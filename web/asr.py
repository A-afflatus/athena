from __future__ import annotations

import asyncio
import json
import os
import queue
import threading
import time
from typing import Callable

import websocket
from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from bootstrap.logger import get_logger
from web.web import app

logger = get_logger(__name__)

# 配置常量
API_KEY = os.environ.get("LLM_QWEN_API_KEY")
if not API_KEY:
    raise ValueError("ASR LLM_QWEN_API_KEY is not set")
QWEN_MODEL = "qwen3-asr-flash-realtime"
BASE_URL = "wss://dashscope.aliyuncs.com/api-ws/v1/realtime"


class ASRWebSocketClient:

    def __init__(
        self,
        on_transcript: Callable[[dict], None],
        on_error: Callable[[str], None],
        on_close: Callable[[], None],
    ):
        self.on_transcript = on_transcript
        self.on_error = on_error
        self.on_close = on_close
        self.ws: websocket.WebSocketApp | None = None
        self.is_running = False
        self.audio_queue: queue.Queue[str] = queue.Queue()
        self.ws_thread: threading.Thread | None = None
        self.audio_thread: threading.Thread | None = None

    def _on_open(self, ws: websocket.WebSocketApp) -> None:
        logger.info("已连接到阿里云 ASR 服务器")
        event_vad = {
            "event_id": f"event_{int(time.time() * 1000)}",
            "type": "session.update",
            "session": {
                "modalities": ["text"],
                "input_audio_format": "pcm",
                "sample_rate": 16000,
                "input_audio_transcription": {
                    "language": "zh",
                },
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.0,
                    "silence_duration_ms": 400,
                },
            },
        }
        logger.debug(
            f"发送会话配置: {json.dumps(event_vad, indent=2, ensure_ascii=False)}"
        )
        ws.send(json.dumps(event_vad))
        # 等待会话更新完成
        time.sleep(0.5)

    def _on_message(self, ws: websocket.WebSocketApp, message: str) -> None:
        try:
            data = json.loads(message)
            logger.debug(
                f"收到 ASR 事件: {json.dumps(data, ensure_ascii=False, indent=2)}"
            )

            event_type = data.get("type", "")
            if event_type == "conversation.item.input_audio_transcription.completed":
                transcript = data.get("transcript", "")
                logger.info(f"最终识别结果: {transcript}")
                self.on_transcript(
                    {
                        "type": "transcription.completed",
                        "transcript": transcript,
                        "data": data,
                    }
                )
            elif event_type == "error":
                error_msg = data.get("error", {}).get("message", "未知错误")
                logger.error(f"ASR 服务错误: {error_msg}")
                self.on_error(error_msg)
        except json.JSONDecodeError as e:
            logger.error(f"解析消息失败: {message}, 错误: {e}")

    def _on_error(self, ws: websocket.WebSocketApp, error: Exception) -> None:
        logger.error(f"WebSocket 错误: {error}")
        self.on_error(str(error))

    def _on_close(
        self, ws: websocket.WebSocketApp, close_status_code: int, close_msg: str
    ) -> None:
        """WebSocket 关闭回调"""
        logger.info(f"WebSocket 连接已关闭: {close_status_code}, {close_msg}")
        self.is_running = False
        self.on_close()

    def _send_audio_loop(self) -> None:
        """音频发送循环（在线程中运行）"""
        time.sleep(0.5)  # 等待会话更新完成
        while self.is_running:
            try:
                # 从队列获取音频数据，超时时间0.1秒
                encoded_data = self.audio_queue.get(timeout=0.1)
                if not self.ws or not self.ws.sock or not self.ws.sock.connected:
                    logger.info("WebSocket 已断开，停止发送音频")
                    break
                event = {
                    "event_id": f"event_{int(time.time() * 1000)}",
                    "type": "input_audio_buffer.append",
                    "audio": encoded_data,
                }
                self.ws.send(json.dumps(event))
                logger.debug(f"发送音频数据块: {event['event_id']}")
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"发送音频数据失败: {e}")
                break

    def start(self) -> None:
        """启动 WebSocket 连接和音频发送线程"""
        if self.is_running:
            logger.warning("ASR 客户端已在运行")
            return

        url = f"{BASE_URL}?model={QWEN_MODEL}"
        headers = [
            f"Authorization: Bearer {API_KEY}",
            "OpenAI-Beta: realtime=v1",
        ]

        self.is_running = True
        self.ws = websocket.WebSocketApp(
            url,
            header=headers,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )

        # 启动 WebSocket 线程
        self.ws_thread = threading.Thread(
            target=self.ws.run_forever, daemon=True, name="ASR-WebSocket"
        )
        self.ws_thread.start()

        # 启动音频发送线程
        self.audio_thread = threading.Thread(
            target=self._send_audio_loop, daemon=True, name="ASR-AudioSender"
        )
        self.audio_thread.start()

        logger.info("ASR 客户端已启动")

    def send_audio(self, audio_data: str) -> None:
        """发送音频数据到队列"""
        if not self.is_running:
            logger.warning("ASR 客户端未运行，无法发送音频")
            return
        self.audio_queue.put_nowait(audio_data)

    def stop(self) -> None:
        """停止 WebSocket 连接"""
        if not self.is_running:
            return

        self.is_running = False
        if self.ws:
            try:
                self.ws.close()
            except Exception as e:
                logger.error(f"关闭 WebSocket 失败: {e}")

        # 等待线程结束
        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=2.0)
        if self.ws_thread and self.ws_thread.is_alive():
            self.ws_thread.join(timeout=2.0)

        logger.info("ASR 客户端已停止")


@app.websocket("/ws/asr")
async def websocket_asr(websocket: WebSocket):
    """WebSocket 实时语音识别接口"""
    asr_client: ASRWebSocketClient | None = None

    try:
        # 接受 WebSocket 连接
        await websocket.accept()
        logger.info("客户端已连接 ASR 服务")

        # 创建 ASR 客户端
        transcript_queue: asyncio.Queue[dict] = asyncio.Queue()
        error_occurred = False

        loop = asyncio.get_event_loop()

        def on_transcript(data: dict) -> None:
            """识别结果回调"""
            asyncio.run_coroutine_threadsafe(transcript_queue.put(data), loop)

        def on_error(error_msg: str) -> None:
            """错误回调"""
            nonlocal error_occurred
            error_occurred = True
            asyncio.run_coroutine_threadsafe(
                transcript_queue.put({"type": "error", "message": error_msg}), loop
            )

        def on_close() -> None:
            """关闭回调"""
            asyncio.run_coroutine_threadsafe(
                transcript_queue.put({"type": "closed"}), loop
            )

        asr_client = ASRWebSocketClient(
            on_transcript=on_transcript, on_error=on_error, on_close=on_close
        )
        asr_client.start()

        # 启动识别结果发送任务
        async def send_transcripts():
            """发送识别结果到客户端"""
            while True:
                try:
                    data = await asyncio.wait_for(transcript_queue.get(), timeout=1.0)
                    if data.get("type") == "closed":
                        break
                    await websocket.send_json(data)
                    if data.get("type") == "error":
                        break
                except asyncio.TimeoutError:
                    # 检查连接状态
                    if websocket.application_state == WebSocketState.DISCONNECTED:
                        break
                    continue

        send_task = asyncio.create_task(send_transcripts())

        # 接收客户端音频数据
        while True:
            # 检查连接状态
            if websocket.application_state == WebSocketState.DISCONNECTED:
                break

            if error_occurred:
                break

            try:
                # 接收音频数据（二进制或文本）
                event = await websocket.receive()
                text = event.get("text")
                if text:
                    event = json.loads(text)
                if event.get("event") == "close":
                    break
                if event.get("event") == "send_audio":
                    audio_data = event.get("data")
                    if audio_data:
                        asr_client.send_audio(audio_data)
            except WebSocketDisconnect:
                logger.info("客户端断开连接")
                break
            except Exception as e:
                logger.error(f"接收消息错误: {e}")
                await websocket.send_json(
                    {"type": "error", "message": f"处理消息失败: {e}"}
                )
                break

        # 取消发送任务
        send_task.cancel()
        try:
            await send_task
        except asyncio.CancelledError:
            pass

    except WebSocketDisconnect:
        logger.info("WebSocket 连接已断开")
    except Exception as e:
        logger.error(f"WebSocket 处理错误: {e}")
    finally:
        # 清理资源
        if asr_client:
            asr_client.stop()
        try:
            await websocket.close()
        except Exception:
            pass
