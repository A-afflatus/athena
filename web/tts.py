"""
FastAPI WebSocket TTS 服务
提供实时文本转语音接口
"""

from __future__ import annotations

import json
import os
import asyncio
from typing import Optional, Any, cast

from dashscope.audio.qwen_tts_realtime import (
    AudioFormat,
    QwenTtsRealtime,
    QwenTtsRealtimeCallback,
)
from dotenv import load_dotenv
from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from web.web import app
from bootstrap.logger import get_logger

load_dotenv()

logger = get_logger(__name__)

TTS_KEY = os.environ.get("LLM_QWEN_API_KEY")
if not TTS_KEY:
    raise ValueError("TTS LLM_QWEN_API_KEY is not set")

TTS_MODEL = "qwen-tts-realtime"


class TtsWebSocketCallback(QwenTtsRealtimeCallback):

    def __init__(
        self,
        message_queue: asyncio.Queue,
        loop: asyncio.AbstractEventLoop,
        complete_event: asyncio.Event,
    ):
        self.message_queue = message_queue
        self.loop = loop
        self.complete_event = complete_event
        # self.file = open('result_24k'+str(time.time())+'.pcm', 'wb')


    def _put_message(self, event: str, data: Any = None):
        self.loop.call_soon_threadsafe(
            self.message_queue.put_nowait, {"event": event, "data": data}
        )

    def on_open(self) -> None:
        self._put_message("connection_opened")

    def on_close(self, close_status_code: int, close_msg: str) -> None:
        logger.info(f"TTS连接已关闭，状态码: {close_status_code}, 消息: {close_msg}")
        self._put_message("connection_closed")
        self.loop.call_soon_threadsafe(self.complete_event.set)

    def on_event(self, response: dict) -> None:
        try:
            event_type = response.get("type", "")
            session_id = response.get("session", {}).get("id", "")
            response_id = response.get("response", {}).get("id", "")
            if event_type == "session.created":
                logger.info(f"TTS会话已创建: {session_id}")
                self._put_message("session_created", {"session_id": session_id})

            elif event_type == "response.audio.delta":
                audio_b64 = response.get("delta", "")
                if audio_b64:
                    # 避免在日志中打印过长的 base64 数据
                    self._put_message("audio_delta", audio_b64)
                    # self.file.write(base64.b64decode(audio_b64))

            elif event_type == "response.done":
                logger.info(f"TTS响应完成: {response_id}")
                self._put_message("response_done", {"response_id": response_id})

            elif event_type == "session.finished":
                logger.info("TTS会话已完成")
                self._put_message("session_finished")
                self.loop.call_soon_threadsafe(self.complete_event.set)

        except Exception as e:
            logger.error(f"TTS事件处理错误: {e}")
            self._put_message("error", {"message": str(e)})


@app.websocket("/ws/tts")
async def websocket_tts(websocket: WebSocket):
    """WebSocket TTS接口"""
    tts_instance: Optional[QwenTtsRealtime] = None
    message_queue: Optional[asyncio.Queue] = None
    complete_event = asyncio.Event()

    try:
        # 接受WebSocket连接
        await websocket.accept()
        # 获取当前事件循环
        loop = asyncio.get_running_loop()
        # 创建消息队列
        message_queue = asyncio.Queue()
        # 创建回调实例
        callback = TtsWebSocketCallback(message_queue, loop, complete_event)
        # 初始化TTS实例
        tts_instance = QwenTtsRealtime(
            model=TTS_MODEL,
            callback=callback,
        )
        tts_instance.apikey = TTS_KEY
        # 连接TTS服务
        tts_instance.connect()
        # 更新会话配置
        tts_instance.update_session(
            voice="Serena",
            response_format=AudioFormat.PCM_24000HZ_MONO_16BIT,
            mode="server_commit",
        )

        # 创建任务来处理消息队列
        async def process_messages():
            """处理来自TTS回调的消息"""
            while True:
                try:
                    message = await asyncio.wait_for(message_queue.get(), timeout=1.0)
                    if message.get("event") != "audio_delta":
                        logger.info(f"发送消息到客户端: {message}")
                    await websocket.send_json(message)
                except asyncio.TimeoutError:
                    # 检查连接状态
                    if websocket.application_state == WebSocketState.DISCONNECTED:
                        break
                except Exception as e:
                    logger.error(f"处理消息时出错: {e}")
                    break

        message_task = asyncio.create_task(process_messages())

        buffer_text = ""

        # 接收客户端消息
        try:
            while True:
                # 接收文本消息
                try:
                    event_data = await websocket.receive_text()
                    logger.info(f"收到TTS请求: {event_data}")
                    event = json.loads(event_data)

                    if event.get("event") == "send_text":
                        text_data = cast(str, event.get("data", ""))
                        buffer_text += text_data.strip()
                        if len(buffer_text) > 10:
                            tts_instance.append_text(buffer_text)
                            buffer_text = ""
                    elif event.get("event") == "close":
                        buffer_text = buffer_text.strip()
                        if buffer_text != "":
                            tts_instance.append_text(buffer_text)
                        break

                except WebSocketDisconnect:
                    logger.info("WebSocket 连接已断开")
                    break
                except Exception as e:
                    logger.error(f"接收消息时出错: {e}")
                    break

        finally:
            # 完成TTS会话
            try:
                if tts_instance:
                    tts_instance.finish()

            except Exception as e:
                logger.error(f"完成TTS会话时出错: {e}")

            # 等待所有消息处理完毕（session_finished 或 connection_closed）
            try:
                await asyncio.wait_for(complete_event.wait(), timeout=10.0)
            except asyncio.TimeoutError:
                logger.warning("等待TTS会话结束超时")

            # 取消消息处理任务
            message_task.cancel()
            try:
                await message_task
            except asyncio.CancelledError:
                pass

    except WebSocketDisconnect:
        logger.info("TTS WebSocket连接已断开")
    except Exception as e:
        logger.error(f"TTS WebSocket处理错误: {e}", exc_info=True)
        try:
            await websocket.send_json({"event": "error", "data": {"message": str(e)}})
        except:
            pass
    finally:
        # 清理资源
        if tts_instance:
            try:
                tts_instance.close()
            except:
                pass
        try:
            await websocket.close()
        except:
            pass
