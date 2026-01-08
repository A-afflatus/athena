"""
FastAPI WebSocket 服务
提供对话接口
"""

from __future__ import annotations

import json
from typing import cast
import uuid

from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from athena import Athena
from athena.context import (
    ChatEvent,
    ChatEventListener,
    ChatRequest,
    DialogueContext,
    UserGender,
    UserType,
)
from web.web import app
from bootstrap.logger import get_logger

logger = get_logger(__name__)


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket 对话接口"""
    try:
        # 必须先接受 WebSocket 连接
        await websocket.accept()

        # 启动Athena
        athena = Athena()
        # 初始化athena
        await athena.init()

        thread_id = str(uuid.uuid4())
        # context = DialogueContext(
        #     user_id="root",
        #     user_type=UserType.OWNER,
        #     user_name="汪京",
        #     user_gender=UserGender.MALE,
        #     user_location="北京市亦庄经济开发区",
        # )


        async def on_chat_start():
            await websocket.send_json({"event":"on_dialogue_start"})

        async def on_chat_end():
            await websocket.send_json({"event":"on_dialogue_end"})

        async def on_chat_event_stream(event: ChatEvent | None):
            if event:
                # 使用 Pydantic 的 model_dump 将模型转换为字典，支持枚举类型序列化
                event_dict = event.model_dump(mode="json")
                await websocket.send_json(event_dict)

        async def on_exit():
            await websocket.close()

        listener = ChatEventListener(
            on_chat_start=on_chat_start,
            on_chat_end=on_chat_end,
            on_chat_event_stream=on_chat_event_stream,
            on_exit=on_exit,
        )

        while True:
            # 检查 WebSocket 连接状态
            if websocket.application_state == WebSocketState.DISCONNECTED:
                break

            # 接收消息
            user_input = await websocket.receive_text()
            user_input = json.loads(user_input)
            input_event = cast(str, user_input.get("event"))
            if input_event == "exit":
                break
            if input_event == "user_input":
                text_input = cast(str, user_input.get("text_input", ""))
                if not text_input or text_input.strip()=="":
                    continue
                emotion = cast(str, user_input.get("emotion", ""))
                context = DialogueContext(
                    user_id="00006",
                    user_name="赵六",
                    user_type=UserType.STRANGER,
                    user_gender=UserGender.MALE,
                    user_emotion=emotion,
                )
                await athena.chat(ChatRequest(thread_id=thread_id, user_input=text_input, context=context),listener)

    except WebSocketDisconnect:
        logger.info("WebSocket 连接已断开")
    except Exception as e:
        logger.error(f"WebSocket 处理错误: {e}", exc_info=True)
        try:
            await websocket.close()
        except:
            pass
