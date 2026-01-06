"""
FastAPI WebSocket 服务
提供对话接口
"""

from __future__ import annotations

import uuid

from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from athena import Athena
from athena.context import ChatEvent, ChatEventListener, ChatRequest, DialogueContext, UserGender, UserType
from web.web import app

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
        context = DialogueContext(
            user_id="00006",
            user_name="赵六",
            user_type=UserType.STRANGER,
            user_gender=UserGender.MALE,
        )

        async def on_chat_start():
            await websocket.send_json({"message": "单轮对话开始"})
        
        async def on_chat_end():
            await websocket.send_json({"message": "单轮对话结束"})
        
        async def on_chat_event_stream(event: ChatEvent | None):
            if event:
                # 使用 Pydantic 的 model_dump 将模型转换为字典，支持枚举类型序列化
                event_dict = event.model_dump(mode='json')
                await websocket.send_json(event_dict)
        
        async def on_exit():
            await websocket.send_json({"message": "退出"})
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
            if user_input == "exit":
                break
            if not user_input.strip():
                continue
            await athena.chat(ChatRequest(thread_id=thread_id, user_input=user_input, context=context), listener)
            
    except WebSocketDisconnect:
        print("WebSocket 连接已断开")
    except Exception as e:
        print(f"WebSocket 处理错误: {e}")
        try:
            await websocket.close()
        except:
            pass


