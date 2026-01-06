"""
FastAPI WebSocket 服务
提供对话接口
"""

from __future__ import annotations

import uuid

from fastapi import WebSocket, WebSocketDisconnect

from athena import Athena
from athena.context import ChatEventListener, ChatRequest, DialogueContext, UserGender, UserType
from web.web import app


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket 对话接口"""
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
    
    async def on_chat_model_stream(chunk):
        await websocket.send_json({"message": f"流式响应: {chunk}"})
    
    async def on_exit():
        await websocket.send_json({"message": "退出"})
        await websocket.close()
    
    listener = ChatEventListener(
        on_chat_start=on_chat_start,
        on_chat_end=on_chat_end,
        on_chat_model_stream=on_chat_model_stream,
        on_exit=on_exit,
    )
    
    try:
        while True:
            # 接收消息
            user_input = await websocket.receive_text()
            if user_input == "exit":
                break
            if not user_input.strip():
                continue
            await athena.chat(ChatRequest(thread_id=thread_id, user_input=user_input, context=context), listener)
            
    except WebSocketDisconnect:
        print("WebSocket 连接已断开")


