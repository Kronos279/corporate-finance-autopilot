from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from app.models.company import AgentTrace
from app.services.agent.orchestrator import run

router = APIRouter()


class AgentAskRequest(BaseModel):
    query: str


@router.post("/{ticker}/ask")
async def ask_agent(ticker: str, request: AgentAskRequest):
    result = await run(ticker, request.query)
    return result


@router.websocket("/ws/{ticker}")
async def agent_ws(websocket: WebSocket, ticker: str):
    await websocket.accept()
    conversation_history: list[dict] = []
    try:
        while True:
            prompt = await websocket.receive_text()

            async def send_trace(trace: AgentTrace) -> None:
                await websocket.send_json({"trace": trace.model_dump()})

            result = await run(ticker, prompt, on_trace=send_trace, conversation_history=conversation_history)

            # Store the exchange in history for follow-up context
            conversation_history.append({"role": "user", "content": f"[Company: {ticker.upper()}] {prompt}"})
            conversation_history.append({"role": "assistant", "content": result.response})

            # Keep history bounded (last 10 exchanges = 20 messages)
            if len(conversation_history) > 20:
                conversation_history = conversation_history[-20:]

            await websocket.send_json({"complete": result.response})
    except WebSocketDisconnect:
        return
