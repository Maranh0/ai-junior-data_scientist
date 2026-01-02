# api_main.py

from fastapi import FastAPI
from pydantic import BaseModel

from agent_cli import build_agent_executor
from metrics_logger import MetricsLogger

app = FastAPI(title="AI Junior Data Scientist API")

agent_executor = build_agent_executor()
metrics_logger = MetricsLogger()


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str
    latency_sec: float
    total_requests: int


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat_with_agent(req: ChatRequest):
    wrapped = metrics_logger.wrap_call(agent_executor, req.message)
    result = wrapped["result"]
    metrics = wrapped["metrics"]

    output = result.get("output", result)

    return ChatResponse(
        reply=output,
        latency_sec=metrics["latency_sec"],
        total_requests=metrics["total_requests"],
    )
