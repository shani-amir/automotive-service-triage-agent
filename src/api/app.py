from threading import Lock
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from langgraph.types import Command
from langsmith import tracing_context
from pydantic import BaseModel, Field

from src.agent.checkpoint import build_memory_checkpointer
from src.agent.graph import build_triage_graph
from src.agent.nodes import MAX_INPUT_CHARS
from src.schemas.decision import HumanReviewDecision
from src.services.response_drafter import GeminiResponseDrafter


class TriageRequest(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_INPUT_CHARS)
    thread_id: str | None = Field(default=None, min_length=1)


def _thread_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _pending_review(snapshot):
    if "human_review" in snapshot.next and snapshot.interrupts:
        return snapshot.interrupts[0]
    return None


def create_app(graph=None, checkpointer=None) -> FastAPI:
    """Build the local API. In production, use PostgresSaver or a Redis-backed checkpointer."""
    if graph is None:
        graph = build_triage_graph(
            response_drafter=GeminiResponseDrafter(),
            checkpointer=build_memory_checkpointer() if checkpointer is None else checkpointer,
        )
    service_name = "automotive-service-triage-agent"
    app = FastAPI(title=service_name)
    # Serialize writes to the same thread within this process.
    thread_locks = [Lock() for _ in range(32)]

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_request: Request, _error: RequestValidationError):
        return JSONResponse(status_code=422, content={"detail": "Invalid request body or parameters."})

    @app.get("/health")
    def health():
        return {"status": "ok", "service": service_name, "hitl_enabled": True}

    @app.post("/api/v1/triage")
    def triage(request: TriageRequest):
        if not request.message.strip():
            raise HTTPException(400, "Message must not be blank.")
        thread_id = request.thread_id if request.thread_id is not None else str(uuid4())
        config = _thread_config(thread_id)
        with thread_locks[hash(thread_id) % len(thread_locks)], tracing_context(enabled=False):
            if graph.get_state(config).created_at is not None:
                raise HTTPException(409, "Thread already exists.")
            result = graph.invoke({"raw_input": request.message}, config=config)

        common = {
            "thread_id": thread_id,
            "status": result["final_status"],
            "decision": result["decision"],
            "erp_status": result["erp_status"],
        }
        if interrupts := result.get("__interrupt__"):
            return {**common, "review_payload": interrupts[0].value}
        return {
            **common,
            "final_reply_he": result["final_reply_he"],
            "internal_note_he": result["internal_note_he"],
            "cited_policy_titles": result["cited_policy_titles"],
        }

    @app.post("/api/v1/triage/{thread_id}/review")
    def review(thread_id: str, review: HumanReviewDecision):
        config = _thread_config(thread_id)
        with thread_locks[hash(thread_id) % len(thread_locks)], tracing_context(enabled=False):
            snapshot = graph.get_state(config)
            if snapshot.created_at is None:
                raise HTTPException(404, "Thread not found.")
            if _pending_review(snapshot) is None:
                raise HTTPException(409, "Thread is not awaiting human review.")
            result = graph.invoke(Command(resume=review.model_dump()), config=config)
        return {
            "thread_id": thread_id,
            "status": result["final_status"],
            "final_reply_he": result["final_reply_he"],
            "human_review": result["human_review"],
        }

    @app.get("/api/v1/triage/{thread_id}")
    def get_triage(thread_id: str):
        snapshot = graph.get_state(_thread_config(thread_id))
        if snapshot.created_at is None:
            raise HTTPException(404, "Thread not found.")
        return {
            "thread_id": thread_id,
            "status": snapshot.values.get("final_status"),
            "is_paused": _pending_review(snapshot) is not None,
            "decision": snapshot.values.get("decision"),
            "final_reply_he": snapshot.values.get("final_reply_he"),
        }

    return app


app = create_app()
