from unittest.mock import Mock

import pytest
from langsmith import tracing_context

from src.agent.checkpoint import build_memory_checkpointer
from src.agent.graph import build_triage_graph
from src.schemas.decision import DraftedOutput
from src.schemas.retrieval import PolicyChunk
from src.schemas.triage import TriageExtraction
from src.services.erp_repo import SQLiteVehicleRepository


@pytest.fixture
def step3_graph_factory():
    def create(*, risk=False, missing=False, checkpointer=True, drafting_error=False):
        llm = Mock()
        llm.invoke.return_value = TriageExtraction(
            intent="accident_towing" if risk or missing else "general_inquiry",
            urgency="high" if risk else "low",
            sentiment="angry" if risk else "neutral",
            license_plate="482-91-302" if risk else None,
            missing_info=["מיקום הרכב, 482-91-302, 052-555-0199"] if risk else [],
            summary_he="פנייה בנושא שירות. 052-555-0199",
        )
        policy = PolicyChunk(
            chunk_id="towing-1", doc_id="towing", title="שירותי דרך", section="גרירה",
            applies_to=["all"], topics=["towing"], score=1, content="מימוש לפי תנאי החוזה.",
        )
        retriever = Mock()
        retriever.retrieve.return_value = [policy]
        drafter = Mock(
            return_value=DraftedOutput(
                draft_reply_he="נשמח לעזור בבדיקת הפנייה.",
                internal_note_he="יש לבדוק את תנאי השירות.",
                cited_policy_titles=[policy.title, "Unknown", policy.title],
            ),
            side_effect=RuntimeError("Gemini unavailable") if drafting_error else None,
        )
        graph = build_triage_graph(
            llm=llm, vehicle_repo=SQLiteVehicleRepository(":memory:"),
            policy_retriever=retriever, response_drafter=drafter,
            checkpointer=build_memory_checkpointer() if checkpointer else None,
        )
        return graph, drafter

    with tracing_context(enabled=False):
        yield create
