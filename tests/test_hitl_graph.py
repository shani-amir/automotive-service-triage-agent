import json

import pytest
from langchain_core.runnables import RunnableLambda
from langgraph.types import Command

from src.agent import nodes
from src.schemas.decision import BusinessDecision, DraftedOutput, HumanReviewDecision
from src.services import response_drafter
from src.services.response_drafter import HOLD_REPLY_HE, GeminiResponseDrafter


@pytest.mark.parametrize("approved, edited, expected", [
    (True, "נציג שירות יבדוק את הזכאות ויעדכן אותך.", "approved_by_human"),
    (True, None, "approved_by_human"),
    (False, None, "rejected_by_human"),
    (False, "הפנייה הועברה לבדיקה נוספת.", "rejected_by_human"),
])
def test_interrupt_and_resume(step3_graph_factory, approved, edited, expected):
    graph, drafter = step3_graph_factory(risk=True)
    config = {"configurable": {"thread_id": "risk"}}
    result = graph.invoke({"raw_input": "raw customer text 052-555-0199"}, config=config)
    packet = result["__interrupt__"][0].value
    assert graph.get_state(config).next == ("human_review",)
    assert result["final_status"] == "awaiting_human_review"
    assert result["final_reply_he"] is None
    assert "angry_sentiment" in packet["reasons"]
    assert packet["erp_status"] == "found"
    assert packet["vehicle_context"]["contract_type"] == "operational_leasing"
    assert packet["cited_policy_titles"] == ["שירותי דרך"]
    assert packet["draft_reply_he"] and packet["internal_note_he"]
    assert packet["suggested_next_action"]
    serialized_packet = json.dumps(packet, ensure_ascii=False)
    for sensitive in ("raw customer text", "052-555-0199", "482-91-302", "F-1001"):
        assert sensitive not in serialized_packet
        assert sensitive not in str(drafter.call_args)

    review = HumanReviewDecision(approved=approved, reviewer_id="agent-7", edited_reply_he=edited)
    resumed = graph.invoke(Command(resume=review.model_dump()), config=config)
    assert resumed["final_status"] == expected
    assert resumed["final_reply_he"] == (edited or (packet["draft_reply_he"] if approved else HOLD_REPLY_HE))
    assert resumed["human_review"] == review
    assert not graph.get_state(config).next
    drafter.assert_called_once()


def test_low_risk_completes_without_interrupt(step3_graph_factory):
    graph, _ = step3_graph_factory()
    config = {"configurable": {"thread_id": "low"}}
    result = graph.invoke({"raw_input": "שאלת מדיניות כללית"}, config=config)
    assert not result.get("__interrupt__")
    assert not graph.get_state(config).next
    assert result["final_status"] == "auto_completed"
    assert result["final_reply_he"] == result["draft_reply_he"]


def test_legacy_graph_uses_fallback_and_marks_review_pending(step3_graph_factory):
    graph, _ = step3_graph_factory(risk=True, checkpointer=False, drafting_error=True)
    result = graph.invoke({"raw_text": "תאונה. נדרש גרר."})
    assert not result.get("__interrupt__")
    assert result["final_status"] == "awaiting_human_review"
    assert result["final_reply_he"] is None
    assert result["draft_reply_he"] == HOLD_REPLY_HE
    assert "לקוח כועס" in result["internal_note_he"]


def test_missing_info_fallback_asks_for_plate(step3_graph_factory):
    graph, _ = step3_graph_factory(missing=True, checkpointer=False, drafting_error=True)
    result = graph.invoke({"raw_input": "אפשר להזמין גרר?"})
    assert result["final_status"] == "needs_customer_info"
    assert "מספר הרכב" in result["final_reply_he"]


def test_gemini_drafter_reuses_structured_model_fallback(monkeypatch):
    calls = []
    expected = DraftedOutput(
        draft_reply_he="נשמח לבדוק את תנאי השירות.",
        internal_note_he="שאלה כללית.",
        cited_policy_titles=[],
    )

    class Client:
        def __init__(self, *, model, **options):
            self.model = model
            assert options == (
                {"timeout": 25, "max_retries": 1}
                | ({} if model.endswith("lite") else {"temperature": 0})
            )

        def with_structured_output(self, schema):
            assert schema is DraftedOutput

            def invoke(messages):
                calls.append(self.model)
                if self.model.endswith("lite"):
                    raise RuntimeError("Primary model offline")
                payload = json.loads(messages[1][1])
                assert payload["decision"]["route"] == "auto_respond"
                assert payload["intent"] == "general_inquiry"
                return expected

            return RunnableLambda(invoke)

    monkeypatch.setattr(nodes, "ChatGoogleGenerativeAI", Client)
    monkeypatch.setattr(nodes, "load_dotenv", lambda: None)
    monkeypatch.setenv("GEMINI_MODEL", "model-lite")
    monkeypatch.setenv("GEMINI_FALLBACK_MODELS", "model-flash")
    monkeypatch.delenv("GEMINI_THINKING_LEVEL", raising=False)
    decision = BusinessDecision(
        route="auto_respond", requires_human_approval=False, reasons=[],
        entitlements={}, suggested_next_action="Answer the general question.",
    )
    response_drafter._get_drafting_llm.cache_clear()
    try:
        draft = GeminiResponseDrafter()(
            decision=decision, intent="general_inquiry", missing_details=[], retrieved_policies=[],
        )
    finally:
        response_drafter._get_drafting_llm.cache_clear()
    assert draft == expected
    assert calls == ["model-lite", "model-flash"]
