from unittest.mock import Mock

from evals.runner import run_evals
from src.agent.nodes import draft_response
from src.schemas.decision import BusinessDecision, DraftedOutput
from src.schemas.triage import TriageExtraction
from src.services.policy_rag import LocalHybridPolicyRetriever
from src.services.response_drafter import asks_for_plate, build_fallback_draft


def test_offline_golden_dataset():
    report = run_evals()
    failures = {case["id"]: case["checks"] for case in report["cases"] if not case["passed"]}
    assert report["total_cases"] == report["passed_cases"] == 10, failures
    assert report["pass_rate"] == 1.0


def test_found_vehicle_does_not_ask_for_plate():
    decision = BusinessDecision(
        route="request_missing_info", requires_human_approval=False,
        reasons=["missing_license_plate", "missing_required_information"],
        entitlements={"contract_status": "active"}, suggested_next_action="Ask for details.",
    )
    state = {
        "decision": decision, "erp_status": "found", "retrieved_policies": [],
        "extraction": TriageExtraction(
            license_plate="482-91-302", intent="maintenance_garage", urgency="low",
            sentiment="neutral", missing_info=["מיקום"], summary_he="תיאום טיפול.",
        ),
    }

    contradictory_draft = Mock(return_value=DraftedOutput(
        draft_reply_he="אנא שלחו את מספר הרישוי.", internal_note_he="תיאום טיפול.",
        cited_policy_titles=[],
    ))

    fallback = build_fallback_draft(decision, "maintenance_garage", [], erp_status="found")
    result = draft_response(state, response_drafter=contradictory_draft)
    assert contradictory_draft.call_args.kwargs["erp_status"] == "found"
    assert "מספר" not in fallback.draft_reply_he
    assert result["draft_reply_he"] == fallback.draft_reply_he
    assert asks_for_plate("אנא שלחו את מספר הרישוי.")
    assert not asks_for_plate("מספר הרכב כבר מופיע במערכת.")


def test_operational_leasing_maintenance_policy_ranks_first():
    chunks = LocalHybridPolicyRetriever().retrieve(
        "מתי צריך לעשות טיפול תקופתי לרכב בליסינג תפעולי ואיך מתאמים במוסך הסדר?",
        contract_type="operational_leasing", intent="maintenance_garage",
    )
    assert chunks[0].doc_id == "warranty_and_maintenance"
    assert "טיפול תקופתי" in chunks[0].section
