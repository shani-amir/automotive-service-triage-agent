import pytest

from src.schemas.triage import TriageExtraction
from src.services.erp_repo import SQLiteVehicleRepository
from src.services.policy_engine import evaluate_business_rules


@pytest.mark.parametrize(
    "changes, erp, plate, route, reason",
    [
        ({"estimated_cost_nis": 5000}, "found", "48291302", "escalate_hitl", "high_estimated_cost"),
        ({"estimated_cost_nis": 4999.99}, "found", "48291302", "auto_respond", "no_escalation_required"),
        ({"sentiment": "angry"}, "skipped_no_plate", None, "escalate_hitl", "angry_sentiment"),
        ({"urgency": "critical"}, "found", "48291302", "escalate_hitl", "critical_urgency"),
        ({}, "found", "5476219", "escalate_hitl", "inactive_contract"),
        ({}, "found", "67128904", "escalate_hitl", "coverage_not_included"),
        ({}, "error", None, "escalate_hitl", "erp_lookup_error"),
        ({}, "not_found", None, "escalate_hitl", "erp_not_found_actionable"),
        ({}, "skipped_no_plate", None, "request_missing_info", "missing_license_plate"),
        ({"missing_info": ["location"]}, "found", "48291302",
         "request_missing_info", "missing_required_information"),
        ({"intent": "general_inquiry", "missing_info": ["plate"]},
         "skipped_no_plate", None, "auto_respond", "general_policy_inquiry"),
        ({"intent": "general_inquiry"}, "not_found", None, "auto_respond", "general_policy_inquiry"),
        ({"intent": "general_inquiry"}, "found", "48291302", "auto_respond", "general_policy_inquiry"),
    ],
)
def test_business_routes(changes, erp, plate, route, reason):
    vehicle = SQLiteVehicleRepository(":memory:").get_by_license_plate(plate) if plate else None
    triage = TriageExtraction(
        **({"intent": "accident_towing", "urgency": "low", "sentiment": "neutral",
            "missing_info": [], "summary_he": "שאלת שירות"} | changes)
    )
    decision = evaluate_business_rules(triage, erp, vehicle, [])
    assert decision.route == route
    assert decision.requires_human_approval == (route == "escalate_hitl")
    assert reason in decision.reasons
    assert decision.suggested_next_action
    if plate == "48291302":
        assert decision.entitlements == {
            "towing_eligible": True, "replacement_car_eligible": True,
            "deductible_nis": 1500, "contract_status": "active",
        }
    if plate == "5476219":
        assert decision.entitlements["towing_eligible"] is False
        assert decision.entitlements["deductible_nis"] is None


def test_cost_override_and_multiple_reasons():
    triage = TriageExtraction(
        intent="accident_towing", urgency="critical", sentiment="angry",
        estimated_cost_nis=1000, missing_info=["location"], summary_he="דרוש גרר",
    )
    decision = evaluate_business_rules(triage, "skipped_no_plate", None, [], 1000)
    assert decision.reasons == [
        "critical_urgency", "angry_sentiment", "high_estimated_cost",
        "missing_license_plate", "missing_required_information",
    ]
