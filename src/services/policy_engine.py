from collections.abc import Sequence

from src.schemas.decision import BusinessDecision
from src.schemas.retrieval import PolicyChunk, VehicleContractContext
from src.schemas.triage import TriageExtraction


def evaluate_business_rules(
    triage: TriageExtraction,
    erp_status: str,
    vehicle_context: VehicleContractContext | None,
    retrieved_policies: Sequence[PolicyChunk],
    high_cost_threshold_nis: float = 5000.0,
) -> BusinessDecision:
    """Route using triage and ERP facts. Retrieved policy informs drafting."""
    vehicle = vehicle_context
    actionable = triage.intent != "general_inquiry"
    active = vehicle is not None and vehicle.contract_status == "active"
    entitlements = {
        "towing_eligible": active and vehicle.includes_towing if vehicle else None,
        "replacement_car_eligible": active and vehicle.includes_replacement_car if vehicle else None,
        "deductible_nis": vehicle.deductible_nis if active else None,
        "contract_status": vehicle.contract_status if vehicle else None,
    }
    escalation_conditions = {
        "critical_urgency": triage.urgency == "critical",
        "angry_sentiment": triage.sentiment == "angry",
        "high_estimated_cost": (
            triage.estimated_cost_nis is not None
            and triage.estimated_cost_nis >= high_cost_threshold_nis
        ),
        "erp_lookup_error": erp_status == "error",
        "erp_not_found_actionable": erp_status == "not_found" and actionable,
        "inactive_contract": vehicle is not None and not active,
        "coverage_not_included": (
            vehicle is not None and triage.intent == "accident_towing" and not vehicle.includes_towing
        ),
    }
    escalation_reasons = [reason for reason, applies in escalation_conditions.items() if applies]
    missing_reasons = []
    if actionable and erp_status == "skipped_no_plate":
        missing_reasons.append("missing_license_plate")
    if actionable and triage.missing_info:
        missing_reasons.append("missing_required_information")
    if escalation_reasons:
        return BusinessDecision(
            route="escalate_hitl", requires_human_approval=True,
            reasons=escalation_reasons + missing_reasons, entitlements=entitlements,
            suggested_next_action="Have a service agent review the case and approve a reply.",
        )
    if missing_reasons:
        return BusinessDecision(
            route="request_missing_info", requires_human_approval=False,
            reasons=missing_reasons, entitlements=entitlements,
            suggested_next_action="Ask the customer for the information needed to continue.",
        )
    return BusinessDecision(
        route="auto_respond", requires_human_approval=False,
        reasons=["general_policy_inquiry" if not actionable else "no_escalation_required"],
        entitlements=entitlements,
        suggested_next_action="Answer using verified entitlements and relevant policy.",
    )
