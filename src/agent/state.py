from typing import Any, Literal, TypedDict

from src.schemas.decision import BusinessDecision, HumanReviewDecision
from src.schemas.retrieval import PolicyChunk, VehicleContractContext
from src.schemas.triage import TriageExtraction


class TriageState(TypedDict, total=False):
    raw_input: str
    raw_text: str
    source_type: Literal["call_transcript", "document"]
    masked_input: str
    redacted_pii_types: list[str]
    extraction: TriageExtraction
    erp_status: Literal["found", "not_found", "skipped_no_plate", "error"]
    vehicle_context: VehicleContractContext | None
    retrieved_policies: list[PolicyChunk]
    retrieval_filters: dict[str, Any]
    decision: BusinessDecision | None
    draft_reply_he: str | None
    internal_note_he: str | None
    cited_policy_titles: list[str]
    human_review: HumanReviewDecision | None
    final_reply_he: str | None
    final_status: Literal[
        "auto_completed", "needs_customer_info", "awaiting_human_review",
        "approved_by_human", "rejected_by_human",
    ] | None
