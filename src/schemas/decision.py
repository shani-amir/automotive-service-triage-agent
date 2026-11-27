from typing import Any, Literal

from pydantic import BaseModel


class BusinessDecision(BaseModel):
    route: Literal["auto_respond", "request_missing_info", "escalate_hitl"]
    requires_human_approval: bool
    reasons: list[str]
    entitlements: dict[str, Any]
    suggested_next_action: str


class DraftedOutput(BaseModel):
    draft_reply_he: str
    internal_note_he: str
    cited_policy_titles: list[str]


class HumanReviewDecision(BaseModel):
    approved: bool
    reviewer_id: str
    edited_reply_he: str | None = None
    reviewer_note: str | None = None
