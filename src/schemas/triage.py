from typing import Literal, Optional

from pydantic import BaseModel, Field


class TriageExtraction(BaseModel):
    license_plate: Optional[str] = Field(
        default=None,
        description="Explicitly stated Israeli 7- or 8-digit vehicle plate. Preserve its format. Never guess.",
        pattern=r"^(?:[0-9]{7,8}|[0-9]{2}-[0-9]{3}-[0-9]{2}|[0-9]{3}-[0-9]{2}-[0-9]{3})$",
    )
    intent: Literal[
        "accident_towing", "maintenance_garage", "leasing_contract",
        "insurance_claim", "billing_invoice", "general_inquiry",
    ] = Field(description="Primary reason the customer needs assistance.")
    urgency: Literal["low", "medium", "high", "critical"] = Field(
        description="Operational priority based on the stated circumstances and immediate safety risk."
    )
    sentiment: Literal["positive", "neutral", "frustrated", "angry"] = Field(
        description="Customer's expressed tone."
    )
    estimated_cost_nis: Optional[float] = Field(
        default=None, description="Explicitly stated estimated cost in NIS. Leave null if absent."
    )
    missing_info: list[str] = Field(
        description="Information still needed to act. Return an empty list if nothing is missing."
    )
    summary_he: str = Field(
        description="Concise operational summary in Hebrew, one or two sentences."
    )
