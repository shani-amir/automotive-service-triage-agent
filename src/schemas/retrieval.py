from typing import Literal

from pydantic import BaseModel


class VehicleContractContext(BaseModel):
    license_plate: str
    normalized_plate: str
    customer_id: str
    customer_name: str
    customer_segment: Literal["private", "fleet_smb", "fleet_enterprise"]
    vehicle_make_model: str
    vehicle_year: int
    contract_type: Literal[
        "operational_leasing", "private_leasing", "manufacturer_warranty", "trade_in_warranty"
    ]
    contract_status: Literal["active", "expired", "suspended"]
    includes_towing: bool
    includes_replacement_car: bool
    deductible_nis: int
    preferred_service_center: str


class PolicyChunk(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    section: str
    applies_to: list[str]
    topics: list[str]
    score: float
    content: str
