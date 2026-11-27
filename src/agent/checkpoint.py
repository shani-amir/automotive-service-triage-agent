from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from src.schemas.decision import BusinessDecision, HumanReviewDecision
from src.schemas.retrieval import PolicyChunk, VehicleContractContext
from src.schemas.triage import TriageExtraction


def build_memory_checkpointer() -> MemorySaver:
    state_models = (
        BusinessDecision, HumanReviewDecision, PolicyChunk, VehicleContractContext, TriageExtraction,
    )
    return MemorySaver(serde=JsonPlusSerializer(
        allowed_msgpack_modules=[(model.__module__, model.__name__) for model in state_models],
    ))
