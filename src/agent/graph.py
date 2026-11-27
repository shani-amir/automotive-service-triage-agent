from functools import partial
from typing import Literal

from langgraph.graph import END, START, StateGraph

from src.agent.nodes import (
    draft_response,
    evaluate_rules,
    extract_triage_node,
    finalize_response,
    human_review,
    lookup_erp,
    mask_pii_node,
    retrieve_policies,
)
from src.agent.state import TriageState
from src.services.erp_repo import VehicleRepository
from src.services.policy_rag import PolicyRetriever
from src.services.response_drafter import ResponseDrafter


def route_after_triage(state: TriageState) -> Literal["lookup_erp", "retrieve_policies"]:
    plate = state["extraction"].license_plate
    return "lookup_erp" if plate and plate.strip() else "retrieve_policies"


def route_after_draft(state: TriageState) -> Literal["human_review", "finalize_response"]:
    return "human_review" if state["decision"].requires_human_approval else "finalize_response"


def build_triage_graph(
    llm=None,
    vehicle_repo: VehicleRepository | None = None,
    policy_retriever: PolicyRetriever | None = None,
    response_drafter: ResponseDrafter | None = None,
    checkpointer=None,
    enable_hitl: bool | None = None,
):
    if enable_hitl is None:
        enable_hitl = checkpointer is not None
    if enable_hitl and checkpointer is None:
        raise ValueError("Human review interrupts require a checkpointer.")

    builder = StateGraph(TriageState)
    builder.add_node("mask_pii", mask_pii_node)
    builder.add_node("extract_triage", partial(extract_triage_node, llm=llm))
    builder.add_node("lookup_erp", partial(lookup_erp, vehicle_repo=vehicle_repo))
    builder.add_node("retrieve_policies", partial(retrieve_policies, policy_retriever=policy_retriever))
    builder.add_node("evaluate_rules", evaluate_rules)
    builder.add_node("draft_response", partial(draft_response, response_drafter=response_drafter))
    builder.add_node("human_review", partial(human_review, enable_hitl=enable_hitl))
    builder.add_node("finalize_response", finalize_response)
    builder.add_edge(START, "mask_pii")
    builder.add_edge("mask_pii", "extract_triage")
    builder.add_conditional_edges("extract_triage", route_after_triage)
    builder.add_edge("lookup_erp", "retrieve_policies")
    builder.add_edge("retrieve_policies", "evaluate_rules")
    builder.add_edge("evaluate_rules", "draft_response")
    builder.add_conditional_edges("draft_response", route_after_draft)
    builder.add_edge("human_review", END)
    builder.add_edge("finalize_response", END)
    return builder.compile(checkpointer=checkpointer)


graph = build_triage_graph()
