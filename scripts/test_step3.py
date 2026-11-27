import json
import os
import sys
from uuid import uuid4

from dotenv import load_dotenv
from fastapi.encoders import jsonable_encoder
from langgraph.types import Command
from langsmith import tracing_context

from scripts.test_step1 import TRANSCRIPT, format_smoke_error
from src.agent.checkpoint import build_memory_checkpointer
from src.agent.graph import build_triage_graph
from src.schemas.decision import HumanReviewDecision
from src.services.response_drafter import GeminiResponseDrafter, asks_for_plate


def print_result(label: str, result: dict) -> None:
    print(label)
    print(json.dumps(jsonable_encoder({
        "triage": result.get("extraction"),
        "erp_status": result.get("erp_status"),
        "decision": result.get("decision"),
        "final_status": result.get("final_status"),
        "final_reply_he": result.get("final_reply_he"),
        "draft_reply_he": result.get("draft_reply_he"),
        "retrieved_policies": [chunk.title for chunk in result.get("retrieved_policies", [])],
    }), ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv()
    if not any(
        os.getenv(name, "").strip() not in {"", "your_key_here"}
        for name in ("GOOGLE_API_KEY", "GEMINI_API_KEY")
    ):
        raise SystemExit("Set GOOGLE_API_KEY or GEMINI_API_KEY before running the smoke test.")

    graph = build_triage_graph(
        response_drafter=GeminiResponseDrafter(), checkpointer=build_memory_checkpointer(),
    )
    maintenance_config = {"configurable": {"thread_id": str(uuid4())}}
    accident_config = {"configurable": {"thread_id": str(uuid4())}}
    try:
        with tracing_context(enabled=False):
            maintenance = graph.invoke({
                "raw_input": "מספר רכב 482-91-302 בליסינג תפעולי. מתי צריך לעשות טיפול תקופתי?",
            }, config=maintenance_config)
            print_result("Scenario 1 — periodic maintenance", maintenance)

            accident = graph.invoke({"raw_input": TRANSCRIPT}, config=accident_config)
            print_result("Scenario 2 — accident before review", accident)
            interrupts = accident.get("__interrupt__")
            failures = []
            reply = maintenance.get("final_reply_he") or ""
            policies = maintenance["retrieved_policies"]
            if (
                maintenance["final_status"] != "auto_completed"
                or maintenance.get("__interrupt__")
                or maintenance["erp_status"] != "found"
                or not policies or policies[0].doc_id != "warranty_and_maintenance"
                or "טיפול" not in reply or asks_for_plate(reply)
                or any(term in reply for term in ("גרירה", "גרר", "רכב חלופי"))
            ):
                failures.append("Maintenance did not auto-complete with a focused, plate-free answer.")
            if not interrupts or accident["final_status"] != "awaiting_human_review":
                failures.append("The accident did not pause for review.")
            if interrupts:
                packet = interrupts[0].value
                if (
                    packet["erp_status"] != "found"
                    or (packet["vehicle_context"] or {}).get("contract_type") != "operational_leasing"
                    or not {"angry_sentiment", "high_estimated_cost"}.issubset(packet["reasons"])
                    or packet["triage"]["estimated_cost_nis"] != 12_000
                    or not {"accidents_and_deductible", "towing_and_roadside"}.intersection(
                        chunk.doc_id for chunk in accident["retrieved_policies"]
                    )
                    or not {"israeli_id", "phone"}.issubset(accident["redacted_pii_types"])
                ):
                    failures.append("The accident lacked expected ERP, policy, PII or risk evidence.")
                review = HumanReviewDecision(
                    approved=True,
                    reviewer_id="smoke-reviewer",
                    edited_reply_he=f'{packet["draft_reply_he"].rstrip()} תודה על הסבלנות.',
                )
                final = graph.invoke(Command(resume=review.model_dump()), config=accident_config)
                print_result("Scenario 2 — accident after approval", final)
                if final["final_status"] != "approved_by_human" or final["final_reply_he"] != review.edited_reply_he:
                    failures.append("The review did not apply the approved edit.")
            if failures:
                raise SystemExit("\n".join(failures))
    except Exception as error:
        raise SystemExit(format_smoke_error(error)) from None


if __name__ == "__main__":
    main()
