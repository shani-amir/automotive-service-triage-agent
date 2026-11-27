import json
import os
import sys

from dotenv import load_dotenv
from langsmith import tracing_context

from scripts.test_step1 import format_smoke_error
from src.agent.graph import graph
from src.services.pii import mask_pii


SCENARIOS = (
    (
        "ליסינג תפעולי עם מספר רכב מוכר",
        "פגעו בי מאחור והרכב לא מניע. הרכב עומד במקום בטוח ואין נפגעים. "
        "מספר הרכב הוא 482-91-302. אני צריך גרר ורכב חלופי כי המוסך צופה יומיים לתיקון. "
        "תעודת הזהות שלי 123456782 והטלפון 052-555-0199.",
        "found",
    ),
    (
        "שאלת מדיניות ללא מספר רכב",
        "מה המדיניות במקרה של תקלה בדרך? איך מזמינים גרר ומתי אפשר לקבל רכב חלופי?",
        "skipped_no_plate",
    ),
    (
        "מספר רכב לא מוכר",
        "מספר הרכב שלי הוא 999-88-777. הרכב לא מניע בחניה ואני צריך גרר. "
        "אפשר לבדוק גם זכאות לרכב חלופי אם התיקון יימשך יומיים?",
        "not_found",
    ),
)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv()
    if not any(
        os.getenv(name, "").strip() not in {"", "your_key_here"}
        for name in ("GOOGLE_API_KEY", "GEMINI_API_KEY")
    ):
        raise SystemExit("Set GOOGLE_API_KEY or GEMINI_API_KEY before running the smoke test.")

    for index, (title, transcript, expected_status) in enumerate(SCENARIOS, start=1):
        print(f"\n{index}. {title}", flush=True)
        try:
            with tracing_context(enabled=False):
                result = graph.invoke({"raw_input": transcript, "source_type": "call_transcript"})
        except Exception as error:
            raise SystemExit(format_smoke_error(error)) from None

        extraction = result["extraction"].model_dump()
        extraction["summary_he"] = mask_pii(extraction["summary_he"]).masked_text
        print("Triage", json.dumps(extraction, ensure_ascii=False, indent=2))
        print("Redacted PII types", ", ".join(result["redacted_pii_types"]) or "none")
        print("ERP status", result["erp_status"])
        vehicle = result["vehicle_context"]
        if vehicle is not None:
            print(
                f"Vehicle {vehicle.license_plate} · {vehicle.vehicle_make_model} {vehicle.vehicle_year}"
                f" · {vehicle.contract_type} ({vehicle.contract_status})"
            )
            print(
                f"Coverage · towing={vehicle.includes_towing} · replacement={vehicle.includes_replacement_car}"
                f" · deductible={vehicle.deductible_nis:,} NIS"
            )
        print("Retrieval filters", json.dumps(result["retrieval_filters"], ensure_ascii=False, indent=2))
        print("Top policy chunks")
        for chunk in result["retrieved_policies"]:
            print(f"  {chunk.score:.3f}  {chunk.title} — {chunk.section}")

        if result["erp_status"] != expected_status:
            raise SystemExit(f"Scenario {index} expected ERP status {expected_status}.")
        if not result["retrieved_policies"]:
            raise SystemExit(f"Scenario {index} returned no policy chunks.")
        if expected_status != "found" and result["retrieval_filters"]["contract_type"] is not None:
            raise SystemExit(f"Scenario {index} unexpectedly applied a contract filter.")
        if expected_status == "found":
            if result["retrieval_filters"]["contract_type"] != "operational_leasing":
                raise SystemExit("The known-plate scenario did not apply the operational leasing filter.")
            if not {"israeli_id", "phone"}.issubset(result["redacted_pii_types"]):
                raise SystemExit("The known-plate scenario did not report the expected PII redactions.")


if __name__ == "__main__":
    main()
