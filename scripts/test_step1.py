import os
import re
import sys
from time import perf_counter
from uuid import UUID

from dotenv import load_dotenv
from google.genai.errors import APIError
from langchain_core.callbacks import BaseCallbackHandler
from langsmith import tracing_context

from src.agent.graph import graph


TRANSCRIPT = (
    "אני תקוע בשוליים של כביש 6 לכיוון צפון. נכנסו בי מאחור והרכב לא מניע. "
    "אני ממש כועס, כבר חצי שעה אני מחכה לעזרה! לא נראה שיש פצועים, אבל צריך גרר דחוף. "
    "מספר הרכב הוא 482-91-302, תעודת הזהות שלי 123456782 והטלפון 052-555-0199. "
    "המוסך העריך שהתיקון יעלה 12000 ש״ח."
)


class NodeTimings(BaseCallbackHandler):
    def __init__(self) -> None:
        self._started: dict[UUID, tuple[str, float]] = {}

    def on_chain_start(self, serialized, inputs, *, run_id, name=None, **kwargs):
        if name in {"mask_pii", "extract_triage"}:
            self._started[run_id] = (name, perf_counter())
            print(f"{name} started", flush=True)

    def on_chain_end(self, outputs, *, run_id, **kwargs):
        self._finish(run_id, "finished")

    def on_chain_error(self, error, *, run_id, **kwargs):
        self._finish(run_id, "failed")

    def _finish(self, run_id: UUID, status: str) -> None:
        if run_id in self._started:
            name, started = self._started.pop(run_id)
            print(f"{name} {status} in {(perf_counter() - started) * 1000:.1f} ms", flush=True)


def format_smoke_error(error: Exception) -> str:
    status = None
    current: BaseException | None = error
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        response = getattr(current, "response", None)
        for candidate in (
            getattr(response, "status_code", None),
            getattr(current, "status_code", None),
            getattr(current, "code", None),
        ):
            if type(candidate) is int and 100 <= candidate <= 599:
                status = candidate
                break
        if status is not None:
            break
        current = current.__cause__ or current.__context__

    detail = str(error.message or "") if isinstance(error, APIError) else str(error)
    # Drop URLs, which can include credentials in query parameters.
    detail = re.sub(r"https?://[^\s'\"]+", "[URL redacted]", detail, flags=re.IGNORECASE)
    for variable in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        key = os.getenv(variable, "").strip()
        if key:
            detail = detail.replace(key, "[redacted]")
    detail = re.sub(r"AIza[A-Za-z0-9_-]+", "[redacted]", detail)
    detail = " ".join(detail.split())[:500] or "No error message provided."
    status_label = f", HTTP {status}" if status is not None else ""
    return f"Smoke test failed ({type(error).__name__}{status_label}): {detail}"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv()
    if os.getenv("GOOGLE_API_KEY", "").strip() in {"", "your_key_here"}:
        raise SystemExit("Set GOOGLE_API_KEY in the project's .env before running the smoke test.")

    try:
        with tracing_context(enabled=False):
            result = graph.invoke(
                {"raw_input": TRANSCRIPT, "source_type": "call_transcript"},
                config={"callbacks": [NodeTimings()]},
            )
    except Exception as error:
        raise SystemExit(format_smoke_error(error)) from None
    print("Masked text:", result["masked_input"])
    print("Redacted PII types:", ", ".join(result["redacted_pii_types"]))
    print(result["extraction"].model_dump_json(indent=2))


if __name__ == "__main__":
    main()
