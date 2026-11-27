import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from evals.runner import run_evals


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Hebrew automotive triage golden dataset.")
    parser.add_argument("--live", action="store_true", help="Use Gemini for extraction and drafting.")
    parser.add_argument("--output", type=Path, help="Write a JSON report to this path.")
    options = parser.parse_args()
    if options.live:
        load_dotenv()
        if not any(
            os.getenv(name, "").strip() not in {"", "your_key_here"}
            for name in ("GOOGLE_API_KEY", "GEMINI_API_KEY")
        ):
            parser.error("Set GOOGLE_API_KEY or GEMINI_API_KEY in .env for live evals.")

    report = run_evals(live=options.live)
    print("| Mode | Passed | Pass rate | Average ms | P95 ms |")
    print("| --- | ---: | ---: | ---: | ---: |")
    print(
        f"| {report['mode']} | {report['passed_cases']}/{report['total_cases']} | "
        f"{report['pass_rate']:.0%} | {report['average_latency_ms']:.1f} | "
        f"{report['p95_latency_ms']:.1f} |"
    )
    print("\n| Check | Pass rate |\n| --- | ---: |")
    for name, rate in report["per_check_pass_rates"].items():
        print(f"| {name} | {rate:.0%} |")
    for case in report["cases"]:
        if not case["passed"]:
            failed = ", ".join(name for name, passed in case["checks"].items() if not passed)
            print(f"\n- {case['id']}: {failed}" + (f" ({case['error']})" if case["error"] else ""))
    if options.output:
        options.output.parent.mkdir(parents=True, exist_ok=True)
        options.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return int(report["passed_cases"] != report["total_cases"])


if __name__ == "__main__":
    raise SystemExit(main())
