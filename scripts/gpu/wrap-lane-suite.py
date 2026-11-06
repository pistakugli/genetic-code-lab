#!/usr/bin/env python3
"""Assemble an ordered-path stability suite from per-model lane reports.

Python equivalent of the suite assembly in
scripts/benchmark-nn-live-ordered-path-stability.ts, so the full
suite + verifier cycle runs on hosts without Node.js (notebooks,
containers, plain GPU boxes). Produces the same three files the TS
release orchestrator writes:

    <output-dir>/report.json    suite report (kind ...-suite)
    <output-dir>/report.md      suite markdown table
    <output-dir>/manifest.json  command/env/git manifest

Each --result flag takes comma-separated key=value pairs:
    id, label, adapter, model, report (repo-relative path to the
    per-model lane report.json). "label" and "adapter" default to "id".

Status per result is "pass" only when the lane report summary says
pass=true (the TS orchestrator uses the child exit code instead; this
wrapper has no child process, so it reads the recorded summary).

Example:
    python scripts/gpu/wrap-lane-suite.py --paraphrase-profile ood \
      --result id=gpt2,label=GPT-2,model=gpt2,\
report=benchmarks/scratch/gpu-env-cpu-live/ood-replacement-transfer-full/report.json \
      --output-dir benchmarks/scratch/gpu-env-cpu-live/suite
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]

SUITES = {
    "ood": "nn-live-ordered-path-ood-stress-suite",
    "standard": "nn-live-ordered-path-stability-suite",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble an ordered-path stability suite report.")
    parser.add_argument("--paraphrase-profile", choices=["standard", "ood"], default="ood")
    parser.add_argument("--lane", default="nn-live-ordered-path-stability")
    parser.add_argument(
        "--result",
        action="append",
        required=True,
        help="comma-separated id,label,adapter,model,report=<path> fields",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--note",
        default="Assembled by scripts/gpu/wrap-lane-suite.py (python suite wrapper).",
    )
    return parser.parse_args()


def parse_result(spec: str, index: int) -> Dict[str, str]:
    fields: Dict[str, str] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise SystemExit(f"--result #{index} field {part!r} is not key=value")
        key, value = part.split("=", 1)
        fields[key.strip()] = value.strip()
    for required in ("id", "report"):
        if required not in fields or not fields[required]:
            raise SystemExit(f"--result #{index} is missing required field {required!r}")
    fields.setdefault("label", fields["id"])
    fields.setdefault("adapter", fields["id"])
    fields.setdefault("model", fields["id"])
    return fields


def git_state() -> Dict[str, Any]:
    def run(args: List[str]) -> str:
        try:
            return subprocess.run(
                args, cwd=REPO_ROOT, capture_output=True, text=True, timeout=30, check=False
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""

    return {
        "commit": run(["git", "rev-parse", "HEAD"]) or None,
        "statusShort": run(["git", "status", "--short"]) or None,
    }


def relevant_env() -> Dict[str, str]:
    keys = [
        "GCL_DEVICE",
        "GCL_PYTHON",
        "GCL_BENCHMARK_VARIANT",
        "GCL_LIVE_STABILITY_MODELS",
        "GCL_LIVE_STABILITY_PARAPHRASE_PROFILE",
        "GCL_LIVE_STABILITY_RESCUE",
        "HF_HOME",
    ]
    return {key: os.environ[key] for key in keys if os.environ.get(key)}


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    for index, spec in enumerate(args.result, start=1):
        fields = parse_result(spec, index)
        report_path = REPO_ROOT / fields["report"]
        if not report_path.is_file():
            raise SystemExit(f"missing lane report: {report_path}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        summary = report.get("summary", {})
        status = "pass" if summary.get("pass") is True else "fail"
        # Mirror the TS orchestrator layout: each model's report lives inside
        # the suite dir under its id, so the suite is self-contained.
        case_dir = output_dir / fields["id"]
        case_dir.mkdir(parents=True, exist_ok=True)
        case_report_path = case_dir / "report.json"
        case_report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        source_markdown = report_path.with_suffix(".md")
        if source_markdown.is_file():
            (case_dir / "report.md").write_text(
                source_markdown.read_text(encoding="utf-8"), encoding="utf-8"
            )
        results.append(
            {
                "id": fields["id"],
                "label": fields["label"],
                "adapter": fields["adapter"],
                "model": fields["model"],
                "status": status,
                "reportPath": str(case_report_path.relative_to(REPO_ROOT)),
                "artifactPath": f"{fields['id']}/report.json",
                "summary": summary,
            }
        )

    passed = [item for item in results if item["status"] == "pass"]
    suite_report = {
        "schemaVersion": 1,
        "kind": SUITES[args.paraphrase_profile],
        "paraphraseProfile": args.paraphrase_profile,
        "synthetic": False,
        "surrogate": False,
        "generatedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "summary": {
            "pass": len(passed) == len(results) and bool(results),
            "cases": len(results),
            "passed": len(passed),
            "failed": [item["id"] for item in results if item["status"] != "pass"],
        },
        "results": results,
    }
    suite_path = output_dir / "report.json"
    suite_path.write_text(json.dumps(suite_report, indent=2) + "\n", encoding="utf-8")

    markdown_lines = [
        "# Live Ordered Path Stability Suite",
        "",
        "This suite replays discovered real-model circuits across paraphrase and "
        "neighboring-layer controls. It does not fit a surrogate.",
        "",
        f"- Paraphrase profile: `{args.paraphrase_profile}`",
        f"- Passed: `{len(passed)}/{len(results)}`",
        "",
        "| Model | Status | Passed cases | Paraphrase pass rate | Layer specificity | Failed subjects | Report |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in results:
        summary = item["summary"]
        passed_cases = summary.get("passedCases", "?")
        case_count = summary.get("caseCount", "?")
        failed_subjects = ",".join(summary.get("failedSubjects", [])) or "[]"
        markdown_lines.append(
            f"| {item['label']} | {item['status']} | {passed_cases}/{case_count} "
            f"| {summary.get('paraphrasePassRate', '?')} "
            f"| {summary.get('layerSpecificityRate', '?')} "
            f"| {failed_subjects} | {item['reportPath']} |"
        )
    (output_dir / "report.md").write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")

    manifest = {
        "schemaVersion": 1,
        "generatedAt": suite_report["generatedAt"],
        "lane": args.lane,
        "variant": output_dir.name,
        "status": "pass" if suite_report["summary"]["pass"] else "fail",
        "command": {
            "argv": [sys.executable, str(Path(__file__).resolve())] + sys.argv[1:],
            "cwd": str(Path.cwd()),
            "python": sys.version.split()[0],
        },
        "env": relevant_env(),
        "git": git_state(),
        "outputs": {
            "report": str(suite_path.relative_to(REPO_ROOT)),
            "markdown": str((output_dir / "report.md").relative_to(REPO_ROOT)),
        },
        "note": args.note,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "pass": suite_report["summary"]["pass"],
                "output": str(suite_path.relative_to(REPO_ROOT)),
                "results": [
                    {"id": item["id"], "status": item["status"]} for item in results
                ],
            },
            indent=2,
        )
    )
    sys.exit(0 if suite_report["summary"]["pass"] else 1)


if __name__ == "__main__":
    main()
