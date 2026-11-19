#!/usr/bin/env python3
"""Integrate the Kaggle LLaMA-3.2 50-fact ordered-path discovery lane.

Pulls the downloaded kernel output into the repo release layout, assembles the
manifest (same shape as the recorded 20-fact lane), and re-runs two local
verifications:

1. the kernel's on-kernel consistency recompute (summary aggregates, percentile
   stats, pass verdict, model identity, config echo, claim boundary) against
   the downloaded report - the same code the kernel ran before download;
2. an independent tokenizer-scan expectation: the recorded case scan order and
   skipped subjects must match a locally recomputed scan of the FACTS pool with
   the provider tokenizer.json (single-token answer validity), from
   kaggle-staging/expected-scan.json (regenerate with scripts/llama50_preflight.py
   semantics if missing).

Usage:
    python3 scripts/gpu/integrate-llama-50-facts.py \
        --download-dir /tmp/gcl-llama-50-out \
        --expected-scan /home/z/my-project/kaggle-staging/expected-scan.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LANE_DIR = REPO_ROOT / "benchmarks/release/nn-live-llama-swiglu-ordered-residual-path-circuit-finder"
VARIANT = "llama-3.2-3b-50-facts-ordered-residual-path"
TARGET_DIR = LANE_DIR / VARIANT
KERNEL_PY = REPO_ROOT / "scripts/gpu/kaggle/llama-50-facts/kernel.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-dir", required=True, help="kaggle kernels output directory")
    parser.add_argument("--expected-scan", default="/home/z/my-project/kaggle-staging/expected-scan.json")
    return parser.parse_args()


def load_kernel_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("llama50_kernel", KERNEL_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git_state() -> dict:
    def run(args: list[str]) -> str:
        return subprocess.run(args, cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()

    return {
        "commit": run(["git", "rev-parse", "HEAD"]),
        "statusShort": run(["git", "status", "--short"]) or "(clean)",
    }


def main() -> None:
    args = parse_args()
    download = Path(args.download_dir)
    report_path = download / "artifacts" / VARIANT / "report.json"
    md_path = download / "artifacts" / VARIANT / "report.md"
    run_info_path = download / "artifacts" / VARIANT / "run-info.json"
    run_summary_path = download / "artifacts" / "run-summary.json"

    missing = [p.name for p in (report_path, md_path, run_info_path) if not p.is_file()]
    if missing:
        raise SystemExit(f"downloaded output is missing files: {missing}")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    summary = report.get("summary", {}) or {}

    # 1. consistency recompute (same code the kernel ran on-kernel).
    kernel = load_kernel_module()
    errors = kernel.consistency_errors(report)
    if errors:
        print("CONSISTENCY ERRORS:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("CONSISTENCY_RECOMPUTE_OK")

    # 2. independent tokenizer-scan expectation.
    scan_errors: list[str] = []
    expected_path = Path(args.expected_scan)
    if expected_path.is_file():
        expected = json.loads(expected_path.read_text(encoding="utf-8"))
        recorded_subjects = [str(case.get("subject")) for case in report.get("cases", [])]
        if recorded_subjects != expected["scanned"]:
            scan_errors.append(
                f"recorded scan order differs from the local tokenizer expectation "
                f"({len(recorded_subjects)} vs {len(expected['scanned'])} entries)"
            )
        recorded_skips = [str(case.get("subject")) for case in report.get("cases", []) if case.get("status") == "skipped"]
        if sorted(recorded_skips) != sorted(expected["expectedSkips"]):
            scan_errors.append("recorded skipped subjects differ from the local tokenizer expectation")
        if summary.get("skippedCases") != len(expected["expectedSkips"]):
            scan_errors.append("skippedCases does not match the local tokenizer expectation")
    else:
        scan_errors.append(f"expected-scan file missing: {expected_path}")
    if scan_errors:
        print("TOKENIZER_SCAN_ERRORS:")
        for error in scan_errors:
            print(f"  - {error}")
    else:
        print("TOKENIZER_SCAN_EXPECTATION_OK (scan order + skip set match)")

    # Integrate into the repo release layout.
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    TARGET_DIR.joinpath("report.json").write_text(report_path.read_text(encoding="utf-8"), encoding="utf-8")
    TARGET_DIR.joinpath("report.md").write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")
    TARGET_DIR.joinpath("kernel-run-info.json").write_text(run_info_path.read_text(encoding="utf-8"), encoding="utf-8")
    if run_summary_path.is_file():
        TARGET_DIR.joinpath("kernel-run-summary.json").write_text(run_summary_path.read_text(encoding="utf-8"), encoding="utf-8")
    for log_candidate in ("gcl-llama-50-facts-ordered-path.log", "kernel.log"):
        candidate = download / log_candidate
        if candidate.is_file():
            TARGET_DIR.joinpath("kernel-log.txt").write_text(candidate.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
            break

    manifest = {
        "schemaVersion": 1,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "lane": "nn-live-llama-swiglu-ordered-residual-path-circuit-finder",
        "variant": VARIANT,
        "status": "pass" if summary.get("pass") else "fail",
        "command": {
            "argv": [
                "kaggle kernels push",
                "scripts/gpu/kaggle/llama-50-facts (kernel.py v1, dataset kuglipista/gcl-llama-ordered-input)",
            ],
            "kernel": "kuglipista/gcl-llama-50-facts-ordered-path",
            "runner": "tools/llm-reader-gpt2-ordered-residual-path-sweep.py",
            "runnerArgv": json.loads(run_info_path.read_text(encoding="utf-8")).get("laneArgv"),
        },
        "env": json.loads(run_info_path.read_text(encoding="utf-8")),
        "git": git_state(),
        "outputs": {
            "reportJson": f"benchmarks/release/nn-live-llama-swiglu-ordered-residual-path-circuit-finder/{VARIANT}/report.json",
            "reportMarkdown": f"benchmarks/release/nn-live-llama-swiglu-ordered-residual-path-circuit-finder/{VARIANT}/report.md",
            "kernelRunInfo": f"benchmarks/release/nn-live-llama-swiglu-ordered-residual-path-circuit-finder/{VARIANT}/kernel-run-info.json",
            "manifest": f"benchmarks/release/nn-live-llama-swiglu-ordered-residual-path-circuit-finder/{VARIANT}/manifest.json",
        },
        "summary": summary,
        "localVerification": {
            "consistencyRecompute": "pass" if not errors else errors,
            "tokenizerScanExpectation": "pass" if not scan_errors else scan_errors,
        },
        "notes": [
            "Real Llama-3.2-3B-Instruct 4-bit CUDA gate through the model-agnostic LLMOracleAdapter, run on a Kaggle T4 kernel.",
            "The recorded 20-fact lane configuration with only case-count/min-valid-cases scaled 20 -> 50 (roadmap item 6: scale the fact set); the full 90-entry FACTS corpus scanned until 50 single-token-valid facts.",
            "The addressable MLP unit is the real SiLU(gate_proj(x))*up_proj(x) pair before down_proj. The benchmark uses causal pair ablations, attention output-path interventions, ordered path controls, and no synthetic teacher or fitted surrogate.",
            "On-kernel consistency recompute passed before download; local integration re-ran the same recompute plus an independent tokenizer-scan expectation over the recorded case order and skip set.",
        ],
    }
    TARGET_DIR.joinpath("manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({
        "integrated": str(TARGET_DIR),
        "summary": {k: summary.get(k) for k in ("pass", "validCases", "passedCases", "failedCases", "skippedCases", "passRate", "combinedGraphPassRate")},
        "consistencyRecompute": "pass" if not errors else "fail",
        "tokenizerScan": "pass" if not scan_errors else "fail",
    }, indent=1))

    if errors or scan_errors:
        raise SystemExit(1)
    print("GCL_LLAMA_50_FACTS_LANE_INTEGRATED_AND_VERIFIED")


if __name__ == "__main__":
    main()
