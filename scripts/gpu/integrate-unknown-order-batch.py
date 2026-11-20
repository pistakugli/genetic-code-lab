#!/usr/bin/env python3
"""Integrate the Kaggle unknown-order ladder batch kernel output.

For every case directory in the downloaded kernel output (M-E1-null,
M-M15, M-L28, L-E1, L-M14, L-L27):

1. run the independent verifier (tools/llm_reader/verify_unknown_order_empirical.py)
   against the downloaded case dir, with the pool from the pools dataset
   download and the case's run-info/teacher-manifest - full grading
   recompute included (teacher-side response tables are present in the
   kernel output);
2. copy the light files (case report, fits, run-info, teacher manifest,
   pool sampler manifest) into the repo release layout under
   benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/
   unknown-order-empirical-<case>-<date>/;
3. write kernel-evidence.json with the heavy-file hashes (solver-input pool
   copy, teacher-side eval-responses) and a lane manifest with the same
   shape as the M-E1 integration.

Usage:
    python3 scripts/gpu/integrate-unknown-order-batch.py \
        --download-dir /tmp/gcl-unknown-order-batch-out \
        --pools-dir /tmp/ds-pools-v2
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LANE_DIR = REPO_ROOT / "benchmarks/release/nn-live-blind-swiglu-oracle-decompiler"
VERIFIER = REPO_ROOT / "tools/llm_reader/verify_unknown_order_empirical.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-dir", required=True)
    parser.add_argument("--pools-dir", required=True)
    parser.add_argument("--variant-date", default="2026-11-20")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def light_fit_files(fit_dir: Path) -> list[Path]:
    wanted = ("solver-report.json", "solver-report.md", "oracle-witness.json", "frozen-recovery.npz", "solver-stdout.log")
    return [fit_dir / name for name in wanted if (fit_dir / name).is_file()]


def main() -> int:
    args = parse_args()
    download = Path(args.download_dir).resolve()
    pools_dir = Path(args.pools_dir).resolve()
    if not download.is_dir():
        raise SystemExit(f"missing download dir: {download}")
    if not pools_dir.is_dir():
        raise SystemExit(f"missing pools dir: {pools_dir}")

    run_summary_path = download / "run-summary.json"
    if not run_summary_path.is_file():
        raise SystemExit(f"missing run summary: {run_summary_path}")
    run_summary = json.loads(run_summary_path.read_text(encoding="utf-8"))

    integrated = []
    failures = []
    for case in run_summary.get("cases", []):
        label = case.get("case")
        if not label:
            continue
        case_dir = download / label
        ladder_dir = case_dir / "ladder"
        if not (ladder_dir / "case-report.json").is_file():
            failures.append(f"{label}: no case report in kernel output")
            continue
        pool_path = pools_dir / f"{label.split('-null')[0]}-empirical-pool.npz"
        sampler_manifest_path = pools_dir / f"{label.split('-null')[0]}-sampler-manifest.json"

        verifier = subprocess.run(
            [
                sys.executable,
                str(VERIFIER),
                "--case-dir", str(ladder_dir),
                "--pool", str(pool_path),
                "--run-info", str(case_dir / "run-info.json"),
                "--teacher-manifest", str(case_dir / "teacher-manifest.json"),
            ],
            capture_output=True,
            text=True,
        )
        if verifier.returncode != 0:
            failures.append(f"{label}: verifier failed:\n{verifier.stdout}{verifier.stderr}")
            continue

        variant = f"unknown-order-empirical-{label.lower()}-{args.variant_date}"
        target = LANE_DIR / variant
        if target.exists():
            shutil.rmtree(target)
        (target / "ladder" / "fits").mkdir(parents=True)
        (target / "pool-provenance").mkdir(parents=True)
        shutil.copy2(case_dir / "run-info.json", target / "run-info.json")
        shutil.copy2(case_dir / "teacher-manifest.json", target / "teacher-manifest.json")
        shutil.copy2(ladder_dir / "case-report.json", target / "ladder" / "case-report.json")
        shutil.copy2(ladder_dir / "case-report.md", target / "ladder" / "case-report.md")
        for fit_dir in sorted((ladder_dir / "fits").glob("m*")):
            if not fit_dir.is_dir():
                continue
            fit_target = target / "ladder" / "fits" / fit_dir.name
            fit_target.mkdir(parents=True)
            for file_path in light_fit_files(fit_dir):
                shutil.copy2(file_path, fit_target / file_path.name)
        if sampler_manifest_path.is_file():
            shutil.copy2(sampler_manifest_path, target / "pool-provenance" / sampler_manifest_path.name)

        heavy = []
        pool_copy = ladder_dir / "solver-inputs" / "empirical-pool.npz"
        if pool_copy.is_file():
            heavy.append(
                {
                    "path": "ladder/solver-inputs/empirical-pool.npz",
                    "sizeBytes": pool_copy.stat().st_size,
                    "sha256": sha256_file(pool_copy),
                    "note": "solver-visible pool copy (x values and split indices only)",
                }
            )
        eval_responses = ladder_dir / "teacher-side" / "eval-responses.npz"
        if eval_responses.is_file():
            heavy.append(
                {
                    "path": "ladder/teacher-side/eval-responses.npz",
                    "sizeBytes": eval_responses.stat().st_size,
                    "sha256": sha256_file(eval_responses),
                    "note": "teacher-side grading tables for the full verifier recompute",
                }
            )

        null_note = None
        case_report = json.loads((ladder_dir / "case-report.json").read_text(encoding="utf-8"))
        if case_report.get("nullControl"):
            null_note = case_report["nullControl"].get("verdict")

        kernel_evidence = {
            "schemaVersion": 1,
            "kind": "gcl-kaggle-kernel-evidence",
            "recordedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "kernels": {
                "batch": {
                    "slug": "kuglipista/gcl-unknown-order-ladder-batch",
                    "kernelVersion": 1,
                    "preregistration": "docs/research/2026-11-19-unknown-order-empirical-swiglu-recovery.md",
                }
            },
            "datasets": {
                "toolsInput": "kuglipista/gcl-unknown-order-input",
                "pools": "kuglipista/gcl-unknown-order-pools",
            },
            "excludedHeavyFiles": heavy,
        }
        (target / "kernel-evidence.json").write_text(json.dumps(kernel_evidence, indent=2) + "\n", encoding="utf-8")

        manifest = {
            "schemaVersion": 1,
            "generatedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "lane": "nn-live-blind-swiglu-oracle-decompiler",
            "variant": variant,
            "status": "pass",
            "command": {"argv": ["kaggle kernel gcl-unknown-order-ladder-batch case " + label], "cwd": "/kaggle/working"},
            "outputs": {
                "reportJson": f"{LANE_DIR.name}/{variant}/ladder/case-report.json",
                "reportMarkdown": f"{LANE_DIR.name}/{variant}/ladder/case-report.md",
                "manifest": f"{LANE_DIR.name}/{variant}/manifest.json",
            },
            "summary": {
                "pass": True,
                "verdict": case_report.get("verdict"),
                "inferredOrder": case_report.get("inferredOrder"),
                "trueM": case.get("trueM"),
                "orderCorrect": case_report.get("inferredOrder") == case.get("trueM"),
                "upperRegionNonIdentifiable": case_report.get("upperRegionNonIdentifiable"),
                "nullControlVerdict": null_note,
            },
            "notes": [
                "Recorded by the batch kernel against the frozen protocol; independently verified locally (selection rule, grades recompute, pool sha256, teacher secret, access audits).",
                "Heavy files stay in the kernel output download; sha256 provenance in kernel-evidence.json.",
            ],
        }
        (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        integrated.append(
            {
                "case": label,
                "variant": variant,
                "verdict": case_report.get("verdict"),
                "inferredOrder": case_report.get("inferredOrder"),
                "trueM": case.get("trueM"),
                "nullControlVerdict": null_note,
            }
        )

    print(json.dumps({"integrated": integrated, "failures": failures}, indent=2))
    if failures:
        for failure in failures:
            print("ERROR: " + failure)
        return 1
    print("UNKNOWN_ORDER_BATCH_INTEGRATED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
