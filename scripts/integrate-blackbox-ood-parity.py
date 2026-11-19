#!/usr/bin/env python3
"""Integrate the Kaggle blackbox OOD parity kernel artifacts into the repo.

Copies the kernel-produced report.json/report.md for both instruct families
into benchmarks/release/nn-live-blackbox-llm-behavior-reader/<variant>/,
assembles the manifest.json for each variant (kernel invocation, git
provenance, summary, notes), and embeds the per-family kernel run info as
kernel-run-info.json (the same integration pattern as the recorded OOD
replacement-transfer lanes).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path("/home/z/my-project/genetic-code-lab")
SRC = Path("/tmp/blackbox-kernel-final/artifacts")
LANE_DIR = REPO / "benchmarks/release/nn-live-blackbox-llm-behavior-reader"

FAMILIES = [
    {
        "key": "llama",
        "variant": "llama-3.2-3b-4bit-output-only-ood-20-facts",
    },
    {
        "key": "mistral",
        "variant": "mistral-7b-4bit-output-only-ood-20-facts",
    },
]

KERNEL_SLUG = "kuglipista/gcl-black-box-ood-parity-llama-3-2-mistral-7b"
KERNEL_VERSION = 4


def git_state() -> dict:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True
    ).stdout.strip()
    status_short = subprocess.run(
        ["git", "status", "--short"], cwd=REPO, capture_output=True, text=True
    ).stdout.strip()
    diff = subprocess.run(
        ["git", "diff"], cwd=REPO, capture_output=True, text=True
    ).stdout
    return {
        "commit": commit,
        "statusShort": status_short,
        "uncommittedDiffSha256": hashlib.sha256(diff.encode("utf-8")).hexdigest(),
    }


def main() -> None:
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    git = git_state()
    run_summary = json.loads((SRC / "run-summary.json").read_text(encoding="utf-8"))
    by_key = {entry["familyKey"]: entry for entry in run_summary["families"]}

    for family in FAMILIES:
        variant = family["variant"]
        src_dir = SRC / variant
        dst_dir = LANE_DIR / variant
        dst_dir.mkdir(parents=True, exist_ok=True)

        report = json.loads((src_dir / "report.json").read_text(encoding="utf-8"))
        summary = report["summary"]

        for name in ("report.json", "report.md"):
            (dst_dir / name).write_text((src_dir / name).read_text(encoding="utf-8"))

        run_info = {
            **by_key[family["key"]],
            "gpuName": run_summary.get("gpuName"),
            "torch": run_summary.get("torch"),
            "pinnedDeps": run_summary.get("pinnedDeps"),
            "kernelSlug": KERNEL_SLUG,
            "kernelVersion": KERNEL_VERSION,
            "kernelDataset": run_summary.get("dataset"),
        }
        (dst_dir / "kernel-run-info.json").write_text(
            json.dumps(run_info, indent=2) + "\n", encoding="utf-8"
        )

        manifest = {
            "schemaVersion": 1,
            "generatedAt": now,
            "lane": "nn-live-blackbox-llm-behavior-reader",
            "variant": variant,
            "status": "pass" if summary.get("pass") else "fail",
            "command": {
                "kernelSlug": KERNEL_SLUG,
                "kernelVersion": KERNEL_VERSION,
                "argv": [
                    "python3",
                    "tools/llm-reader-blackbox-behavior.py",
                    f"--model {run_info['model']}",
                    "--case-count 20",
                    f"--min-valid-cases {run_info['minValidCases']}",
                    "--paraphrase-count 6",
                    "--paraphrase-profile ood",
                    "--device cuda",
                    "--load-in-4bit",
                ],
                "note": "produced by the Kaggle GPU kernel; see kernel-run-info.json",
            },
            "env": {
                "gpu": run_summary.get("gpuName"),
                "torch": run_summary.get("torch"),
                "pinnedDeps": run_summary.get("pinnedDeps"),
            },
            "git": git,
            "outputs": {
                "reportJson": f"benchmarks/release/nn-live-blackbox-llm-behavior-reader/{variant}/report.json",
                "reportMarkdown": f"benchmarks/release/nn-live-blackbox-llm-behavior-reader/{variant}/report.md",
                "manifest": f"benchmarks/release/nn-live-blackbox-llm-behavior-reader/{variant}/manifest.json",
            },
            "summary": summary,
            "notes": [
                "Output/logit-only whole-model factual behavior reader on the quantized unsloth instruct provider, OOD paraphrase profile, full 20-fact pool.",
                "Solver receives prompt strings, target token ids, and next-token logits only; no weights, hooks, hidden activations, source ids, or ablation handles.",
                "Paraphrase protocol: the fixed six answer-blind OOD families; every family prompt is scored and kept with no margin-based selection.",
                "Produced on Kaggle T4 by kernel kuglipista/gcl-black-box-ood-parity-llama-3-2-mistral-7b v4 against dataset kuglipista/gcl-blackbox-ood-input (repo commit 1c98d5a + --load-in-4bit); on-kernel consistency recompute passed (family table, case aggregates, access audit, model identity).",
                f"Skipped facts are recorded, not dropped: {summary.get('skippedCases')} case(s) with non-single-token answers for this tokenizer.",
            ],
        }
        (dst_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        print(f"integrated {variant}: status={manifest['status']}")

    print("done")


if __name__ == "__main__":
    main()
