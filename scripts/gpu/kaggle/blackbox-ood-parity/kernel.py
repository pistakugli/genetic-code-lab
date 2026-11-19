#!/usr/bin/env python3
"""GCL output-only OOD paraphrase parity for LLaMA-3.2-3B and Mistral-7B (4-bit).

Roadmap item 5 remaining gate: instruct-family parity through the unchanged
prompt-to-logits-only solver boundary. Runs
tools/llm-reader-blackbox-behavior.py (dataset
kuglipista/gcl-blackbox-ood-input, refreshed to repo commit 1c98d5a + the
--load-in-4bit extension) with the OOD paraphrase profile over the full
20-fact pool for both quantized unsloth providers, on a Kaggle T4 GPU.

Steps:
1. print environment evidence (GPU, torch, versions);
2. install the pinned runtime (transformers/accelerate/bitsandbytes, the
   versions used by the recorded family artifacts);
3. pre-download both 4-bit providers into HF_HOME with a hard pre-flight
   gate (never run a cache-blind lane; v12 lesson);
4. per family: run the black-box behavior reader (OOD profile, full pool,
   min-valid-cases set by the tokenizer's single-token answer validity:
   19 for LLaMA-3.2, 9 for Mistral-7B; skipped facts are recorded in the
   artifact, not dropped);
5. on-kernel consistency recompute: family table, case aggregates, access
   audit, and model identity are recomputed from the raw case fields, so a
   malformed artifact is a kernel error;
6. write run-info per family (embedded in run-summary.json) and archive the
   artifact directories under /kaggle/working/artifacts.

An honest scientific fail (runner exit 1 with a consistent report) is a
recorded finding, not a kernel error.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

DS = Path("/kaggle/input/gcl-blackbox-ood-input")
TEMP = Path("/kaggle/temp")


def _dataset_listing() -> list[str]:
    rows: list[str] = []
    for root, _dirs, files in os.walk("/kaggle/input"):
        for name in files:
            rows.append(str(Path(root) / name))
    return sorted(rows)


def _resolve_tools_dir() -> Path:
    """Locate the runner + llm_reader package in the attached datasets.

    The dataset was uploaded with --dir-mode zip, and the kernel's mount name
    for the dataset is not guaranteed to match the dataset slug, so the
    pre-flight searches the whole /kaggle/input tree for the runner and the
    llm_reader package, unpacking any zip archive to /kaggle/temp if needed.
    The full mount listing is printed for provenance.
    """
    print("[dataset] /kaggle/input listing:", flush=True)
    for row in _dataset_listing():
        print(f"[dataset] {row}", flush=True)

    candidates: list[Path] = []
    for row in _dataset_listing():
        if row.endswith("llm-reader-blackbox-behavior.py"):
            candidates.append(Path(row).parent)
        if row.endswith(".zip"):
            target = TEMP / (Path(row).stem + "-unpacked")
            target.mkdir(parents=True, exist_ok=True)
            extract = sh(["unzip", "-o", row, "-d", str(target)])
            if extract.returncode != 0:
                print(f"[dataset] unzip failed: {row}", flush=True)
                continue
            print(f"[dataset] extracted {row} -> {target}", flush=True)
            candidates.append(target)
            candidates.append(target / "tools")

    for candidate in candidates:
        if (candidate / "llm-reader-blackbox-behavior.py").is_file() and (
            candidate / "llm_reader" / "ood_paraphrases.py"
        ).is_file():
            return candidate
    raise SystemExit(
        "dataset pre-flight failed: llm-reader-blackbox-behavior.py and the "
        "llm_reader package are missing from /kaggle/input"
    )


TOOLS: Path
RUNNER: Path
WORK = Path("/kaggle/working")
ART = WORK / "artifacts"
HF_HOME = "/kaggle/temp/hf-cache"

PINNED = ["transformers==5.5.0", "accelerate==1.13.0", "bitsandbytes==0.49.2"]
PINNED_REPO_COMMIT = "1c98d5a"
PINNED_DATASET = "kuglipista/gcl-blackbox-ood-input"

FAMILIES = [
    {
        "key": "llama",
        "model": "unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit",
        "minValidCases": 19,
        "variant": "llama-3.2-3b-4bit-output-only-ood-20-facts",
    },
    {
        "key": "mistral",
        "model": "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
        "minValidCases": 9,
        "variant": "mistral-7b-4bit-output-only-ood-20-facts",
    },
]


def sh(cmd: list[str], env: dict | None = None) -> subprocess.CompletedProcess:
    print(f"+ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, env=env)


def env_info() -> dict:
    import torch

    return {
        "torch": torch.__version__,
        "cudaAvailable": bool(torch.cuda.is_available()),
        "gpuName": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpuCount": int(torch.cuda.device_count()),
        "python": sys.version.split()[0],
        "machine": os.uname().nodename if hasattr(os, "uname") else None,
    }


def run_family(family: dict) -> tuple[int, dict]:
    """Run one family lane; returns (runner exit code, run-info)."""
    variant_dir = ART / family["variant"]
    variant_dir.mkdir(parents=True, exist_ok=True)
    out_json = variant_dir / "report.json"
    out_md = variant_dir / "report.md"

    cmd = [
        sys.executable,
        str(RUNNER),
        "--model", family["model"],
        "--case-count", "20",
        "--min-valid-cases", str(family["minValidCases"]),
        "--control-count", "8",
        "--paraphrase-count", "6",
        "--paraphrase-profile", "ood",
        "--wrong-answer-count", "4",
        "--max-target-rank", "100",
        "--min-pass-rate", "1.0",
        "--min-subject-lift", "0.05",
        "--min-relation-lift", "0.02",
        "--min-paraphrase-retention-rate", "0.50",
        "--control-seed", "260709",
        "--device", "cuda",
        "--load-in-4bit",
        "--local-files-only",
        "--output-json", str(out_json),
        "--output-md", str(out_md),
    ]
    env = dict(os.environ)
    env["HF_HOME"] = HF_HOME
    started = time.time()
    result = sh(cmd, env=env)
    elapsed = round(time.time() - started)
    print(f"[{family['key']}] exit={result.returncode} elapsed={elapsed}s", flush=True)

    info = {
        "familyKey": family["key"],
        "model": family["model"],
        "variant": family["variant"],
        "minValidCases": family["minValidCases"],
        "runnerExit": result.returncode,
        "elapsedSeconds": elapsed,
        "pinnedRepoCommit": PINNED_REPO_COMMIT,
        "dataset": PINNED_DATASET,
    }
    return result.returncode, info


def consistency_errors(report: dict, family: dict) -> list[str]:
    """Recompute the recorded aggregates from the raw case fields.

    This is the on-kernel integrity check: every summary number must be
    recomputable from the raw measurements, the access audit must state the
    logit-only boundary, and the model identity must match the selected
    family. A mismatch is a kernel error, not a scientific finding.
    """
    errors: list[str] = []
    prefix = f"{family['key']}: "

    access = report.get("access", {}) or {}
    if access.get("solverAccess") != "prompt-to-next-token-logits-only":
        errors.append(prefix + "access audit does not state the logit-only boundary")
    if access.get("weightsUsedBySolver") is not False or access.get("hooksUsedBySolver") is not False:
        errors.append(prefix + "access audit admits weights or hooks")
    if access.get("activationsUsedBySolver") is not False:
        errors.append(prefix + "access audit admits activations")

    model = report.get("model", {}) or {}
    if model.get("id") != family["model"]:
        errors.append(prefix + "model identity does not match the family")
    if model.get("device") != "cuda":
        errors.append(prefix + "model device is not cuda")
    if model.get("quantized") is not True:
        errors.append(prefix + "model identity does not record the 4-bit provider")

    summary = report.get("summary", {}) or {}
    cases = report.get("cases", [])
    if not isinstance(cases, list) or not cases:
        errors.append(prefix + "no measured cases")
        return errors
    valid = [case for case in cases if isinstance(case, dict) and case.get("valid")]
    passed = [case for case in valid if case.get("pass")]
    if summary.get("validCases") != len(valid):
        errors.append(prefix + "validCases does not match the raw cases")
    if summary.get("passedCases") != len(passed):
        errors.append(prefix + "passedCases does not match the raw cases")
    if summary.get("skippedCases") != len(cases) - len(valid):
        errors.append(prefix + "skippedCases does not match the raw cases")
    expected_pass_rate = len(passed) / max(1, len(valid))
    recorded_pass_rate = summary.get("passRate")
    if not isinstance(recorded_pass_rate, (int, float)) or abs(float(recorded_pass_rate) - expected_pass_rate) > 1e-9:
        errors.append(prefix + "passRate does not match the raw cases")
    min_pass_rate = summary.get("thresholds", {}).get("minPassRate")
    expected_summary_pass = (
        len(valid) >= int(family["minValidCases"])
        and isinstance(min_pass_rate, (int, float))
        and expected_pass_rate >= float(min_pass_rate)
    )
    if summary.get("pass") is not expected_summary_pass:
        errors.append(prefix + "summary pass/fail does not match the recorded gates")

    paraphrases = [
        item
        for case in valid
        for item in case.get("paraphrases", [])
        if isinstance(item, dict) and isinstance(item.get("family"), str)
    ]
    for case in valid:
        case_items = [item for item in case.get("paraphrases", []) if isinstance(item, dict)]
        case_retention = (
            sum(1 for item in case_items if float(item.get("answerPoolMargin", 0.0)) > 0.0)
            / max(1, len(case_items))
        )
        recorded = case.get("metrics", {}).get("paraphraseRetentionRate")
        if not isinstance(recorded, (int, float)) or abs(float(recorded) - case_retention) > 1e-9:
            errors.append(prefix + f"case {case.get('subject')} retention does not match raw paraphrases")

    family_summary = summary.get("paraphraseFamilySummary", {})
    if not isinstance(family_summary, dict) or not family_summary:
        errors.append(prefix + "per-family table is missing")
    else:
        grouped: dict[str, list[dict]] = {}
        for item in paraphrases:
            grouped.setdefault(item["family"], []).append(item)
        if set(grouped) != set(family_summary):
            errors.append(prefix + "family table does not cover the recorded families")
        for fam, rows in sorted(grouped.items()):
            row = family_summary.get(fam, {})
            retained = sum(1 for item in rows if float(item.get("answerPoolMargin", 0.0)) > 0.0)
            if row.get("count") != len(rows):
                errors.append(prefix + f"{fam} count does not match raw paraphrases")
            if row.get("retained") != retained:
                errors.append(prefix + f"{fam} retained count does not match raw paraphrases")
            expected_rate = retained / max(1, len(rows))
            if not isinstance(row.get("retentionRate"), (int, float)) or abs(float(row["retentionRate"]) - expected_rate) > 1e-9:
                errors.append(prefix + f"{fam} retention rate does not match raw paraphrases")

    # Answer-leak structural check: no OOD prompt may contain its case answer.
    for case in valid:
        answer = str(case.get("answer") or "")
        if not answer:
            errors.append(prefix + f"case {case.get('subject')} records no answer")
            continue
        for item in case.get("paraphrases", []):
            if isinstance(item, dict) and answer.casefold() in str(item.get("prompt", "")).casefold():
                errors.append(prefix + f"paraphrase prompt leaks the answer for {case.get('subject')}")
                break

    if summary.get("paraphraseProfile") != "ood":
        errors.append(prefix + "summary does not record the OOD paraphrase profile")
    return errors


def main() -> None:
    global TOOLS, RUNNER
    ART.mkdir(parents=True, exist_ok=True)
    TOOLS = _resolve_tools_dir()
    RUNNER = TOOLS / "llm-reader-blackbox-behavior.py"
    print(f"[dataset] tools dir: {TOOLS}", flush=True)
    info = env_info()
    print(json.dumps(info, indent=1), flush=True)

    install = sh([sys.executable, "-m", "pip", "install", "-q", *PINNED])
    print(f"pip exit={install.returncode}", flush=True)
    if install.returncode != 0:
        raise SystemExit(1)

    # Pre-download both providers with a hard cache pre-flight gate.
    download_env = dict(os.environ)
    download_env["HF_HOME"] = HF_HOME
    for family in FAMILIES:
        probe = sh(
            [
                sys.executable,
                "-c",
                "from huggingface_hub import snapshot_download; "
                f"snapshot_download('{family['model']}'); print('downloaded')",
            ],
            env=download_env,
        )
        if probe.returncode != 0:
            raise SystemExit(f"model pre-download failed: {family['model']}")
        cache_dir = f"{HF_HOME}/hub/models--{family['model'].replace('/', '--')}"
        if not os.path.isdir(cache_dir):
            raise SystemExit(f"pre-downloaded model missing from HF_HOME: {family['model']}")

    kernel_ok = True
    run_infos: list[dict] = []
    for family in FAMILIES:
        rc, run_info = run_family(family)
        variant_dir = ART / family["variant"]
        report_path = variant_dir / "report.json"
        if not report_path.is_file():
            print(f"[{family['key']}] runner produced no report; kernel error", flush=True)
            kernel_ok = False
            run_info["consistencyErrors"] = ["runner produced no report.json"]
            run_infos.append(run_info)
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        errors = consistency_errors(report, family)
        run_info["consistencyErrors"] = errors
        run_info["summaryPass"] = bool(report.get("summary", {}).get("pass"))
        run_info["torch"] = info["torch"]
        run_infos.append(run_info)
        if errors:
            print(f"[{family['key']}] consistency errors: {errors}", flush=True)
            kernel_ok = False
        else:
            print(f"[{family['key']}] ARTIFACT_CONSISTENT (summary pass={run_info['summaryPass']})", flush=True)

    (ART / "run-summary.json").write_text(json.dumps({
        "kind": "gcl-kaggle-blackbox-ood-parity-run",
        "pinnedRepoCommit": PINNED_REPO_COMMIT,
        "dataset": PINNED_DATASET,
        "pinnedDeps": PINNED,
        **info,
        "families": run_infos,
    }, indent=2) + "\n", encoding="utf-8")
    print("GCL_KAGGLE_BLACKBOX_OOD_PARITY_SUMMARY " + json.dumps(run_infos), flush=True)

    if not kernel_ok:
        print("GCL_KAGGLE_BLACKBOX_OOD_PARITY_KERNEL_ERROR", flush=True)
        raise SystemExit(1)
    print("GCL_KAGGLE_BLACKBOX_OOD_PARITY_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
