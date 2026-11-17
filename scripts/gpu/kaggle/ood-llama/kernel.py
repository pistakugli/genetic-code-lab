#!/usr/bin/env python3
"""GCL ordered-path OOD replacement transfer for LLaMA-3.2 (SwiGLU), third model family.

Runs tools/llm-reader-ordered-path-stability.py against the recorded
LLaMA-3.2-3B 4-bit circuit-finder report with the OOD paraphrase profile and
rescue-broken-paraphrases enabled, on a Kaggle T4 GPU.

Steps:
1. print environment evidence (GPU, torch, versions);
2. install the pinned runtime used by the recorded circuit-finder report;
3. probe replay tolerance on one subject (cross-hardware replay check);
4. run the full 4-subject OOD replacement-transfer replay with rescue;
5. run the pinned verifier's case-report check on the produced report
   (on-kernel smoke verification);
6. write run-info.json with environment and invocation details.

v2: the dataset is refreshed to repo commit b59234a, so the runner emits the
current why-report layer (paraphraseBreaks classification of every broken
paraphrase and replacement attempted flags). v1 recorded the full lane
correctly but its dataset carried an older why_report.py, which the pinned
verifier rejected at integration time. This version also runs the pinned
verifier's verify_case_report on the produced case report on-kernel: an
artifact that does not verify is a kernel error, while an honest scientific
fail (runner exit 1) is a recorded finding. The lane suite (report.json,
report.md, manifest.json) is assembled after the run by the repo wrapper
scripts/gpu/wrap-lane-suite.py at integration time, and re-verified there.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

DS = Path("/kaggle/input/gcl-llama-ood-input")
TOOLS = DS / "tools"
INPUT_REPORT = DS / "input" / "report.json"
WORK = Path("/kaggle/working")
OUT_DIR = WORK / "llama"
MODEL_ID = "unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit"

PINNED = ["transformers==5.5.0", "accelerate==1.13.0", "bitsandbytes==0.49.2"]
PINNED_REPO_COMMIT = "b59234a"


def sh(cmd: list[str]) -> subprocess.CompletedProcess:
    print(f"+ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd)


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


def run_stability(
    subjects: str,
    case_count: int,
    paraphrase_count: int,
    max_replay_error: float,
    tag: str,
    ratio: float = 0.05,
) -> tuple[int, Path, Path]:
    out_json = WORK / f"{tag}-report.json"
    out_md = WORK / f"{tag}-report.md"
    cmd = [
        sys.executable,
        str(TOOLS / "llm-reader-ordered-path-stability.py"),
        "--adapter", "llama-swiglu",
        "--model", MODEL_ID,
        "--input-report", str(INPUT_REPORT),
        "--subjects", subjects,
        "--case-count", str(case_count),
        "--paraphrase-count", str(paraphrase_count),
        "--paraphrase-profile", "ood",
        "--min-original-drop", "0.005",
        "--min-paraphrase-drop-ratio", str(ratio),
        "--min-paraphrase-pass-rate", "0.66",
        "--layer-shifts=-2,-1,1,2",
        "--min-layer-control-count", "2",
        "--min-layer-margin", "0.0001",
        "--min-layer-specificity-rate", "0.75",
        "--rescue-broken-paraphrases",
        "--max-rescue-attempts", "2",
        "--rescue-layer-shifts=-3,-2,-1,1,2,3",
        "--replacement-candidate-count", "10",
        "--replacement-max-pair-attempts", "32",
        "--replacement-max-set-attempts", "24",
        "--max-replay-error", str(max_replay_error),
        "--device", "cuda",
        "--output-json", str(out_json),
        "--output-md", str(out_md),
    ]
    started = time.time()
    result = sh(cmd)
    elapsed = round(time.time() - started)
    print(f"[{tag}] exit={result.returncode} elapsed={elapsed}s", flush=True)
    return result.returncode, out_json, out_md


def replay_tolerance_ok(report_path: Path, tolerance: float) -> bool | None:
    """Read a probe report and decide whether cross-hardware replay is exact enough.

    Returns True/False when the report exists, None when it does not. A broken
    OOD paraphrase is expected in this lane and is NOT a tolerance signal; only
    the original-replay pass rate and the recorded max replay error decide.
    """
    if not report_path.is_file():
        return None
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    summary = report.get("summary", {}) or {}
    original_rate = summary.get("originalReplayPassRate")
    max_error = summary.get("maxReplayError")
    if original_rate != 1.0:
        return False
    if isinstance(max_error, (int, float)) and not isinstance(max_error, bool):
        return float(max_error) <= tolerance
    return False


def verify_case_report() -> tuple[int, Path]:
    """Run the pinned verifier's case-report check on-kernel.

    Imports the verifier module straight from the dataset tools and calls
    verify_case_report on the produced case report: no suite assembly is
    needed on-kernel (the suite wrapper assumes a repo-relative layout), and
    this is the exact check the repo verifier will re-run at integration.
    """
    verifier_log = WORK / "llama-case-verifier.txt"
    if not (OUT_DIR / "report.json").is_file():
        verifier_log.write_text("case report missing\n", encoding="utf-8")
        return 1, verifier_log
    sys.path.insert(0, str(TOOLS))
    from llm_reader.verify_ordered_path_ood_stress import verify_case_report as check  # noqa: E402

    case = json.loads((OUT_DIR / "report.json").read_text(encoding="utf-8"))
    errors = check(case, MODEL_ID, "llama-swiglu")
    if errors:
        text = "\n".join(f"ERROR: {error}" for error in errors) + "\n"
    else:
        text = "ORDERED_PATH_OOD_STRESS_CASE_REPORT_VERIFIED\n"
    verifier_log.write_text(text, encoding="utf-8")
    print(text.strip(), flush=True)
    return (1 if errors else 0), verifier_log


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    info = env_info()
    print(json.dumps(info, indent=1), flush=True)

    install = sh([sys.executable, "-m", "pip", "install", "-q", *PINNED])
    print(f"pip exit={install.returncode}", flush=True)

    # Probe: one subject, one OOD paraphrase, strict recorded replay tolerance.
    # Records best-case cross-hardware exactness evidence (v1 measured France
    # and Italy exact, Germany/Spain off by exactly one 4-bit logit quantum).
    probe_rc, probe_json, _ = run_stability("France", 1, 1, 1e-5, "probe")
    strict_ok = replay_tolerance_ok(probe_json, 1e-5)

    # Full lane at the measured cross-hardware bounds:
    # - replay tolerance 0.125 = exactly one 4-bit logit quantum (all observed
    #   llama drops are multiples of 1/8; borderline subjects can flip one
    #   quantum across GPU architectures);
    # - paraphrase ratio 0.5 = strict-persistence OOD gate (the path must
    #   retain >=50% of its source-prompt causal effect; measured attenuation
    #   breaks get rescued and diagnosed by the why-report).
    tolerance = 0.125
    full_rc, full_json, full_md = run_stability(
        "France,Germany,Italy,Spain", 4, 6, tolerance, "full", ratio=0.5
    )
    if full_json.is_file():
        OUT_DIR.joinpath("report.json").write_text(full_json.read_text(encoding="utf-8"))
    if full_md.is_file():
        OUT_DIR.joinpath("report.md").write_text(full_md.read_text(encoding="utf-8"))

    verify_rc, verifier_log = verify_case_report()

    run_info = {
        **info,
        "pinnedDeps": PINNED,
        "pinnedRepoCommit": PINNED_REPO_COMMIT,
        "probeStrictTolerance": 1e-5,
        "probeStrictReplayOk": strict_ok,
        "replayToleranceUsed": tolerance,
        "replayToleranceRationale": "one 4-bit logit quantum; measured cross-GPU bound",
        "paraphraseDropRatioUsed": 0.5,
        "paraphraseDropRatioRationale": "strict-persistence OOD gate; attenuation breaks are rescued and diagnosed",
        "fullExit": full_rc,
        "caseVerifierExit": verify_rc,
        "modelId": MODEL_ID,
        "inputReport": str(INPUT_REPORT),
        "dataset": "kuglipista/gcl-llama-ood-input",
    }
    (OUT_DIR / "run-info.json").write_text(json.dumps(run_info, indent=2) + "\n", encoding="utf-8")
    print("RUN_INFO " + json.dumps(run_info), flush=True)

    if full_json.is_file() and verify_rc == 0:
        print("GCL_LLAMA_OOD_REPLACEMENT_TRANSFER_ARTIFACT_VERIFIED", flush=True)
        if full_rc == 0:
            print("GCL_LLAMA_OOD_REPLACEMENT_TRANSFER_PASS", flush=True)
        else:
            # Honest scientific fail: the lane report and why-report are recorded
            # and verified; this is a finding, not a kernel crash.
            print("GCL_LLAMA_OOD_REPLACEMENT_TRANSFER_FAIL_RECORDED", flush=True)
        return

    print("GCL_LLAMA_OOD_REPLACEMENT_TRANSFER_KERNEL_ERROR", flush=True)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
