#!/usr/bin/env python3
"""GCL unknown-order ladder batch kernel (Kaggle T4, headless).

Runs the five freshly preregistered cases of the unknown-order empirical
SwiGLU recovery protocol (docs/research/2026-11-19-unknown-order-empirical-swiglu-recovery.md,
"Fresh preregistration" table) plus the M-E1 null-permutation control that
the recorded M-E1 kernel did not exercise:

  - M-E1-null: Mistral-7B 4-bit layer 1, true m=6 (the recorded case's
    teacher parameters), run with --null-control so the preregistered
    permutation null control is finally recorded;
  - M-M15 (true m=8), M-L28 (true m=12): Mistral-7B 4-bit layers 15/28;
  - L-E1 (true m=6), L-M14 (true m=8), L-L27 (true m=12):
    LLaMA-3.2-3B 4-bit layers 1/14/27.

All protocol knobs are the frozen ones from the recorded M-E1 run: ladder
4,5,6,8,10,12,14,16,18,20,24, r-threshold 0.05, oracle budget 1.5M per
fit, max 9 fits, quantization step 1e-6, fit-train-count 16000, solver
timeout 10000s. The teacher parameters for the five new cases are the
fresh preregistration's mechanical rule (early/middle/late layers get
m = 6/8/12, unit/channel LCG seed pairs advancing by two from M-E1's
261119/261120), chosen before any run against them.

Per case: draw the frozen LCG units/channel, write the teacher manifest,
run the ladder runner against the recorded empirical pool, record
run-info. A runner exit of 1 (no sufficient order) is an honest recorded
outcome; only missing artifacts fail the kernel.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

WORK = "/kaggle/working"
STAGE = "/tmp/gcl-work"
VENV = "/tmp/gcl-venv"
HF_HOME = "/tmp/hf-cache"
ART = WORK

MISTRAL_MODEL = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
LLAMA_MODEL = "unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit"

LADDER = "4,5,6,8,10,12,14,16,18,20,24"
R_THRESHOLD = 0.05
ORACLE_BUDGET = 1_500_000
MAX_FITS = 9
QUANTIZATION_STEP = 1e-06
SOLVER_TIMEOUT = 10_000
FIT_TRAIN_COUNT = 16000

CASES = [
    {
        "label": "M-E1-null",
        "model": MISTRAL_MODEL,
        "layer": 1,
        "true_m": 6,
        "unit_seed": 261119,
        "channel_seed": 261120,
        "intermediate": 14336,
        "hidden": 4096,
        "null_control": True,
    },
    {
        "label": "M-M15",
        "model": MISTRAL_MODEL,
        "layer": 15,
        "true_m": 8,
        "unit_seed": 261121,
        "channel_seed": 261122,
        "intermediate": 14336,
        "hidden": 4096,
        "null_control": False,
    },
    {
        "label": "M-L28",
        "model": MISTRAL_MODEL,
        "layer": 28,
        "true_m": 12,
        "unit_seed": 261123,
        "channel_seed": 261124,
        "intermediate": 14336,
        "hidden": 4096,
        "null_control": False,
    },
    {
        "label": "L-E1",
        "model": LLAMA_MODEL,
        "layer": 1,
        "true_m": 6,
        "unit_seed": 261125,
        "channel_seed": 261126,
        "intermediate": 8192,
        "hidden": 3072,
        "null_control": False,
    },
    {
        "label": "L-M14",
        "model": LLAMA_MODEL,
        "layer": 14,
        "true_m": 8,
        "unit_seed": 261127,
        "channel_seed": 261128,
        "intermediate": 8192,
        "hidden": 3072,
        "null_control": False,
    },
    {
        "label": "L-L27",
        "model": LLAMA_MODEL,
        "layer": 27,
        "true_m": 12,
        "unit_seed": 261129,
        "channel_seed": 261130,
        "intermediate": 8192,
        "hidden": 3072,
        "null_control": False,
    },
]

TOOLS = ""
REQS = ""
LOG_PATH = f"{WORK}/kernel-progress.log"


def log(message: str) -> None:
    line = f"[ladder-batch] {message}"
    print(line, flush=True)
    try:
        with open(LOG_PATH, "a") as handle:
            handle.write(line + "\n")
    except OSError:
        pass


def run(cmd, *, capture: bool = False, check: bool = True, env=None, timeout=None):
    printed = " ".join(str(part) for part in cmd[:6]) + (" ..." if len(cmd) > 6 else "")
    log(f"$ {printed}")
    result = subprocess.run(cmd, capture_output=capture, text=True, env=env, timeout=timeout)
    if capture and result.stdout:
        print(result.stdout[-3000:], flush=True)
    if capture and result.stderr:
        print("[stderr] " + result.stderr[-2000:], flush=True)
    if check and result.returncode != 0:
        raise RuntimeError(f"command failed ({result.returncode}): {printed}\n{result.stderr[-3000:]}")
    return result


def lcg_draws(seed: int, count: int, modulus: int) -> list[int]:
    state = int(seed)
    draws: list[int] = []
    seen: set[int] = set()
    while len(draws) < count:
        state = (state * 1103515245 + 12345) % (2**31)
        value = state % modulus
        if value == 0 or value in seen:
            continue
        seen.add(value)
        draws.append(value)
    return draws


def locate_tools_and_requirements() -> tuple[str, str]:
    deadline = time.time() + 300
    while True:
        tools_candidates: list[str] = []
        req_candidates: list[str] = []
        for root in ("/kaggle/input", "/kaggle/input/datasets"):
            if not os.path.isdir(root):
                continue
            tools_candidates += [str(p) for p in Path(root).rglob("unknown_order_swiglu_recovery.py")]
            req_candidates += [str(p) for p in Path(root).rglob("gcl-python-gpu.txt")]
        tools_candidates = [p for p in tools_candidates if "/tmp/" not in p and "/kaggle/working/" not in p]
        req_candidates = [p for p in req_candidates if "/tmp/" not in p and "/kaggle/working/" not in p]
        if tools_candidates and req_candidates:
            return os.path.dirname(tools_candidates[0]), req_candidates[0]
        if time.time() >= deadline:
            raise RuntimeError(
                f"could not locate tools/requirements after 300s: tools={tools_candidates[:3]} reqs={req_candidates[:3]}"
            )
        log(f"inputs incomplete; rechecking in 30s (tools={len(tools_candidates)} reqs={len(req_candidates)})")
        time.sleep(30)


def locate_pool(case_label: str) -> str:
    base_label = case_label.split("-null")[0]
    deadline = time.time() + 300
    while True:
        pool_candidates: list[str] = []
        for root in ("/kaggle/input", "/kaggle/input/datasets"):
            if not os.path.isdir(root):
                continue
            pool_candidates += [str(p) for p in Path(root).rglob("*empirical-pool.npz")]
        pool_candidates = [p for p in pool_candidates if base_label in p and "/tmp/" not in p]
        if pool_candidates:
            return pool_candidates[0]
        if time.time() >= deadline:
            raise RuntimeError(f"could not locate pool for {case_label} after 300s")
        log(f"pool for {case_label} incomplete; rechecking in 30s")
        time.sleep(30)


def env_info() -> dict:
    import torch

    return {
        "torch": torch.__version__,
        "cudaAvailable": bool(torch.cuda.is_available()),
        "gpuName": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "python": sys.version.split()[0],
    }


def stage_tools_work(tools_dir: str) -> str:
    target = f"{STAGE}/tools/llm_reader"
    if os.path.exists(f"{STAGE}/tools"):
        shutil.rmtree(f"{STAGE}/tools")
    os.makedirs(target, exist_ok=True)
    for entry in os.listdir(tools_dir):
        if entry.endswith(".py"):
            shutil.copy2(f"{tools_dir}/{entry}", f"{target}/{entry}")
    return f"{target}/unknown_order_swiglu_recovery.py"


def run_case(case: dict, runner_path: str, python: str, env: dict, git_commit: str) -> dict:
    label = case["label"]
    case_dir = f"{ART}/{label}"
    os.makedirs(case_dir, exist_ok=True)
    units = lcg_draws(case["unit_seed"], case["true_m"], case["intermediate"])
    channel = lcg_draws(case["channel_seed"], 1, case["hidden"])[0]
    log(f"case {label}: m={case['true_m']} layer={case['layer']} channel={channel} null={case['null_control']}")
    manifest_path = f"{case_dir}/teacher-manifest.json"
    with open(manifest_path, "w") as handle:
        json.dump(
            {
                "model": case["model"],
                "layer": case["layer"],
                "units": units,
                "outputChannel": channel,
                "device": "cuda",
                "localFilesOnly": True,
                "quantizationStep": QUANTIZATION_STEP,
                "maxOracleCalls": ORACLE_BUDGET,
                "batchSize": 2048,
            },
            handle,
            indent=2,
        )

    pool_path = locate_pool(label)
    log(f"case {label}: pool={pool_path}")
    runner_args = [
        python,
        runner_path,
        "--output-dir", f"{case_dir}/ladder",
        "--empirical-pool", pool_path,
        "--teacher-manifest", manifest_path,
        "--ladder", LADDER,
        "--r-threshold", str(R_THRESHOLD),
        "--oracle-budget", str(ORACLE_BUDGET),
        "--max-fits", str(MAX_FITS),
        "--quantization-step", str(QUANTIZATION_STEP),
        "--solver-timeout", str(SOLVER_TIMEOUT),
        "--fit-train-count", str(FIT_TRAIN_COUNT),
    ]
    if case["null_control"]:
        runner_args.append("--null-control")

    ladder_started = time.time()
    ladder_exit = 1
    timed_out = False
    try:
        result = run(runner_args, env=env, check=False, timeout=39_000)
        ladder_exit = result.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
        log(f"case {label}: ladder runner hit the kernel-level timeout; artifacts stay as recorded")

    report_path = f"{case_dir}/ladder/case-report.json"
    case_report: dict = {}
    if os.path.exists(report_path):
        with open(report_path) as handle:
            case_report = json.load(handle)
    info = {
        "kind": "gcl-unknown-order-ladder-kernel",
        "case": label,
        "kernelVersion": 1,
        "repoCommit": git_commit,
        "preregistration": "docs/research/2026-11-19-unknown-order-empirical-swiglu-recovery.md",
        "elapsedSeconds": time.time() - ladder_started,
        "ladderExit": ladder_exit,
        "timedOut": timed_out,
        "teacherSecret": {"trueM": case["true_m"], "unitCount": len(units), "outputChannel": channel},
        "ladderConfig": {
            "ladder": LADDER,
            "rThreshold": R_THRESHOLD,
            "oracleBudgetPerFit": ORACLE_BUDGET,
            "maxFits": MAX_FITS,
            "nullControl": case["null_control"],
        },
        "caseVerdict": case_report.get("verdict"),
        "inferredOrder": case_report.get("inferredOrder"),
        "upperRegionNonIdentifiable": case_report.get("upperRegionNonIdentifiable"),
        "nullControlVerdict": (
            (case_report.get("nullControl") or {}).get("verdict") if case_report.get("nullControl") else None
        ),
    }
    with open(f"{case_dir}/run-info.json", "w") as handle:
        handle.write(json.dumps(info, indent=2) + "\n")
    log(
        f"case {label}: done in {info['elapsedSeconds']:.0f}s; "
        f"verdict={info['caseVerdict']} inferredOrder={info['inferredOrder']}"
    )
    return info


def main() -> int:
    started = time.time()
    log("kernel v1 start (batch: five fresh-preregistration cases + M-E1 null control)")
    global TOOLS, REQS
    TOOLS, REQS = locate_tools_and_requirements()
    log(f"tools={TOOLS} requirements={REQS}")

    try:
        run([sys.executable, "-m", "pip", "install", "--quiet", "uv"])
        run([sys.executable, "-m", "uv", "venv", "--python", "3.12", VENV])
        run([sys.executable, "-m", "uv", "pip", "install", "--python", f"{VENV}/bin/python", "-r", REQS])
        PYTHON = f"{VENV}/bin/python"
    except RuntimeError as error:
        log(f"uv provisioning failed ({error}); fallback to base interpreter")
        PYTHON = sys.executable
        run([PYTHON, "-m", "pip", "install", "--quiet", "-r", REQS])

    env = dict(os.environ)
    env["GCL_DEVICE"] = "cuda"
    env["HF_HOME"] = HF_HOME

    for model in (MISTRAL_MODEL, LLAMA_MODEL):
        log(f"pre-downloading {model}")
        run(
            [PYTHON, "-c", f"from huggingface_hub import snapshot_download; snapshot_download('{model}')"],
            env=env,
            capture=True,
            timeout=2400,
        )
    for model in (MISTRAL_MODEL, LLAMA_MODEL):
        cache_dir = f"{HF_HOME}/hub/models--{model.replace('/', '--')}"
        if not os.path.isdir(cache_dir):
            raise RuntimeError(f"model missing from cache: {cache_dir}")

    runner_path = stage_tools_work(TOOLS)
    log(f"staged runner: {runner_path}")

    git_commit = "origin/main@1532d6f (pinned upstream; tools staged from kuglipista/gcl-unknown-order-input)"

    infos = []
    for case in CASES:
        try:
            info = run_case(case, runner_path, PYTHON, env, git_commit)
        except Exception as error:  # a failed case must not kill the batch
            import traceback

            log(f"case {case['label']} FAILED with exception: {error}")
            log(traceback.format_exc())
            info = {"case": case["label"], "kernelError": str(error)}
        infos.append(info)

    summary = {
        "kind": "gcl-unknown-order-ladder-batch-kernel",
        "kernelVersion": 1,
        "env": env_info(),
        "repoCommit": git_commit,
        "preregistration": "docs/research/2026-11-19-unknown-order-empirical-swiglu-recovery.md",
        "cases": [
            {
                "case": info.get("case"),
                "verdict": info.get("caseVerdict"),
                "inferredOrder": info.get("inferredOrder"),
                "trueM": info.get("teacherSecret", {}).get("trueM"),
                "nullControlVerdict": info.get("nullControlVerdict"),
                "timedOut": info.get("timedOut"),
            }
            for info in infos
        ],
        "elapsedSeconds": time.time() - started,
    }
    with open(f"{ART}/run-summary.json", "w") as handle:
        handle.write(json.dumps(summary, indent=2) + "\n")
    print("GCL_KAGGLE_UNKNOWN_ORDER_BATCH_SUMMARY " + json.dumps(summary), flush=True)

    missing = [
        info.get("case")
        for info in infos
        if not os.path.exists(f"{ART}/{info.get('case')}/ladder/case-report.json")
    ]
    if missing:
        log(f"cases without a recorded case report: {missing}")
        return 1
    print("GCL_KAGGLE_UNKNOWN_ORDER_BATCH_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        import traceback

        with open(LOG_PATH, "a") as handle:
            handle.write("[FATAL] kernel traceback:\n" + traceback.format_exc() + "\n")
        print(traceback.format_exc(), flush=True)
        raise
