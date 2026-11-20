#!/usr/bin/env python3
"""GCL unknown-order ladder kernel, case M-E1 (Kaggle T4, headless).

Preregistered protocol (docs/research/2026-11-19-unknown-order-empirical-swiglu-recovery.md),
case M-E1: Mistral-7B 4-bit, layer 1, true m=6 (teacher-side secret),
LCG unit draws seed 261119 (intermediate 14336), output channel seed 261120
(hidden 4096). The ladder runner never sees the true m: it fits frozen
ladder candidates against the unknown-order oracle (ready frame carries no
unitCount) on the recorded empirical pool, early-stops at the first
sufficient candidate, extends two extra steps, and writes the case report.

Steps:
1. environment evidence + pinned GPU stack;
2. pre-download the Mistral provider into HF_HOME (hard cache gate);
3. draw the frozen LCG units/channel and write the teacher manifest;
4. run the ladder runner with the frozen ladder/budget/threshold;
5. copy the case report into the kernel output and record run-info.

The true m and the unit list exist only in the teacher manifest (passed to
the oracle server by inherited descriptor) and in run-info metadata for the
post-freeze verifier; the solver command never mentions them.
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
DS = "/kaggle/input/gcl-unknown-order-input"
STAGE = "/tmp/gcl-work"
POOLS_DS = "/kaggle/input/gcl-unknown-order-pools"
UNPACK = "/tmp/gcl-input"
POOLS_UNPACK = "/tmp/gcl-pools"
TOOLS = ""
REQS = ""
POOL_PATH = ""
ART = f"{WORK}/m-e1"
VENV = "/tmp/gcl-venv"
HF_HOME = "/tmp/hf-cache"

MODEL = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
LAYER = 1
TRUE_M = 6  # teacher-side; never passed to the solver
INTERMEDIATE = 14336
HIDDEN = 4096
UNIT_LCG_SEED = 261119
CHANNEL_LCG_SEED = 261120
CASE_LABEL = "M-E1"

LADDER = "4,5,6,8,10,12,14,16,18,20,24"
R_THRESHOLD = 0.05
ORACLE_BUDGET = 1_500_000
MAX_FITS = 9
QUANTIZATION_STEP = 1e-6
SOLVER_TIMEOUT = 10_000


def log(message: str) -> None:
    print(f"[ladder-m-e1] {message}", flush=True)


def run(cmd: list[str], *, capture: bool = False, check: bool = True, env=None, timeout=None):
    printed = " ".join(cmd[:6]) + (" ..." if len(cmd) > 6 else "")
    log(f"$ {printed}")
    result = subprocess.run(cmd, capture_output=capture, text=True, env=env, timeout=timeout)
    if capture and result.stdout:
        print(result.stdout[-4000:], flush=True)
    if capture and result.stderr:
        print(result.stderr[-4000:], flush=True)
    if check and result.returncode != 0:
        raise RuntimeError(f"command failed ({result.returncode}): {printed}\n{result.stderr[-3000:]}")
    return result


def lcg_draws(seed: int, count: int, modulus: int) -> list[int]:
    """Frozen rule from the preregistration:
    state = seed; repeat state = (state * 1103515245 + 12345) mod 2**31;
    draw = state mod modulus; skip duplicates and zero.
    """
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


def locate_inputs() -> tuple[str, str, str]:
    """Kaggle mounts datasets inconsistently between the legacy
    /kaggle/input/<slug> and the new /kaggle/input/datasets/<owner>/<slug>
    layouts, and large pool files can appear late on the mount. Diagnose the
    full mount tree, wait briefly for the pool, and locate tools, requirements
    and this case's pool by content across both layouts."""
    try:
        log(f"kaggle input mounts: {os.listdir('/kaggle/input')}")
        for root in ("/kaggle/input", "/kaggle/input/datasets"):
            if os.path.isdir(root):
                log(f"{root} entries: {os.listdir(root)}")
        datasets_owner = "/kaggle/input/datasets/kuglipista"
        if os.path.isdir(datasets_owner):
            log(f"{datasets_owner} entries: {os.listdir(datasets_owner)}")
            for entry in os.listdir(datasets_owner):
                inner = f"{datasets_owner}/{entry}"
                if os.path.isdir(inner):
                    log(f"{inner} entries: {os.listdir(inner)[:20]}")
        legacy = "/kaggle/input/gcl-unknown-order-pools"
        if os.path.isdir(legacy):
            log(f"{legacy} entries: {os.listdir(legacy)}")
    except OSError as error:
        log(f"input listing failed: {error}")
    tools_dir = ""
    reqs_path = ""
    pool_path = ""
    deadline = time.time() + 300
    while True:
        tools_candidates: list[str] = []
        req_candidates: list[str] = []
        pool_candidates: list[str] = []
        for root in ("/kaggle/input", "/kaggle/input/datasets"):
            if not os.path.isdir(root):
                continue
            tools_candidates += [str(p) for p in Path(root).rglob("unknown_order_swiglu_recovery.py")]
            req_candidates += [str(p) for p in Path(root).rglob("gcl-python-gpu.txt")]
            pool_candidates += [str(p) for p in Path(root).rglob("*empirical-pool.npz")]
        tools_candidates = [p for p in tools_candidates if "/tmp/" not in p and "/kaggle/working/" not in p]
        req_candidates = [p for p in req_candidates if "/tmp/" not in p and "/kaggle/working/" not in p]
        pool_candidates = [p for p in pool_candidates if CASE_LABEL in p]
        if tools_candidates and req_candidates and pool_candidates:
            tools_dir = os.path.dirname(tools_candidates[0])
            reqs_path = req_candidates[0]
            pool_path = pool_candidates[0]
            break
        if time.time() >= deadline:
            raise RuntimeError(
                "could not locate inputs after 300s: "
                f"tools={tools_candidates[:3]} requirements={req_candidates[:3]} pools={pool_candidates[:3]}"
            )
        log("inputs incomplete; rechecking in 30s "
            f"(tools={len(tools_candidates)} reqs={len(req_candidates)} pools={len(pool_candidates)})")
        time.sleep(30)
    return tools_dir, reqs_path, pool_path


def env_info() -> dict:
    import torch

    return {
        "torch": torch.__version__,
        "cudaAvailable": bool(torch.cuda.is_available()),
        "gpuName": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "python": sys.version.split()[0],
    }


def stage_tools_work(tools_dir: str) -> str:
    """The input dataset carries the tools flat at its root; the ladder runner
    and its oracle server import the repo package layout (tools.llm_reader.x),
    so stage the flat files into a repo-like tree and run from there. The
    runner's own REPO_ROOT resolution (parents[2] of __file__) then points at
    the staged root and every subprocess import resolves."""
    target = f"{STAGE}/tools/llm_reader"
    os.makedirs(target, exist_ok=True)
    for entry in os.listdir(tools_dir):
        if entry.endswith(".py"):
            shutil.copy2(f"{tools_dir}/{entry}", f"{target}/{entry}")
    return f"{target}/unknown_order_swiglu_recovery.py"


def main() -> int:
    started = time.time()
    log("kernel v5 start")
    global TOOLS, REQS, POOL_PATH
    TOOLS, REQS, POOL_PATH = locate_inputs()
    log(f"tools={TOOLS} requirements={REQS} pool={POOL_PATH}")
    for required in (TOOLS, REQS, POOL_PATH):
        if not os.path.exists(required):
            raise RuntimeError(f"input missing: {required}; refusing to run")

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

    log(f"pre-downloading {MODEL}")
    run([PYTHON, "-c",
         f"from huggingface_hub import snapshot_download; snapshot_download('{MODEL}')"],
        env=env, capture=True, timeout=2400)
    cache_dir = f"{HF_HOME}/hub/models--{MODEL.replace('/', '--')}"
    if not os.path.isdir(cache_dir):
        raise RuntimeError(f"model missing from cache: {cache_dir}")

    # --- teacher manifest (LCG frozen draws; teacher-side only) ----------------
    units = lcg_draws(UNIT_LCG_SEED, TRUE_M, INTERMEDIATE)
    channel = lcg_draws(CHANNEL_LCG_SEED, 1, HIDDEN)[0]
    log(f"teacher draws ready: m={TRUE_M} units=[..{TRUE_M} values..] channel={channel}")
    os.makedirs(ART, exist_ok=True)
    manifest_path = f"{ART}/teacher-manifest.json"
    with open(manifest_path, "w") as handle:
        json.dump({
            "model": MODEL,
            "layer": LAYER,
            "units": units,
            "outputChannel": channel,
            "device": "cuda",
            "localFilesOnly": True,
            "quantizationStep": QUANTIZATION_STEP,
            "maxOracleCalls": ORACLE_BUDGET,
            "batchSize": 2048,
        }, handle, indent=2)

    # --- ladder run -------------------------------------------------------------
    runner_path = stage_tools_work(TOOLS)
    log(f"staged runner: {runner_path}")
    ladder_started = time.time()
    ladder_exit = 1
    timed_out = False
    try:
        result = run([PYTHON, runner_path,
                      "--output-dir", f"{ART}/ladder",
                      "--empirical-pool", POOL_PATH,
                      "--teacher-manifest", manifest_path,
                      "--ladder", LADDER,
                      "--r-threshold", str(R_THRESHOLD),
                      "--oracle-budget", str(ORACLE_BUDGET),
                      "--max-fits", str(MAX_FITS),
                      "--quantization-step", str(QUANTIZATION_STEP),
                      "--solver-timeout", str(SOLVER_TIMEOUT),
                      "--fit-train-count", "16000"],
                     env=env, check=False, timeout=39_000)
        ladder_exit = result.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
        log("ladder runner hit the kernel-level timeout; artifacts stay as recorded")

    report_path = f"{ART}/ladder/case-report.json"
    case_report = {}
    if os.path.exists(report_path):
        with open(report_path) as handle:
            case_report = json.load(handle)

    info = {
        "kind": "gcl-unknown-order-ladder-kernel",
        "case": "M-E1",
        "kernelVersion": 5,
        "repoCommit": "db5f33f3a9ce749fbc6eeed063cbdfad4958aedf",
        "preregistration": "docs/research/2026-11-19-unknown-order-empirical-swiglu-recovery.md",
        "elapsedSeconds": time.time() - started,
        "ladderSeconds": time.time() - ladder_started,
        "ladderExit": ladder_exit,
        "timedOut": timed_out,
        "env": env_info(),
        "teacherSecret": {"trueM": TRUE_M, "unitCount": len(units), "outputChannel": channel},
        "ladderConfig": {
            "ladder": LADDER,
            "rThreshold": R_THRESHOLD,
            "oracleBudgetPerFit": ORACLE_BUDGET,
            "maxFits": MAX_FITS,
        },
        "caseVerdict": case_report.get("verdict"),
        "inferredOrder": case_report.get("inferredOrder"),
        "upperRegionNonIdentifiable": case_report.get("upperRegionNonIdentifiable"),
    }
    with open(f"{ART}/run-info.json", "w") as handle:
        handle.write(json.dumps(info, indent=2) + "\n")
    log(f"done in {info['elapsedSeconds']:.0f}s; verdict={info['caseVerdict']} inferredOrder={info['inferredOrder']}")
    # exit code 1 from the runner is an honest scientific outcome (no sufficient
    # order); only missing artifacts are kernel errors
    if not os.path.exists(report_path) and not timed_out:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
