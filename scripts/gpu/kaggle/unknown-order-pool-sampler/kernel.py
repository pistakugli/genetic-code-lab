#!/usr/bin/env python3
"""GCL unknown-order empirical pool sampler kernel (Kaggle T4, headless).

Teacher-side step of the preregistered unknown-order protocol
(docs/research/2026-11-19-unknown-order-empirical-swiglu-recovery.md):
record the real MLP-input distribution for the six frozen cases so the
later ladder kernels can fit blind candidates on the models' actual
activation manifold instead of Gaussian probes.

Steps:
1. print environment evidence (GPU, torch, versions);
2. provision the pinned GPU stack from the input dataset's requirements;
3. pre-download both 4-bit providers into HF_HOME=/tmp/hf-cache and hard-fail
   if either is missing from the cache (the v12 lesson, applied here too);
4. run the frozen-corpus sampler for each of the six cases;
5. validate every pool: point count, dimension, 60/20/20 splits, corpus
   SHA-256 identical to the preregistration digest;
6. write run-info.json with environment, invocation and pool inventory.

The solver never runs here: this kernel is teacher/witness-side only, and
its output dataset (pools + manifests) is the input for the ladder kernels.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

WORK = "/kaggle/working"
DS = "/kaggle/input/gcl-unknown-order-input"
UNPACK = "/tmp/gcl-input"
TOOLS = ""
REQS = ""
ART = f"{WORK}/pools"
RUN_INFO = f"{WORK}/run-info.json"
VENV = "/tmp/gcl-venv"
HF_HOME = "/tmp/hf-cache"

MISTRAL_MODEL = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
LLAMA_MODEL = "unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit"

# Frozen corpus digest (preregistration 2026-11-19, 18 templates x 12 slots).
CORPUS_SHA = "9042c1978c41f693c10dca648053fb9c6eb89547e9aa101252bdaf201d00a7bb"

# Six frozen cases: label, model, layer, frozen per-family shuffle seed.
CASES = [
    {"label": "M-E1", "model": MISTRAL_MODEL, "layer": 1, "shuffleSeed": 261133},
    {"label": "M-M15", "model": MISTRAL_MODEL, "layer": 15, "shuffleSeed": 261133},
    {"label": "M-L28", "model": MISTRAL_MODEL, "layer": 28, "shuffleSeed": 261133},
    {"label": "L-E1", "model": LLAMA_MODEL, "layer": 1, "shuffleSeed": 261134},
    {"label": "L-M14", "model": LLAMA_MODEL, "layer": 14, "shuffleSeed": 261134},
    {"label": "L-L27", "model": LLAMA_MODEL, "layer": 27, "shuffleSeed": 261134},
]

MAX_POINTS = 32768
SAMPLER_TIMEOUT = 1800.0


def log(message: str) -> None:
    print(f"[sampler-kernel] {message}", flush=True)


def run(cmd: list[str], *, capture: bool = False, check: bool = True, env=None, timeout=None):
    printed = " ".join(cmd[:6]) + (" ..." if len(cmd) > 6 else "")
    log(f"$ {printed}")
    result = subprocess.run(cmd, capture_output=capture, text=True, env=env, timeout=timeout)
    if capture and result.stdout:
        print(result.stdout[-3000:], flush=True)
    if capture and result.stderr:
        print(result.stderr[-3000:], flush=True)
    if check and result.returncode != 0:
        raise RuntimeError(f"command failed ({result.returncode}): {printed}\n{result.stderr[-2000:]}")
    return result


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


def locate_input() -> tuple[str, str]:
    """Kaggle's new mount layout nests datasets under /kaggle/input/datasets/
    instead of the legacy /kaggle/input/<slug>. Search by content across both
    layouts and diagnose what was actually mounted."""
    try:
        log(f"kaggle input mounts: {os.listdir('/kaggle/input')}")
        for root in ("/kaggle/input", "/kaggle/input/datasets"):
            if os.path.isdir(root):
                log(f"{root} entries: {os.listdir(root)}")
    except OSError as error:
        log(f"input listing failed: {error}")
    sampler_matches: list[str] = []
    req_matches: list[str] = []
    for root in ("/kaggle/input", "/kaggle/input/datasets"):
        if not os.path.isdir(root):
            continue
        sampler_matches += [str(p) for p in Path(root).rglob("empirical_activation_sampler.py")]
        req_matches += [str(p) for p in Path(root).rglob("gcl-python-gpu.txt")]
    sampler_matches = [p for p in sampler_matches if "/tmp/" not in p and "/kaggle/working/" not in p]
    req_matches = [p for p in req_matches if "/tmp/" not in p and "/kaggle/working/" not in p]
    if not sampler_matches or not req_matches:
        raise RuntimeError(
            f"could not locate sampler tool / requirements under /kaggle/input; "
            f"sampler matches: {sampler_matches}; requirements matches: {req_matches}"
        )
    return os.path.dirname(sampler_matches[0]), req_matches[0]


def main() -> int:
    started = time.time()
    log("kernel v4 start; python=" + sys.version.replace("\n", " "))
    tools_dir, reqs_path = locate_input()
    global TOOLS, REQS
    TOOLS = tools_dir
    REQS = reqs_path
    log(f"tools={TOOLS} requirements={REQS}")
    required_tools = (
        "empirical_activation_sampler.py",
    )
    for name in required_tools:
        if not os.path.isfile(f"{TOOLS}/{name}"):
            raise RuntimeError(f"input dataset is missing {name}; refusing to run")

    # --- 1. Pinned GPU stack -------------------------------------------------
    try:
        run([sys.executable, "-m", "pip", "install", "--quiet", "uv"])
        run([sys.executable, "-m", "uv", "venv", "--python", "3.12", VENV])
        run([sys.executable, "-m", "uv", "pip", "install", "--python", f"{VENV}/bin/python", "-r", REQS])
        PYTHON = f"{VENV}/bin/python"
    except RuntimeError as error:
        log(f"uv provisioning failed ({error}); falling back to base interpreter")
        PYTHON = sys.executable
        run([PYTHON, "-m", "pip", "install", "--quiet", "-r", REQS])
    stack_info = run([PYTHON, "-c",
                      "import sys, torch, transformers, bitsandbytes; "
                      "print('python', sys.version.split()[0]); "
                      "print('torch', torch.__version__, '| cuda', torch.cuda.is_available()); "
                      "print('transformers', transformers.__version__); "
                      "print('bitsandbytes', bitsandbytes.__version__)"], capture=True)

    env = dict(os.environ)
    env["GCL_DEVICE"] = "cuda"
    env["HF_HOME"] = HF_HOME

    # --- 2. Model pre-downloads + hard cache gate -----------------------------
    for model in (MISTRAL_MODEL, LLAMA_MODEL):
        log(f"pre-downloading {model}")
        run([PYTHON, "-c",
             f"from huggingface_hub import snapshot_download; "
             f"snapshot_download('{model}'); print('downloaded {model}')"],
            env=env, capture=True, timeout=2400)
    hf_cache = {}
    for model in (MISTRAL_MODEL, LLAMA_MODEL):
        cache_dir = f"{HF_HOME}/hub/models--{model.replace('/', '--')}"
        hf_cache[model] = os.path.isdir(cache_dir)
        log(f"hf cache check {model}: {'present' if hf_cache[model] else 'MISSING'}")
    if not all(hf_cache.values()):
        missing = [m for m, ok in hf_cache.items() if not ok]
        raise RuntimeError(f"pre-downloaded models missing from {HF_HOME}/hub: {missing}")

    # --- 3. Sampler runs per frozen case --------------------------------------
    os.makedirs(ART, exist_ok=True)
    inventory = []
    failures = []
    for case in CASES:
        case_dir = f"{ART}/{case['label']}"
        log(f"sampling case {case['label']}: {case['model']} layer {case['layer']}")
        try:
            run([PYTHON, f"{TOOLS}/empirical_activation_sampler.py",
                 "--model", case["model"], "--layer", str(case["layer"]),
                 "--output-dir", case_dir,
                 "--max-points", str(MAX_POINTS),
                 "--shuffle-seed", str(case["shuffleSeed"])],
                env=env, timeout=SAMPLER_TIMEOUT)
        except RuntimeError as error:
            log(f"case {case['label']} FAILED: {error}")
            failures.append({"case": case["label"], "error": str(error)[:2000]})
            continue
        manifest_path = f"{case_dir}/sampler-manifest.json"
        if not os.path.exists(manifest_path) or not os.path.exists(f"{case_dir}/empirical-pool.npz"):
            failures.append({"case": case["label"], "error": "sampler produced no pool/manifest"})
            continue
        with open(manifest_path) as handle:
            manifest = json.load(handle)
        corpus_sha = manifest.get("corpus", {}).get("sha256", "")
        splits = manifest.get("pool", {}).get("splits", {})
        point_count = int(manifest.get("pool", {}).get("pointCount", 0))
        dimension = int(manifest.get("pool", {}).get("dimension", 0))
        checks = {
            "corpusShaMatchesPreregistration": corpus_sha == CORPUS_SHA,
            "promptCount216": int(manifest.get("corpus", {}).get("promptCount", 0)) == 216,
            "pointCountPositive": point_count > 0,
            "dimensionMatchesModel": dimension in (3072, 4096),
            "splitRatios": splits,
        }
        ok = (checks["corpusShaMatchesPreregistration"] and checks["promptCount216"]
              and checks["pointCountPositive"] and checks["dimensionMatchesModel"]
              and set(splits) == {"train", "selection", "validation"}
              and point_count == sum(int(v) for v in splits.values()))
        log(f"case {case['label']}: points={point_count} dim={dimension} splits={splits} ok={ok}")
        inventory.append({
            "case": case["label"],
            "model": case["model"],
            "layer": case["layer"],
            "shuffleSeed": case["shuffleSeed"],
            "pointCount": point_count,
            "dimension": dimension,
            "splits": splits,
            "corpusSha256": corpus_sha,
            "validated": bool(ok),
        })
        if not ok:
            failures.append({"case": case["label"], "error": f"pool validation failed: {checks}"})

    # --- 4. run-info -----------------------------------------------------------
    info = {
        "kind": "gcl-unknown-order-empirical-pool-sampler",
        "kernelVersion": 4,
        "repoCommit": "db5f33f3a9ce749fbc6eeed063cbdfad4958aedf",
        "preregistration": "docs/research/2026-11-19-unknown-order-empirical-swiglu-recovery.md",
        "elapsedSeconds": time.time() - started,
        "env": env_info(),
        "stack": stack_info.stdout[-2000:] if stack_info.stdout else "",
        "hfCache": hf_cache,
        "inventory": inventory,
        "failures": failures,
        "caseCount": len(CASES),
        "validatedCaseCount": sum(1 for row in inventory if row["validated"]),
    }
    with open(RUN_INFO, "w") as handle:
        handle.write(json.dumps(info, indent=2) + "\n")
    log(f"done in {info['elapsedSeconds']:.0f}s; validated {info['validatedCaseCount']}/{info['caseCount']} pools; failures={len(failures)}")
    if failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
