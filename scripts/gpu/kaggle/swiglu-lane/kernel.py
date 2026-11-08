"""GCL blind SwiGLU oracle-decompiler lane kernel for Kaggle (headless).

Kernel version 13: witness-only rerun — fixes the v12 witness environment
regression.

What broke in v12: the kernel pre-downloaded the model providers into
HF_HOME=/tmp/hf-cache (and passed that env to the downloads and to the lane
F v2 solver — which is why lane F ran), but ran the quadratic-identity
witness tool WITHOUT the env. The witness spawns its oracle subprocess with
an inherited environment, so the oracle looked in the default
~/.cache/huggingface, found nothing, and died with LocalEntryNotFoundError
under --local-files-only — before a single oracle call. Both new witnesses
(Mistral, Qwen) failed this way in ~6s each while the lane F v2 case ran on.

Version 13:

  - propagates the kernel env (HF_HOME=/tmp/hf-cache, GCL_DEVICE=cuda)
    end-to-end: kernel -> witness tool -> oracle subprocess;
  - hard-fails early if a pre-downloaded model is not present under
    $HF_HOME/hub after the download step (the exact silent path that v12
    missed);
  - reruns ONLY the two quadratic-identity witnesses: Mistral-7B (4096D)
    and Qwen2.5-7B (3584D), layer 1, checks 256, scale 0.35, seed 260621,
    tolerance 0.02. The lane F v2 case is unaffected by the bug and keeps
    running in the v12 session; its evidence lands from that version. The
    LLaMA-family witness passed in kernel v11 and is landed in origin/main.

Runs on a free Kaggle GPU session (T4): clones origin/main, provisions the
pinned GPU stack, pre-downloads the two model providers, runs both
witnesses. Everything lands under /kaggle/working/artifacts/ and is
downloaded with `kaggle kernels output`.
"""

import json
import os
import shutil
import subprocess
import sys
import time
import traceback

WORK = "/kaggle/working"
REPO = "/tmp/gcl-repo"
VENV = "/tmp/gcl-venv"
ART = f"{WORK}/artifacts"
LOG_PATH = f"{ART}/kernel-progress.log"

LANE_ROOT = "benchmarks/release/nn-live-blind-swiglu-oracle-decompiler"
TOOL_PATH = "tools/llm_reader/blind_swiglu_oracle_decompiler.py"
WITNESS_PATH = "tools/llm_reader/swiglu_quadratic_witness.py"

MISTRAL_MODEL = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
QWEN_MODEL = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"

HF_HOME = "/tmp/hf-cache"
WITNESS_LAYER = 1
WITNESS_TIMEOUT_SECONDS = 5400.0

MISTRAL_WITNESS_VARIANT = "quadratic-witness-mistral-4096d-layer1-kaggle-t4-2026-11-08"
QWEN_WITNESS_VARIANT = "quadratic-witness-qwen-3584d-layer1-kaggle-t4-2026-11-08"

WITNESS_SPECS = [
    {
        "label": "mistral",
        "variant": MISTRAL_WITNESS_VARIANT,
        "model": MISTRAL_MODEL,
        "intermediate": 14336,
        "work_dir": "/tmp/gcl-witness-work-mistral",
    },
    {
        "label": "qwen",
        "variant": QWEN_WITNESS_VARIANT,
        "model": QWEN_MODEL,
        "intermediate": 18944,
        "work_dir": "/tmp/gcl-witness-work-qwen",
    },
]


def log(message):
    line = f"[{time.strftime('%H:%M:%S')}] {message}"
    print(line, flush=True)
    try:
        with open(LOG_PATH, "a") as handle:
            handle.write(line + "\n")
    except OSError:
        pass


def run(cmd, cwd=None, env=None, check=True, capture=False, timeout=None):
    log(f"RUN {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, env=env, check=False, text=True,
                            capture_output=capture, timeout=timeout)
    if capture:
        print(result.stdout[-4000:] if result.stdout else "", flush=True)
        if result.stderr:
            print("[stderr] " + result.stderr[-2000:], flush=True)
    if result.returncode != 0:
        log(f"FAILED ({result.returncode}): {' '.join(cmd)}")
        if capture and result.stderr:
            log("stderr tail: " + result.stderr[-6000:])
    if check and result.returncode != 0:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(cmd)}")
    return result


def write_manifest(output_dir, variant, status, summary, notes, argv, env, git_info):
    manifest = {
        "schemaVersion": 1,
        "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
        "lane": "nn-live-blind-swiglu-oracle-decompiler",
        "variant": variant,
        "status": status,
        "command": {"argv": argv, "cwd": REPO, "python": "3.12"},
        "env": env,
        "git": git_info,
        "outputs": {
            "reportJson": f"{LANE_ROOT}/{variant}/report.json",
            "reportMarkdown": f"{LANE_ROOT}/{variant}/report.md",
            "manifest": f"{LANE_ROOT}/{variant}/manifest.json",
        },
        "summary": summary,
        "notes": notes,
    }
    with open(f"{output_dir}/manifest.json", "w") as handle:
        handle.write(json.dumps(manifest, indent=2) + "\n")


def run_witness(spec, python, git_info, env):
    """One quadratic-identity witness run; returns (pass, oracle_calls, report).

    The kernel env (HF_HOME!) is passed to the witness process; the witness
    tool spawns its oracle subprocess with an inherited environment, so the
    env reaches the oracle end-to-end. v12 failed exactly here.
    """
    witness_dir = f"{ART}/{LANE_ROOT}/{spec['variant']}"
    os.makedirs(witness_dir, exist_ok=True)
    witness_started = time.time()
    timed_out = False
    result = None
    try:
        result = run([python, f"{REPO}/{WITNESS_PATH}",
                      "--model", spec["model"], "--layer", str(WITNESS_LAYER),
                      "--device", "cuda",
                      "--intermediate-size", str(spec["intermediate"]),
                      "--checks", "256", "--scale", "0.35", "--seed", "260621",
                      "--tolerance", "0.02", "--work-dir", spec["work_dir"],
                      "--local-files-only",
                      "--output-json", f"{witness_dir}/report.json",
                      "--output-md", f"{witness_dir}/report.md"],
                     cwd=REPO, check=False, env=env,
                     timeout=WITNESS_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        timed_out = True
        log(f"witness {spec['label']}: TIMEOUT after {WITNESS_TIMEOUT_SECONDS}s")
    witness_report = {}
    if os.path.exists(f"{witness_dir}/report.json"):
        try:
            with open(f"{witness_dir}/report.json") as handle:
                witness_report = json.load(handle)
        except json.JSONDecodeError:
            pass
    witness_pass = bool(not timed_out and result is not None
                        and result.returncode == 0
                        and witness_report.get("pass"))
    log(f"witness {spec['label']}: pass={witness_pass} "
        f"({time.time() - witness_started:.1f}s)")
    if os.path.exists(f"{spec['work_dir']}/witness-oracle-stderr.log"):
        try:
            shutil.copy(f"{spec['work_dir']}/witness-oracle-stderr.log",
                        f"{ART}/witness-oracle-stderr-{spec['label']}.log")
        except OSError:
            pass
    write_manifest(witness_dir, spec["variant"],
                   "pass" if witness_pass else "fail",
                   {"pass": witness_report.get("pass"),
                    "oracleCalls": witness_report.get("oracleCalls"),
                    "identities": list(witness_report.get("results", {}).keys())},
                   [f"SwiGLU quadratic-identity structural witness (oracle-only, "
                    f"full-mlp-vector mode), {spec['label']} family: certifies the "
                    f"target block's even part is exactly a vector-valued quadratic "
                    f"map — bias-free SiLU-gated structure — from f(x) queries alone. "
                    f"The LLaMA-family witness passed in kernel v11 and is landed in "
                    f"origin/main; this run extends the evidence class to "
                    f"{spec['label']}.",
                    "Kernel v13 rerun: v12's witness invocations did not propagate "
                    "HF_HOME to the oracle subprocess, so both witnesses died with "
                    "LocalEntryNotFoundError before any oracle call; v13 passes the "
                    "kernel env end-to-end (kernel -> witness -> oracle).",
                    "Tool: tools/llm_reader/swiglu_quadratic_witness.py."],
                   [python, f"{REPO}/{WITNESS_PATH}", "(see witness report)"],
                   {"GCL_WITNESS_SEED": "260621", "GCL_WITNESS_CHECKS": "256",
                    "GCL_WITNESS_TOLERANCE": "0.02", "HF_HOME": HF_HOME},
                   git_info)
    return witness_pass, witness_report.get("oracleCalls"), witness_report


def main():
    os.makedirs(ART, exist_ok=True)
    started = time.time()
    log("kernel v13 start; python=" + sys.version.replace("\n", " "))

    # --- 1. Clone origin/main (witness tool + oracle live in the repo) -------
    if os.path.exists(REPO):
        shutil.rmtree(REPO)
    run(["git", "clone", "--quiet", "https://github.com/pistakugli/genetic-code-lab.git", REPO])
    os.chdir(REPO)
    run(["git", "log", "--oneline", "-3"])
    for required in (TOOL_PATH, WITNESS_PATH,
                     "scripts/gpu/gpu-smoke.py",
                     "requirements/gcl-python-gpu.txt",
                     "README-kaggle-t4-2026-11-08.md"):
        if not os.path.exists(f"{REPO}/{required}"):
            raise RuntimeError(f"origin/main is missing {required}; refusing to run")
    git_info = {
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                                 text=True).stdout.strip(),
        "statusShort": subprocess.run(["git", "status", "--short"], cwd=REPO, capture_output=True,
                                      text=True).stdout.strip()[:400],
    }
    log(f"repo commit {git_info['commit']}")

    # --- 2. Provision the pinned GPU stack ----------------------------------
    try:
        run([sys.executable, "-m", "pip", "install", "--quiet", "uv"])
        run([sys.executable, "-m", "uv", "venv", "--python", "3.12", VENV])
        run([sys.executable, "-m", "uv", "pip", "install", "--python", f"{VENV}/bin/python",
             "-r", "requirements/gcl-python-gpu.txt"])
        PYTHON = f"{VENV}/bin/python"
    except RuntimeError as error:
        log(f"uv provisioning failed: {error}; falling back to base interpreter")
        PYTHON = sys.executable
        run([PYTHON, "-m", "pip", "install", "--quiet", "-r", "requirements/gcl-python-gpu.txt"])
    run([PYTHON, "-c",
         "import sys, torch, transformers, bitsandbytes; print('python', sys.version.split()[0]); "
         "print('torch', torch.__version__, '| cuda build', torch.version.cuda, "
         "'| cuda', torch.cuda.is_available(), "
         "'| gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None); "
         "print('transformers', transformers.__version__, '| bitsandbytes', bitsandbytes.__version__)"],
        capture=True)

    # The single env every downstream subprocess gets (THE v13 FIX: v12 gave
    # this to the downloads and the lane solver but not to the witnesses).
    env = dict(os.environ)
    env["GCL_DEVICE"] = "cuda"
    env["HF_HOME"] = HF_HOME

    # --- 3. Host validation + model pre-downloads ----------------------------
    run([PYTHON, "scripts/gpu/gpu-smoke.py", "--device", "cuda",
         "--output-json", f"{ART}/gpu-smoke.json"], env=env, capture=True)
    for model in (MISTRAL_MODEL, QWEN_MODEL):
        log(f"pre-downloading model {model}")
        run([PYTHON, "-c",
             f"from huggingface_hub import snapshot_download; "
             f"snapshot_download('{model}'); print('downloaded {model}')"],
            env=env, capture=True, timeout=1800)

    # Hard gate: the cache must actually contain both providers before any
    # --local-files-only subprocess is allowed to run (v12 missed this).
    hf_cache = {}
    for model in (MISTRAL_MODEL, QWEN_MODEL):
        cache_dir = f"{HF_HOME}/hub/models--{model.replace('/', '--')}"
        hf_cache[model] = os.path.isdir(cache_dir)
        log(f"hf cache check {model}: {'present' if hf_cache[model] else 'MISSING'}")
    if not all(hf_cache.values()):
        missing = [m for m, ok in hf_cache.items() if not ok]
        raise RuntimeError(f"pre-downloaded models missing from {HF_HOME}/hub: "
                           f"{missing}; refusing to run cache-blind witnesses")

    # --- 4. Quadratic-identity witnesses: Mistral + Qwen (LLaMA landed v11) --
    witness_results = {}
    for spec in WITNESS_SPECS:
        witness_pass, oracle_calls, _ = run_witness(spec, PYTHON, git_info, env)
        witness_results[spec["label"]] = {
            "pass": witness_pass,
            "oracleCalls": oracle_calls,
        }

    # --- 5. Run summary --------------------------------------------------------
    torch_version = subprocess.run([PYTHON, "-c", "import torch;print(torch.__version__)"],
                                   capture_output=True, text=True).stdout.strip()
    summary = {
        "kind": "gcl-kaggle-swiglu-lane-run",
        "kernelVersion": 13,
        "purpose": "witness-only rerun with the oracle subprocess env fixed "
                   "(v12 ran both new witnesses cache-blind)",
        "lanes": {
            "quadratic-witness-mistral": witness_results["mistral"],
            "quadratic-witness-qwen": witness_results["qwen"],
        },
        "allLanesPassed": bool(witness_results["mistral"]["pass"]
                               and witness_results["qwen"]["pass"]),
        "hfCache": hf_cache,
        "elapsedSeconds": round(time.time() - started, 1),
        "torch": torch_version,
        "repoCommit": git_info["commit"],
    }
    with open(f"{ART}/run-summary.json", "w") as handle:
        handle.write(json.dumps(summary, indent=2) + "\n")
    print("GCL_KAGGLE_SWIGLU_LANE_SUMMARY " + json.dumps(summary), flush=True)

    if not summary["allLanesPassed"]:
        raise RuntimeError("a quadratic-identity witness failed verification")
    print("GCL_KAGGLE_SWIGLU_LANE_COMPLETE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        with open(LOG_PATH, "a") as handle:
            handle.write("[FATAL] kernel traceback:\n" + traceback.format_exc() + "\n")
        print(traceback.format_exc(), flush=True)
        raise
