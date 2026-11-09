"""GCL blind SwiGLU oracle-decompiler lane G kernel for Kaggle (headless).

Kernel version 15: beat-the-baseline lanes — 16-unit blocks on the two
families v3 only swept at 8 units, plus a 24-unit LLaMA frontier bracket.

What this kernel is for: the recorded v3 T4 baselines are mistral 8-unit
(2 cases, best R2 0.9973), llama 8-unit (8 cases, R2 0.9999988), llama
layer-20 16-unit (1 case, R2 0.9999999974) and qwen 8-unit (8 cases,
R2 0.9999999957). Lane E later proved the recorded solver recovers a
16-unit LLaMA layer-1 block at R2 0.9999872 / MAE 0.00072 with the
forward-coordinate oracle-only protocol. Kernel v14 then mapped the upper
frontier: 48 units fails hard (R2 0.0104 at 10 rows/param, 0/48 paired
atoms, clean audit). So the recorded frontier is 16 pass / 48 fail.

Version 15 extends the *winning* side of that frontier to the other two
model families and brackets it:

  - lane G1: Mistral-7B (4096D) layer 1, the first 16 distinct Park-Miller
    LCG draws (seed 260621) over its 14,336 intermediate units — 2x the
    unit count of the v3 mistral baseline, at lane E's exact
    identifiability density (16,000 train rows, 15.4 rows/parameter,
    activeRank 32, 384 directions, 512 restarts, structured starts).
    Target: R2 >= 0.999 against v3 mistral's 0.9973.
  - lane G2: Qwen2.5-7B (3584D) layer 1, the first 16 distinct LCG draws
    over its 18,944 intermediate units — 2x the unit count of the v3 qwen
    baseline, same lane E protocol. Target: R2 >= 0.999.
  - lane G3 (frontier bracket, best-effort): LLaMA-3.2-3B layer 1, the
    first 24 distinct LCG draws over 8192 (a verified subset of the failed
    48-unit list, so G3 probes the exact block whose upper half v14 could
    not recover) at 10 rows/parameter, activeRank 48, 576 directions.
    Recorded pass or fail; does not gate the kernel claim.

Cases G1 and G2 gate the kernel exit; G3 is recorded either way.

Defensive machinery inherited from v13/v14: one kernel env (HF_HOME +
GCL_DEVICE) for every subprocess, and a hard pre-flight gate that refuses
to run any solver whose provider is not actually in the HF cache (the v12
witness regression). Never push over a running kernel — this runs under
its own fresh slug.

Runs on a free Kaggle GPU session (T4): clones origin/main, provisions the
pinned GPU stack, pre-downloads the three model providers, runs the three
cases. Everything lands under /kaggle/working/artifacts/ and is downloaded
with `kaggle kernels output`.
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

HF_HOME = "/tmp/hf-cache"
LCG_SEED = 260621

LLAMA_MODEL = "unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit"
MISTRAL_MODEL = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
QWEN_MODEL = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"

LANE_E_MANIFEST = (
    f"{LANE_ROOT}/llama-3072d-layer1-units27-57-413-500-1607-1970-2215-2529-2847"
    "-3530-3786-4033-4236-4906-7283-7720-16atom-forward-coordinate-oracle-only"
    "/manifest.json"
)

LANE_G_TIMEOUT_SECONDS = {
    "mistral-16": 2.5 * 3600.0,
    "qwen-16": 2.5 * 3600.0,
    "llama-24": 3.0 * 3600.0,
}

# spec per case: label, model, dimension, intermediate size, unit count,
# train count, validation count, direction count, variant tag, gating flag.
CASE_SPECS = [
    {
        "label": "mistral-16",
        "model": MISTRAL_MODEL,
        "dimension": 4096,
        "intermediate": 14336,
        "units": 16,
        "train": 16000,   # lane E parity: 15.4 rows per free parameter
        "validation": 8192,
        "directions": 384,  # 12 per rank dimension (activeRank 32), lane E parity
        "variant": "mistral-4096d-layer1-lcg16-16atom-forward-coordinate-oracle-only-kaggle-t4-2026-11-09",
        "gates": True,
    },
    {
        "label": "qwen-16",
        "model": QWEN_MODEL,
        "dimension": 3584,
        "intermediate": 18944,
        "units": 16,
        "train": 16000,
        "validation": 8192,
        "directions": 384,
        "variant": "qwen-3584d-layer1-lcg16-16atom-forward-coordinate-oracle-only-kaggle-t4-2026-11-09",
        "gates": True,
    },
    {
        "label": "llama-24",
        "model": LLAMA_MODEL,
        "dimension": 3072,
        "intermediate": 8192,
        "units": 24,
        "train": 23280,   # 10 rows per free parameter (24*97*10)
        "validation": 8192,
        "directions": 576,  # 12 per rank dimension (activeRank 48)
        "variant": "llama-3072d-layer1-lcg24-24atom-forward-coordinate-oracle-only-kaggle-t4-2026-11-09",
        "gates": False,   # frontier bracket: recorded either way
    },
]

FIT_RESTARTS = 512  # lane E parity
FIT_RESTART_CHUNK_SIZE = 16
MAX_ORACLE_CALLS = 2400000


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


# ---------------------------------------------------------------------------
# Deterministic unit selection (anti-cherry-pick): Park-Miller LCG, seed
# 260621 — the same stream every recorded lane uses. Each case draws its
# units over its own model's full intermediate size.
# ---------------------------------------------------------------------------

def make_random(seed):
    state = int(seed) % 2147483647
    if state <= 0:
        state += 2147483646

    def rnd():
        nonlocal state
        state = (state * 16807) % 2147483647
        return (state - 1) / 2147483646

    return rnd


def generate_units(count, intermediate):
    random = make_random(LCG_SEED)
    unit_set = set()
    while len(unit_set) < count:
        unit_set.add(int(random() * intermediate))
    return sorted(unit_set)


def build_case_args(spec, units, output_json, output_md):
    """Case argv: the recorded lane E configuration, verbatim except the
    model, units/atom-count/active-rank, train+validation counts, and
    direction count — each scaled to keep the lane E identifiability
    margins. Every other solver-environment knob is inherited unchanged."""
    return [
        "--model", spec["model"],
        "--layer", "1",
        "--unit", str(units[0]),
        "--units", ",".join(str(u) for u in units),
        "--device", "cuda",
        "--oracle-quantization-step", "0",
        "--active-rank", str(2 * spec["units"]),
        "--atom-count", str(spec["units"]),
        "--gradient-mode", "forward-coordinate",
        "--basis-moment", "centered",
        "--probe-points", "33",
        "--direction-count", str(spec["directions"]),
        "--subspace-count", "0",
        "--cubic-direction-count", "0",
        "--cubic-step", "0.35",
        "--cubic-steps", "",
        "--cubic-holdout-count", "0",
        "--finite-difference-step", "0.04",
        "--finite-difference-steps", "",
        "--input-scale", "0.35",
        "--train-scale-mixture", "",
        "--train-sampling-mode", "gaussian",
        "--validation-scale", "0.35",
        "--train-count", str(spec["train"]),
        "--validation-count", str(spec["validation"]),
        "--fit-steps", "2600",
        "--fit-learning-rate", "0.025",
        "--fit-restarts", str(FIT_RESTARTS),
        "--fit-restart-chunk-size", str(FIT_RESTART_CHUNK_SIZE),
        "--fit-mode", "batched",
        "--recovery-path", "nonlinear-fit",
        "--fit-gradient-count", "256",
        "--fit-gradient-mode", "forward",
        "--fit-gradient-step", "0.04",
        "--fit-gradient-steps", "",
        "--fit-gradient-weight", "1.0",
        "--fit-selection-count", "0",
        "--fit-selection-scale", "0",
        "--selection-tail-weight", "0",
        "--selection-tail-quantile", "0.995",
        "--selection-worst-weight", "0",
        "--seed", "260914",
        "--counterexample-count", "0",
        "--counterexample-repeat", "4",
        "--counterexample-jitter-count", "0",
        "--counterexample-jitter-scale", "0.025",
        "--tail-probe-count", "0",
        "--tail-probe-repeat", "4",
        "--tail-probe-scale", "0",
        "--rescue-train-target", "0",
        "--rescue-gradient-target", "0",
        "--rescue-steps", "0",
        "--rescue-restarts", "0",
        "--min-r2", "0.995",
        "--max-mae", "0.015",
        "--max-worst-error", "0.12",
        "--min-subspace-cosine", "0.999",
        "--min-paired-atom-cosine", "0.99",
        "--max-oracle-calls", str(MAX_ORACLE_CALLS),
        "--polish-top-k", "1",
        "--backfit-cycles", "0",
        "--tail-polish-steps", "1100",
        "--tail-polish-weight", "2.2",
        "--tail-polish-fraction", "0.30",
        "--tail-polish-lr", "0.003",
        "--tail-polish-loss", "topk",
        "--robust-scalar-calibration",
        "--scalar-calibration-steps", "900",
        "--scalar-calibration-lr", "0.01",
        "--scalar-calibration-tail-weight", "2.0",
        "--scalar-calibration-tail-fraction", "0.25",
        "--disable-adaptive-rescue",
        "--progress",
        "--local-files-only",
        "--output-json", output_json,
        "--output-md", output_md,
    ]


def assert_case_report(report, spec, units):
    """Gate assertions replicating the TS sweep assertCaseReport."""
    failures = []
    summary = report.get("summary", {})
    metrics = summary.get("metrics", {})
    alignment = summary.get("sourceAlignment", {})
    audit = report.get("accessAudit", {})
    if not summary.get("pass"):
        failures.append("summary.pass=false")
    if summary.get("dimension") != spec["dimension"]:
        failures.append(f"dimension={summary.get('dimension')}")
    if summary.get("layer") != 1:
        failures.append(f"layer={summary.get('layer')}")
    if summary.get("unit") != units[0]:
        failures.append(f"unit={summary.get('unit')}")
    report_units = summary.get("units")
    if report_units != units:
        failures.append(f"units mismatch (len {len(report_units or [])} vs {len(units)})")
    if (summary.get("unitCount") or 0) != len(units):
        failures.append(f"unitCount={summary.get('unitCount')}")
    if (summary.get("atomCount") or 0) != len(units):
        failures.append(f"atomCount={summary.get('atomCount')}")
    if summary.get("activeRank") != 2 * len(units):
        failures.append(f"activeRank={summary.get('activeRank')}")
    if metrics.get("r2", float("-inf")) < 0.995:
        failures.append(f"r2={metrics.get('r2')}<0.995")
    if metrics.get("mae", float("inf")) > 0.015:
        failures.append(f"mae={metrics.get('mae')}>0.015")
    if metrics.get("worstError", float("inf")) > 0.12:
        failures.append(f"worstError={metrics.get('worstError')}>0.12")
    if alignment.get("subspaceMinPrincipalCosine", float("-inf")) < 0.999:
        failures.append(f"subspace={alignment.get('subspaceMinPrincipalCosine')}<0.999")
    paired_min = alignment.get("pairedAtomMinJointCosine")
    paired_count = alignment.get("pairedAtomCountAbove099")
    if paired_min is None:
        failures.append("pairedAtomMinJointCosine=missing")
    elif paired_min < 0.99:
        failures.append(f"pairedAtomMinJointCosine={paired_min}<0.99")
    if paired_count is None:
        failures.append("pairedAtomCount=missing")
    elif paired_count < len(units):
        failures.append(f"pairedAtomCount={paired_count}<{len(units)}")
    if audit.get("solverAccess") != "oracle-only":
        failures.append(f"solverAccess={audit.get('solverAccess')}")
    if audit.get("weightsUsedBySolver") is not False:
        failures.append("weightsUsedBySolver!=false")
    if audit.get("hooksUsedBySolver") is not False:
        failures.append("hooksUsedBySolver!=false")
    if audit.get("sourceIdsUsedBySolver") is not False:
        failures.append("sourceIdsUsedBySolver!=false")
    if audit.get("postHocWitnessOnly") is not True:
        failures.append("postHocWitnessOnly!=true")
    return failures


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


def run_case(spec, python, git_info, base_env, lane_e_env):
    """One lane G case; returns a result dict for the run summary."""
    units = generate_units(spec["units"], spec["intermediate"])
    # llama-24 bracket must be a subset of the failed 48-unit v14 list.
    if spec["label"] == "llama-24":
        units_48 = generate_units(48, spec["intermediate"])
        if not set(units) <= set(units_48):
            raise RuntimeError("llama-24 unit list is not a subset of the v14 "
                               "48-unit LCG stream; refusing to run")
    params = spec["units"] * (2 * (2 * spec["units"]) + 1)
    log(f"lane G [{spec['label']}]: {spec['units']} LCG units (seed {LCG_SEED}) "
        f"over {spec['intermediate']} first={units[0]} last={units[-1]}; "
        f"{params} params, {spec['train']} train rows = "
        f"{spec['train'] / params:.1f} rows/param")

    output_dir = f"{REPO}/{LANE_ROOT}/{spec['variant']}"
    os.makedirs(output_dir, exist_ok=True)
    for stale in ("report.json", "report.md", "manifest.json"):
        try:
            os.remove(f"{output_dir}/{stale}")
        except FileNotFoundError:
            pass
    output_json = f"{LANE_ROOT}/{spec['variant']}/report.json"
    output_md = f"{LANE_ROOT}/{spec['variant']}/report.md"
    argv = [python, f"{REPO}/{TOOL_PATH}",
            *build_case_args(spec, units, output_json, output_md)]

    case_env = dict(base_env)
    for key, value in lane_e_env.items():
        if key == "GCL_PYTHON":
            continue
        case_env[key] = value
    units_csv = ",".join(str(u) for u in units)
    for key, value in {
        "GCL_BENCHMARK_VARIANT": spec["variant"],
        "GCL_BLIND_SWIGLU_ACTIVE_RANK": str(2 * spec["units"]),
        "GCL_BLIND_SWIGLU_ATOM_COUNT": str(spec["units"]),
        "GCL_BLIND_SWIGLU_UNITS": units_csv,
        "GCL_BLIND_SWIGLU_TRAIN_COUNT": str(spec["train"]),
        "GCL_BLIND_SWIGLU_VALIDATION_COUNT": str(spec["validation"]),
        "GCL_BLIND_SWIGLU_FIT_RESTARTS": str(FIT_RESTARTS),
        "GCL_BLIND_SWIGLU_FIT_RESTART_CHUNK_SIZE": str(FIT_RESTART_CHUNK_SIZE),
        "GCL_BLIND_SWIGLU_MAX_ORACLE_CALLS": str(MAX_ORACLE_CALLS),
        "GCL_BLIND_SWIGLU_DIRECTION_COUNT": str(spec["directions"]),
        "GCL_BLIND_SWIGLU_STRUCTURED_STARTS": "1",
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
        "OMP_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "GCL_BLIND_SWIGLU_SWEEP_VARIANT": spec["variant"],
        "GCL_BLIND_SWIGLU_SWEEP_CASES": (
            f"1:{'+'.join(str(u) for u in units)}:0.995:0.015:0.999:0.12"),
        "GCL_BLIND_SWIGLU_TIMEOUT_MS": str(LANE_G_TIMEOUT_SECONDS[spec["label"]] * 1000),
    }.items():
        case_env[key] = value
    manifest_env = {k: v for k, v in case_env.items() if k not in base_env}

    timeout = LANE_G_TIMEOUT_SECONDS[spec["label"]]
    log(f"lane G [{spec['label']}]: solver start (timeout {timeout}s)")
    case_started = time.time()
    result = None
    try:
        result = subprocess.run(argv, cwd=REPO, env=case_env, text=True,
                                stdout=None, stderr=subprocess.PIPE,
                                timeout=timeout)
        stderr_tail = [line for line in (result.stderr or "").split("\n")
                       if line.strip() and not line.strip().startswith(("UserWarning", "warnings.warn"))][-6:]
    except subprocess.TimeoutExpired:
        stderr_tail = [f"process killed by timeout after {timeout}s"]
        log(f"lane G [{spec['label']}]: TIMEOUT after {timeout}s")
    elapsed = round(time.time() - case_started, 1)

    failures = []
    report = None
    report_json_path = f"{output_dir}/report.json"
    if os.path.exists(report_json_path):
        try:
            with open(report_json_path) as handle:
                report = json.load(handle)
        except json.JSONDecodeError as error:
            failures.append(f"report-json-parse-error: {error}")
    else:
        failures.append("missing report.json")
    if report is not None:
        failures += assert_case_report(report, spec, units)
    if result is not None and result.returncode != 0:
        failures.append(f"process exit {result.returncode}")
    if stderr_tail:
        failures.append("stderr tail: " + " | ".join(stderr_tail))

    write_manifest(output_dir, spec["variant"],
                   "pass" if not failures else "fail",
                   report.get("summary") if report else None,
                   ["Per-case report for the blind SwiGLU oracle-only lane G "
                    f"({spec['label']}, Kaggle T4): {spec['model']} layer 1, "
                    f"{spec['units']} LCG-drawn units (seed {LCG_SEED} over "
                    f"{spec['intermediate']}), {spec['units']} atoms, "
                    f"activeRank {2 * spec['units']}.",
                    ("Beat-the-baseline lane: the recorded v3 T4 baseline for "
                     "this family swept 8-unit blocks; lane G doubles the unit "
                     "count at the recorded lane E identifiability density "
                     f"({spec['train']} train rows = "
                     f"{spec['train'] / params:.1f} rows per free parameter, "
                     f"direction count {spec['directions']} = 12 per rank "
                     "dimension, restarts 512, structured starts on).")
                    if spec["gates"] else
                    ("Frontier bracket: the first 24 distinct LCG draws — a "
                     "verified subset of the v14 48-unit list that failed at "
                     "10 rows/parameter — probing whether the joint recovery "
                     "frontier sits above 16 units; recorded pass or fail."),
                    "The solver process receives only f(x) batch outputs. Source "
                    "weights are requested only after recovery for post-hoc "
                    "witness alignment.",
                    f"kaggle-t4 case elapsed seconds: {elapsed}",
                    *([f"gate failures: {'; '.join(failures)}"] if failures else [])],
                   argv, manifest_env, git_info)
    case_pass = not failures
    log(f"lane G [{spec['label']}]: {'PASS' if case_pass else 'FAIL ' + '; '.join(failures[:6])} ({elapsed}s)")
    summary_json = report.get("summary") if report else {}
    metrics = summary_json.get("metrics", {}) if summary_json else {}
    return {
        "label": spec["label"],
        "variant": spec["variant"],
        "model": spec["model"],
        "dimension": spec["dimension"],
        "pass": case_pass,
        "unitCount": spec["units"],
        "atomCount": spec["units"],
        "activeRank": 2 * spec["units"],
        "trainCount": spec["train"],
        "rowsPerParameter": round(spec["train"] / params, 2),
        "failures": failures[:12],
        "elapsedSeconds": elapsed,
        "r2": metrics.get("r2"),
        "mae": metrics.get("mae"),
        "worstError": metrics.get("worstError"),
        "rawOracleCalls": summary_json.get("rawOracleCalls"),
        "gating": spec["gates"],
    }


def main():
    os.makedirs(ART, exist_ok=True)
    started = time.time()
    log("kernel v15 start; python=" + sys.version.replace("\n", " "))

    # --- 1. Clone origin/main (lane infra lives in the repo) ------------------
    if os.path.exists(REPO):
        shutil.rmtree(REPO)
    run(["git", "clone", "--quiet", "https://github.com/pistakugli/genetic-code-lab.git", REPO])
    os.chdir(REPO)
    run(["git", "log", "--oneline", "-3"])
    for required in (TOOL_PATH, LANE_E_MANIFEST,
                     "scripts/gpu/gpu-smoke.py",
                     "requirements/gcl-python-gpu.txt",
                     "README-kaggle-t4-2026-11-08.md",
                     "docs/research/2026-11-08-fullblock-joint-swiglu-recovery.md"):
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

    # One env for every downstream subprocess (v13/v14 lesson).
    env = dict(os.environ)
    env["GCL_DEVICE"] = "cuda"
    env["HF_HOME"] = HF_HOME

    # --- 3. Host validation + model pre-downloads ------------------------------
    run([PYTHON, "scripts/gpu/gpu-smoke.py", "--device", "cuda",
         "--output-json", f"{ART}/gpu-smoke.json"], env=env, capture=True)
    for model in (MISTRAL_MODEL, QWEN_MODEL, LLAMA_MODEL):
        log(f"pre-downloading model {model}")
        run([PYTHON, "-c",
             f"from huggingface_hub import snapshot_download; "
             f"snapshot_download('{model}'); print('downloaded {model}')"],
            env=env, capture=True, timeout=1800)

    # Hard gate: every provider must actually be in the HF cache.
    hf_cache = {}
    for model in (MISTRAL_MODEL, QWEN_MODEL, LLAMA_MODEL):
        cache_dir = f"{HF_HOME}/hub/models--{model.replace('/', '--')}"
        hf_cache[model] = os.path.isdir(cache_dir)
        log(f"hf cache check {model}: {'present' if hf_cache[model] else 'MISSING'}")
    if not all(hf_cache.values()):
        missing = [m for m, ok in hf_cache.items() if not ok]
        raise RuntimeError(f"pre-downloaded models missing from {HF_HOME}/hub: "
                           f"{missing}; refusing to run cache-blind solvers")

    # Recorded lane E solver environment, inherited verbatim.
    with open(f"{REPO}/{LANE_E_MANIFEST}") as handle:
        lane_e_env = json.load(handle)["env"]

    # --- 4. Lane G cases -------------------------------------------------------
    results = []
    for spec in CASE_SPECS:
        try:
            results.append(run_case(spec, PYTHON, git_info, env, lane_e_env))
        except Exception as error:
            log(f"lane G [{spec['label']}]: crashed: {error}")
            results.append({
                "label": spec["label"], "variant": spec["variant"],
                "model": spec["model"], "dimension": spec["dimension"],
                "pass": False, "unitCount": spec["units"],
                "atomCount": spec["units"], "activeRank": 2 * spec["units"],
                "trainCount": spec["train"],
                "rowsPerParameter": round(
                    spec["train"] / (spec["units"] * (2 * (2 * spec["units"]) + 1)), 2),
                "failures": [f"case crashed: {error}"], "elapsedSeconds": 0.0,
                "r2": None, "mae": None, "worstError": None,
                "rawOracleCalls": None, "gating": spec["gates"],
            })

    # --- 5. Collect artifacts --------------------------------------------------
    for spec in CASE_SPECS:
        src = f"{REPO}/{LANE_ROOT}/{spec['variant']}"
        if os.path.isdir(src):
            dst = f"{ART}/{LANE_ROOT}/{spec['variant']}"
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copytree(src, dst, dirs_exist_ok=True)
            log(f"collected lane artifacts: {spec['variant']}")

    # --- 6. Run summary --------------------------------------------------------
    torch_version = subprocess.run([PYTHON, "-c", "import torch;print(torch.__version__)"],
                                   capture_output=True, text=True).stdout.strip()
    gated = [r for r in results if r["gating"]]
    bracket = [r for r in results if not r["gating"]]
    summary = {
        "kind": "gcl-kaggle-swiglu-lane-run",
        "kernelVersion": 15,
        "purpose": "beat-the-baseline lanes: 16-unit blocks on Mistral and Qwen "
                   "(v3 swept those families at 8 units) plus a 24-unit LLaMA "
                   "frontier bracket (v14 proved 48 units fails at 10 rows/param)",
        "lanes": {r["label"]: r for r in results},
        "gatedLanesPassed": all(r["pass"] for r in gated),
        "allLanesPassed": all(r["pass"] for r in results),
        "hfCache": hf_cache,
        "elapsedSeconds": round(time.time() - started, 1),
        "torch": torch_version,
        "repoCommit": git_info["commit"],
    }
    with open(f"{ART}/run-summary.json", "w") as handle:
        handle.write(json.dumps(summary, indent=2) + "\n")
    print("GCL_KAGGLE_SWIGLU_LANE_SUMMARY " + json.dumps(summary), flush=True)

    if not summary["gatedLanesPassed"]:
        failed = [r["label"] for r in gated if not r["pass"]]
        raise RuntimeError(f"gating lane G cases failed: {failed}")
    if bracket and not all(r["pass"] for r in bracket):
        log(f"frontier bracket recorded as fail: "
            f"{[r['label'] for r in bracket if not r['pass']]}")
    print("GCL_KAGGLE_SWIGLU_LANE_COMPLETE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        with open(LOG_PATH, "a") as handle:
            handle.write("[FATAL] kernel traceback:\n" + traceback.format_exc() + "\n")
        print(traceback.format_exc(), flush=True)
        raise
