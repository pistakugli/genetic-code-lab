"""GCL blind SwiGLU oracle-decompiler lane kernel for Kaggle (headless).

Kernel version 12: lane F v2 (48 units, identifiability-matched) + the
quadratic-identity witness extended to Mistral and Qwen.

Version 11 (2026-11-08) landed the first real-model quadratic-identity
witness PASS on LLaMA-3.2-3B layer 1 (all identity residuals ~1e-6 relative,
6,145 f(x)-only oracle calls — landed in origin/main), but lane F v1 failed
informatively: the 128-unit configuration had 65,664 free parameters on
32,000 train rows (0.49 rows/parameter; 250 rows/atom, inside the 3-4m
spurious zone and far under the ~6m fiber bound recorded in
docs/research/2026-11-08-fullblock-joint-swiglu-recovery.md). The fit
interpolated the training set to MAE 4.06e-14 while validation R2 was
-39.61, CEGIS accepted 0 atoms, and the run was killed by its own 5 h case
timeout. Version 12:

  - lane F v2: 48 units (the first 48 distinct Park-Miller draws of the same
    seed-260621 stream — a verified subset of the v1 128-unit list, so v2
    recovers a sub-block of the very block v1 attempted), 48 atoms,
    activeRank 96, direction count 1152 (12/rank-dim, lane E parity),
    train 92,640 rows = exactly 10 rows/parameter (1,930 rows/atom = 1.9x
    lane E's per-atom density, 6.7x the 6m fiber bound), restarts 512 and
    structured starts on (both lane E parity), oracle budget 2,400,000,
    case timeout 8 h. The recorded lane E adaptive solver environment is
    inherited verbatim from the lane E manifest as in v11.
  - witnesses: the quadratic-identity structural witness on the Mistral-7B
    (4096D) and Qwen2.5-7B (3584D) layer-1 blocks — the LLaMA witness
    already passed in v11 and is landed; v12 extends the evidence class to
    the other two model families.

Runs on a free Kaggle GPU session (T4): clones origin/main, provisions the
pinned GPU stack, pre-downloads the three model providers, runs both
witnesses, then lane F v2. Everything lands under /kaggle/working/artifacts/
and is downloaded with `kaggle kernels output`.
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

MODEL = "unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit"
MISTRAL_MODEL = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
QWEN_MODEL = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"

LANE_F_UNITS = 48
LANE_F_V1_UNITS = 128  # v1 attempted 128 units; v2 units must be a subset
LANE_F_SEED = 260621
LANE_F_INTERMEDIATE = 8192
LANE_F_LAYER = 1
LANE_F_VARIANT = (
    "llama-3072d-layer1-lcg48-48atom-forward-coordinate-oracle-only"
    "-kaggle-t4-2026-11-08"
)
LANE_F_TRAIN_COUNT = 92640  # exactly 10 rows per free parameter (48*193*10)
LANE_F_DIRECTION_COUNT = 1152  # 12 per rank dimension, lane E parity
LANE_F_FIT_RESTARTS = 512  # lane E parity
LANE_F_FIT_RESTART_CHUNK_SIZE = 16

MISTRAL_WITNESS_VARIANT = "quadratic-witness-mistral-4096d-layer1-kaggle-t4-2026-11-08"
QWEN_WITNESS_VARIANT = "quadratic-witness-qwen-3584d-layer1-kaggle-t4-2026-11-08"

LANE_E_MANIFEST = (
    f"{LANE_ROOT}/llama-3072d-layer1-units27-57-413-500-1607-1970-2215-2529-2847"
    "-3530-3786-4033-4236-4906-7283-7720-16atom-forward-coordinate-oracle-only"
    "/manifest.json"
)

KERNEL_TIME_BUDGET_SECONDS = 8.75 * 3600.0
LANE_F_TIMEOUT_SECONDS = 8.0 * 3600.0

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


# ---------------------------------------------------------------------------
# Deterministic unit selection (anti-cherry-pick): Park-Miller LCG, seed
# 260621. Lane F v2 uses the first 48 distinct draws of the same stream v1
# drew 128 from, and the kernel asserts the subset property before solving.
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


def generate_units(count):
    random = make_random(LANE_F_SEED)
    unit_set = set()
    while len(unit_set) < count:
        unit_set.add(int(random() * LANE_F_INTERMEDIATE))
    return sorted(unit_set)


def build_case_args(units, output_json, output_md):
    """Lane F v2 argv: the recorded lane E configuration, density-matched.

    Only the block-shape and data knobs differ from lane E's recorded argv:
    units / atom-count / active-rank, train+validation counts, direction
    count, fit restarts, and the oracle budget — each chosen to keep the
    lane E identifiability margins (rows per parameter, rows per atom, rows
    per rank dimension) while scaling the block 3x. Every other
    solver-environment knob (gradient mode, probe geometry, polish,
    calibration, seeds, gates) is inherited unchanged.
    """
    return [
        "--model", MODEL,
        "--layer", str(LANE_F_LAYER),
        "--unit", str(units[0]),
        "--units", ",".join(str(u) for u in units),
        "--device", "cuda",
        "--oracle-quantization-step", "0",
        "--active-rank", str(2 * LANE_F_UNITS),
        "--atom-count", str(LANE_F_UNITS),
        "--gradient-mode", "forward-coordinate",
        "--basis-moment", "centered",
        "--probe-points", "33",
        "--direction-count", str(LANE_F_DIRECTION_COUNT),
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
        "--train-count", str(LANE_F_TRAIN_COUNT),
        "--validation-count", "8192",
        "--fit-steps", "2600",
        "--fit-learning-rate", "0.025",
        "--fit-restarts", str(LANE_F_FIT_RESTARTS),
        "--fit-restart-chunk-size", str(LANE_F_FIT_RESTART_CHUNK_SIZE),
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
        "--max-oracle-calls", "2400000",
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


def assert_case_report(report, units):
    """Gate assertions replicating the TS sweep assertCaseReport for lane F."""
    failures = []
    summary = report.get("summary", {})
    metrics = summary.get("metrics", {})
    alignment = summary.get("sourceAlignment", {})
    audit = report.get("accessAudit", {})
    if not summary.get("pass"):
        failures.append("summary.pass=false")
    if summary.get("dimension") != 3072:
        failures.append(f"dimension={summary.get('dimension')}")
    if summary.get("layer") != LANE_F_LAYER:
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


def run_witness(spec, python, git_info):
    """One quadratic-identity witness run; returns (pass, oracle_calls, report)."""
    witness_dir = f"{ART}/{LANE_ROOT}/{spec['variant']}"
    os.makedirs(witness_dir, exist_ok=True)
    witness_started = time.time()
    result = run([python, f"{REPO}/{WITNESS_PATH}",
                  "--model", spec["model"], "--layer", str(LANE_F_LAYER),
                  "--device", "cuda",
                  "--intermediate-size", str(spec["intermediate"]),
                  "--checks", "256", "--scale", "0.35", "--seed", "260621",
                  "--tolerance", "0.02", "--work-dir", spec["work_dir"],
                  "--local-files-only",
                  "--output-json", f"{witness_dir}/report.json",
                  "--output-md", f"{witness_dir}/report.md"],
                 cwd=REPO, check=False)
    witness_report = {}
    if os.path.exists(f"{witness_dir}/report.json"):
        try:
            with open(f"{witness_dir}/report.json") as handle:
                witness_report = json.load(handle)
        except json.JSONDecodeError:
            pass
    witness_pass = bool(result.returncode == 0 and witness_report.get("pass"))
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
                    "Tool: tools/llm_reader/swiglu_quadratic_witness.py."],
                   [python, f"{REPO}/{WITNESS_PATH}", "(see witness report)"],
                   {"GCL_WITNESS_SEED": "260621", "GCL_WITNESS_CHECKS": "256",
                    "GCL_WITNESS_TOLERANCE": "0.02"}, git_info)
    return witness_pass, witness_report.get("oracleCalls"), witness_report


def main():
    os.makedirs(ART, exist_ok=True)
    started = time.time()
    log("kernel v12 start; python=" + sys.version.replace("\n", " "))

    # --- 1. Clone origin/main (lane infra + witness tool live in the repo) ---
    if os.path.exists(REPO):
        shutil.rmtree(REPO)
    run(["git", "clone", "--quiet", "https://github.com/pistakugli/genetic-code-lab.git", REPO])
    os.chdir(REPO)
    run(["git", "log", "--oneline", "-3"])
    for required in (TOOL_PATH, WITNESS_PATH, LANE_E_MANIFEST,
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

    env = dict(os.environ)
    env["GCL_DEVICE"] = "cuda"
    env["HF_HOME"] = "/tmp/hf-cache"

    # --- 3. Host validation + model pre-downloads ----------------------------
    run([PYTHON, "scripts/gpu/gpu-smoke.py", "--device", "cuda",
         "--output-json", f"{ART}/gpu-smoke.json"], env=env, capture=True)
    for model in (MODEL, MISTRAL_MODEL, QWEN_MODEL):
        log(f"pre-downloading model {model}")
        run([PYTHON, "-c",
             f"from huggingface_hub import snapshot_download; "
             f"snapshot_download('{model}'); print('downloaded {model}')"],
            env=env, capture=True, timeout=1800)

    # --- 4. Quadratic-identity witnesses: Mistral + Qwen (LLaMA landed in v11)
    witness_results = {}
    for spec in WITNESS_SPECS:
        witness_pass, oracle_calls, _ = run_witness(spec, PYTHON, git_info)
        witness_results[spec["label"]] = {
            "pass": witness_pass,
            "oracleCalls": oracle_calls,
        }

    # --- 5. Lane F v2: 48-unit block recovery, density-matched ---------------
    units = generate_units(LANE_F_UNITS)
    units_v1 = generate_units(LANE_F_V1_UNITS)
    if not set(units) <= set(units_v1):
        raise RuntimeError("lane F v2 unit list is not a subset of the v1 "
                           "128-unit LCG stream; refusing to run")
    params = LANE_F_UNITS * (2 * (2 * LANE_F_UNITS) + 1)
    log(f"lane F v2: {LANE_F_UNITS} LCG units (seed {LANE_F_SEED}) "
        f"first={units[0]} last={units[-1]}; subset of v1 128-unit list: "
        f"{set(units) <= set(units_v1)}; {params} params, "
        f"{LANE_F_TRAIN_COUNT} train rows = "
        f"{LANE_F_TRAIN_COUNT / params:.1f} rows/param")
    output_dir = f"{REPO}/{LANE_ROOT}/{LANE_F_VARIANT}"
    os.makedirs(output_dir, exist_ok=True)
    for stale in ("report.json", "report.md", "manifest.json"):
        try:
            os.remove(f"{output_dir}/{stale}")
        except FileNotFoundError:
            pass
    output_json = f"{LANE_ROOT}/{LANE_F_VARIANT}/report.json"
    output_md = f"{LANE_ROOT}/{LANE_F_VARIANT}/report.md"
    argv = [PYTHON, f"{REPO}/{TOOL_PATH}", *build_case_args(units, output_json, output_md)]

    # Recorded lane E solver environment (adaptive machinery), verbatim except
    # the block-shape and data knobs scaled with constant identifiability
    # margins for the 48-unit block.
    with open(f"{REPO}/{LANE_E_MANIFEST}") as handle:
        lane_e_env = json.load(handle)["env"]
    lane_f_env = dict(env)
    for key, value in lane_e_env.items():
        if key == "GCL_PYTHON":
            continue
        lane_f_env[key] = value
    units_csv = ",".join(str(u) for u in units)
    for key, value in {
        "GCL_BENCHMARK_VARIANT": LANE_F_VARIANT,
        "GCL_BLIND_SWIGLU_ACTIVE_RANK": str(2 * LANE_F_UNITS),
        "GCL_BLIND_SWIGLU_ATOM_COUNT": str(LANE_F_UNITS),
        "GCL_BLIND_SWIGLU_UNITS": units_csv,
        "GCL_BLIND_SWIGLU_TRAIN_COUNT": str(LANE_F_TRAIN_COUNT),
        "GCL_BLIND_SWIGLU_VALIDATION_COUNT": "8192",
        "GCL_BLIND_SWIGLU_FIT_RESTARTS": str(LANE_F_FIT_RESTARTS),
        "GCL_BLIND_SWIGLU_FIT_RESTART_CHUNK_SIZE": str(LANE_F_FIT_RESTART_CHUNK_SIZE),
        "GCL_BLIND_SWIGLU_MAX_ORACLE_CALLS": "2400000",
        "GCL_BLIND_SWIGLU_DIRECTION_COUNT": str(LANE_F_DIRECTION_COUNT),
        "GCL_BLIND_SWIGLU_STRUCTURED_STARTS": "1",
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
        "OMP_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "GCL_BLIND_SWIGLU_SWEEP_VARIANT": LANE_F_VARIANT,
        "GCL_BLIND_SWIGLU_SWEEP_CASES": (
            f"1:{'+'.join(str(u) for u in units)}:0.995:0.015:0.999:0.12"),
        "GCL_BLIND_SWIGLU_TIMEOUT_MS": str(LANE_F_TIMEOUT_SECONDS * 1000),
    }.items():
        lane_f_env[key] = value
    manifest_env = {k: v for k, v in lane_f_env.items() if k not in env}

    log(f"lane F v2: solver start (timeout {LANE_F_TIMEOUT_SECONDS}s)")
    case_started = time.time()
    try:
        result = subprocess.run(argv, cwd=REPO, env=lane_f_env, text=True,
                                stdout=None, stderr=subprocess.PIPE,
                                timeout=LANE_F_TIMEOUT_SECONDS)
        stderr_tail = [line for line in (result.stderr or "").split("\n")
                       if line.strip() and not line.strip().startswith(("UserWarning", "warnings.warn"))][-6:]
    except subprocess.TimeoutExpired:
        result = None
        stderr_tail = [f"process killed by timeout after {LANE_F_TIMEOUT_SECONDS}s"]
        log(f"lane F v2: TIMEOUT after {LANE_F_TIMEOUT_SECONDS}s")
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
        failures += assert_case_report(report, units)
    if result is not None and result.returncode != 0:
        failures.append(f"process exit {result.returncode}")
    if stderr_tail:
        failures.append("stderr tail: " + " | ".join(stderr_tail))

    write_manifest(output_dir, LANE_F_VARIANT,
                   "pass" if not failures else "fail",
                   report.get("summary") if report else None,
                   ["Per-case report for the blind SwiGLU oracle-only lane F v2 "
                    "(Kaggle T4): LLaMA-3.2-3B layer 1, 48 LCG-drawn units (seed "
                    "260621, first 48 distinct draws — a verified subset of the "
                    "lane F v1 128-unit list), 48 atoms, activeRank 96.",
                    "Identifiability-matched scale-up of the recorded lane E "
                    "environment: 92,640 train rows = 10 rows per free parameter "
                    "(9,264 params), 1,930 rows per atom (1.9x lane E), 6.7x the "
                    "~6m fiber bound from the full-block research note; direction "
                    "count 1152 = 12 per rank dimension (lane E parity); restarts "
                    "512 and structured starts on (lane E parity). Lane F v1 "
                    "(128 units, 0.49 rows/param) failed by interpolation and is "
                    "documented in README-kaggle-t4-2026-11-08.md.",
                    "The solver process receives only f(x) batch outputs. Source "
                    "weights are requested only after recovery for post-hoc "
                    "witness alignment.",
                    f"kaggle-t4 case elapsed seconds: {elapsed}",
                    *([f"gate failures: {'; '.join(failures)}"] if failures else [])],
                   argv, manifest_env, git_info)
    lane_f_pass = not failures
    log(f"lane F v2: {'PASS' if lane_f_pass else 'FAIL ' + '; '.join(failures[:6])} ({elapsed}s)")
    summary_json = report.get("summary") if report else {}
    log(f"lane F v2 summary: r2={summary_json.get('metrics', {}).get('r2')} "
        f"rawOracleCalls={summary_json.get('rawOracleCalls')}")

    # --- 6. Collect artifacts --------------------------------------------------
    for variant in (LANE_F_VARIANT, MISTRAL_WITNESS_VARIANT, QWEN_WITNESS_VARIANT):
        src = f"{REPO}/{LANE_ROOT}/{variant}"
        if os.path.isdir(src):
            dst = f"{ART}/{LANE_ROOT}/{variant}"
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copytree(src, dst, dirs_exist_ok=True)
            log(f"collected lane artifacts: {variant}")

    # --- 7. Run summary --------------------------------------------------------
    torch_version = subprocess.run([PYTHON, "-c", "import torch;print(torch.__version__)"],
                                   capture_output=True, text=True).stdout.strip()
    summary = {
        "kind": "gcl-kaggle-swiglu-lane-run",
        "kernelVersion": 12,
        "lanes": {
            "llama-layer1-48unit": {
                "pass": lane_f_pass,
                "unitCount": LANE_F_UNITS,
                "atomCount": LANE_F_UNITS,
                "activeRank": 2 * LANE_F_UNITS,
                "trainCount": LANE_F_TRAIN_COUNT,
                "rowsPerParameter": round(LANE_F_TRAIN_COUNT / params, 2),
                "failures": failures[:12],
                "elapsedSeconds": elapsed,
                "minR2": summary_json.get("metrics", {}).get("r2"),
                "maxMae": summary_json.get("metrics", {}).get("mae"),
                "rawOracleCalls": summary_json.get("rawOracleCalls"),
            },
            "quadratic-witness-mistral": witness_results["mistral"],
            "quadratic-witness-qwen": witness_results["qwen"],
        },
        "allLanesPassed": bool(lane_f_pass
                               and witness_results["mistral"]["pass"]
                               and witness_results["qwen"]["pass"]),
        "elapsedSeconds": round(time.time() - started, 1),
        "torch": torch_version,
        "repoCommit": git_info["commit"],
    }
    with open(f"{ART}/run-summary.json", "w") as handle:
        handle.write(json.dumps(summary, indent=2) + "\n")
    print("GCL_KAGGLE_SWIGLU_LANE_SUMMARY " + json.dumps(summary), flush=True)

    if not summary["allLanesPassed"]:
        raise RuntimeError("lane F v2 or a quadratic-identity witness failed verification")
    print("GCL_KAGGLE_SWIGLU_LANE_COMPLETE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        with open(LOG_PATH, "a") as handle:
            handle.write("[FATAL] kernel traceback:\n" + traceback.format_exc() + "\n")
        print(traceback.format_exc(), flush=True)
        raise
