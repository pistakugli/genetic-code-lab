"""GCL GPU lane kernel for Kaggle (headless, artifact-producing).

Runs on a free Kaggle GPU session (enable_gpu): clones the public GitHub
repo at origin/main, applies the two local patches carried in the attached
dataset, provisions the pinned GPU stack, validates the host with
gpu-smoke.py, runs the ordered-path OOD stress lane live on the GPU with the
recorded France configuration (plus a two-subject extended run), assembles
release-format suites, and verifies them with the recomputing verifier.

Everything lands under /kaggle/working/artifacts/ and is downloaded with
`kaggle kernels output`. The kernel exits non-zero if the primary verified
suite fails; the extended run is informational.
"""

import glob
import json
import os
import shutil
import subprocess
import sys
import time

WORK = "/kaggle/working"
REPO = f"{WORK}/repo"
VENV = f"{WORK}/venv"
ART = f"{WORK}/artifacts"
INPUT = "/kaggle/input"

PRIMARY_SUITE = "benchmarks/release/nn-live-ordered-path-stability/kaggle-t4-ood-2026-11-06"
EXTENDED_SUITE = "benchmarks/release/nn-live-ordered-path-stability/kaggle-t4-ood-2026-11-06-extended"

LANE_ARGS = [
    "--adapter", "gpt2",
    "--model", "gpt2",
    "--local-files-only",
    "--input-report", "../benchmarks/release/nn-live-gpt2-ordered-residual-path-circuit-finder/gpt2-50-facts-ordered-residual-path/report.json",
    "--paraphrase-count", "6",
    "--paraphrase-profile", "ood",
    "--layer-shifts=-2,-1,1,2",
    "--min-layer-control-count", "2",
    "--min-layer-margin", "0.001",
    "--min-original-drop", "0.05",
    "--min-paraphrase-drop-ratio", "0.10",
    "--min-paraphrase-pass-rate", "0.66",
    "--rescue-broken-paraphrases",
    "--rescue-layer-shifts=-3,-2,-1,1,2,3",
    "--max-rescue-attempts", "2",
    "--replacement-candidate-count", "10",
    "--replacement-max-pair-attempts", "32",
    "--replacement-max-set-attempts", "24",
    "--min-layer-specificity-rate", "0.8",
    "--max-replay-error", "1e-5",
    "--device", "cuda",
]


def run(cmd, cwd=None, env=None, check=True, capture=False):
    print(f"+ {' '.join(cmd)}", flush=True)
    result = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        check=False,
        text=True,
        capture_output=capture,
    )
    if capture:
        print(result.stdout[-4000:] if result.stdout else "", flush=True)
        if result.stderr:
            print("[stderr] " + result.stderr[-2000:], flush=True)
    if check and result.returncode != 0:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(cmd)}")
    return result


def main():
    os.makedirs(ART, exist_ok=True)
    started = time.time()

    # --- 1. Clone the repository and apply the local patches ---------------
    if os.path.exists(REPO):
        shutil.rmtree(REPO)
    run(["git", "clone", "--quiet", "https://github.com/pistakugli/genetic-code-lab.git", REPO])
    os.chdir(REPO)
    patches = sorted(glob.glob(f"{INPUT}/gcl-gpu-lane-patches/*.patch"))
    if not patches:
        raise RuntimeError("no patches found in the attached dataset")
    for patch in patches:
        run(["git", "-c", "user.name=gcl-kernel", "-c", "user.email=gcl@kernel.local", "am", "--keep-cr", patch])
    run(["git", "log", "--oneline", "-3"])

    # --- 2. Provision the pinned GPU stack ---------------------------------
    def provision_python():
        candidates = [sys.executable]
        if "/usr/bin/python3" != sys.executable:
            candidates.append("/usr/bin/python3")
        for base in candidates:
            try:
                run([base, "-m", "venv", VENV])
                return f"{VENV}/bin/python"
            except RuntimeError as error:
                print(f"[kernel] venv failed for {base}: {error}; trying next", flush=True)
        print("[kernel] venv unavailable; installing pinned stack into the running interpreter", flush=True)
        return sys.executable

    PYTHON = provision_python()
    run([PYTHON, "-m", "pip", "install", "--quiet", "-r", "requirements/gcl-python-gpu.txt"])
    run([PYTHON, "-c",
         "import torch; print('kernel torch:', torch.__version__, '| cuda:', torch.cuda.is_available(), '| gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"])

    env = dict(os.environ)
    env["GCL_DEVICE"] = "cuda"
    env["HF_HOME"] = f"{WORK}/hf-cache"

    # --- 3. Host validation (also warms the model cache for the lane) -------
    run([PYTHON, "scripts/gpu/gpu-smoke.py",
         "--device", "cuda",
         "--output-json", f"{ART}/gpu-smoke.json"], env=env)

    # --- 4. Primary lane run: recorded France configuration ------------------
    os.chdir(f"{REPO}/tools")
    run([PYTHON, "llm-reader-ordered-path-stability.py",
         *LANE_ARGS,
         "--subjects", "France", "--case-count", "1",
         "--output-json", f"../{PRIMARY_SUITE}/gpt2/report.json",
         "--output-md", f"../{PRIMARY_SUITE}/gpt2/report.md"], env=env)
    os.chdir(REPO)
    run([PYTHON, "scripts/gpu/wrap-lane-suite.py",
         "--paraphrase-profile", "ood",
         "--result", f"id=gpt2,label=GPT-2,adapter=gpt2,model=gpt2,report={PRIMARY_SUITE}/gpt2/report.json",
         "--output-dir", PRIMARY_SUITE,
         "--note", "Kaggle GPU kernel run (headless); recorded France OOD-stress configuration."], env=env)

    # --- 5. Extended lane run: two subjects, informational --------------------
    extended_ok = True
    try:
        os.chdir(f"{REPO}/tools")
        run([PYTHON, "llm-reader-ordered-path-stability.py",
             *LANE_ARGS,
             "--subjects", "France,Germany", "--case-count", "2",
             "--output-json", f"../{EXTENDED_SUITE}/gpt2/report.json",
             "--output-md", f"../{EXTENDED_SUITE}/gpt2/report.md"], env=env)
        os.chdir(REPO)
        run([PYTHON, "scripts/gpu/wrap-lane-suite.py",
             "--paraphrase-profile", "ood",
             "--result", f"id=gpt2,label=GPT-2,adapter=gpt2,model=gpt2,report={EXTENDED_SUITE}/gpt2/report.json",
             "--output-dir", EXTENDED_SUITE,
             "--note", "Kaggle GPU kernel extended run (France+Germany); informational."], env=env)
    except RuntimeError as error:
        print(f"[kernel] extended run failed (non-fatal): {error}", flush=True)
        extended_ok = False

    # --- 6. Independent verification ------------------------------------------
    os.chdir(f"{REPO}/tools")
    primary_verify = run([PYTHON, "-m", "llm_reader.verify_ordered_path_ood_stress",
                          "--report", f"../{PRIMARY_SUITE}/report.json"], env=env, check=False, capture=True)
    with open(f"{ART}/verifier-primary.txt", "w") as handle:
        handle.write(primary_verify.stdout + "\n[stderr]\n" + primary_verify.stderr)
    print("[kernel] primary verifier exit:", primary_verify.returncode, flush=True)

    extended_verify = run([PYTHON, "-m", "llm_reader.verify_ordered_path_ood_stress",
                           "--report", f"../{EXTENDED_SUITE}/report.json"], env=env, check=False, capture=True)
    with open(f"{ART}/verifier-extended.txt", "w") as handle:
        handle.write(extended_verify.stdout + "\n[stderr]\n" + extended_verify.stderr)
    print("[kernel] extended verifier exit:", extended_verify.returncode, flush=True)

    # --- 7. Collect artifacts + run summary -----------------------------------
    torch_version = subprocess.run([PYTHON, "-c", "import torch;print(torch.__version__)"],
                                   capture_output=True, text=True).stdout.strip()
    summary = {
        "kind": "gcl-kaggle-gpu-lane-run",
        "primarySuite": PRIMARY_SUITE,
        "extendedSuite": EXTENDED_SUITE,
        "primaryVerifierExit": primary_verify.returncode,
        "extendedVerifierExit": extended_verify.returncode,
        "extendedRunCompleted": extended_ok,
        "elapsedSeconds": round(time.time() - started, 1),
        "torch": torch_version,
    }
    with open(f"{ART}/run-summary.json", "w") as handle:
        handle.write(json.dumps(summary, indent=2))
    print("GCL_KAGGLE_GPU_LANE_SUMMARY " + json.dumps(summary), flush=True)

    if primary_verify.returncode != 0:
        raise RuntimeError("primary suite failed verification")
    print("GCL_KAGGLE_GPU_LANE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
