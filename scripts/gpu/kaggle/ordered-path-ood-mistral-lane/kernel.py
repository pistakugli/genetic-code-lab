"""GCL ordered-path OOD replacement-transfer lane kernel for Kaggle (Mistral-7B, headless).

Runs on a free Kaggle GPU session (T4): clones the public GitHub repo at
origin/main, applies the local patches carried in the attached dataset,
provisions the pinned GPU stack, validates the host with gpu-smoke.py, then
replays the recorded Mistral-7B France ordered residual path (attention L31
h3 + MLP L31 n9943, position 5) across the six held-out OOD paraphrase
families with layer-shift rescue, replacement-path search, cross-family
transfer, and the merged why-report layer attached to every recorded break.

The recorded artifact was produced on an RTX 3070 (sm86) with the same pinned
library stack (requirements/gcl-python-gpu.txt); this kernel replays it on a
Kaggle T4 (sm75). Four-bit bitsandbytes kernels differ across GPU
architectures, so bit-exact replay is not expected and the lane uses
--max-replay-error 0.1 (20% of the recorded combined drop 0.5); the actual
replay error is recorded in the artifact either way.

Output suite:
  benchmarks/release/nn-live-ordered-path-stability/ood-replacement-transfer-mistral-7b-2026-11-14

Everything lands under /kaggle/working/artifacts/ and is downloaded with
`kaggle kernels output`. The kernel exits non-zero only if the verified suite
is inconsistent (verification failure); an honest lane-gate failure (OOD
paraphrase robustness below threshold) is a recorded measurement, not an
error.
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

PRIMARY_SUITE = (
    "benchmarks/release/nn-live-ordered-path-stability/"
    "ood-replacement-transfer-mistral-7b-2026-11-14"
)

LANE_ARGS = [
    "--adapter", "mistral-swiglu",
    "--model", "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "--input-report",
    "../benchmarks/release/nn-live-mistral-swiglu-ordered-residual-path-circuit-finder/"
    "mistral-7b-instruct-v0.3-10-facts-ordered-residual-path/report.json",
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
    "--max-rescue-attempts", "6",
    "--replacement-candidate-count", "10",
    "--replacement-max-pair-attempts", "32",
    "--replacement-max-set-attempts", "24",
    "--min-layer-specificity-rate", "0.8",
    # Recorded on RTX 3070 (sm86), replayed here on T4 (sm75): 4-bit bnb
    # kernels are not bit-exact across architectures, so the tolerance is
    # 0.1 absolute logit units (20% of the recorded combined drop 0.5) instead
    # of the same-machine 1e-5. The artifact records the actual error.
    "--max-replay-error", "0.1",
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
    patches = sorted(glob.glob(f"{INPUT}/gcl-ordered-path-ood-mistral-lane-patches/*.patch"))
    if not patches:
        raise RuntimeError("no patches found in the attached dataset")
    for patch in patches:
        run(["git", "-c", "user.name=gcl-kernel", "-c", "user.email=gcl@kernel.local", "am", "--keep-cr", patch])
    run(["git", "log", "--oneline", "-5"])

    # Anti-drift guard: the recorded Mistral France case must carry the exact
    # recorded path this lane claims to replay.
    recorded = json.load(open(
        "benchmarks/release/nn-live-mistral-swiglu-ordered-residual-path-circuit-finder/"
        "mistral-7b-instruct-v0.3-10-facts-ordered-residual-path/report.json"))
    france = next(c for c in recorded["cases"] if c.get("subject") == "France")
    nodes = france["orderedResidualPath"]["nodes"]
    assert france["combined"]["targetLogitDrop"] == 0.5, "recorded France combined drop drifted"
    assert nodes["attention"] == [{"layer": 31, "head": 3, "ablateTokenPositions": [5]}], \
        "recorded France attention path drifted"
    assert nodes["mlp"] == [{"layer": 31, "neuron": 9943, "ablateTokenPositions": [5]}], \
        "recorded France MLP path drifted"
    assert france["readTokenIndex"] == 5, "recorded France read index drifted"
    print("[kernel] recorded Mistral France case matches the lane anchors", flush=True)

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

    # --- 3. Host validation ---------------------------------------------------
    run([PYTHON, "scripts/gpu/gpu-smoke.py",
         "--device", "cuda",
         "--output-json", f"{ART}/gpu-smoke.json"], env=env)

    # --- 4. Primary lane run: recorded Mistral France configuration ---------
    os.chdir(f"{REPO}/tools")
    run([PYTHON, "llm-reader-ordered-path-stability.py",
         *LANE_ARGS,
         "--subjects", "France", "--case-count", "1",
         "--output-json", f"../{PRIMARY_SUITE}/mistral/report.json",
         "--output-md", f"../{PRIMARY_SUITE}/mistral/report.md"], env=env)
    os.chdir(REPO)
    # An honest lane-gate failure (OOD robustness below threshold) is a
    # recorded measurement, not an execution error: the wrapper exits 1 when
    # the lane summary says pass=false, and the suite must still be assembled
    # and verified so the artifact and why-report ship.
    wrap = run([PYTHON, "scripts/gpu/wrap-lane-suite.py",
                "--paraphrase-profile", "ood",
                "--result", f"id=mistral,label=Mistral/SwiGLU,adapter=mistral-swiglu,"
                f"model=unsloth/mistral-7b-instruct-v0.3-bnb-4bit,report={PRIMARY_SUITE}/mistral/report.json",
                "--output-dir", PRIMARY_SUITE,
                "--note", "Kaggle T4 kernel run (headless); recorded Mistral-7B France "
                "OOD replacement-transfer configuration, why-report layer attached."],
               env=env, check=False)

    # --- 5. Independent verification ------------------------------------------
    os.chdir(f"{REPO}/tools")
    primary_verify = run([PYTHON, "-m", "llm_reader.verify_ordered_path_ood_stress",
                          "--report", f"../{PRIMARY_SUITE}/report.json"], env=env, check=False, capture=True)
    with open(f"{ART}/verifier-primary.txt", "w") as handle:
        handle.write(primary_verify.stdout + "\n[stderr]\n" + primary_verify.stderr)
    print("[kernel] primary verifier exit:", primary_verify.returncode, flush=True)

    # --- 6. Collect artifacts + run summary -----------------------------------
    torch_version = subprocess.run([PYTHON, "-c", "import torch;print(torch.__version__)"],
                                   capture_output=True, text=True).stdout.strip()
    lane_report = json.load(open(f"{REPO}/{PRIMARY_SUITE}/mistral/report.json"))
    summary = {
        "kind": "gcl-kaggle-ordered-path-ood-mistral-lane-run",
        "primarySuite": PRIMARY_SUITE,
        "suiteWrapperExit": wrap.returncode,
        "primaryVerifierExit": primary_verify.returncode,
        "laneGatePass": bool(lane_report.get("summary", {}).get("pass")),
        "paraphrasePassRate": lane_report.get("summary", {}).get("paraphrasePassRate"),
        "maxReplayError": lane_report.get("summary", {}).get("maxReplayError"),
        "elapsedSeconds": round(time.time() - started, 1),
        "torch": torch_version,
    }
    with open(f"{ART}/run-summary.json", "w") as handle:
        handle.write(json.dumps(summary, indent=2))
    print("GCL_KAGGLE_OOD_MISTRAL_LANE_SUMMARY " + json.dumps(summary), flush=True)

    if primary_verify.returncode != 0:
        raise RuntimeError("primary suite failed verification")
    print("GCL_KAGGLE_OOD_MISTRAL_LANE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
