#!/usr/bin/env python3
"""GCL LLaMA-3.2 50-fact ordered-residual-path circuit discovery (Kaggle T4).

Roadmap item 6 first gate: scale the fact set. The recorded
nn-live-llama-swiglu-ordered-residual-path-circuit-finder lane stopped at 20
valid facts (repo commit 5628833, run on the developer's local GPU); this
kernel scales the identical discovery protocol, layer plan, thresholds, and
claim boundary to the GPT-2 lane's 50-valid-fact pool through the full FACTS
corpus (90 entries; the tokenizer's single-token answer validity decides the
recorded pool - a pre-flight over tokenizer.json measured 62 valid facts, so
50 is reachable without weakening any gate).

Steps:
1. print environment evidence (GPU, torch, versions);
2. install the pinned runtime used by the recorded family artifacts;
3. resolve the dataset mount by content (dir-mode zip upload mounts a raw
   tools.zip; unpack any zip under /kaggle/input and search the whole input
   tree for the runner, printing the full mount listing for provenance);
4. pre-download the 4-bit provider into HF_HOME with a hard pre-flight gate
   (never run a cache-blind lane; v12 lesson);
5. run tools/llm-reader-gpt2-ordered-residual-path-sweep.py with the recorded
   20-fact configuration (mlp/attention layers 20,24,27, fallback
   16,18,20,22,24,26,27, top-neurons 8, top-logits 8, control-count 3,
   control-seed 260610, max-graph-nodes 3, max-graph-heads 2, thresholds
   0.90/0.90/0.01/0.001/0.001/0.0001, paraphrase-count 1, negative-fact-count
   1, unrelated-logit-count 4, wrong-answer-count 2,
   attention-max-target-rank 200, no component hardening, target-causal MLP
   fallback allowed) with only case-count/min-valid-cases scaled 20 -> 50;
6. on-kernel consistency recompute: summary aggregates, percentile stats,
   pass verdict, model identity, config echo, and claim-boundary fields are
   recomputed from the raw case fields with the runner's own percentile
   algorithm (a malformed artifact is a kernel error; an honest scientific
   fail is a recorded finding);
7. write run-info.json and archive the artifact directory under
   /kaggle/working/artifacts.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

DS_HINTS = ("gcl-llama-ordered-input",)
TEMP = Path("/kaggle/temp")
WORK = Path("/kaggle/working")
ART = WORK / "artifacts"
HF_HOME = "/kaggle/temp/hf-cache"

MODEL_ID = "unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit"
ADAPTER_ID = "llama-swiglu"
VARIANT = "llama-3.2-3b-50-facts-ordered-residual-path"
RUNNER_NAME = "llm-reader-gpt2-ordered-residual-path-sweep.py"

PINNED = ["transformers==5.5.0", "accelerate==1.13.0", "bitsandbytes==0.49.2"]
PINNED_REPO_COMMIT = "81cb491"
PINNED_DATASET = "kuglipista/gcl-llama-ordered-input"

# The recorded 20-fact lane configuration, with only case-count and
# min-valid-cases scaled 20 -> 50 (roadmap item 6: scale the fact set).
LANE_ARGV = [
    "--adapter", ADAPTER_ID,
    "--model", MODEL_ID,
    "--case-count", "50",
    "--min-valid-cases", "50",
    "--mlp-layers", "20,24,27",
    "--mlp-fallback-layers", "16,18,20,22,24,26,27",
    "--attention-layers", "20,24,27",
    "--attention-fallback-layers", "16,18,20,22,24,26,27",
    "--top-neurons", "8",
    "--top-logits", "8",
    "--control-count", "3",
    "--control-prompt", "The weather in Berlin is",
    "--control-seed", "260610",
    "--max-graph-nodes", "3",
    "--max-graph-heads", "2",
    "--min-pass-rate", "0.90",
    "--min-component-pass-rate", "0.90",
    "--min-target-drop", "0.01",
    "--min-margin-over-control-p95", "0.001",
    "--min-path-activation-delta", "0.001",
    "--min-path-margin-over-control-p95", "0.0001",
    "--wrong-answer-count", "2",
    "--attention-max-target-rank", "200",
    "--paraphrase-count", "1",
    "--negative-fact-count", "1",
    "--unrelated-logit-count", "4",
    "--no-require-component-hardening",
    "--allow-target-causal-mlp-fallback",
    "--device", "cuda",
    "--local-files-only",
]


def sh(cmd: list[str], env: dict | None = None, cwd: str | None = None) -> subprocess.CompletedProcess:
    print(f"+ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, env=env, cwd=cwd)


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


def _dataset_listing() -> list[str]:
    rows: list[str] = []
    for root, _dirs, files in os.walk("/kaggle/input"):
        for name in files:
            rows.append(str(Path(root) / name))
    return sorted(rows)


def _resolve_tools_dir() -> Path:
    """Locate the ordered-path sweep runner + llm_reader package in the mounts."""
    print("[dataset] /kaggle/input listing:", flush=True)
    for row in _dataset_listing():
        print(f"[dataset] {row}", flush=True)

    candidates: list[Path] = []
    for row in _dataset_listing():
        if row.endswith(RUNNER_NAME):
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
        if (candidate / RUNNER_NAME).is_file() and (
            candidate / "llm_reader" / "adapters" / "registry.py"
        ).is_file():
            return candidate
    raise SystemExit(
        "dataset pre-flight failed: the ordered residual path sweep runner and "
        "the llm_reader adapter package are missing from /kaggle/input"
    )


def _percentile(values: list[float], q: float) -> float:
    """Byte-identical reimplementation of factual.percentile (linear interpolation)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    position = min(len(ordered) - 1, max(0.0, (len(ordered) - 1) * q))
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    fraction = position - lower
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction)


def run_discovery(tools: Path) -> tuple[int, Path, Path, int]:
    variant_dir = ART / VARIANT
    variant_dir.mkdir(parents=True, exist_ok=True)
    out_json = variant_dir / "report.json"
    out_md = variant_dir / "report.md"

    cmd = [
        sys.executable,
        str(tools / RUNNER_NAME),
        *LANE_ARGV,
        "--output-json", str(out_json),
        "--output-md", str(out_md),
    ]
    env = dict(os.environ)
    env["HF_HOME"] = HF_HOME
    started = time.time()
    result = sh(cmd, env=env, cwd=str(tools))
    elapsed = round(time.time() - started)
    print(f"[llama-50] exit={result.returncode} elapsed={elapsed}s", flush=True)
    return result.returncode, out_json, out_md, elapsed


def consistency_errors(report: dict) -> list[str]:
    """Recompute the recorded summary from the raw case fields.

    Every aggregate must be recomputable from the raw measurements with the
    runner's own percentile algorithm, the pass verdict must match the
    recorded thresholds, and the model identity/config echo must match the
    pinned lane. A mismatch is a kernel error, not a scientific finding.
    """
    errors: list[str] = []
    prefix = "llama-50: "

    if report.get("synthetic") is not False or report.get("surrogate") is not False:
        errors.append(prefix + "report does not state a live non-synthetic non-surrogate run")
    if not str(report.get("mode", "")).startswith("live-"):
        errors.append(prefix + "report mode is not live")

    model = report.get("model", {}) or {}
    if model.get("id") != MODEL_ID:
        errors.append(prefix + "model identity does not match the pinned provider")
    if model.get("device") != "cuda":
        errors.append(prefix + "model device is not cuda")
    if model.get("adapter") != ADAPTER_ID:
        errors.append(prefix + "model adapter does not match the lane")
    if model.get("quantized") is not True:
        errors.append(prefix + "model identity does not record the 4-bit provider")
    if model.get("activationFamily") != "swiglu":
        errors.append(prefix + "activation family is not swiglu")
    if model.get("localFilesOnly") is not True:
        errors.append(prefix + "model identity does not record local-files-only loading")

    oracle = report.get("oracleAdapter", {}) or {}
    if oracle.get("id") != ADAPTER_ID or oracle.get("contract") != "LLMOracleAdapter":
        errors.append(prefix + "oracle adapter block does not match the lane")

    config = report.get("config", {}) or {}
    expected_config = {
        "mlpLayers": [20, 24, 27],
        "mlpFallbackLayers": [16, 18, 20, 22, 24, 26, 27],
        "attentionLayers": [20, 24, 27],
        "attentionFallbackLayers": [16, 18, 20, 22, 24, 26, 27],
        "caseCount": 50,
        "minValidCases": 50,
        "minPassRate": 0.9,
        "minComponentPassRate": 0.9,
        "controlCount": 3,
        "attentionMaxTargetRank": 200,
        "minPathActivationDelta": 0.001,
        "minPathMarginOverControlP95": 0.0001,
        "requireComponentHardening": False,
        "controlSeedMode": "sha256-subject-derived",
        "device": "cuda",
        "adapter": ADAPTER_ID,
    }
    for key, expected in expected_config.items():
        if config.get(key) != expected:
            errors.append(prefix + f"config {key} does not echo the pinned lane ({config.get(key)!r} != {expected!r})")

    claim = report.get("claimBoundary", {}) or {}
    if not claim.get("provesIfPasses") or not claim.get("doesNotProve"):
        errors.append(prefix + "claim boundary is missing")

    summary = report.get("summary", {}) or {}
    cases = report.get("cases", [])
    if not isinstance(cases, list) or not cases:
        errors.append(prefix + "no measured cases")
        return errors
    valid = [case for case in cases if isinstance(case, dict) and case.get("status") != "skipped"]
    passed = [case for case in valid if case.get("status") == "passed"]
    combined_passed = [case for case in valid if (case.get("combined", {}) or {}).get("status") == "passed"]
    if summary.get("validCases") != len(valid):
        errors.append(prefix + "validCases does not match the raw cases")
    if summary.get("passedCases") != len(passed):
        errors.append(prefix + "passedCases does not match the raw cases")
    if summary.get("failedCases") != len(valid) - len(passed):
        errors.append(prefix + "failedCases does not match the raw cases")
    if summary.get("skippedCases") != len(cases) - len(valid):
        errors.append(prefix + "skippedCases does not match the raw cases")
    expected_pass_rate = len(passed) / max(1, len(valid))
    if not isinstance(summary.get("passRate"), (int, float)) or abs(float(summary["passRate"]) - expected_pass_rate) > 1e-12:
        errors.append(prefix + "passRate does not match the raw cases")
    expected_combined_rate = len(combined_passed) / max(1, len(valid))
    if not isinstance(summary.get("combinedGraphPassRate"), (int, float)) or abs(float(summary["combinedGraphPassRate"]) - expected_combined_rate) > 1e-12:
        errors.append(prefix + "combinedGraphPassRate does not match the raw cases")
    if summary.get("failedSubjects") != [case["subject"] for case in valid if case.get("status") != "passed"]:
        errors.append(prefix + "failedSubjects does not match the raw cases")

    def path_field(case: dict, key: str) -> float:
        patch = (case.get("orderedResidualPath", {}) or {}).get("pathPatch", {}) or {}
        return float(patch.get(key, 0.0))

    path_deltas = [path_field(case, "selectedPathActivationDelta") for case in valid]
    control_p95 = [path_field(case, "controlPathP95ActivationDelta") for case in valid]
    path_lifts = [path_field(case, "liftOverControlPathP95") for case in valid]
    combined_drops = [float((case.get("combined", {}) or {}).get("targetLogitDrop", 0.0)) for case in valid]
    percentile_checks = {
        "medianSelectedPathActivationDelta": _percentile(path_deltas, 0.50),
        "p10SelectedPathActivationDelta": _percentile(path_deltas, 0.10),
        "medianControlPathP95ActivationDelta": _percentile(control_p95, 0.50),
        "medianOrderedPathLiftOverControlP95": _percentile(path_lifts, 0.50),
        "medianCombinedTargetLogitDrop": _percentile(combined_drops, 0.50),
    }
    for key, expected in percentile_checks.items():
        recorded = summary.get(key)
        if not isinstance(recorded, (int, float)) or abs(float(recorded) - expected) > 1e-9:
            errors.append(prefix + f"{key} does not recompute from the raw cases")

    expected_summary_pass = (
        len(valid) >= int(config.get("minValidCases", 0))
        and expected_pass_rate >= float(config.get("minPassRate", 1.0))
        and expected_combined_rate >= float(config.get("minComponentPassRate", 1.0))
    )
    if summary.get("pass") is not expected_summary_pass:
        errors.append(prefix + "summary pass/fail does not match the recorded gates")

    # Structural per-case checks: every valid case must carry the measured fields.
    for case in valid:
        subject = case.get("subject")
        if not subject or not case.get("answer"):
            errors.append(prefix + "a valid case records no subject/answer")
        target = case.get("target", {}) or {}
        if not isinstance(target.get("tokenId"), int):
            errors.append(prefix + f"case {subject} records no target token")
        if not str(case.get("prompt", "")).strip():
            errors.append(prefix + f"case {subject} records no prompt")
    return errors


def main() -> None:
    ART.mkdir(parents=True, exist_ok=True)
    tools = _resolve_tools_dir()
    print(f"[dataset] tools dir: {tools}", flush=True)
    info = env_info()
    print(json.dumps(info, indent=1), flush=True)

    install = sh([sys.executable, "-m", "pip", "install", "-q", *PINNED])
    print(f"pip exit={install.returncode}", flush=True)
    if install.returncode != 0:
        raise SystemExit(1)

    # Pre-download the 4-bit provider with a hard cache pre-flight gate.
    download_env = dict(os.environ)
    download_env["HF_HOME"] = HF_HOME
    probe = sh(
        [
            sys.executable,
            "-c",
            "from huggingface_hub import snapshot_download; "
            f"snapshot_download('{MODEL_ID}'); print('downloaded')",
        ],
        env=download_env,
    )
    if probe.returncode != 0:
        raise SystemExit(f"model pre-download failed: {MODEL_ID}")
    cache_dir = f"{HF_HOME}/hub/models--{MODEL_ID.replace('/', '--')}"
    if not os.path.isdir(cache_dir):
        raise SystemExit(f"pre-downloaded model missing from HF_HOME: {MODEL_ID}")

    runner_rc, report_json, report_md, elapsed = run_discovery(tools)

    run_info: dict = {
        "kind": "gcl-kaggle-llama-50-facts-ordered-residual-path-run",
        "modelId": MODEL_ID,
        "variant": VARIANT,
        "pinnedRepoCommit": PINNED_REPO_COMMIT,
        "dataset": PINNED_DATASET,
        "pinnedDeps": PINNED,
        "runnerExit": runner_rc,
        "elapsedSeconds": elapsed,
        "laneArgv": LANE_ARGV,
        "datasetHints": DS_HINTS,
        **info,
    }

    if not report_json.is_file():
        print("[llama-50] runner produced no report; kernel error", flush=True)
        run_info["consistencyErrors"] = ["runner produced no report.json"]
        (ART / "run-summary.json").write_text(json.dumps(run_info, indent=2) + "\n", encoding="utf-8")
        print("GCL_LLAMA_50_FACTS_ORDERED_PATH_KERNEL_ERROR", flush=True)
        raise SystemExit(1)

    report = json.loads(report_json.read_text(encoding="utf-8"))
    errors = consistency_errors(report)
    run_info["consistencyErrors"] = errors
    run_info["summaryPass"] = bool(report.get("summary", {}).get("pass"))
    (ART / VARIANT / "run-info.json").write_text(json.dumps(run_info, indent=2) + "\n", encoding="utf-8")
    (ART / "run-summary.json").write_text(json.dumps(run_info, indent=2) + "\n", encoding="utf-8")
    print("GCL_LLAMA_50_FACTS_ORDERED_PATH_SUMMARY " + json.dumps({
        "runnerExit": runner_rc,
        "summaryPass": run_info["summaryPass"],
        "validCases": report.get("summary", {}).get("validCases"),
        "passedCases": report.get("summary", {}).get("passedCases"),
        "skippedCases": report.get("summary", {}).get("skippedCases"),
        "elapsedSeconds": elapsed,
    }), flush=True)

    if errors:
        print(f"[llama-50] consistency errors: {errors}", flush=True)
        print("GCL_LLAMA_50_FACTS_ORDERED_PATH_KERNEL_ERROR", flush=True)
        raise SystemExit(1)

    print("[llama-50] ARTIFACT_CONSISTENT (summary pass=%s)" % run_info["summaryPass"], flush=True)
    if runner_rc == 0 and run_info["summaryPass"]:
        print("GCL_LLAMA_50_FACTS_ORDERED_PATH_PASS", flush=True)
    else:
        # Honest scientific fail: the discovery report is recorded and
        # internally consistent; this is a finding, not a kernel crash.
        print("GCL_LLAMA_50_FACTS_ORDERED_PATH_FAIL_RECORDED", flush=True)


if __name__ == "__main__":
    main()
