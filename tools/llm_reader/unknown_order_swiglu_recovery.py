#!/usr/bin/env python3
"""Unknown-order SwiGLU recovery ladder runner (preregistered protocol).

Executes the frozen order-inference protocol of
docs/research/2026-11-19-unknown-order-empirical-swiglu-recovery.md:
fit a ladder of candidate atom counts against an eval-only oracle whose
ready frame carries no unitCount, grade each candidate on the empirical
selection pool, early-stop at the first sufficient candidate, extend two
extra ladder steps for the non-identifiability check, and report the
inferred order (or an explicit no-sufficient-order verdict).

Access separation is the one from `isolated_swiglu_cold_start.py`:
- the teacher manifest travels by inherited descriptor to the oracle
  server only; the solver subprocess gets a sanitized environment and is
  strace-witnessed when strace is available;
- the solver-visible pool copy carries x values and split indices only;
- teacher-side response tables (selection/validation/OOD f values) live in
  a teacher-side subdirectory the solver command never mentions.

This runner is solver-side infrastructure: it knows the ladder and the
grading rule, and it computes grades from teacher-side response tables plus
frozen fits, but it never reads source weights (synthetic teachers except
their own tables by construction, real teachers answer only through the
oracle protocol's ready frame and the runner's own teacher tables).

Sanity mode (--synthetic-teacher) runs the same protocol against a recorded
synthetic block through `synthetic_swiglu_oracle_server.py`; reports are
labeled synthetic and are never protocol evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.llm_reader.blind_swiglu_oracle_decompiler import (
    predict_swiglu,
    regression_metrics,
)
from tools.llm_reader.blind_swiglu_solver_worker import configure_solver_environment
from tools.llm_reader.isolated_swiglu_cold_start import (
    classify_solver_file_access,
    sanitized_env,
    sanitized_solver_env,
    sha256_file,
)

TOOLS = REPO_ROOT / "tools" / "llm_reader"
DECOMPILER = TOOLS / "blind_swiglu_oracle_decompiler.py"
ORACLE_SERVER = TOOLS / "swiglu_oracle_server.py"
SYNTHETIC_SERVER = TOOLS / "synthetic_swiglu_oracle_server.py"


def parse_ladder(raw: str) -> list[int]:
    ladder = [int(value) for value in raw.split(",") if value.strip()]
    if not ladder or any(value <= 0 for value in ladder) or len(set(ladder)) != len(ladder):
        raise RuntimeError(f"ladder must be a non-empty list of distinct positive ints, got {raw!r}")
    return sorted(ladder)


def sha256_file_quiet(path: Path) -> str:
    return sha256_file(path)


def load_pool(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as data:
        x = np.asarray(data["x"], dtype=np.float64)
        indices = {
            name: np.asarray(data[name], dtype=np.int64)
            for name in ("train", "selection", "validation")
        }
    total = int(x.shape[0])
    seen: set[int] = set()
    for name, idx in indices.items():
        if idx.ndim != 1 or idx.size == 0 or idx.min() < 0 or idx.max() >= total:
            raise RuntimeError(f"pool split {name!r} invalid for shape {x.shape}")
        overlap = seen.intersection(idx.tolist())
        if overlap:
            raise RuntimeError(f"pool splits overlap at {name!r}")
        seen.update(idx.tolist())
    return x, indices


def synthetic_teacher_fn(npz_path: Path) -> tuple[Callable[[np.ndarray], np.ndarray], dict[str, Any]]:
    with np.load(npz_path, allow_pickle=False) as data:
        gate = np.asarray(data["gate"], dtype=np.float64)
        up = np.asarray(data["up"], dtype=np.float64)
        scales = np.asarray(data["scales"], dtype=np.float64)
        bias = float(np.asarray(data["bias"]).item()) if "bias" in data else 0.0
    dimension = int(gate.shape[1])

    def teacher_fn(points: np.ndarray) -> np.ndarray:
        values = np.asarray(points, dtype=np.float64)
        gate_pre = values @ gate.T
        up_pre = values @ up.T
        activated = gate_pre / (1.0 + np.exp(-gate_pre))
        return (activated * up_pre) @ scales + bias

    meta = {"kind": "synthetic", "dimension": dimension, "atomCount": int(gate.shape[0])}
    return teacher_fn, meta


def real_teacher_fn(manifest_path: Path) -> tuple[Callable[[np.ndarray], np.ndarray], dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    from tools.llm_reader.swiglu_oracle_server import load_teacher

    gate, up, scales, metadata = load_teacher(manifest)

    def to_numpy(value: object) -> np.ndarray:
        # load_teacher answers torch host tensors; the grading side is numpy
        # throughout (a numpy @ Tensor mix raised in the first real-model run)
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        return np.asarray(value, dtype=np.float64)

    gate_np, up_np, scales_np = to_numpy(gate), to_numpy(up), to_numpy(scales)
    bias = 0.0

    def teacher_fn(points: np.ndarray) -> np.ndarray:
        values = np.asarray(points, dtype=np.float64)
        gate_pre = values @ gate_np.T
        up_pre = values @ up_np.T
        activated = gate_pre / (1.0 + np.exp(-gate_pre))
        return (activated * up_pre) @ scales_np + bias

    meta = {"kind": "real-model", "dimension": int(metadata["hiddenSize"]),
            "model": str(manifest.get("model", "")), "layer": int(manifest.get("layer", -1))}
    return teacher_fn, meta


def relative_mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    std = float(np.std(actual))
    if std <= 0.0:
        raise RuntimeError("teacher response std is zero on the graded pool; r is undefined")
    return float(np.mean(np.abs(np.asarray(actual) - np.asarray(predicted))) / std)


def evaluate_frozen_fit(
    frozen_fit_path: Path,
    pool_x: np.ndarray,
    split_indices: dict[str, np.ndarray],
    teacher_fn: Callable[[np.ndarray], np.ndarray],
    ood_points: np.ndarray,
) -> dict[str, float]:
    with np.load(frozen_fit_path, allow_pickle=False) as data:
        basis = np.asarray(data["basis"], dtype=np.float64)
        gates = np.asarray(data["gates"], dtype=np.float64)
        ups = np.asarray(data["ups"], dtype=np.float64)
        scales = np.asarray(data["scales"], dtype=np.float64)
        bias = float(np.asarray(data["bias"]).item())
    fit = {"gates": gates, "ups": ups, "scales": scales, "bias": bias}
    if basis.shape[1] != pool_x.shape[1]:
        raise RuntimeError(f"frozen fit basis dimension {basis.shape} != pool dimension {pool_x.shape[1]}")

    def f_hat(points: np.ndarray) -> np.ndarray:
        return predict_swiglu(points @ basis.T, fit)

    grades: dict[str, float] = {}
    for name, split in (("selection", "selection"), ("validation", "validation")):
        points = pool_x[split_indices[split]]
        grades[f"r{name.title()}"] = relative_mae(teacher_fn(points), f_hat(points))
    ood_actual = teacher_fn(ood_points)
    grades["rOod"] = relative_mae(ood_actual, f_hat(ood_points))
    metrics = regression_metrics(teacher_fn(pool_x[split_indices["validation"]]),
                                 f_hat(pool_x[split_indices["validation"]]))
    grades["validationR2"] = float(metrics["r2"])
    grades["validationMae"] = float(metrics["mae"])
    return grades


def run_one_fit(
    candidate: int,
    *,
    output_dir: Path,
    pool_x: np.ndarray,
    pool_splits: dict[str, np.ndarray],
    dimension: int,
    oracle_budget: int,
    quantization_step: float,
    solver_timeout_seconds: int,
    solver_seed: int,
    oracle_server_command: list[str],
    oracle_manifest: dict[str, Any],
    pool_copy_path: Path,
    fit_train_count: int,
) -> dict[str, Any]:
    fit_dir = output_dir / f"fits" / f"m{candidate}"
    fit_dir.mkdir(parents=True, exist_ok=True)
    socket_path = fit_dir / "oracle.sock"
    witness_path = fit_dir / "oracle-witness.json"
    frozen_fit_path = fit_dir / "frozen-recovery.npz"
    solver_json = fit_dir / "solver-report.json"
    solver_md = fit_dir / "solver-report.md"
    solver_stdout = fit_dir / "solver-stdout.log"
    solver_stderr = fit_dir / "solver-stderr.log"
    strace_path = fit_dir / "solver-file-access.strace"
    oracle_stderr = fit_dir / "oracle-stderr.log"

    validation_gate_count = int(pool_splits["validation"].size) // 2
    selection_count = int(pool_splits["selection"].size)
    train_count = min(fit_train_count, int(pool_splits["train"].size))
    if validation_gate_count <= 0 or selection_count <= 0 or train_count <= 0:
        raise RuntimeError("empirical pool splits are too small for the frozen counts")

    solver_command = [
        sys.executable, str(DECOMPILER),
        "--oracle-socket", str(socket_path),
        "--require-opaque-oracle",
        "--unknown-order",
        "--defer-posthoc-truth",
        "--frozen-fit-output", str(frozen_fit_path),
        "--recovery-path", "asymptotic-cold-start",
        "--expected-dimension", str(dimension),
        "--active-rank", str(candidate * 2),
        "--atom-count", str(candidate),
        "--gradient-mode", "symmetric-multiscale-coordinate",
        "--basis-moment", "centered",
        "--probe-points", "65",
        "--finite-difference-steps", "1.0,2.0",
        "--cubic-direction-count", "2048",
        "--cubic-holdout-count", "512",
        "--cubic-steps", "0.35,0.25",
        "--train-count", str(train_count),
        "--train-sampling-mode", "empirical",
        "--empirical-inputs-path", str(pool_copy_path),
        "--input-scale", "0.45",
        "--auto-input-scale",
        "--auto-input-scale-candidates", "0.45,0.6,0.75,0.9",
        "--auto-input-scale-probe-count", "512",
        "--auto-input-scale-quantization-ratio", "1000",
        "--validation-count", str(validation_gate_count),
        "--validation-scale", "0.65",
        "--fit-gradient-count", "256",
        "--fit-gradient-mode", "forward",
        "--fit-gradient-step", "0.04",
        "--fit-gradient-steps", "0.04",
        "--fit-selection-count", str(selection_count),
        "--fit-selection-scale", "0.65",
        "--disable-adaptive-rescue",
        "--seed", str(solver_seed),
        "--min-r2", "0.999",
        "--max-mae", "0.001",
        "--max-worst-error", "0.005",
        "--min-subspace-cosine", "0.999",
        "--max-oracle-calls", str(oracle_budget),
        "--output-json", str(solver_json),
        "--output-md", str(solver_md),
        "--progress",
    ]
    forbidden_flags = {"--model", "--layer", "--unit", "--units", "--output-channel"}
    violated = sorted(forbidden_flags.intersection(solver_command))
    if violated:
        raise RuntimeError(f"solver command leaks teacher locators: {violated}")

    manifest_read, manifest_write = os.pipe()
    resolved_server_command = [
        (
            {"@FD@": str(manifest_read), "@SOCKET@": str(socket_path), "@WITNESS@": str(witness_path)}.get(token, token)
            if isinstance(token, str) else token
        )
        for token in oracle_server_command
    ]
    env = sanitized_env()
    solver_env = sanitized_solver_env()
    configure_solver_environment()
    strace = shutil.which("strace")
    launch = solver_command
    if strace:
        launch = [
            strace, "-f", "-qq", "-e", "trace=open,openat,statx,readlink",
            "-o", str(strace_path), "--", *solver_command,
        ]
    started_ns = time.time_ns()
    oracle_process: subprocess.Popen[bytes] | None = None
    solver_process: subprocess.Popen[bytes] | None = None
    solver_exit: int | None = None
    oracle_exit: int | None = None
    try:
        with oracle_stderr.open("wb") as oracle_err:
            oracle_process = subprocess.Popen(
                resolved_server_command,
                cwd=str(REPO_ROOT),
                env=env,
                stderr=oracle_err,
                pass_fds=(manifest_read,),
                close_fds=True,
            )
        os.close(manifest_read)
        manifest_read = -1
        with os.fdopen(manifest_write, "w", encoding="utf-8") as stream:
            json.dump(oracle_manifest, stream)
        manifest_write = -1
        socket_deadline = time.monotonic() + 300
        while not socket_path.exists():
            if oracle_process.poll() is not None:
                raise RuntimeError(
                    f"oracle exited before socket ready: {oracle_process.returncode}; "
                    f"stderr={oracle_stderr.read_text(encoding='utf-8', errors='replace')[-4000:]}"
                )
            if time.monotonic() >= socket_deadline:
                raise RuntimeError("timed out waiting for oracle socket")
            time.sleep(0.05)
        with solver_stdout.open("wb") as out, solver_stderr.open("wb") as err:
            solver_process = subprocess.Popen(
                launch,
                cwd=str(REPO_ROOT),
                env=solver_env,
                stdout=out,
                stderr=err,
                close_fds=True,
            )
            solver_exit = solver_process.wait(timeout=solver_timeout_seconds)
        oracle_exit = oracle_process.wait(timeout=60)
    finally:
        for descriptor in (manifest_read, manifest_write):
            if descriptor >= 0:
                os.close(descriptor)
        if solver_process is not None and solver_process.poll() is None:
            solver_process.kill()
        if oracle_process is not None and oracle_process.poll() is None:
            oracle_process.kill()
    elapsed_seconds = (time.time_ns() - started_ns) / 1e9

    witness: dict[str, Any] = {}
    if witness_path.exists():
        witness = json.loads(witness_path.read_text(encoding="utf-8"))
    access_rows: list[dict[str, Any]] = []
    if strace and strace_path.exists():
        classification = classify_solver_file_access(
            strace_path.read_text(encoding="utf-8", errors="replace"),
            sensitive_control_paths=[],
        )
        access_rows = classification.get("modelAccessRows", [])

    if not frozen_fit_path.exists():
        # A candidate whose asymptotic structure court refuses produced no frozen fit.
        # For the ladder that is a legitimate underfit outcome, not a lane error:
        # record the candidate as failed (never sufficient) and keep climbing.
        return {
            "candidate": int(candidate),
            "fitDir": str(fit_dir),
            "fitFailed": True,
            "failureKind": "no-frozen-fit",
            "solverExit": solver_exit,
            "oracleExit": oracle_exit,
            "elapsedSeconds": elapsed_seconds,
            "oracleCalls": int(witness.get("queryCount", -1)) if witness else -1,
            "straceModelAccessRows": access_rows,
            "frozenFitPath": None,
        }
    if not solver_json.exists():
        raise RuntimeError(
            f"candidate {candidate}: frozen fit exists but solver report is missing "
            f"(exit={solver_exit}); this combination is a lane error, not an underfit"
        )
    solver_report = json.loads(solver_json.read_text(encoding="utf-8"))
    return {
        "candidate": int(candidate),
        "fitDir": str(fit_dir),
        "solverExit": solver_exit,
        "oracleExit": oracle_exit,
        "elapsedSeconds": elapsed_seconds,
        "oracleCalls": int(witness.get("queryCount", -1)),
        "solverGatePassed": bool(solver_report.get("summary", {}).get("passed", solver_exit == 0)),
        "straceModelAccessRows": access_rows,
        "frozenFitPath": str(frozen_fit_path),
    }


def build_oracle_server_command(base: list[str] | None, *, synthetic: bool) -> list[str]:
    if base:
        resolved = [str(part) for part in base]
        for placeholder in ("@FD@", "@SOCKET@", "@WITNESS@"):
            if not any(placeholder in token for token in resolved):
                raise RuntimeError(f"oracle server command must contain {placeholder} placeholder")
        return resolved
    server = SYNTHETIC_SERVER if synthetic else ORACLE_SERVER
    return [
        sys.executable, str(server),
        "--hide-unit-count",
        "--manifest-fd", "@FD@",
        "--socket", "@SOCKET@",
        "--witness", "@WITNESS@",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Unknown-order SwiGLU recovery ladder runner")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--empirical-pool", required=True)
    parser.add_argument("--ladder", default="4,5,6,8,10,12,14,16,18,20,24")
    parser.add_argument("--r-threshold", type=float, default=0.05)
    parser.add_argument("--extra-ladder-steps", type=int, default=2)
    parser.add_argument("--oracle-budget", type=int, default=1_500_000)
    parser.add_argument("--max-fits", type=int, default=9)
    parser.add_argument("--quantization-step", type=float, default=1e-6)
    parser.add_argument("--teacher-manifest", default="")
    parser.add_argument("--synthetic-teacher", default="")
    parser.add_argument("--oracle-server-command", nargs="+", default=None,
                        help="server command with @FD@/@SOCKET@/@WITNESS@ placeholders (default: repo oracle server, unknown-order mode)")
    parser.add_argument("--solver-timeout", type=int, default=14_400)
    parser.add_argument("--solver-seed", type=int, default=260723)
    parser.add_argument("--fit-train-count", type=int, default=16_000)
    parser.add_argument("--ood-count", type=int, default=4096)
    parser.add_argument("--ood-scale", type=float, default=0.65)
    parser.add_argument("--ood-seed", type=int, default=261135)
    parser.add_argument("--null-control", action="store_true",
                        help="also run the preregistered permutation null control after the real ladder")
    parser.add_argument("--null-seed", type=int, default=261131)
    parser.add_argument("--null-permutation-seed", type=int, default=261131)
    args = parser.parse_args()

    if bool(args.teacher_manifest) == bool(args.synthetic_teacher):
        raise RuntimeError("exactly one of --teacher-manifest or --synthetic-teacher is required")
    if args.r_threshold <= 0.0 or args.extra_ladder_steps < 0 or args.oracle_budget <= 0 or args.max_fits <= 0:
        raise RuntimeError("thresholds/budgets must be positive")
    if not 0.0 < args.quantization_step or args.quantization_step > 0.1:
        raise RuntimeError("quantization step out of sane range")

    pool_path = Path(args.empirical_pool).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pool_x, pool_splits = load_pool(pool_path)
    dimension = int(pool_x.shape[1])

    if args.synthetic_teacher:
        teacher_fn, teacher_meta = synthetic_teacher_fn(Path(args.synthetic_teacher).resolve())
    else:
        teacher_fn, teacher_meta = real_teacher_fn(Path(args.teacher_manifest).resolve())
    if teacher_meta["dimension"] != dimension:
        raise RuntimeError(f"teacher dimension {teacher_meta['dimension']} != pool dimension {dimension}")

    # solver-visible pool copy: x and split indices only, never responses
    solver_inputs_dir = output_dir / "solver-inputs"
    solver_inputs_dir.mkdir(parents=True, exist_ok=True)
    pool_copy_path = solver_inputs_dir / "empirical-pool.npz"
    np.savez_compressed(
        pool_copy_path,
        x=pool_x.astype(np.float32),
        train=pool_splits["train"],
        selection=pool_splits["selection"],
        validation=pool_splits["validation"],
    )
    teacher_dir = output_dir / "teacher-side"
    teacher_dir.mkdir(parents=True, exist_ok=True)

    selection_points = pool_x[pool_splits["selection"]]
    validation_points = pool_x[pool_splits["validation"]]
    ood_points = np.random.default_rng(args.ood_seed).normal(
        0.0, args.ood_scale, size=(args.ood_count, dimension)
    )
    selection_responses = teacher_fn(selection_points)
    validation_responses = teacher_fn(validation_points)
    ood_responses = teacher_fn(ood_points)
    np.savez_compressed(
        teacher_dir / "eval-responses.npz",
        selection_x=selection_points,
        selection_f=selection_responses,
        validation_x=validation_points,
        validation_f=validation_responses,
        ood_x=ood_points,
        ood_f=ood_responses,
    )

    ladder = parse_ladder(args.ladder)
    oracle_server_command = build_oracle_server_command(args.oracle_server_command, synthetic=bool(args.synthetic_teacher))
    if args.synthetic_teacher:
        oracle_manifest = {
            "syntheticTeacher": str(Path(args.synthetic_teacher).resolve()),
            "quantizationStep": float(args.quantization_step),
            "maxOracleCalls": int(args.oracle_budget),
        }
    else:
        manifest = json.loads(Path(args.teacher_manifest).read_text(encoding="utf-8"))
        oracle_manifest = {
            "model": manifest["model"],
            "layer": int(manifest["layer"]),
            "units": [int(value) for value in manifest["units"]],
            "outputChannel": int(manifest["outputChannel"]),
            "device": str(manifest.get("device", "cuda")),
            "localFilesOnly": bool(manifest.get("localFilesOnly", True)),
            "quantizationStep": float(args.quantization_step),
            "maxOracleCalls": int(args.oracle_budget),
            "batchSize": int(manifest.get("batchSize", 2048)),
        }

    def grade_fit(fit_record: dict[str, Any]) -> dict[str, Any]:
        if fit_record.get("fitFailed"):
            merged = dict(fit_record)
            merged["sufficient"] = False
            return merged
        grades = evaluate_frozen_fit(
            Path(fit_record["frozenFitPath"]),
            pool_x,
            pool_splits,
            teacher_fn,
            ood_points,
        )
        merged = dict(fit_record)
        merged.update(grades)
        merged["sufficient"] = bool(grades["rSelection"] <= args.r_threshold)
        return merged

    fits: list[dict[str, Any]] = []
    traversed: list[int] = []
    inferred_order: int | None = None
    upper_region_non_identifiable = False
    first_sufficient_index: int | None = None
    for index, candidate in enumerate(ladder):
        if len(fits) >= args.max_fits:
            break
        record = run_one_fit(
            candidate,
            output_dir=output_dir,
            pool_x=pool_x,
            pool_splits=pool_splits,
            dimension=dimension,
            oracle_budget=args.oracle_budget,
            quantization_step=args.quantization_step,
            solver_timeout_seconds=args.solver_timeout,
            solver_seed=args.solver_seed,
            oracle_server_command=oracle_server_command,
            oracle_manifest=oracle_manifest,
            pool_copy_path=pool_copy_path,
            fit_train_count=args.fit_train_count,
        )
        graded = grade_fit(record)
        fits.append(graded)
        traversed.append(candidate)
        print(json.dumps({
            "candidate": graded["candidate"],
            "rSelection": graded.get("rSelection"),
            "rValidation": graded.get("rValidation"),
            "rOod": graded.get("rOod"),
            "sufficient": graded["sufficient"],
            "fitFailed": bool(graded.get("fitFailed")),
            "oracleCalls": graded["oracleCalls"],
        }), flush=True)
        if graded["sufficient"]:
            if first_sufficient_index is None:
                first_sufficient_index = index
            if index == first_sufficient_index:
                inferred_order = candidate
                extra = [value for value in ladder[index + 1:]][: args.extra_ladder_steps]
                if len(extra) < args.extra_ladder_steps:
                    upper_region_non_identifiable = False
                    break
                continue
        if inferred_order is not None and index >= first_sufficient_index + args.extra_ladder_steps:
            break

    if first_sufficient_index is not None:
        sufficient_block = [fit for fit in fits if fit["sufficient"]]
        first_r = fits[first_sufficient_index]["rSelection"]
        later_sufficient = [fit for fit in sufficient_block if fit["candidate"] > inferred_order]
        upper_region_non_identifiable = bool(
            later_sufficient
            and all(fit["rSelection"] <= 2.0 * first_r for fit in later_sufficient)
        )

    null_control: dict[str, Any] | None = None
    if args.null_control:
        if inferred_order is not None:
            null_candidates = [value for value in ladder if value <= inferred_order]
        else:
            null_candidates = list(traversed)
        null_fits: list[dict[str, Any]] = []
        permutation = np.random.default_rng(args.null_seed).permutation(selection_points.shape[0])
        permuted_selection = selection_responses[permutation]
        null_server_command = [str(part) for part in oracle_server_command]
        if not any("--permute-responses-seed" in str(part) for part in null_server_command):
            # insert the permutation flag right before the manifest-fd placeholder; both repo
            # servers accept it, and the synthetic server implements it over the batch
            insert_at = len(null_server_command)
            for position, token in enumerate(null_server_command):
                if str(token).startswith("--manifest-fd"):
                    insert_at = position
                    break
            null_server_command[insert_at:insert_at] = [
                "--permute-responses-seed", str(args.null_permutation_seed),
            ]
        for candidate in null_candidates:
            if len(null_fits) >= args.max_fits:
                break
            record = run_one_fit(
                candidate,
                output_dir=output_dir / "null-control",
                pool_x=pool_x,
                pool_splits=pool_splits,
                dimension=dimension,
                oracle_budget=args.oracle_budget,
                quantization_step=args.quantization_step,
                solver_timeout_seconds=args.solver_timeout,
                solver_seed=args.solver_seed,
                oracle_server_command=null_server_command,
                oracle_manifest=oracle_manifest,
                pool_copy_path=pool_copy_path,
                fit_train_count=args.fit_train_count,
            )
            # null grading: same frozen fit machinery against the frozen permuted
            # response vector; the oracle-side permutation and the grading
            # permutation are independent structure-free draws by design
            if record.get("fitFailed"):
                merged = dict(record)
                merged["rNullSelection"] = None
                merged["sufficient"] = False
                merged["gradesReference"] = "permutation-nulled responses"
                null_fits.append(merged)
                print(json.dumps({
                    "nullCandidate": merged["candidate"],
                    "rNullSelection": None,
                    "sufficient": False,
                    "fitFailed": True,
                    "oracleCalls": merged["oracleCalls"],
                }), flush=True)
                continue
            with np.load(record["frozenFitPath"], allow_pickle=False) as data:
                basis = np.asarray(data["basis"], dtype=np.float64)
                fit = {
                    "gates": np.asarray(data["gates"], dtype=np.float64),
                    "ups": np.asarray(data["ups"], dtype=np.float64),
                    "scales": np.asarray(data["scales"], dtype=np.float64),
                    "bias": float(np.asarray(data["bias"]).item()),
                }
            f_hat_selection = predict_swiglu(selection_points @ basis.T, fit)
            r_null = relative_mae(permuted_selection, f_hat_selection)
            merged = dict(record)
            merged["rNullSelection"] = r_null
            merged["sufficient"] = bool(r_null <= args.r_threshold)
            merged["gradesReference"] = "permutation-nulled responses"
            null_fits.append(merged)
            print(json.dumps({
                "nullCandidate": merged["candidate"],
                "rNullSelection": merged["rNullSelection"],
                "sufficient": merged["sufficient"],
                "oracleCalls": merged["oracleCalls"],
            }), flush=True)
        null_control = {
            "candidates": null_candidates,
            "gradingPermutationSeed": int(args.null_seed),
            "serverPermutationSeed": int(args.null_permutation_seed),
            "fits": [
                {key: value for key, value in fit.items() if key != "frozenFitPath"}
                for fit in null_fits
            ],
            "verdict": (
                "no-sufficient-order"
                if not any(fit["sufficient"] for fit in null_fits)
                else "SUFFICIENT-UNDER-PERMUTATION"
            ),
        }

    verdict = "inferred-order" if inferred_order is not None else "no-sufficient-order"
    report = {
        "schemaVersion": 1,
        "kind": "unknown-order-swiglu-ladder-report",
        "synthetic": bool(args.synthetic_teacher),
        "teacher": teacher_meta,
        "pool": {
            "path": str(pool_path),
            "sha256": sha256_file_quiet(pool_path),
            "pointCount": int(pool_x.shape[0]),
            "dimension": dimension,
            "splits": {name: int(idx.size) for name, idx in pool_splits.items()},
        },
        "protocol": {
            "ladder": ladder,
            "rThreshold": float(args.r_threshold),
            "extraLadderSteps": int(args.extra_ladder_steps),
            "oracleBudgetPerFit": int(args.oracle_budget),
            "maxFits": int(args.max_fits),
            "quantizationStep": float(args.quantization_step),
            "oodCount": int(args.ood_count),
            "oodSeed": int(args.ood_seed),
        },
        "fits": [
            {key: value for key, value in fit.items() if key not in ("frozenFitPath",)}
            for fit in fits
        ],
        "inferredOrder": inferred_order,
        "verdict": verdict,
        "upperRegionNonIdentifiable": bool(upper_region_non_identifiable),
        "nullControl": null_control,
    }
    report_path = output_dir / "case-report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Unknown-order SwiGLU recovery ladder report",
        "",
        f"- synthetic teacher: {report['synthetic']}",
        f"- pool: {report['pool']['pointCount']} points, dimension {dimension}, splits {report['pool']['splits']}",
        f"- ladder: {ladder} (early-stop threshold r_sel <= {args.r_threshold})",
        "",
        "| candidate | r_sel | r_eval | r_ood | sufficient | fit failed | oracle calls |",
        "|---|---|---|---|---|---|---|",
    ]
    for fit in fits:
        lines.append(
            f"| {fit['candidate']} | "
            + (f"{fit['rSelection']:.5f}" if "rSelection" in fit else "-")
            + " | " + (f"{fit['rValidation']:.5f}" if "rValidation" in fit else "-")
            + " | " + (f"{fit['rOod']:.5f}" if "rOod" in fit else "-")
            + f" | {fit['sufficient']} | {bool(fit.get('fitFailed'))} | {fit['oracleCalls']} |"
        )
    lines += [
        "",
        f"- inferred order: {inferred_order if inferred_order is not None else 'none'}",
        f"- verdict: {verdict}",
        f"- upper-region non-identifiable: {upper_region_non_identifiable}",
    ]
    if null_control:
        lines += [
            "",
            "## Null permutation control",
            "",
            f"- candidates: {null_control['candidates']}",
            f"- verdict: {null_control['verdict']}",
        ]
    (output_dir / "case-report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), "inferredOrder": inferred_order, "verdict": verdict}, indent=2))
    return 0 if verdict == "inferred-order" else 1


if __name__ == "__main__":
    raise SystemExit(main())
