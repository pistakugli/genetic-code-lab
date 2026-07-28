#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.llm_reader.blind_swiglu_oracle_decompiler import (
    apply_fit_env_overrides,
    build_parser as build_decompiler_parser,
    run_solver,
)


def configure_solver_environment() -> None:
    values = {
        "GCL_BLIND_SWIGLU_INDEPENDENT_RNG": "1",
        "GCL_BLIND_SWIGLU_VALIDATION_FULL_TAIL_POLISH": "0",
        "GCL_BLIND_SWIGLU_ASYMPTOTIC_STITCH_LINE_COUNT": "48",
        "GCL_BLIND_SWIGLU_ASYMPTOTIC_STITCH_SAMPLES_PER_LINE": "1537",
        # Boundary crossings are located from output differences along a probe line, and
        # the oracle quantizes to 1e-6. A narrow range packs the samples close together,
        # which shrinks those differences and lets quantization dominate the localization.
        # Widening the range costs no extra queries because the sample count is unchanged.
        # Measured on mistral layer 28: range 6 gives R2 0.978190 and worst 1.195e-04,
        # range 12 gives 0.992712 and 5.846e-05, range 24 gives 0.993663 and 5.421e-05.
        # 12 is taken because 24 only adds 0.001 for twice the deviation from the setting
        # the passing lanes were proven on. The passing mistral layer-11 block improves
        # too, R2 0.999999920 to 0.999999950 and worst 9.755e-05 to 3.206e-05, so this is
        # not a weak-block trade.
        "GCL_BLIND_SWIGLU_ASYMPTOTIC_STITCH_LINE_RANGE": "12",
        # Which range wins is block-dependent, so it is chosen rather than assumed: the
        # stitch runs at 12 and at 6 and the candidate with the lower error on the solver's
        # own selection sample is kept. That sample is a separate oracle draw from the
        # validation set the gate scores, so nothing is fitted to the graded points.
        # Measured on llama layer 1, which prefers the narrower range: the search returns
        # R2 0.998556 and worst 2.785e-03, matching a fixed range-6 run exactly, against
        # R2 0.998301 and worst 3.123e-03 for a fixed range 12. One extra stitch costs
        # about 120k queries inside the 1.5M budget.
        "GCL_BLIND_SWIGLU_ASYMPTOTIC_STITCH_LINE_RANGE_LADDER": "6",
        "GCL_BLIND_SWIGLU_ASYMPTOTIC_STITCH_OFFSET_SCALE": "0.75",
        "GCL_BLIND_SWIGLU_ASYMPTOTIC_STITCH_SCALE": "512",
        "GCL_BLIND_SWIGLU_ASYMPTOTIC_STITCH_HESSIAN_NORMAL_DELTA": "0.10",
        "GCL_BLIND_SWIGLU_ASYMPTOTIC_STITCH_HESSIAN_REPRESENTATIVES": "8",
        "GCL_BLIND_SWIGLU_ASYMPTOTIC_STITCH_MIN_GATE_SUPPORT": "4",
        "GCL_BLIND_SWIGLU_ASYMPTOTIC_STITCH_MIN_UP_SUPPORT": "3",
        "GCL_BLIND_SWIGLU_ASYMPTOTIC_COLD_START_GATE_NORM": "0.25",
        "GCL_BLIND_SWIGLU_ASYMPTOTIC_COLD_START_ORIENTATION_SAMPLES": "4096",
        "GCL_BLIND_SWIGLU_ASYMPTOTIC_STITCH_STRUCTURE_CALIBRATION": "1",
        "GCL_BLIND_SWIGLU_ASYMPTOTIC_STITCH_STRUCTURE_REFINE": "1",
        "GCL_BLIND_SWIGLU_ASYMPTOTIC_STITCH_VALUE_POLISH": "0",
        "GCL_BLIND_SWIGLU_GATE_NORM_CALIBRATION_STEPS": "200",
        "GCL_BLIND_SWIGLU_GATE_NORM_CALIBRATION_RESTARTS": "5",
        "GCL_BLIND_SWIGLU_GATE_NORM_CALIBRATION_LR": "0.8",
        "GCL_BLIND_SWIGLU_GATE_NORM_CALIBRATION_MAX_NORM_RATIO": "8",
        "GCL_BLIND_SWIGLU_STRUCTURE_REFINE_STEPS": "220",
        "GCL_BLIND_SWIGLU_STRUCTURE_REFINE_RESTARTS": "3",
        "GCL_BLIND_SWIGLU_STRUCTURE_REFINE_LR": "0.6",
    }
    for key, value in values.items():
        os.environ[key] = value


def build_worker_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Opaque eval-only SwiGLU recovery worker")
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--dimension", type=int, required=True)
    parser.add_argument("--atom-count", type=int, required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--frozen-output", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser


def main() -> int:
    parser = build_worker_parser()
    worker = parser.parse_args()
    if worker.dimension <= 0 or worker.atom_count <= 0 or worker.budget <= 0:
        parser.error("dimension, atom-count, and budget must be positive")
    configure_solver_environment()
    decompiler_args = build_decompiler_parser().parse_args([
        "--oracle-socket", worker.endpoint,
        "--require-opaque-oracle",
        "--defer-posthoc-truth",
        "--frozen-fit-output", worker.frozen_output,
        "--recovery-path", "asymptotic-cold-start",
        "--expected-dimension", str(worker.dimension),
        "--active-rank", str(worker.atom_count * 2),
        "--atom-count", str(worker.atom_count),
        "--gradient-mode", "symmetric-multiscale-coordinate",
        "--basis-moment", "centered",
        "--probe-points", "65",
        "--finite-difference-steps", "1.0,2.0",
        "--cubic-direction-count", "2048",
        "--cubic-holdout-count", "512",
        "--cubic-steps", "0.35,0.25",
        "--train-count", "16000",
        "--train-sampling-mode", "gaussian",
        # A fixed input scale is not selection-independent. Blocks differ in output
        # amplitude by orders of magnitude, and 0.45 leaves a weak block's signal too
        # close to the oracle quantization floor to fit accurately. The solver now picks
        # the scale per block from oracle-observed amplitude; 0.45 stays the first
        # candidate, so a block that does not need the adaptation is unaffected apart
        # from one 512-call probe batch. The 0.9 ceiling is measured: 1.1 is already past
        # the optimum and 1.4 destroys structure recovery outright.
        "--input-scale", "0.45",
        "--auto-input-scale",
        "--auto-input-scale-candidates", "0.45,0.6,0.75,0.9",
        "--auto-input-scale-probe-count", "512",
        "--auto-input-scale-quantization-ratio", "1000",
        "--validation-count", "4096",
        "--validation-scale", "0.65",
        "--fit-gradient-count", "256",
        "--fit-gradient-mode", "forward",
        "--fit-gradient-step", "0.04",
        "--fit-gradient-steps", "0.04",
        "--fit-selection-count", "4096",
        "--fit-selection-scale", "0.65",
        "--disable-adaptive-rescue",
        "--seed", "260723",
        "--min-r2", "0.999",
        "--max-mae", "0.001",
        "--max-worst-error", "0.005",
        "--min-subspace-cosine", "0.999",
        "--max-oracle-calls", str(worker.budget),
        "--output-json", worker.output_json,
        "--output-md", worker.output_md,
        "--progress",
    ])
    apply_fit_env_overrides(decompiler_args)
    return run_solver(decompiler_args)


if __name__ == "__main__":
    raise SystemExit(main())
