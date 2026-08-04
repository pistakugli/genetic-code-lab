#!/usr/bin/env python3
"""Find what maximally drives a recovered unit, then check the real unit agrees.

The third falsifiable claim, and the one closest to what people mean by a unit's "meaning"
without ever using the word. If the recovered parameters describe the unit, then inputs
constructed to maximise the recovered activation should also maximise the real one, and inputs
that leave the recovered unit cold should leave the real one cold.

The test does not search over text. It searches over the residual stream directly, where the
recovered model is a closed-form function and the answer can be computed rather than hunted:
for `silu(g.x)(u.x)` under a norm budget the drive is largest where `x` aligns with both
directions, so a family of candidates spanned by `g` and `u` is enough to rank them.

Reported as rank correlation between predicted and actual drive over that family, plus the
actual activation at the predicted optimum as a fraction of the best actual activation seen.
A unit whose recovery is wrong scores near zero on both, which is what makes this falsifiable
rather than illustrative.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def candidate_inputs(gate: np.ndarray, up: np.ndarray, norm: float, count: int) -> np.ndarray:
    """Inputs spanned by the unit's own two directions, at fixed norm.

    The drive `silu(g.x)(u.x)` depends on `x` only through `g.x` and `u.x`, so the plane those
    two span contains the maximiser at any fixed norm. Sweeping angles in that plane is a
    complete search, not a sample of one.
    """
    gate_unit = gate / np.linalg.norm(gate)
    residual = up - (up @ gate_unit) * gate_unit
    residual_norm = float(np.linalg.norm(residual))
    second = residual / residual_norm if residual_norm > 1e-12 else np.roll(gate_unit, 1)
    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    return norm * (np.cos(angles)[:, None] * gate_unit[None, :] + np.sin(angles)[:, None] * second[None, :])


def drive(gate: np.ndarray, up: np.ndarray, points: np.ndarray) -> np.ndarray:
    z = points @ gate
    return (z / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))) * (points @ up)


def rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    # Guard on the signals, not on their ranks. `argsort(argsort(x))` of a constant array is
    # 0..n-1, which has plenty of variance, so a degenerate unit scored a perfect 1.0 - the
    # best possible result for the one case that carries no information at all.
    if float(np.std(left)) < 1e-12 or float(np.std(right)) < 1e-12:
        return 0.0
    order_left = np.argsort(np.argsort(left)).astype(np.float64)
    order_right = np.argsort(np.argsort(right)).astype(np.float64)
    return float(np.corrcoef(order_left, order_right)[0, 1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovered-npz", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--norm", type=float, default=20.0)
    parser.add_argument("--candidates", type=int, default=360)
    parser.add_argument("--max-units", type=int, default=16)
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM

    from tools.llm_reader.blind_swiglu_oracle_decompiler import dequantized_weight

    recovered = np.load(args.recovered_npz)
    gates = recovered["gates_in_basis"] @ recovered["basis"]
    gates = gates / np.linalg.norm(gates, axis=1, keepdims=True) * recovered["gate_norms"][:, None]
    ups = recovered["up_directions"] / np.linalg.norm(recovered["up_directions"], axis=1, keepdims=True)
    units = recovered["units"].tolist()

    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True, device_map="cuda")
    mlp = model.model.layers[args.layer].mlp
    true_gate = dequantized_weight(mlp.gate_proj).detach().float().cpu().numpy()[units]
    true_up = dequantized_weight(mlp.up_proj).detach().float().cpu().numpy()[units]

    similarity = np.abs(
        (gates / np.linalg.norm(gates, axis=1, keepdims=True))
        @ (true_gate / np.linalg.norm(true_gate, axis=1, keepdims=True)).T
    )
    assignment, quality = similarity.argmax(axis=1), similarity.max(axis=1)

    rows = []
    for solution in np.argsort(-quality)[: args.max_units]:
        if quality[solution] < 0.99:
            continue
        source = int(assignment[solution])
        points = candidate_inputs(gates[solution], ups[solution], args.norm, args.candidates)
        predicted = drive(gates[solution], ups[solution], points)
        actual = drive(true_gate[source], true_up[source], points)

        best_predicted = int(np.argmax(np.abs(predicted)))
        captured = float(np.abs(actual[best_predicted]) / max(np.max(np.abs(actual)), 1e-30))
        rows.append({
            "unit": units[source],
            "rankCorrelation": rank_correlation(np.abs(predicted), np.abs(actual)),
            "actualAtPredictedOptimum": captured,
        })

    correlations = np.array([row["rankCorrelation"] for row in rows]) if rows else np.zeros(0)
    captured = np.array([row["actualAtPredictedOptimum"] for row in rows]) if rows else np.zeros(0)
    report = {
        "model": args.model, "layer": args.layer, "unitsTested": len(rows),
        "candidatesPerUnit": args.candidates, "inputNorm": args.norm,
        "rankCorrelationMedian": float(np.median(correlations)) if correlations.size else None,
        "rankCorrelationMin": float(np.min(correlations)) if correlations.size else None,
        "actualAtPredictedOptimumMedian": float(np.median(captured)) if captured.size else None,
        "actualAtPredictedOptimumMin": float(np.min(captured)) if captured.size else None,
        "perUnit": rows,
        "claimBoundary": [
            "Ranks inputs by how hard they drive a recovered unit, and checks the real unit ranks them the same.",
            "The search is over the residual stream, not over text: it says what drives the unit, not which words do.",
            "Says nothing about what the unit is for.",
        ],
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(f"{rendered}\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
