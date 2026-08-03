#!/usr/bin/env python3
"""Predict what a recovered unit will do on real text, before running the model.

This is the first of the three falsifiable claims neuron semantics has to make. Not a label
on a unit, a number: given `g_i`, `u_i` and the gate norm recovered by
`jacobian_boundary_recovery.py`, the activation `silu(g_i.x)(u_i.x)` is computable for any
input, so it can be predicted for a real prompt and checked against what the model actually
does.

Measured on LLaMA-3.2-3B layer 15, 32 units, 42 tokens of ordinary English: correlation
between predicted and actual activation has median 0.9837 and minimum 0.9375, with 94% of
units above 0.95. The gap from 1.0 is expected - the up direction is recovered as a direction,
so activations are right up to one scale factor per unit.

What this does and does not license. It licenses saying the unit computes a specific function
of the residual stream and that the function has been checked on text the recovery never saw.
It does not license saying what the unit is "for": 32 of 8192 units in 1 of 28 layers is
0.014% of the network, attention is untouched, and how these units compose across layers into
behaviour is not addressed here at all.
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


def predicted_activation(gates: np.ndarray, ups: np.ndarray, points: np.ndarray) -> np.ndarray:
    """`silu(g.x)(u.x)` for each unit and each input row."""
    z = points @ gates.T
    return (z / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))) * (points @ ups.T)


def correlation_per_unit(predicted: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Absolute correlation per unit, skipping any column that does not vary.

    Correlation rather than error, because the up direction is recovered without its norm:
    the prediction is right up to a scale, and a scale-sensitive metric would report that as
    a failure of the recovery rather than as the known missing constant it is.
    """
    scores = []
    for index in range(predicted.shape[1]):
        left, right = predicted[:, index], actual[:, index]
        if float(np.std(left)) < 1e-12 or float(np.std(right)) < 1e-12:
            continue
        scores.append(abs(float(np.corrcoef(left, right)[0, 1])))
    return np.asarray(scores)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovered-npz", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--text", action="append", required=True)
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from tools.llm_reader.blind_swiglu_oracle_decompiler import dequantized_weight, silu_np

    recovered = np.load(args.recovered_npz)
    gates = recovered["gates_in_basis"] @ recovered["basis"]
    gates = gates / np.linalg.norm(gates, axis=1, keepdims=True) * recovered["gate_norms"][:, None]
    ups = recovered["up_directions"]
    ups = ups / np.linalg.norm(ups, axis=1, keepdims=True)
    units = recovered["units"].tolist()

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True, device_map="cuda")
    model.eval()

    captured: dict[str, np.ndarray] = {}
    handle = model.model.layers[args.layer].mlp.register_forward_hook(
        lambda module, inputs, output: captured.__setitem__("x", inputs[0].detach().float().cpu().numpy())
    )
    rows = []
    with torch.no_grad():
        for text in args.text:
            model(**tokenizer(text, return_tensors="pt").to("cuda"))
            rows.append(captured["x"][0])
    handle.remove()
    points = np.concatenate(rows, axis=0)

    # Post-hoc witness: the true activation, read after the prediction is fixed.
    mlp = model.model.layers[args.layer].mlp
    true_gate = dequantized_weight(mlp.gate_proj).detach().float().cpu().numpy()[units]
    true_up = dequantized_weight(mlp.up_proj).detach().float().cpu().numpy()[units]
    actual = silu_np(points @ true_gate.T) * (points @ true_up.T)

    predicted = predicted_activation(gates, ups, points)
    normalised = gates / np.linalg.norm(gates, axis=1, keepdims=True)
    reference = true_gate / np.linalg.norm(true_gate, axis=1, keepdims=True)
    similarity = np.abs(normalised @ reference.T)
    assignment, quality = similarity.argmax(axis=1), similarity.max(axis=1)

    keep = quality >= 0.99
    scores = correlation_per_unit(predicted[:, keep], actual[:, assignment[keep]])
    report = {
        "model": args.model,
        "layer": args.layer,
        "unitCount": len(units),
        "unitsScored": int(scores.size),
        "tokens": int(points.shape[0]),
        "activationCorrelationMedian": float(np.median(scores)) if scores.size else None,
        "activationCorrelationMin": float(np.min(scores)) if scores.size else None,
        "shareAbove095": float(np.mean(scores > 0.95)) if scores.size else None,
        "shareAbove099": float(np.mean(scores > 0.99)) if scores.size else None,
        "claimBoundary": [
            "Predicts the activation of recovered units on text the recovery never saw.",
            "Up directions are recovered without their norm, so predictions hold up to one scale per unit.",
            "Says nothing about what a unit is for, about attention, or about composition across layers.",
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
