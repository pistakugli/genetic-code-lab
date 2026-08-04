#!/usr/bin/env python3
"""Predict what silencing a unit does to the model's output, then silence it and check.

The second of the three falsifiable claims. Prediction says the recovered parameters describe
the unit's function; this says they describe its *effect*. A recovered unit contributes
`d_i * silu(g_i.x)(u_i.x)` to the residual stream, so zeroing it should move the layer output
by exactly minus that, and the prediction can be compared against a real forward pass with a
hook that zeroes the same unit.

This is a harder claim than prediction and it can fail on its own: the up direction is
recovered without its norm, so the predicted delta is right in direction but not in magnitude
until that scale is fitted from the data. Direction and magnitude are therefore reported
separately - cosine says whether the mechanism is understood, relative error says whether it
is quantified.

What it does not license: any statement about what the unit is for. A unit whose removal is
predicted perfectly may still have no describable role.
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


def predicted_delta(
    gate: np.ndarray,
    up: np.ndarray,
    output_direction: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    """Change in layer output from removing one unit: `-d_i * silu(g_i.x)(u_i.x)`."""
    z = points @ gate
    activation = (z / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))) * (points @ up)
    return -np.outer(activation, output_direction)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovered-npz", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--text", action="append", required=True)
    parser.add_argument("--max-units", type=int, default=8)
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from tools.llm_reader.blind_swiglu_oracle_decompiler import dequantized_weight

    recovered = np.load(args.recovered_npz)
    gates = recovered["gates_in_basis"] @ recovered["basis"]
    gates = gates / np.linalg.norm(gates, axis=1, keepdims=True) * recovered["gate_norms"][:, None]
    ups = recovered["up_directions"] / np.linalg.norm(recovered["up_directions"], axis=1, keepdims=True)
    outputs = recovered["output_directions"]
    units = recovered["units"].tolist()

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True, device_map="cuda")
    model.eval()
    mlp = model.model.layers[args.layer].mlp

    # Match recovered solutions to source units so the same unit is silenced in both worlds.
    true_gate = dequantized_weight(mlp.gate_proj).detach().float().cpu().numpy()[units]
    similarity = np.abs(
        (gates / np.linalg.norm(gates, axis=1, keepdims=True))
        @ (true_gate / np.linalg.norm(true_gate, axis=1, keepdims=True)).T
    )
    assignment, quality = similarity.argmax(axis=1), similarity.max(axis=1)

    captured: dict[str, np.ndarray] = {}
    handle = mlp.register_forward_hook(
        lambda module, inputs, output: captured.__setitem__("x", inputs[0].detach().float().cpu().numpy())
    )
    input_rows = []
    with torch.no_grad():
        for text in args.text:
            encoded = tokenizer(text, return_tensors="pt").to("cuda")
            model(**encoded)
            input_rows.append(captured["x"][0])
    handle.remove()
    points = np.concatenate(input_rows, axis=0)

    order = np.argsort(-quality)[: args.max_units]
    rows = []
    for solution in order:
        if quality[solution] < 0.99:
            continue
        unit = units[assignment[solution]]

        # What silencing the unit changes, taken in one pass rather than as the difference of
        # two. `down_proj` is linear and carries no bias, so
        # `down_proj(p) - down_proj(p with i zeroed) = down_proj(e_i p_i)` exactly, and that
        # single product is the whole effect of the unit.
        #
        # Differencing two forward passes measured the same quantity through near-total
        # cancellation: one unit moves the layer output by about 0.16 while the outputs
        # themselves are far larger, so what survived was the 4-bit kernel's disagreement with
        # itself. Checked against the model's own weights, that route had a ceiling of 0.95 to
        # 0.99 rather than 1.0 - it could not tell a good recovery from a mediocre one,
        # because its noise was the same size as the difference it was asked to resolve.
        def isolate(module, inputs, output, index=unit):
            product = module.act_fn(module.gate_proj(inputs[0])) * module.up_proj(inputs[0])
            only = torch.zeros_like(product)
            only[..., index] = product[..., index]
            return module.down_proj(only)

        hook = mlp.register_forward_hook(isolate)
        with torch.no_grad():
            contribution = mlp(torch.tensor(points[None, :, :], device="cuda")).detach().float().cpu().numpy()[0]
        hook.remove()
        actual = -contribution

        prediction = predicted_delta(gates[solution], ups[solution], outputs[solution], points)
        # One scale per unit, because the up direction has no norm. Fitted on the same data,
        # so the magnitude number is weaker evidence than the direction number and is
        # reported as such.
        denominator = float(np.sum(prediction * prediction))
        scale = float(np.sum(prediction * actual) / denominator) if denominator > 0 else 0.0
        scaled = scale * prediction
        cosine = float(
            np.sum(prediction * actual)
            / max(np.linalg.norm(prediction) * np.linalg.norm(actual), 1e-30)
        )
        relative = float(np.linalg.norm(scaled - actual) / max(np.linalg.norm(actual), 1e-30))
        rows.append({"unit": int(unit), "gateCosine": float(quality[solution]),
                     "deltaCosine": abs(cosine), "relativeErrorAfterScale": relative})

    cosines = np.array([row["deltaCosine"] for row in rows]) if rows else np.zeros(0)
    errors = np.array([row["relativeErrorAfterScale"] for row in rows]) if rows else np.zeros(0)
    report = {
        "model": args.model, "layer": args.layer, "unitsTested": len(rows),
        "tokens": int(points.shape[0]),
        "deltaCosineMedian": float(np.median(cosines)) if cosines.size else None,
        "deltaCosineMin": float(np.min(cosines)) if cosines.size else None,
        "relativeErrorMedian": float(np.median(errors)) if errors.size else None,
        "perUnit": rows,
        "claimBoundary": [
            "Predicts the change in layer output from silencing a recovered unit, checked against a real hook.",
            "Magnitude uses one scale per unit fitted on the same data, because the up direction has no norm; the cosine does not.",
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
