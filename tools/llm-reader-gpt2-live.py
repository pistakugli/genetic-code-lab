#!/usr/bin/env python3

import argparse
import json
import math
import os
import sys
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Dict, List

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as transformers_logging

transformers_logging.disable_progress_bar()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read a live GPT-2/LLM forward pass and explain real hooked MLP neuron reactions."
    )
    parser.add_argument("--model", default="gpt2", help="HuggingFace causal LM id or local model path.")
    parser.add_argument("--prompt", required=True, help="Prompt to run through the live model.")
    parser.add_argument("--layer", type=int, default=0, help="Transformer block index to inspect.")
    parser.add_argument("--token-index", type=int, default=-1, help="Prompt token index to explain; negative indexes are supported.")
    parser.add_argument("--top-neurons", type=int, default=12, help="Number of top MLP source neurons to report.")
    parser.add_argument("--top-logits", type=int, default=8, help="Number of next-token logits to report.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"], help="Runtime device.")
    parser.add_argument("--local-files-only", action="store_true", help="Do not download model files; use local cache/path only.")
    parser.add_argument("--output", help="Optional JSON report output path.")
    parser.add_argument("--format", choices=["text", "json"], default="text", help="Stdout format.")
    return parser.parse_args()


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def gelu(value: torch.Tensor) -> torch.Tensor:
    return 0.5 * value * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (value + 0.044715 * value.pow(3))))


def to_float(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


def clean_token(tokenizer: Any, token_id: int) -> str:
    text = tokenizer.decode([int(token_id)])
    return text.replace("\n", "\\n")


def top_values(values: torch.Tensor, count: int) -> List[Dict[str, Any]]:
    limit = min(max(0, count), int(values.numel()))
    if limit == 0:
        return []
    top = torch.topk(values.detach().cpu(), k=limit)
    return [
        {"index": int(index), "value": float(value)}
        for value, index in zip(top.values, top.indices)
    ]


def ensure_gpt2_mlp_shape(model: Any, layer: int) -> Any:
    blocks = getattr(getattr(model, "transformer", None), "h", None)
    if blocks is None:
        raise RuntimeError("Model does not expose GPT-2-style transformer.h blocks.")
    if layer < 0 or layer >= len(blocks):
        raise RuntimeError(f"Layer {layer} is outside available GPT-2 block range 0..{len(blocks) - 1}.")
    mlp = blocks[layer].mlp
    for name in ("c_fc", "act", "c_proj"):
        if not hasattr(mlp, name):
            raise RuntimeError(f"Layer {layer} MLP does not expose GPT-2-style {name}.")
    return mlp


def normalize_token_index(index: int, token_count: int) -> int:
    resolved = token_count + index if index < 0 else index
    if resolved < 0 or resolved >= token_count:
        raise RuntimeError(f"token-index {index} is outside prompt token range 0..{token_count - 1}.")
    return resolved


def build_neuron_report(
    tokenizer: Any,
    token_ids: torch.Tensor,
    mlp_input: torch.Tensor,
    preactivation: torch.Tensor,
    activation: torch.Tensor,
    c_fc_weight: torch.Tensor,
    c_fc_bias: torch.Tensor,
    c_proj_weight: torch.Tensor,
    mlp_output: torch.Tensor,
    token_index: int,
    top_neurons: int
) -> List[Dict[str, Any]]:
    seq_len = int(token_ids.numel())
    selected_input = mlp_input[token_index]
    selected_activation = activation[token_index]
    selected_preactivation = preactivation[token_index]
    selected_output = mlp_output[token_index]
    projection_norms = torch.linalg.vector_norm(c_proj_weight, dim=1)
    selected_contribution_norms = selected_activation.abs() * projection_norms
    output_norm = torch.linalg.vector_norm(selected_output) + 1e-12
    top = torch.topk(selected_contribution_norms.detach().cpu(), k=min(top_neurons, selected_contribution_norms.numel()))
    reports: List[Dict[str, Any]] = []
    for score_cpu, neuron_cpu in zip(top.values, top.indices):
        neuron = int(neuron_cpu)
        input_weights = c_fc_weight[:, neuron]
        raw_drivers = selected_input * input_weights
        driver_scores = raw_drivers.abs().detach().cpu()
        driver_top = torch.topk(driver_scores, k=min(8, driver_scores.numel()))
        contribution_vector = selected_activation[neuron] * c_proj_weight[neuron]
        signed_output_projection = torch.dot(contribution_vector, selected_output) / output_norm
        token_scores = activation[:, neuron].abs() * projection_norms[neuron]
        token_top = torch.topk(token_scores.detach().cpu(), k=min(5, seq_len))
        reports.append({
            "neuron": neuron,
            "activation": to_float(selected_activation[neuron]),
            "preactivation": to_float(selected_preactivation[neuron]),
            "bias": to_float(c_fc_bias[neuron]),
            "projectionNorm": to_float(projection_norms[neuron]),
            "contributionNorm": float(score_cpu),
            "signedOutputProjection": to_float(signed_output_projection),
            "reactsMostOnPromptTokens": [
                {
                    "position": int(position),
                    "token": clean_token(tokenizer, int(token_ids[position])),
                    "activation": to_float(activation[int(position), neuron]),
                    "contributionNorm": to_float(token_scores[int(position)])
                }
                for position in token_top.indices
            ],
            "topResidualDrivers": [
                {
                    "residualDimension": int(index),
                    "signedContributionToPreactivation": to_float(raw_drivers[int(index)]),
                    "inputValue": to_float(selected_input[int(index)]),
                    "weight": to_float(input_weights[int(index)])
                }
                for index in driver_top.indices
            ],
            "why": (
                "This MLP source neuron reacts when the hooked residual stream input pushes "
                "its c_fc preactivation through GELU. The listed residual dimensions are the "
                "largest signed drivers for this prompt token; its c_proj row turns the GELU "
                "activation into the reported residual-stream contribution."
            )
        })
    return reports


def make_text_report(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("GCL live LLM reader")
    lines.append(f"model: {report['model']['id']}")
    lines.append(f"prompt: {report['prompt']}")
    lines.append(f"layer: {report['selection']['layer']}")
    lines.append(
        f"token: #{report['selection']['tokenIndex']} {json.dumps(report['selection']['token'])}"
    )
    lines.append("")
    lines.append("next-token logits:")
    for item in report["nextToken"]["topLogits"]:
        lines.append(f"  {item['rank']:>2}. {json.dumps(item['token'])} logit={item['logit']:.6f}")
    lines.append("")
    lines.append("top hooked MLP source neurons:")
    for rank, item in enumerate(report["neurons"], start=1):
        lines.append(
            f"  {rank:>2}. neuron {item['neuron']} "
            f"pre={item['preactivation']:.6f} act={item['activation']:.6f} "
            f"contrib_norm={item['contributionNorm']:.6f} "
            f"signed_out_proj={item['signedOutputProjection']:.6f}"
        )
        token_bits = ", ".join(
            f"#{tok['position']} {json.dumps(tok['token'])}:act={tok['activation']:.4f}"
            for tok in item["reactsMostOnPromptTokens"][:3]
        )
        lines.append(f"      reacts most on: {token_bits}")
        driver_bits = ", ".join(
            f"d{drv['residualDimension']}={drv['signedContributionToPreactivation']:.4f}"
            for drv in item["topResidualDrivers"][:4]
        )
        lines.append(f"      strongest residual drivers: {driver_bits}")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.top_neurons <= 0:
        raise RuntimeError("--top-neurons must be positive.")
    if args.top_logits <= 0:
        raise RuntimeError("--top-logits must be positive.")

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=args.local_files_only)
    model.eval()
    model.to(device)
    mlp = ensure_gpt2_mlp_shape(model, args.layer)

    encoded = tokenizer(args.prompt, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    token_ids = encoded["input_ids"][0].detach().cpu()
    token_index = normalize_token_index(args.token_index, int(token_ids.numel()))

    captures: Dict[str, torch.Tensor] = {}

    def capture_c_fc(module: Any, inputs: Any, output: torch.Tensor) -> None:
        captures["mlp_input"] = inputs[0].detach().cpu()[0]
        captures["preactivation"] = output.detach().cpu()[0]

    def capture_act(module: Any, inputs: Any, output: torch.Tensor) -> None:
        captures["activation"] = output.detach().cpu()[0]

    def capture_c_proj(module: Any, inputs: Any, output: torch.Tensor) -> None:
        captures["mlp_output"] = output.detach().cpu()[0]

    handles = [
        mlp.c_fc.register_forward_hook(capture_c_fc),
        mlp.act.register_forward_hook(capture_act),
        mlp.c_proj.register_forward_hook(capture_c_proj)
    ]
    with torch.no_grad():
        outputs = model(**encoded)
    for handle in handles:
        handle.remove()

    required = {"mlp_input", "preactivation", "activation", "mlp_output"}
    missing = sorted(required - set(captures))
    if missing:
        raise RuntimeError(f"Missing hook captures: {missing}")

    logits = outputs.logits.detach().cpu()[0, token_index]
    top_logits = torch.topk(logits, k=min(args.top_logits, logits.numel()))
    c_fc_weight = mlp.c_fc.weight.detach().cpu()
    c_fc_bias = mlp.c_fc.bias.detach().cpu()
    c_proj_weight = mlp.c_proj.weight.detach().cpu()
    neurons = build_neuron_report(
        tokenizer,
        token_ids,
        captures["mlp_input"],
        captures["preactivation"],
        captures["activation"],
        c_fc_weight,
        c_fc_bias,
        c_proj_weight,
        captures["mlp_output"],
        token_index,
        args.top_neurons
    )

    report = {
        "schemaVersion": 1,
        "mode": "live-real-model-hook-reader",
        "synthetic": False,
        "surrogate": False,
        "model": {
            "id": args.model,
            "class": model.__class__.__name__,
            "device": device,
            "localFilesOnly": bool(args.local_files_only)
        },
        "runtimeDependencies": {
            "python": sys.executable,
            "torch": torch.__version__,
            "transformers": package_version("transformers"),
            "numpy": package_version("numpy"),
            "torchao": package_version("torchao")
        },
        "prompt": args.prompt,
        "tokens": [
            {"position": index, "id": int(token_id), "text": clean_token(tokenizer, int(token_id))}
            for index, token_id in enumerate(token_ids)
        ],
        "selection": {
            "layer": args.layer,
            "tokenIndex": token_index,
            "tokenId": int(token_ids[token_index]),
            "token": clean_token(tokenizer, int(token_ids[token_index])),
            "component": f"transformer.h.{args.layer}.mlp"
        },
        "hookShapes": {key: list(value.shape) for key, value in captures.items()},
        "nextToken": {
            "topLogits": [
                {
                    "rank": rank + 1,
                    "tokenId": int(index),
                    "token": clean_token(tokenizer, int(index)),
                    "logit": float(value)
                }
                for rank, (value, index) in enumerate(zip(top_logits.values, top_logits.indices))
            ]
        },
        "neurons": neurons,
        "claim": {
            "does": "Loads the real model, runs a real forward pass, hooks real MLP activations, and reports source neuron activations/contributions for the selected prompt token.",
            "doesNot": "Does not use synthetic teachers, does not fit a surrogate, and does not claim full black-box extraction of every hidden mechanism."
        }
    }

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=2)
            file.write("\n")

    if args.format == "json":
        print(json.dumps(report, indent=2))
    else:
        print(make_text_report(report))


if __name__ == "__main__":
    main()
