#!/usr/bin/env python3

import argparse
import json
import math
import os
import random
import sys
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Dict, List

import torch
import torch.nn.functional as F

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as transformers_logging

from llm_reader.adapters.registry import create_adapter_for_model_type

transformers_logging.disable_progress_bar()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explain hooked GPT-2 MLP neurons or instrumented SwiGLU gate/value channels."
    )
    parser.add_argument("--model", default="gpt2", help="HuggingFace causal LM id or local model path.")
    parser.add_argument("--prompt", required=True, help="Prompt to run through the live model.")
    parser.add_argument("--layer", type=int, default=0, help="Transformer block index to inspect.")
    parser.add_argument(
        "--layers",
        default="",
        help="For SwiGLU models, comma-separated layer indexes or 'all'; defaults to --layer.",
    )
    parser.add_argument("--token-index", type=int, default=-1, help="Prompt token index to explain; negative indexes are supported.")
    parser.add_argument("--top-neurons", type=int, default=12, help="Number of top MLP neurons or SwiGLU channels to report.")
    parser.add_argument("--top-logits", type=int, default=8, help="Number of next-token logits to report.")
    parser.add_argument("--ablation-mode", choices=["selected-token", "all-tokens", "none"], default="selected-token", help="Run real causal neuron ablations for reported neurons.")
    parser.add_argument("--negative-controls", action=argparse.BooleanOptionalAction, default=True, help="Run negative-control ablations.")
    parser.add_argument(
        "--control-count",
        type=int,
        default=50,
        help="GPT-2 control count per family; SwiGLU uses one active and one inactive control per selected layer.",
    )
    parser.add_argument("--control-prompt", default="The weather in Berlin is", help="Different real prompt used to select off-prompt top neurons.")
    parser.add_argument("--control-seed", type=int, default=1337, help="Deterministic seed for random negative-control neuron selection.")
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


def top_logit_items(tokenizer: Any, logits: torch.Tensor, count: int) -> List[Dict[str, Any]]:
    top = torch.topk(logits.detach().cpu(), k=min(count, logits.numel()))
    return [
        {
            "rank": rank + 1,
            "tokenId": int(index),
            "token": clean_token(tokenizer, int(index)),
            "logit": float(value)
        }
        for rank, (value, index) in enumerate(zip(top.values, top.indices))
    ]


def run_with_mlp_captures(model: Any, mlp: Any, encoded: Dict[str, torch.Tensor]) -> tuple[Any, Dict[str, torch.Tensor]]:
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
    try:
        with torch.no_grad():
            outputs = model(**encoded)
    finally:
        for handle in handles:
            handle.remove()

    required = {"mlp_input", "preactivation", "activation", "mlp_output"}
    missing = sorted(required - set(captures))
    if missing:
        raise RuntimeError(f"Missing hook captures: {missing}")
    return outputs, captures


def run_activation_ablation_logits(
    model: Any,
    mlp: Any,
    encoded: Dict[str, torch.Tensor],
    neuron: int,
    ablate_positions: List[int],
    read_token_index: int
) -> torch.Tensor:
    seq_len = int(encoded["input_ids"].shape[1])
    valid_positions = sorted(set(position for position in ablate_positions if 0 <= position < seq_len))
    if not valid_positions:
        raise RuntimeError("Ablation requires at least one valid token position.")

    def zero_neuron_activation(module: Any, inputs: Any, output: torch.Tensor) -> torch.Tensor:
        patched = output.clone()
        for position in valid_positions:
            patched[:, position, neuron] = 0
        return patched

    handle = mlp.act.register_forward_hook(zero_neuron_activation)
    try:
        with torch.no_grad():
            return model(**encoded).logits.detach().cpu()[0, read_token_index]
    finally:
        handle.remove()


def logit_effects(
    tokenizer: Any,
    baseline_logits: torch.Tensor,
    ablated_logits: torch.Tensor,
    candidate_token_ids: List[int]
) -> List[Dict[str, Any]]:
    baseline_cpu = baseline_logits.detach().cpu()
    effects = []
    for token_id in candidate_token_ids:
        baseline = float(baseline_cpu[token_id])
        ablated = float(ablated_logits[token_id])
        effects.append({
            "tokenId": int(token_id),
            "token": clean_token(tokenizer, int(token_id)),
            "baselineLogit": baseline,
            "ablatedLogit": ablated,
            "logitDropWhenRemoved": baseline - ablated
        })
    return sorted(effects, key=lambda effect: abs(effect["logitDropWhenRemoved"]), reverse=True)


def max_abs_effect(effects: List[Dict[str, Any]]) -> float:
    if not effects:
        return 0.0
    return max(abs(float(effect["logitDropWhenRemoved"])) for effect in effects)


def summarize_control_items(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    values = sorted(float(item.get("maxAbsLogitDrop", 0.0)) for item in items)
    def percentile(q: float) -> float:
        if not values:
            return 0.0
        index = min(len(values) - 1, max(0, int(round((len(values) - 1) * q))))
        return values[index]
    return {
        "count": len(items),
        "maxAbsLogitDrop": max(values) if values else 0.0,
        "meanAbsLogitDrop": sum(values) / len(values) if values else 0.0,
        "p50AbsLogitDrop": percentile(0.50),
        "p90AbsLogitDrop": percentile(0.90),
        "p95AbsLogitDrop": percentile(0.95)
    }


def attach_causal_ablation_reports(
    model: Any,
    mlp: Any,
    encoded: Dict[str, torch.Tensor],
    tokenizer: Any,
    baseline_logits: torch.Tensor,
    layer: int,
    token_index: int,
    neurons: List[Dict[str, Any]],
    candidate_token_ids: List[int],
    mode: str
) -> None:
    if mode == "none":
        return

    for item in neurons:
        neuron = int(item["neuron"])
        ablate_positions = [token_index] if mode == "selected-token" else list(range(int(encoded["input_ids"].shape[1])))
        ablated_logits = run_activation_ablation_logits(model, mlp, encoded, neuron, ablate_positions, token_index)
        effects = logit_effects(tokenizer, baseline_logits, ablated_logits, candidate_token_ids)

        item["causalAblation"] = {
            "mode": mode,
            "intervention": f"zero transformer.h.{layer}.mlp.act[..., {neuron}]",
            "tokenIndex": token_index,
            "topLogitsAfterRemoval": top_logit_items(tokenizer, ablated_logits, min(8, len(candidate_token_ids))),
            "candidateLogitEffects": effects,
            "maxAbsLogitDrop": max_abs_effect(effects)
        }


def make_control_item(
    *,
    model: Any,
    mlp: Any,
    encoded: Dict[str, torch.Tensor],
    tokenizer: Any,
    baseline_logits: torch.Tensor,
    layer: int,
    read_token_index: int,
    neuron: int,
    ablate_positions: List[int],
    candidate_token_ids: List[int],
    metadata: Dict[str, Any]
) -> Dict[str, Any]:
    ablated_logits = run_activation_ablation_logits(model, mlp, encoded, neuron, ablate_positions, read_token_index)
    effects = logit_effects(tokenizer, baseline_logits, ablated_logits, candidate_token_ids)
    return {
        **metadata,
        "neuron": int(neuron),
        "intervention": f"zero transformer.h.{layer}.mlp.act[..., {int(neuron)}] at token positions {ablate_positions}",
        "readTokenIndex": read_token_index,
        "ablateTokenPositions": ablate_positions,
        "topLogitsAfterRemoval": top_logit_items(tokenizer, ablated_logits, min(8, len(candidate_token_ids))),
        "candidateLogitEffects": effects,
        "maxAbsLogitDrop": max_abs_effect(effects)
    }


def pick_amplitude_matched_controls(
    activation: torch.Tensor,
    target_neurons: List[int],
    excluded: set[int],
    count: int,
    rng: random.Random
) -> List[Dict[str, Any]]:
    abs_activation = activation.abs()
    choices: List[Dict[str, Any]] = []
    used = set(excluded)
    neuron_count = int(abs_activation.numel())
    if not target_neurons:
        return choices
    target_index = 0
    while len(choices) < count:
        target = target_neurons[target_index % len(target_neurons)]
        target_index += 1
        target_abs = float(abs_activation[target])
        candidates = [
            index for index in range(neuron_count)
            if index not in used and float(abs_activation[index]) > 0.05
        ]
        if not candidates:
            break
        nearest = sorted(candidates, key=lambda index: abs(float(abs_activation[index]) - target_abs))[:64]
        chosen = rng.choice(nearest)
        used.add(chosen)
        choices.append({
            "neuron": chosen,
            "matchedToNeuron": int(target),
            "matchedToActivationAbs": target_abs,
            "activationAbs": float(abs_activation[chosen]),
            "activation": float(activation[chosen])
        })
    return choices


def pick_inactive_controls(
    activation: torch.Tensor,
    excluded: set[int],
    count: int,
    rng: random.Random
) -> tuple[List[Dict[str, Any]], float]:
    abs_activation = activation.abs()
    threshold = max(1e-6, float(torch.quantile(abs_activation, 0.05)))
    candidates = [
        index for index in range(int(abs_activation.numel()))
        if index not in excluded and float(abs_activation[index]) <= threshold
    ]
    rng.shuffle(candidates)
    choices = [
        {
            "neuron": int(index),
            "activationAbs": float(abs_activation[index]),
            "activation": float(activation[index])
        }
        for index in candidates[:count]
    ]
    return choices, threshold


def build_negative_control_report(
    *,
    model: Any,
    mlp: Any,
    tokenizer: Any,
    encoded: Dict[str, torch.Tensor],
    baseline_logits: torch.Tensor,
    activation: torch.Tensor,
    token_ids: torch.Tensor,
    layer: int,
    token_index: int,
    neurons: List[Dict[str, Any]],
    candidate_token_ids: List[int],
    c_proj_weight: torch.Tensor,
    args: argparse.Namespace
) -> Dict[str, Any]:
    rng = random.Random(args.control_seed)
    control_count = max(1, int(args.control_count))
    baseline_activation = activation[token_index]
    primary_neurons = [int(item["neuron"]) for item in neurons]
    primary_set = set(primary_neurons)
    used_control_neurons = set(primary_neurons)
    groups: List[Dict[str, Any]] = []

    matched_items = []
    matched_choices = pick_amplitude_matched_controls(baseline_activation, primary_neurons, used_control_neurons, control_count, rng)
    used_control_neurons.update(int(choice["neuron"]) for choice in matched_choices)
    for choice in matched_choices:
        matched_items.append(make_control_item(
            model=model,
            mlp=mlp,
            encoded=encoded,
            tokenizer=tokenizer,
            baseline_logits=baseline_logits,
            layer=layer,
            read_token_index=token_index,
            neuron=int(choice["neuron"]),
            ablate_positions=[token_index],
            candidate_token_ids=candidate_token_ids,
            metadata={
                "kind": "random-active-same-amplitude",
                "matchedToNeuron": choice["matchedToNeuron"],
                "matchedToActivationAbs": choice["matchedToActivationAbs"],
                "activationAbs": choice["activationAbs"],
                "activation": choice["activation"]
            }
        ))
    groups.append({
        "kind": "random-active-same-amplitude",
        "requestedControl": "random active neurons with same activation amplitude",
        "summary": summarize_control_items(matched_items),
        "items": matched_items
    })

    inactive_choices, inactive_threshold = pick_inactive_controls(baseline_activation, used_control_neurons, control_count, rng)
    used_control_neurons.update(int(choice["neuron"]) for choice in inactive_choices)
    inactive_items = []
    for choice in inactive_choices:
        inactive_items.append(make_control_item(
            model=model,
            mlp=mlp,
            encoded=encoded,
            tokenizer=tokenizer,
            baseline_logits=baseline_logits,
            layer=layer,
            read_token_index=token_index,
            neuron=int(choice["neuron"]),
            ablate_positions=[token_index],
            candidate_token_ids=candidate_token_ids,
            metadata={
                "kind": "random-inactive",
                "inactiveThresholdAbs": inactive_threshold,
                "activationAbs": choice["activationAbs"],
                "activation": choice["activation"]
            }
        ))
    groups.append({
        "kind": "random-inactive",
        "requestedControl": "random inactive neurons",
        "summary": summarize_control_items(inactive_items),
        "inactiveThresholdAbs": inactive_threshold,
        "items": inactive_items
    })

    control_encoded = tokenizer(args.control_prompt, return_tensors="pt")
    device = encoded["input_ids"].device
    control_encoded = {key: value.to(device) for key, value in control_encoded.items()}
    control_outputs, control_captures = run_with_mlp_captures(model, mlp, control_encoded)
    del control_outputs
    control_token_ids = control_encoded["input_ids"][0].detach().cpu()
    control_token_index = int(control_token_ids.numel()) - 1
    control_activation = control_captures["activation"][control_token_index]
    projection_norms = torch.linalg.vector_norm(c_proj_weight, dim=1)
    control_scores = control_activation.abs() * projection_norms
    off_prompt_items = []
    skipped_primary_overlap = 0
    skipped_prior_control_overlap = 0
    for neuron_tensor in torch.argsort(control_scores.detach().cpu(), descending=True):
        neuron = int(neuron_tensor)
        if neuron in primary_set:
            skipped_primary_overlap += 1
            continue
        if neuron in used_control_neurons:
            skipped_prior_control_overlap += 1
            continue
        if len(off_prompt_items) >= control_count:
            break
        score = float(control_scores[neuron])
        off_prompt_items.append(make_control_item(
            model=model,
            mlp=mlp,
            encoded=encoded,
            tokenizer=tokenizer,
            baseline_logits=baseline_logits,
            layer=layer,
            read_token_index=token_index,
            neuron=neuron,
            ablate_positions=[token_index],
            candidate_token_ids=candidate_token_ids,
            metadata={
                "kind": "top-neuron-from-other-prompt",
                "controlPrompt": args.control_prompt,
                "controlTokenIndex": control_token_index,
                "controlToken": clean_token(tokenizer, int(control_token_ids[control_token_index])),
                "controlPromptContributionScore": score,
                "controlPromptActivation": float(control_activation[neuron]),
                "selectedPromptActivation": float(baseline_activation[neuron]),
                "overlapsPrimaryTopNeurons": False,
                "overlapsPriorControlNeurons": False
            }
        ))
    used_control_neurons.update(int(item["neuron"]) for item in off_prompt_items)
    groups.append({
        "kind": "top-neuron-from-other-prompt",
        "requestedControl": "top neurons from a different prompt, excluding primary and prior control-family overlap",
        "summary": summarize_control_items(off_prompt_items),
        "skippedPrimaryOverlap": skipped_primary_overlap,
        "skippedPriorControlOverlap": skipped_prior_control_overlap,
        "items": off_prompt_items
    })

    other_position_items = []
    other_positions = [position for position in range(int(token_ids.numel())) if position != token_index]
    for neuron in primary_neurons:
        for position in other_positions:
            other_position_items.append(make_control_item(
                model=model,
                mlp=mlp,
                encoded=encoded,
                tokenizer=tokenizer,
                baseline_logits=baseline_logits,
                layer=layer,
                read_token_index=token_index,
                neuron=neuron,
                ablate_positions=[position],
                candidate_token_ids=candidate_token_ids,
                metadata={
                    "kind": "same-neuron-other-token-position",
                    "sourceNeuron": neuron,
                    "otherTokenIndex": position,
                    "otherToken": clean_token(tokenizer, int(token_ids[position])),
                    "activationAtOtherToken": float(activation[position, neuron])
                }
            ))
    groups.append({
        "kind": "same-neuron-other-token-position",
        "requestedControl": "same top neurons ablated at other token positions",
        "summary": summarize_control_items(other_position_items),
        "sourceNeuronCount": len(primary_neurons),
        "otherTokenPositionCount": len(other_positions),
        "items": other_position_items
    })

    requested_control_coverage = {
        "randomActiveNeuronsSameAmplitude": {
            "present": len(matched_items) > 0,
            "kind": "random-active-same-amplitude",
            "itemCount": len(matched_items)
        },
        "randomInactiveNeurons": {
            "present": len(inactive_items) > 0,
            "kind": "random-inactive",
            "itemCount": len(inactive_items)
        },
        "topNeuronsFromDifferentPrompt": {
            "present": len(off_prompt_items) > 0,
            "kind": "top-neuron-from-other-prompt",
            "itemCount": len(off_prompt_items),
            "skippedPrimaryOverlap": skipped_primary_overlap,
            "skippedPriorControlOverlap": skipped_prior_control_overlap
        },
        "sameNeuronsOnOtherTokenPositions": {
            "present": len(other_position_items) > 0,
            "kind": "same-neuron-other-token-position",
            "itemCount": len(other_position_items),
            "sourceNeuronCount": len(primary_neurons),
            "otherTokenPositionCount": len(other_positions)
        }
    }
    cross_family_overlap = []
    for left_index, left_group in enumerate(groups):
        left_neurons = set(int(item["neuron"]) for item in left_group["items"])
        for right_group in groups[left_index + 1:]:
            right_neurons = set(int(item["neuron"]) for item in right_group["items"])
            overlap = sorted(left_neurons & right_neurons)
            cross_family_overlap.append({
                "left": left_group["kind"],
                "right": right_group["kind"],
                "count": len(overlap),
                "neurons": overlap[:25]
            })

    return {
        "enabled": True,
        "seed": int(args.control_seed),
        "controlCount": control_count,
        "controlPrompt": args.control_prompt,
        "requestedControlCoverage": requested_control_coverage,
        "crossFamilyNeuronOverlap": cross_family_overlap,
        "candidateTokens": [
            {"tokenId": int(token_id), "token": clean_token(tokenizer, int(token_id))}
            for token_id in candidate_token_ids
        ],
        "groups": groups
    }


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
        ablation = item.get("causalAblation")
        if ablation:
            effect_bits = ", ".join(
                f"{json.dumps(effect['token'])}:drop={effect['logitDropWhenRemoved']:.4f}"
                for effect in ablation["candidateLogitEffects"][:4]
            )
            lines.append(f"      causal ablation: {effect_bits}")
    controls = report.get("negativeControls")
    if controls and controls.get("enabled"):
        lines.append("")
        lines.append("negative-control ablations:")
        coverage = controls.get("requestedControlCoverage", {})
        for name, item in coverage.items():
            lines.append(
                f"  requested {name}: present={str(item.get('present')).lower()} "
                f"items={item.get('itemCount', 0)}"
            )
        for group in controls["groups"]:
            summary = group["summary"]
            lines.append(
                f"  {group['kind']}: count={summary['count']} "
                f"max_abs_drop={summary['maxAbsLogitDrop']:.6f} "
                f"mean_abs_drop={summary['meanAbsLogitDrop']:.6f} "
                f"p95_abs_drop={summary['p95AbsLogitDrop']:.6f}"
            )
    return "\n".join(lines)


def parse_layer_selection(value: str, layer_count: int, fallback_layer: int) -> List[int]:
    if not value.strip():
        layers = [fallback_layer]
    elif value.strip().lower() == "all":
        layers = list(range(layer_count))
    else:
        try:
            layers = [int(item.strip()) for item in value.split(",") if item.strip()]
        except ValueError as error:
            raise RuntimeError("--layers must be 'all' or a comma-separated list of integers.") from error
        if not layers:
            raise RuntimeError("--layers must include at least one layer.")
    if len(set(layers)) != len(layers):
        raise RuntimeError("--layers must not contain duplicate layer indexes.")
    invalid = [layer for layer in layers if layer < 0 or layer >= layer_count]
    if invalid:
        raise RuntimeError(
            f"Layer indexes {invalid} are outside model block range 0..{layer_count - 1}."
        )
    return layers


def _capture_swiglu_layers(
    model: Any,
    mlps: Dict[int, Any],
    encoded: Dict[str, torch.Tensor],
) -> tuple[Any, Dict[int, Dict[str, torch.Tensor]]]:
    captures: Dict[int, Dict[str, torch.Tensor]] = {layer: {} for layer in mlps}
    handles = []
    for layer, mlp in mlps.items():
        module = mlp._mlp

        def capture_input(module: Any, inputs: tuple[Any, ...], layer_index: int = layer) -> None:
            captures[layer_index]["mlpInput"] = inputs[0].detach().cpu()[0].float()

        def capture_gate(module: Any, inputs: tuple[Any, ...], output: torch.Tensor, layer_index: int = layer) -> None:
            captures[layer_index]["gatePreactivation"] = output.detach().cpu()[0].float()

        def capture_up(module: Any, inputs: tuple[Any, ...], output: torch.Tensor, layer_index: int = layer) -> None:
            captures[layer_index]["upProjection"] = output.detach().cpu()[0].float()

        def capture_activation(module: Any, inputs: tuple[Any, ...], layer_index: int = layer) -> None:
            captures[layer_index]["activation"] = inputs[0].detach().cpu()[0].float()

        def capture_mlp_output(module: Any, inputs: tuple[Any, ...], output: torch.Tensor, layer_index: int = layer) -> None:
            captures[layer_index]["mlpOutput"] = output.detach().cpu()[0].float()

        handles.extend([
            module.gate_proj.register_forward_pre_hook(capture_input),
            module.gate_proj.register_forward_hook(capture_gate),
            module.up_proj.register_forward_hook(capture_up),
            module.down_proj.register_forward_pre_hook(capture_activation),
            module.down_proj.register_forward_hook(capture_mlp_output),
        ])
    try:
        with torch.no_grad():
            outputs = model(**encoded)
    finally:
        for handle in handles:
            handle.remove()
    required = {"mlpInput", "gatePreactivation", "upProjection", "activation", "mlpOutput"}
    for layer, values in captures.items():
        missing = sorted(required - set(values))
        if missing:
            raise RuntimeError(f"Missing SwiGLU captures for layer {layer}: {missing}")
    return outputs, captures


def _top_signed_drivers(
    input_vector: torch.Tensor,
    weight_vector: torch.Tensor,
    bias: float,
) -> List[Dict[str, Any]]:
    contributions = input_vector.float() * weight_vector.float()
    indices = torch.topk(contributions.abs(), k=min(8, contributions.numel())).indices
    drivers = [
        {
            "mlpInputDimension": int(index),
            "inputValue": float(input_vector[int(index)]),
            "weight": float(weight_vector[int(index)]),
            "signedContribution": float(contributions[int(index)]),
        }
        for index in indices
    ]
    if bias:
        drivers.append({"bias": bias, "signedContribution": bias})
    return drivers


def _swiglu_unit_report(
    tokenizer: Any,
    mlps: Dict[int, Any],
    captures: Dict[int, Dict[str, torch.Tensor]],
    layer: int,
    unit: int,
    token_ids: torch.Tensor,
    token_index: int,
) -> Dict[str, Any]:
    mlp = mlps[layer]
    values = captures[layer]
    module = mlp._mlp
    gate_weight = module.gate_proj.weight.detach().cpu().float()[unit]
    up_weight = module.up_proj.weight.detach().cpu().float()[unit]
    down_row = mlp.c_proj.weight.detach().cpu().float()[unit]
    gate_bias_tensor = getattr(module.gate_proj, "bias", None)
    up_bias_tensor = getattr(module.up_proj, "bias", None)
    gate_bias = float(gate_bias_tensor.detach().cpu()[unit]) if gate_bias_tensor is not None else 0.0
    up_bias = float(up_bias_tensor.detach().cpu()[unit]) if up_bias_tensor is not None else 0.0
    mlp_input = values["mlpInput"][token_index]
    gate_pre = values["gatePreactivation"][token_index, unit]
    gate_activation = F.silu(gate_pre)
    up_projection = values["upProjection"][token_index, unit]
    activation = values["activation"][token_index, unit]
    contribution = activation * down_row
    output = values["mlpOutput"][token_index]
    output_norm = torch.linalg.vector_norm(output).clamp_min(1e-12)
    contribution_norm = torch.linalg.vector_norm(contribution)
    alignment = torch.dot(contribution, output) / (
        contribution_norm * output_norm
    ).clamp_min(1e-12)
    token_scores = values["activation"][:, unit].abs() * torch.linalg.vector_norm(down_row)
    token_positions = torch.topk(token_scores, k=min(5, token_scores.numel())).indices
    return {
        "layer": layer,
        "unit": unit,
        "unitType": "SwiGLU gate/value pair channel",
        "gatePreactivation": float(gate_pre),
        "gateActivationSiLU": float(gate_activation),
        "upProjection": float(up_projection),
        "productActivation": float(activation),
        "gateBias": gate_bias,
        "upBias": up_bias,
        "downProjectionNorm": float(torch.linalg.vector_norm(down_row)),
        "contributionNorm": float(contribution_norm),
        "contributionAlignmentWithLayerMlpOutput": float(alignment),
        "topGateInputDrivers": _top_signed_drivers(mlp_input, gate_weight, gate_bias),
        "topUpInputDrivers": _top_signed_drivers(mlp_input, up_weight, up_bias),
        "reactivatesAtPromptTokens": [
            {
                "position": int(position),
                "token": clean_token(tokenizer, int(token_ids[position])),
                "productActivation": float(values["activation"][int(position), unit]),
                "contributionNorm": float(token_scores[int(position)]),
            }
            for position in token_positions
        ],
        "mechanisticDescription": (
            "The normalized MLP input drives a learned gate projection and value projection. "
            "SiLU(gate) multiplies the value channel; the resulting scalar scales one row "
            "of the down projection and adds that vector to this block's MLP output."
        ),
    }


def _run_swiglu_explanation(args: argparse.Namespace, config: Any, device: str) -> Dict[str, Any]:
    model_type = str(getattr(config, "model_type", ""))
    adapter = create_adapter_for_model_type(model_type)
    if adapter.activation_family != "swiglu":
        raise RuntimeError(
            f"The multi-layer instrumented explainer currently supports SwiGLU adapters; "
            f"model_type={model_type!r} uses {adapter.activation_family!r}."
        )
    tokenizer, model = adapter.load_model_and_tokenizer(
        args.model,
        local_files_only=args.local_files_only,
        device=device,
    )
    layer_count = int(getattr(model.config, "num_hidden_layers"))
    layers = parse_layer_selection(args.layers, layer_count, args.layer)
    mlps = adapter.mlp_modules(model, layers)
    encoded = tokenizer(args.prompt, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    token_ids = encoded["input_ids"][0].detach().cpu()
    token_index = normalize_token_index(args.token_index, int(token_ids.numel()))
    outputs, captures = _capture_swiglu_layers(model, mlps, encoded)
    internal_operator_checks: Dict[str, Any] = {}
    for layer, mlp in mlps.items():
        values = captures[layer]
        reconstructed_product = F.silu(values["gatePreactivation"]) * values["upProjection"]
        product_error = float((reconstructed_product - values["activation"]).abs().max())
        projection_weight = mlp.c_proj.weight.detach().cpu().float()
        reconstructed_output = values["activation"] @ projection_weight
        down_bias = getattr(mlp._mlp.down_proj, "bias", None)
        if down_bias is not None:
            reconstructed_output = reconstructed_output + down_bias.detach().cpu().float()
        output_error = float((reconstructed_output - values["mlpOutput"]).abs().max())
        product_matches = torch.allclose(
            reconstructed_product,
            values["activation"],
            rtol=2e-4,
            atol=2e-5,
        )
        output_matches = torch.allclose(
            reconstructed_output,
            values["mlpOutput"],
            rtol=2e-4,
            atol=2e-5,
        )
        if not product_matches or not output_matches:
            raise RuntimeError(
                f"SwiGLU hook reconstruction mismatch at layer {layer}: "
                f"product_max_abs_error={product_error}, "
                f"down_projection_max_abs_error={output_error}."
            )
        internal_operator_checks[str(layer)] = {
            "gateValueProductMatchesHook": product_matches,
            "productMaxAbsError": product_error,
            "downProjectionMatchesHook": output_matches,
            "downProjectionMaxAbsError": output_error,
            "check": "SiLU(gate_proj(mlp_input))*up_proj(mlp_input), then down_proj",
        }
    baseline_logits = outputs.logits.detach().cpu()[0, token_index].float()
    top_logits = torch.topk(
        baseline_logits,
        k=min(args.top_logits, baseline_logits.numel()),
    )
    candidate_token_ids = [int(index) for index in top_logits.indices]

    candidates: List[Dict[str, Any]] = []
    for layer, mlp in mlps.items():
        projection = mlp.c_proj.weight.detach().cpu().float()
        scores = captures[layer]["activation"][token_index].abs() * torch.linalg.vector_norm(
            projection,
            dim=1,
        )
        for unit_tensor in torch.topk(scores, k=min(args.top_neurons, scores.numel())).indices:
            unit = int(unit_tensor)
            item = _swiglu_unit_report(
                tokenizer,
                mlps,
                captures,
                layer,
                unit,
                token_ids,
                token_index,
            )
            item["selectionScore"] = float(scores[unit])
            candidates.append(item)
    candidates.sort(key=lambda item: (-item["selectionScore"], item["layer"], item["unit"]))
    units = candidates[:args.top_neurons]

    control_items: List[Dict[str, Any]] = []
    if args.ablation_mode != "none":
        ablate_positions = (
            [token_index]
            if args.ablation_mode == "selected-token"
            else list(range(int(token_ids.numel())))
        )
        for item in units:
            ablated_logits = adapter.run_activation_ablation_logits(
                model,
                mlps[int(item["layer"])],
                encoded,
                int(item["unit"]),
                ablate_positions,
                token_index,
            )
            effects = logit_effects(
                tokenizer,
                baseline_logits,
                ablated_logits,
                candidate_token_ids,
            )
            item["causalAblation"] = {
                "mode": args.ablation_mode,
                "intervention": (
                    f"zero SwiGLU product channel {item['unit']} at layer {item['layer']}"
                ),
                "tokenPositions": ablate_positions,
                "candidateLogitEffects": effects,
                "topLogitsAfterRemoval": top_logit_items(
                    tokenizer,
                    ablated_logits,
                    min(args.top_logits, len(candidate_token_ids)),
                ),
                "maxAbsLogitDrop": max_abs_effect(effects),
            }
        if args.negative_controls:
            rng = random.Random(args.control_seed)
            selected_by_layer: Dict[int, List[int]] = {}
            for item in units:
                selected_by_layer.setdefault(int(item["layer"]), []).append(int(item["unit"]))
            for layer, selected_units in selected_by_layer.items():
                activation = captures[layer]["activation"][token_index]
                exclusions = set(selected_units)
                matched = adapter.pick_amplitude_matched_controls(
                    activation,
                    selected_units,
                    exclusions,
                    1,
                    rng,
                )
                if matched:
                    exclusions.add(int(matched[0]["neuron"]))
                inactive, threshold = adapter.pick_inactive_controls(
                    activation,
                    exclusions,
                    1,
                    rng,
                )
                for kind, choices in (
                    ("active-amplitude-matched", matched),
                    ("inactive", inactive),
                ):
                    for choice in choices:
                        control_unit = int(choice["neuron"])
                        ablated_logits = adapter.run_activation_ablation_logits(
                            model,
                            mlps[layer],
                            encoded,
                            control_unit,
                            [token_index],
                            token_index,
                        )
                        effects = logit_effects(
                            tokenizer,
                            baseline_logits,
                            ablated_logits,
                            candidate_token_ids,
                        )
                        control_items.append({
                            "layer": layer,
                            "unit": control_unit,
                            "kind": kind,
                            "selectionMetadata": choice,
                            "inactiveThresholdAbs": threshold if kind == "inactive" else None,
                            "candidateLogitEffects": effects,
                            "maxAbsLogitDrop": max_abs_effect(effects),
                        })

    selected_effect_by_layer: Dict[int, float] = {}
    for item in units:
        layer = int(item["layer"])
        selected_effect_by_layer[layer] = max(
            selected_effect_by_layer.get(layer, 0.0),
            float(item.get("causalAblation", {}).get("maxAbsLogitDrop", 0.0)),
        )
    max_control_by_layer: Dict[int, float] = {}
    for item in control_items:
        layer = int(item["layer"])
        max_control_by_layer[layer] = max(
            max_control_by_layer.get(layer, 0.0),
            float(item["maxAbsLogitDrop"]),
        )
    control_comparison = {
        str(layer): {
            "maxSelectedUnitAbsLogitEffect": selected_effect_by_layer[layer],
            "maxMeasuredControlAbsLogitEffect": max_control_by_layer.get(layer, 0.0),
            "selectedToMaxControlRatio": (
                selected_effect_by_layer[layer] / max_control_by_layer[layer]
                if max_control_by_layer.get(layer, 0.0) > 0.0
                else None
            ),
        }
        for layer in selected_effect_by_layer
    }

    return {
        "schemaVersion": 1,
        "mode": "instrumented-swiglu-mechanistic-reader",
        "synthetic": False,
        "surrogate": False,
        "analysisAccess": "instrumented-real-model-weights-hooks-and-causal-ablations",
        "model": adapter.model_identity(model, args.model, device, bool(args.local_files_only)),
        "runtimeDependencies": {
            "python": sys.executable,
            "torch": torch.__version__,
            "transformers": package_version("transformers"),
            "numpy": package_version("numpy"),
            "torchao": package_version("torchao"),
        },
        "prompt": args.prompt,
        "tokens": [
            {"position": index, "id": int(token_id), "text": clean_token(tokenizer, int(token_id))}
            for index, token_id in enumerate(token_ids)
        ],
        "selection": {
            "layers": layers,
            "tokenIndex": token_index,
            "tokenId": int(token_ids[token_index]),
            "token": clean_token(tokenizer, int(token_ids[token_index])),
            "component": "SwiGLU gate/value pair channel before down projection",
        },
        "hookShapes": {
            str(layer): {key: list(value.shape) for key, value in values.items()}
            for layer, values in captures.items()
        },
        "internalOperatorChecks": internal_operator_checks,
        "nextToken": {"topLogits": top_logit_items(tokenizer, baseline_logits, args.top_logits)},
        "units": units,
        "negativeControls": {
            "enabled": bool(args.negative_controls and args.ablation_mode != "none"),
            "selection": "one active-amplitude-matched and one inactive control per selected layer",
            "items": control_items,
            "comparisonByLayer": control_comparison,
        },
        "accounting": {
            "baselineForwardPasses": 1,
            "unitAblationForwardPasses": len(units) if args.ablation_mode != "none" else 0,
            "negativeControlForwardPasses": len(control_items),
            "totalForwardPasses": (
                1
                + (len(units) if args.ablation_mode != "none" else 0)
                + len(control_items)
            ),
        },
        "claim": {
            "does": (
                "Uses real model hooks and weights to decompose selected SwiGLU gate/value "
                "channels, show their strongest MLP-input drivers, and measure full-model "
                "next-token logit changes when individual product channels are removed."
            ),
            "doesNot": (
                "Does not recover a complete unique reasoning trace, prove semantic necessity, "
                "or represent an oracle-only black-box analysis."
            ),
        },
    }


def make_swiglu_text_report(report: Dict[str, Any]) -> str:
    lines = [
        "GCL instrumented SwiGLU reader",
        f"model: {report['model']['id']}",
        f"layers: {report['selection']['layers']}",
        f"token: #{report['selection']['tokenIndex']} {json.dumps(report['selection']['token'])}",
        "access: real weights + hooks + causal ablations (not black-box)",
        "",
        "next-token logits:",
    ]
    for item in report["nextToken"]["topLogits"]:
        lines.append(f"  {item['rank']:>2}. {json.dumps(item['token'])} logit={item['logit']:.6f}")
    lines.extend(["", "highest-scoring SwiGLU channels:"])
    verified_layers = sum(
        1 for check in report["internalOperatorChecks"].values()
        if check["gateValueProductMatchesHook"] and check["downProjectionMatchesHook"]
    )
    lines.append(
        f"internal forward check: exact SiLU-gate/value products and down projections "
        f"reconstructed in {verified_layers}/{len(report['internalOperatorChecks'])} scanned layers"
    )
    for rank, item in enumerate(report["units"], start=1):
        lines.append(
            f"  {rank:>2}. layer {item['layer']} unit {item['unit']} "
            f"gate={item['gatePreactivation']:.4f} silu={item['gateActivationSiLU']:.4f} "
            f"value={item['upProjection']:.4f} product={item['productActivation']:.4f}"
        )
        gate = ", ".join(
            (
                f"d{driver['mlpInputDimension']}:{driver['signedContribution']:.3f}"
                if "mlpInputDimension" in driver
                else f"bias:{driver['signedContribution']:.3f}"
            )
            for driver in item["topGateInputDrivers"][:3]
        )
        up = ", ".join(
            (
                f"d{driver['mlpInputDimension']}:{driver['signedContribution']:.3f}"
                if "mlpInputDimension" in driver
                else f"bias:{driver['signedContribution']:.3f}"
            )
            for driver in item["topUpInputDrivers"][:3]
        )
        lines.append(f"      strongest gate-input terms: {gate}")
        lines.append(f"      strongest value-input terms: {up}")
        ablation = item.get("causalAblation")
        if ablation:
            effects = ", ".join(
                f"{json.dumps(effect['token'])}:{effect['logitDropWhenRemoved']:+.4f}"
                for effect in ablation["candidateLogitEffects"][:3]
            )
            lines.append(f"      full-model effect when removed: {effects}")
    controls = report["negativeControls"]
    if controls["enabled"]:
        lines.extend(["", "matched/inactive control ablations:"])
        for item in controls["items"]:
            effects = ", ".join(
                f"{json.dumps(effect['token'])}:{effect['logitDropWhenRemoved']:+.4f}"
                for effect in item["candidateLogitEffects"][:2]
            )
            lines.append(
                f"  layer {item['layer']} unit {item['unit']} ({item['kind']}): {effects}"
            )
        for layer, comparison in controls["comparisonByLayer"].items():
            ratio = comparison["selectedToMaxControlRatio"]
            lines.append(
                f"  layer {layer}: max selected/control effect ratio="
                f"{ratio:.3f}" if ratio is not None
                else f"  layer {layer}: measured control effect is zero"
            )
    lines.extend(["", report["claim"]["doesNot"]])
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

    config = AutoConfig.from_pretrained(args.model, local_files_only=args.local_files_only)
    model_type = str(getattr(config, "model_type", ""))
    if model_type in {"qwen2", "llama", "mistral"}:
        report = _run_swiglu_explanation(args, config, device)
        if args.output:
            os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as file:
                json.dump(report, file, indent=2)
                file.write("\n")
        print(json.dumps(report, indent=2) if args.format == "json" else make_swiglu_text_report(report))
        return
    if model_type != "gpt2":
        raise RuntimeError(
            "The live mechanistic explainer currently supports GPT-2 GELU and "
            "Qwen2/LLaMA/Mistral SwiGLU architectures."
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=args.local_files_only)
    model.eval()
    model.to(device)
    mlp = ensure_gpt2_mlp_shape(model, args.layer)

    encoded = tokenizer(args.prompt, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    token_ids = encoded["input_ids"][0].detach().cpu()
    token_index = normalize_token_index(args.token_index, int(token_ids.numel()))

    outputs, captures = run_with_mlp_captures(model, mlp, encoded)
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
    candidate_token_ids = [int(index) for index in top_logits.indices]
    attach_causal_ablation_reports(
        model,
        mlp,
        encoded,
        tokenizer,
        logits,
        args.layer,
        token_index,
        neurons,
        candidate_token_ids,
        args.ablation_mode
    )
    negative_controls = None
    if args.negative_controls and args.ablation_mode != "none":
        negative_controls = build_negative_control_report(
            model=model,
            mlp=mlp,
            tokenizer=tokenizer,
            encoded=encoded,
            baseline_logits=logits,
            activation=captures["activation"],
            token_ids=token_ids,
            layer=args.layer,
            token_index=token_index,
            neurons=neurons,
            candidate_token_ids=candidate_token_ids,
            c_proj_weight=c_proj_weight,
            args=args
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
            "topLogits": top_logit_items(tokenizer, logits, args.top_logits)
        },
        "neurons": neurons,
        "negativeControls": negative_controls,
        "claim": {
            "does": "Loads the real model, runs a real forward pass, hooks real MLP activations, and performs real activation ablations to measure selected-neuron effects on next-token logits.",
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
