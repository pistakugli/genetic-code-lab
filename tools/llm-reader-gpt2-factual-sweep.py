#!/usr/bin/env python3

import argparse
import importlib.util
import json
import math
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as transformers_logging

transformers_logging.disable_progress_bar()


FACTS: List[Dict[str, str]] = [
    {"subject": "France", "answer": "Paris"},
    {"subject": "Germany", "answer": "Berlin"},
    {"subject": "Italy", "answer": "Rome"},
    {"subject": "Spain", "answer": "Madrid"},
    {"subject": "Portugal", "answer": "Lisbon"},
    {"subject": "Greece", "answer": "Athens"},
    {"subject": "Egypt", "answer": "Cairo"},
    {"subject": "Japan", "answer": "Tokyo"},
    {"subject": "China", "answer": "Beijing"},
    {"subject": "Russia", "answer": "Moscow"},
    {"subject": "Canada", "answer": "Ottawa"},
    {"subject": "Australia", "answer": "Canberra"},
    {"subject": "Brazil", "answer": "Brasilia"},
    {"subject": "Sweden", "answer": "Stockholm"},
    {"subject": "Norway", "answer": "Oslo"},
    {"subject": "Denmark", "answer": "Copenhagen"},
    {"subject": "Finland", "answer": "Helsinki"},
    {"subject": "Poland", "answer": "Warsaw"},
    {"subject": "Hungary", "answer": "Budapest"},
    {"subject": "Austria", "answer": "Vienna"},
    {"subject": "Belgium", "answer": "Brussels"},
    {"subject": "Netherlands", "answer": "Amsterdam"},
    {"subject": "Ireland", "answer": "Dublin"},
    {"subject": "Turkey", "answer": "Ankara"},
    {"subject": "Iran", "answer": "Tehran"},
    {"subject": "Iraq", "answer": "Baghdad"},
    {"subject": "Syria", "answer": "Damascus"},
    {"subject": "Jordan", "answer": "Amman"},
    {"subject": "Israel", "answer": "Jerusalem"},
    {"subject": "Saudi Arabia", "answer": "Riyadh"},
    {"subject": "Pakistan", "answer": "Islamabad"},
    {"subject": "Bangladesh", "answer": "Dhaka"},
    {"subject": "Thailand", "answer": "Bangkok"},
    {"subject": "Vietnam", "answer": "Hanoi"},
    {"subject": "Indonesia", "answer": "Jakarta"},
    {"subject": "Philippines", "answer": "Manila"},
    {"subject": "South Korea", "answer": "Seoul"},
    {"subject": "Belarus", "answer": "Minsk"},
    {"subject": "Estonia", "answer": "Tallinn"},
    {"subject": "Latvia", "answer": "Riga"},
    {"subject": "Lithuania", "answer": "Vilnius"},
    {"subject": "Romania", "answer": "Bucharest"},
    {"subject": "Bulgaria", "answer": "Sofia"},
    {"subject": "Serbia", "answer": "Belgrade"},
    {"subject": "Croatia", "answer": "Zagreb"},
    {"subject": "Slovakia", "answer": "Bratislava"},
    {"subject": "Czech Republic", "answer": "Prague"},
    {"subject": "Switzerland", "answer": "Bern"},
    {"subject": "Morocco", "answer": "Rabat"},
    {"subject": "Algeria", "answer": "Algiers"},
    {"subject": "Tunisia", "answer": "Tunis"},
    {"subject": "Kenya", "answer": "Nairobi"},
    {"subject": "Nigeria", "answer": "Abuja"},
    {"subject": "Ghana", "answer": "Accra"},
    {"subject": "Senegal", "answer": "Dakar"},
    {"subject": "Mali", "answer": "Bamako"},
    {"subject": "Niger", "answer": "Niamey"},
    {"subject": "Sudan", "answer": "Khartoum"},
    {"subject": "Uganda", "answer": "Kampala"},
    {"subject": "Tanzania", "answer": "Dodoma"},
    {"subject": "Rwanda", "answer": "Kigali"},
    {"subject": "Chile", "answer": "Santiago"},
    {"subject": "Peru", "answer": "Lima"},
    {"subject": "Colombia", "answer": "Bogota"},
    {"subject": "Venezuela", "answer": "Caracas"},
    {"subject": "Cuba", "answer": "Havana"},
    {"subject": "Jamaica", "answer": "Kingston"},
    {"subject": "Uruguay", "answer": "Montevideo"},
    {"subject": "Paraguay", "answer": "Asuncion"},
    {"subject": "Ecuador", "answer": "Quito"},
    {"subject": "largest planet", "prompt": "The largest planet in the solar system is", "answer": "Jupiter"},
    {"subject": "smallest planet", "prompt": "The smallest planet in the solar system is", "answer": "Mercury"},
    {"subject": "nearest star", "prompt": "The nearest star to Earth is the", "answer": "Sun"},
    {"subject": "grass color", "prompt": "The color of grass is", "answer": "green"},
    {"subject": "opposite of hot", "prompt": "The opposite of hot is", "answer": "cold"},
    {"subject": "number after nine", "prompt": "The number after nine is", "answer": "ten"},
    {"subject": "month after January", "prompt": "The month after January is", "answer": "February"},
    {"subject": "frozen water", "prompt": "Water freezes into", "answer": "ice"},
    {"subject": "baby dog", "prompt": "A baby dog is called a", "answer": "puppy"},
    {"subject": "female parent", "prompt": "A female parent is called a", "answer": "mother"},
    {"subject": "male parent", "prompt": "A male parent is called a", "answer": "father"},
    {"subject": "medical profession", "prompt": "A person who treats sick people is a", "answer": "doctor"},
    {"subject": "school profession", "prompt": "A person who teaches students is a", "answer": "teacher"},
    {"subject": "largest ocean", "prompt": "The largest ocean is the", "answer": "Pacific"},
    {"subject": "language of Spain", "prompt": "The main language of Spain is", "answer": "Spanish"},
    {"subject": "breathing gas", "prompt": "Humans breathe", "answer": "oxygen"},
    {"subject": "precious metal gold", "prompt": "The metal in a gold ring is", "answer": "gold"},
    {"subject": "day after Sunday", "prompt": "The day after Sunday is", "answer": "Monday"},
    {"subject": "cold season", "prompt": "The cold season is", "answer": "winter"},
    {"subject": "warm season", "prompt": "The warm season is", "answer": "summer"},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a real GPT-2 factual-prompt causal-neuron sweep with negative controls."
    )
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--case-count", type=int, default=50)
    parser.add_argument("--min-valid-cases", type=int, default=50)
    parser.add_argument("--layers", default="0,3,6,9,11")
    parser.add_argument("--top-neurons", type=int, default=6)
    parser.add_argument("--top-logits", type=int, default=8)
    parser.add_argument("--control-count", type=int, default=8)
    parser.add_argument("--control-prompt", default="The weather in Berlin is")
    parser.add_argument("--control-seed", type=int, default=260603)
    parser.add_argument("--max-graph-nodes", type=int, default=4)
    parser.add_argument("--min-pass-rate", type=float, default=0.55)
    parser.add_argument("--min-target-drop", type=float, default=0.05)
    parser.add_argument("--min-margin-over-control-p95", type=float, default=0.02)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


def load_live_helper() -> Any:
    helper_path = Path(__file__).resolve().with_name("llm-reader-gpt2-live.py")
    spec = importlib.util.spec_from_file_location("gcl_live_reader", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load helper module from {helper_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


live = load_live_helper()


def parse_layers(value: str) -> List[int]:
    layers = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        parsed = int(item)
        if parsed < 0:
            raise RuntimeError("--layers must contain non-negative integers.")
        layers.append(parsed)
    if not layers:
        raise RuntimeError("--layers must contain at least one layer.")
    return sorted(set(layers))


def percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return float(ordered[index])


def summarize_drops(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    drops = [float(item["targetLogitDrop"]) for item in items]
    positive = [max(0.0, value) for value in drops]
    absolute = [abs(value) for value in drops]
    return {
        "count": len(items),
        "maxPositiveTargetLogitDrop": max(positive) if positive else 0.0,
        "meanPositiveTargetLogitDrop": sum(positive) / len(positive) if positive else 0.0,
        "p50PositiveTargetLogitDrop": percentile(positive, 0.50),
        "p90PositiveTargetLogitDrop": percentile(positive, 0.90),
        "p95PositiveTargetLogitDrop": percentile(positive, 0.95),
        "maxAbsTargetLogitDrop": max(absolute) if absolute else 0.0,
        "p95AbsTargetLogitDrop": percentile(absolute, 0.95),
    }


def encode_single_answer_token(tokenizer: Any, answer: str) -> Tuple[int, str] | None:
    text = " " + answer
    encoded = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(encoded) != 1:
        return None
    return int(encoded[0]), live.clean_token(tokenizer, int(encoded[0]))


def run_with_multi_layer_captures(
    model: Any,
    mlps: Dict[int, Any],
    encoded: Dict[str, torch.Tensor],
) -> Tuple[Any, Dict[int, Dict[str, torch.Tensor]]]:
    captures: Dict[int, Dict[str, torch.Tensor]] = {layer: {} for layer in mlps}
    handles = []
    for layer, mlp in mlps.items():
        def capture_act(module: Any, inputs: Any, output: torch.Tensor, layer_index: int = layer) -> None:
            captures[layer_index]["activation"] = output.detach().cpu()[0]
        handles.append(mlp.act.register_forward_hook(capture_act))
    try:
        with torch.no_grad():
            outputs = model(**encoded)
    finally:
        for handle in handles:
            handle.remove()
    missing = [layer for layer, values in captures.items() if "activation" not in values]
    if missing:
        raise RuntimeError(f"Missing activation captures for layers: {missing}")
    return outputs, captures


def target_rank(logits: torch.Tensor, target_token_id: int) -> int:
    values = logits.detach().cpu()
    target = float(values[target_token_id])
    return int((values > target).sum().item()) + 1


def top_logit_tokens(tokenizer: Any, logits: torch.Tensor, count: int) -> List[Dict[str, Any]]:
    top = torch.topk(logits.detach().cpu(), k=min(max(1, count), int(logits.numel())))
    return [
        {
            "rank": index + 1,
            "tokenId": int(token_id),
            "token": live.clean_token(tokenizer, int(token_id)),
            "logit": float(value),
        }
        for index, (value, token_id) in enumerate(zip(top.values, top.indices))
    ]


def ablate_target_drop(
    model: Any,
    mlp: Any,
    encoded: Dict[str, torch.Tensor],
    baseline_logits: torch.Tensor,
    neuron: int,
    ablate_positions: List[int],
    token_index: int,
    target_token_id: int,
) -> float:
    ablated = live.run_activation_ablation_logits(
        model,
        mlp,
        encoded,
        int(neuron),
        ablate_positions,
        token_index,
    )
    return float(baseline_logits[target_token_id] - ablated[target_token_id])


def run_graph_ablation_logits(
    model: Any,
    mlps: Dict[int, Any],
    encoded: Dict[str, torch.Tensor],
    interventions: List[Dict[str, Any]],
    read_token_index: int,
) -> torch.Tensor:
    seq_len = int(encoded["input_ids"].shape[1])
    grouped: Dict[int, Dict[int, List[int]]] = {}
    for intervention in interventions:
        layer = int(intervention["layer"])
        neuron = int(intervention["neuron"])
        positions = sorted(set(int(position) for position in intervention.get("ablateTokenPositions", [])))
        valid_positions = [position for position in positions if 0 <= position < seq_len]
        if not valid_positions:
            raise RuntimeError("Graph ablation requires at least one valid token position per intervention.")
        grouped.setdefault(layer, {}).setdefault(neuron, [])
        grouped[layer][neuron] = sorted(set(grouped[layer][neuron] + valid_positions))
    if not grouped:
        raise RuntimeError("Graph ablation requires at least one intervention.")

    handles = []
    for layer, neurons in grouped.items():
        if layer not in mlps:
            raise RuntimeError(f"Graph ablation references unscanned layer {layer}.")

        def patch_activation(module: Any, inputs: Any, output: torch.Tensor, layer_index: int = layer) -> torch.Tensor:
            patched = output.clone()
            for neuron, positions in grouped[layer_index].items():
                for position in positions:
                    patched[:, position, neuron] = 0
            return patched

        handles.append(mlps[layer].act.register_forward_hook(patch_activation))
    try:
        with torch.no_grad():
            return model(**encoded).logits.detach().cpu()[0, read_token_index]
    finally:
        for handle in handles:
            handle.remove()


def graph_target_drop(
    *,
    model: Any,
    mlps: Dict[int, Any],
    encoded: Dict[str, torch.Tensor],
    baseline_logits: torch.Tensor,
    interventions: List[Dict[str, Any]],
    token_index: int,
    target_token_id: int,
) -> float:
    ablated = run_graph_ablation_logits(model, mlps, encoded, interventions, token_index)
    return float(baseline_logits[target_token_id] - ablated[target_token_id])


def token_support_edges(
    *,
    tokenizer: Any,
    token_ids: torch.Tensor,
    activation_matrix: torch.Tensor,
    c_proj_weight: torch.Tensor,
    layer: int,
    neuron: int,
    limit: int = 3,
) -> List[Dict[str, Any]]:
    projection_norm = float(torch.linalg.vector_norm(c_proj_weight[neuron]))
    scores = activation_matrix[:, neuron].abs() * projection_norm
    top = torch.topk(scores.detach().cpu(), k=min(limit, int(scores.numel())))
    return [
        {
            "source": f"token:{int(position)}",
            "target": f"mlp:{layer}:{neuron}",
            "kind": "token-activation-support",
            "tokenIndex": int(position),
            "token": live.clean_token(tokenizer, int(token_ids[int(position)])),
            "activation": float(activation_matrix[int(position), neuron]),
            "supportScore": float(score),
        }
        for score, position in zip(top.values, top.indices)
    ]


def build_candidate_records(
    *,
    model: Any,
    tokenizer: Any,
    encoded: Dict[str, torch.Tensor],
    baseline_logits: torch.Tensor,
    captures: Dict[int, Dict[str, torch.Tensor]],
    mlps: Dict[int, Any],
    token_ids: torch.Tensor,
    token_index: int,
    target_token_id: int,
    top_neurons: int,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for layer, mlp in mlps.items():
        activation_matrix = captures[layer]["activation"]
        activation = activation_matrix[token_index]
        c_proj_weight = mlp.c_proj.weight.detach().cpu()
        projection_norms = torch.linalg.vector_norm(c_proj_weight, dim=1)
        scores = activation.abs() * projection_norms
        top = torch.topk(scores.detach().cpu(), k=min(top_neurons, int(scores.numel())))
        for rank, (score, neuron_tensor) in enumerate(zip(top.values, top.indices), start=1):
            neuron = int(neuron_tensor)
            drop = ablate_target_drop(
                model,
                mlp,
                encoded,
                baseline_logits,
                neuron,
                [token_index],
                token_index,
                target_token_id,
            )
            records.append({
                "layer": int(layer),
                "rankWithinLayer": rank,
                "neuron": neuron,
                "selectionScore": float(score),
                "activation": float(activation[neuron]),
                "targetLogitDrop": drop,
                "tokenSupportEdges": token_support_edges(
                    tokenizer=tokenizer,
                    token_ids=token_ids,
                    activation_matrix=activation_matrix,
                    c_proj_weight=c_proj_weight,
                    layer=int(layer),
                    neuron=neuron,
                ),
            })
    return sorted(records, key=lambda item: item["targetLogitDrop"], reverse=True)


def make_control_drop_item(
    *,
    model: Any,
    mlp: Any,
    encoded: Dict[str, torch.Tensor],
    baseline_logits: torch.Tensor,
    layer: int,
    neuron: int,
    token_index: int,
    target_token_id: int,
    metadata: Dict[str, Any],
    ablate_positions: List[int] | None = None,
) -> Dict[str, Any]:
    positions = [token_index] if ablate_positions is None else ablate_positions
    drop = ablate_target_drop(
        model,
        mlp,
        encoded,
        baseline_logits,
        neuron,
        positions,
        token_index,
        target_token_id,
    )
    return {
        **metadata,
        "layer": int(layer),
        "neuron": int(neuron),
        "ablateTokenPositions": positions,
        "targetLogitDrop": drop,
    }


def build_controls(
    *,
    model: Any,
    tokenizer: Any,
    encoded: Dict[str, torch.Tensor],
    baseline_logits: torch.Tensor,
    mlp: Any,
    layer: int,
    activation: torch.Tensor,
    token_ids: torch.Tensor,
    token_index: int,
    target_token_id: int,
    primary_neuron: int,
    candidate_neurons: List[int],
    control_count: int,
    control_prompt: str,
    rng: random.Random,
) -> Dict[str, Any]:
    baseline_activation = activation[token_index]
    used = set(candidate_neurons)
    groups = []

    matched_choices = live.pick_amplitude_matched_controls(
        baseline_activation,
        [primary_neuron],
        used,
        control_count,
        rng,
    )
    used.update(int(choice["neuron"]) for choice in matched_choices)
    matched_items = [
        make_control_drop_item(
            model=model,
            mlp=mlp,
            encoded=encoded,
            baseline_logits=baseline_logits,
            layer=layer,
            neuron=int(choice["neuron"]),
            token_index=token_index,
            target_token_id=target_token_id,
            metadata={
                "kind": "random-active-same-amplitude",
                "matchedToNeuron": int(choice["matchedToNeuron"]),
                "matchedToActivationAbs": float(choice["matchedToActivationAbs"]),
                "activationAbs": float(choice["activationAbs"]),
            },
        )
        for choice in matched_choices
    ]
    groups.append({
        "kind": "random-active-same-amplitude",
        "summary": summarize_drops(matched_items),
        "items": matched_items,
    })

    inactive_choices, inactive_threshold = live.pick_inactive_controls(
        baseline_activation,
        used,
        control_count,
        rng,
    )
    used.update(int(choice["neuron"]) for choice in inactive_choices)
    inactive_items = [
        make_control_drop_item(
            model=model,
            mlp=mlp,
            encoded=encoded,
            baseline_logits=baseline_logits,
            layer=layer,
            neuron=int(choice["neuron"]),
            token_index=token_index,
            target_token_id=target_token_id,
            metadata={
                "kind": "random-inactive",
                "inactiveThresholdAbs": float(inactive_threshold),
                "activationAbs": float(choice["activationAbs"]),
            },
        )
        for choice in inactive_choices
    ]
    groups.append({
        "kind": "random-inactive",
        "summary": summarize_drops(inactive_items),
        "inactiveThresholdAbs": float(inactive_threshold),
        "items": inactive_items,
    })

    device = encoded["input_ids"].device
    control_encoded = tokenizer(control_prompt, return_tensors="pt")
    control_encoded = {key: value.to(device) for key, value in control_encoded.items()}
    _, control_captures = run_with_multi_layer_captures(model, {layer: mlp}, control_encoded)
    control_token_index = int(control_encoded["input_ids"].shape[1]) - 1
    control_activation = control_captures[layer]["activation"][control_token_index]
    projection_norms = torch.linalg.vector_norm(mlp.c_proj.weight.detach().cpu(), dim=1)
    control_scores = control_activation.abs() * projection_norms
    off_prompt_items = []
    skipped_prior = 0
    for neuron_tensor in torch.argsort(control_scores.detach().cpu(), descending=True):
        neuron = int(neuron_tensor)
        if neuron in used:
            skipped_prior += 1
            continue
        if len(off_prompt_items) >= control_count:
            break
        used.add(neuron)
        off_prompt_items.append(make_control_drop_item(
            model=model,
            mlp=mlp,
            encoded=encoded,
            baseline_logits=baseline_logits,
            layer=layer,
            neuron=neuron,
            token_index=token_index,
            target_token_id=target_token_id,
            metadata={
                "kind": "top-neuron-from-other-prompt",
                "controlPrompt": control_prompt,
                "controlPromptContributionScore": float(control_scores[neuron]),
                "selectedPromptActivation": float(baseline_activation[neuron]),
            },
        ))
    groups.append({
        "kind": "top-neuron-from-other-prompt",
        "summary": summarize_drops(off_prompt_items),
        "skippedPriorControlOverlap": skipped_prior,
        "items": off_prompt_items,
    })

    other_positions = [position for position in range(int(token_ids.numel())) if position != token_index]
    other_position_items = [
        make_control_drop_item(
            model=model,
            mlp=mlp,
            encoded=encoded,
            baseline_logits=baseline_logits,
            layer=layer,
            neuron=primary_neuron,
            token_index=token_index,
            target_token_id=target_token_id,
            ablate_positions=[position],
            metadata={
                "kind": "same-neuron-other-token-position",
                "sourceNeuron": int(primary_neuron),
                "otherTokenIndex": int(position),
                "otherToken": live.clean_token(tokenizer, int(token_ids[position])),
                "activationAtOtherToken": float(activation[position, primary_neuron]),
            },
        )
        for position in other_positions
    ]
    groups.append({
        "kind": "same-neuron-other-token-position",
        "summary": summarize_drops(other_position_items),
        "sourceNeuronCount": 1,
        "otherTokenPositionCount": len(other_positions),
        "items": other_position_items,
    })

    overlap = []
    for left_index, left in enumerate(groups):
        left_neurons = set(int(item["neuron"]) for item in left["items"])
        for right in groups[left_index + 1:]:
            right_neurons = set(int(item["neuron"]) for item in right["items"])
            shared = sorted(left_neurons & right_neurons)
            overlap.append({
                "left": left["kind"],
                "right": right["kind"],
                "count": len(shared),
                "neurons": shared[:25],
            })
    return {
        "controlCount": control_count,
        "groups": groups,
        "crossFamilyNeuronOverlap": overlap,
    }


def select_circuit_candidates(
    candidates: List[Dict[str, Any]],
    threshold: float,
    max_graph_nodes: int,
) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    used = set()
    for candidate in candidates:
        pair = (int(candidate["layer"]), int(candidate["neuron"]))
        if pair in used:
            continue
        if float(candidate["targetLogitDrop"]) >= threshold or not selected:
            selected.append(candidate)
            used.add(pair)
        if len(selected) >= max_graph_nodes:
            break
    return selected


def control_graphs_from_groups(
    *,
    model: Any,
    mlps: Dict[int, Any],
    encoded: Dict[str, torch.Tensor],
    baseline_logits: torch.Tensor,
    controls: Dict[str, Any],
    graph_size: int,
    token_index: int,
    target_token_id: int,
) -> Dict[str, Any]:
    graph_items = []
    for group in controls["groups"]:
        items = group["items"][:graph_size]
        if not items:
            continue
        interventions = [
            {
                "layer": int(item["layer"]),
                "neuron": int(item["neuron"]),
                "ablateTokenPositions": [int(position) for position in item.get("ablateTokenPositions", [token_index])],
            }
            for item in items
        ]
        drop = graph_target_drop(
            model=model,
            mlps=mlps,
            encoded=encoded,
            baseline_logits=baseline_logits,
            interventions=interventions,
            token_index=token_index,
            target_token_id=target_token_id,
        )
        graph_items.append({
            "kind": group["kind"],
            "nodeCount": len(interventions),
            "targetLogitDrop": drop,
            "interventions": interventions,
        })
    return {
        "items": graph_items,
        "summary": summarize_drops(graph_items),
    }


def build_circuit_graph(
    *,
    model: Any,
    tokenizer: Any,
    mlps: Dict[int, Any],
    encoded: Dict[str, torch.Tensor],
    baseline_logits: torch.Tensor,
    token_ids: torch.Tensor,
    token_index: int,
    target_token_id: int,
    target_token: str,
    candidates: List[Dict[str, Any]],
    controls: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    control_p95 = max(group["summary"]["p95PositiveTargetLogitDrop"] for group in controls["groups"])
    threshold = max(float(args.min_target_drop), control_p95 + float(args.min_margin_over_control_p95))
    selected = select_circuit_candidates(candidates, threshold, max(1, int(args.max_graph_nodes)))
    interventions = [
        {
            "layer": int(candidate["layer"]),
            "neuron": int(candidate["neuron"]),
            "ablateTokenPositions": [token_index],
        }
        for candidate in selected
    ]
    joint_drop = graph_target_drop(
        model=model,
        mlps=mlps,
        encoded=encoded,
        baseline_logits=baseline_logits,
        interventions=interventions,
        token_index=token_index,
        target_token_id=target_token_id,
    )
    control_graphs = control_graphs_from_groups(
        model=model,
        mlps=mlps,
        encoded=encoded,
        baseline_logits=baseline_logits,
        controls=controls,
        graph_size=len(interventions),
        token_index=token_index,
        target_token_id=target_token_id,
    )
    control_graph_p95 = float(control_graphs["summary"]["p95PositiveTargetLogitDrop"])
    graph_margin = joint_drop - control_graph_p95
    pass_graph = (
        len(selected) > 0
        and joint_drop >= float(args.min_target_drop)
        and graph_margin >= float(args.min_margin_over_control_p95)
        and all(pair["count"] == 0 for pair in controls["crossFamilyNeuronOverlap"])
    )

    nodes = [
        {"id": "fact", "kind": "fact-prompt"},
        {"id": "selected-circuit", "kind": "selected-causal-circuit", "selectedNeuronCount": len(selected)},
        {"id": f"target:{target_token_id}", "kind": "target-token", "tokenId": target_token_id, "token": target_token},
    ]
    nodes.extend(
        {"id": f"token:{index}", "kind": "prompt-token", "position": index, "token": live.clean_token(tokenizer, int(token_id))}
        for index, token_id in enumerate(token_ids)
    )
    for candidate in selected:
        nodes.append({
            "id": f"mlp:{candidate['layer']}:{candidate['neuron']}",
            "kind": "mlp-neuron",
            "layer": int(candidate["layer"]),
            "neuron": int(candidate["neuron"]),
            "selectionScore": float(candidate["selectionScore"]),
            "activation": float(candidate["activation"]),
        })
    nodes.extend(
        {"id": f"control:{item['kind']}", "kind": "control-graph", "controlKind": item["kind"]}
        for item in control_graphs["items"]
    )

    edges = [
        {"source": "fact", "target": f"target:{target_token_id}", "kind": "expected-answer", "answerToken": target_token}
    ]
    for candidate in selected:
        edges.extend(candidate["tokenSupportEdges"])
        edges.append({
            "source": f"mlp:{candidate['layer']}:{candidate['neuron']}",
            "target": f"target:{target_token_id}",
            "kind": "causal-target-logit-drop",
            "individualTargetLogitDrop": float(candidate["targetLogitDrop"]),
        })
    edges.append({
        "source": "selected-circuit",
        "target": f"target:{target_token_id}",
        "kind": "joint-graph-ablation",
        "jointTargetLogitDrop": joint_drop,
    })
    for item in control_graphs["items"]:
        edges.append({
            "source": f"control:{item['kind']}",
            "target": f"target:{target_token_id}",
            "kind": "control-graph-ablation",
            "targetLogitDrop": float(item["targetLogitDrop"]),
        })

    return {
        "kind": "gpt2-live-factual-circuit-graph",
        "selectedNodeCount": len(selected),
        "nodes": nodes,
        "edges": edges,
        "selectedInterventions": interventions,
        "selectedCandidates": selected,
        "jointAblation": {
            "targetLogitDrop": joint_drop,
            "controlGraphP95PositiveTargetLogitDrop": control_graph_p95,
            "controlGraphMaxPositiveTargetLogitDrop": float(control_graphs["summary"]["maxPositiveTargetLogitDrop"]),
            "marginOverControlGraphP95": graph_margin,
            "liftOverControlGraphP95": joint_drop / max(1e-9, control_graph_p95),
        },
        "controlGraphs": control_graphs,
        "gate": {
            "pass": pass_graph,
            "thresholdForIndividualCandidateSelection": threshold,
            "minTargetDrop": float(args.min_target_drop),
            "minMarginOverControlP95": float(args.min_margin_over_control_p95),
        },
    }


def run_case(
    *,
    model: Any,
    tokenizer: Any,
    mlps: Dict[int, Any],
    fact: Dict[str, str],
    args: argparse.Namespace,
    rng: random.Random,
) -> Dict[str, Any]:
    prompt = fact.get("prompt") or f"The capital of {fact['subject']} is"
    encoded = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    encoded = {key: value.to(device) for key, value in encoded.items()}
    token_ids = encoded["input_ids"][0].detach().cpu()
    token_index = int(token_ids.numel()) - 1
    target = encode_single_answer_token(tokenizer, fact["answer"])
    if target is None:
        return {
            "subject": fact["subject"],
            "answer": fact["answer"],
            "prompt": prompt,
            "status": "skipped",
            "skipReason": "answer is not one GPT-2 token with a leading space",
        }
    target_token_id, target_token = target
    outputs, captures = run_with_multi_layer_captures(model, mlps, encoded)
    baseline_logits = outputs.logits.detach().cpu()[0, token_index]
    candidates = build_candidate_records(
        model=model,
        tokenizer=tokenizer,
        encoded=encoded,
        baseline_logits=baseline_logits,
        captures=captures,
        mlps=mlps,
        token_ids=token_ids,
        token_index=token_index,
        target_token_id=target_token_id,
        top_neurons=args.top_neurons,
    )
    best = candidates[0] if candidates else None
    if best is None:
        return {
            "subject": fact["subject"],
            "answer": fact["answer"],
            "prompt": prompt,
            "status": "failed",
            "failureReason": "no candidate neurons",
        }
    best_layer = int(best["layer"])
    best_mlp = mlps[best_layer]
    controls = build_controls(
        model=model,
        tokenizer=tokenizer,
        encoded=encoded,
        baseline_logits=baseline_logits,
        mlp=best_mlp,
        layer=best_layer,
        activation=captures[best_layer]["activation"],
        token_ids=token_ids,
        token_index=token_index,
        target_token_id=target_token_id,
        primary_neuron=int(best["neuron"]),
        candidate_neurons=[int(item["neuron"]) for item in candidates if int(item["layer"]) == best_layer],
        control_count=args.control_count,
        control_prompt=args.control_prompt,
        rng=rng,
    )
    control_p95 = max(group["summary"]["p95PositiveTargetLogitDrop"] for group in controls["groups"])
    control_max = max(group["summary"]["maxPositiveTargetLogitDrop"] for group in controls["groups"])
    best_drop = float(best["targetLogitDrop"])
    margin = best_drop - control_p95
    circuit_graph = build_circuit_graph(
        model=model,
        tokenizer=tokenizer,
        mlps=mlps,
        encoded=encoded,
        baseline_logits=baseline_logits,
        token_ids=token_ids,
        token_index=token_index,
        target_token_id=target_token_id,
        target_token=target_token,
        candidates=candidates,
        controls=controls,
        args=args,
    )
    pass_case = bool(circuit_graph["gate"]["pass"])
    return {
        "subject": fact["subject"],
        "answer": fact["answer"],
        "prompt": prompt,
        "status": "passed" if pass_case else "failed",
        "target": {
            "tokenId": target_token_id,
            "token": target_token,
            "baselineLogit": float(baseline_logits[target_token_id]),
            "baselineRank": target_rank(baseline_logits, target_token_id),
        },
        "promptTokens": [
            {"position": index, "id": int(token_id), "text": live.clean_token(tokenizer, int(token_id))}
            for index, token_id in enumerate(token_ids)
        ],
        "topNextLogits": top_logit_tokens(tokenizer, baseline_logits, args.top_logits),
        "bestCandidate": best,
        "candidateCount": len(candidates),
        "topCandidates": candidates[:12],
        "controls": controls,
        "circuitGraph": circuit_graph,
        "caseMetrics": {
            "bestTargetLogitDrop": best_drop,
            "controlP95PositiveTargetLogitDrop": control_p95,
            "controlMaxPositiveTargetLogitDrop": control_max,
            "marginOverControlP95": margin,
            "liftOverControlP95": best_drop / max(1e-9, control_p95),
            "graphSelectedNodeCount": circuit_graph["selectedNodeCount"],
            "graphJointTargetLogitDrop": circuit_graph["jointAblation"]["targetLogitDrop"],
            "graphControlP95PositiveTargetLogitDrop": circuit_graph["jointAblation"]["controlGraphP95PositiveTargetLogitDrop"],
            "graphMarginOverControlP95": circuit_graph["jointAblation"]["marginOverControlGraphP95"],
            "graphLiftOverControlP95": circuit_graph["jointAblation"]["liftOverControlGraphP95"],
        },
        "gate": {
            "minTargetDrop": args.min_target_drop,
            "minMarginOverControlP95": args.min_margin_over_control_p95,
            "graphGate": circuit_graph["gate"],
            "pass": pass_case,
        },
    }


def make_markdown(report: Dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Live GPT-2 Fact Circuit Finder",
        "",
        "This benchmark uses a real GPT-2 model, real forward-pass hooks, automatic layer/neuron scanning, graph construction, joint graph ablations, and graph-level negative controls. It is not synthetic and does not fit a surrogate.",
        "",
        f"- Pass: `{summary['pass']}`",
        f"- Valid cases: `{summary['validCases']}`",
        f"- Passed cases: `{summary['passedCases']}`",
        f"- Pass rate: `{summary['passRate']}`",
        f"- Median graph joint target-logit drop: `{summary['medianGraphJointTargetLogitDrop']}`",
        f"- Median graph control p95 target-logit drop: `{summary['medianGraphControlP95PositiveTargetLogitDrop']}`",
        f"- Median graph lift over control p95: `{summary['medianGraphLiftOverControlP95']}`",
        f"- Mean selected graph nodes: `{summary['meanGraphSelectedNodeCount']}`",
        f"- Layers scanned: `{report['config']['layers']}`",
        f"- Top neurons per layer: `{report['config']['topNeurons']}`",
        f"- Controls per family: `{report['config']['controlCount']}`",
        f"- Max graph nodes: `{report['config']['maxGraphNodes']}`",
        "",
        "## Claim Boundary",
        "",
        report["claimScope"]["provesIfPasses"],
        "",
        report["claimScope"]["doesNotProve"],
        "",
        "## Cases",
        "",
        "| subject | answer | status | target rank | graph nodes | graph drop | graph control p95 | graph margin | best layer | best neuron |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["cases"]:
        if row.get("status") == "skipped":
            lines.append(f"| {row['subject']} | {row['answer']} | skipped | - | - | - | - | - | - | - |")
            continue
        best = row["bestCandidate"]
        metrics = row["caseMetrics"]
        lines.append(
            f"| {row['subject']} | {row['answer']} | {row['status']} | "
            f"{row['target']['baselineRank']} | "
            f"{metrics['graphSelectedNodeCount']} | "
            f"{metrics['graphJointTargetLogitDrop']:.6f} | "
            f"{metrics['graphControlP95PositiveTargetLogitDrop']:.6f} | "
            f"{metrics['graphMarginOverControlP95']:.6f} | "
            f"{best['layer']} | {best['neuron']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.case_count <= 0 or args.min_valid_cases <= 0:
        raise RuntimeError("case counts must be positive.")
    if args.top_neurons <= 0 or args.control_count <= 0:
        raise RuntimeError("--top-neurons and --control-count must be positive.")
    layers = parse_layers(args.layers)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=args.local_files_only)
    model.eval()
    model.to(device)
    mlps = {layer: live.ensure_gpt2_mlp_shape(model, layer) for layer in layers}
    rng = random.Random(args.control_seed)

    cases = []
    valid_seen = 0
    for fact in FACTS:
        if valid_seen >= args.case_count:
            break
        row = run_case(model=model, tokenizer=tokenizer, mlps=mlps, fact=fact, args=args, rng=rng)
        cases.append(row)
        if row.get("status") != "skipped":
            valid_seen += 1

    valid_cases = [row for row in cases if row.get("status") != "skipped"]
    passed_cases = [row for row in valid_cases if row.get("status") == "passed"]
    failed_cases = [row for row in valid_cases if row.get("status") == "failed"]
    skipped_cases = [row for row in cases if row.get("status") == "skipped"]
    best_drops = [float(row["caseMetrics"]["bestTargetLogitDrop"]) for row in valid_cases]
    control_p95 = [float(row["caseMetrics"]["controlP95PositiveTargetLogitDrop"]) for row in valid_cases]
    lifts = [float(row["caseMetrics"]["liftOverControlP95"]) for row in valid_cases]
    graph_drops = [float(row["caseMetrics"]["graphJointTargetLogitDrop"]) for row in valid_cases]
    graph_control_p95 = [float(row["caseMetrics"]["graphControlP95PositiveTargetLogitDrop"]) for row in valid_cases]
    graph_lifts = [float(row["caseMetrics"]["graphLiftOverControlP95"]) for row in valid_cases]
    graph_node_counts = [int(row["caseMetrics"]["graphSelectedNodeCount"]) for row in valid_cases]
    selected_graph_neurons = []
    layer_histogram: Dict[str, int] = {}
    for row in valid_cases:
        for intervention in row["circuitGraph"]["selectedInterventions"]:
            key = f"{intervention['layer']}:{intervention['neuron']}"
            selected_graph_neurons.append(key)
            layer_key = str(intervention["layer"])
            layer_histogram[layer_key] = layer_histogram.get(layer_key, 0) + 1
    pass_rate = len(passed_cases) / max(1, len(valid_cases))
    pass_gate = len(valid_cases) >= args.min_valid_cases and pass_rate >= args.min_pass_rate

    report = {
        "schemaVersion": 1,
        "mode": "live-gpt2-fact-circuit-finder",
        "synthetic": False,
        "surrogate": False,
        "generatedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "model": {
            "id": args.model,
            "class": model.__class__.__name__,
            "device": device,
            "localFilesOnly": bool(args.local_files_only),
        },
        "runtimeDependencies": {
            "python": sys.executable,
            "torch": torch.__version__,
            "transformers": live.package_version("transformers"),
            "numpy": live.package_version("numpy"),
            "torchao": live.package_version("torchao"),
        },
        "config": {
            "caseCount": args.case_count,
            "minValidCases": args.min_valid_cases,
            "layers": layers,
            "topNeurons": args.top_neurons,
            "topLogits": args.top_logits,
            "controlCount": args.control_count,
            "controlPrompt": args.control_prompt,
            "controlSeed": args.control_seed,
            "maxGraphNodes": args.max_graph_nodes,
            "minPassRate": args.min_pass_rate,
            "minTargetDrop": args.min_target_drop,
            "minMarginOverControlP95": args.min_margin_over_control_p95,
        },
        "summary": {
            "pass": pass_gate,
            "validCases": len(valid_cases),
            "passedCases": len(passed_cases),
            "failedCases": len(failed_cases),
            "skippedCases": len(skipped_cases),
            "passRate": pass_rate,
            "medianBestTargetLogitDrop": percentile(best_drops, 0.50),
            "p10BestTargetLogitDrop": percentile(best_drops, 0.10),
            "medianControlP95PositiveTargetLogitDrop": percentile(control_p95, 0.50),
            "medianLiftOverControlP95": percentile(lifts, 0.50),
            "minLiftOverControlP95": min(lifts) if lifts else 0.0,
            "medianGraphJointTargetLogitDrop": percentile(graph_drops, 0.50),
            "p10GraphJointTargetLogitDrop": percentile(graph_drops, 0.10),
            "medianGraphControlP95PositiveTargetLogitDrop": percentile(graph_control_p95, 0.50),
            "medianGraphLiftOverControlP95": percentile(graph_lifts, 0.50),
            "minGraphLiftOverControlP95": min(graph_lifts) if graph_lifts else 0.0,
            "meanGraphSelectedNodeCount": sum(graph_node_counts) / max(1, len(graph_node_counts)),
            "uniqueSelectedGraphNeurons": len(set(selected_graph_neurons)),
            "selectedGraphLayerHistogram": layer_histogram,
            "failedSubjects": [row["subject"] for row in failed_cases],
            "skippedSubjects": [row["subject"] for row in skipped_cases],
        },
        "claimScope": {
            "provesIfPasses": "Benchmark-scoped evidence that the real GPT-2 live pipeline can automatically scan layers/top activation-contribution neurons, build a token/neuron/target graph, and validate the graph through joint causal ablation above graph-level negative controls across a factual prompt set.",
            "doesNotProve": "This is not attention-head/path recovery yet, not full GPT-2 FFN extraction, not arbitrary LLM understanding, and not output-only original-neuron identity.",
        },
        "cases": cases,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_md)), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    with open(args.output_md, "w", encoding="utf-8") as file:
        file.write(make_markdown(report))

    print(json.dumps({
        "pass": pass_gate,
        "validCases": len(valid_cases),
        "passedCases": len(passed_cases),
        "passRate": pass_rate,
        "reportJson": args.output_json,
        "reportMarkdown": args.output_md,
    }, indent=2))
    if not pass_gate:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
