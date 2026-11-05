#!/usr/bin/env python3

import argparse
import json
import math
import os
import random
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

from llm_reader.adapters.gpt2 import GPT2OracleAdapter


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
    parser.add_argument("--only-subjects", default="")
    parser.add_argument("--layers", default="6,9,11")
    parser.add_argument("--fallback-layers", default="0,1,2,3,4,5,6,7,8,9,10,11")
    parser.add_argument("--top-neurons", type=int, default=50)
    parser.add_argument("--top-logits", type=int, default=8)
    parser.add_argument("--control-count", type=int, default=8)
    parser.add_argument("--control-prompt", default="The weather in Berlin is")
    parser.add_argument("--control-seed", type=int, default=260603)
    parser.add_argument("--max-graph-nodes", type=int, default=5)
    parser.add_argument("--min-pass-rate", type=float, default=0.55)
    parser.add_argument("--min-target-drop", type=float, default=0.05)
    parser.add_argument("--min-margin-over-control-p95", type=float, default=0.02)
    parser.add_argument("--paraphrase-count", type=int, default=2)
    parser.add_argument("--negative-fact-count", type=int, default=2)
    parser.add_argument("--wrong-answer-count", type=int, default=3)
    parser.add_argument("--unrelated-logit-count", type=int, default=8)
    parser.add_argument("--min-paraphrase-pass-rate", type=float, default=0.70)
    parser.add_argument("--min-wrong-answer-specificity-rate", type=float, default=0.85)
    parser.add_argument("--min-negative-fact-specificity-rate", type=float, default=0.85)
    parser.add_argument("--min-unrelated-preservation-rate", type=float, default=0.85)
    parser.add_argument("--max-wrong-answer-drop-ratio", type=float, default=0.55)
    parser.add_argument("--max-candidate-wrong-answer-drop-ratio", type=float, default=0.70)
    parser.add_argument("--allow-target-causal-mlp-fallback", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-negative-fact-drop-ratio", type=float, default=0.55)
    parser.add_argument("--max-unrelated-abs-drop-ratio", type=float, default=0.65)
    parser.add_argument("--strict-rescue-candidate-count", type=int, default=14)
    parser.add_argument("--strict-rescue-pair-candidate-count", type=int, default=12)
    parser.add_argument("--strict-rescue-max-combo-size", type=int, default=2)
    parser.add_argument("--strict-rescue-max-attempts", type=int, default=72)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


live = GPT2OracleAdapter()


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
    position = min(len(ordered) - 1, max(0.0, (len(ordered) - 1) * q))
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    fraction = position - lower
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction)


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


def prompt_for_fact(fact: Dict[str, str]) -> str:
    return fact.get("prompt") or f"The capital of {fact['subject']} is"


def paraphrase_prompts_for_fact(fact: Dict[str, str], count: int) -> List[str]:
    if count <= 0:
        return []
    if "paraphrases" in fact:
        values = fact["paraphrases"]
        if isinstance(values, list):
            return [str(value) for value in values[:count]]
    if "prompt" in fact:
        return []
    subject = fact["subject"]
    candidates = [
        f"The capital city of {subject} is",
        f"In {subject}, the capital is",
        f"In {subject}, the capital city is",
        f"The capital of the country {subject} is",
        f"{subject}'s capital is",
        f"For {subject}, the capital is",
    ]
    return candidates[:count]


def build_answer_pool(tokenizer: Any) -> List[Dict[str, Any]]:
    pool = []
    seen = set()
    for fact in FACTS:
        target = encode_single_answer_token(tokenizer, fact["answer"])
        if target is None:
            continue
        token_id, token = target
        if token_id in seen:
            continue
        seen.add(token_id)
        pool.append({
            "subject": fact["subject"],
            "answer": fact["answer"],
            "tokenId": token_id,
            "token": token,
            "hasPrompt": "prompt" in fact,
        })
    return pool


def choose_wrong_answer_tokens(
    answer_pool: List[Dict[str, Any]],
    target_token_id: int,
    count: int,
) -> List[Dict[str, Any]]:
    if count <= 0:
        return []
    choices = [item for item in answer_pool if int(item["tokenId"]) != int(target_token_id)]
    return choices[:count]


def choose_unrelated_answer_tokens(
    answer_pool: List[Dict[str, Any]],
    fact: Dict[str, str],
    excluded_token_ids: set[int],
    count: int,
) -> List[Dict[str, Any]]:
    if count <= 0:
        return []
    source_has_prompt = "prompt" in fact
    preferred = [
        item
        for item in answer_pool
        if int(item["tokenId"]) not in excluded_token_ids and bool(item.get("hasPrompt")) != source_has_prompt
    ]
    fallback = [
        item
        for item in answer_pool
        if int(item["tokenId"]) not in excluded_token_ids and item not in preferred
    ]
    return (preferred + fallback)[:count]


def choose_negative_facts(fact: Dict[str, str], count: int) -> List[Dict[str, str]]:
    if count <= 0:
        return []
    preferred = [candidate for candidate in FACTS if ("prompt" in candidate) != ("prompt" in fact)]
    fallback = [candidate for candidate in FACTS if candidate not in preferred]
    selected: List[Dict[str, str]] = []
    for candidate in preferred + fallback:
        if candidate["subject"] == fact["subject"]:
            continue
        if candidate["answer"] == fact["answer"]:
            continue
        selected.append(candidate)
        if len(selected) >= count:
            break
    return selected


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


def graph_ablation_logits_for_interventions(
    *,
    model: Any,
    mlps: Dict[int, Any],
    encoded: Dict[str, torch.Tensor],
    interventions: List[Dict[str, Any]],
    read_token_index: int,
) -> torch.Tensor:
    return run_graph_ablation_logits(model, mlps, encoded, interventions, read_token_index)


def translate_interventions_to_token(
    interventions: List[Dict[str, Any]],
    token_index: int,
) -> List[Dict[str, Any]]:
    return [
        {
            "layer": int(item["layer"]),
            "neuron": int(item["neuron"]),
            "ablateTokenPositions": [int(token_index)],
        }
        for item in interventions
    ]


def translate_interventions_preserving_positions(
    interventions: List[Dict[str, Any]],
    seq_len: int,
    fallback_token_index: int,
) -> List[Dict[str, Any]]:
    translated = []
    for item in interventions:
        positions = [
            int(position)
            for position in item.get("ablateTokenPositions", [])
            if 0 <= int(position) < seq_len
        ]
        if not positions:
            positions = [fallback_token_index]
        translated.append({
            "layer": int(item["layer"]),
            "neuron": int(item["neuron"]),
            "ablateTokenPositions": sorted(set(positions)),
        })
    return translated


def translate_interventions_to_active_support(
    interventions: List[Dict[str, Any]],
    captures: Dict[int, Dict[str, torch.Tensor]],
    fallback_token_index: int,
    support_limit: int = 3,
) -> List[Dict[str, Any]]:
    translated = []
    for item in interventions:
        layer = int(item["layer"])
        neuron = int(item["neuron"])
        activation_matrix = captures[layer]["activation"]
        scores = activation_matrix[:, neuron].abs()
        limit = min(max(1, support_limit), int(scores.numel()))
        top = torch.topk(scores.detach().cpu(), k=limit)
        positions = {fallback_token_index}
        positions.update(int(position) for position in top.indices)
        translated.append({
            "layer": layer,
            "neuron": neuron,
            "ablateTokenPositions": sorted(positions),
        })
    return translated


def candidate_ablation_positions(candidate: Dict[str, Any], token_index: int) -> List[int]:
    explicit_positions = candidate.get("ablateTokenPositions")
    if isinstance(explicit_positions, list) and explicit_positions:
        return sorted(set(int(position) for position in explicit_positions))
    positions = {int(token_index)}
    for edge in candidate.get("tokenSupportEdges", []):
        if edge.get("kind") == "token-activation-support":
            positions.add(int(edge["tokenIndex"]))
    return sorted(positions)


def summarize_validation_items(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarise validation items, counting only the controls that actually ran.

    A control whose answer is not a single GPT-2 token is appended with
    ``status="skipped"``, ``pass=True`` and ``targetLogitDrop=0.0`` so the record of the
    attempt survives. This function used to count every item in both ``count`` and
    ``passed``, so a control that never ran raised the specificity pass rate, and its
    placeholder 0.0 drop entered the median, pulling a specificity statistic in the
    flattering direction. Case-level skips are already excluded elsewhere in this file; this
    is the item-level path catching up.

    Skipped items are reported separately rather than dropped, because how many controls
    could not be built is itself worth seeing.
    """
    skipped = [item for item in items if item.get("status") == "skipped"]
    evaluated = [item for item in items if item.get("status") != "skipped"]
    if not evaluated:
        return {
            "count": 0,
            "passed": 0,
            "passRate": 0.0,
            "skippedCount": len(skipped),
            "itemCount": len(items),
        }
    passed = [item for item in evaluated if item.get("pass")]
    target_drops = [float(item.get("targetLogitDrop", 0.0)) for item in evaluated]
    ratios = [float(item.get("ratioToPrimaryGraphDrop", 0.0)) for item in evaluated]
    return {
        "count": len(evaluated),
        "passed": len(passed),
        "passRate": len(passed) / len(evaluated),
        "skippedCount": len(skipped),
        "itemCount": len(items),
        "medianTargetLogitDrop": percentile(target_drops, 0.50),
        "maxTargetLogitDrop": max(target_drops) if target_drops else 0.0,
        "medianRatioToPrimaryGraphDrop": percentile(ratios, 0.50),
        "maxRatioToPrimaryGraphDrop": max(ratios) if ratios else 0.0,
    }


def summarize_unrelated_items(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not items:
        return {"count": 0, "passed": 0, "passRate": 0.0}
    passed = [item for item in items if item.get("pass")]
    ratios = [float(item.get("absDropRatioToPrimaryGraphDrop", 0.0)) for item in items]
    return {
        "count": len(items),
        "passed": len(passed),
        "passRate": len(passed) / len(items),
        "medianAbsDropRatioToPrimaryGraphDrop": percentile(ratios, 0.50),
        "maxAbsDropRatioToPrimaryGraphDrop": max(ratios) if ratios else 0.0,
    }


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
    wrong_answer_targets: List[Dict[str, Any]],
    top_neurons: int,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    lm_head_weight = live.unembedding_weight(model).detach().cpu()
    target_unembed = lm_head_weight[target_token_id]
    wrong_token_ids = [int(item["tokenId"]) for item in wrong_answer_targets]
    wrong_unembeds = lm_head_weight[wrong_token_ids] if wrong_token_ids else None
    for layer, mlp in mlps.items():
        activation_matrix = captures[layer]["activation"]
        activation = activation_matrix[token_index]
        c_proj_weight = mlp.c_proj.weight.detach().cpu()
        projection_norms = torch.linalg.vector_norm(c_proj_weight, dim=1)
        norm_scores = activation.abs() * projection_norms
        target_projection = torch.mv(c_proj_weight, target_unembed)
        signed_target_scores = activation * target_projection
        if wrong_unembeds is not None and int(wrong_unembeds.numel()) > 0:
            wrong_projection = torch.matmul(c_proj_weight, wrong_unembeds.T)
            signed_wrong_scores = activation.unsqueeze(1) * wrong_projection
            max_wrong_scores = torch.max(signed_wrong_scores, dim=1).values
        else:
            max_wrong_scores = torch.zeros_like(signed_target_scores)
        contrast_scores = signed_target_scores - max_wrong_scores
        candidate_indices = set()
        for score_tensor in (norm_scores, signed_target_scores, contrast_scores):
            limit = min(top_neurons, int(score_tensor.numel()))
            if limit <= 0:
                continue
            top = torch.topk(score_tensor.detach().cpu(), k=limit)
            candidate_indices.update(int(neuron) for neuron in top.indices)
        ranked_indices = sorted(
            candidate_indices,
            key=lambda neuron: (
                float(contrast_scores[neuron]),
                float(signed_target_scores[neuron]),
                float(norm_scores[neuron]),
            ),
            reverse=True,
        )
        for rank, neuron in enumerate(ranked_indices, start=1):
            ablated_logits = live.run_activation_ablation_logits(
                model,
                mlp,
                encoded,
                int(neuron),
                [token_index],
                token_index,
            )
            drop = float(baseline_logits[target_token_id] - ablated_logits[target_token_id])
            wrong_drops = []
            for wrong in wrong_answer_targets:
                wrong_token_id = int(wrong["tokenId"])
                wrong_drop = float(baseline_logits[wrong_token_id] - ablated_logits[wrong_token_id])
                wrong_drops.append({
                    "subject": wrong["subject"],
                    "answer": wrong["answer"],
                    "tokenId": wrong_token_id,
                    "token": wrong["token"],
                    "logitDrop": wrong_drop,
                })
            max_wrong_drop = max([float(item["logitDrop"]) for item in wrong_drops] + [0.0])
            target_specific_margin = drop - max_wrong_drop
            records.append({
                "layer": int(layer),
                "rankWithinLayer": rank,
                "neuron": neuron,
                "selectionScore": float(contrast_scores[neuron]),
                "normSelectionScore": float(norm_scores[neuron]),
                "targetProjectionScore": float(signed_target_scores[neuron]),
                "contrastiveTargetScore": float(contrast_scores[neuron]),
                "activation": float(activation[neuron]),
                "targetLogitDrop": drop,
                "ablateTokenPositions": [int(token_index)],
                "wrongAnswerLogitDrops": wrong_drops,
                "maxWrongAnswerLogitDrop": max_wrong_drop,
                "targetSpecificMargin": target_specific_margin,
                "targetSpecificLift": drop / max(1e-9, max_wrong_drop),
                "tokenSupportEdges": token_support_edges(
                    tokenizer=tokenizer,
                    token_ids=token_ids,
                    activation_matrix=activation_matrix,
                    c_proj_weight=c_proj_weight,
                    layer=int(layer),
                    neuron=neuron,
                ),
            })
    return sorted(records, key=lambda item: (item["targetSpecificMargin"], item["targetLogitDrop"]), reverse=True)


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
    min_target_drop: float,
    max_wrong_answer_drop_ratio: float,
) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    used = set()
    ranked = sorted(
        candidates,
        key=lambda item: (
            float(item.get("targetSpecificLift", 0.0)),
            float(item.get("targetSpecificMargin", 0.0)),
            float(item.get("targetLogitDrop", 0.0)),
        ),
        reverse=True,
    )
    for candidate in ranked:
        pair = (int(candidate["layer"]), int(candidate["neuron"]))
        if pair in used:
            continue
        target_drop = float(candidate["targetLogitDrop"])
        wrong_drop = max(0.0, float(candidate.get("maxWrongAnswerLogitDrop", 0.0)))
        wrong_ratio = wrong_drop / max(1e-9, target_drop)
        if (
            target_drop >= min_target_drop
            and float(candidate.get("targetSpecificMargin", 0.0)) > 0
            and wrong_ratio <= max_wrong_answer_drop_ratio
        ):
            selected.append(candidate)
            used.add(pair)
        if len(selected) >= max_graph_nodes:
            break
    return selected


def select_target_causal_candidates(
    candidates: List[Dict[str, Any]],
    max_graph_nodes: int,
    min_target_drop: float,
) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    used = set()
    ranked = sorted(
        candidates,
        key=lambda item: (
            float(item.get("targetLogitDrop", 0.0)),
            float(item.get("targetSpecificMargin", 0.0)),
            float(item.get("targetProjectionScore", 0.0)),
        ),
        reverse=True,
    )
    for candidate in ranked:
        pair = (int(candidate["layer"]), int(candidate["neuron"]))
        if pair in used:
            continue
        if float(candidate.get("targetLogitDrop", 0.0)) < min_target_drop:
            continue
        selected.append(candidate)
        used.add(pair)
        if len(selected) >= max_graph_nodes:
            break
    return selected


def strict_rescue_candidate_pool(
    candidates: List[Dict[str, Any]],
    max_candidates: int,
    min_target_drop: float,
) -> List[Dict[str, Any]]:
    pool: List[Dict[str, Any]] = []
    used = set()
    ranked = sorted(
        candidates,
        key=lambda item: (
            float(item.get("targetSpecificMargin", 0.0)),
            float(item.get("targetLogitDrop", 0.0)),
            float(item.get("targetProjectionScore", 0.0)),
        ),
        reverse=True,
    )
    for candidate in ranked:
        pair = (int(candidate["layer"]), int(candidate["neuron"]))
        if pair in used:
            continue
        if float(candidate.get("targetLogitDrop", 0.0)) < min_target_drop:
            continue
        if float(candidate.get("targetSpecificMargin", 0.0)) <= 0:
            continue
        pool.append(candidate)
        used.add(pair)
        if len(pool) >= max_candidates:
            break
    return pool


def strict_rescue_candidate_sets(
    pool: List[Dict[str, Any]],
    max_graph_nodes: int,
    max_combo_size: int,
    pair_candidate_count: int,
    target_drop_hint: float,
) -> List[List[Dict[str, Any]]]:
    max_size = max(1, min(max_graph_nodes, max_combo_size, len(pool)))
    sets: List[List[Dict[str, Any]]] = []
    singleton_order = sorted(
        pool,
        key=lambda item: (
            abs(float(item.get("targetLogitDrop", 0.0)) - target_drop_hint),
            max(0.0, float(item.get("maxWrongAnswerLogitDrop", 0.0))),
            -float(item.get("targetSpecificMargin", 0.0)),
        ),
    )
    sets.extend([candidate] for candidate in singleton_order)
    for size in range(2, max_size + 1):
        combo_pool = singleton_order[:max(1, min(pair_candidate_count, len(singleton_order)))]
        combo_target = target_drop_hint * size
        combos = sorted(
            combinations(combo_pool, size),
            key=lambda combo: (
                abs(sum(float(candidate.get("targetLogitDrop", 0.0)) for candidate in combo) - combo_target),
                max(float(candidate.get("targetLogitDrop", 0.0)) for candidate in combo),
                -sum(float(candidate.get("targetSpecificMargin", 0.0)) for candidate in combo),
            ),
        )
        sets.extend(list(combo) for combo in combos)
    return sets


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


def interventions_from_candidates(selected: List[Dict[str, Any]], token_index: int) -> List[Dict[str, Any]]:
    return [
        {
            "layer": int(candidate["layer"]),
            "neuron": int(candidate["neuron"]),
            "ablateTokenPositions": candidate_ablation_positions(candidate, token_index),
        }
        for candidate in selected
    ]


def evaluate_selected_graph(
    *,
    model: Any,
    mlps: Dict[int, Any],
    encoded: Dict[str, torch.Tensor],
    baseline_logits: torch.Tensor,
    controls: Dict[str, Any],
    selected: List[Dict[str, Any]],
    token_index: int,
    target_token_id: int,
    min_target_drop: float,
    min_margin_over_control_p95: float,
) -> Dict[str, Any]:
    interventions = interventions_from_candidates(selected, token_index)
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
        and joint_drop >= min_target_drop
        and graph_margin >= min_margin_over_control_p95
        and all(pair["count"] == 0 for pair in controls["crossFamilyNeuronOverlap"])
    )
    return {
        "selected": selected,
        "interventions": interventions,
        "jointDrop": joint_drop,
        "controlGraphs": control_graphs,
        "controlGraphP95": control_graph_p95,
        "graphMargin": graph_margin,
        "pass": pass_graph,
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
    forced_selected: List[Dict[str, Any]] | None = None,
    force_exact_selection: bool = False,
) -> Dict[str, Any]:
    control_p95 = max(group["summary"]["p95PositiveTargetLogitDrop"] for group in controls["groups"])
    threshold = max(float(args.min_target_drop), control_p95 + float(args.min_margin_over_control_p95))
    if forced_selected is None:
        selected_pool = select_circuit_candidates(
            candidates,
            threshold,
            max(1, int(args.max_graph_nodes)),
            float(args.min_target_drop),
            float(args.max_candidate_wrong_answer_drop_ratio),
        )
        selection_source = "ranked-target-specific-candidates"
        if not selected_pool and bool(getattr(args, "allow_target_causal_mlp_fallback", False)):
            selected_pool = select_target_causal_candidates(
                candidates,
                max(1, int(args.max_graph_nodes)),
                float(args.min_target_drop),
            )
            selection_source = "ranked-target-causal-fallback-candidates"
    else:
        selected_pool = forced_selected[:max(1, int(args.max_graph_nodes))]
        selection_source = "forced-candidate-set"
    if not selected_pool:
        return {
            "kind": "gpt2-live-factual-circuit-graph",
            "selectedNodeCount": 0,
            "nodes": [
                {"id": "fact", "kind": "fact-prompt"},
                {"id": f"target:{target_token_id}", "kind": "target-token", "tokenId": target_token_id, "token": target_token},
            ],
            "edges": [
                {"source": "fact", "target": f"target:{target_token_id}", "kind": "expected-answer", "answerToken": target_token}
            ],
            "selectedInterventions": [],
            "selectedCandidates": [],
            "jointAblation": {
                "targetLogitDrop": 0.0,
                "controlGraphP95PositiveTargetLogitDrop": 0.0,
                "controlGraphMaxPositiveTargetLogitDrop": 0.0,
                "marginOverControlGraphP95": 0.0,
                "liftOverControlGraphP95": 0.0,
            },
            "controlGraphs": {"items": [], "summary": summarize_drops([])},
            "gate": {
                "pass": False,
                "failureReason": "no target-specific candidates passed selection threshold",
                "thresholdForIndividualCandidateSelection": threshold,
                "allowTargetCausalMlpFallback": bool(getattr(args, "allow_target_causal_mlp_fallback", False)),
                "minTargetDrop": float(args.min_target_drop),
                "minMarginOverControlP95": float(args.min_margin_over_control_p95),
            },
        }
    graph_attempts = []
    chosen = None
    sizes = [len(selected_pool)] if force_exact_selection else list(range(1, len(selected_pool) + 1))
    for size in sizes:
        attempt = evaluate_selected_graph(
            model=model,
            mlps=mlps,
            encoded=encoded,
            baseline_logits=baseline_logits,
            controls=controls,
            selected=selected_pool[:size],
            token_index=token_index,
            target_token_id=target_token_id,
            min_target_drop=float(args.min_target_drop),
            min_margin_over_control_p95=float(args.min_margin_over_control_p95),
        )
        graph_attempts.append({
            "nodeCount": size,
            "targetLogitDrop": float(attempt["jointDrop"]),
            "controlGraphP95PositiveTargetLogitDrop": float(attempt["controlGraphP95"]),
            "marginOverControlGraphP95": float(attempt["graphMargin"]),
            "pass": bool(attempt["pass"]),
        })
        if bool(attempt["pass"]):
            chosen = attempt
            break
    if chosen is None:
        chosen = max(
            (
                evaluate_selected_graph(
                    model=model,
                    mlps=mlps,
                    encoded=encoded,
                    baseline_logits=baseline_logits,
                    controls=controls,
                    selected=selected_pool[:size],
                    token_index=token_index,
                    target_token_id=target_token_id,
                    min_target_drop=float(args.min_target_drop),
                    min_margin_over_control_p95=float(args.min_margin_over_control_p95),
                )
                for size in sizes
            ),
            key=lambda item: (float(item["graphMargin"]), float(item["jointDrop"])),
        )
    selected = chosen["selected"]
    interventions = chosen["interventions"]
    joint_drop = float(chosen["jointDrop"])
    control_graphs = chosen["controlGraphs"]
    control_graph_p95 = float(chosen["controlGraphP95"])
    graph_margin = float(chosen["graphMargin"])
    pass_graph = bool(chosen["pass"])
    selection_mode = (
        "forced-exact"
        if force_exact_selection
        else ("minimal-passing-prefix" if pass_graph else "best-margin-prefix")
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
        "graphSelection": {
            "mode": selection_mode,
            "source": selection_source,
            "candidatePoolNodeCount": len(selected_pool),
            "attempts": graph_attempts,
        },
        "gate": {
            "pass": pass_graph,
            "thresholdForIndividualCandidateSelection": threshold,
            "minTargetDrop": float(args.min_target_drop),
            "minMarginOverControlP95": float(args.min_margin_over_control_p95),
        },
    }


def run_graph_validation_prompt(
    *,
    model: Any,
    tokenizer: Any,
    mlps: Dict[int, Any],
    prompt: str,
    target_token_id: int,
    target_token: str,
    selected_interventions: List[Dict[str, Any]],
    primary_graph_drop: float,
    min_target_drop: float,
    min_drop_ratio: float,
    device: torch.device,
    kind: str,
    metadata: Dict[str, Any],
    positionMode: str = "last-token",
) -> Dict[str, Any]:
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    token_index = int(encoded["input_ids"].shape[1]) - 1
    token_ids = encoded["input_ids"][0].detach().cpu()
    if positionMode == "active-support":
        outputs, captures = run_with_multi_layer_captures(model, mlps, encoded)
    else:
        captures = {}
        with torch.no_grad():
            outputs = model(**encoded)
    baseline_logits = outputs.logits.detach().cpu()[0, token_index]
    if positionMode == "preserve-valid-positions":
        translated = translate_interventions_preserving_positions(
            selected_interventions,
            int(encoded["input_ids"].shape[1]),
            token_index,
        )
    elif positionMode == "active-support":
        translated = translate_interventions_to_active_support(
            selected_interventions,
            captures,
            token_index,
        )
    else:
        translated = translate_interventions_to_token(selected_interventions, token_index)
    ablated_logits = graph_ablation_logits_for_interventions(
        model=model,
        mlps=mlps,
        encoded=encoded,
        interventions=translated,
        read_token_index=token_index,
    )
    drop = float(baseline_logits[target_token_id] - ablated_logits[target_token_id])
    ratio = drop / max(1e-9, primary_graph_drop)
    return {
        "kind": kind,
        **metadata,
        "prompt": prompt,
        "tokenIndex": token_index,
        "promptTokens": [
            {"position": index, "id": int(token_id), "text": live.clean_token(tokenizer, int(token_id))}
            for index, token_id in enumerate(token_ids)
        ],
        "targetTokenId": target_token_id,
        "targetToken": target_token,
        "targetBaselineLogit": float(baseline_logits[target_token_id]),
        "targetBaselineRank": target_rank(baseline_logits, target_token_id),
        "targetLogitDrop": drop,
        "ratioToPrimaryGraphDrop": ratio,
        "minTargetDrop": min_target_drop,
        "minDropRatioToPrimary": min_drop_ratio,
        "positionMode": positionMode,
        "pass": drop >= min_target_drop and ratio >= min_drop_ratio,
        "translatedInterventions": translated,
    }


def build_hardening_validations(
    *,
    model: Any,
    tokenizer: Any,
    mlps: Dict[int, Any],
    fact: Dict[str, str],
    answer_pool: List[Dict[str, Any]],
    encoded: Dict[str, torch.Tensor],
    baseline_logits: torch.Tensor,
    token_index: int,
    target_token_id: int,
    target_token: str,
    selected_interventions: List[Dict[str, Any]],
    primary_graph_drop: float,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    device = encoded["input_ids"].device
    primary_ablated_logits = graph_ablation_logits_for_interventions(
        model=model,
        mlps=mlps,
        encoded=encoded,
        interventions=selected_interventions,
        read_token_index=token_index,
    )
    wrong_answers = []
    for wrong in choose_wrong_answer_tokens(answer_pool, target_token_id, int(args.wrong_answer_count)):
        drop = float(baseline_logits[int(wrong["tokenId"])] - primary_ablated_logits[int(wrong["tokenId"])])
        ratio = drop / max(1e-9, primary_graph_drop)
        wrong_answers.append({
            "kind": "wrong-answer-target-token",
            "subject": wrong["subject"],
            "answer": wrong["answer"],
            "tokenId": int(wrong["tokenId"]),
            "token": wrong["token"],
            "baselineLogit": float(baseline_logits[int(wrong["tokenId"])]),
            "logitDrop": drop,
            "targetLogitDrop": drop,
            "ratioToPrimaryGraphDrop": ratio,
            "maxAllowedRatioToPrimary": float(args.max_wrong_answer_drop_ratio),
            "pass": ratio <= float(args.max_wrong_answer_drop_ratio),
        })

    excluded = {int(target_token_id)}
    excluded.update(int(item["tokenId"]) for item in wrong_answers)
    unrelated = []
    for unrelated_answer in choose_unrelated_answer_tokens(answer_pool, fact, excluded, int(args.unrelated_logit_count)):
        token_id = int(unrelated_answer["tokenId"])
        baseline = float(baseline_logits[token_id])
        ablated = float(primary_ablated_logits[token_id])
        drop = baseline - ablated
        ratio = abs(drop) / max(1e-9, primary_graph_drop)
        unrelated.append({
            "kind": "unrelated-factual-answer-logit-preservation",
            "subject": unrelated_answer["subject"],
            "answer": unrelated_answer["answer"],
            "tokenId": token_id,
            "token": unrelated_answer["token"],
            "baselineLogit": baseline,
            "ablatedLogit": ablated,
            "logitDrop": drop,
            "absDropRatioToPrimaryGraphDrop": ratio,
            "maxAllowedAbsDropRatioToPrimary": float(args.max_unrelated_abs_drop_ratio),
            "pass": ratio <= float(args.max_unrelated_abs_drop_ratio),
        })

    paraphrase_min_ratio = 0.35
    paraphrases = [
        run_graph_validation_prompt(
            model=model,
            tokenizer=tokenizer,
            mlps=mlps,
            prompt=prompt,
            target_token_id=target_token_id,
            target_token=target_token,
            selected_interventions=selected_interventions,
            primary_graph_drop=primary_graph_drop,
            min_target_drop=float(args.min_target_drop),
            min_drop_ratio=paraphrase_min_ratio,
            device=device,
            kind="paraphrase-same-fact",
            metadata={"subject": fact["subject"], "answer": fact["answer"]},
            positionMode="active-support",
        )
        for prompt in paraphrase_prompts_for_fact(fact, int(args.paraphrase_count))
    ]

    negative_facts = []
    for negative in choose_negative_facts(fact, int(args.negative_fact_count)):
        negative_target = encode_single_answer_token(tokenizer, negative["answer"])
        if negative_target is None:
            negative_facts.append({
                "kind": "negative-fact-correct-answer-token",
                "sourceSubject": fact["subject"],
                "negativeSubject": negative["subject"],
                "negativeAnswer": negative["answer"],
                "status": "skipped",
                "skipReason": "negative fact answer is not one GPT-2 token with a leading space",
                "pass": True,
                "targetLogitDrop": 0.0,
                "ratioToPrimaryGraphDrop": 0.0,
            })
            continue
        negative_token_id, negative_token = negative_target
        prompt = prompt_for_fact(negative)
        item = run_graph_validation_prompt(
            model=model,
            tokenizer=tokenizer,
            mlps=mlps,
            prompt=prompt,
            target_token_id=negative_token_id,
            target_token=negative_token,
            selected_interventions=selected_interventions,
            primary_graph_drop=primary_graph_drop,
            min_target_drop=-1e9,
            min_drop_ratio=-1e9,
            device=device,
            kind="negative-fact-correct-answer-token",
            metadata={
                "sourceSubject": fact["subject"],
                "negativeSubject": negative["subject"],
                "negativeAnswer": negative["answer"],
                "originalAnswer": fact["answer"],
            },
            positionMode="preserve-valid-positions",
        )
        ratio = float(item["targetLogitDrop"]) / max(1e-9, primary_graph_drop)
        item["ratioToPrimaryGraphDrop"] = ratio
        item["maxAllowedRatioToPrimary"] = float(args.max_negative_fact_drop_ratio)
        item["pass"] = ratio <= float(args.max_negative_fact_drop_ratio)
        negative_facts.append(item)

    paraphrase_summary = summarize_validation_items(paraphrases)
    wrong_summary = summarize_validation_items(wrong_answers)
    negative_summary = summarize_validation_items(negative_facts)
    unrelated_summary = summarize_unrelated_items(unrelated)
    pass_hardening = (
        (paraphrase_summary["count"] == 0 or paraphrase_summary["passRate"] >= float(args.min_paraphrase_pass_rate))
        and (wrong_summary["count"] == 0 or wrong_summary["passRate"] >= float(args.min_wrong_answer_specificity_rate))
        and (negative_summary["count"] == 0 or negative_summary["passRate"] >= float(args.min_negative_fact_specificity_rate))
        and (unrelated_summary["count"] == 0 or unrelated_summary["passRate"] >= float(args.min_unrelated_preservation_rate))
    )
    return {
        "pass": pass_hardening,
        "thresholds": {
            "minParaphrasePassRate": float(args.min_paraphrase_pass_rate),
            "minWrongAnswerSpecificityRate": float(args.min_wrong_answer_specificity_rate),
            "minNegativeFactSpecificityRate": float(args.min_negative_fact_specificity_rate),
            "minUnrelatedPreservationRate": float(args.min_unrelated_preservation_rate),
            "maxWrongAnswerDropRatio": float(args.max_wrong_answer_drop_ratio),
            "maxCandidateWrongAnswerDropRatio": float(args.max_candidate_wrong_answer_drop_ratio),
            "maxNegativeFactDropRatio": float(args.max_negative_fact_drop_ratio),
            "maxUnrelatedAbsDropRatio": float(args.max_unrelated_abs_drop_ratio),
            "paraphraseMinDropRatio": paraphrase_min_ratio,
        },
        "paraphrases": {
            "summary": paraphrase_summary,
            "items": paraphrases,
        },
        "wrongAnswerTargets": {
            "summary": wrong_summary,
            "items": wrong_answers,
        },
        "negativeFacts": {
            "summary": negative_summary,
            "items": negative_facts,
        },
        "unrelatedLogits": {
            "summary": unrelated_summary,
            "items": unrelated,
        },
    }


def run_case(
    *,
    model: Any,
    tokenizer: Any,
    mlps: Dict[int, Any],
    fact: Dict[str, str],
    answer_pool: List[Dict[str, Any]],
    args: argparse.Namespace,
    rng: random.Random,
    prompt: str | None = None,
) -> Dict[str, Any]:
    prompt = prompt if prompt is not None else prompt_for_fact(fact)
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
    wrong_answer_targets = choose_wrong_answer_tokens(answer_pool, target_token_id, int(args.wrong_answer_count))
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
        wrong_answer_targets=wrong_answer_targets,
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

    def validate_graph(graph: Dict[str, Any]) -> Dict[str, Any]:
        if graph["selectedInterventions"]:
            return build_hardening_validations(
                model=model,
                tokenizer=tokenizer,
                mlps=mlps,
                fact=fact,
                answer_pool=answer_pool,
                encoded=encoded,
                baseline_logits=baseline_logits,
                token_index=token_index,
                target_token_id=target_token_id,
                target_token=target_token,
                selected_interventions=graph["selectedInterventions"],
                primary_graph_drop=float(graph["jointAblation"]["targetLogitDrop"]),
                args=args,
            )
        return {
            "pass": False,
            "failureReason": "no selected graph interventions to validate",
            "thresholds": {},
            "paraphrases": {"summary": summarize_validation_items([]), "items": []},
            "wrongAnswerTargets": {"summary": summarize_validation_items([]), "items": []},
            "negativeFacts": {"summary": summarize_validation_items([]), "items": []},
            "unrelatedLogits": {"summary": summarize_unrelated_items([]), "items": []},
        }

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
    hardening = validate_graph(circuit_graph)
    rescue_attempts = []
    if not (bool(circuit_graph["gate"]["pass"]) and bool(hardening["pass"])):
        rescue_target_hint = max(float(args.min_target_drop), control_p95 + float(args.min_margin_over_control_p95))
        rescue_pool = strict_rescue_candidate_pool(
            candidates,
            max(1, int(args.strict_rescue_candidate_count)),
            float(args.min_target_drop),
        )
        rescue_sets = strict_rescue_candidate_sets(
            rescue_pool,
            max(1, int(args.max_graph_nodes)),
            max(1, int(args.strict_rescue_max_combo_size)),
            max(1, int(args.strict_rescue_pair_candidate_count)),
            rescue_target_hint,
        )
        for candidate_set in rescue_sets:
            if len(rescue_attempts) >= max(1, int(args.strict_rescue_max_attempts)):
                break
            rescue_graph = build_circuit_graph(
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
                forced_selected=candidate_set,
                force_exact_selection=True,
            )
            rescue_hardening = validate_graph(rescue_graph) if bool(rescue_graph["gate"]["pass"]) else {"pass": False}
            rescue_pass = bool(rescue_graph["gate"]["pass"]) and bool(rescue_hardening["pass"])
            rescue_attempts.append({
                "nodeCount": len(candidate_set),
                "nodes": [
                    {"layer": int(candidate["layer"]), "neuron": int(candidate["neuron"])}
                    for candidate in candidate_set
                ],
                "graphPass": bool(rescue_graph["gate"]["pass"]),
                "hardeningPass": bool(rescue_hardening["pass"]),
                "pass": rescue_pass,
                "graphDrop": float(rescue_graph["jointAblation"]["targetLogitDrop"]),
                "graphControlP95": float(rescue_graph["jointAblation"]["controlGraphP95PositiveTargetLogitDrop"]),
            })
            if rescue_pass:
                circuit_graph = rescue_graph
                hardening = rescue_hardening
                break
    circuit_graph["strictRescue"] = {
        "attempted": len(rescue_attempts) > 0,
        "attempts": rescue_attempts,
        "candidateCount": int(args.strict_rescue_candidate_count),
        "pairCandidateCount": int(args.strict_rescue_pair_candidate_count),
        "maxComboSize": int(args.strict_rescue_max_combo_size),
        "maxAttempts": int(args.strict_rescue_max_attempts),
        "selectedAfterRescue": bool(circuit_graph["gate"]["pass"]) and bool(hardening["pass"]),
    }
    pass_case = bool(circuit_graph["gate"]["pass"]) and bool(hardening["pass"])
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
        "orderedRescueCandidates": candidates,
        "controls": controls,
        "circuitGraph": circuit_graph,
        "hardening": hardening,
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
            "hardeningGate": {
                "pass": hardening["pass"],
                "thresholds": hardening["thresholds"],
            },
            "hardeningCasePass": bool(hardening["pass"]),
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
        f"- Strict case definition: `{summary['strictCaseDefinition']}`",
        f"- Passed cases: `{summary['passedCases']}`",
        f"- Graph-passed cases: `{summary['graphPassedCases']}`",
        f"- Hardening-passed cases: `{summary['hardeningPassedCases']}`",
        f"- Pass rate: `{summary['passRate']}`",
        f"- Median graph joint target-logit drop: `{summary['medianGraphJointTargetLogitDrop']}`",
        f"- Median graph control p95 target-logit drop: `{summary['medianGraphControlP95PositiveTargetLogitDrop']}`",
        f"- Median graph lift over control p95: `{summary['medianGraphLiftOverControlP95']}`",
        f"- Mean selected graph nodes: `{summary['meanGraphSelectedNodeCount']}`",
        f"- Paraphrase validation pass rate: `{summary['hardening']['paraphrasePassRate']}`",
        f"- Wrong-answer specificity pass rate: `{summary['hardening']['wrongAnswerSpecificityPassRate']}`",
        f"- Negative-fact specificity pass rate: `{summary['hardening']['negativeFactSpecificityPassRate']}`",
        f"- Unrelated-logit preservation pass rate: `{summary['hardening']['unrelatedLogitPreservationPassRate']}`",
        f"- Primary layers scanned: `{report['config']['layers']}`",
        f"- Fallback layers scanned on strict failures: `{report['config']['fallbackLayers']}`",
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
        "| subject | answer | status | target rank | graph nodes | graph drop | graph control p95 | graph margin | hardening | best layer | best neuron |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["cases"]:
        if row.get("status") == "skipped":
            lines.append(f"| {row['subject']} | {row['answer']} | skipped | - | - | - | - | - | - | - | - |")
            continue
        best = row["bestCandidate"]
        metrics = row["caseMetrics"]
        hardening = row["hardening"]
        lines.append(
            f"| {row['subject']} | {row['answer']} | {row['status']} | "
            f"{row['target']['baselineRank']} | "
            f"{metrics['graphSelectedNodeCount']} | "
            f"{metrics['graphJointTargetLogitDrop']:.6f} | "
            f"{metrics['graphControlP95PositiveTargetLogitDrop']:.6f} | "
            f"{metrics['graphMarginOverControlP95']:.6f} | "
            f"{hardening['pass']} | "
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
    for name in ("paraphrase_count", "negative_fact_count", "wrong_answer_count", "unrelated_logit_count"):
        if getattr(args, name) < 0:
            raise RuntimeError(f"--{name.replace('_', '-')} must be non-negative.")
    only_subjects = {item.strip() for item in args.only_subjects.split(",") if item.strip()}
    layers = parse_layers(args.layers)
    fallback_layers = parse_layers(args.fallback_layers) if args.fallback_layers.strip() else []
    use_fallback_layers = bool(fallback_layers) and set(fallback_layers) != set(layers)
    all_required_layers = sorted(set(layers) | set(fallback_layers))
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")

    tokenizer, model = live.load_model_and_tokenizer(args.model, local_files_only=args.local_files_only, device=device)
    all_mlps = live.mlp_modules(model, all_required_layers)
    primary_mlps = {layer: all_mlps[layer] for layer in layers}
    fallback_mlps = {layer: all_mlps[layer] for layer in fallback_layers} if use_fallback_layers else {}
    rng = random.Random(args.control_seed)
    answer_pool = build_answer_pool(tokenizer)
    if len(answer_pool) < max(1, args.wrong_answer_count):
        raise RuntimeError("Not enough single-token answer targets for wrong-answer controls.")

    cases = []
    valid_seen = 0
    for fact in FACTS:
        if only_subjects and fact["subject"] not in only_subjects:
            continue
        if valid_seen >= args.case_count:
            break
        row = run_case(model=model, tokenizer=tokenizer, mlps=primary_mlps, fact=fact, answer_pool=answer_pool, args=args, rng=rng)
        expansion = {
            "attempted": False,
            "primaryLayers": layers,
            "fallbackLayers": fallback_layers if use_fallback_layers else [],
            "resolved": row.get("status") == "passed",
        }
        if use_fallback_layers and row.get("status") == "failed":
            primary_summary = {
                "status": row.get("status"),
                "graphPass": bool(row.get("gate", {}).get("graphGate", {}).get("pass", False)),
                "hardeningPass": bool(row.get("gate", {}).get("hardeningGate", {}).get("pass", False)),
                "graphDrop": row.get("caseMetrics", {}).get("graphJointTargetLogitDrop", 0.0),
                "graphControlP95": row.get("caseMetrics", {}).get("graphControlP95PositiveTargetLogitDrop", 0.0),
                "selectedNodes": [
                    {"layer": int(item["layer"]), "neuron": int(item["neuron"])}
                    for item in row.get("circuitGraph", {}).get("selectedCandidates", [])
                ],
            }
            fallback_row = run_case(
                model=model,
                tokenizer=tokenizer,
                mlps=fallback_mlps,
                fact=fact,
                answer_pool=answer_pool,
                args=args,
                rng=rng,
            )
            expansion = {
                "attempted": True,
                "primaryLayers": layers,
                "fallbackLayers": fallback_layers,
                "primary": primary_summary,
                "resolved": fallback_row.get("status") == "passed",
            }
            row = fallback_row
        row["adaptiveLayerExpansion"] = expansion
        cases.append(row)
        if row.get("status") != "skipped":
            valid_seen += 1
            graph_pass = bool(row.get("gate", {}).get("graphGate", {}).get("pass", False))
            hardening_pass = bool(row.get("gate", {}).get("hardeningGate", {}).get("pass", False))
            graph_drop = row.get("caseMetrics", {}).get("graphJointTargetLogitDrop", 0.0)
            rescue = row.get("circuitGraph", {}).get("strictRescue", {})
            print(
                json.dumps({
                    "case": valid_seen,
                    "subject": row.get("subject"),
                    "status": row.get("status"),
                    "graphPass": graph_pass,
                    "hardeningPass": hardening_pass,
                    "graphDrop": graph_drop,
                    "rescueAttempts": len(rescue.get("attempts", [])),
                    "adaptiveLayerExpansion": bool(row.get("adaptiveLayerExpansion", {}).get("attempted", False)),
                }),
                flush=True,
            )

    valid_cases = [row for row in cases if row.get("status") != "skipped"]
    passed_cases = [row for row in valid_cases if row.get("status") == "passed"]
    failed_cases = [row for row in valid_cases if row.get("status") == "failed"]
    skipped_cases = [row for row in cases if row.get("status") == "skipped"]
    graph_passed_cases = [row for row in valid_cases if bool(row.get("gate", {}).get("graphGate", {}).get("pass", False))]
    hardening_passed_cases = [row for row in valid_cases if bool(row.get("gate", {}).get("hardeningGate", {}).get("pass", False))]
    best_drops = [float(row["caseMetrics"]["bestTargetLogitDrop"]) for row in valid_cases]
    control_p95 = [float(row["caseMetrics"]["controlP95PositiveTargetLogitDrop"]) for row in valid_cases]
    lifts = [float(row["caseMetrics"]["liftOverControlP95"]) for row in valid_cases]
    graph_drops = [float(row["caseMetrics"]["graphJointTargetLogitDrop"]) for row in valid_cases]
    graph_control_p95 = [float(row["caseMetrics"]["graphControlP95PositiveTargetLogitDrop"]) for row in valid_cases]
    graph_lifts = [float(row["caseMetrics"]["graphLiftOverControlP95"]) for row in valid_cases]
    graph_node_counts = [int(row["caseMetrics"]["graphSelectedNodeCount"]) for row in valid_cases]
    paraphrase_items = [
        item
        for row in valid_cases
        for item in row["hardening"]["paraphrases"]["items"]
    ]
    wrong_answer_items = [
        item
        for row in valid_cases
        for item in row["hardening"]["wrongAnswerTargets"]["items"]
    ]
    negative_fact_items = [
        item
        for row in valid_cases
        for item in row["hardening"]["negativeFacts"]["items"]
    ]
    unrelated_items = [
        item
        for row in valid_cases
        for item in row["hardening"]["unrelatedLogits"]["items"]
    ]
    paraphrase_summary = summarize_validation_items(paraphrase_items)
    wrong_answer_summary = summarize_validation_items(wrong_answer_items)
    negative_fact_summary = summarize_validation_items(negative_fact_items)
    unrelated_summary = summarize_unrelated_items(unrelated_items)
    hardening_failed_cases = [
        row["subject"]
        for row in valid_cases
        if not bool(row.get("hardening", {}).get("pass", False))
    ]
    selected_graph_neurons = []
    layer_histogram: Dict[str, int] = {}
    for row in valid_cases:
        for intervention in row["circuitGraph"]["selectedInterventions"]:
            key = f"{intervention['layer']}:{intervention['neuron']}"
            selected_graph_neurons.append(key)
            layer_key = str(intervention["layer"])
            layer_histogram[layer_key] = layer_histogram.get(layer_key, 0) + 1
    pass_rate = len(passed_cases) / max(1, len(valid_cases))
    hardening_gate = (
        (paraphrase_summary["count"] == 0 or paraphrase_summary["passRate"] >= float(args.min_paraphrase_pass_rate))
        and (wrong_answer_summary["count"] == 0 or wrong_answer_summary["passRate"] >= float(args.min_wrong_answer_specificity_rate))
        and (negative_fact_summary["count"] == 0 or negative_fact_summary["passRate"] >= float(args.min_negative_fact_specificity_rate))
        and (unrelated_summary["count"] == 0 or unrelated_summary["passRate"] >= float(args.min_unrelated_preservation_rate))
    )
    pass_gate = len(valid_cases) >= args.min_valid_cases and pass_rate >= args.min_pass_rate and hardening_gate

    report = {
        "schemaVersion": 1,
        "mode": "live-gpt2-fact-circuit-finder",
        "synthetic": False,
        "surrogate": False,
        "generatedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "model": live.model_identity(model, args.model, device, bool(args.local_files_only)),
        "oracleAdapter": {
            "id": live.adapter_id,
            "activationFamily": live.activation_family,
            "contract": "LLMOracleAdapter",
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
            "onlySubjects": sorted(only_subjects),
            "layers": layers,
            "fallbackLayers": fallback_layers if use_fallback_layers else [],
            "topNeurons": args.top_neurons,
            "topLogits": args.top_logits,
            "controlCount": args.control_count,
            "controlPrompt": args.control_prompt,
            "controlSeed": args.control_seed,
            "maxGraphNodes": args.max_graph_nodes,
            "minPassRate": args.min_pass_rate,
            "minTargetDrop": args.min_target_drop,
            "minMarginOverControlP95": args.min_margin_over_control_p95,
            "paraphraseCount": args.paraphrase_count,
            "negativeFactCount": args.negative_fact_count,
            "wrongAnswerCount": args.wrong_answer_count,
            "unrelatedLogitCount": args.unrelated_logit_count,
            "minParaphrasePassRate": args.min_paraphrase_pass_rate,
            "minWrongAnswerSpecificityRate": args.min_wrong_answer_specificity_rate,
            "minNegativeFactSpecificityRate": args.min_negative_fact_specificity_rate,
            "minUnrelatedPreservationRate": args.min_unrelated_preservation_rate,
            "maxWrongAnswerDropRatio": args.max_wrong_answer_drop_ratio,
            "maxCandidateWrongAnswerDropRatio": args.max_candidate_wrong_answer_drop_ratio,
            "maxNegativeFactDropRatio": args.max_negative_fact_drop_ratio,
            "maxUnrelatedAbsDropRatio": args.max_unrelated_abs_drop_ratio,
            "strictRescueCandidateCount": args.strict_rescue_candidate_count,
            "strictRescuePairCandidateCount": args.strict_rescue_pair_candidate_count,
            "strictRescueMaxComboSize": args.strict_rescue_max_combo_size,
            "strictRescueMaxAttempts": args.strict_rescue_max_attempts,
        },
        "summary": {
            "pass": pass_gate,
            "validCases": len(valid_cases),
            "passedCases": len(passed_cases),
            "graphPassedCases": len(graph_passed_cases),
            "hardeningPassedCases": len(hardening_passed_cases),
            "strictCaseDefinition": "graphGate.pass && hardeningGate.pass",
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
            "hardening": {
                "pass": hardening_gate,
                "paraphrasePassRate": paraphrase_summary["passRate"],
                "paraphraseCount": paraphrase_summary["count"],
                "wrongAnswerSpecificityPassRate": wrong_answer_summary["passRate"],
                "wrongAnswerCount": wrong_answer_summary["count"],
                "negativeFactSpecificityPassRate": negative_fact_summary["passRate"],
                "negativeFactCount": negative_fact_summary["count"],
                "unrelatedLogitPreservationPassRate": unrelated_summary["passRate"],
                "unrelatedLogitCount": unrelated_summary["count"],
                "paraphraseSummary": paraphrase_summary,
                "wrongAnswerSummary": wrong_answer_summary,
                "negativeFactSummary": negative_fact_summary,
                "unrelatedLogitSummary": unrelated_summary,
                "failedSubjects": hardening_failed_cases,
            },
            "failedSubjects": [row["subject"] for row in failed_cases],
            "skippedSubjects": [row["subject"] for row in skipped_cases],
        },
        "claimScope": {
            "provesIfPasses": "Benchmark-scoped evidence that the real GPT-2 live pipeline can automatically scan layers/top activation-contribution neurons, build a token/neuron/target graph, validate the graph through joint causal ablation above graph-level controls, and harden it against paraphrase, wrong-answer, negative-fact, and unrelated-logit checks across a factual prompt set.",
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
