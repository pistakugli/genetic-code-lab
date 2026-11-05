#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

from llm_reader.adapters.registry import adapter_choices, create_adapter


def load_tool_module(filename: str, module_name: str) -> Any:
    source_path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load helper module from {source_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


factual = load_tool_module("llm-reader-gpt2-factual-sweep.py", "gcl_factual_sweep")
attention = load_tool_module("llm-reader-gpt2-attention-path-sweep.py", "gcl_attention_sweep")
combined = load_tool_module("llm-reader-gpt2-combined-circuit-sweep.py", "gcl_combined_sweep")
live = create_adapter("gpt2")


def set_live_adapter(adapter_id: str) -> None:
    global live
    live = create_adapter(adapter_id)
    factual.live = live
    attention.live = live
    attention.factual.live = live
    combined.live = live
    combined.factual.live = live
    combined.attention.live = live
    combined.attention.factual.live = live


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a real LLM ordered residual attention->MLP->target path-patching sweep."
    )
    parser.add_argument("--adapter", choices=adapter_choices(), default="gpt2")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--case-count", type=int, default=50)
    parser.add_argument("--min-valid-cases", type=int, default=50)
    parser.add_argument("--only-subjects", default="")
    parser.add_argument("--mlp-layers", default="6,9,11")
    parser.add_argument("--mlp-fallback-layers", default="0,1,2,3,4,5,6,7,8,9,10,11")
    parser.add_argument("--attention-layers", default="6,9,11")
    parser.add_argument("--attention-fallback-layers", default="0,1,2,3,4,5,6,7,8,9,10,11")
    parser.add_argument("--top-neurons", type=int, default=50)
    parser.add_argument("--top-logits", type=int, default=8)
    parser.add_argument("--control-count", type=int, default=8)
    parser.add_argument("--control-prompt", default="The weather in Berlin is")
    parser.add_argument("--control-seed", type=int, default=260607)
    parser.add_argument("--max-graph-nodes", type=int, default=5)
    parser.add_argument("--max-graph-heads", type=int, default=3)
    parser.add_argument("--min-pass-rate", type=float, default=0.90)
    parser.add_argument("--min-component-pass-rate", type=float, default=0.90)
    parser.add_argument("--min-target-drop", type=float, default=0.05)
    parser.add_argument("--min-margin-over-control-p95", type=float, default=0.02)
    parser.add_argument("--min-path-activation-delta", type=float, default=0.005)
    parser.add_argument("--min-path-margin-over-control-p95", type=float, default=0.001)
    parser.add_argument("--require-component-hardening", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ordered-rescue-candidate-count", type=int, default=10)
    parser.add_argument("--ordered-rescue-max-pair-attempts", type=int, default=32)
    parser.add_argument("--ordered-rescue-set-candidate-count", type=int, default=10)
    parser.add_argument("--ordered-rescue-max-set-attempts", type=int, default=64)
    parser.add_argument("--wrong-answer-count", type=int, default=3)
    parser.add_argument("--attention-max-target-rank", type=int, default=100)
    parser.add_argument("--paraphrase-count", type=int, default=2)
    parser.add_argument("--negative-fact-count", type=int, default=2)
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


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stable_subject_seed(base_seed: int, subject: str) -> int:
    digest = hashlib.sha256(f"{int(base_seed)}:{subject}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def namespace_for_combined(args: argparse.Namespace) -> argparse.Namespace:
    values = vars(args).copy()
    values["control_seed"] = int(args.control_seed)
    return argparse.Namespace(**values)


def encode_prompt(tokenizer: Any, prompt: str, device: torch.device | str) -> Dict[str, torch.Tensor]:
    encoded = tokenizer(prompt, return_tensors="pt")
    return {key: value.to(device) for key, value in encoded.items()}


def group_attention_interventions(
    interventions: List[Dict[str, Any]],
    attentions: Dict[int, Any],
    seq_len: int,
) -> Dict[int, Dict[int, List[int]]]:
    grouped: Dict[int, Dict[int, List[int]]] = {}
    for intervention in interventions:
        layer = int(intervention["layer"])
        head = int(intervention["head"])
        positions = sorted(set(int(position) for position in intervention.get("ablateTokenPositions", [])))
        valid_positions = [position for position in positions if 0 <= position < seq_len]
        if not valid_positions:
            continue
        if layer not in attentions:
            raise RuntimeError(f"Ordered path references unavailable attention layer {layer}.")
        if head < 0 or head >= live.attention_head_count(attentions[layer]):
            raise RuntimeError(f"Ordered path references invalid head {head} for layer {layer}.")
        grouped.setdefault(layer, {}).setdefault(head, [])
        grouped[layer][head] = sorted(set(grouped[layer][head] + valid_positions))
    return grouped


def run_mlp_captures_with_attention_patch(
    *,
    model: Any,
    mlps: Dict[int, Any],
    attentions: Dict[int, Any],
    encoded: Dict[str, torch.Tensor],
    attention_interventions: List[Dict[str, Any]],
    read_token_index: int,
) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
    seq_len = int(encoded["input_ids"].shape[1])
    grouped_heads = group_attention_interventions(attention_interventions, attentions, seq_len)
    captures: Dict[int, torch.Tensor] = {}
    handles = []

    for layer, mlp in mlps.items():
        def capture_act(module: Any, inputs: Any, output: torch.Tensor, layer_index: int = layer) -> None:
            captures[layer_index] = output.detach().cpu()[0]

        handles.append(mlp.act.register_forward_hook(capture_act))

    for layer, heads in grouped_heads.items():
        attention_module = attentions[layer]
        head_dim = int(attention_module.head_dim)

        def patch_attention_path(module: Any, inputs: Any, layer_index: int = layer) -> tuple[torch.Tensor, ...]:
            patched = inputs[0].clone()
            for head, positions in grouped_heads[layer_index].items():
                start = int(head) * head_dim
                end = start + head_dim
                for position in positions:
                    patched[:, position, start:end] = 0
            return (patched,) + tuple(inputs[1:])

        handles.append(attention_module.c_proj.register_forward_pre_hook(patch_attention_path))

    try:
        with torch.no_grad():
            logits = model(**encoded).logits.detach().cpu()[0, read_token_index]
    finally:
        for handle in handles:
            handle.remove()

    missing = [layer for layer in mlps if layer not in captures]
    if missing:
        raise RuntimeError(f"Missing ordered path MLP captures for layers: {missing}")
    return logits, captures


def path_activation_items(
    *,
    baseline_captures: Dict[int, torch.Tensor],
    patched_captures: Dict[int, torch.Tensor],
    mlp_interventions: List[Dict[str, Any]],
    path_positions: List[int],
) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    allowed_positions = set(int(position) for position in path_positions)
    for intervention in mlp_interventions:
        layer = int(intervention["layer"])
        neuron = int(intervention["neuron"])
        positions = [int(position) for position in intervention.get("ablateTokenPositions", [])]
        for position in positions:
            if allowed_positions and position not in allowed_positions:
                continue
            if layer not in baseline_captures or layer not in patched_captures:
                continue
            if position < 0 or position >= int(baseline_captures[layer].shape[0]):
                continue
            before = float(baseline_captures[layer][position, neuron])
            after = float(patched_captures[layer][position, neuron])
            items.append({
                "layer": layer,
                "neuron": neuron,
                "tokenPosition": position,
                "baselineActivation": before,
                "attentionPatchedActivation": after,
                "signedDelta": before - after,
                "absDelta": abs(before - after),
            })
    return items


def summarize_path_items(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    values = [float(item["absDelta"]) for item in items]
    signed = [float(item["signedDelta"]) for item in items]
    return {
        "count": len(items),
        "meanAbsDelta": sum(values) / max(1, len(values)),
        "maxAbsDelta": max(values) if values else 0.0,
        "p50AbsDelta": factual.percentile(values, 0.50),
        "p95AbsDelta": factual.percentile(values, 0.95),
        "meanSignedDelta": sum(signed) / max(1, len(signed)),
    }


def score_path_activation(
    *,
    baseline_captures: Dict[int, torch.Tensor],
    patched_captures: Dict[int, torch.Tensor],
    mlp_interventions: List[Dict[str, Any]],
    path_positions: List[int],
) -> Dict[str, Any]:
    items = path_activation_items(
        baseline_captures=baseline_captures,
        patched_captures=patched_captures,
        mlp_interventions=mlp_interventions,
        path_positions=path_positions,
    )
    summary = summarize_path_items(items)
    return {
        "items": items,
        "summary": summary,
        "pathActivationDelta": float(summary["meanAbsDelta"]),
    }


def evaluate_path_controls(
    *,
    model: Any,
    mlps: Dict[int, Any],
    attentions: Dict[int, Any],
    encoded: Dict[str, torch.Tensor],
    baseline_captures: Dict[int, torch.Tensor],
    combined_controls: Dict[str, Any],
    read_token_index: int,
) -> Dict[str, Any]:
    items: List[Dict[str, Any]] = []
    for control in combined_controls.get("items", []):
        mlp_interventions = control.get("mlpInterventions", [])
        attention_interventions = control.get("attentionInterventions", [])
        if not mlp_interventions or not attention_interventions:
            continue
        _, patched_captures = run_mlp_captures_with_attention_patch(
            model=model,
            mlps=mlps,
            attentions=attentions,
            encoded=encoded,
            attention_interventions=attention_interventions,
            read_token_index=read_token_index,
        )
        score = score_path_activation(
            baseline_captures=baseline_captures,
            patched_captures=patched_captures,
            mlp_interventions=mlp_interventions,
            path_positions=attention_path_positions(attention_interventions, read_token_index),
        )
        items.append({
            "kind": control.get("kind"),
            "controlClass": control.get("controlClass"),
            "mlpNodeCount": control.get("mlpNodeCount", len(mlp_interventions)),
            "attentionHeadCount": control.get("attentionHeadCount", len(attention_interventions)),
            "pathActivationDelta": score["pathActivationDelta"],
            "activationSummary": score["summary"],
            "mlpInterventions": mlp_interventions,
            "attentionInterventions": attention_interventions,
        })
    full_controls = [item for item in items if item.get("controlClass") == "full-control"]
    attention_swaps = [item for item in items if item.get("controlClass") == "attention-swap"]
    mlp_swaps = [item for item in items if item.get("controlClass") == "mlp-swap"]
    return {
        "items": items,
        "summary": summarize_path_controls(items),
        "gateSummary": summarize_path_controls(full_controls),
        "controlClassSummaries": {
            "attention-swap": summarize_path_controls(attention_swaps),
            "mlp-swap": summarize_path_controls(mlp_swaps),
            "full-control": summarize_path_controls(full_controls),
        },
    }


def summarize_path_controls(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    values = [float(item.get("pathActivationDelta", 0.0)) for item in items]
    return {
        "count": len(items),
        "maxPathActivationDelta": max(values) if values else 0.0,
        "p50PathActivationDelta": factual.percentile(values, 0.50),
        "p90PathActivationDelta": factual.percentile(values, 0.90),
        "p95PathActivationDelta": factual.percentile(values, 0.95),
    }


def attention_path_positions(attention_interventions: List[Dict[str, Any]], fallback_position: int) -> List[int]:
    positions = {
        int(position)
        for intervention in attention_interventions
        for position in intervention.get("ablateTokenPositions", [])
    }
    if not positions:
        positions.add(int(fallback_position))
    return sorted(positions)


def compact_combined(row: Dict[str, Any]) -> Dict[str, Any]:
    graph = row.get("combinedGraph", {})
    joint = graph.get("jointAblation", {})
    return {
        "status": row.get("status"),
        "componentPass": row.get("componentPass"),
        "selectedMlpNodeCount": len(graph.get("nodes", {}).get("mlp", [])),
        "selectedAttentionHeadCount": len(graph.get("nodes", {}).get("attention", [])),
        "targetLogitDrop": joint.get("targetLogitDrop", 0.0),
        "controlGraphP95PositiveTargetLogitDrop": joint.get("controlGraphP95PositiveTargetLogitDrop", 0.0),
        "marginOverControlGraphP95": joint.get("marginOverControlGraphP95", 0.0),
        "liftOverControlGraphP95": joint.get("liftOverControlGraphP95", 0.0),
    }


def mlp_intervention_from_candidate(candidate: Dict[str, Any], read_token_index: int) -> Dict[str, Any]:
    return {
        "layer": int(candidate["layer"]),
        "neuron": int(candidate["neuron"]),
        "ablateTokenPositions": factual.candidate_ablation_positions(candidate, read_token_index),
    }


def attention_intervention_from_candidate(candidate: Dict[str, Any], read_token_index: int) -> Dict[str, Any]:
    positions = [int(position) for position in candidate.get("ablateTokenPositions", [read_token_index])]
    return {
        "layer": int(candidate["layer"]),
        "head": int(candidate["head"]),
        "ablateTokenPositions": positions if positions else [read_token_index],
    }


def layer_balanced_candidates(candidates: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    by_layer: Dict[int, List[Dict[str, Any]]] = {}
    for candidate in candidates:
        by_layer.setdefault(int(candidate["layer"]), []).append(candidate)
    selected: List[Dict[str, Any]] = []
    depth = 0
    ordered_layers = sorted(by_layer)
    while len(selected) < limit:
        added = False
        for layer in ordered_layers:
            layer_candidates = by_layer[layer]
            if depth < len(layer_candidates):
                selected.append(layer_candidates[depth])
                added = True
                if len(selected) >= limit:
                    break
        if not added:
            break
        depth += 1
    return selected


def intervention_key(intervention: Dict[str, Any], unit_key: str) -> Tuple[int, int, Tuple[int, ...]]:
    return (
        int(intervention["layer"]),
        int(intervention[unit_key]),
        tuple(int(position) for position in intervention.get("ablateTokenPositions", [])),
    )


def ranked_unique_interventions(
    attempts: List[Dict[str, Any]],
    field: str,
    unit_key: str,
    limit: int,
) -> List[Dict[str, Any]]:
    ranked = sorted(
        [attempt for attempt in attempts if bool(attempt.get("orderValid"))],
        key=lambda item: (
            -float(item.get("combined", {}).get("combinedDrop", 0.0)),
            -float(item.get("pathMargin", -1e9)),
            -float(item.get("pathDelta", 0.0)),
        ),
    )
    selected: List[Dict[str, Any]] = []
    seen = set()
    for attempt in ranked:
        intervention = attempt[field]
        key = intervention_key(intervention, unit_key)
        if key in seen:
            continue
        seen.add(key)
        selected.append(intervention)
        if len(selected) >= limit:
            break
    return selected


def raw_combined_drop(
    *,
    model: Any,
    mlps: Dict[int, Any],
    attentions: Dict[int, Any],
    encoded: Dict[str, torch.Tensor],
    baseline_logits: torch.Tensor,
    selected_mlp: List[Dict[str, Any]],
    selected_attention: List[Dict[str, Any]],
    read_token_index: int,
    target_token_id: int,
) -> float:
    ablated_logits = combined.run_combined_ablation_logits(
        model=model,
        mlps=mlps,
        attentions=attentions,
        encoded=encoded,
        mlp_interventions=selected_mlp,
        attention_interventions=selected_attention,
        read_token_index=read_token_index,
    )
    return combined.target_drop_from_logits(baseline_logits, ablated_logits, target_token_id)


def candidate_set_key(candidate_set: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        candidate_set["topology"],
        tuple(sorted(intervention_key(item, "neuron") for item in candidate_set["mlps"])),
        tuple(sorted(intervention_key(item, "head") for item in candidate_set["attentions"])),
    )


def pareto_shortlist(candidate_sets: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    if limit <= 0:
        return []
    raw_quota = (limit + 1) // 2
    path_quota = limit - raw_quota
    ranked_raw = sorted(candidate_sets, key=lambda item: -float(item["rawCombinedDrop"]))
    ranked_path = sorted(
        candidate_sets,
        key=lambda item: (
            -float(item["estimatedPathDelta"]),
            -float(item["rawCombinedDrop"]),
        ),
    )
    selected: List[Dict[str, Any]] = []
    seen = set()
    for ranked, quota in ((ranked_raw, raw_quota), (ranked_path, path_quota)):
        added = 0
        for item in ranked:
            key = candidate_set_key(item)
            if key in seen:
                continue
            selected.append(item)
            seen.add(key)
            added += 1
            if added >= quota:
                break
    if len(selected) < limit:
        for item in ranked_raw:
            key = candidate_set_key(item)
            if key in seen:
                continue
            selected.append(item)
            seen.add(key)
            if len(selected) >= limit:
                break
    return selected


def evaluate_ordered_attempt(
    *,
    model: Any,
    mlps: Dict[int, Any],
    attentions: Dict[int, Any],
    encoded: Dict[str, torch.Tensor],
    baseline_logits: torch.Tensor,
    baseline_captures: Dict[int, torch.Tensor],
    mlp_row: Dict[str, Any],
    attention_row: Dict[str, Any],
    selected_mlp: List[Dict[str, Any]],
    selected_attention: List[Dict[str, Any]],
    read_token_index: int,
    target_token_id: int,
    component_pass: bool,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    combined_attempt = combined.evaluate_combined_attempt(
        model=model,
        mlps=mlps,
        attentions=attentions,
        encoded=encoded,
        baseline_logits=baseline_logits,
        mlp_row=mlp_row,
        attention_row=attention_row,
        selected_mlp=selected_mlp,
        selected_attention=selected_attention,
        read_token_index=read_token_index,
        target_token_id=target_token_id,
        min_target_drop=float(args.min_target_drop),
        min_margin_over_control_p95=float(args.min_margin_over_control_p95),
    )
    _, patched_captures = run_mlp_captures_with_attention_patch(
        model=model,
        mlps=mlps,
        attentions=attentions,
        encoded=encoded,
        attention_interventions=selected_attention,
        read_token_index=read_token_index,
    )
    path_score = score_path_activation(
        baseline_captures=baseline_captures,
        patched_captures=patched_captures,
        mlp_interventions=selected_mlp,
        path_positions=attention_path_positions(selected_attention, read_token_index),
    )
    path_controls = evaluate_path_controls(
        model=model,
        mlps=mlps,
        attentions=attentions,
        encoded=encoded,
        baseline_captures=baseline_captures,
        combined_controls=combined_attempt["controls"],
        read_token_index=read_token_index,
    )
    path_control_p95 = float(path_controls["gateSummary"]["p95PathActivationDelta"])
    path_delta = float(path_score["pathActivationDelta"])
    path_margin = path_delta - path_control_p95
    component_gate = component_pass or not bool(args.require_component_hardening)
    return {
        "mlp": selected_mlp[0],
        "attention": selected_attention[0],
        "mlps": selected_mlp,
        "attentions": selected_attention,
        "orderValid": True,
        "componentPass": component_pass,
        "componentHardeningRequired": bool(args.require_component_hardening),
        "componentGatePass": component_gate,
        "combined": combined_attempt,
        "pathScore": path_score,
        "pathControls": path_controls,
        "pathControlP95": path_control_p95,
        "pathDelta": path_delta,
        "pathMargin": path_margin,
        "pass": (
            component_gate
            and bool(combined_attempt.get("pass"))
            and path_delta >= float(args.min_path_activation_delta)
            and path_margin >= float(args.min_path_margin_over_control_p95)
        ),
    }


def scan_ordered_pair(
    *,
    model: Any,
    mlps: Dict[int, Any],
    attentions: Dict[int, Any],
    encoded: Dict[str, torch.Tensor],
    baseline_logits: torch.Tensor,
    baseline_captures: Dict[int, torch.Tensor],
    mlp_intervention: Dict[str, Any],
    attention_intervention: Dict[str, Any],
    read_token_index: int,
    target_token_id: int,
) -> Dict[str, Any]:
    raw_drop = raw_combined_drop(
        model=model,
        mlps=mlps,
        attentions=attentions,
        encoded=encoded,
        baseline_logits=baseline_logits,
        selected_mlp=[mlp_intervention],
        selected_attention=[attention_intervention],
        read_token_index=read_token_index,
        target_token_id=target_token_id,
    )
    _, patched_captures = run_mlp_captures_with_attention_patch(
        model=model,
        mlps=mlps,
        attentions=attentions,
        encoded=encoded,
        attention_interventions=[attention_intervention],
        read_token_index=read_token_index,
    )
    path_score = score_path_activation(
        baseline_captures=baseline_captures,
        patched_captures=patched_captures,
        mlp_interventions=[mlp_intervention],
        path_positions=attention_path_positions([attention_intervention], read_token_index),
    )
    path_delta = float(path_score["pathActivationDelta"])
    return {
        "topology": "one-attention-one-mlp",
        "mlps": [mlp_intervention],
        "attentions": [attention_intervention],
        "rawCombinedDrop": raw_drop,
        "estimatedPathDelta": path_delta,
    }


def compact_combined_attempt(
    *,
    component_pass: bool,
    chosen: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "status": "passed" if component_pass and bool(chosen.get("pass")) else "failed",
        "componentPass": component_pass,
        "selectedMlpNodeCount": len(chosen.get("selectedMlp", [])),
        "selectedAttentionHeadCount": len(chosen.get("selectedAttention", [])),
        "targetLogitDrop": float(chosen.get("combinedDrop", 0.0)),
        "controlGraphP95PositiveTargetLogitDrop": float(chosen.get("controlP95", 0.0)),
        "marginOverControlGraphP95": float(chosen.get("margin", 0.0)),
        "liftOverControlGraphP95": float(chosen.get("lift", 0.0)),
    }


def select_ordered_rescue_attempt(
    *,
    model: Any,
    tokenizer: Any,
    mlps: Dict[int, Any],
    attentions: Dict[int, Any],
    primary_mlps: Dict[int, Any],
    fallback_mlps: Dict[int, Any],
    primary_attentions: Dict[int, Any],
    fallback_attentions: Dict[int, Any],
    encoded: Dict[str, torch.Tensor],
    baseline_logits: torch.Tensor,
    baseline_captures: Dict[int, torch.Tensor],
    fact: Dict[str, str],
    answer_pool: List[Dict[str, Any]],
    args: argparse.Namespace,
    rng: random.Random,
    read_token_index: int,
    target_token_id: int,
    prompt: str | None = None,
) -> Dict[str, Any] | None:
    mlp_row = combined.run_component_case(
        component="mlp",
        model=model,
        tokenizer=tokenizer,
        primary_modules=primary_mlps,
        fallback_modules=fallback_mlps,
        fact=fact,
        answer_pool=answer_pool,
        args=args,
        rng=rng,
        prompt=prompt,
    )
    attention_row = combined.run_component_case(
        component="attention",
        model=model,
        tokenizer=tokenizer,
        primary_modules=primary_attentions,
        fallback_modules=fallback_attentions,
        fact=fact,
        answer_pool=answer_pool,
        args=args,
        rng=rng,
        prompt=prompt,
    )
    if mlp_row.get("status") == "skipped" or attention_row.get("status") == "skipped":
        return None
    component_pass = mlp_row.get("status") == "passed" and attention_row.get("status") == "passed"
    rescue_candidate_count = max(1, int(args.ordered_rescue_candidate_count))
    mlp_candidates = layer_balanced_candidates(
        mlp_row.get("orderedRescueCandidates", mlp_row.get("topCandidates", [])),
        rescue_candidate_count,
    )
    attention_candidates = layer_balanced_candidates(
        attention_row.get("orderedRescueCandidates", attention_row.get("candidateHeads", [])),
        rescue_candidate_count,
    )
    pair_candidates: List[Dict[str, Any]] = []
    for mlp_candidate in mlp_candidates:
        mlp_intervention = mlp_intervention_from_candidate(mlp_candidate, read_token_index)
        mlp_layer = int(mlp_intervention["layer"])
        for attention_candidate in attention_candidates:
            attention_intervention = attention_intervention_from_candidate(attention_candidate, read_token_index)
            attention_layer = int(attention_intervention["layer"])
            order_valid = attention_layer <= mlp_layer
            if not order_valid:
                continue
            pair_candidates.append(scan_ordered_pair(
                model=model,
                mlps=mlps,
                attentions=attentions,
                encoded=encoded,
                baseline_logits=baseline_logits,
                baseline_captures=baseline_captures,
                mlp_intervention=mlp_intervention,
                attention_intervention=attention_intervention,
                read_token_index=read_token_index,
                target_token_id=target_token_id,
            ))
    if not pair_candidates:
        return None
    max_pair_attempts = max(1, int(args.ordered_rescue_max_pair_attempts))
    pair_finalists = pareto_shortlist(pair_candidates, max_pair_attempts)
    attempts: List[Dict[str, Any]] = []
    for pair_candidate in pair_finalists:
        attempt = evaluate_ordered_attempt(
            model=model,
            mlps=mlps,
            attentions=attentions,
            encoded=encoded,
            baseline_logits=baseline_logits,
            baseline_captures=baseline_captures,
            mlp_row=mlp_row,
            attention_row=attention_row,
            selected_mlp=pair_candidate["mlps"],
            selected_attention=pair_candidate["attentions"],
            read_token_index=read_token_index,
            target_token_id=target_token_id,
            component_pass=component_pass,
            args=args,
        )
        attempt["topology"] = pair_candidate["topology"]
        attempt["estimatedPathDelta"] = float(pair_candidate["estimatedPathDelta"])
        attempt["rawCombinedDrop"] = float(pair_candidate["rawCombinedDrop"])
        attempts.append(attempt)
    passing = [attempt for attempt in attempts if bool(attempt.get("pass"))]
    if not passing:
        set_candidate_count = max(2, int(args.ordered_rescue_set_candidate_count))
        ranked_mlps = ranked_unique_interventions(attempts, "mlp", "neuron", set_candidate_count)
        ranked_attentions = ranked_unique_interventions(attempts, "attention", "head", set_candidate_count)
        pair_path_deltas = {
            (
                intervention_key(attempt["mlp"], "neuron"),
                intervention_key(attempt["attention"], "head"),
            ): float(attempt.get("pathDelta", 0.0))
            for attempt in attempts
            if bool(attempt.get("orderValid")) and attempt.get("mlp") and attempt.get("attention")
        }
        set_candidates: List[Dict[str, Any]] = []
        for attention_intervention in ranked_attentions:
            for mlp_pair in itertools.combinations(ranked_mlps, 2):
                if int(attention_intervention["layer"]) > min(int(item["layer"]) for item in mlp_pair):
                    continue
                selected_mlp = list(mlp_pair)
                selected_attention = [attention_intervention]
                set_candidates.append({
                    "topology": "one-attention-two-mlp",
                    "mlps": selected_mlp,
                    "attentions": selected_attention,
                    "estimatedPathDelta": sum(
                        pair_path_deltas.get((
                            intervention_key(item, "neuron"),
                            intervention_key(attention_intervention, "head"),
                        ), 0.0)
                        for item in selected_mlp
                    ) / len(selected_mlp),
                    "rawCombinedDrop": raw_combined_drop(
                        model=model,
                        mlps=mlps,
                        attentions=attentions,
                        encoded=encoded,
                        baseline_logits=baseline_logits,
                        selected_mlp=selected_mlp,
                        selected_attention=selected_attention,
                        read_token_index=read_token_index,
                        target_token_id=target_token_id,
                    ),
                })
        for mlp_intervention in ranked_mlps:
            for attention_pair in itertools.combinations(ranked_attentions, 2):
                if max(int(item["layer"]) for item in attention_pair) > int(mlp_intervention["layer"]):
                    continue
                selected_mlp = [mlp_intervention]
                selected_attention = list(attention_pair)
                set_candidates.append({
                    "topology": "two-attention-one-mlp",
                    "mlps": selected_mlp,
                    "attentions": selected_attention,
                    "estimatedPathDelta": sum(
                        pair_path_deltas.get((
                            intervention_key(mlp_intervention, "neuron"),
                            intervention_key(item, "head"),
                        ), 0.0)
                        for item in selected_attention
                    ) / len(selected_attention),
                    "rawCombinedDrop": raw_combined_drop(
                        model=model,
                        mlps=mlps,
                        attentions=attentions,
                        encoded=encoded,
                        baseline_logits=baseline_logits,
                        selected_mlp=selected_mlp,
                        selected_attention=selected_attention,
                        read_token_index=read_token_index,
                        target_token_id=target_token_id,
                    ),
                })
        joint_mlps = ranked_mlps[:min(8, len(ranked_mlps))]
        joint_attentions = ranked_attentions[:min(8, len(ranked_attentions))]
        for mlp_pair in itertools.combinations(joint_mlps, 2):
            for attention_pair in itertools.combinations(joint_attentions, 2):
                if max(int(item["layer"]) for item in attention_pair) > min(int(item["layer"]) for item in mlp_pair):
                    continue
                selected_mlp = list(mlp_pair)
                selected_attention = list(attention_pair)
                pair_scores = [
                    pair_path_deltas.get((
                        intervention_key(mlp_item, "neuron"),
                        intervention_key(attention_item, "head"),
                    ), 0.0)
                    for mlp_item in selected_mlp
                    for attention_item in selected_attention
                ]
                set_candidates.append({
                    "topology": "two-attention-two-mlp",
                    "mlps": selected_mlp,
                    "attentions": selected_attention,
                    "estimatedPathDelta": sum(pair_scores) / max(1, len(pair_scores)),
                    "rawCombinedDrop": raw_combined_drop(
                        model=model,
                        mlps=mlps,
                        attentions=attentions,
                        encoded=encoded,
                        baseline_logits=baseline_logits,
                        selected_mlp=selected_mlp,
                        selected_attention=selected_attention,
                        read_token_index=read_token_index,
                        target_token_id=target_token_id,
                    ),
                })
        max_set_attempts = max(2, int(args.ordered_rescue_max_set_attempts))
        ranked_sets: List[Dict[str, Any]] = []
        topology_names = ("one-attention-two-mlp", "two-attention-one-mlp", "two-attention-two-mlp")
        base_quota = max_set_attempts // len(topology_names)
        remainder = max_set_attempts % len(topology_names)
        for topology_index, topology in enumerate(topology_names):
            topology_quota = base_quota + (1 if topology_index < remainder else 0)
            topology_candidates = [item for item in set_candidates if item["topology"] == topology]
            ranked_sets.extend(pareto_shortlist(topology_candidates, topology_quota))
        for candidate_set in ranked_sets:
            attempt = evaluate_ordered_attempt(
                model=model,
                mlps=mlps,
                attentions=attentions,
                encoded=encoded,
                baseline_logits=baseline_logits,
                baseline_captures=baseline_captures,
                mlp_row=mlp_row,
                attention_row=attention_row,
                selected_mlp=candidate_set["mlps"],
                selected_attention=candidate_set["attentions"],
                read_token_index=read_token_index,
                target_token_id=target_token_id,
                component_pass=component_pass,
                args=args,
            )
            attempt["topology"] = candidate_set["topology"]
            attempt["estimatedPathDelta"] = float(candidate_set["estimatedPathDelta"])
            attempt["rawCombinedDrop"] = float(candidate_set["rawCombinedDrop"])
            attempts.append(attempt)
        passing = [attempt for attempt in attempts if bool(attempt.get("pass"))]
    if passing:
        chosen = sorted(
            passing,
            key=lambda item: (
                len(item.get("mlps", [item["mlp"]])) + len(item.get("attentions", [item["attention"]])),
                max(int(entry["layer"]) for entry in item.get("attentions", [item["attention"]])),
                max(int(entry["layer"]) for entry in item.get("mlps", [item["mlp"]])),
                -float(item["pathMargin"]),
                -float(item["combined"]["combinedDrop"]),
            ),
        )[0]
        component_count = len(chosen.get("mlps", [chosen["mlp"]])) + len(chosen.get("attentions", [chosen["attention"]]))
        mode = "ordered-rescue-minimal-layer-passing-pair" if component_count == 2 else "ordered-rescue-minimal-passing-set"
    else:
        scored = [attempt for attempt in attempts if bool(attempt.get("orderValid"))]
        if not scored:
            return None
        chosen = max(
            scored,
            key=lambda item: (
                float(item.get("pathMargin", -1e9)),
                float(item.get("pathDelta", 0.0)),
                float(item.get("combined", {}).get("combinedDrop", 0.0)),
            ),
        )
        mode = "ordered-rescue-best-margin-pair"
    ranked_for_audit = sorted(
        [attempt for attempt in attempts if bool(attempt.get("orderValid"))],
        key=lambda item: (
            not bool(item.get("pass")),
            -float(item.get("pathMargin", -1e9)),
            -float(item.get("combined", {}).get("combinedDrop", 0.0)),
        ),
    )
    recorded_attempts = ranked_for_audit[:16]
    for topology in (
        "one-attention-one-mlp",
        "one-attention-two-mlp",
        "two-attention-one-mlp",
        "two-attention-two-mlp",
    ):
        topology_attempts = [
            attempt
            for attempt in ranked_for_audit
            if attempt.get("topology", "one-attention-one-mlp") == topology
        ]
        for attempt in topology_attempts[:8]:
            if attempt not in recorded_attempts:
                recorded_attempts.append(attempt)
    if chosen not in recorded_attempts:
        recorded_attempts.append(chosen)
    return {
        "mode": mode,
        "componentPass": component_pass,
        "chosen": chosen,
        "candidateCount": len(pair_candidates) + len(set_candidates) if not passing else len(pair_candidates),
        "attemptCount": len(attempts),
        "passingAttemptCount": len(passing),
        "recordedAttemptCount": len(recorded_attempts),
        "attempts": [
            {
                "mlp": attempt.get("mlp"),
                "attention": attempt.get("attention"),
                "mlps": attempt.get("mlps", [attempt.get("mlp")] if attempt.get("mlp") else []),
                "attentions": attempt.get("attentions", [attempt.get("attention")] if attempt.get("attention") else []),
                "orderValid": bool(attempt.get("orderValid")),
                "topology": attempt.get("topology", "one-attention-one-mlp"),
                "estimatedPathDelta": float(attempt.get("estimatedPathDelta", attempt.get("pathDelta", 0.0))),
                "rawCombinedDrop": float(attempt.get("rawCombinedDrop", attempt.get("combined", {}).get("combinedDrop", 0.0))),
                "combinedDrop": float(attempt.get("combined", {}).get("combinedDrop", 0.0)),
                "combinedMargin": float(attempt.get("combined", {}).get("margin", 0.0)),
                "pathDelta": float(attempt.get("pathDelta", 0.0)),
                "pathControlP95": float(attempt.get("pathControlP95", 0.0)),
                "pathMargin": float(attempt.get("pathMargin", 0.0)),
                "componentHardeningRequired": bool(attempt.get("componentHardeningRequired", True)),
                "componentHardeningPass": bool(attempt.get("componentPass", False)),
                "componentGatePass": bool(attempt.get("componentGatePass", False)),
                "pass": bool(attempt.get("pass")),
                "failureReason": attempt.get("failureReason"),
            }
            for attempt in recorded_attempts
        ],
    }


def run_ordered_case(
    *,
    model: Any,
    tokenizer: Any,
    mlps: Dict[int, Any],
    attentions: Dict[int, Any],
    primary_mlps: Dict[int, Any],
    fallback_mlps: Dict[int, Any],
    primary_attentions: Dict[int, Any],
    fallback_attentions: Dict[int, Any],
    fact: Dict[str, str],
    answer_pool: List[Dict[str, Any]],
    args: argparse.Namespace,
    rng: random.Random,
    prompt: str | None = None,
) -> Dict[str, Any]:
    combined_row = combined.run_combined_case(
        model=model,
        tokenizer=tokenizer,
        mlps=mlps,
        attentions=attentions,
        primary_mlps=primary_mlps,
        fallback_mlps=fallback_mlps,
        primary_attentions=primary_attentions,
        fallback_attentions=fallback_attentions,
        fact=fact,
        answer_pool=answer_pool,
        args=namespace_for_combined(args),
        rng=rng,
        prompt=prompt,
    )
    if combined_row.get("status") == "skipped":
        return combined_row

    graph = combined_row.get("combinedGraph", {})
    selected_mlp = graph.get("nodes", {}).get("mlp", [])
    selected_attention = graph.get("nodes", {}).get("attention", [])
    if not selected_mlp or not selected_attention:
        return {
            **combined_row,
            "status": "failed",
            "failureReason": "ordered path requires selected MLP and attention interventions",
            "combined": compact_combined(combined_row),
        }

    prompt = prompt if prompt is not None else factual.prompt_for_fact(fact)
    target = factual.encode_single_answer_token(tokenizer, fact["answer"])
    if target is None:
        return {"subject": fact["subject"], "answer": fact["answer"], "prompt": prompt, "status": "skipped", "reason": "answer is not a single token"}
    target_token_id, target_token = target
    device = next(model.parameters()).device
    encoded = encode_prompt(tokenizer, prompt, device)
    token_ids = encoded["input_ids"][0].detach().cpu()
    read_token_index = int(token_ids.numel()) - 1

    baseline_logits, baseline_captures = run_mlp_captures_with_attention_patch(
        model=model,
        mlps=mlps,
        attentions=attentions,
        encoded=encoded,
        attention_interventions=[],
        read_token_index=read_token_index,
    )
    attention_logits, attention_patched_captures = run_mlp_captures_with_attention_patch(
        model=model,
        mlps=mlps,
        attentions=attentions,
        encoded=encoded,
        attention_interventions=selected_attention,
        read_token_index=read_token_index,
    )
    selected_path = score_path_activation(
        baseline_captures=baseline_captures,
        patched_captures=attention_patched_captures,
        mlp_interventions=selected_mlp,
        path_positions=attention_path_positions(selected_attention, read_token_index),
    )
    controls = evaluate_path_controls(
        model=model,
        mlps=mlps,
        attentions=attentions,
        encoded=encoded,
        baseline_captures=baseline_captures,
        combined_controls=graph.get("controlGraphs", {}),
        read_token_index=read_token_index,
    )
    control_p95 = float(controls["gateSummary"]["p95PathActivationDelta"])
    path_delta = float(selected_path["pathActivationDelta"])
    path_margin = path_delta - control_p95
    attention_target_drop = float(baseline_logits[target_token_id] - attention_logits[target_token_id])
    combined_summary = compact_combined(combined_row)
    selection_mode = "combined-minimal-prefix"
    ordered_rescue = None
    combined_pass = combined_summary.get("status") == "passed"
    pass_case = (
        combined_pass
        and path_delta >= float(args.min_path_activation_delta)
        and path_margin >= float(args.min_path_margin_over_control_p95)
    )
    if not pass_case:
        ordered_rescue = select_ordered_rescue_attempt(
            model=model,
            tokenizer=tokenizer,
            mlps=mlps,
            attentions=attentions,
            primary_mlps=primary_mlps,
            fallback_mlps=fallback_mlps,
            primary_attentions=primary_attentions,
            fallback_attentions=fallback_attentions,
            encoded=encoded,
            baseline_logits=baseline_logits,
            baseline_captures=baseline_captures,
            fact=fact,
            answer_pool=answer_pool,
            args=args,
            rng=rng,
            read_token_index=read_token_index,
            target_token_id=target_token_id,
            prompt=prompt,
        )
        if ordered_rescue is not None:
            rescue_chosen = ordered_rescue["chosen"]
            rescue_combined = rescue_chosen.get("combined", {})
            selected_mlp = rescue_combined.get("selectedMlp", [rescue_chosen["mlp"]])
            selected_attention = rescue_combined.get("selectedAttention", [rescue_chosen["attention"]])
            selected_path = rescue_chosen["pathScore"]
            controls = rescue_chosen["pathControls"]
            control_p95 = float(rescue_chosen["pathControlP95"])
            path_delta = float(rescue_chosen["pathDelta"])
            path_margin = float(rescue_chosen["pathMargin"])
            combined_summary = compact_combined_attempt(
                component_pass=bool(rescue_chosen.get("componentGatePass")),
                chosen=rescue_combined,
            )
            combined_pass = combined_summary.get("status") == "passed"
            pass_case = bool(rescue_chosen.get("pass"))
            selection_mode = str(ordered_rescue["mode"])

    return {
        "subject": fact["subject"],
        "answer": fact["answer"],
        "prompt": prompt,
        "status": "passed" if pass_case else "failed",
        "target": {
            "tokenId": int(target_token_id),
            "token": target_token,
            "baselineLogit": float(baseline_logits[target_token_id]),
            "baselineRank": factual.target_rank(baseline_logits, target_token_id),
        },
        "readTokenIndex": read_token_index,
        "readToken": live.clean_token(tokenizer, int(token_ids[read_token_index])),
        "combined": combined_summary,
        "orderedResidualPath": {
            "kind": f"{live.adapter_id}-live-ordered-residual-attention-mlp-target-path",
            "nodes": {
                "attention": selected_attention,
                "mlp": selected_mlp,
                "target": {"tokenId": int(target_token_id), "token": target_token},
            },
            "edges": [
                {
                    "source": f"attention:{item['layer']}:{item['head']}",
                    "target": "residual-stream",
                    "kind": "attention-head-residual-write",
                    "ablateTokenPositions": item.get("ablateTokenPositions", []),
                }
                for item in selected_attention
            ] + [
                {
                    "source": "residual-stream",
                    "target": f"mlp:{item['layer']}:{item['neuron']}",
                    "kind": "residual-to-mlp-activation-dependency",
                    "ablateTokenPositions": item.get("ablateTokenPositions", []),
                }
                for item in selected_mlp
            ] + [{
                "source": "selected-mlp",
                "target": f"target:{target_token_id}",
                "kind": "mlp-to-target-logit-causal-ablation",
                "targetLogitDrop": graph.get("jointAblation", {}).get("mlpOnlyTargetLogitDrop", 0.0),
            }],
            "pathPatch": {
                "attentionOnlyTargetLogitDrop": attention_target_drop,
                "selectedPathActivationDelta": path_delta,
                "selectedPathActivationSummary": selected_path["summary"],
                "selectedPathActivationItems": selected_path["items"],
                "controlPathP95ActivationDelta": control_p95,
                "controlPathMaxActivationDelta": float(controls["gateSummary"]["maxPathActivationDelta"]),
                "marginOverControlPathP95": path_margin,
                "liftOverControlPathP95": path_delta / max(1e-9, control_p95),
            },
            "controlPaths": controls,
            "selection": {
                "mode": selection_mode,
                "rescueAttempted": ordered_rescue is not None,
                "rescueCandidateCount": ordered_rescue.get("candidateCount", 0) if ordered_rescue is not None else 0,
                "rescueAttemptCount": ordered_rescue.get("attemptCount", 0) if ordered_rescue is not None else 0,
                "rescuePassingAttemptCount": ordered_rescue.get("passingAttemptCount", 0) if ordered_rescue is not None else 0,
                "rescueRecordedAttemptCount": ordered_rescue.get("recordedAttemptCount", 0) if ordered_rescue is not None else 0,
                "rescueAttempts": ordered_rescue.get("attempts", []) if ordered_rescue is not None else [],
            },
            "gate": {
                "pass": pass_case,
                "requiresCombinedGraphPass": True,
                "requiresComponentHardening": bool(args.require_component_hardening),
                "minPathActivationDelta": float(args.min_path_activation_delta),
                "minPathMarginOverControlP95": float(args.min_path_margin_over_control_p95),
            },
        },
    }


def make_report_md(report: Dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Live LLM Ordered Residual Path Circuit Finder",
        "",
        "This benchmark uses a real causal-LM model, real attention-path hooks, real MLP activation hooks, and ordered attention->residual->MLP->target path evidence. It is not synthetic and does not fit a surrogate.",
        "",
        "## Summary",
        "",
        f"- Pass: `{summary['pass']}`",
        f"- Valid cases: `{summary['validCases']}`",
        f"- Passed cases: `{summary['passedCases']}`",
        f"- Pass rate: `{summary['passRate']}`",
        f"- Combined graph pass rate: `{summary['combinedGraphPassRate']}`",
        f"- Median selected path activation delta: `{summary['medianSelectedPathActivationDelta']}`",
        f"- Median control path p95 activation delta: `{summary['medianControlPathP95ActivationDelta']}`",
        f"- Median ordered path lift over control p95: `{summary['medianOrderedPathLiftOverControlP95']}`",
        f"- Median combined target-logit drop: `{summary['medianCombinedTargetLogitDrop']}`",
        "",
        "## Claim Boundary",
        "",
        report["claimBoundary"]["provesIfPasses"],
        "",
        report["claimBoundary"]["doesNotProve"],
        "",
        "## Cases",
        "",
        "| subject | status | combined | path delta | control p95 | margin | lift |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["cases"]:
        if row.get("status") == "skipped":
            continue
        path = row.get("orderedResidualPath", {}).get("pathPatch", {})
        lines.append(
            f"| {row['subject']} | {row['status']} | {row.get('combined', {}).get('status')} | "
            f"{float(path.get('selectedPathActivationDelta', 0.0)):.6f} | "
            f"{float(path.get('controlPathP95ActivationDelta', 0.0)):.6f} | "
            f"{float(path.get('marginOverControlPathP95', 0.0)):.6f} | "
            f"{float(path.get('liftOverControlPathP95', 0.0)):.6f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    set_live_adapter(args.adapter)
    if args.case_count <= 0 or args.min_valid_cases <= 0:
        raise RuntimeError("case counts must be positive.")
    only_subjects = {item.strip() for item in args.only_subjects.split(",") if item.strip()}
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")

    tokenizer, model = live.load_model_and_tokenizer(args.model, local_files_only=args.local_files_only, device=device)
    mlp_layers = factual.parse_layers(args.mlp_layers)
    mlp_fallback_layers = factual.parse_layers(args.mlp_fallback_layers) if args.mlp_fallback_layers.strip() else []
    attention_layers = attention.parse_layers(args.attention_layers)
    attention_fallback_layers = attention.parse_layers(args.attention_fallback_layers) if args.attention_fallback_layers.strip() else []
    all_mlp_layers = sorted(set(mlp_layers) | set(mlp_fallback_layers))
    all_attention_layers = sorted(set(attention_layers) | set(attention_fallback_layers))
    mlps = live.mlp_modules(model, all_mlp_layers)
    attentions = live.attention_modules(model, all_attention_layers)
    primary_mlps = {layer: mlps[layer] for layer in mlp_layers}
    fallback_mlps = {layer: mlps[layer] for layer in mlp_fallback_layers} if mlp_fallback_layers else {}
    primary_attentions = {layer: attentions[layer] for layer in attention_layers}
    fallback_attentions = {layer: attentions[layer] for layer in attention_fallback_layers} if attention_fallback_layers else {}
    answer_pool = factual.build_answer_pool(tokenizer)
    cases = []
    valid_seen = 0
    for fact in factual.FACTS:
        if only_subjects and fact["subject"] not in only_subjects:
            continue
        if valid_seen >= args.case_count:
            break
        case_rng = random.Random(stable_subject_seed(args.control_seed, fact["subject"]))
        row = run_ordered_case(
            model=model,
            tokenizer=tokenizer,
            mlps=mlps,
            attentions=attentions,
            primary_mlps=primary_mlps,
            fallback_mlps=fallback_mlps,
            primary_attentions=primary_attentions,
            fallback_attentions=fallback_attentions,
            fact=fact,
            answer_pool=answer_pool,
            args=args,
            rng=case_rng,
        )
        if row.get("status") != "skipped":
            valid_seen += 1
        cases.append(row)
        path = row.get("orderedResidualPath", {}).get("pathPatch", {})
        print(json.dumps({
            "case": valid_seen,
            "subject": row["subject"],
            "status": row["status"],
            "combined": row.get("combined", {}).get("status"),
            "pathDelta": path.get("selectedPathActivationDelta", 0.0),
            "pathMargin": path.get("marginOverControlPathP95", 0.0),
        }))
        sys.stdout.flush()

    valid_cases = [row for row in cases if row.get("status") != "skipped"]
    passed_cases = [row for row in valid_cases if row.get("status") == "passed"]
    combined_passed = [row for row in valid_cases if row.get("combined", {}).get("status") == "passed"]
    path_deltas = [float(row.get("orderedResidualPath", {}).get("pathPatch", {}).get("selectedPathActivationDelta", 0.0)) for row in valid_cases]
    control_p95 = [float(row.get("orderedResidualPath", {}).get("pathPatch", {}).get("controlPathP95ActivationDelta", 0.0)) for row in valid_cases]
    path_lifts = [float(row.get("orderedResidualPath", {}).get("pathPatch", {}).get("liftOverControlPathP95", 0.0)) for row in valid_cases]
    combined_drops = [float(row.get("combined", {}).get("targetLogitDrop", 0.0)) for row in valid_cases]
    pass_rate = len(passed_cases) / max(1, len(valid_cases))
    combined_rate = len(combined_passed) / max(1, len(valid_cases))
    pass_summary = (
        len(valid_cases) >= args.min_valid_cases
        and pass_rate >= args.min_pass_rate
        and combined_rate >= args.min_component_pass_rate
    )
    report = {
        "schemaVersion": 1,
        "kind": f"{live.adapter_id}-live-ordered-residual-path-circuit-finder",
        "mode": f"live-{live.adapter_id}-ordered-residual-attention-mlp-target-path",
        "synthetic": False,
        "surrogate": False,
        "generatedAt": now_iso(),
        "model": live.model_identity(model, args.model, device, args.local_files_only),
        "oracleAdapter": {
            "id": live.adapter_id,
            "activationFamily": live.activation_family,
            "contract": "LLMOracleAdapter",
        },
        "runtimeDependencies": {
            "python": sys.executable,
            "torch": live.package_version("torch"),
            "transformers": live.package_version("transformers"),
            "numpy": live.package_version("numpy"),
            "accelerate": live.package_version("accelerate"),
            "bitsandbytes": live.package_version("bitsandbytes"),
        },
        "config": {
            "mlpLayers": mlp_layers,
            "mlpFallbackLayers": mlp_fallback_layers,
            "attentionLayers": attention_layers,
            "attentionFallbackLayers": attention_fallback_layers,
            "caseCount": args.case_count,
            "minValidCases": args.min_valid_cases,
            "minPassRate": args.min_pass_rate,
            "minComponentPassRate": args.min_component_pass_rate,
            "controlCount": args.control_count,
            "attentionMaxTargetRank": args.attention_max_target_rank,
            "minPathActivationDelta": args.min_path_activation_delta,
            "minPathMarginOverControlP95": args.min_path_margin_over_control_p95,
            "requireComponentHardening": bool(args.require_component_hardening),
            "orderedRescueCandidateCount": int(args.ordered_rescue_candidate_count),
            "orderedRescueMaxPairAttempts": int(args.ordered_rescue_max_pair_attempts),
            "orderedRescueSetCandidateCount": int(args.ordered_rescue_set_candidate_count),
            "orderedRescueMaxSetAttempts": int(args.ordered_rescue_max_set_attempts),
            "controlSeedMode": "sha256-subject-derived",
            "device": device,
            "adapter": args.adapter,
        },
        "summary": {
            "pass": pass_summary,
            "validCases": len(valid_cases),
            "passedCases": len(passed_cases),
            "failedCases": len(valid_cases) - len(passed_cases),
            "passRate": pass_rate,
            "combinedGraphPassRate": combined_rate,
            "medianSelectedPathActivationDelta": factual.percentile(path_deltas, 0.50),
            "p10SelectedPathActivationDelta": factual.percentile(path_deltas, 0.10),
            "medianControlPathP95ActivationDelta": factual.percentile(control_p95, 0.50),
            "medianOrderedPathLiftOverControlP95": factual.percentile(path_lifts, 0.50),
            "medianCombinedTargetLogitDrop": factual.percentile(combined_drops, 0.50),
            "failedSubjects": [row["subject"] for row in valid_cases if row.get("status") != "passed"],
            "skippedCases": len([row for row in cases if row.get("status") == "skipped"]),
        },
        "cases": cases,
        "claimBoundary": {
            "provesIfPasses": f"Benchmark-scoped evidence that the real {live.adapter_id} pipeline can discover selected attention heads and MLP units for factual prompts, then validate an ordered attention->residual->MLP->target path by showing selected attention interventions change selected MLP activations above full-control path interventions while the combined circuit remains causally effective on the target logit.",
            "doesNotProve": "This is not full-model circuit recovery, not exhaustive residual-stream patching for every edge, not full FFN extraction, not arbitrary LLM understanding, not output-only original-neuron identity, and not a broad SOTA claim.",
        },
    }
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    output_md.write_text(make_report_md(report), encoding="utf-8")
    print(json.dumps({
        "pass": pass_summary,
        "validCases": len(valid_cases),
        "passedCases": len(passed_cases),
        "passRate": pass_rate,
        "reportJson": str(output_json),
        "reportMarkdown": str(output_md),
    }, indent=2))
    if not pass_summary:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
