#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

from llm_reader.adapters.gpt2 import GPT2OracleAdapter


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
live = GPT2OracleAdapter()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a real GPT-2 combined attention+MLP factual circuit graph sweep."
    )
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
    parser.add_argument("--control-seed", type=int, default=260606)
    parser.add_argument("--max-graph-nodes", type=int, default=5)
    parser.add_argument("--max-graph-heads", type=int, default=3)
    parser.add_argument("--min-pass-rate", type=float, default=0.90)
    parser.add_argument("--min-target-drop", type=float, default=0.05)
    parser.add_argument("--min-margin-over-control-p95", type=float, default=0.02)
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
    parser.add_argument("--max-negative-fact-drop-ratio", type=float, default=0.55)
    parser.add_argument("--max-unrelated-abs-drop-ratio", type=float, default=0.65)
    parser.add_argument("--strict-rescue-candidate-count", type=int, default=14)
    parser.add_argument("--strict-rescue-pair-candidate-count", type=int, default=12)
    parser.add_argument("--strict-rescue-max-combo-size", type=int, default=2)
    parser.add_argument("--strict-rescue-max-attempts", type=int, default=72)
    parser.add_argument("--min-component-pass-rate", type=float, default=0.90)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def namespace_for_mlp(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        top_neurons=args.top_neurons,
        top_logits=args.top_logits,
        control_count=args.control_count,
        control_prompt=args.control_prompt,
        max_graph_nodes=args.max_graph_nodes,
        min_pass_rate=args.min_pass_rate,
        min_target_drop=args.min_target_drop,
        min_margin_over_control_p95=args.min_margin_over_control_p95,
        paraphrase_count=args.paraphrase_count,
        negative_fact_count=args.negative_fact_count,
        wrong_answer_count=args.wrong_answer_count,
        unrelated_logit_count=args.unrelated_logit_count,
        min_paraphrase_pass_rate=args.min_paraphrase_pass_rate,
        min_wrong_answer_specificity_rate=args.min_wrong_answer_specificity_rate,
        min_negative_fact_specificity_rate=args.min_negative_fact_specificity_rate,
        min_unrelated_preservation_rate=args.min_unrelated_preservation_rate,
        max_wrong_answer_drop_ratio=args.max_wrong_answer_drop_ratio,
        max_candidate_wrong_answer_drop_ratio=args.max_candidate_wrong_answer_drop_ratio,
        max_negative_fact_drop_ratio=args.max_negative_fact_drop_ratio,
        max_unrelated_abs_drop_ratio=args.max_unrelated_abs_drop_ratio,
        strict_rescue_candidate_count=args.strict_rescue_candidate_count,
        strict_rescue_pair_candidate_count=args.strict_rescue_pair_candidate_count,
        strict_rescue_max_combo_size=args.strict_rescue_max_combo_size,
        strict_rescue_max_attempts=args.strict_rescue_max_attempts,
    )


def namespace_for_attention(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        control_count=args.control_count,
        control_prompt=args.control_prompt,
        max_graph_heads=args.max_graph_heads,
        min_pass_rate=args.min_pass_rate,
        min_graph_drop=args.min_target_drop,
        min_margin_over_control_p95=args.min_margin_over_control_p95,
        wrong_answer_count=args.wrong_answer_count,
        max_wrong_answer_drop_ratio=0.80,
        max_target_rank=args.attention_max_target_rank,
    )


def encode_prompt(tokenizer: Any, prompt: str, device: torch.device | str) -> Dict[str, torch.Tensor]:
    encoded = tokenizer(prompt, return_tensors="pt")
    return {key: value.to(device) for key, value in encoded.items()}


def group_mlp_interventions(
    interventions: List[Dict[str, Any]],
    seq_len: int,
) -> Dict[int, Dict[int, List[int]]]:
    grouped: Dict[int, Dict[int, List[int]]] = {}
    for intervention in interventions:
        layer = int(intervention["layer"])
        neuron = int(intervention["neuron"])
        positions = sorted(set(int(position) for position in intervention.get("ablateTokenPositions", [])))
        valid_positions = [position for position in positions if 0 <= position < seq_len]
        if not valid_positions:
            continue
        grouped.setdefault(layer, {}).setdefault(neuron, [])
        grouped[layer][neuron] = sorted(set(grouped[layer][neuron] + valid_positions))
    return grouped


def group_attention_interventions(
    interventions: List[Dict[str, Any]],
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
        grouped.setdefault(layer, {}).setdefault(head, [])
        grouped[layer][head] = sorted(set(grouped[layer][head] + valid_positions))
    return grouped


def run_combined_ablation_logits(
    *,
    model: Any,
    mlps: Dict[int, Any],
    attentions: Dict[int, Any],
    encoded: Dict[str, torch.Tensor],
    mlp_interventions: List[Dict[str, Any]],
    attention_interventions: List[Dict[str, Any]],
    read_token_index: int,
) -> torch.Tensor:
    seq_len = int(encoded["input_ids"].shape[1])
    grouped_mlps = group_mlp_interventions(mlp_interventions, seq_len)
    grouped_heads = group_attention_interventions(attention_interventions, seq_len)
    if not grouped_mlps and not grouped_heads:
        raise RuntimeError("Combined ablation requires at least one MLP or attention intervention.")

    handles = []
    for layer, neurons in grouped_mlps.items():
        if layer not in mlps:
            raise RuntimeError(f"Combined MLP ablation references unavailable layer {layer}.")

        def patch_activation(module: Any, inputs: Any, output: torch.Tensor, layer_index: int = layer) -> torch.Tensor:
            patched = output.clone()
            for neuron, positions in grouped_mlps[layer_index].items():
                for position in positions:
                    patched[:, position, neuron] = 0
            return patched

        handles.append(mlps[layer].act.register_forward_hook(patch_activation))

    for layer, heads in grouped_heads.items():
        if layer not in attentions:
            raise RuntimeError(f"Combined attention ablation references unavailable layer {layer}.")
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
            return model(**encoded).logits.detach().cpu()[0, read_token_index]
    finally:
        for handle in handles:
            handle.remove()


def mlp_control_interventions(group: Dict[str, Any], size: int, fallback_index: int) -> List[Dict[str, Any]]:
    return [
        {
            "layer": int(item["layer"]),
            "neuron": int(item["neuron"]),
            "ablateTokenPositions": [int(position) for position in item.get("ablateTokenPositions", [fallback_index])],
        }
        for item in group.get("items", [])[:size]
    ]


def attention_control_interventions(group: Dict[str, Any], size: int, fallback_index: int) -> List[Dict[str, Any]]:
    return [
        {
            "layer": int(item["layer"]),
            "head": int(item["head"]),
            "ablateTokenPositions": [int(position) for position in item.get("ablateTokenPositions", [fallback_index])],
        }
        for item in group.get("items", [])[:size]
    ]


def target_drop_from_logits(
    baseline_logits: torch.Tensor,
    ablated_logits: torch.Tensor,
    target_token_id: int,
) -> float:
    return float(baseline_logits[target_token_id] - ablated_logits[target_token_id])


def evaluate_combined_controls(
    *,
    model: Any,
    mlps: Dict[int, Any],
    attentions: Dict[int, Any],
    encoded: Dict[str, torch.Tensor],
    baseline_logits: torch.Tensor,
    mlp_row: Dict[str, Any],
    attention_row: Dict[str, Any],
    selected_mlp: List[Dict[str, Any]],
    selected_attention: List[Dict[str, Any]],
    read_token_index: int,
    target_token_id: int,
) -> Dict[str, Any]:
    mlp_size = max(1, len(selected_mlp))
    attention_size = max(1, len(selected_attention))
    mlp_groups = mlp_row.get("controls", {}).get("groups", [])
    attention_groups = attention_row.get("attentionGraph", {}).get("controls", {}).get("groups", [])
    items: List[Dict[str, Any]] = []

    def add_item(kind: str, control_class: str, mlp_items: List[Dict[str, Any]], attention_items: List[Dict[str, Any]]) -> None:
        if not mlp_items and not attention_items:
            return
        ablated = run_combined_ablation_logits(
            model=model,
            mlps=mlps,
            attentions=attentions,
            encoded=encoded,
            mlp_interventions=mlp_items,
            attention_interventions=attention_items,
            read_token_index=read_token_index,
        )
        items.append({
            "kind": kind,
            "controlClass": control_class,
            "mlpNodeCount": len(mlp_items),
            "attentionHeadCount": len(attention_items),
            "targetLogitDrop": target_drop_from_logits(baseline_logits, ablated, target_token_id),
            "mlpInterventions": mlp_items,
            "attentionInterventions": attention_items,
        })

    for group in attention_groups:
        attention_controls = attention_control_interventions(group, attention_size, read_token_index)
        add_item(f"selected-mlp-plus-{group.get('kind', 'attention-control')}", "attention-swap", selected_mlp, attention_controls)

    for group in mlp_groups:
        mlp_controls = mlp_control_interventions(group, mlp_size, read_token_index)
        add_item(f"{group.get('kind', 'mlp-control')}-plus-selected-attention", "mlp-swap", mlp_controls, selected_attention)

    for mlp_group in mlp_groups:
        mlp_controls = mlp_control_interventions(mlp_group, mlp_size, read_token_index)
        if not mlp_controls:
            continue
        for attention_group in attention_groups:
            attention_controls = attention_control_interventions(attention_group, attention_size, read_token_index)
            if not attention_controls:
                continue
            add_item(
                f"{mlp_group.get('kind', 'mlp-control')}-plus-{attention_group.get('kind', 'attention-control')}",
                "full-control",
                mlp_controls,
                attention_controls,
            )

    full_control_items = [item for item in items if item.get("controlClass") == "full-control"]
    return {
        "items": items,
        "summary": factual.summarize_drops(items),
        "gateSummary": factual.summarize_drops(full_control_items),
        "controlClassSummaries": {
            control_class: factual.summarize_drops([item for item in items if item.get("controlClass") == control_class])
            for control_class in ("attention-swap", "mlp-swap", "full-control")
        },
    }


def evaluate_combined_attempt(
    *,
    model: Any,
    mlps: Dict[int, Any],
    attentions: Dict[int, Any],
    encoded: Dict[str, torch.Tensor],
    baseline_logits: torch.Tensor,
    mlp_row: Dict[str, Any],
    attention_row: Dict[str, Any],
    selected_mlp: List[Dict[str, Any]],
    selected_attention: List[Dict[str, Any]],
    read_token_index: int,
    target_token_id: int,
    min_target_drop: float,
    min_margin_over_control_p95: float,
) -> Dict[str, Any]:
    combined_logits = run_combined_ablation_logits(
        model=model,
        mlps=mlps,
        attentions=attentions,
        encoded=encoded,
        mlp_interventions=selected_mlp,
        attention_interventions=selected_attention,
        read_token_index=read_token_index,
    )
    combined_drop = target_drop_from_logits(baseline_logits, combined_logits, target_token_id)
    controls = evaluate_combined_controls(
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
    )
    control_p95 = float(controls["gateSummary"]["p95PositiveTargetLogitDrop"])
    margin = combined_drop - control_p95
    return {
        "selectedMlp": selected_mlp,
        "selectedAttention": selected_attention,
        "combinedDrop": combined_drop,
        "controls": controls,
        "controlP95": control_p95,
        "controlMax": float(controls["gateSummary"]["maxPositiveTargetLogitDrop"]),
        "margin": margin,
        "lift": combined_drop / max(1e-9, control_p95),
        "pass": combined_drop >= min_target_drop and margin >= min_margin_over_control_p95,
    }


def select_combined_graph_attempt(
    *,
    model: Any,
    mlps: Dict[int, Any],
    attentions: Dict[int, Any],
    encoded: Dict[str, torch.Tensor],
    baseline_logits: torch.Tensor,
    mlp_row: Dict[str, Any],
    attention_row: Dict[str, Any],
    selected_mlp: List[Dict[str, Any]],
    selected_attention: List[Dict[str, Any]],
    read_token_index: int,
    target_token_id: int,
    min_target_drop: float,
    min_margin_over_control_p95: float,
) -> Dict[str, Any]:
    attempts: List[Dict[str, Any]] = []
    mlp_sizes = range(1, len(selected_mlp) + 1)
    attention_sizes = range(1, len(selected_attention) + 1)
    for mlp_size in mlp_sizes:
        for attention_size in attention_sizes:
            attempt = evaluate_combined_attempt(
                model=model,
                mlps=mlps,
                attentions=attentions,
                encoded=encoded,
                baseline_logits=baseline_logits,
                mlp_row=mlp_row,
                attention_row=attention_row,
                selected_mlp=selected_mlp[:mlp_size],
                selected_attention=selected_attention[:attention_size],
                read_token_index=read_token_index,
                target_token_id=target_token_id,
                min_target_drop=min_target_drop,
                min_margin_over_control_p95=min_margin_over_control_p95,
            )
            attempts.append(attempt)
    passing = [attempt for attempt in attempts if bool(attempt["pass"])]
    if passing:
        chosen = sorted(
            passing,
            key=lambda item: (
                len(item["selectedMlp"]) + len(item["selectedAttention"]),
                -float(item["margin"]),
                -float(item["combinedDrop"]),
            ),
        )[0]
        selection_mode = "minimal-passing-combined-prefix"
    else:
        chosen = max(attempts, key=lambda item: (float(item["margin"]), float(item["combinedDrop"])))
        selection_mode = "best-margin-combined-prefix"
    return {
        **chosen,
        "selectionMode": selection_mode,
        "attempts": [
            {
                "mlpNodeCount": len(attempt["selectedMlp"]),
                "attentionHeadCount": len(attempt["selectedAttention"]),
                "targetLogitDrop": float(attempt["combinedDrop"]),
                "controlGraphP95PositiveTargetLogitDrop": float(attempt["controlP95"]),
                "marginOverControlGraphP95": float(attempt["margin"]),
                "pass": bool(attempt["pass"]),
            }
            for attempt in attempts
        ],
    }


def run_component_case(
    *,
    component: str,
    model: Any,
    tokenizer: Any,
    primary_modules: Dict[int, Any],
    fallback_modules: Dict[int, Any],
    fact: Dict[str, str],
    answer_pool: List[Dict[str, Any]],
    args: argparse.Namespace,
    rng: random.Random,
) -> Dict[str, Any]:
    if component == "mlp":
        row = factual.run_case(
            model=model,
            tokenizer=tokenizer,
            mlps=primary_modules,
            fact=fact,
            answer_pool=answer_pool,
            args=namespace_for_mlp(args),
            rng=rng,
        )
        if row.get("status") == "failed" and fallback_modules and set(fallback_modules) != set(primary_modules):
            fallback_row = factual.run_case(
                model=model,
                tokenizer=tokenizer,
                mlps=fallback_modules,
                fact=fact,
                answer_pool=answer_pool,
                args=namespace_for_mlp(args),
                rng=rng,
            )
            if fallback_row.get("status") == "passed":
                fallback_row["adaptiveLayerExpansion"] = True
                return fallback_row
        return row
    if component == "attention":
        row = attention.run_case(
            model=model,
            tokenizer=tokenizer,
            attentions=primary_modules,
            fact=fact,
            answer_pool=answer_pool,
            args=namespace_for_attention(args),
            rng=rng,
        )
        if row.get("status") == "failed" and fallback_modules and set(fallback_modules) != set(primary_modules):
            fallback_row = attention.run_case(
                model=model,
                tokenizer=tokenizer,
                attentions=fallback_modules,
                fact=fact,
                answer_pool=answer_pool,
                args=namespace_for_attention(args),
                rng=rng,
            )
            if fallback_row.get("status") == "passed":
                fallback_row["adaptiveLayerExpansion"] = True
                return fallback_row
        return row
    raise RuntimeError(f"Unknown component {component}")


def compact_mlp_row(row: Dict[str, Any]) -> Dict[str, Any]:
    graph = row.get("circuitGraph", {})
    return {
        "status": row.get("status"),
        "target": row.get("target"),
        "selectedNodeCount": graph.get("selectedNodeCount", 0),
        "selectedInterventions": graph.get("selectedInterventions", []),
        "jointAblation": graph.get("jointAblation", {}),
        "hardeningSummary": {
            "pass": row.get("hardening", {}).get("pass"),
            "paraphrase": row.get("hardening", {}).get("paraphrases", {}).get("summary"),
            "wrongAnswer": row.get("hardening", {}).get("wrongAnswerTargets", {}).get("summary"),
            "negativeFacts": row.get("hardening", {}).get("negativeFacts", {}).get("summary"),
            "unrelatedLogits": row.get("hardening", {}).get("unrelatedLogits", {}).get("summary"),
        },
    }


def compact_attention_row(row: Dict[str, Any]) -> Dict[str, Any]:
    graph = row.get("attentionGraph", {})
    return {
        "status": row.get("status"),
        "target": row.get("target"),
        "selectedHeadCount": len(row.get("selectedHeads", [])),
        "selectedInterventions": graph.get("selectedInterventions", []),
        "jointAblation": graph.get("jointAblation", {}),
    }


def run_combined_case(
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
) -> Dict[str, Any]:
    prompt = factual.prompt_for_fact(fact)
    target = factual.encode_single_answer_token(tokenizer, fact["answer"])
    if target is None:
        return {"subject": fact["subject"], "answer": fact["answer"], "prompt": prompt, "status": "skipped", "reason": "answer is not a single token"}
    target_token_id, target_token = target
    device = next(model.parameters()).device
    encoded = encode_prompt(tokenizer, prompt, device)
    token_ids = encoded["input_ids"][0].detach().cpu()
    read_token_index = int(token_ids.numel()) - 1
    with torch.no_grad():
        baseline_logits = model(**encoded).logits.detach().cpu()[0, read_token_index]

    mlp_row = run_component_case(
        component="mlp",
        model=model,
        tokenizer=tokenizer,
        primary_modules=primary_mlps,
        fallback_modules=fallback_mlps,
        fact=fact,
        answer_pool=answer_pool,
        args=args,
        rng=rng,
    )
    attention_row = run_component_case(
        component="attention",
        model=model,
        tokenizer=tokenizer,
        primary_modules=primary_attentions,
        fallback_modules=fallback_attentions,
        fact=fact,
        answer_pool=answer_pool,
        args=args,
        rng=rng,
    )
    if mlp_row.get("status") == "skipped" or attention_row.get("status") == "skipped":
        return {"subject": fact["subject"], "answer": fact["answer"], "prompt": prompt, "status": "skipped", "reason": "component skipped"}

    selected_mlp = mlp_row.get("circuitGraph", {}).get("selectedInterventions", [])
    selected_attention = attention_row.get("attentionGraph", {}).get("selectedInterventions", [])
    component_pass = mlp_row.get("status") == "passed" and attention_row.get("status") == "passed"
    if not selected_mlp or not selected_attention:
        return {
            "subject": fact["subject"],
            "answer": fact["answer"],
            "prompt": prompt,
            "status": "failed",
            "failureReason": "missing selected MLP or attention interventions",
            "componentPass": component_pass,
            "mlp": compact_mlp_row(mlp_row),
            "attention": compact_attention_row(attention_row),
        }

    chosen = select_combined_graph_attempt(
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
    selected_mlp = chosen["selectedMlp"]
    selected_attention = chosen["selectedAttention"]
    combined_drop = float(chosen["combinedDrop"])
    controls = chosen["controls"]
    control_p95 = float(chosen["controlP95"])
    margin = float(chosen["margin"])
    mlp_drop = float(mlp_row.get("circuitGraph", {}).get("jointAblation", {}).get("targetLogitDrop", 0.0))
    attention_drop = float(attention_row.get("attentionGraph", {}).get("jointAblation", {}).get("targetLogitDrop", 0.0))
    pass_case = (
        component_pass
        and bool(chosen["pass"])
    )
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
        "componentPass": component_pass,
        "mlp": compact_mlp_row(mlp_row),
        "attention": compact_attention_row(attention_row),
        "combinedGraph": {
            "kind": "gpt2-live-factual-attention-mlp-combined-circuit-graph",
            "nodes": {
                "mlp": selected_mlp,
                "attention": selected_attention,
            },
            "edges": [
                {
                    "source": f"attention:{item['layer']}:{item['head']}",
                    "target": "selected-circuit",
                    "kind": "selected-attention-path",
                    "ablateTokenPositions": item.get("ablateTokenPositions", []),
                }
                for item in selected_attention
            ] + [
                {
                    "source": f"mlp:{item['layer']}:{item['neuron']}",
                    "target": "selected-circuit",
                    "kind": "selected-mlp-node",
                    "ablateTokenPositions": item.get("ablateTokenPositions", []),
                }
                for item in selected_mlp
            ] + [{
                "source": "selected-circuit",
                "target": f"target:{target_token_id}",
                "kind": "joint-attention-mlp-ablation",
                "targetLogitDrop": combined_drop,
            }],
            "jointAblation": {
                "targetLogitDrop": combined_drop,
                "mlpOnlyTargetLogitDrop": mlp_drop,
                "attentionOnlyTargetLogitDrop": attention_drop,
                "controlGraphP95PositiveTargetLogitDrop": control_p95,
                "controlGraphMaxPositiveTargetLogitDrop": float(chosen["controlMax"]),
                "marginOverControlGraphP95": margin,
                "liftOverControlGraphP95": float(chosen["lift"]),
            },
            "controlGraphs": controls,
            "graphSelection": {
                "mode": chosen["selectionMode"],
                "attempts": chosen["attempts"],
            },
            "gate": {
                "pass": pass_case,
                "requiresComponentPass": True,
                "minTargetDrop": float(args.min_target_drop),
                "minMarginOverControlP95": float(args.min_margin_over_control_p95),
            },
        },
    }


def make_report_md(report: Dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Live GPT-2 Combined Attention+MLP Circuit Finder",
        "",
        "This benchmark uses a real GPT-2 model, real attention-path hooks, real MLP activation hooks, automatic component discovery, and simultaneous attention+MLP causal ablation. It is not synthetic and does not fit a surrogate.",
        "",
        "## Summary",
        "",
        f"- Pass: `{summary['pass']}`",
        f"- Valid cases: `{summary['validCases']}`",
        f"- Passed cases: `{summary['passedCases']}`",
        f"- Pass rate: `{summary['passRate']}`",
        f"- MLP component pass rate: `{summary['mlpComponentPassRate']}`",
        f"- Attention component pass rate: `{summary['attentionComponentPassRate']}`",
        f"- Median combined target-logit drop: `{summary['medianCombinedTargetLogitDrop']}`",
        f"- Median combined control p95: `{summary['medianCombinedControlP95PositiveTargetLogitDrop']}`",
        f"- Median combined lift over control p95: `{summary['medianCombinedLiftOverControlP95']}`",
        f"- Mean selected MLP nodes: `{summary['meanSelectedMlpNodes']}`",
        f"- Mean selected attention heads: `{summary['meanSelectedAttentionHeads']}`",
        "",
        "## Claim Boundary",
        "",
        report["claimBoundary"]["provesIfPasses"],
        "",
        report["claimBoundary"]["doesNotProve"],
        "",
        "## Cases",
        "",
        "| subject | status | mlp | attention | combined drop | control p95 | margin |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["cases"]:
        if row.get("status") == "skipped":
            continue
        graph = row.get("combinedGraph", {}).get("jointAblation", {})
        lines.append(
            f"| {row['subject']} | {row['status']} | {row.get('mlp', {}).get('status')} | "
            f"{row.get('attention', {}).get('status')} | {float(graph.get('targetLogitDrop', 0.0)):.6f} | "
            f"{float(graph.get('controlGraphP95PositiveTargetLogitDrop', 0.0)):.6f} | "
            f"{float(graph.get('marginOverControlGraphP95', 0.0)):.6f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
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
    rng = random.Random(args.control_seed)

    cases = []
    valid_seen = 0
    for fact in factual.FACTS:
        if only_subjects and fact["subject"] not in only_subjects:
            continue
        if valid_seen >= args.case_count:
            break
        row = run_combined_case(
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
            rng=rng,
        )
        if row.get("status") != "skipped":
            valid_seen += 1
        cases.append(row)
        graph = row.get("combinedGraph", {}).get("jointAblation", {})
        print(json.dumps({
            "case": valid_seen,
            "subject": row["subject"],
            "status": row["status"],
            "mlp": row.get("mlp", {}).get("status"),
            "attention": row.get("attention", {}).get("status"),
            "combinedDrop": graph.get("targetLogitDrop", 0.0),
            "margin": graph.get("marginOverControlGraphP95", 0.0),
        }))
        sys.stdout.flush()

    valid_cases = [row for row in cases if row.get("status") != "skipped"]
    passed_cases = [row for row in valid_cases if row.get("status") == "passed"]
    component_mlp_passed = [row for row in valid_cases if row.get("mlp", {}).get("status") == "passed"]
    component_attention_passed = [row for row in valid_cases if row.get("attention", {}).get("status") == "passed"]
    graph_drops = [float(row.get("combinedGraph", {}).get("jointAblation", {}).get("targetLogitDrop", 0.0)) for row in valid_cases]
    control_p95 = [float(row.get("combinedGraph", {}).get("jointAblation", {}).get("controlGraphP95PositiveTargetLogitDrop", 0.0)) for row in valid_cases]
    graph_lifts = [float(row.get("combinedGraph", {}).get("jointAblation", {}).get("liftOverControlGraphP95", 0.0)) for row in valid_cases]
    mlp_counts = [int(row.get("mlp", {}).get("selectedNodeCount", 0)) for row in valid_cases]
    attention_counts = [int(row.get("attention", {}).get("selectedHeadCount", 0)) for row in valid_cases]
    pass_rate = len(passed_cases) / max(1, len(valid_cases))
    mlp_component_rate = len(component_mlp_passed) / max(1, len(valid_cases))
    attention_component_rate = len(component_attention_passed) / max(1, len(valid_cases))
    pass_summary = (
        len(valid_cases) >= args.min_valid_cases
        and pass_rate >= args.min_pass_rate
        and mlp_component_rate >= args.min_component_pass_rate
        and attention_component_rate >= args.min_component_pass_rate
    )
    report = {
        "schemaVersion": 1,
        "kind": "gpt2-live-factual-attention-mlp-combined-circuit-finder",
        "mode": "live-gpt2-attention-mlp-combined-circuit-finder",
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
            "device": device,
        },
        "summary": {
            "pass": pass_summary,
            "validCases": len(valid_cases),
            "passedCases": len(passed_cases),
            "failedCases": len(valid_cases) - len(passed_cases),
            "passRate": pass_rate,
            "mlpComponentPassRate": mlp_component_rate,
            "attentionComponentPassRate": attention_component_rate,
            "medianCombinedTargetLogitDrop": factual.percentile(graph_drops, 0.50),
            "p10CombinedTargetLogitDrop": factual.percentile(graph_drops, 0.10),
            "medianCombinedControlP95PositiveTargetLogitDrop": factual.percentile(control_p95, 0.50),
            "medianCombinedLiftOverControlP95": factual.percentile(graph_lifts, 0.50),
            "meanSelectedMlpNodes": sum(mlp_counts) / max(1, len(mlp_counts)),
            "meanSelectedAttentionHeads": sum(attention_counts) / max(1, len(attention_counts)),
            "failedSubjects": [row["subject"] for row in valid_cases if row.get("status") != "passed"],
        },
        "cases": cases,
        "claimBoundary": {
            "provesIfPasses": "Benchmark-scoped evidence that the real GPT-2 live pipeline can discover selected MLP nodes and attention heads, build one token/head/neuron/target graph, and validate the union through simultaneous causal ablation above combined controls across a factual prompt set.",
            "doesNotProve": "This is not full GPT-2 circuit recovery, not ordered path-patching through every residual stream edge, not full FFN extraction, not arbitrary LLM understanding, not output-only original-neuron identity, and not a broad SOTA claim.",
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
