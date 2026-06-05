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


def load_factual_sweep() -> Any:
    source_path = Path(__file__).resolve().parent / "llm-reader-gpt2-factual-sweep.py"
    spec = importlib.util.spec_from_file_location("gcl_factual_sweep", source_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load factual sweep helpers from {source_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


factual = load_factual_sweep()
live = GPT2OracleAdapter()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a real GPT-2 factual-prompt attention-head/path graph sweep."
    )
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--case-count", type=int, default=50)
    parser.add_argument("--min-valid-cases", type=int, default=50)
    parser.add_argument("--only-subjects", default="")
    parser.add_argument("--layers", default="6,9,11")
    parser.add_argument("--fallback-layers", default="0,1,2,3,4,5,6,7,8,9,10,11")
    parser.add_argument("--control-count", type=int, default=8)
    parser.add_argument("--control-prompt", default="The weather in Berlin is")
    parser.add_argument("--control-seed", type=int, default=260605)
    parser.add_argument("--max-graph-heads", type=int, default=3)
    parser.add_argument("--min-pass-rate", type=float, default=0.90)
    parser.add_argument("--min-graph-drop", type=float, default=0.05)
    parser.add_argument("--min-margin-over-control-p95", type=float, default=0.02)
    parser.add_argument("--wrong-answer-count", type=int, default=3)
    parser.add_argument("--max-wrong-answer-drop-ratio", type=float, default=0.80)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_layers(value: str) -> List[int]:
    layers = factual.parse_layers(value)
    return sorted(set(int(layer) for layer in layers))


def encode_prompt(tokenizer: Any, prompt: str, device: torch.device | str) -> Dict[str, torch.Tensor]:
    encoded = tokenizer(prompt, return_tensors="pt")
    return {key: value.to(device) for key, value in encoded.items()}


def top_logit_token_id(logits: torch.Tensor) -> int:
    return int(torch.argmax(logits.detach().cpu()).item())


def target_drop(
    *,
    model: Any,
    attentions: Dict[int, Any],
    encoded: Dict[str, torch.Tensor],
    baseline_logits: torch.Tensor,
    interventions: List[Dict[str, Any]],
    read_token_index: int,
    target_token_id: int,
) -> Tuple[float, torch.Tensor]:
    ablated = live.run_attention_heads_ablation_logits(
        model,
        attentions,
        encoded,
        interventions,
        read_token_index,
    )
    return float(baseline_logits[target_token_id] - ablated[target_token_id]), ablated


def scan_attention_heads(
    *,
    model: Any,
    attentions: Dict[int, Any],
    encoded: Dict[str, torch.Tensor],
    baseline_logits: torch.Tensor,
    read_token_index: int,
    target_token_id: int,
    wrong_answer_targets: List[Dict[str, Any]],
    ablate_positions: List[int],
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for layer, attention in attentions.items():
        for head in range(live.attention_head_count(attention)):
            interventions = [{
                "layer": int(layer),
                "head": int(head),
                "ablateTokenPositions": ablate_positions,
            }]
            drop, ablated_logits = target_drop(
                model=model,
                attentions={int(layer): attention},
                encoded=encoded,
                baseline_logits=baseline_logits,
                interventions=interventions,
                read_token_index=read_token_index,
                target_token_id=target_token_id,
            )
            wrong_drops = []
            for wrong in wrong_answer_targets:
                wrong_token_id = int(wrong["tokenId"])
                wrong_drops.append({
                    "subject": wrong["subject"],
                    "answer": wrong["answer"],
                    "tokenId": wrong_token_id,
                    "token": wrong["token"],
                    "logitDrop": float(baseline_logits[wrong_token_id] - ablated_logits[wrong_token_id]),
                })
            max_wrong_drop = max([float(item["logitDrop"]) for item in wrong_drops] + [0.0])
            candidates.append({
                "layer": int(layer),
                "head": int(head),
                "component": f"transformer.h.{int(layer)}.attn.head[{int(head)}]",
                "ablateTokenPositions": list(ablate_positions),
                "targetLogitDrop": drop,
                "wrongAnswerLogitDrops": wrong_drops,
                "maxWrongAnswerLogitDrop": max_wrong_drop,
                "targetSpecificMargin": drop - max_wrong_drop,
                "targetSpecificLift": drop / max(1e-9, max_wrong_drop),
            })
    return sorted(
        candidates,
        key=lambda item: (
            float(item["targetSpecificMargin"]),
            float(item["targetLogitDrop"]),
            float(item["targetSpecificLift"]),
        ),
        reverse=True,
    )


def select_attention_graph(
    candidates: List[Dict[str, Any]],
    max_graph_heads: int,
    min_graph_drop: float,
) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    seen = set()
    for candidate in candidates:
        key = (int(candidate["layer"]), int(candidate["head"]))
        if key in seen:
            continue
        drop = float(candidate["targetLogitDrop"])
        if drop < min_graph_drop:
            continue
        selected.append(candidate)
        seen.add(key)
        if len(selected) >= max_graph_heads:
            break
    return selected


def summarize_drops(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    return factual.summarize_drops(items)


def graph_interventions(selected: List[Dict[str, Any]], token_index: int) -> List[Dict[str, Any]]:
    return [
        {
            "layer": int(item["layer"]),
            "head": int(item["head"]),
            "ablateTokenPositions": [int(position) for position in item.get("ablateTokenPositions", [token_index])],
        }
        for item in selected
    ]


def control_graphs(
    *,
    model: Any,
    attentions: Dict[int, Any],
    encoded: Dict[str, torch.Tensor],
    baseline_logits: torch.Tensor,
    controls: Dict[str, Any],
    graph_size: int,
    read_token_index: int,
    target_token_id: int,
) -> Dict[str, Any]:
    items = []
    for group in controls["groups"]:
        selected = group["items"][:graph_size]
        if not selected:
            continue
        interventions = graph_interventions(selected, read_token_index)
        drop, _ = target_drop(
            model=model,
            attentions=attentions,
            encoded=encoded,
            baseline_logits=baseline_logits,
            interventions=interventions,
            read_token_index=read_token_index,
            target_token_id=target_token_id,
        )
        items.append({
            "kind": group["kind"],
            "headCount": len(interventions),
            "targetLogitDrop": drop,
            "interventions": interventions,
        })
    return {"items": items, "summary": summarize_drops(items)}


def make_control_item(
    *,
    model: Any,
    attentions: Dict[int, Any],
    encoded: Dict[str, torch.Tensor],
    baseline_logits: torch.Tensor,
    layer: int,
    head: int,
    positions: List[int],
    read_token_index: int,
    target_token_id: int,
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    interventions = [{"layer": int(layer), "head": int(head), "ablateTokenPositions": positions}]
    drop, _ = target_drop(
        model=model,
        attentions=attentions,
        encoded=encoded,
        baseline_logits=baseline_logits,
        interventions=interventions,
        read_token_index=read_token_index,
        target_token_id=target_token_id,
    )
    return {
        **metadata,
        "layer": int(layer),
        "head": int(head),
        "component": f"transformer.h.{int(layer)}.attn.head[{int(head)}]",
        "ablateTokenPositions": positions,
        "targetLogitDrop": drop,
    }


def build_controls(
    *,
    model: Any,
    tokenizer: Any,
    attentions: Dict[int, Any],
    encoded: Dict[str, torch.Tensor],
    baseline_logits: torch.Tensor,
    selected: List[Dict[str, Any]],
    token_ids: torch.Tensor,
    read_token_index: int,
    target_token_id: int,
    control_count: int,
    control_prompt: str,
    rng: random.Random,
) -> Dict[str, Any]:
    used = {(int(item["layer"]), int(item["head"])) for item in selected}
    all_heads = [
        (int(layer), int(head))
        for layer, attention in attentions.items()
        for head in range(live.attention_head_count(attention))
    ]
    groups = []

    random_pool = [item for item in all_heads if item not in used]
    rng.shuffle(random_pool)
    random_items = [
        make_control_item(
            model=model,
            attentions=attentions,
            encoded=encoded,
            baseline_logits=baseline_logits,
            layer=layer,
            head=head,
            positions=[read_token_index],
            read_token_index=read_token_index,
            target_token_id=target_token_id,
            metadata={"kind": "random-attention-head"},
        )
        for layer, head in random_pool[:control_count]
    ]
    groups.append({"kind": "random-attention-head", "summary": summarize_drops(random_items), "items": random_items})

    device = encoded["input_ids"].device
    control_encoded = encode_prompt(tokenizer, control_prompt, device)
    control_read_index = int(control_encoded["input_ids"].shape[1]) - 1
    with torch.no_grad():
        control_logits = model(**control_encoded).logits.detach().cpu()[0, control_read_index]
    control_target = top_logit_token_id(control_logits)
    off_prompt_candidates = scan_attention_heads(
        model=model,
        attentions=attentions,
        encoded=control_encoded,
        baseline_logits=control_logits,
        read_token_index=control_read_index,
        target_token_id=control_target,
        wrong_answer_targets=[],
        ablate_positions=[control_read_index],
    )
    off_prompt_items = []
    for candidate in off_prompt_candidates:
        key = (int(candidate["layer"]), int(candidate["head"]))
        if key in used:
            continue
        off_prompt_items.append(make_control_item(
            model=model,
            attentions=attentions,
            encoded=encoded,
            baseline_logits=baseline_logits,
            layer=int(candidate["layer"]),
            head=int(candidate["head"]),
            positions=[read_token_index],
            read_token_index=read_token_index,
            target_token_id=target_token_id,
            metadata={
                "kind": "top-head-from-other-prompt",
                "controlPrompt": control_prompt,
                "controlPromptTargetTokenId": int(control_target),
                "controlPromptTargetToken": live.clean_token(tokenizer, int(control_target)),
                "controlPromptTargetDrop": float(candidate["targetLogitDrop"]),
            },
        ))
        if len(off_prompt_items) >= control_count:
            break
    groups.append({
        "kind": "top-head-from-other-prompt",
        "summary": summarize_drops(off_prompt_items),
        "items": off_prompt_items,
    })

    if selected:
        primary = selected[0]
        other_position_items = [
            make_control_item(
                model=model,
                attentions=attentions,
                encoded=encoded,
                baseline_logits=baseline_logits,
                layer=int(primary["layer"]),
                head=int(primary["head"]),
                positions=[int(position)],
                read_token_index=read_token_index,
                target_token_id=target_token_id,
                metadata={
                    "kind": "same-head-other-token-position",
                    "sourceHead": int(primary["head"]),
                    "sourceLayer": int(primary["layer"]),
                    "otherTokenIndex": int(position),
                    "otherToken": live.clean_token(tokenizer, int(token_ids[int(position)])),
                },
            )
            for position in range(int(token_ids.numel()))
            if int(position) != int(read_token_index)
        ]
    else:
        other_position_items = []
    groups.append({
        "kind": "same-head-other-token-position",
        "summary": summarize_drops(other_position_items),
        "items": other_position_items,
    })

    return {"controlCount": control_count, "groups": groups}


def run_case(
    *,
    model: Any,
    tokenizer: Any,
    attentions: Dict[int, Any],
    fact: Dict[str, str],
    answer_pool: List[Dict[str, Any]],
    args: argparse.Namespace,
    rng: random.Random,
) -> Dict[str, Any]:
    target = factual.encode_single_answer_token(tokenizer, fact["answer"])
    if target is None:
        return {"subject": fact["subject"], "status": "skipped", "reason": "answer is not a single token"}
    target_token_id, target_token = target
    wrong_answer_targets = factual.choose_wrong_answer_tokens(answer_pool, target_token_id, args.wrong_answer_count)
    prompt = factual.prompt_for_fact(fact)
    device = next(model.parameters()).device
    encoded = encode_prompt(tokenizer, prompt, device)
    token_ids = encoded["input_ids"][0].detach().cpu()
    read_token_index = int(token_ids.numel()) - 1
    with torch.no_grad():
        baseline_logits = model(**encoded).logits.detach().cpu()[0, read_token_index]

    candidates = scan_attention_heads(
        model=model,
        attentions=attentions,
        encoded=encoded,
        baseline_logits=baseline_logits,
        read_token_index=read_token_index,
        target_token_id=target_token_id,
        wrong_answer_targets=wrong_answer_targets,
        ablate_positions=[read_token_index],
    )
    selected = select_attention_graph(
        candidates,
        max_graph_heads=max(1, int(args.max_graph_heads)),
        min_graph_drop=float(args.min_graph_drop),
    )
    controls = build_controls(
        model=model,
        tokenizer=tokenizer,
        attentions=attentions,
        encoded=encoded,
        baseline_logits=baseline_logits,
        selected=selected,
        token_ids=token_ids,
        read_token_index=read_token_index,
        target_token_id=target_token_id,
        control_count=max(1, int(args.control_count)),
        control_prompt=args.control_prompt,
        rng=rng,
    )
    interventions = graph_interventions(selected, read_token_index)
    if interventions:
        joint_drop, joint_logits = target_drop(
            model=model,
            attentions=attentions,
            encoded=encoded,
            baseline_logits=baseline_logits,
            interventions=interventions,
            read_token_index=read_token_index,
            target_token_id=target_token_id,
        )
    else:
        joint_drop, joint_logits = 0.0, baseline_logits
    wrong_graph_drops = []
    for wrong in wrong_answer_targets:
        token_id = int(wrong["tokenId"])
        wrong_graph_drops.append({
            "subject": wrong["subject"],
            "answer": wrong["answer"],
            "tokenId": token_id,
            "token": wrong["token"],
            "logitDrop": float(baseline_logits[token_id] - joint_logits[token_id]),
        })
    max_wrong_graph_drop = max([float(item["logitDrop"]) for item in wrong_graph_drops] + [0.0])
    controls_graph = control_graphs(
        model=model,
        attentions=attentions,
        encoded=encoded,
        baseline_logits=baseline_logits,
        controls=controls,
        graph_size=max(1, len(interventions)),
        read_token_index=read_token_index,
        target_token_id=target_token_id,
    )
    control_p95 = float(controls_graph["summary"]["p95PositiveTargetLogitDrop"])
    margin = joint_drop - control_p95
    wrong_ratio = max_wrong_graph_drop / max(1e-9, joint_drop)
    pass_case = (
        len(selected) > 0
        and joint_drop >= float(args.min_graph_drop)
        and margin >= float(args.min_margin_over_control_p95)
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
        "topLogits": factual.top_logit_tokens(tokenizer, baseline_logits, 8),
        "candidateHeads": candidates[:12],
        "selectedHeads": selected,
        "attentionGraph": {
            "kind": "gpt2-live-factual-attention-path-graph",
            "selectedInterventions": interventions,
            "jointAblation": {
                "targetLogitDrop": joint_drop,
                "controlGraphP95PositiveTargetLogitDrop": control_p95,
                "marginOverControlGraphP95": margin,
                "liftOverControlGraphP95": joint_drop / max(1e-9, control_p95),
                "wrongAnswerLogitDrops": wrong_graph_drops,
                "maxWrongAnswerLogitDrop": max_wrong_graph_drop,
                "maxWrongAnswerDropRatio": wrong_ratio,
                "wrongAnswerSpecificityIsGated": False,
            },
            "controlGraphs": controls_graph,
            "controls": controls,
            "gate": {
                "pass": pass_case,
                "minGraphDrop": float(args.min_graph_drop),
                "minMarginOverControlP95": float(args.min_margin_over_control_p95),
                "maxWrongAnswerDropRatio": float(args.max_wrong_answer_drop_ratio),
                "wrongAnswerSpecificityIsGated": False,
            },
        },
    }


def make_report_md(report: Dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Live GPT-2 Attention Path Circuit Finder",
        "",
        "This benchmark uses a real GPT-2 model, real attention-path hooks, automatic layer/head scanning, joint attention-head ablations, and attention-head negative controls. It is not synthetic and does not fit a surrogate.",
        "",
        "## Summary",
        "",
        f"- Pass: `{summary['pass']}`",
        f"- Valid cases: `{summary['validCases']}`",
        f"- Passed cases: `{summary['passedCases']}`",
        f"- Pass rate: `{summary['passRate']}`",
        f"- Median attention graph target-logit drop: `{summary['medianGraphJointTargetLogitDrop']}`",
        f"- Median attention control p95 target-logit drop: `{summary['medianGraphControlP95PositiveTargetLogitDrop']}`",
        f"- Median attention graph lift over control p95: `{summary['medianGraphLiftOverControlP95']}`",
        f"- Mean selected attention heads: `{summary['meanSelectedHeadCount']}`",
        f"- Unique selected attention heads: `{summary['uniqueSelectedHeads']}`",
        f"- Median wrong-answer collateral ratio: `{summary['medianWrongAnswerDropRatio']}`",
        "",
        "## Claim Boundary",
        "",
        report["claimBoundary"]["provesIfPasses"],
        "",
        report["claimBoundary"]["doesNotProve"],
        "",
        "## Cases",
        "",
        "| subject | status | target rank | heads | graph drop | control p95 | margin | wrong ratio |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["cases"]:
        if row.get("status") == "skipped":
            continue
        graph = row["attentionGraph"]["jointAblation"]
        lines.append(
            f"| {row['subject']} | {row['status']} | {row['target']['baselineRank']} | "
            f"{len(row['selectedHeads'])} | {graph['targetLogitDrop']:.6f} | "
            f"{graph['controlGraphP95PositiveTargetLogitDrop']:.6f} | "
            f"{graph['marginOverControlGraphP95']:.6f} | {graph['maxWrongAnswerDropRatio']:.6f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if args.case_count <= 0 or args.min_valid_cases <= 0:
        raise RuntimeError("case counts must be positive.")
    if args.max_graph_heads <= 0 or args.control_count <= 0:
        raise RuntimeError("--max-graph-heads and --control-count must be positive.")
    only_subjects = {item.strip() for item in args.only_subjects.split(",") if item.strip()}
    layers = parse_layers(args.layers)
    fallback_layers = parse_layers(args.fallback_layers) if args.fallback_layers.strip() else []
    all_required_layers = sorted(set(layers) | set(fallback_layers))
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")

    tokenizer, model = live.load_model_and_tokenizer(args.model, local_files_only=args.local_files_only, device=device)
    all_attentions = live.attention_modules(model, all_required_layers)
    primary_attentions = {layer: all_attentions[layer] for layer in layers}
    fallback_attentions = {layer: all_attentions[layer] for layer in fallback_layers} if fallback_layers else {}
    rng = random.Random(args.control_seed)
    answer_pool = factual.build_answer_pool(tokenizer)

    cases = []
    valid_seen = 0
    for fact in factual.FACTS:
        if only_subjects and fact["subject"] not in only_subjects:
            continue
        if valid_seen >= args.case_count:
            break
        row = run_case(
            model=model,
            tokenizer=tokenizer,
            attentions=primary_attentions,
            fact=fact,
            answer_pool=answer_pool,
            args=args,
            rng=rng,
        )
        if row["status"] != "skipped":
            valid_seen += 1
        if row["status"] == "failed" and fallback_attentions and set(fallback_attentions) != set(primary_attentions):
            fallback_row = run_case(
                model=model,
                tokenizer=tokenizer,
                attentions=fallback_attentions,
                fact=fact,
                answer_pool=answer_pool,
                args=args,
                rng=rng,
            )
            if fallback_row["status"] == "passed":
                fallback_row["adaptiveLayerExpansion"] = True
                row = fallback_row
            else:
                row["fallbackAttempt"] = {
                    "status": fallback_row["status"],
                    "graphDrop": fallback_row.get("attentionGraph", {}).get("jointAblation", {}).get("targetLogitDrop"),
                    "margin": fallback_row.get("attentionGraph", {}).get("jointAblation", {}).get("marginOverControlGraphP95"),
                }
        cases.append(row)
        print(json.dumps({
            "case": len([item for item in cases if item.get("status") != "skipped"]),
            "subject": row["subject"],
            "status": row["status"],
            "graphDrop": row.get("attentionGraph", {}).get("jointAblation", {}).get("targetLogitDrop", 0.0),
            "margin": row.get("attentionGraph", {}).get("jointAblation", {}).get("marginOverControlGraphP95", 0.0),
            "adaptiveLayerExpansion": bool(row.get("adaptiveLayerExpansion", False)),
        }))
        sys.stdout.flush()

    valid_cases = [row for row in cases if row.get("status") != "skipped"]
    passed_cases = [row for row in valid_cases if row.get("status") == "passed"]
    graph_drops = [float(row["attentionGraph"]["jointAblation"]["targetLogitDrop"]) for row in valid_cases]
    control_p95 = [float(row["attentionGraph"]["jointAblation"]["controlGraphP95PositiveTargetLogitDrop"]) for row in valid_cases]
    graph_lifts = [float(row["attentionGraph"]["jointAblation"]["liftOverControlGraphP95"]) for row in valid_cases]
    head_counts = [len(row["selectedHeads"]) for row in valid_cases]
    wrong_ratios = [
        float(row["attentionGraph"]["jointAblation"]["maxWrongAnswerDropRatio"])
        for row in valid_cases
    ]
    selected_heads = [
        f"{int(head['layer'])}:{int(head['head'])}"
        for row in valid_cases
        for head in row["selectedHeads"]
    ]
    pass_rate = len(passed_cases) / max(1, len(valid_cases))
    pass_summary = len(valid_cases) >= args.min_valid_cases and pass_rate >= args.min_pass_rate
    report = {
        "schemaVersion": 1,
        "kind": "gpt2-live-factual-attention-path-circuit-finder",
        "mode": "live-gpt2-attention-path-circuit-finder",
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
            "layers": layers,
            "fallbackLayers": fallback_layers,
            "caseCount": args.case_count,
            "minValidCases": args.min_valid_cases,
            "controlCount": args.control_count,
            "maxGraphHeads": args.max_graph_heads,
            "minPassRate": args.min_pass_rate,
            "minGraphDrop": args.min_graph_drop,
            "minMarginOverControlP95": args.min_margin_over_control_p95,
            "wrongAnswerCount": args.wrong_answer_count,
            "maxWrongAnswerDropRatio": args.max_wrong_answer_drop_ratio,
            "device": device,
        },
        "summary": {
            "pass": pass_summary,
            "validCases": len(valid_cases),
            "passedCases": len(passed_cases),
            "failedCases": len(valid_cases) - len(passed_cases),
            "passRate": pass_rate,
            "medianGraphJointTargetLogitDrop": factual.percentile(graph_drops, 0.50),
            "p10GraphJointTargetLogitDrop": factual.percentile(graph_drops, 0.10),
            "medianGraphControlP95PositiveTargetLogitDrop": factual.percentile(control_p95, 0.50),
            "medianGraphLiftOverControlP95": factual.percentile(graph_lifts, 0.50),
            "meanSelectedHeadCount": sum(head_counts) / max(1, len(head_counts)),
            "uniqueSelectedHeads": len(set(selected_heads)),
            "medianWrongAnswerDropRatio": factual.percentile(wrong_ratios, 0.50),
            "wrongAnswerSpecificityIsGated": False,
            "failedSubjects": [row["subject"] for row in valid_cases if row["status"] != "passed"],
        },
        "cases": cases,
        "claimBoundary": {
            "provesIfPasses": "Benchmark-scoped evidence that the real GPT-2 live pipeline can scan attention heads, build token/head/target path graph evidence, and validate selected attention paths through causal head ablation above attention-head controls across a factual prompt set.",
            "doesNotProve": "This is not answer-specific factual identity. Attention heads can carry broad task or token-class effects, so wrong-answer collateral is reported but not gated in this Stage-K path graph. This is also not full attention-circuit recovery across all model components, not attention-to-MLP composition recovery, not full GPT-2 FFN extraction, not arbitrary LLM understanding, and not output-only original-neuron identity.",
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
