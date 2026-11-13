#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import torch

from llm_reader.adapters.registry import adapter_choices, create_adapter
from llm_reader.ood_paraphrases import (
    ood_paraphrase_cases_for_fact,
    summarize_paraphrases_by_family,
)
from llm_reader.why_report import attach_why_analysis, render_why_markdown


def load_tool_module(filename: str, module_name: str) -> Any:
    source_path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load helper module from {source_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ordered = load_tool_module(
    "llm-reader-gpt2-ordered-residual-path-sweep.py",
    "gcl_ordered_path_sweep",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay discovered real-model circuits across paraphrases and neighboring-layer controls."
    )
    parser.add_argument("--adapter", choices=adapter_choices(), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input-report", required=True)
    parser.add_argument("--subjects", default="France,Germany,Italy")
    parser.add_argument("--case-count", type=int, default=3)
    parser.add_argument("--paraphrase-count", type=int, default=1)
    parser.add_argument("--paraphrase-profile", choices=["standard", "ood"], default="standard")
    parser.add_argument("--min-original-drop", type=float, default=0.001)
    parser.add_argument("--min-paraphrase-drop-ratio", type=float, default=0.20)
    parser.add_argument("--min-paraphrase-pass-rate", type=float, default=0.66)
    parser.add_argument("--layer-shifts", default="-1,1")
    parser.add_argument("--min-layer-control-count", type=int, default=1)
    parser.add_argument("--min-layer-margin", type=float, default=0.0001)
    parser.add_argument(
        "--rescue-broken-paraphrases",
        action="store_true",
        help="when a paraphrase breaks the path, search for the path that replaced it",
    )
    parser.add_argument(
        "--max-rescue-attempts",
        type=int,
        default=6,
        help="rescue at most this many broken paraphrases; 6 covers every OOD family",
    )
    parser.add_argument("--rescue-layer-shifts", default="-3,-2,-1,1,2,3")
    parser.add_argument("--replacement-candidate-count", type=int, default=10)
    parser.add_argument("--replacement-max-pair-attempts", type=int, default=32)
    parser.add_argument("--replacement-max-set-attempts", type=int, default=24)
    parser.add_argument("--min-layer-specificity-rate", type=float, default=0.66)
    parser.add_argument("--max-replay-error", type=float, default=1e-5)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def encode_prompt(tokenizer: Any, prompt: str, device: torch.device | str) -> Dict[str, torch.Tensor]:
    encoded = tokenizer(prompt, return_tensors="pt")
    return {key: value.to(device) for key, value in encoded.items()}


def remap_positions(
    interventions: List[Dict[str, Any]],
    original_read_index: int,
    new_read_index: int,
) -> List[Dict[str, Any]]:
    remapped: List[Dict[str, Any]] = []
    for intervention in interventions:
        positions = []
        for position in intervention.get("ablateTokenPositions", [original_read_index]):
            offset = int(position) - int(original_read_index)
            positions.append(max(0, min(new_read_index, new_read_index + offset)))
        remapped.append({
            **intervention,
            "ablateTokenPositions": sorted(set(positions)),
        })
    return remapped


def parse_layer_shifts(value: str) -> List[int]:
    shifts = []
    seen = set()
    for raw in value.split(","):
        item = raw.strip()
        if not item:
            continue
        shift = int(item)
        if shift == 0 or shift in seen:
            continue
        seen.add(shift)
        shifts.append(shift)
    if not shifts:
        raise ValueError("--layer-shifts must include at least one non-zero integer shift")
    return shifts


def shift_layers(
    interventions: List[Dict[str, Any]],
    shift: int,
    layer_count: int,
) -> List[Dict[str, Any]] | None:
    shifted: List[Dict[str, Any]] = []
    for intervention in interventions:
        layer = int(intervention["layer"]) + int(shift)
        if layer < 0 or layer >= layer_count:
            return None
        shifted.append({**intervention, "layer": layer})
    return shifted


def run_drop(
    *,
    model: Any,
    tokenizer: Any,
    mlps: Dict[int, Any],
    attentions: Dict[int, Any],
    prompt: str,
    target_token_id: int,
    mlp_interventions: List[Dict[str, Any]],
    attention_interventions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    device = next(model.parameters()).device
    encoded = encode_prompt(tokenizer, prompt, device)
    read_index = int(encoded["input_ids"].shape[1]) - 1
    with torch.no_grad():
        baseline_logits = model(**encoded).logits.detach().cpu()[0, read_index]
    ablated_logits = ordered.combined.run_combined_ablation_logits(
        model=model,
        mlps=mlps,
        attentions=attentions,
        encoded=encoded,
        mlp_interventions=mlp_interventions,
        attention_interventions=attention_interventions,
        read_token_index=read_index,
    )
    return {
        "readTokenIndex": read_index,
        "targetLogitDrop": float(baseline_logits[target_token_id] - ablated_logits[target_token_id]),
    }


def replacement_search_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        control_seed=260607,
        top_neurons=50,
        top_logits=8,
        control_count=8,
        control_prompt="The weather in Berlin is",
        max_graph_nodes=5,
        max_graph_heads=3,
        min_pass_rate=0.90,
        min_component_pass_rate=0.90,
        min_target_drop=0.05,
        min_graph_drop=0.05,
        min_margin_over_control_p95=0.02,
        min_path_activation_delta=0.005,
        min_path_margin_over_control_p95=0.001,
        require_component_hardening=True,
        ordered_rescue_candidate_count=max(1, int(args.replacement_candidate_count)),
        ordered_rescue_max_pair_attempts=max(1, int(args.replacement_max_pair_attempts)),
        ordered_rescue_set_candidate_count=max(2, int(args.replacement_candidate_count)),
        ordered_rescue_max_set_attempts=max(2, int(args.replacement_max_set_attempts)),
        wrong_answer_count=3,
        attention_max_target_rank=0,
        max_target_rank=0,
        paraphrase_count=2,
        negative_fact_count=2,
        unrelated_logit_count=8,
        min_paraphrase_pass_rate=0.70,
        min_wrong_answer_specificity_rate=0.85,
        min_negative_fact_specificity_rate=0.85,
        min_unrelated_preservation_rate=0.85,
        max_wrong_answer_drop_ratio=0.55,
        max_candidate_wrong_answer_drop_ratio=0.70,
        allow_target_causal_mlp_fallback=False,
        max_negative_fact_drop_ratio=0.55,
        max_unrelated_abs_drop_ratio=0.65,
        strict_rescue_candidate_count=14,
        strict_rescue_pair_candidate_count=12,
        strict_rescue_max_combo_size=2,
        strict_rescue_max_attempts=72,
    )


def rediscover_replacement_path(
    *,
    prompt: str,
    source: Dict[str, Any],
    model: Any,
    tokenizer: Any,
    mlps: Dict[int, Any],
    attentions: Dict[int, Any],
    answer_pool: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    fact = {"subject": str(source["subject"]), "answer": str(source["answer"])}
    seed = int.from_bytes(
        hashlib.sha256(f"{fact['subject']}:{prompt}".encode("utf-8")).digest()[:8],
        "big",
    )
    row = ordered.run_ordered_case(
        model=model,
        tokenizer=tokenizer,
        mlps=mlps,
        attentions=attentions,
        primary_mlps=mlps,
        fallback_mlps={},
        primary_attentions=attentions,
        fallback_attentions={},
        fact=fact,
        answer_pool=answer_pool,
        args=replacement_search_args(args),
        rng=random.Random(seed),
        prompt=prompt,
    )
    path = row.get("orderedResidualPath", {})
    nodes = path.get("nodes", {})
    metrics = path.get("pathPatch", {})
    encoded = encode_prompt(tokenizer, prompt, next(model.parameters()).device)
    return {
        "prompt": prompt,
        "status": row.get("status", "failed"),
        "found": row.get("status") == "passed",
        "failureReason": row.get("failureReason", row.get("reason", row.get("skipReason"))),
        "selectionMode": path.get("selection", {}).get("mode"),
        "readTokenIndex": int(encoded["input_ids"].shape[1]) - 1,
        "searchLayers": sorted(set(mlps) | set(attentions)),
        "mlpInterventions": nodes.get("mlp", []),
        "attentionInterventions": nodes.get("attention", []),
        "targetLogitDrop": row.get("combined", {}).get("targetLogitDrop"),
        "pathActivationDelta": metrics.get("selectedPathActivationDelta"),
        "pathMarginOverControlP95": metrics.get("marginOverControlPathP95"),
        "searchScope": "all MLP neurons and attention heads in the source and configured rescue-shift layers",
    }


def measure_replacement_path_transfer(
    *,
    replacement: Dict[str, Any],
    source_family: str,
    prompt_cases: List[Dict[str, str]],
    model: Any,
    tokenizer: Any,
    mlps: Dict[int, Any],
    attentions: Dict[int, Any],
    target_token_id: int,
    min_target_drop: float,
    min_drop_ratio: float,
) -> Dict[str, Any]:
    heldout_cases = [
        item
        for item in prompt_cases
        if item["family"] != source_family and item["prompt"] != replacement["prompt"]
    ]
    if replacement.get("status") != "passed":
        return {
            "status": "not_applicable",
            "reason": "replacement_path_not_found",
            "sourceFamily": source_family,
            "expectedHeldoutFamilyCount": len(heldout_cases),
            "items": [],
            "summary": {"attempted": 0, "passed": 0, "passRate": None},
        }
    if not heldout_cases:
        return {
            "status": "not_applicable",
            "reason": "no_heldout_prompt_families",
            "sourceFamily": source_family,
            "expectedHeldoutFamilyCount": 0,
            "items": [],
            "summary": {"attempted": 0, "passed": 0, "passRate": None},
        }

    discovery_drop = float(replacement["targetLogitDrop"])
    discovery_read_index = int(replacement["readTokenIndex"])
    items = []
    for prompt_case in heldout_cases:
        encoded = encode_prompt(
            tokenizer,
            prompt_case["prompt"],
            next(model.parameters()).device,
        )
        read_index = int(encoded["input_ids"].shape[1]) - 1
        replay = run_drop(
            model=model,
            tokenizer=tokenizer,
            mlps=mlps,
            attentions=attentions,
            prompt=prompt_case["prompt"],
            target_token_id=target_token_id,
            mlp_interventions=remap_positions(
                replacement["mlpInterventions"],
                discovery_read_index,
                read_index,
            ),
            attention_interventions=remap_positions(
                replacement["attentionInterventions"],
                discovery_read_index,
                read_index,
            ),
        )
        target_drop = float(replay["targetLogitDrop"])
        ratio = target_drop / max(discovery_drop, 1e-9)
        items.append({
            "family": prompt_case["family"],
            "prompt": prompt_case["prompt"],
            "tokenCount": read_index + 1,
            "targetLogitDrop": target_drop,
            "ratioToDiscovery": ratio,
            "pass": target_drop >= min_target_drop and ratio >= min_drop_ratio,
        })

    attempted = len(items)
    passed = sum(bool(item["pass"]) for item in items)
    return {
        "status": "passed" if attempted and passed == attempted else "failed",
        "sourceFamily": source_family,
        "sourcePrompt": replacement["prompt"],
        "discoveryTargetLogitDrop": discovery_drop,
        "minTargetLogitDrop": min_target_drop,
        "minDropRatio": min_drop_ratio,
        "expectedHeldoutFamilyCount": len(heldout_cases),
        "items": items,
        "summary": {
            "attempted": attempted,
            "passed": passed,
            "passRate": passed / attempted if attempted else None,
        },
        "claimBoundary": (
            "Held-out family replay measures whether the prompt-specific discovered path "
            "retains its causal target-logit effect on the other configured OOD formulations; "
            "it does not establish semantic circuit identity or general prompt robustness."
        ),
    }


def make_markdown(report: Dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Live Ordered Path OOD Stress" if summary["paraphraseProfile"] == "ood" else "# Live Ordered Path Stability",
        "",
        "This gate replays already-discovered real-model circuits. It does not fit a surrogate.",
        "",
        f"- Pass: `{summary['pass']}`",
        f"- Cases: `{summary['caseCount']}`",
        f"- Original replay pass rate: `{summary['originalReplayPassRate']}`",
        f"- Paraphrase profile: `{summary['paraphraseProfile']}`",
        f"- Paraphrase pass rate: `{summary['paraphrasePassRate']}`",
        f"- Neighbor-layer specificity rate: `{summary['layerSpecificityRate']}`",
        f"- Maximum replay error: `{summary['maxReplayError']}`",
        "",
        "| subject | replay | paraphrase | layer margin | status |",
        "|---|---:|---:|---:|---:|",
    ]
    for case in report["cases"]:
        lines.append(
            f"| {case['subject']} | {case['originalReplay']['targetLogitDrop']:.6f} | "
            f"{case['paraphraseSummary']['passRate']:.6f} | "
            f"{case['layerControls']['marginOverBestPositiveNeighbor']:.6f} | {case['status']} |"
        )
    lines.extend([
        "",
        "## Paraphrase family results",
        "",
        "| family | count | passed | pass rate | mean drop ratio | mean tokens |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for family, result in report["summary"]["paraphraseFamilySummary"].items():
        lines.append(
            f"| {family} | {result['count']} | {result['passed']} | "
            f"{result['passRate']:.6f} | {result['meanDropRatio']:.6f} | "
            f"{result['meanTokenCount']:.2f} |"
        )
    rescue_rows = [
        (case["subject"], rescue)
        for case in report["cases"]
        for rescue in case.get("paraphraseRescue", {}).get("items", [])
    ]
    if rescue_rows:
        lines.extend([
            "",
            "## Paraphrase rescue",
            "",
            "Same-component layer-shift hints; these are controls, not replacement-path discovery.",
            "",
            "| subject | found | best shift | broken drop | best drop |",
            "|---|---:|---:|---:|---:|",
        ])
        for subject, rescue in rescue_rows:
            best_drop = rescue.get("bestDrop")
            lines.append(
                f"| {subject} | {rescue['found']} | "
                f"{rescue.get('bestLayerShift', 'n/a')} | "
                f"{rescue['brokenDrop']:.6f} | "
                f"{'n/a' if best_drop is None else f'{best_drop:.6f}'} |"
            )
    replacement_rows = [
        (case["subject"], rescue, rescue["replacementPathSearch"])
        for case in report["cases"]
        for rescue in case.get("paraphraseRescue", {}).get("items", [])
        if isinstance(rescue.get("replacementPathSearch"), dict)
    ]
    if replacement_rows:
        lines.extend([
            "",
            "## Replacement path rediscovery",
            "",
            "Components are reselected on each broken prompt from the loaded source and configured rescue-shift layers. A pass requires the existing combined-ablation, ordered-path and control gates.",
            "",
            "| subject | status | MLP nodes | attention heads | target drop | path delta | path margin |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for subject, _, result in replacement_rows:
            target_drop = result.get("targetLogitDrop")
            path_delta = result.get("pathActivationDelta")
            path_margin = result.get("pathMarginOverControlP95")
            lines.append(
                f"| {subject} | {result['status']} | "
                f"{len(result.get('mlpInterventions', []))} | "
                f"{len(result.get('attentionInterventions', []))} | "
                f"{'n/a' if target_drop is None else f'{float(target_drop):.6f}'} | "
                f"{'n/a' if path_delta is None else f'{float(path_delta):.6f}'} | "
                f"{'n/a' if path_margin is None else f'{float(path_margin):.6f}'} |"
            )
        transfer_rows = [
            (case["subject"], result["crossFamilyTransfer"])
            for case in report["cases"]
            for rescue in case.get("paraphraseRescue", {}).get("items", [])
            for result in [rescue.get("replacementPathSearch", {})]
            if isinstance(result, dict)
            and isinstance(result.get("crossFamilyTransfer"), dict)
        ]
        if transfer_rows:
            lines.extend([
                "",
                "## Held-out family transfer",
                "",
                "Each replacement path is measured on the other OOD prompt families, excluding the family used for path selection.",
                "",
                "| subject | source family | status | held-out families passed |",
                "|---|---|---:|---:|",
            ])
            for subject, transfer in transfer_rows:
                transfer_summary = transfer.get("summary", {})
                lines.append(
                    f"| {subject} | {transfer.get('sourceFamily', 'n/a')} | "
                    f"{transfer.get('status', 'n/a')} | "
                    f"{transfer_summary.get('passed', 0)}/"
                    f"{transfer_summary.get('attempted', 0)} |"
                )
    return "\n".join(lines + render_why_markdown(report)) + "\n"


def main() -> None:
    args = parse_args()
    if args.case_count <= 0 or args.paraphrase_count <= 0:
        raise RuntimeError("--case-count and --paraphrase-count must be positive.")
    if args.max_rescue_attempts < 0:
        raise RuntimeError("--max-rescue-attempts must be non-negative.")
    if min(
        args.replacement_candidate_count,
        args.replacement_max_pair_attempts,
        args.replacement_max_set_attempts,
    ) <= 0:
        raise RuntimeError("Replacement search candidate and attempt budgets must be positive.")
    report = json.loads(Path(args.input_report).read_text(encoding="utf-8"))
    selected_subjects = {item.strip() for item in args.subjects.split(",") if item.strip()}
    source_cases = [
        case
        for case in report.get("cases", [])
        if case.get("status") == "passed" and case.get("subject") in selected_subjects
    ][:max(1, int(args.case_count))]
    if len(source_cases) < int(args.case_count):
        raise RuntimeError(
            f"Stability gate requires {args.case_count} passed source cases, found {len(source_cases)}."
        )

    adapter = create_adapter(args.adapter)
    ordered.set_live_adapter(args.adapter)
    device_request = os.environ.get("GCL_DEVICE") or args.device
    device = "cuda" if device_request == "auto" and torch.cuda.is_available() else device_request
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    layer_shifts = parse_layer_shifts(args.layer_shifts)
    rescue_shifts = parse_layer_shifts(args.rescue_layer_shifts) if args.rescue_broken_paraphrases else []
    tokenizer, model = adapter.load_model_and_tokenizer(
        args.model,
        local_files_only=args.local_files_only,
        device=device,
    )
    layer_count = int(getattr(model.config, "num_hidden_layers"))
    all_layers = set()
    for case in source_cases:
        nodes = case["orderedResidualPath"]["nodes"]
        all_layers.update(int(item["layer"]) for item in nodes["mlp"])
        all_layers.update(int(item["layer"]) for item in nodes["attention"])
        for shift in list(layer_shifts) + list(rescue_shifts):
            all_layers.update(
                layer
                for item in nodes["mlp"] + nodes["attention"]
                for layer in [int(item["layer"]) + shift]
                if 0 <= layer < layer_count
            )
    mlps = adapter.mlp_modules(model, sorted(all_layers))
    attentions = adapter.attention_modules(model, sorted(all_layers))
    answer_pool = ordered.factual.build_answer_pool(tokenizer) if args.rescue_broken_paraphrases else []

    cases: List[Dict[str, Any]] = []
    for source in source_cases:
        target = ordered.factual.encode_single_answer_token(tokenizer, source["answer"])
        if target is None:
            raise RuntimeError(f"Stability target is not one token for {source['subject']}.")
        target_token_id, target_token = target
        nodes = source["orderedResidualPath"]["nodes"]
        original_read_index = int(source["readTokenIndex"])
        selected_mlp = remap_positions(nodes["mlp"], original_read_index, original_read_index)
        selected_attention = remap_positions(nodes["attention"], original_read_index, original_read_index)
        original = run_drop(
            model=model,
            tokenizer=tokenizer,
            mlps=mlps,
            attentions=attentions,
            prompt=source["prompt"],
            target_token_id=target_token_id,
            mlp_interventions=selected_mlp,
            attention_interventions=selected_attention,
        )
        reported_drop = float(source["combined"]["targetLogitDrop"])
        replay_error = abs(float(original["targetLogitDrop"]) - reported_drop)
        original_pass = (
            float(original["targetLogitDrop"]) >= float(args.min_original_drop)
            and replay_error <= float(args.max_replay_error)
        )

        paraphrase_items = []
        fact = {"subject": source["subject"], "answer": source["answer"]}
        if args.paraphrase_profile == "ood":
            prompt_cases = ood_paraphrase_cases_for_fact(fact, int(args.paraphrase_count))
        else:
            prompt_cases = [
                {"profile": "standard", "family": "standard-template", "prompt": prompt}
                for prompt in ordered.factual.paraphrase_prompts_for_fact(
                    fact, int(args.paraphrase_count)
                )
            ]
        for prompt_case in prompt_cases:
            prompt = prompt_case["prompt"]
            encoded = encode_prompt(tokenizer, prompt, next(model.parameters()).device)
            new_read_index = int(encoded["input_ids"].shape[1]) - 1
            replay = run_drop(
                model=model,
                tokenizer=tokenizer,
                mlps=mlps,
                attentions=attentions,
                prompt=prompt,
                target_token_id=target_token_id,
                mlp_interventions=remap_positions(nodes["mlp"], original_read_index, new_read_index),
                attention_interventions=remap_positions(nodes["attention"], original_read_index, new_read_index),
            )
            ratio = float(replay["targetLogitDrop"]) / max(1e-9, float(original["targetLogitDrop"]))
            paraphrase_items.append({
                "profile": prompt_case["profile"],
                "family": prompt_case["family"],
                "prompt": prompt,
                "tokenCount": int(replay["readTokenIndex"]) + 1,
                "targetLogitDrop": float(replay["targetLogitDrop"]),
                "ratioToOriginal": ratio,
                "pass": (
                    float(replay["targetLogitDrop"]) >= float(args.min_original_drop)
                    and ratio >= float(args.min_paraphrase_drop_ratio)
                ),
            })
        paraphrase_pass_rate = sum(bool(item["pass"]) for item in paraphrase_items) / max(1, len(paraphrase_items))

        # When a paraphrase breaks the path, find where the effect went.
        #
        # Recording `pass: false` and moving on treats one prompt's circuit as the circuit and
        # a broken replay as the model being unreadable. Neither holds: the behaviour survives
        # the paraphrase, so some path carries it, and which layers took over is the finding.
        # A stability lane without this can say a path is fragile; it cannot say what the
        # model does instead, which is the question worth asking.
        #
        # The search here is deliberately narrow - the same path shifted across layers, scored
        # on the paraphrase that broke it. That is not a full circuit rediscovery, and the
        # artifact says so; it answers "did the effect move, and where to" without pretending
        # to answer "what is the new circuit in full".
        rescues = []
        if args.rescue_broken_paraphrases:
            broken = [item for item in paraphrase_items if not item["pass"]]
            for item in broken[: int(args.max_rescue_attempts)]:
                encoded = encode_prompt(tokenizer, item["prompt"], next(model.parameters()).device)
                new_read_index = int(encoded["input_ids"].shape[1]) - 1
                base_mlp = remap_positions(nodes["mlp"], original_read_index, new_read_index)
                base_attention = remap_positions(nodes["attention"], original_read_index, new_read_index)
                attempts = []
                for shift in rescue_shifts:
                    shifted_mlp = shift_layers(base_mlp, shift, layer_count)
                    shifted_attention = shift_layers(base_attention, shift, layer_count)
                    if shifted_mlp is None or shifted_attention is None:
                        continue
                    replay = run_drop(
                        model=model, tokenizer=tokenizer, mlps=mlps, attentions=attentions,
                        prompt=item["prompt"], target_token_id=target_token_id,
                        mlp_interventions=shifted_mlp, attention_interventions=shifted_attention,
                    )
                    attempts.append({"layerShift": shift, "targetLogitDrop": float(replay["targetLogitDrop"])})
                best = max(attempts, key=lambda row: row["targetLogitDrop"]) if attempts else None
                recovered = bool(
                    best is not None
                    and best["targetLogitDrop"] >= float(args.min_original_drop)
                    and best["targetLogitDrop"] > float(item["targetLogitDrop"])
                )
                replacement_path = rediscover_replacement_path(
                    prompt=item["prompt"],
                    source=source,
                    model=model,
                    tokenizer=tokenizer,
                    mlps=mlps,
                    attentions=attentions,
                    answer_pool=answer_pool,
                    args=args,
                )
                replacement_path["crossFamilyTransfer"] = measure_replacement_path_transfer(
                    replacement=replacement_path,
                    source_family=item["family"],
                    prompt_cases=prompt_cases,
                    model=model,
                    tokenizer=tokenizer,
                    mlps=mlps,
                    attentions=attentions,
                    target_token_id=target_token_id,
                    min_target_drop=float(args.min_original_drop),
                    min_drop_ratio=float(args.min_paraphrase_drop_ratio),
                )
                rescues.append({
                    "prompt": item["prompt"],
                    "family": item["family"],
                    "brokenDrop": float(item["targetLogitDrop"]),
                    "found": recovered,
                    "bestLayerShift": None if best is None else int(best["layerShift"]),
                    "bestDrop": None if best is None else float(best["targetLogitDrop"]),
                    "attempts": attempts,
                    "searchScope": "layer-shifted replay of the same path",
                    "replacementPathSearch": replacement_path,
                })
        rescued = sum(1 for row in rescues if row.get("found")) if rescues else 0

        layer_items = []
        for shift in layer_shifts:
            shifted_mlp = shift_layers(selected_mlp, shift, layer_count)
            shifted_attention = shift_layers(selected_attention, shift, layer_count)
            if shifted_mlp is None or shifted_attention is None:
                continue
            replay = run_drop(
                model=model,
                tokenizer=tokenizer,
                mlps=mlps,
                attentions=attentions,
                prompt=source["prompt"],
                target_token_id=target_token_id,
                mlp_interventions=shifted_mlp,
                attention_interventions=shifted_attention,
            )
            layer_items.append({"shift": shift, "targetLogitDrop": float(replay["targetLogitDrop"])})
        positive_neighbors = [max(0.0, float(item["targetLogitDrop"])) for item in layer_items]
        best_neighbor = max(positive_neighbors) if positive_neighbors else 0.0
        layer_margin = float(original["targetLogitDrop"]) - best_neighbor
        layer_pass = (
            len(layer_items) >= int(args.min_layer_control_count)
            and layer_margin >= float(args.min_layer_margin)
        )
        case_pass = (
            original_pass
            and paraphrase_pass_rate >= float(args.min_paraphrase_pass_rate)
            and layer_pass
        )
        cases.append({
            "subject": source["subject"],
            "answer": source["answer"],
            "target": {"tokenId": int(target_token_id), "token": target_token},
            "paraphraseRescue": {
                "attempted": len(rescues),
                "found": rescued,
                "items": rescues,
            },
            "status": "passed" if case_pass else "failed",
            "sourceCircuit": nodes,
            "originalReplay": {
                **original,
                "reportedTargetLogitDrop": reported_drop,
                "absoluteReplayError": replay_error,
                "pass": original_pass,
            },
            "paraphrases": paraphrase_items,
            "paraphraseSummary": {
                "count": len(paraphrase_items),
                "passed": sum(bool(item["pass"]) for item in paraphrase_items),
                "passRate": paraphrase_pass_rate,
                "families": summarize_paraphrases_by_family(paraphrase_items),
            },
            "layerControls": {
                "items": layer_items,
                "requestedShifts": layer_shifts,
                "validControlCount": len(layer_items),
                "minLayerControlCount": int(args.min_layer_control_count),
                "bestPositiveNeighborTargetLogitDrop": best_neighbor,
                "marginOverBestPositiveNeighbor": layer_margin,
                "pass": layer_pass,
            },
        })

    original_rate = sum(bool(case["originalReplay"]["pass"]) for case in cases) / len(cases)
    paraphrase_items = [item for case in cases for item in case["paraphrases"]]
    paraphrase_rate = sum(bool(item["pass"]) for item in paraphrase_items) / max(1, len(paraphrase_items))
    layer_rate = sum(bool(case["layerControls"]["pass"]) for case in cases) / len(cases)
    max_replay_error = max(float(case["originalReplay"]["absoluteReplayError"]) for case in cases)
    all_cases_pass = all(case["status"] == "passed" for case in cases)
    pass_summary = (
        all_cases_pass
        and original_rate == 1.0
        and paraphrase_rate >= float(args.min_paraphrase_pass_rate)
        and layer_rate >= float(args.min_layer_specificity_rate)
    )
    output = {
        "schemaVersion": 1,
        "kind": (
            f"{adapter.adapter_id}-live-ordered-path-ood-stress"
            if args.paraphrase_profile == "ood"
            else f"{adapter.adapter_id}-live-ordered-path-stability"
        ),
        "synthetic": False,
        "surrogate": False,
        "generatedAt": now_iso(),
        "model": adapter.model_identity(model, args.model, device, args.local_files_only),
        "sourceReport": str(Path(args.input_report).resolve()),
        "config": {
            "subjects": sorted(selected_subjects),
            "caseCount": int(args.case_count),
            "paraphraseCount": int(args.paraphrase_count),
            "paraphraseProfile": args.paraphrase_profile,
            "layerShifts": layer_shifts,
            "minLayerControlCount": int(args.min_layer_control_count),
            "minOriginalDrop": float(args.min_original_drop),
            "minParaphraseDropRatio": float(args.min_paraphrase_drop_ratio),
            "minParaphrasePassRate": float(args.min_paraphrase_pass_rate),
            "minLayerMargin": float(args.min_layer_margin),
            "minLayerSpecificityRate": float(args.min_layer_specificity_rate),
            "maxReplayError": float(args.max_replay_error),
            "replacementPathSearchEnabled": bool(args.rescue_broken_paraphrases),
            "replacementPathTransferEnabled": (
                bool(args.rescue_broken_paraphrases)
                and args.paraphrase_profile == "ood"
            ),
            "replacementCandidateCount": int(args.replacement_candidate_count),
            "replacementMaxPairAttempts": int(args.replacement_max_pair_attempts),
            "replacementMaxSetAttempts": int(args.replacement_max_set_attempts),
            "maxRescueAttempts": int(args.max_rescue_attempts),
        },
        "summary": {
            "pass": pass_summary,
            "caseCount": len(cases),
            "passedCases": sum(case["status"] == "passed" for case in cases),
            "originalReplayPassRate": original_rate,
            "paraphraseProfile": args.paraphrase_profile,
            "paraphrasePassRate": paraphrase_rate,
            "paraphraseFamilySummary": summarize_paraphrases_by_family(paraphrase_items),
            "layerSpecificityRate": layer_rate,
            "maxReplayError": max_replay_error,
            "failedSubjects": [case["subject"] for case in cases if case["status"] != "passed"],
            "replacementPathSearch": {
                "attempted": sum(
                    "replacementPathSearch" in rescue
                    for case in cases
                    for rescue in case.get("paraphraseRescue", {}).get("items", [])
                ),
                "passed": sum(
                    bool(rescue["replacementPathSearch"].get("found"))
                    for case in cases
                    for rescue in case.get("paraphraseRescue", {}).get("items", [])
                    if "replacementPathSearch" in rescue
                ),
            },
            "replacementPathTransfer": {
                "attempted": sum(
                    int(rescue["replacementPathSearch"]["crossFamilyTransfer"]["summary"]["attempted"])
                    for case in cases
                    for rescue in case.get("paraphraseRescue", {}).get("items", [])
                    if "replacementPathSearch" in rescue
                    and isinstance(rescue["replacementPathSearch"].get("crossFamilyTransfer"), dict)
                ),
                "passed": sum(
                    int(rescue["replacementPathSearch"]["crossFamilyTransfer"]["summary"]["passed"])
                    for case in cases
                    for rescue in case.get("paraphraseRescue", {}).get("items", [])
                    if "replacementPathSearch" in rescue
                    and isinstance(rescue["replacementPathSearch"].get("crossFamilyTransfer"), dict)
                ),
            },
        },
        "cases": cases,
        "claimBoundary": {
            "provesIfPasses": "The exact discovered circuit interventions replay on the source prompts, preserve target-causal effects across the configured standard or held-out OOD paraphrase families, and outperform neighboring-layer shifted controls for the configured real-model cases.",
            "doesNotProve": "This does not prove neuron identity across model families, semantic equivalence of internal circuits, full-model circuit recovery, semantic validity beyond the curated capital-question templates, or arbitrary prompt stability.",
        },
    }
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    # North-star step 7 for this lane: every recorded break gets a why-report
    # (break classification, where the effect went, replacement comparison, and
    # transfer diagnosis) computed from the measurements already collected.
    attach_why_analysis(output)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    output_md.write_text(make_markdown(output), encoding="utf-8")
    print(json.dumps({"summary": output["summary"], "reportJson": str(output_json)}, indent=2))
    if not pass_summary:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
