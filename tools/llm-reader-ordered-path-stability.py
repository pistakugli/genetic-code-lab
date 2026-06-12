#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

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
    parser.add_argument("--min-original-drop", type=float, default=0.001)
    parser.add_argument("--min-paraphrase-drop-ratio", type=float, default=0.20)
    parser.add_argument("--min-paraphrase-pass-rate", type=float, default=0.66)
    parser.add_argument("--layer-shifts", default="-1,1")
    parser.add_argument("--min-layer-control-count", type=int, default=1)
    parser.add_argument("--min-layer-margin", type=float, default=0.0001)
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


def make_markdown(report: Dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Live Ordered Path Stability",
        "",
        "This gate replays already-discovered real-model circuits. It does not fit a surrogate.",
        "",
        f"- Pass: `{summary['pass']}`",
        f"- Cases: `{summary['caseCount']}`",
        f"- Original replay pass rate: `{summary['originalReplayPassRate']}`",
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
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
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
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    layer_shifts = parse_layer_shifts(args.layer_shifts)
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
        for shift in layer_shifts:
            all_layers.update(
                layer
                for item in nodes["mlp"] + nodes["attention"]
                for layer in [int(item["layer"]) + shift]
                if 0 <= layer < layer_count
            )
    mlps = adapter.mlp_modules(model, sorted(all_layers))
    attentions = adapter.attention_modules(model, sorted(all_layers))

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
        for prompt in ordered.factual.paraphrase_prompts_for_fact(fact, int(args.paraphrase_count)):
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
                "prompt": prompt,
                "targetLogitDrop": float(replay["targetLogitDrop"]),
                "ratioToOriginal": ratio,
                "pass": (
                    float(replay["targetLogitDrop"]) >= float(args.min_original_drop)
                    and ratio >= float(args.min_paraphrase_drop_ratio)
                ),
            })
        paraphrase_pass_rate = sum(bool(item["pass"]) for item in paraphrase_items) / max(1, len(paraphrase_items))

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
        "kind": f"{adapter.adapter_id}-live-ordered-path-stability",
        "synthetic": False,
        "surrogate": False,
        "generatedAt": now_iso(),
        "model": adapter.model_identity(model, args.model, device, args.local_files_only),
        "sourceReport": str(Path(args.input_report).resolve()),
        "config": {
            "subjects": sorted(selected_subjects),
            "caseCount": int(args.case_count),
            "paraphraseCount": int(args.paraphrase_count),
            "layerShifts": layer_shifts,
            "minLayerControlCount": int(args.min_layer_control_count),
            "minOriginalDrop": float(args.min_original_drop),
            "minParaphraseDropRatio": float(args.min_paraphrase_drop_ratio),
            "minParaphrasePassRate": float(args.min_paraphrase_pass_rate),
            "minLayerMargin": float(args.min_layer_margin),
            "minLayerSpecificityRate": float(args.min_layer_specificity_rate),
            "maxReplayError": float(args.max_replay_error),
        },
        "summary": {
            "pass": pass_summary,
            "caseCount": len(cases),
            "passedCases": sum(case["status"] == "passed" for case in cases),
            "originalReplayPassRate": original_rate,
            "paraphrasePassRate": paraphrase_rate,
            "layerSpecificityRate": layer_rate,
            "maxReplayError": max_replay_error,
            "failedSubjects": [case["subject"] for case in cases if case["status"] != "passed"],
        },
        "cases": cases,
        "claimBoundary": {
            "provesIfPasses": "The exact discovered circuit interventions replay on the source prompts, preserve target-causal effects across configured paraphrases, and outperform neighboring-layer shifted controls for the configured real-model cases.",
            "doesNotProve": "This does not prove neuron identity across model families, semantic equivalence of internal circuits, full-model circuit recovery, or arbitrary prompt stability.",
        },
    }
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    output_md.write_text(make_markdown(output), encoding="utf-8")
    print(json.dumps({"summary": output["summary"], "reportJson": str(output_json)}, indent=2))
    if not pass_summary:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
