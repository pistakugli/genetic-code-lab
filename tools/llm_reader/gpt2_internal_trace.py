#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_REPORT = (
    REPOSITORY_ROOT
    / "benchmarks/release/nn-live-gpt2-ordered-residual-path-circuit-finder"
    / "gpt2-50-facts-ordered-residual-path/report.json"
)
REQUIRED_SUBJECTS = ("France", "Germany", "Italy", "Portugal", "Greece")
NEIGHBOR_LAYER_OFFSETS = (-1, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture layerwise GPT-2 computation and causally intervene on existing ordered paths."
    )
    parser.add_argument("--source-report", type=Path, default=DEFAULT_SOURCE_REPORT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--protocol", choices=("legacy", "autoregressive"), default="legacy")
    parser.add_argument("--case-set", choices=("regression-five", "pilot-five", "all-50"), default="regression-five")
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--control-count", type=int, default=100)
    parser.add_argument("--control-magnitude-tolerance", type=float, default=0.2)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def primary_tensor(output: Any) -> torch.Tensor:
    value = output[0] if isinstance(output, (tuple, list)) else output
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Expected a tensor output, received {type(value).__name__}.")
    return value


def ordered_source_cases(source_report: Dict[str, Any]) -> List[Dict[str, Any]]:
    by_subject: Dict[str, Dict[str, Any]] = {}
    for row in source_report.get("cases", []):
        subject = row.get("subject")
        if subject in REQUIRED_SUBJECTS and row.get("status") == "passed":
            if subject in by_subject:
                raise ValueError(f"Source report contains duplicate passed cases for {subject}.")
            by_subject[subject] = row
    missing = [subject for subject in REQUIRED_SUBJECTS if subject not in by_subject]
    if missing:
        raise ValueError(f"Source report is missing required passed cases: {', '.join(missing)}")
    return [by_subject[subject] for subject in REQUIRED_SUBJECTS]


def all_passed_source_cases(source_report: Dict[str, Any]) -> List[Dict[str, Any]]:
    cases = [row for row in source_report.get("cases", []) if row.get("status") == "passed"]
    expected_count = int(source_report.get("config", {}).get("caseCount", 0))
    if expected_count <= 0 or len(cases) != expected_count:
        raise ValueError(
            f"Source artifact declares {expected_count} cases but has {len(cases)} passed cases."
        )
    subjects = [row.get("subject") for row in cases]
    if any(not isinstance(subject, str) or not subject for subject in subjects):
        raise ValueError("Every passed source case must have a non-empty subject.")
    if len(set(subjects)) != len(subjects):
        raise ValueError("Source artifact contains duplicate passed-case subjects.")
    return cases


def conv1d_head_writes(
    context: torch.Tensor,
    projection_weight: torch.Tensor,
    head_count: int,
) -> torch.Tensor:
    if context.shape[-1] % head_count:
        raise ValueError("Attention projection input width is not divisible by the head count.")
    head_dim = int(context.shape[-1]) // head_count
    writes = [
        context[..., head * head_dim : (head + 1) * head_dim]
        @ projection_weight[head * head_dim : (head + 1) * head_dim, :]
        for head in range(head_count)
    ]
    return torch.stack(writes, dim=-2)


def mlp_neuron_first_order_attribution(
    activations: torch.Tensor,
    projection_weight: torch.Tensor,
    output_gradient: torch.Tensor,
) -> torch.Tensor:
    return activations * (projection_weight @ output_gradient)


def summarize_logits(logits: torch.Tensor, target_id: int, tokenizer: Any, count: int = 8) -> Dict[str, Any]:
    flat = logits.detach().float().cpu().reshape(-1)
    if target_id < 0 or target_id >= flat.numel():
        raise ValueError(f"Target token id {target_id} is outside the model vocabulary.")
    probabilities = torch.softmax(flat, dim=-1)
    top = torch.topk(flat, k=min(count, flat.numel()))
    items = [
        {
            "rank": rank + 1,
            "tokenId": int(token_id),
            "token": tokenizer.decode([int(token_id)]).replace("\n", "\\n"),
            "logit": float(logit),
            "probability": float(probabilities[token_id]),
        }
        for rank, (logit, token_id) in enumerate(zip(top.values, top.indices))
    ]
    target_rank = int((flat > flat[target_id]).sum()) + 1
    greedy_id = int(torch.argmax(flat))
    return {
        "target": {
            "tokenId": int(target_id),
            "token": tokenizer.decode([target_id]).replace("\n", "\\n"),
            "logit": float(flat[target_id]),
            "probability": float(probabilities[target_id]),
            "rank": target_rank,
        },
        "greedyNextToken": {
            "tokenId": greedy_id,
            "token": tokenizer.decode([greedy_id]).replace("\n", "\\n"),
            "logit": float(flat[greedy_id]),
            "probability": float(probabilities[greedy_id]),
        },
        "topTokens": items,
    }


def selected_path_nodes(source_case: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    nodes = source_case.get("orderedResidualPath", {}).get("nodes", {})
    attention = nodes.get("attention", [])
    mlp = nodes.get("mlp", [])
    if not attention or not mlp:
        raise ValueError(f"{source_case.get('subject')}: source case does not contain both path component types.")
    return attention, mlp


def choose_recorded_controls(source_case: Dict[str, Any]) -> List[Dict[str, Any]]:
    items = source_case.get("orderedResidualPath", {}).get("controlPaths", {}).get("items", [])
    selected: List[Dict[str, Any]] = []
    for desired_kind in (
        "selected-mlp-plus-random-attention-head",
        "random-active-same-amplitude-plus-selected-attention",
    ):
        match = next((item for item in items if item.get("kind") == desired_kind), None)
        if match is None:
            raise ValueError(
                f"{source_case.get('subject')}: source artifact lacks recorded control {desired_kind}."
            )
        selected.append(match)
    return selected


class GPT2TraceCapture:
    def __init__(self, model: Any):
        self.model = model
        self.blocks = list(model.transformer.h)
        self.layers = len(self.blocks)
        self.values: Dict[str, Any] = {}
        self.handles: List[Any] = []
        self.gradient_outputs: List[Tuple[str, torch.Tensor]] = []

    def __enter__(self) -> "GPT2TraceCapture":
        transformer = self.model.transformer
        self._hook_tensor(transformer.wte, "token_embedding")
        self._hook_tensor(transformer.wpe, "position_embedding")
        self._hook_tensor(transformer.drop, "embedding_output")
        self._hook_tensor(transformer.ln_f, "final_norm_output", retain_grad=False)
        self.handles.append(
            transformer.ln_f.register_forward_pre_hook(
                lambda _module, inputs: self.values.__setitem__("final_norm_input", inputs[0])
            )
        )
        self.handles.append(
            self.model.lm_head.register_forward_hook(
                lambda _module, _inputs, output: self.values.__setitem__("lm_head_logits", primary_tensor(output))
            )
        )
        for layer, block in enumerate(self.blocks):
            self.handles.append(
                block.register_forward_pre_hook(
                    lambda _module, inputs, layer_index=layer: self._set_layer("block_input", layer_index, inputs[0])
                )
            )
            self.handles.append(
                block.register_forward_hook(
                    lambda _module, _inputs, output, layer_index=layer: self._set_layer(
                        "block_output", layer_index, primary_tensor(output)
                    )
                )
            )
            self._hook_tensor(block.attn, f"attention_output_{layer}")
            self._hook_tensor(block.mlp.c_fc, f"mlp_preactivation_{layer}", retain_grad=False)
            self._hook_tensor(block.mlp.act, f"mlp_activation_{layer}", retain_grad=False)
            self._hook_tensor(block.mlp.c_proj, f"mlp_output_{layer}")
            self.handles.append(
                block.attn.c_proj.register_forward_pre_hook(
                    lambda _module, inputs, layer_index=layer: self._set_layer(
                        "attention_context", layer_index, inputs[0]
                    )
                )
            )
            self.handles.append(
                block.mlp.c_fc.register_forward_pre_hook(
                    lambda _module, inputs, layer_index=layer: self._set_layer(
                        "mlp_input", layer_index, inputs[0]
                    )
                )
            )
        return self

    def _set_layer(self, name: str, layer: int, tensor: torch.Tensor) -> None:
        self.values.setdefault(name, {})[layer] = tensor

    def _hook_tensor(self, module: Any, name: str, retain_grad: bool = True) -> None:
        def capture(_module: Any, _inputs: Any, output: Any) -> None:
            tensor = primary_tensor(output)
            self.values[name] = tensor
            if retain_grad and tensor.requires_grad:
                tensor.retain_grad()
                self.gradient_outputs.append((name, tensor))

        self.handles.append(module.register_forward_hook(capture))

    def __exit__(self, _exc_type: Any, _exc_value: Any, _traceback: Any) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()


def _ordered_layer_tensors(capture: GPT2TraceCapture, name: str) -> List[torch.Tensor]:
    values = capture.values.get(name, {})
    missing = [layer for layer in range(capture.layers) if layer not in values]
    if missing:
        raise RuntimeError(f"Missing {name} hook values for layers {missing}.")
    return [values[layer] for layer in range(capture.layers)]


def _tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().float().cpu().numpy()


def _scalar(value: torch.Tensor) -> float:
    return float(value.detach().float().cpu())


def full_distribution_change(
    baseline_logits: torch.Tensor,
    changed_logits: torch.Tensor,
) -> Dict[str, float]:
    baseline_prob = torch.softmax(baseline_logits.detach().float().cpu(), dim=-1)
    changed_prob = torch.softmax(changed_logits.detach().float().cpu(), dim=-1)
    midpoint = (baseline_prob + changed_prob) * 0.5
    epsilon = torch.finfo(baseline_prob.dtype).tiny
    js = 0.5 * (
        torch.sum(baseline_prob * (torch.log(baseline_prob.clamp_min(epsilon)) - torch.log(midpoint.clamp_min(epsilon))))
        + torch.sum(changed_prob * (torch.log(changed_prob.clamp_min(epsilon)) - torch.log(midpoint.clamp_min(epsilon))))
    )
    js = js.clamp_min(0.0)
    absolute_delta = torch.abs(changed_prob - baseline_prob)
    return {
        "totalVariation": float(absolute_delta.sum() * 0.5),
        "l1Probability": float(absolute_delta.sum()),
        "jensenShannonDivergence": float(js),
        "maxAbsoluteProbabilityChange": float(absolute_delta.max()),
    }


def interaction_from_logits(
    baseline_logits: torch.Tensor,
    group_logits: torch.Tensor,
    individual_logits: List[torch.Tensor],
    target_id: int,
) -> Dict[str, float]:
    baseline = baseline_logits.detach().float().cpu()
    group = group_logits.detach().float().cpu()
    individuals = [item.detach().float().cpu() for item in individual_logits]
    if not individuals:
        raise ValueError("Interaction requires at least one individual intervention.")
    group_drop = baseline[target_id] - group[target_id]
    individual_drops = [baseline[target_id] - item[target_id] for item in individuals]
    interaction = group_drop - sum(individual_drops)
    full_interaction = group_drop.new_full(group.shape, 1 - len(individuals)) * baseline - group
    for item in individuals:
        full_interaction += item
    return {
        "groupLogitDrop": float(group_drop),
        "sumIndividualLogitDrops": float(sum(individual_drops)),
        "logitDropNonAdditivity": float(interaction),
        "fullVocabularyLogitInteractionL2": float(torch.linalg.vector_norm(full_interaction)),
        "fullVocabularyLogitInteractionMaxAbs": float(full_interaction.abs().max()),
    }


def _ensure_positions(interventions: Iterable[Dict[str, Any]], token_count: int, label: str) -> List[Dict[str, Any]]:
    checked = []
    for intervention in interventions:
        layer = int(intervention["layer"])
        positions = sorted({int(position) for position in intervention.get("ablateTokenPositions", [])})
        if not positions or any(position < 0 or position >= token_count for position in positions):
            raise ValueError(f"{label} intervention requires valid token positions.")
        checked.append({**intervention, "layer": layer, "ablateTokenPositions": positions})
    return checked


def run_intervention(
    *,
    model: Any,
    encoded: Dict[str, torch.Tensor],
    read_index: int,
    target_id: int,
    baseline_summary: Dict[str, Any],
    tokenizer: Any,
    mlp_interventions: List[Dict[str, Any]],
    attention_interventions: List[Dict[str, Any]],
    baseline_mlp_activations: Dict[int, torch.Tensor],
    baseline_attention_context: Dict[int, torch.Tensor],
    mode: str,
    baseline_logits: torch.Tensor | None = None,
    retain_logits: bool = False,
) -> Dict[str, Any]:
    token_count = int(encoded["input_ids"].shape[1])
    mlp_sites = _ensure_positions(mlp_interventions, token_count, "MLP")
    attention_sites = _ensure_positions(attention_interventions, token_count, "Attention")
    if mode not in ("remove", "restore"):
        raise ValueError(f"Unsupported intervention mode: {mode}")

    handles = []
    blocks = model.transformer.h
    try:
        for intervention in mlp_sites:
            layer = intervention["layer"]
            neuron = int(intervention["neuron"])
            inner_width = int(baseline_mlp_activations[layer].shape[-1]) if 0 <= layer < len(blocks) else 0
            if layer < 0 or layer >= len(blocks) or neuron < 0 or neuron >= inner_width:
                raise ValueError(f"Invalid MLP site layer={layer}, neuron={neuron}.")
            baseline = baseline_mlp_activations[layer]

            def patch_mlp(
                _module: Any,
                _inputs: Any,
                output: torch.Tensor,
                neuron_index: int = neuron,
                positions: List[int] = intervention["ablateTokenPositions"],
                original: torch.Tensor = baseline,
            ) -> torch.Tensor:
                patched = output.clone()
                for position in positions:
                    patched[:, position, neuron_index] = (
                        0 if mode == "remove" else original[:, position, neuron_index]
                    )
                return patched

            handles.append(blocks[layer].mlp.act.register_forward_hook(patch_mlp))

        by_layer: Dict[int, List[Dict[str, Any]]] = {}
        for intervention in attention_sites:
            layer = intervention["layer"]
            head = int(intervention["head"])
            head_count = int(blocks[layer].attn.num_heads)
            if head < 0 or head >= head_count:
                raise ValueError(f"Invalid attention head {head} for layer {layer}.")
            by_layer.setdefault(layer, []).append(intervention)

        for layer, interventions in by_layer.items():
            module = blocks[layer].attn.c_proj
            baseline = baseline_attention_context[layer]
            head_dim = int(blocks[layer].attn.head_dim)

            def patch_attention(
                _module: Any,
                inputs: Tuple[torch.Tensor, ...],
                layer_interventions: List[Dict[str, Any]] = interventions,
                original_context: torch.Tensor = baseline,
                dimension: int = head_dim,
            ) -> Tuple[torch.Tensor, ...]:
                patched = inputs[0].clone()
                for intervention in layer_interventions:
                    head = int(intervention["head"])
                    start = head * dimension
                    end = start + dimension
                    for position in intervention["ablateTokenPositions"]:
                        patched[:, position, start:end] = (
                            0
                            if mode == "remove"
                            else original_context[:, position, start:end]
                        )
                return (patched,) + tuple(inputs[1:])

            handles.append(module.register_forward_pre_hook(patch_attention))

        with torch.no_grad():
            logits = model(**encoded, use_cache=False).logits[0, read_index].detach().float().cpu()
    finally:
        for handle in reversed(handles):
            handle.remove()

    summary = summarize_logits(logits, target_id, tokenizer, count=5)
    baseline_target = baseline_summary["target"]
    baseline_greedy = baseline_summary["greedyNextToken"]
    intervention_greedy_logit = float(logits[baseline_greedy["tokenId"]])
    result = {
        "mode": mode,
        "mlpSites": mlp_sites,
        "attentionSites": attention_sites,
        "targetLogit": summary["target"]["logit"],
        "targetProbability": summary["target"]["probability"],
        "targetRank": summary["target"]["rank"],
        "targetLogitDrop": baseline_target["logit"] - summary["target"]["logit"],
        "targetProbabilityChange": summary["target"]["probability"] - baseline_target["probability"],
        "greedyTokenAfterIntervention": summary["greedyNextToken"],
        "baselineGreedyLogitAfterIntervention": intervention_greedy_logit,
        "baselineGreedyLogitDrop": baseline_greedy["logit"] - intervention_greedy_logit,
        "baselineGreedyProbabilityChange": (
            float(torch.softmax(logits, dim=-1)[baseline_greedy["tokenId"]])
            - baseline_greedy["probability"]
        ),
        "greedyPredictionChanged": (
            summary["greedyNextToken"]["tokenId"] != baseline_greedy["tokenId"]
        ),
        "topTokensAfterIntervention": summary["topTokens"],
    }
    if baseline_logits is not None:
        result["fullDistributionChange"] = full_distribution_change(baseline_logits, logits)
    if retain_logits:
        result["_outputLogits"] = logits
    return result


def _component_magnitude(
    *,
    kind: str,
    node: Dict[str, Any],
    model: Any,
    capture: GPT2TraceCapture,
) -> float:
    layer = int(node["layer"])
    positions = [int(position) for position in node["ablateTokenPositions"]]
    if kind == "attention":
        head = int(node["head"])
        head_dim = int(model.transformer.h[layer].attn.head_dim)
        start = head * head_dim
        end = start + head_dim
        context = capture.values["attention_context"][layer][0]
        weight = model.transformer.h[layer].attn.c_proj.weight
        values = [
            torch.linalg.vector_norm(context[position, start:end] @ weight[start:end, :])
            for position in positions
        ]
    else:
        neuron = int(node["neuron"])
        activations = capture.values[f"mlp_activation_{layer}"][0]
        values = [activations[position, neuron].abs() for position in positions]
    return float(torch.stack(values).mean().detach().float().cpu())


def matched_random_control_groups(
    *,
    attention_nodes: List[Dict[str, Any]],
    mlp_nodes: List[Dict[str, Any]],
    model: Any,
    capture: GPT2TraceCapture,
    count: int,
    tolerance: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if count <= 0:
        raise ValueError("Matched control count must be positive.")
    if tolerance < 0:
        raise ValueError("Magnitude tolerance cannot be negative.")
    entries = [("attention", node) for node in attention_nodes] + [("mlp", node) for node in mlp_nodes]
    if not entries:
        raise ValueError("A matched control group requires at least one selected component.")

    selected_by_layer: Dict[Tuple[str, int], set[int]] = {}
    for kind, node in entries:
        unit = int(node["head"] if kind == "attention" else node["neuron"])
        selected_by_layer.setdefault((kind, int(node["layer"])), set()).add(unit)

    pools: List[List[Dict[str, Any]]] = []
    candidate_pools: List[List[Dict[str, Any]]] = []
    pool_metadata = []
    for kind, selected_node in entries:
        layer = int(selected_node["layer"])
        positions = [int(position) for position in selected_node["ablateTokenPositions"]]
        selected_unit = int(selected_node["head"] if kind == "attention" else selected_node["neuron"])
        selected_magnitude = _component_magnitude(
            kind=kind, node=selected_node, model=model, capture=capture
        )
        candidates = []
        width = (
            int(model.config.n_head)
            if kind == "attention"
            else int(capture.values[f"mlp_activation_{layer}"].shape[-1])
        )
        for unit in range(width):
            if unit in selected_by_layer[(kind, layer)]:
                continue
            candidate_node = {
                "layer": layer,
                "ablateTokenPositions": positions,
                "head" if kind == "attention" else "neuron": unit,
            }
            magnitude = _component_magnitude(
                kind=kind, node=candidate_node, model=model, capture=capture
            )
            relative_error = abs(magnitude - selected_magnitude) / max(abs(selected_magnitude), 1e-8)
            candidates.append({
                "node": candidate_node,
                "magnitude": magnitude,
                "relativeMagnitudeError": relative_error,
            })
        if not candidates:
            raise ValueError(f"No non-selected {kind} controls exist in layer {layer}.")
        candidates.sort(
            key=lambda item: (
                item["relativeMagnitudeError"],
                int(item["node"]["head"] if kind == "attention" else item["node"]["neuron"]),
            )
        )
        within_tolerance = [item for item in candidates if item["relativeMagnitudeError"] <= tolerance]
        if within_tolerance:
            pool = within_tolerance
            match_status = "within-tolerance"
        else:
            pool = [min(candidates, key=lambda item: item["relativeMagnitudeError"])]
            match_status = "nearest-fallback"
        pools.append(pool)
        candidate_pools.append(candidates)
        pool_metadata.append({
            "componentType": kind,
            "layer": layer,
            "tokenPositions": positions,
            "selectedUnit": selected_unit,
            "selectedActivationMagnitude": selected_magnitude,
            "candidateCount": len(candidates),
            "withinToleranceCandidateCount": len(within_tolerance),
            "controlPoolCount": len(pool),
            "matchingStatus": match_status,
            "relativeMagnitudeTolerance": tolerance,
        })

    while math.prod(len(pool) for pool in pools) < count:
        expandable = [
            index for index, pool in enumerate(pools)
            if len(pool) < len(candidate_pools[index])
        ]
        if not expandable:
            break
        selected_index = min(
            expandable,
            key=lambda index: (
                candidate_pools[index][len(pools[index])]["relativeMagnitudeError"],
                index,
            ),
        )
        pools[selected_index].append(candidate_pools[selected_index][len(pools[selected_index])])
        metadata = pool_metadata[selected_index]
        metadata["controlPoolCount"] = len(pools[selected_index])
        metadata["matchingStatus"] = (
            "within-tolerance"
            if all(item["relativeMagnitudeError"] <= tolerance for item in pools[selected_index])
            else "expanded-nearest-after-available-matches"
        )

    rng = random.Random(seed)
    controls: List[Dict[str, Any]] = []
    unique_keys = set()
    attempts = 0
    max_attempts = max(1000, count * 500)
    while len(controls) < count and attempts < max_attempts:
        attempts += 1
        chosen = [rng.choice(pool) for pool in pools]
        units_seen: Dict[Tuple[str, int], set[int]] = {}
        valid = True
        for (kind, _selected), candidate in zip(entries, chosen):
            node = candidate["node"]
            unit = int(node["head"] if kind == "attention" else node["neuron"])
            key = (kind, int(node["layer"]))
            if unit in units_seen.setdefault(key, set()):
                valid = False
                break
            units_seen[key].add(unit)
        if not valid:
            continue
        key = tuple(
            sorted(
                (
                    kind,
                    int(candidate["node"]["layer"]),
                    int(candidate["node"]["head"] if kind == "attention" else candidate["node"]["neuron"]),
                    tuple(candidate["node"]["ablateTokenPositions"]),
                )
                for (kind, _), candidate in zip(entries, chosen)
            )
        )
        if key in unique_keys:
            continue
        unique_keys.add(key)
        control = {"attention": [], "mlp": [], "magnitudeMatches": []}
        for (kind, _selected), candidate in zip(entries, chosen):
            node = candidate["node"]
            control["attention" if kind == "attention" else "mlp"].append(node)
            control["magnitudeMatches"].append({
                "componentType": kind,
                "layer": node["layer"],
                "unit": node["head"] if kind == "attention" else node["neuron"],
                "selectedMagnitude": pool_metadata[len(control["magnitudeMatches"])]["selectedActivationMagnitude"],
                "controlMagnitude": candidate["magnitude"],
                "relativeMagnitudeError": candidate["relativeMagnitudeError"],
                "withinTolerance": candidate["relativeMagnitudeError"] <= tolerance,
            })
        controls.append(control)

    if not controls:
        raise RuntimeError("Unable to construct even one unique matched control group.")
    unique_controls = list(controls)
    while len(controls) < count:
        controls.append(dict(rng.choice(unique_controls)))
    for index, control in enumerate(controls):
        control["controlIndex"] = index

    metadata = {
        "seed": seed,
        "requestedSampleCount": count,
        "uniqueSampleCount": len(unique_controls),
        "samplingWithReplacement": len(unique_controls) < count,
        "componentPools": pool_metadata,
        "exactStructuralMatch": True,
        "magnitudeMatchedUniqueGroupCount": sum(
            all(match["withinTolerance"] for match in control["magnitudeMatches"])
            for control in unique_controls
        ),
    }
    return controls, metadata


def run_batched_control_interventions(
    *,
    model: Any,
    encoded: Dict[str, torch.Tensor],
    read_index: int,
    baseline_logits: torch.Tensor,
    baseline_summary: Dict[str, Any],
    tokenizer: Any,
    controls: List[Dict[str, Any]],
    batch_size: int = 32,
) -> List[Dict[str, Any]]:
    if batch_size <= 0:
        raise ValueError("Control intervention batch size must be positive.")
    result_rows: List[Dict[str, Any]] = []
    blocks = model.transformer.h
    baseline = baseline_logits.detach().float().cpu()
    baseline_probabilities = torch.softmax(baseline, dim=-1)
    greedy_id = int(baseline_summary["greedyNextToken"]["tokenId"])
    target_id = int(baseline_summary["target"]["tokenId"])

    for start in range(0, len(controls), batch_size):
        batch_controls = controls[start : start + batch_size]
        batch_count = len(batch_controls)
        batch_encoded = {
            key: value.repeat(batch_count, *([1] * (value.ndim - 1)))
            for key, value in encoded.items()
        }
        mlp_by_layer: Dict[int, Dict[int, List[Dict[str, Any]]]] = {}
        attention_by_layer: Dict[int, Dict[int, List[Dict[str, Any]]]] = {}
        for row, control in enumerate(batch_controls):
            for node in control["mlp"]:
                mlp_by_layer.setdefault(int(node["layer"]), {}).setdefault(row, []).append(node)
            for node in control["attention"]:
                attention_by_layer.setdefault(int(node["layer"]), {}).setdefault(row, []).append(node)
        handles = []
        try:
            for layer, row_nodes in mlp_by_layer.items():
                module = blocks[layer].mlp.act

                def patch_mlp(
                    _module: Any,
                    _inputs: Any,
                    output: torch.Tensor,
                    interventions: Dict[int, List[Dict[str, Any]]] = row_nodes,
                ) -> torch.Tensor:
                    patched = output.clone()
                    for row_index, nodes in interventions.items():
                        for node in nodes:
                            neuron = int(node["neuron"])
                            for position in node["ablateTokenPositions"]:
                                patched[row_index, int(position), neuron] = 0
                    return patched

                handles.append(module.register_forward_hook(patch_mlp))

            for layer, row_nodes in attention_by_layer.items():
                module = blocks[layer].attn.c_proj
                head_dim = int(blocks[layer].attn.head_dim)

                def patch_attention(
                    _module: Any,
                    inputs: Tuple[torch.Tensor, ...],
                    interventions: Dict[int, List[Dict[str, Any]]] = row_nodes,
                    dimension: int = head_dim,
                ) -> Tuple[torch.Tensor, ...]:
                    patched = inputs[0].clone()
                    for row_index, nodes in interventions.items():
                        for node in nodes:
                            first = int(node["head"]) * dimension
                            last = first + dimension
                            for position in node["ablateTokenPositions"]:
                                patched[row_index, int(position), first:last] = 0
                    return (patched,) + tuple(inputs[1:])

                handles.append(module.register_forward_pre_hook(patch_attention))

            with torch.no_grad():
                changed = model(**batch_encoded, use_cache=False).logits[:, read_index, :].detach().float().cpu()
        finally:
            for handle in reversed(handles):
                handle.remove()

        probabilities = torch.softmax(changed, dim=-1)
        midpoint = (baseline_probabilities.unsqueeze(0) + probabilities) * 0.5
        epsilon = torch.finfo(probabilities.dtype).tiny
        abs_delta = torch.abs(probabilities - baseline_probabilities.unsqueeze(0))
        js = 0.5 * (
            (
                baseline_probabilities.unsqueeze(0)
                * (torch.log(baseline_probabilities.unsqueeze(0).clamp_min(epsilon)) - torch.log(midpoint.clamp_min(epsilon)))
            ).sum(dim=-1)
            + (
                probabilities
                * (torch.log(probabilities.clamp_min(epsilon)) - torch.log(midpoint.clamp_min(epsilon)))
            ).sum(dim=-1)
        )
        js = js.clamp_min(0.0)
        changed_greedy = changed.argmax(dim=-1)
        top_values, top_ids = torch.topk(changed, k=min(5, changed.shape[-1]), dim=-1)
        for row, control in enumerate(batch_controls):
            result_rows.append({
                "controlIndex": control["controlIndex"],
                "attention": control["attention"],
                "mlp": control["mlp"],
                "magnitudeMatches": control["magnitudeMatches"],
                "targetLogitDrop": float(baseline[target_id] - changed[row, target_id]),
                "targetProbabilityChange": float(probabilities[row, target_id] - baseline_probabilities[target_id]),
                "baselineGreedyLogitDrop": float(baseline[greedy_id] - changed[row, greedy_id]),
                "baselineGreedyProbabilityChange": float(probabilities[row, greedy_id] - baseline_probabilities[greedy_id]),
                "greedyPredictionAfterIntervention": {
                    "tokenId": int(changed_greedy[row]),
                    "token": tokenizer.decode([int(changed_greedy[row])]).replace("\n", "\\n"),
                    "logit": float(top_values[row, 0]),
                    "probability": float(probabilities[row, changed_greedy[row]]),
                },
                "greedyPredictionChanged": int(changed_greedy[row]) != greedy_id,
                "fullDistributionChange": {
                    "totalVariation": float(abs_delta[row].sum() * 0.5),
                    "l1Probability": float(abs_delta[row].sum()),
                    "jensenShannonDivergence": float(js[row]),
                    "maxAbsoluteProbabilityChange": float(abs_delta[row].max()),
                },
                "topFiveAfterIntervention": [
                    {
                        "rank": rank + 1,
                        "tokenId": int(top_ids[row, rank]),
                        "token": tokenizer.decode([int(top_ids[row, rank])]).replace("\n", "\\n"),
                        "logit": float(top_values[row, rank]),
                        "probability": float(probabilities[row, top_ids[row, rank]]),
                    }
                    for rank in range(top_ids.shape[1])
                ],
            })
    return result_rows


def capture_prediction_trace(
    *,
    capture: GPT2TraceCapture,
    model: Any,
    read_index: int,
    gradients: Dict[str, torch.Tensor],
    archive_arrays: Dict[str, np.ndarray],
    prefix: str,
    context_ids: torch.Tensor,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    layers = capture.layers
    head_count = int(model.config.n_head)
    hidden = int(model.config.n_embd)
    layer_trace = []
    reconstruction_errors = []
    head_residual_writes = []
    head_attributions = []
    mlp_attributions = []
    mlp_write_norms = []
    head_top_rows = []
    mlp_top_rows = []
    block_inputs = _ordered_layer_tensors(capture, "block_input")
    block_outputs = _ordered_layer_tensors(capture, "block_output")
    attention_contexts = _ordered_layer_tensors(capture, "attention_context")
    mlp_inputs = _ordered_layer_tensors(capture, "mlp_input")

    for layer in range(layers):
        block_input = block_inputs[layer][0, read_index]
        block_output = block_outputs[layer][0, read_index]
        attention_output = capture.values[f"attention_output_{layer}"][0, read_index]
        mlp_output = capture.values[f"mlp_output_{layer}"][0, read_index]
        attention_context = attention_contexts[layer]
        attention_weight = model.transformer.h[layer].attn.c_proj.weight
        mlp_weight = model.transformer.h[layer].mlp.c_proj.weight
        attention_bias = model.transformer.h[layer].attn.c_proj.bias
        mlp_bias = model.transformer.h[layer].mlp.c_proj.bias
        projected_heads = conv1d_head_writes(attention_context, attention_weight, head_count)
        head_write = projected_heads[0, read_index]
        head_gradient = gradients[f"attention_output_{layer}"][0, read_index]
        head_attr = torch.einsum("hd,d->h", head_write, head_gradient)
        mlp_activation = capture.values[f"mlp_activation_{layer}"][0, read_index]
        mlp_gradient = gradients[f"mlp_output_{layer}"][0, read_index]
        neuron_attr = mlp_neuron_first_order_attribution(mlp_activation, mlp_weight, mlp_gradient)
        neuron_write_norm = mlp_activation.abs() * torch.linalg.vector_norm(mlp_weight, dim=1)
        reconstruction_error = _scalar(
            (block_output - block_input - attention_output - mlp_output).abs().max()
        )
        reconstruction_errors.append(reconstruction_error)
        head_residual_writes.append(head_write)
        head_attributions.append(head_attr)
        mlp_attributions.append(neuron_attr)
        mlp_write_norms.append(neuron_write_norm)
        head_top_rows.append(_top_contribution_rows(
            kind="attention-head",
            layer=layer,
            attributions=head_attr,
        ))
        mlp_top_rows.append(_top_contribution_rows(
            kind="mlp-neuron",
            layer=layer,
            attributions=neuron_attr,
        ))
        layer_trace.append({
            "layer": layer,
            "residualInputL2": _scalar(torch.linalg.vector_norm(block_input)),
            "attentionResidualWriteL2": _scalar(torch.linalg.vector_norm(attention_output)),
            "mlpResidualWriteL2": _scalar(torch.linalg.vector_norm(mlp_output)),
            "residualOutputL2": _scalar(torch.linalg.vector_norm(block_output)),
            "attentionChangeL2": _scalar(torch.linalg.vector_norm(attention_output)),
            "mlpChangeL2": _scalar(torch.linalg.vector_norm(mlp_output)),
            "residualAddMaxAbsReconstructionError": reconstruction_error,
            "attentionFirstOrderPredictedTokenAttribution": _scalar(torch.dot(attention_output, head_gradient)),
            "mlpFirstOrderPredictedTokenAttribution": _scalar(torch.dot(mlp_output, mlp_gradient)),
            "attentionBiasFirstOrderPredictedTokenAttribution": _scalar(torch.dot(attention_bias, head_gradient)),
            "mlpBiasFirstOrderPredictedTokenAttribution": _scalar(torch.dot(mlp_bias, mlp_gradient)),
        })

    input_embedding = capture.values["token_embedding"][0, read_index]
    position_embedding = capture.values["position_embedding"][0, read_index]
    embedding_output = capture.values["embedding_output"][0, read_index]
    final_norm_input = capture.values["final_norm_input"][0, read_index]
    final_norm_output = capture.values["final_norm_output"][0, read_index]
    full_logits = capture.values["lm_head_logits"][0, read_index]
    archive_arrays[f"{prefix}_input_ids"] = _tensor_to_numpy(context_ids[0])
    archive_arrays[f"{prefix}_read_token_index"] = np.asarray(read_index, dtype=np.int64)
    for suffix, value in (
        ("token_embedding", input_embedding),
        ("position_embedding", position_embedding),
        ("embedding_output", embedding_output),
        ("residual_input", torch.stack([value[0, read_index] for value in block_inputs])),
        ("residual_output", torch.stack([value[0, read_index] for value in block_outputs])),
        ("attention_output", torch.stack([capture.values[f"attention_output_{layer}"][0, read_index] for layer in range(layers)])),
        ("attention_context_by_head", torch.stack([
            attention_contexts[layer][0, read_index].reshape(head_count, hidden // head_count)
            for layer in range(layers)
        ])),
        ("attention_head_residual_write", torch.stack(head_residual_writes)),
        ("attention_projection_bias", torch.stack([
            model.transformer.h[layer].attn.c_proj.bias.detach() for layer in range(layers)
        ])),
        ("mlp_input", torch.stack([value[0, read_index] for value in mlp_inputs])),
        ("mlp_preactivation", torch.stack([
            capture.values[f"mlp_preactivation_{layer}"][0, read_index] for layer in range(layers)
        ])),
        ("mlp_activation", torch.stack([
            capture.values[f"mlp_activation_{layer}"][0, read_index] for layer in range(layers)
        ])),
        ("mlp_output", torch.stack([
            capture.values[f"mlp_output_{layer}"][0, read_index] for layer in range(layers)
        ])),
        ("mlp_projection_bias", torch.stack([
            model.transformer.h[layer].mlp.c_proj.bias.detach() for layer in range(layers)
        ])),
        ("mlp_neuron_first_order_predicted_token_attribution", torch.stack(mlp_attributions)),
        ("mlp_neuron_residual_write_norm", torch.stack(mlp_write_norms)),
        ("attention_head_first_order_predicted_token_attribution", torch.stack(head_attributions)),
        ("final_norm_input", final_norm_input),
        ("final_norm_output", final_norm_output),
        ("lm_head_logits", full_logits),
    ):
        archive_arrays[f"{prefix}_{suffix}"] = _tensor_to_numpy(value)

    return layer_trace, {
        "maxResidualAddReconstructionError": max(reconstruction_errors),
        "hiddenSize": hidden,
        "attentionHeadCount": head_count,
        "topAttentionHeadAttributionsByLayer": head_top_rows,
        "topMlpNeuronAttributionsByLayer": mlp_top_rows,
    }


def teacher_forced_intervention_record(
    *,
    baseline_logits: torch.Tensor,
    changed_logits: torch.Tensor,
    target_id: int,
    tokenizer: Any,
) -> Dict[str, Any]:
    baseline = summarize_logits(baseline_logits, target_id, tokenizer, count=5)
    changed = summarize_logits(changed_logits, target_id, tokenizer, count=5)
    return {
        "targetTokenId": target_id,
        "targetToken": baseline["target"]["token"],
        "baselineTargetLogit": baseline["target"]["logit"],
        "changedTargetLogit": changed["target"]["logit"],
        "targetLogitDrop": baseline["target"]["logit"] - changed["target"]["logit"],
        "baselineTargetProbability": baseline["target"]["probability"],
        "changedTargetProbability": changed["target"]["probability"],
        "targetProbabilityChange": changed["target"]["probability"] - baseline["target"]["probability"],
        "baselineTargetRank": baseline["target"]["rank"],
        "changedTargetRank": changed["target"]["rank"],
        "greedyTokenAfterIntervention": changed["greedyNextToken"],
        "fullDistributionChange": full_distribution_change(baseline_logits, changed_logits),
        "topFiveAfterIntervention": changed["topTokens"],
    }


def _intervention_sites(
    *,
    model: Any,
    encoded: Dict[str, torch.Tensor],
    read_index: int,
    predicted_token_id: int,
    baseline_summary: Dict[str, Any],
    baseline_logits: torch.Tensor,
    tokenizer: Any,
    baseline_mlp_activations: Dict[int, torch.Tensor],
    baseline_attention_context: Dict[int, torch.Tensor],
    attention_sites: List[Dict[str, Any]],
    mlp_sites: List[Dict[str, Any]],
    mode: str,
    retain_logits: bool = False,
) -> Dict[str, Any]:
    return run_intervention(
        model=model,
        encoded=encoded,
        read_index=read_index,
        target_id=predicted_token_id,
        baseline_summary=baseline_summary,
        tokenizer=tokenizer,
        mlp_interventions=mlp_sites,
        attention_interventions=attention_sites,
        baseline_mlp_activations=baseline_mlp_activations,
        baseline_attention_context=baseline_attention_context,
        mode=mode,
        baseline_logits=baseline_logits,
        retain_logits=retain_logits,
    )


def _empirical_percentile(values: List[float], value: float) -> float:
    if not values:
        raise ValueError("Empirical percentile requires at least one control measurement.")
    less = sum(item < value for item in values)
    equal = sum(item == value for item in values)
    return 100.0 * (less + 0.5 * equal) / len(values)


def _top_contribution_rows(
    *,
    kind: str,
    layer: int,
    attributions: torch.Tensor,
    count: int = 5,
) -> List[Dict[str, Any]]:
    selected = torch.argsort(attributions.abs(), descending=True)[:count]
    return [
        {
            "componentType": kind,
            "layer": layer,
            "unit": int(index),
            "firstOrderPredictedTokenAttribution": float(attributions[index].detach().float().cpu()),
        }
        for index in selected
    ]


def explain_autoregressive_case(case: Dict[str, Any]) -> str:
    case_lines = [
        f"## {case['subject']}: {case['prompt']}",
        "",
        f"Source fact token: `{case['teacherForced']['sourceFactToken']}`; "
        f"generated {len(case['generation']['steps'])} greedy tokens. "
        "All causal effects below use the unchanged baseline-generated context for that step.",
        "",
        "### Teacher-forced score (not free-running generation)",
        "",
        f"The source artifact's manually selected fact token had rank "
        f"{case['teacherForced']['sourceFactTarget']['rank']} on the original prompt; "
        f"the model's actual first greedy token was "
        f"`{case['generation']['steps'][0]['generatedToken']['token']}`. "
        f"Selected-path ablation changed the teacher-forced target logit by "
        f"{case['teacherForced']['selectedPathTargetLogitDrop']:.6f}.",
        "",
    ]
    for step in case["generation"]["steps"]:
        baseline = step["baseline"]
        causal = step["causal"]
        generated = step["generatedToken"]
        interaction = causal["interaction"]
        specificity = causal["specificity"]
        case_lines.extend([
            f"### Token {step['stepIndex'] + 1}: `{generated['token']}` (id {generated['tokenId']})",
            "",
            f"Context token IDs: `{step['contextTokenIds']}`. The actual greedy token had "
            f"logit {baseline['greedyNextToken']['logit']:.6f}, probability "
            f"{baseline['greedyNextToken']['probability']:.8g}; top-five competitors: "
            + ", ".join(
                f"`{item['token']}` ({item['probability']:.5g})"
                for item in baseline["topTokens"]
            )
            + ".",
            "",
            "**WHAT:** The unchanged source path includes "
            + ", ".join(
                f"{node['componentType']} L{node['layer']} unit {node['unit']} "
                f"at positions {node['tokenPositions']}"
                for node in step["causal"]["selectedComponents"]
            )
            + f". Every layer records residual/attention/MLP values; the largest first-order "
            f"contributors are stored per layer and in the raw tensor archive.",
            "",
            "**HOW:** The computation was replayed on the exact growing context. "
            f"Layerwise residual-add maximum error was "
            f"{step['trace']['maxResidualAddReconstructionError']:.3g}. The archive stores "
            f"the read-position residual inputs/outputs, each attention output and per-head "
            f"projected writes, MLP inputs/preactivations/activations/outputs, per-neuron "
            f"first-order target contributions, final normalization, and full vocabulary logits.",
            "",
            "**WHY:** Ablating the discovered group changed the actual next-token logit by "
            f"{causal['selectedPath']['remove']['targetLogitDrop']:.6f} and its probability by "
            f"{causal['selectedPath']['remove']['targetProbabilityChange']:.8g}; restoration "
            f"maximum full-logit error was {causal['selectedPath']['restoreMaxLogitError']:.3g}. "
            f"Its signed target-logit drop was at percentile "
            f"{specificity['empiricalPercentile']:.1f} (absolute-effect percentile "
            f"{specificity['absoluteEffectPercentile']:.1f}) among "
            f"{specificity['sampleCount']} deterministic structurally matched-control draws "
            f"({specificity['uniqueControlCount']} unique groups). The interaction term "
            f"(group drop minus summed individual drops) was "
            f"{interaction['logitDropNonAdditivity']:.6f} "
            f"({interaction['interpretation']}).",
            "",
            "**WHAT ELSE:** After path ablation the greedy token "
            f"{'changed' if causal['selectedPath']['remove']['greedyPredictionChanged'] else 'did not change'}; "
            f"the altered top five and full-vocabulary TV/JS changes are recorded. Matched controls "
            f"had {specificity['magnitudeMatchedControlCount']}/{specificity['sampleCount']} "
            f"groups within the preregistered activation-magnitude tolerance.",
            "",
            "**LIMITATIONS:** This is a local-weight, hook-instrumented measurement. "
            "Ablations estimate causal influence on one next-token distribution conditional on "
            "the baseline context; when an intervention changes the greedy token, later "
            "free-running contexts are not regenerated from that altered output and are not "
            "interpreted as isolated downstream effects. Path sufficiency is not tested, and "
            "first-order attributions do not assign semantic meanings to units.",
            "",
        ])
    return "\n".join(case_lines)


def run_autoregressive_case(
    *,
    source_case: Dict[str, Any],
    case_index: int,
    model: Any,
    tokenizer: Any,
    archive_arrays: Dict[str, np.ndarray],
    source_sha256: str,
    max_new_tokens: int,
    control_count: int,
    magnitude_tolerance: float,
) -> Dict[str, Any]:
    prompt = str(source_case["prompt"])
    encoded_prompt = tokenizer(
        prompt,
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    offsets = encoded_prompt.pop("offset_mapping")[0].tolist()
    device = next(model.parameters()).device
    prompt_ids = [int(value) for value in encoded_prompt["input_ids"][0].tolist()]
    read_index = len(prompt_ids) - 1
    if read_index != int(source_case["readTokenIndex"]):
        raise ValueError(f"{source_case['subject']}: source read-token index differs from prompt tokenization.")
    source_target_id = int(source_case["target"]["tokenId"])
    source_target_token = tokenizer.decode([source_target_id]).replace("\n", "\\n")
    if source_target_token != source_case["target"]["token"]:
        raise ValueError(f"{source_case['subject']}: source target token ID/text mismatch.")
    attention_nodes, mlp_nodes = selected_path_nodes(source_case)
    source_nodes = [
        {"componentType": "attention", **node} for node in attention_nodes
    ] + [
        {"componentType": "mlp", **node} for node in mlp_nodes
    ]
    selected_component_summary = [
        {
            "componentType": "attention",
            "layer": int(node["layer"]),
            "unit": int(node["head"]),
            "tokenPositions": list(node["ablateTokenPositions"]),
        }
        for node in attention_nodes
    ] + [
        {
            "componentType": "mlp",
            "layer": int(node["layer"]),
            "unit": int(node["neuron"]),
            "tokenPositions": list(node["ablateTokenPositions"]),
        }
        for node in mlp_nodes
    ]
    prefix = f"case_{case_index:02d}_{hashlib.sha256(str(source_case['subject']).encode()).hexdigest()[:10]}"
    context_ids = list(prompt_ids)
    generated_ids: List[int] = []
    steps = []
    teacher_forced = None
    termination = "max-new-tokens"
    started = time.perf_counter()

    for step_index in range(max_new_tokens):
        input_ids = torch.tensor([context_ids], dtype=torch.long, device=device)
        encoded = {"input_ids": input_ids}
        current_read_index = len(context_ids) - 1
        model.zero_grad(set_to_none=True)
        with GPT2TraceCapture(model) as capture:
            outputs = model(**encoded, use_cache=False)
            full_logits = outputs.logits[0, current_read_index]
            actual_token_id = int(torch.argmax(full_logits.detach()))
            actual_summary = summarize_logits(full_logits, actual_token_id, tokenizer, count=5)
            gradient_refs = capture.gradient_outputs + [
                (f"block_output_{layer}", capture.values["block_output"][layer])
                for layer in range(capture.layers)
            ]
            gradient_values = torch.autograd.grad(
                full_logits[actual_token_id],
                [tensor for _, tensor in gradient_refs],
                retain_graph=False,
                allow_unused=False,
            )
            gradients = {
                name: value.detach()
                for (name, _tensor), value in zip(gradient_refs, gradient_values)
            }
            trace, trace_meta = capture_prediction_trace(
                capture=capture,
                model=model,
                read_index=current_read_index,
                gradients=gradients,
                archive_arrays=archive_arrays,
                prefix=f"{prefix}_step_{step_index:02d}",
                context_ids=input_ids,
            )
            baseline_attention_context = _ordered_layer_tensors(capture, "attention_context")
            baseline_mlp_activations = {
                layer: capture.values[f"mlp_activation_{layer}"] for layer in range(capture.layers)
            }
            source_seed_material = (
                f"{source_sha256}:{source_case['subject']}:{step_index}:"
                + ",".join(str(token) for token in context_ids)
            )
            control_seed = int(hashlib.sha256(source_seed_material.encode()).hexdigest()[:16], 16)
            controls, control_metadata = matched_random_control_groups(
                attention_nodes=attention_nodes,
                mlp_nodes=mlp_nodes,
                model=model,
                capture=capture,
                count=control_count,
                tolerance=magnitude_tolerance,
                seed=control_seed,
            )

        baseline_logits_cpu = full_logits.detach().float().cpu()
        baseline_summary = actual_summary
        predicted_token = actual_summary["greedyNextToken"]
        selected_remove = _intervention_sites(
            model=model,
            encoded=encoded,
            read_index=current_read_index,
            predicted_token_id=actual_token_id,
            baseline_summary=baseline_summary,
            baseline_logits=baseline_logits_cpu,
            tokenizer=tokenizer,
            baseline_mlp_activations=baseline_mlp_activations,
            baseline_attention_context=baseline_attention_context,
            attention_sites=attention_nodes,
            mlp_sites=mlp_nodes,
            mode="remove",
            retain_logits=True,
        )
        selected_remove_logits = selected_remove.pop("_outputLogits")
        selected_restore = _intervention_sites(
            model=model,
            encoded=encoded,
            read_index=current_read_index,
            predicted_token_id=actual_token_id,
            baseline_summary=baseline_summary,
            baseline_logits=baseline_logits_cpu,
            tokenizer=tokenizer,
            baseline_mlp_activations=baseline_mlp_activations,
            baseline_attention_context=baseline_attention_context,
            attention_sites=attention_nodes,
            mlp_sites=mlp_nodes,
            mode="restore",
            retain_logits=True,
        )
        selected_restore_logits = selected_restore.pop("_outputLogits")
        restore_max_logit_error = float((selected_restore_logits - baseline_logits_cpu).abs().max())

        individual_rows = []
        individual_logits = []
        for component in source_nodes:
            kind = component["componentType"]
            node = {key: value for key, value in component.items() if key != "componentType"}
            changed = _intervention_sites(
                model=model,
                encoded=encoded,
                read_index=current_read_index,
                predicted_token_id=actual_token_id,
                baseline_summary=baseline_summary,
                baseline_logits=baseline_logits_cpu,
                tokenizer=tokenizer,
                baseline_mlp_activations=baseline_mlp_activations,
                baseline_attention_context=baseline_attention_context,
                attention_sites=[node] if kind == "attention" else [],
                mlp_sites=[node] if kind == "mlp" else [],
                mode="remove",
                retain_logits=True,
            )
            individual_logits.append(changed.pop("_outputLogits"))
            individual_rows.append({
                "component": component,
                "remove": changed,
            })
        interaction = interaction_from_logits(
            baseline_logits_cpu,
            selected_remove_logits,
            individual_logits,
            actual_token_id,
        )
        normalized_interaction = abs(interaction["logitDropNonAdditivity"]) / max(
            abs(interaction["sumIndividualLogitDrops"]), 1e-8
        )
        if normalized_interaction <= 0.1:
            interaction["interpretation"] = "approximately-additive"
        elif interaction["logitDropNonAdditivity"] < 0:
            interaction["interpretation"] = "subadditive-overlap-consistent"
        else:
            interaction["interpretation"] = "superadditive-synergy-consistent"
        interaction["normalizedAbsoluteNonAdditivity"] = normalized_interaction
        interaction_vector = (
            (1 - len(individual_logits)) * baseline_logits_cpu - selected_remove_logits
        )
        for changed_logits in individual_logits:
            interaction_vector += changed_logits

        batched_controls = run_batched_control_interventions(
            model=model,
            encoded=encoded,
            read_index=current_read_index,
            baseline_logits=baseline_logits_cpu,
            baseline_summary=baseline_summary,
            tokenizer=tokenizer,
            controls=controls,
        )
        selected_drop = selected_remove["targetLogitDrop"]
        control_drops = [item["targetLogitDrop"] for item in batched_controls]
        absolute_effect_percentile = _empirical_percentile(
            [abs(value) for value in control_drops],
            abs(selected_drop),
        )
        signed_percentile = _empirical_percentile(control_drops, selected_drop)
        percentile = signed_percentile
        magnitude_matched_count = sum(
            all(match["withinTolerance"] for match in item["magnitudeMatches"])
            for item in batched_controls
        )
        unique_control_count = control_metadata["uniqueSampleCount"]
        specificity_pass = (
            percentile >= 95.0
            and magnitude_matched_count >= math.ceil(control_count * 0.95)
            and unique_control_count >= 20
            and control_metadata["exactStructuralMatch"]
        )
        specificity = {
            "empiricalPercentile": percentile,
            "percentileMetric": "signed-target-logit-drop",
            "absoluteEffectPercentile": absolute_effect_percentile,
            "signedEmpiricalPercentile": signed_percentile,
            "sampleCount": len(batched_controls),
            "uniqueControlCount": unique_control_count,
            "magnitudeMatchedControlCount": magnitude_matched_count,
            "magnitudeMatchCoverage": magnitude_matched_count / len(batched_controls),
            "structurallyMatched": control_metadata["exactStructuralMatch"],
            "strictSpecificityThresholdPercentile": 95.0,
            "passesStrictSpecificity": specificity_pass,
            "interpretation": (
                "path-exceeds-matched-controls-at-preregistered-threshold"
                if specificity_pass
                else "selectivity-not-established-by-matched-controls"
            ),
        }

        for suffix, value in (
            ("baseline_logits", baseline_logits_cpu),
            ("selected_path_ablated_logits", selected_remove_logits),
            ("selected_path_restored_logits", selected_restore_logits),
            ("interaction_full_vocabulary_logit_delta", interaction_vector),
        ):
            archive_arrays[f"{prefix}_step_{step_index:02d}_{suffix}"] = _tensor_to_numpy(value)
        for component_index, changed_logits in enumerate(individual_logits):
            archive_arrays[
                f"{prefix}_step_{step_index:02d}_individual_{component_index}_ablated_logits"
            ] = _tensor_to_numpy(changed_logits)

        teacher_forced_step = None
        if step_index == 0:
            source_target_summary = summarize_logits(
                baseline_logits_cpu, source_target_id, tokenizer, count=5
            )
            teacher_group = teacher_forced_intervention_record(
                baseline_logits=baseline_logits_cpu,
                changed_logits=selected_remove_logits,
                target_id=source_target_id,
                tokenizer=tokenizer,
            )
            teacher_singles = [
                teacher_forced_intervention_record(
                    baseline_logits=baseline_logits_cpu,
                    changed_logits=changed_logits,
                    target_id=source_target_id,
                    tokenizer=tokenizer,
                )
                for changed_logits in individual_logits
            ]
            teacher_forced = {
                "mode": "teacher-forced scoring of the source artifact token on the original prompt",
                "sourceFactToken": source_target_summary["target"]["token"],
                "sourceFactTarget": source_target_summary["target"],
                "originalGreedyToken": source_target_summary["greedyNextToken"],
                "selectedPathTargetLogitDrop": teacher_group["targetLogitDrop"],
                "selectedPathTargetProbabilityChange": teacher_group["targetProbabilityChange"],
                "selectedPathFullDistributionChange": teacher_group["fullDistributionChange"],
                "individualComponentTargetLogitDrops": [
                    {
                        "component": component,
                        "targetLogitDrop": summary["targetLogitDrop"],
                        "targetProbabilityChange": summary["targetProbabilityChange"],
                    }
                    for component, summary in zip(source_nodes, teacher_singles)
                ],
            }
            teacher_forced_step = teacher_forced

        step = {
            "stepIndex": step_index,
            "contextTokenIds": list(context_ids),
            "generatedPrefixTokenIds": list(generated_ids),
            "readTokenIndex": current_read_index,
            "baseline": baseline_summary,
            "generatedToken": predicted_token,
            "trace": {
                "layerCount": len(trace),
                "layerTrace": trace,
                "maxResidualAddReconstructionError": trace_meta["maxResidualAddReconstructionError"],
                "topAttentionHeadAttributionsByLayer": trace_meta["topAttentionHeadAttributionsByLayer"],
                "topMlpNeuronAttributionsByLayer": trace_meta["topMlpNeuronAttributionsByLayer"],
            },
            "causal": {
                "interventionOperator": "zero selected attention head context slice before c_proj; zero selected MLP neuron activation; same operator for controls",
                "selectedComponents": selected_component_summary,
                "selectedPath": {
                    "remove": selected_remove,
                    "restore": selected_restore,
                    "restoreMaxLogitError": restore_max_logit_error,
                    "restorePredictionMatchesBaseline": int(selected_restore["greedyTokenAfterIntervention"]["tokenId"]) == actual_token_id,
                },
                "individualComponentAblations": individual_rows,
                "interaction": interaction,
                "matchedControlSampling": control_metadata,
                "matchedControls": batched_controls,
                "specificity": specificity,
                "causalInfluence": {
                    "logitDropMagnitude": abs(selected_drop),
                    "measurableAtTolerance1e-6": abs(selected_drop) > 1e-6,
                    "predictionChangedAfterAblation": selected_remove["greedyPredictionChanged"],
                },
                "causalSufficiency": {
                    "status": "not-tested",
                    "reason": "No complement ablation was run that removes all non-path computation while preserving the discovered path.",
                },
            },
            "teacherForcedSourceTargetAtStepZero": teacher_forced_step,
            "rawArchivePrefix": f"{prefix}_step_{step_index:02d}",
        }
        steps.append(step)
        generated_ids.append(actual_token_id)
        if tokenizer.eos_token_id is not None and actual_token_id == int(tokenizer.eos_token_id):
            termination = "eos-token"
            break
        context_ids.append(actual_token_id)

    if not steps:
        raise RuntimeError(f"{source_case['subject']}: autoregressive run produced no prediction steps.")
    step_times = time.perf_counter() - started
    step_trace_ok = all(
        step["trace"]["layerCount"] == len(model.transformer.h)
        and step["trace"]["maxResidualAddReconstructionError"] <= 2e-4
        for step in steps
    )
    greedy_reconstruction_ok = all(
        step["generatedToken"]["tokenId"] == step["baseline"]["greedyNextToken"]["tokenId"]
        and step["generatedToken"]["tokenId"] == int(
            np.argmax(archive_arrays[f"{step['rawArchivePrefix']}_baseline_logits"])
        )
        for step in steps
    )
    selected_effects_ok = all(
        step["causal"]["causalInfluence"]["measurableAtTolerance1e-6"]
        for step in steps
    )
    specificity_ok = all(
        step["causal"]["specificity"]["passesStrictSpecificity"]
        for step in steps
    )
    report_case = {
        "subject": source_case["subject"],
        "answer": source_case["answer"],
        "prompt": prompt,
        "promptSha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "tokenization": {
            "promptTokenIds": prompt_ids,
            "promptTokens": [tokenizer.convert_ids_to_tokens(token) for token in prompt_ids],
            "promptCharacterOffsets": [[int(start), int(end)] for start, end in offsets],
            "readTokenIndex": read_index,
        },
        "sourcePath": {"attention": attention_nodes, "mlp": mlp_nodes},
        "teacherForced": teacher_forced,
        "generation": {
            "algorithm": "greedy-autoregressive; recompute full context at each step; no intervention feeds into later contexts",
            "maxNewTokens": max_new_tokens,
            "generatedTokenIds": generated_ids,
            "generatedText": tokenizer.decode(generated_ids),
            "termination": termination,
            "steps": steps,
        },
        "criteria": {
            "A_internalComputationalTrace": {
                "status": "pass" if step_trace_ok else "fail",
                "traceLayersComplete": step_trace_ok,
            },
            "B_correctTokenByTokenReconstruction": {
                "status": "pass" if greedy_reconstruction_ok else "fail",
                "everyGeneratedTokenEqualsUnmodifiedGreedyArgmax": greedy_reconstruction_ok,
            },
            "C_causalInfluence": {
                "status": "pass" if selected_effects_ok else "fail",
                "everyStepHasTargetLogitChangeOver1e-6": selected_effects_ok,
            },
            "D_matchedControlSpecificity": {
                "status": "pass" if specificity_ok else "fail",
                "everyStepPasses95thPercentileAndMatchingCoverageGate": specificity_ok,
            },
            "E_interactionAndRedundancyMeasurement": {
                "status": "pass" if all(math.isfinite(step["causal"]["interaction"]["logitDropNonAdditivity"]) for step in steps) else "fail",
                "perStepNonAdditivityRecorded": True,
            },
            "F_actualGreedyPredictionExplanation": {
                "status": "pass" if greedy_reconstruction_ok and all(step["causal"]["selectedPath"]["remove"].get("topTokensAfterIntervention") for step in steps) else "fail",
                "freeRunningTokensAndConditionalInterventionsRecorded": True,
            },
            "G_sourceCaseReproducibility": {
                "status": "pass",
                "exactSourcePromptPathTargetAndControlSeedRecorded": True,
            },
            "causalSufficiency": {
                "status": "not-tested",
                "reason": "The experiment tests influence and matched specificity, not path-only sufficiency.",
            },
        },
        "runtimeSeconds": step_times,
    }
    report_case["explanationMarkdown"] = explain_autoregressive_case(report_case)
    return report_case


def raw_layer_trace(
    *,
    capture: GPT2TraceCapture,
    model: Any,
    read_index: int,
    target_id: int,
    gradients: Dict[str, torch.Tensor],
    archive_arrays: Dict[str, np.ndarray],
    case_prefix: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    layers = capture.layers
    block_inputs = _ordered_layer_tensors(capture, "block_input")
    block_outputs = _ordered_layer_tensors(capture, "block_output")
    attention_outputs = [capture.values[f"attention_output_{layer}"] for layer in range(layers)]
    attention_context = _ordered_layer_tensors(capture, "attention_context")
    mlp_inputs = _ordered_layer_tensors(capture, "mlp_input")
    mlp_pre = [capture.values[f"mlp_preactivation_{layer}"] for layer in range(layers)]
    mlp_activation = [capture.values[f"mlp_activation_{layer}"] for layer in range(layers)]
    mlp_outputs = [capture.values[f"mlp_output_{layer}"] for layer in range(layers)]

    head_count = int(model.config.n_head)
    head_writes = []
    head_attributions = []
    neuron_attributions = []
    neuron_write_norms = []
    neuron_projection_biases = []
    layer_trace = []
    head_summary = []
    reconstruction_errors = []

    for layer in range(layers):
        attention_write = attention_outputs[layer]
        mlp_write = mlp_outputs[layer]
        attention_gradient = gradients[f"attention_output_{layer}"]
        mlp_gradient = gradients[f"mlp_output_{layer}"]
        context = attention_context[layer]
        attention_weight = model.transformer.h[layer].attn.c_proj.weight
        attention_bias = model.transformer.h[layer].attn.c_proj.bias
        mlp_weight = model.transformer.h[layer].mlp.c_proj.weight
        mlp_bias = model.transformer.h[layer].mlp.c_proj.bias
        writes = conv1d_head_writes(context, attention_weight, head_count)
        head_writes.append(writes)
        head_attr = torch.einsum(
            "hd,d->h",
            writes[0, read_index],
            attention_gradient[0, read_index],
        )
        head_attributions.append(head_attr)
        activations = mlp_activation[layer][0, read_index]
        attr = mlp_neuron_first_order_attribution(activations, mlp_weight, mlp_gradient[0, read_index])
        neuron_attributions.append(attr)
        neuron_write_norms.append(
            activations.abs() * torch.linalg.vector_norm(mlp_weight, dim=1)
        )
        neuron_projection_biases.append(mlp_bias.detach())

        block_input = block_inputs[layer][0, read_index]
        block_output = block_outputs[layer][0, read_index]
        attention_vector = attention_write[0, read_index]
        mlp_vector = mlp_write[0, read_index]
        reconstructed = block_inputs[layer] + attention_write + mlp_write
        reconstruction_error = _scalar((block_outputs[layer] - reconstructed).abs().max())
        reconstruction_errors.append(reconstruction_error)

        layer_trace.append({
            "layer": layer,
            "residualInputL2": _scalar(torch.linalg.vector_norm(block_input)),
            "attentionResidualWriteL2": _scalar(torch.linalg.vector_norm(attention_vector)),
            "mlpResidualWriteL2": _scalar(torch.linalg.vector_norm(mlp_vector)),
            "residualOutputL2": _scalar(torch.linalg.vector_norm(block_output)),
            "residualAddMaxAbsReconstructionError": reconstruction_error,
            "attentionFirstOrderTargetLogitAttribution": _scalar(torch.dot(attention_vector, attention_gradient[0, read_index])),
            "mlpFirstOrderTargetLogitAttribution": _scalar(torch.dot(mlp_vector, mlp_gradient[0, read_index])),
            "attentionBiasFirstOrderTargetLogitAttribution": _scalar(torch.dot(attention_bias, attention_gradient[0, read_index])),
            "mlpBiasFirstOrderTargetLogitAttribution": _scalar(torch.dot(mlp_bias, mlp_gradient[0, read_index])),
        })

        ranked_heads = torch.argsort(head_attr.abs(), descending=True)[:3]
        for rank, head_tensor in enumerate(ranked_heads, start=1):
            head = int(head_tensor)
            head_write = writes[0, read_index, head]
            head_summary.append({
                "layer": layer,
                "head": head,
                "rankByAbsFirstOrderAttributionInLayer": rank,
                "residualWriteL2": _scalar(torch.linalg.vector_norm(head_write)),
                "firstOrderTargetLogitAttribution": _scalar(head_attr[head]),
            })

    final_names = {
        "token_embedding": "token_embedding",
        "position_embedding": "position_embedding",
        "embedding_output": "embedding_output",
        "final_norm_input": "final_norm_input",
        "final_norm_output": "final_norm_output",
        "lm_head_logits": "lm_head_logits",
    }
    for capture_name, archive_name in final_names.items():
        archive_arrays[f"{case_prefix}_{archive_name}"] = _tensor_to_numpy(capture.values[capture_name][0])

    archive_arrays[f"{case_prefix}_block_input"] = _tensor_to_numpy(torch.stack(block_inputs)[:, 0])
    archive_arrays[f"{case_prefix}_block_output"] = _tensor_to_numpy(torch.stack(block_outputs)[:, 0])
    archive_arrays[f"{case_prefix}_attention_residual_write"] = _tensor_to_numpy(torch.stack(attention_outputs)[:, 0])
    archive_arrays[f"{case_prefix}_attention_context_by_head"] = _tensor_to_numpy(
        torch.stack(attention_context)[:, 0].reshape(layers, int(attention_context[0].shape[1]), head_count, -1)
    )
    archive_arrays[f"{case_prefix}_attention_head_residual_write"] = _tensor_to_numpy(torch.stack(head_writes)[:, 0])
    archive_arrays[f"{case_prefix}_attention_projection_bias"] = _tensor_to_numpy(
        torch.stack([model.transformer.h[layer].attn.c_proj.bias for layer in range(layers)])
    )
    archive_arrays[f"{case_prefix}_mlp_input"] = _tensor_to_numpy(torch.stack(mlp_inputs)[:, 0])
    archive_arrays[f"{case_prefix}_mlp_preactivation"] = _tensor_to_numpy(torch.stack(mlp_pre)[:, 0])
    archive_arrays[f"{case_prefix}_mlp_activation"] = _tensor_to_numpy(torch.stack(mlp_activation)[:, 0])
    archive_arrays[f"{case_prefix}_mlp_residual_write"] = _tensor_to_numpy(torch.stack(mlp_outputs)[:, 0])
    archive_arrays[f"{case_prefix}_mlp_neuron_first_order_target_logit_attribution"] = _tensor_to_numpy(
        torch.stack(neuron_attributions)
    )
    archive_arrays[f"{case_prefix}_mlp_neuron_residual_write_norm"] = _tensor_to_numpy(
        torch.stack(neuron_write_norms)
    )
    archive_arrays[f"{case_prefix}_mlp_projection_bias"] = _tensor_to_numpy(torch.stack(neuron_projection_biases))
    archive_arrays[f"{case_prefix}_attention_head_first_order_target_logit_attribution"] = _tensor_to_numpy(
        torch.stack(head_attributions)
    )
    archive_arrays[f"{case_prefix}_residual_target_logit_gradient"] = _tensor_to_numpy(
        torch.stack([gradients[f"block_output_{layer}"][0, read_index] for layer in range(layers)])
    )
    archive_arrays[f"{case_prefix}_attention_output_target_logit_gradient"] = _tensor_to_numpy(
        torch.stack([gradients[f"attention_output_{layer}"][0, read_index] for layer in range(layers)])
    )
    archive_arrays[f"{case_prefix}_mlp_output_target_logit_gradient"] = _tensor_to_numpy(
        torch.stack([gradients[f"mlp_output_{layer}"][0, read_index] for layer in range(layers)])
    )
    return layer_trace, head_summary


def intervention_pair(
    *,
    label: str,
    mlp_interventions: List[Dict[str, Any]],
    attention_interventions: List[Dict[str, Any]],
    model: Any,
    encoded: Dict[str, torch.Tensor],
    read_index: int,
    target_id: int,
    baseline_summary: Dict[str, Any],
    tokenizer: Any,
    baseline_mlp_activations: Dict[int, torch.Tensor],
    baseline_attention_context: Dict[int, torch.Tensor],
) -> Dict[str, Any]:
    removed = run_intervention(
        model=model,
        encoded=encoded,
        read_index=read_index,
        target_id=target_id,
        baseline_summary=baseline_summary,
        tokenizer=tokenizer,
        mlp_interventions=mlp_interventions,
        attention_interventions=attention_interventions,
        baseline_mlp_activations=baseline_mlp_activations,
        baseline_attention_context=baseline_attention_context,
        mode="remove",
    )
    restored = run_intervention(
        model=model,
        encoded=encoded,
        read_index=read_index,
        target_id=target_id,
        baseline_summary=baseline_summary,
        tokenizer=tokenizer,
        mlp_interventions=mlp_interventions,
        attention_interventions=attention_interventions,
        baseline_mlp_activations=baseline_mlp_activations,
        baseline_attention_context=baseline_attention_context,
        mode="restore",
    )
    return {"label": label, "remove": removed, "restore": restored}


def _top_neurons(
    model: Any,
    capture: GPT2TraceCapture,
    gradients: Dict[str, torch.Tensor],
    layer: int,
    read_index: int,
    count: int = 12,
) -> List[Dict[str, Any]]:
    activation = capture.values[f"mlp_activation_{layer}"][0, read_index]
    preactivation = capture.values[f"mlp_preactivation_{layer}"][0, read_index]
    weight = model.transformer.h[layer].mlp.c_proj.weight
    attribution = mlp_neuron_first_order_attribution(
        activation, weight, gradients[f"mlp_output_{layer}"][0, read_index]
    )
    write_norm = activation.abs() * torch.linalg.vector_norm(weight, dim=1)
    indices = torch.argsort(attribution.abs(), descending=True)[:count]
    return [
        {
            "neuron": int(index),
            "activation": _scalar(activation[index]),
            "preactivation": _scalar(preactivation[index]),
            "residualWriteNorm": _scalar(write_norm[index]),
            "firstOrderTargetLogitAttribution": _scalar(attribution[index]),
        }
        for index in indices
    ]


def explain_case(case: Dict[str, Any]) -> str:
    target = case["baseline"]["target"]
    greedy = case["baseline"]["greedyNextToken"]
    path = case["sourcePath"]
    attention = path["attention"][0]
    mlp = path["mlp"][0]
    effects = case["causalInterventions"]
    attention_effect = effects["selectedAttention"]["remove"]
    mlp_effect = effects["selectedMlp"]["remove"]
    joint_effect = effects["selectedPath"]["remove"]
    restored = effects["selectedPath"]["restore"]
    random_head_control = effects["sourceControls"][0]
    random_neuron_control = effects["sourceControls"][1]
    control_attention = random_head_control["remove"]
    control_mlp = random_neuron_control["remove"]
    target_is_greedy = target["tokenId"] == greedy["tokenId"]
    attention_control_relation = (
        "exceeded"
        if joint_effect["targetLogitDrop"] > control_attention["targetLogitDrop"]
        else "did not exceed"
    )
    neuron_control_relation = (
        "exceeded"
        if control_mlp["targetLogitDrop"] > joint_effect["targetLogitDrop"]
        else "did not exceed"
    )
    neighbor_effects = ", ".join(
        f"{item['label']}: {item['remove']['targetLogitDrop']:.6f}"
        for item in effects["neighboringLayers"]
    )
    lines = [
        f"### {case['subject']}: {case['prompt']}",
        "",
        f"Target token: `{target['token']}` (id {target['tokenId']}); baseline rank {target['rank']} "
        f"with logit {target['logit']:.6f} and probability {target['probability']:.8g}. "
        f"Greedy next token is `{greedy['token']}` (id {greedy['tokenId']}, "
        f"probability {greedy['probability']:.8g}); the target "
        f"{'is' if target_is_greedy else 'is not'} the greedy next token.",
        "",
        "**WHAT?**",
        f"The source artifact selects attention head {attention['head']} at layer {attention['layer']} "
        f"and MLP neuron {mlp['neuron']} at layer {mlp['layer']}. The selected MLP's measured "
        f"activation at the read position is {case['selectedMlpActivation']:.6f}; its local "
        f"first-order target-logit attribution is {case['selectedMlpFirstOrderAttribution']:.6f}.",
        "",
        "**HOW?**",
        f"The input tokenizes to {len(case['tokenization']['tokenIds'])} tokens. The recorded trace has "
        f"{len(case['layerTrace'])} transformer layers. At the selected attention site, the "
        f"head residual-write norm is {case['selectedHeadResidualWriteL2']:.6f}; at the selected "
        f"MLP site, the selected neuron's projected residual-contribution norm is "
        f"{case['selectedMlpNeuronResidualWriteContributionL2']:.6f}. "
        f"Each layer's residual addition was reconstructed from its block input, attention write, "
        f"and MLP write with maximum absolute error {case['maxResidualAddReconstructionError']:.3g}.",
        "",
        "**WHY?**",
        f"Removing the selected attention component changed the target logit by "
        f"{attention_effect['targetLogitDrop']:.6f}; removing the selected MLP neuron changed it by "
        f"{mlp_effect['targetLogitDrop']:.6f}; removing both changed it by "
        f"{joint_effect['targetLogitDrop']:.6f} and changed target probability by "
        f"{joint_effect['targetProbabilityChange']:.8g}. Restoring the original selected "
        f"activations returned the target logit to within "
        f"{abs(restored['targetLogit'] - target['logit']):.3g} of baseline. The selected path "
        f"{attention_control_relation} its source-recorded "
        f"{random_head_control['label']} control "
        f"({control_attention['targetLogitDrop']:.6f}). The source-recorded "
        f"{random_neuron_control['label']} control changed the target logit by "
        f"{control_mlp['targetLogitDrop']:.6f}; it {neuron_control_relation} the selected path "
        f"effect. Neighbor-layer removal effects were {neighbor_effects}. These comparisons "
        f"measure this prompt's interventions; they do not assign semantic labels or establish "
        f"a unique complete circuit.",
        "",
        "**LIMITATIONS?**",
        "This is an instrumented, model-specific trace using local GPT-2 weights and hooks, not "
        "black-box decompilation. First-order neuron/head attributions are local gradient "
        "approximations; the reported ablations are the causal evidence. The trace does not "
        "explain all 12 layers' learned semantics, all token-to-token attention probabilities, "
        "or the full model's behavior outside these five prompts. "
        + (
            "The target is not the greedy next token in this case, so the intervention supports "
            "a claim about its scored logit, not that it was the token actually emitted."
            if not target_is_greedy
            else "The target is the greedy next token for this prompt, but this five-case trace "
            "does not establish general factual behavior."
        ),
        "",
    ]
    return "\n".join(lines)


def run_case(
    *,
    source_case: Dict[str, Any],
    case_index: int,
    model: Any,
    tokenizer: Any,
    archive_arrays: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    prompt = str(source_case["prompt"])
    encoded_raw = tokenizer(
        prompt,
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    offsets = encoded_raw.pop("offset_mapping")[0].tolist()
    encoded = {name: value.to(next(model.parameters()).device) for name, value in encoded_raw.items()}
    token_ids = encoded["input_ids"][0]
    read_index = int(source_case["readTokenIndex"])
    if read_index != token_ids.numel() - 1:
        raise ValueError(f"{source_case['subject']}: source read-token index differs from the original prompt.")
    target_id = int(source_case["target"]["tokenId"])
    target_token = tokenizer.decode([target_id]).replace("\n", "\\n")
    if target_token != source_case["target"]["token"]:
        raise ValueError(f"{source_case['subject']}: source target token text does not match tokenizer id.")
    attention_nodes, mlp_nodes = selected_path_nodes(source_case)
    for node in attention_nodes + mlp_nodes:
        for position in node.get("ablateTokenPositions", []):
            if int(position) < 0 or int(position) >= token_ids.numel():
                raise ValueError(f"{source_case['subject']}: source path position is outside the prompt.")

    prefix = f"case_{case_index}_{source_case['subject'].lower()}"
    position_ids = list(range(int(token_ids.numel())))
    baseline_started = time.perf_counter()
    model.zero_grad(set_to_none=True)
    with GPT2TraceCapture(model) as capture:
        outputs = model(**encoded, use_cache=False)
        logits = outputs.logits[0, read_index]
        target_logit = logits[target_id]
        gradient_refs = capture.gradient_outputs + [
            (f"block_output_{layer}", capture.values["block_output"][layer])
            for layer in range(capture.layers)
        ]
        gradient_values = torch.autograd.grad(
            target_logit,
            [tensor for _, tensor in gradient_refs],
            retain_graph=False,
            allow_unused=False,
        )
        gradients = {
            name: value.detach()
            for (name, _tensor), value in zip(gradient_refs, gradient_values)
        }
        baseline_summary = summarize_logits(logits, target_id, tokenizer)
        baseline_attention_context = _ordered_layer_tensors(capture, "attention_context")
        baseline_mlp_activations = {
            layer: capture.values[f"mlp_activation_{layer}"] for layer in range(capture.layers)
        }

        layer_trace, head_summary = raw_layer_trace(
            capture=capture,
            model=model,
            read_index=read_index,
            target_id=target_id,
            gradients=gradients,
            archive_arrays=archive_arrays,
            case_prefix=prefix,
        )
        archive_arrays[f"{prefix}_token_ids"] = _tensor_to_numpy(token_ids)
        archive_arrays[f"{prefix}_position_ids"] = np.asarray(position_ids, dtype=np.int64)
        archive_arrays[f"{prefix}_input_offsets"] = np.asarray(offsets, dtype=np.int64)
        archive_arrays[f"{prefix}_target_logits"] = _tensor_to_numpy(logits)
        baseline = {**baseline_summary, "allLogitsArchiveKey": f"{prefix}_target_logits"}

        selected_mlp_site = mlp_nodes[0]
        selected_attention_site = attention_nodes[0]
        selected_mlp_layer = int(selected_mlp_site["layer"])
        selected_mlp_neuron = int(selected_mlp_site["neuron"])
        selected_attention_layer = int(selected_attention_site["layer"])
        selected_attention_head = int(selected_attention_site["head"])
        mlp_activation = baseline_mlp_activations[selected_mlp_layer][
            0, int(selected_mlp_site["ablateTokenPositions"][0]), selected_mlp_neuron
        ]
        mlp_gradient = gradients[f"mlp_output_{selected_mlp_layer}"][0, read_index]
        mlp_weight = model.transformer.h[selected_mlp_layer].mlp.c_proj.weight
        mlp_attributions = mlp_neuron_first_order_attribution(
            capture.values[f"mlp_activation_{selected_mlp_layer}"][0, read_index],
            mlp_weight,
            mlp_gradient,
        )
        selected_mlp_attribution = _scalar(mlp_attributions[selected_mlp_neuron])
        attention_writes = conv1d_head_writes(
            baseline_attention_context[selected_attention_layer],
            model.transformer.h[selected_attention_layer].attn.c_proj.weight,
            int(model.config.n_head),
        )
        selected_head_write = attention_writes[
            0,
            int(selected_attention_site["ablateTokenPositions"][0]),
            selected_attention_head,
        ]
        selected_head_write_l2 = _scalar(torch.linalg.vector_norm(selected_head_write))
        selected_mlp_write_l2 = _scalar(
            mlp_activation.abs()
            * torch.linalg.vector_norm(mlp_weight[selected_mlp_neuron])
        )
        selected_head_attribution = _scalar(
            gradients[f"attention_output_{selected_attention_layer}"][0, read_index].dot(
                attention_writes[0, read_index, selected_attention_head]
            )
        )
        selected_neuron_records = _top_neurons(
            model, capture, gradients, selected_mlp_layer, read_index
        )

        source_controls = choose_recorded_controls(source_case)
        selected_attention = intervention_pair(
            label="selected-attention-component",
            mlp_interventions=[],
            attention_interventions=attention_nodes,
            model=model,
            encoded=encoded,
            read_index=read_index,
            target_id=target_id,
            baseline_summary=baseline,
            tokenizer=tokenizer,
            baseline_mlp_activations=baseline_mlp_activations,
            baseline_attention_context=baseline_attention_context,
        )
        selected_mlp = intervention_pair(
            label="selected-mlp-component",
            mlp_interventions=mlp_nodes,
            attention_interventions=[],
            model=model,
            encoded=encoded,
            read_index=read_index,
            target_id=target_id,
            baseline_summary=baseline,
            tokenizer=tokenizer,
            baseline_mlp_activations=baseline_mlp_activations,
            baseline_attention_context=baseline_attention_context,
        )
        selected_path = intervention_pair(
            label="selected-ordered-path",
            mlp_interventions=mlp_nodes,
            attention_interventions=attention_nodes,
            model=model,
            encoded=encoded,
            read_index=read_index,
            target_id=target_id,
            baseline_summary=baseline,
            tokenizer=tokenizer,
            baseline_mlp_activations=baseline_mlp_activations,
            baseline_attention_context=baseline_attention_context,
        )
        control_pairs = []
        for control in source_controls:
            control_pairs.append(intervention_pair(
                label=control["kind"],
                mlp_interventions=control["mlpInterventions"],
                attention_interventions=control["attentionInterventions"],
                model=model,
                encoded=encoded,
                read_index=read_index,
                target_id=target_id,
                baseline_summary=baseline,
                tokenizer=tokenizer,
                baseline_mlp_activations=baseline_mlp_activations,
                baseline_attention_context=baseline_attention_context,
            ))

        neighboring = []
        for offset in NEIGHBOR_LAYER_OFFSETS:
            attention_layer = selected_attention_layer + offset
            mlp_layer = selected_mlp_layer + offset
            if 0 <= attention_layer < capture.layers:
                neighboring.append(intervention_pair(
                    label=f"attention-head-{selected_attention_head}-layer-{attention_layer}",
                    mlp_interventions=[],
                    attention_interventions=[{
                        **selected_attention_site,
                        "layer": attention_layer,
                    }],
                    model=model,
                    encoded=encoded,
                    read_index=read_index,
                    target_id=target_id,
                    baseline_summary=baseline,
                    tokenizer=tokenizer,
                    baseline_mlp_activations=baseline_mlp_activations,
                    baseline_attention_context=baseline_attention_context,
                ))
            if 0 <= mlp_layer < capture.layers:
                neighboring.append(intervention_pair(
                    label=f"mlp-neuron-{selected_mlp_neuron}-layer-{mlp_layer}",
                    mlp_interventions=[{**selected_mlp_site, "layer": mlp_layer}],
                    attention_interventions=[],
                    model=model,
                    encoded=encoded,
                    read_index=read_index,
                    target_id=target_id,
                    baseline_summary=baseline,
                    tokenizer=tokenizer,
                    baseline_mlp_activations=baseline_mlp_activations,
                    baseline_attention_context=baseline_attention_context,
                ))

        baseline_elapsed = time.perf_counter() - baseline_started
        case = {
            "subject": source_case["subject"],
            "answer": source_case["answer"],
            "prompt": prompt,
            "promptSha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "tokenization": {
                "tokenIds": [int(token) for token in token_ids.detach().cpu().tolist()],
                "tokens": [tokenizer.convert_ids_to_tokens(int(token)) for token in token_ids.detach().cpu().tolist()],
                "decodedTokens": [
                    tokenizer.decode([int(token)]).replace("\n", "\\n")
                    for token in token_ids.detach().cpu().tolist()
                ],
                "characterOffsets": [[int(start), int(end)] for start, end in offsets],
                "readTokenIndex": read_index,
                "readToken": tokenizer.decode([int(token_ids[read_index])]).replace("\n", "\\n"),
                "tokenEmbeddingArchiveKey": f"{prefix}_token_embedding",
                "positionEmbeddingArchiveKey": f"{prefix}_position_embedding",
                "embeddingOutputArchiveKey": f"{prefix}_embedding_output",
            },
            "target": {
                "tokenId": target_id,
                "token": target_token,
                "sourceTargetRank": source_case["target"].get("baselineRank"),
            },
            "sourcePath": {
                "attention": attention_nodes,
                "mlp": mlp_nodes,
                "sourceControls": source_controls,
            },
            "baseline": baseline,
            "layerTrace": layer_trace,
            "attentionHeadContributions": head_summary,
            "selectedMlpNeuronAttributions": selected_neuron_records,
            "selectedHeadResidualWriteL2": selected_head_write_l2,
            "selectedHeadFirstOrderAttribution": selected_head_attribution,
            "selectedMlpActivation": _scalar(mlp_activation),
            "selectedMlpNeuronResidualWriteContributionL2": float(selected_mlp_write_l2),
            "selectedMlpFirstOrderAttribution": selected_mlp_attribution,
            "maxResidualAddReconstructionError": max(
                row["residualAddMaxAbsReconstructionError"] for row in layer_trace
            ),
            "causalInterventions": {
                "selectedAttention": selected_attention,
                "selectedMlp": selected_mlp,
                "selectedPath": selected_path,
                "neighboringLayers": neighboring,
                "sourceControls": control_pairs,
            },
            "timingSeconds": {"traceAndAllCausalInterventions": baseline_elapsed},
            "rawArchivePrefix": prefix,
        }
        case["explanationMarkdown"] = explain_case(case)
        return case


def write_markdown(report: Dict[str, Any], path: Path) -> None:
    comparison_rows = []
    for case in report["cases"]:
        attention = case["sourcePath"]["attention"][0]
        mlp = case["sourcePath"]["mlp"][0]
        comparison_rows.append(
            f"| {case['subject']} | `{case['prompt']}` | `{case['baseline']['target']['token']}` "
            f"(rank {case['baseline']['target']['rank']}) | "
            f"L{attention['layer']} H{attention['head']} | "
            f"L{mlp['layer']} N{mlp['neuron']} | "
            f"{case['baseline']['target']['probability']:.6g} | "
            f"{case['causalInterventions']['selectedPath']['remove']['targetLogitDrop']:.6f} |"
        )
    lines = [
        "# GPT-2 internal computation trace",
        "",
        f"- Model: `{report['model']['id']}` (`{report['model']['class']}`)",
        f"- Device: `{report['runtime']['device']}`",
        f"- Source artifact SHA-256: `{report['sourceArtifact']['sha256']}`",
        f"- Overall instrumentation status: `{report['capabilityAssessment']['internalAccess']}`",
        f"- Causal validation: `{report['capabilityAssessment']['causalValidation']}`",
        f"- Explanation scope: `{report['capabilityAssessment']['scope']}`",
        "",
        "This is an instrumented witness run using the local model and hooks. It is not a black-box decompilation claim.",
        "",
        "## Factual-input comparison",
        "",
        "| Subject | Exact prompt | Target/rank | Selected attention | Selected MLP | Target probability | Joint removal logit drop |",
        "|---|---|---|---|---|---:|---:|",
        *comparison_rows,
        "",
        "All five cases use the same capture and intervention procedure. Different subject prompts select different MLP neurons and sometimes different attention layers; this is an observed difference in these source paths, not a semantic interpretation of those units.",
        "",
        "## Per-case explanations",
        "",
    ]
    for case in report["cases"]:
        lines.append(case["explanationMarkdown"])
    lines.extend([
        "## Trace fields captured",
        "",
        "Raw arrays are stored in the adjacent compressed NPZ archive. The archive contains token and position embeddings, every block input/output residual, attention projection contexts and per-head residual writes, MLP inputs/preactivations/activations/residual writes, final normalization, full vocabulary logits, and first-order attribution arrays.",
        "",
        "Attention head and MLP neuron first-order values are gradient-times-write local approximations. They are reported separately from causal ablation results.",
        "",
        "## Claim boundary",
        "",
        report["claimBoundary"]["doesNotEstablish"],
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_autoregressive_markdown(report: Dict[str, Any], path: Path) -> None:
    lines = [
        "# GPT-2 autoregressive causal computation trace",
        "",
        f"- Protocol: `{report['protocol']}`",
        f"- Access mode: `{report['accessMode']}`",
        f"- Model: `{report['model']['id']}` (`{report['model']['class']}`), device `{report['runtime']['device']}`",
        f"- Source artifact SHA-256: `{report['sourceArtifact']['sha256']}`",
        f"- Passed source cases: {report['summary']['caseCount']}",
        f"- Generated prediction steps: {report['summary']['generatedPredictionCount']}",
        f"- Deterministic matched controls per prediction: {report['protocolConfig']['controlCount']}",
        f"- Maximum generation length: {report['protocolConfig']['maxNewTokens']}",
        "",
        "The source paths and prompts are read-only inputs from the existing ordered-path artifact. ",
        "Each causal test is a one-step intervention on the unchanged baseline-generated context; ",
        "intervened tokens are not fed into later baseline contexts. Teacher-forced source-token scores ",
        "are reported separately from free-running predictions.",
        "",
        "## Per-case acceptance criteria",
        "",
        "| Subject | A trace | B greedy reconstruction | C influence | D matched specificity | E interaction measured | F actual greedy explained | G provenance | Sufficiency | Greedy tokens |",
        "|---|---|---|---|---|---|---|---|---|---:|",
    ]
    criterion_keys = (
        "A_internalComputationalTrace",
        "B_correctTokenByTokenReconstruction",
        "C_causalInfluence",
        "D_matchedControlSpecificity",
        "E_interactionAndRedundancyMeasurement",
        "F_actualGreedyPredictionExplanation",
        "G_sourceCaseReproducibility",
    )
    for case in report["cases"]:
        statuses = [case["criteria"][key]["status"] for key in criterion_keys]
        sufficiency = case["criteria"]["causalSufficiency"]["status"]
        generated = " ".join(
            step["generatedToken"]["token"].replace("|", "\\|")
            for step in case["generation"]["steps"]
        ).strip()
        lines.append(
            f"| {case['subject']} | {' | '.join(statuses)} | {sufficiency} | "
            f"{generated} |"
        )
        lines.append("")
        lines.append(case["explanationMarkdown"])
        lines.append("")
    lines.extend([
        "## Protocol and claim limits",
        "",
        report["claimBoundary"]["doesNotEstablish"],
        "",
        "## Artifacts",
        "",
        "- `report.json`: every case, step, intervention, control draw, and explanation.",
        "- `raw-tensors.npz`: read-position transformer traces and full-vocabulary baseline/selected-path/individual logits.",
        "- `manifest.json`: wrapper provenance and generated variant metadata.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def run_autoregressive_experiment(
    *,
    args: argparse.Namespace,
    source_path: Path,
    source_report: Dict[str, Any],
    source_cases: List[Dict[str, Any]],
    source_sha256: str,
    output_dir: Path,
    model: Any,
    tokenizer: Any,
    model_load_seconds: float,
    started: float,
) -> Dict[str, Any]:
    archive_arrays: Dict[str, np.ndarray] = {}
    trace_started = time.perf_counter()
    cases = [
        run_autoregressive_case(
            source_case=source_case,
            case_index=index,
            model=model,
            tokenizer=tokenizer,
            archive_arrays=archive_arrays,
            source_sha256=source_sha256,
            max_new_tokens=args.max_new_tokens,
            control_count=args.control_count,
            magnitude_tolerance=args.control_magnitude_tolerance,
        )
        for index, source_case in enumerate(source_cases)
    ]
    if next(model.parameters()).device.type == "cuda":
        torch.cuda.synchronize()
    trace_seconds = time.perf_counter() - trace_started
    archive_path = output_dir / "raw-tensors.npz"
    np.savez_compressed(archive_path, **archive_arrays)
    archive_meta = {
        "file": archive_path.name,
        "sha256": sha256_file(archive_path),
        "arrays": {
            name: {"shape": list(array.shape), "dtype": str(array.dtype)}
            for name, array in sorted(archive_arrays.items())
        },
    }
    criteria_keys = (
        "A_internalComputationalTrace",
        "B_correctTokenByTokenReconstruction",
        "C_causalInfluence",
        "D_matchedControlSpecificity",
        "E_interactionAndRedundancyMeasurement",
        "F_actualGreedyPredictionExplanation",
        "G_sourceCaseReproducibility",
    )
    criteria_summary = {
        key: {
            "pass": sum(case["criteria"][key]["status"] == "pass" for case in cases),
            "fail": sum(case["criteria"][key]["status"] == "fail" for case in cases),
        }
        for key in criteria_keys
    }
    total_predictions = sum(len(case["generation"]["steps"]) for case in cases)
    properties = (
        torch.cuda.get_device_properties(torch.cuda.current_device())
        if next(model.parameters()).device.type == "cuda"
        else None
    )
    gpu_info = None if properties is None else {
        "name": properties.name,
        "totalMemoryBytes": int(properties.total_memory),
        "peakAllocatedBytes": int(torch.cuda.max_memory_allocated()),
        "peakReservedBytes": int(torch.cuda.max_memory_reserved()),
    }
    report = {
        "schemaVersion": 1,
        "kind": "gpt2-autoregressive-causal-computation-trace",
        "synthetic": False,
        "surrogate": False,
        "accessMode": "instrumented-witness",
        "protocol": args.protocol,
        "generatedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "sourceArtifact": {
            "path": str(source_path.relative_to(REPOSITORY_ROOT)),
            "sha256": source_sha256,
            "sourceGeneratedAt": source_report.get("generatedAt"),
            "originalConfig": source_report.get("config"),
            "selectedSubjects": [case["subject"] for case in source_cases],
            "passedSourceCaseCount": len(source_cases),
        },
        "model": {
            "id": args.model,
            "class": model.__class__.__name__,
            "device": str(next(model.parameters()).device),
            "localFilesOnly": True,
            "config": model.config.to_dict(),
            "repositoryCommitHash": getattr(model.config, "_commit_hash", None),
            "tokenizerClass": tokenizer.__class__.__name__,
            "tokenizerNameOrPath": getattr(tokenizer, "name_or_path", None),
        },
        "runtime": {
            "device": str(next(model.parameters()).device),
            "gpu": gpu_info,
            "pythonVersion": sys.version.split()[0],
            "numpyVersion": np.__version__,
            "torchVersion": torch.__version__,
            "transformersVersion": package_version("transformers"),
            "modelLoadSeconds": model_load_seconds,
            "traceAndInterventionSeconds": trace_seconds,
            "totalSeconds": time.perf_counter() - started,
            "cudaWasAvailable": bool(torch.cuda.is_available()),
        },
        "protocolConfig": {
            "caseSet": args.case_set,
            "maxNewTokens": args.max_new_tokens,
            "controlCount": args.control_count,
            "controlMagnitudeRelativeTolerance": args.control_magnitude_tolerance,
            "specificityPassThresholdPercentile": 95.0,
            "specificityMinimumMagnitudeMatchedControls": math.ceil(args.control_count * 0.95),
            "specificityMinimumUniqueControlGroups": 20,
            "randomSeedDerivation": "first 64 bits SHA256(sourceArtifactSha256:subject:stepIndex:contextTokenIds)",
            "controlInterventionBatchSize": 32,
            "interventionOperator": "zero MLP activation units and attention c_proj input slices at the frozen source token positions",
            "generation": "greedy autoregressive; recompute the full context at each prediction; stop at EOS or maxNewTokens",
            "teacherForcedAndFreeRunningSeparated": True,
            "intervenedTokensNeverFedToLaterBaselineContexts": True,
            "fullDistributionMetrics": ["totalVariation", "l1Probability", "jensenShannonDivergence", "maxAbsoluteProbabilityChange"],
        },
        "rawTensorArchive": archive_meta,
        "summary": {
            "caseCount": len(cases),
            "generatedPredictionCount": total_predictions,
            "criteria": criteria_summary,
            "allSourceCasesPresent": [case["subject"] for case in cases] == [case["subject"] for case in source_cases],
            "maxResidualAddReconstructionError": max(
                step["trace"]["maxResidualAddReconstructionError"]
                for case in cases
                for step in case["generation"]["steps"]
            ),
            "maxPathRestoreLogitError": max(
                step["causal"]["selectedPath"]["restoreMaxLogitError"]
                for case in cases
                for step in case["generation"]["steps"]
            ),
        },
        "capabilityAssessment": {
            "internalTrace": "read-position-quantitative-all-layers",
            "tokenByToken": "greedy-context-reconstruction-with-full-prefix-recorded",
            "causalInfluence": "selected-path-individual-group-removal-and-restoration",
            "specificity": "100-deterministic-structurally-matched-control-groups-per-generated-prediction",
            "sufficiency": "not-tested",
            "scope": f"{len(cases)} frozen passed source cases, exact original prompts and paths",
        },
        "claimBoundary": {
            "proves": "For the recorded local GPT-2 model, this run records every unmodified greedy autoregressive next-token context, layerwise read-position internals, and one-step selected-path/component/control interventions on each unchanged baseline context.",
            "doesNotEstablish": "This instrumented hook-based experiment is not black-box decompilation or a complete semantic circuit. Causal influence is distinct from matched-control specificity; path sufficiency is not tested. If an intervention changes a token, later steps remain conditioned on the unmodified baseline-generated context. No semantic meanings are assigned to individual units.",
        },
        "cases": cases,
    }
    report_json = output_dir / "report.json"
    report_md = output_dir / "report.md"
    report_json.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_autoregressive_markdown(report, report_md)
    return report


def main() -> None:
    args = parse_args()
    if args.max_new_tokens <= 0:
        raise SystemExit("--max-new-tokens must be positive.")
    if args.control_count <= 0:
        raise SystemExit("--control-count must be positive.")
    if args.control_magnitude_tolerance < 0:
        raise SystemExit("--control-magnitude-tolerance cannot be negative.")
    if args.protocol == "legacy" and args.case_set != "regression-five":
        raise SystemExit("The legacy protocol is reserved for --case-set regression-five.")
    source_path = args.source_report.resolve()
    output_dir = args.output_dir.resolve()
    if not source_path.is_file():
        raise SystemExit(f"Source report does not exist: {source_path}")
    source_report = json.loads(source_path.read_text(encoding="utf-8"))
    if args.case_set == "all-50":
        source_cases = all_passed_source_cases(source_report)
    else:
        source_cases = ordered_source_cases(source_report)
    if args.case_set == "all-50" and len(source_cases) != 50:
        raise SystemExit(f"Expected all 50 passed source cases, found {len(source_cases)}.")
    source_sha256 = sha256_file(source_path)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable.")
    device = "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"

    started = time.perf_counter()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True)
    model.eval()
    model.to(device)
    if model.__class__.__name__ != "GPT2LMHeadModel":
        raise RuntimeError(f"Expected GPT2LMHeadModel, got {model.__class__.__name__}.")
    if not hasattr(model, "transformer") or not hasattr(model.transformer, "h"):
        raise RuntimeError("Loaded model does not expose GPT-2 transformer blocks.")
    if device == "cuda":
        torch.cuda.synchronize()
    model_load_seconds = time.perf_counter() - started

    output_dir.mkdir(parents=True, exist_ok=True)
    if args.protocol == "autoregressive":
        report = run_autoregressive_experiment(
            args=args,
            source_path=source_path,
            source_report=source_report,
            source_cases=source_cases,
            source_sha256=source_sha256,
            output_dir=output_dir,
            model=model,
            tokenizer=tokenizer,
            model_load_seconds=model_load_seconds,
            started=started,
        )
        print(json.dumps({
            "marker": "GPT2_AUTOREGRESSIVE_CAUSAL_TRACE_EXPERIMENT_WRITTEN",
            "caseCount": report["summary"]["caseCount"],
            "generatedPredictionCount": report["summary"]["generatedPredictionCount"],
            "device": report["runtime"]["device"],
            "report": str(output_dir / "report.json"),
            "rawTensors": str(output_dir / "raw-tensors.npz"),
        }))
        return

    archive_arrays: Dict[str, np.ndarray] = {}
    case_started = time.perf_counter()
    cases = [
        run_case(
            source_case=source_case,
            case_index=index,
            model=model,
            tokenizer=tokenizer,
            archive_arrays=archive_arrays,
        )
        for index, source_case in enumerate(source_cases)
    ]
    if device == "cuda":
        torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - case_started
    archive_path = output_dir / "raw-tensors.npz"
    np.savez_compressed(archive_path, **archive_arrays)
    archive_meta = {
        "file": archive_path.name,
        "sha256": sha256_file(archive_path),
        "arrays": {
            name: {"shape": list(array.shape), "dtype": str(array.dtype)}
            for name, array in sorted(archive_arrays.items())
        },
    }

    max_restore_error = max(
        abs(
            pair["restore"]["targetLogit"]
            - case["baseline"]["target"]["logit"]
        )
        for case in cases
        for pair in (
            case["causalInterventions"]["selectedAttention"],
            case["causalInterventions"]["selectedMlp"],
            case["causalInterventions"]["selectedPath"],
        )
    )
    target_is_greedy = [
        case["baseline"]["target"]["tokenId"] == case["baseline"]["greedyNextToken"]["tokenId"]
        for case in cases
    ]
    model_config = model.config.to_dict()
    gpu_info = None
    if device == "cuda":
        properties = torch.cuda.get_device_properties(torch.cuda.current_device())
        gpu_info = {
            "name": properties.name,
            "totalMemoryBytes": int(properties.total_memory),
            "peakAllocatedBytes": int(torch.cuda.max_memory_allocated()),
            "peakReservedBytes": int(torch.cuda.max_memory_reserved()),
        }
    total_runtime_seconds = time.perf_counter() - started
    report = {
        "schemaVersion": 1,
        "kind": "gpt2-instrumented-internal-computation-trace",
        "synthetic": False,
        "surrogate": False,
        "accessMode": "instrumented-witness",
        "generatedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "sourceArtifact": {
            "path": str(source_path.relative_to(REPOSITORY_ROOT)),
            "sha256": sha256_file(source_path),
            "selectedSubjects": list(REQUIRED_SUBJECTS),
            "originalConfig": source_report.get("config"),
            "originalGeneratedAt": source_report.get("generatedAt"),
        },
        "model": {
            "id": args.model,
            "class": model.__class__.__name__,
            "device": device,
            "localFilesOnly": True,
            "config": model_config,
            "repositoryCommitHash": getattr(model.config, "_commit_hash", None),
            "tokenizerClass": tokenizer.__class__.__name__,
            "tokenizerNameOrPath": getattr(tokenizer, "name_or_path", None),
        },
        "runtime": {
            "device": device,
            "gpu": gpu_info,
            "pythonVersion": sys.version.split()[0],
            "numpyVersion": np.__version__,
            "transformersVersion": package_version("transformers"),
            "modelLoadSeconds": model_load_seconds,
            "fiveCaseTraceAndInterventionsSeconds": inference_seconds,
            "totalSeconds": total_runtime_seconds,
            "cudaWasAvailable": bool(torch.cuda.is_available()),
            "torchVersion": torch.__version__,
        },
        "rawTensorArchive": archive_meta,
        "summary": {
            "caseCount": len(cases),
            "allFiveRequiredSubjectsPresent": [case["subject"] for case in cases] == list(REQUIRED_SUBJECTS),
            "targetWasGreedyNextTokenCases": sum(target_is_greedy),
            "targetNotGreedyNextTokenCases": len(cases) - sum(target_is_greedy),
            "maxResidualAddReconstructionError": max(
                case["maxResidualAddReconstructionError"] for case in cases
            ),
            "maxRestoreTargetLogitError": max_restore_error,
            "selectedPathInterventionEffects": [
                {
                    "subject": case["subject"],
                    "targetToken": case["baseline"]["target"]["token"],
                    "targetLogitDrop": case["causalInterventions"]["selectedPath"]["remove"]["targetLogitDrop"],
                    "targetProbabilityChange": case["causalInterventions"]["selectedPath"]["remove"]["targetProbabilityChange"],
                }
                for case in cases
            ],
        },
        "capabilityAssessment": {
            "internalAccess": "measured" if len(cases) == len(REQUIRED_SUBJECTS) else "incomplete",
            "computationalTrace": "layerwise-quantitative",
            "causalValidation": "measured-interventions-with-restoration-neighbor-and-source-controls",
            "explanatoryOutput": "five-evidence-grounded-case-explanations",
            "reproducibility": "source-hash-prompts-token-ids-config-raw-tensors-and-intervention-outputs-recorded",
            "scope": "five configured GPT-2 capital-fact prompts only",
        },
        "captureCoverage": {
            "tokenAndPositionEmbeddings": True,
            "residualStreamInputAndOutputEveryLayer": True,
            "attentionBlockOutputEveryLayer": True,
            "attentionHeadProjectedResidualWritesEveryLayer": True,
            "attentionProbabilityMatrices": False,
            "mlpInputPreactivationAndActivationEveryLayer": True,
            "mlpNeuronFirstOrderContributionsAtReadPosition": True,
            "mlpNeuronCausalAblations": "selected-path-and-recorded-control-components",
            "finalNormalizationAndFullUnembeddingLogits": True,
            "targetAndTopCompetingTokens": True,
        },
        "claimBoundary": {
            "proves": "For these five exact prompts on the recorded local GPT-2 model, GCL captured actual internal tensors and produced quantitative layerwise traces; recorded component removals and restorations measure causal changes to target-token scores.",
            "doesNotEstablish": "This is not a black-box solver result, complete LLM decompilation, semantic interpretation of units, a complete circuit, exhaustive attention/MLP causal decomposition, generalization beyond these five facts, or proof that the factual target was emitted when it was not the greedy next token.",
        },
        "cases": cases,
    }
    report_json = output_dir / "report.json"
    report_md = output_dir / "report.md"
    report_json.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_markdown(report, report_md)
    print(json.dumps({
        "marker": "GPT2_INTERNAL_TRACE_EXPERIMENT_WRITTEN",
        "report": str(report_json),
        "rawTensors": str(archive_path),
        "caseCount": len(cases),
        "device": device,
    }))


if __name__ == "__main__":
    main()
