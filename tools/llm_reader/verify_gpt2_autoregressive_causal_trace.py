#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
SOURCE_REPORT = (
    ROOT
    / "benchmarks/release/nn-live-gpt2-ordered-residual-path-circuit-finder"
    / "gpt2-50-facts-ordered-residual-path/report.json"
)
PRIOR_FIVE_DIR = ROOT / "benchmarks/release/nn-live-gpt2-internal-trace/five-facts"
PRIOR_FIVE_HASHES = {
    "report.json": "0c7198539c1796969c130c2f252d6ea271c97001ffe556afd3f6819854e2c586",
    "report.md": "a651b28742d40a40d69589f35596b695b8e9e7f74a8d7e67a4336605fa8276b2",
    "raw-tensors.npz": "f5d8f89b92d94be32f449181b4233a34359515e92fa09d824d58923d1bd32d37",
    "manifest.json": "fdac01dd330b9f12650a1c24863c04b47947482ba50207fb2d4afdb53612336a",
}
EXPECTED_SOURCE_SHA256 = "f24428b69b9b1fb2d750dfbf2ea55a065fc1d6cc740c415d01ac6ccb94e3715b"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _close(actual: Any, expected: Any, tolerance: float = 3e-4) -> bool:
    return (
        isinstance(actual, (int, float))
        and not isinstance(actual, bool)
        and isinstance(expected, (int, float))
        and not isinstance(expected, bool)
        and math.isfinite(float(actual))
        and math.isfinite(float(expected))
        and math.isclose(float(actual), float(expected), rel_tol=tolerance, abs_tol=tolerance)
    )


def _distribution_metrics(baseline_logits: np.ndarray, changed_logits: np.ndarray) -> Dict[str, float]:
    base = torch.softmax(torch.as_tensor(baseline_logits, dtype=torch.float64), dim=-1)
    changed = torch.softmax(torch.as_tensor(changed_logits, dtype=torch.float64), dim=-1)
    midpoint = (base + changed) * 0.5
    epsilon = torch.finfo(base.dtype).tiny
    delta = (changed - base).abs()
    js = 0.5 * (
        (base * (base.clamp_min(epsilon).log() - midpoint.clamp_min(epsilon).log())).sum()
        + (changed * (changed.clamp_min(epsilon).log() - midpoint.clamp_min(epsilon).log())).sum()
    )
    return {
        "totalVariation": float(delta.sum() * 0.5),
        "l1Probability": float(delta.sum()),
        "jensenShannonDivergence": float(js),
        "maxAbsoluteProbabilityChange": float(delta.max()),
    }


def _source_cases(source_report: Dict[str, Any]) -> List[Dict[str, Any]]:
    cases = [row for row in source_report.get("cases", []) if row.get("status") == "passed"]
    expected = int(source_report.get("config", {}).get("caseCount", 0))
    if len(cases) != expected:
        return []
    return cases


def _report_case_for_subject(case: Dict[str, Any], source_case: Dict[str, Any]) -> List[str]:
    errors = []
    subject = source_case.get("subject", "<unknown>")
    prefix = f"{subject}: "
    if case.get("subject") != subject or case.get("prompt") != source_case.get("prompt"):
        errors.append(prefix + "source subject or exact prompt differs")
    if case.get("answer") != source_case.get("answer"):
        errors.append(prefix + "source answer differs")
    source_nodes = source_case.get("orderedResidualPath", {}).get("nodes", {})
    if case.get("sourcePath", {}).get("attention") != source_nodes.get("attention"):
        errors.append(prefix + "attention path differs from frozen source artifact")
    if case.get("sourcePath", {}).get("mlp") != source_nodes.get("mlp"):
        errors.append(prefix + "MLP path differs from frozen source artifact")
    if hashlib.sha256(str(case.get("prompt", "")).encode()).hexdigest() != case.get("promptSha256"):
        errors.append(prefix + "prompt hash mismatch")
    return errors


def _verify_regression(report_path: Path) -> List[str]:
    errors = []
    sys.path.insert(0, str(ROOT / "tools"))
    from llm_reader.verify_gpt2_internal_trace import verify_report

    errors.extend(verify_report(report_path))
    for name, expected in PRIOR_FIVE_HASHES.items():
        path = PRIOR_FIVE_DIR / name
        if not path.is_file() or _sha256(path) != expected:
            errors.append(f"prior five-case artifact changed: {name}")
    prior = json.loads((PRIOR_FIVE_DIR / "report.json").read_text(encoding="utf-8"))
    current = json.loads(report_path.read_text(encoding="utf-8"))
    if current.get("sourceArtifact", {}).get("sha256") != EXPECTED_SOURCE_SHA256:
        errors.append("regression run used a different frozen source artifact")
    if [case.get("subject") for case in current.get("cases", [])] != [
        "France", "Germany", "Italy", "Portugal", "Greece"
    ]:
        errors.append("regression run does not contain the exact five frozen cases")
    old_by_subject = {case["subject"]: case for case in prior.get("cases", [])}
    for case in current.get("cases", []):
        baseline = old_by_subject.get(case.get("subject"))
        if baseline is None:
            errors.append(f"{case.get('subject')}: no prior baseline result exists")
            continue
        for path, expected, actual in (
            ("prompt", baseline.get("prompt"), case.get("prompt")),
            ("target", baseline.get("target"), case.get("target")),
            (
                "path",
                baseline.get("sourcePath"),
                case.get("sourcePath"),
            ),
        ):
            if expected != actual:
                errors.append(f"{case['subject']}: regression {path} differs from prior report")
        for group, field in (
            ("baseline", "logit"),
            ("baseline", "probability"),
        ):
            if not _close(
                case.get("baseline", {}).get("target", {}).get(field),
                baseline.get("baseline", {}).get("target", {}).get(field),
                tolerance=1e-5,
            ):
                errors.append(f"{case['subject']}: prior baseline target {field} changed")
        for field in ("targetLogitDrop", "targetProbabilityChange"):
            if not _close(
                case.get("causalInterventions", {}).get("selectedPath", {}).get("remove", {}).get(field),
                baseline.get("causalInterventions", {}).get("selectedPath", {}).get("remove", {}).get(field),
                tolerance=1e-5,
            ):
                errors.append(f"{case['subject']}: prior selected-path {field} changed")
    return errors


def _verify_step(
    *,
    case: Dict[str, Any],
    step: Dict[str, Any],
    source_case: Dict[str, Any],
    arrays: Any,
    model: Any,
    tokenizer: Any,
    device: torch.device,
) -> List[str]:
    errors = []
    subject = case["subject"]
    prefix = step.get("rawArchivePrefix")
    if not isinstance(prefix, str):
        return [f"{subject}: step archive prefix is missing"]
    context_ids = [int(value) for value in step.get("contextTokenIds", [])]
    required_shapes = {
        "input_ids": (len(context_ids),),
        "read_token_index": (),
        "token_embedding": (int(model.config.n_embd),),
        "position_embedding": (int(model.config.n_embd),),
        "embedding_output": (int(model.config.n_embd),),
        "residual_input": (int(model.config.n_layer), int(model.config.n_embd)),
        "residual_output": (int(model.config.n_layer), int(model.config.n_embd)),
        "attention_output": (int(model.config.n_layer), int(model.config.n_embd)),
        "attention_context_by_head": (
            int(model.config.n_layer),
            int(model.config.n_head),
            int(model.config.n_embd) // int(model.config.n_head),
        ),
        "attention_head_residual_write": (
            int(model.config.n_layer),
            int(model.config.n_head),
            int(model.config.n_embd),
        ),
        "attention_projection_bias": (int(model.config.n_layer), int(model.config.n_embd)),
        "mlp_input": (int(model.config.n_layer), int(model.config.n_embd)),
        "mlp_preactivation": (int(model.config.n_layer), int(model.config.n_inner or model.config.n_embd * 4)),
        "mlp_activation": (int(model.config.n_layer), int(model.config.n_inner or model.config.n_embd * 4)),
        "mlp_output": (int(model.config.n_layer), int(model.config.n_embd)),
        "mlp_projection_bias": (int(model.config.n_layer), int(model.config.n_embd)),
        "mlp_neuron_first_order_predicted_token_attribution": (
            int(model.config.n_layer),
            int(model.config.n_inner or model.config.n_embd * 4),
        ),
        "mlp_neuron_residual_write_norm": (
            int(model.config.n_layer),
            int(model.config.n_inner or model.config.n_embd * 4),
        ),
        "attention_head_first_order_predicted_token_attribution": (
            int(model.config.n_layer), int(model.config.n_head)
        ),
        "final_norm_input": (int(model.config.n_embd),),
        "final_norm_output": (int(model.config.n_embd),),
        "baseline_logits": (int(model.config.vocab_size),),
        "selected_path_ablated_logits": (int(model.config.vocab_size),),
        "selected_path_restored_logits": (int(model.config.vocab_size),),
        "interaction_full_vocabulary_logit_delta": (int(model.config.vocab_size),),
    }
    loaded = {}
    for suffix, shape in required_shapes.items():
        key = f"{prefix}_{suffix}"
        if key not in arrays:
            errors.append(f"{subject}: raw tensor {key} is missing")
            continue
        value = arrays[key]
        if tuple(value.shape) != shape:
            errors.append(f"{subject}: raw tensor {key} shape {value.shape} != {shape}")
        if not np.isfinite(value).all():
            errors.append(f"{subject}: raw tensor {key} contains non-finite data")
        loaded[suffix] = value
    for component_index in range(len(case["sourcePath"]["attention"]) + len(case["sourcePath"]["mlp"])):
        key = f"{prefix}_individual_{component_index}_ablated_logits"
        if key not in arrays:
            errors.append(f"{subject}: individual component logits {component_index} are missing")
        elif tuple(arrays[key].shape) != (int(model.config.vocab_size),):
            errors.append(f"{subject}: individual component logits {component_index} shape mismatch")
        else:
            loaded[f"individual_{component_index}_ablated_logits"] = arrays[key]
    if len(loaded) != len(required_shapes) + len(case["sourcePath"]["attention"]) + len(case["sourcePath"]["mlp"]):
        return errors

    if not np.array_equal(loaded["input_ids"], np.asarray(context_ids, dtype=np.int64)):
        errors.append(f"{subject}: archived context token IDs differ from recorded context")
    if int(loaded["read_token_index"]) != len(context_ids) - 1 or step.get("readTokenIndex") != len(context_ids) - 1:
        errors.append(f"{subject}: read index is not the final context token")
    input_tensor = torch.tensor([context_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        outputs = model(input_ids=input_tensor, use_cache=False)
    actual_logits = outputs.logits[0, -1].detach().float().cpu().numpy()
    baseline_logits = loaded["baseline_logits"]
    if not np.allclose(actual_logits, baseline_logits, rtol=3e-4, atol=3e-4):
        errors.append(f"{subject}: archived full-vocabulary logits do not reproduce on local GPT-2")
    greedy_id = int(np.argmax(actual_logits))
    if step.get("generatedToken", {}).get("tokenId") != greedy_id:
        errors.append(f"{subject}: recorded generated token is not the recomputed greedy argmax")
    if step.get("baseline", {}).get("greedyNextToken", {}).get("tokenId") != greedy_id:
        errors.append(f"{subject}: baseline greedy token record differs from recomputed logits")
    expected_top = np.argsort(actual_logits)[-5:][::-1].tolist()
    observed_top = [int(row.get("tokenId", -1)) for row in step.get("baseline", {}).get("topTokens", [])]
    if observed_top != expected_top:
        errors.append(f"{subject}: recorded baseline top five differ from full logits")
    probabilities = torch.softmax(torch.as_tensor(actual_logits, dtype=torch.float64), dim=-1).numpy()
    target_record = step.get("baseline", {}).get("target", {})
    if target_record.get("tokenId") != greedy_id or not _close(target_record.get("probability"), probabilities[greedy_id]):
        errors.append(f"{subject}: prediction target score does not describe the actual greedy token")

    if tokenizer is not None:
        prompt = case["prompt"]
        encoded_prompt = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        if encoded_prompt != case.get("tokenization", {}).get("promptTokenIds"):
            errors.append(f"{subject}: prompt tokenization does not reproduce")
    read_index = len(context_ids) - 1
    token_embedding = model.transformer.wte(input_tensor)[0, read_index].detach().float().cpu().numpy()
    position_ids = torch.arange(len(context_ids), device=device).unsqueeze(0)
    position_embedding = model.transformer.wpe(position_ids)[0, read_index].detach().float().cpu().numpy()
    for name, measured in (
        ("token_embedding", token_embedding),
        ("position_embedding", position_embedding),
        ("embedding_output", token_embedding + position_embedding),
    ):
        if not np.allclose(loaded[name], measured, rtol=2e-4, atol=2e-4):
            errors.append(f"{subject}: {name} differs from local GPT-2 embedding tables")
    if not np.allclose(
        loaded["final_norm_output"],
        model.transformer.ln_f(torch.as_tensor(loaded["final_norm_input"]).to(device)).detach().float().cpu().numpy(),
        rtol=3e-4,
        atol=3e-4,
    ):
        errors.append(f"{subject}: final normalization output does not reproduce")

    n_heads = int(model.config.n_head)
    hidden = int(model.config.n_embd)
    head_dim = hidden // n_heads
    selected_nodes = (
        [("attention", node) for node in case["sourcePath"]["attention"]]
        + [("mlp", node) for node in case["sourcePath"]["mlp"]]
    )
    for layer in range(int(model.config.n_layer)):
        block = model.transformer.h[layer]
        residual_in = loaded["residual_input"][layer]
        residual_out = loaded["residual_output"][layer]
        context_heads = loaded["attention_context_by_head"][layer]
        projected = np.stack([
            context_heads[head] @ block.attn.c_proj.weight[
                head * head_dim : (head + 1) * head_dim
            ].detach().float().cpu().numpy()
            for head in range(n_heads)
        ])
        if not np.allclose(projected, loaded["attention_head_residual_write"][layer], rtol=3e-4, atol=3e-4):
            errors.append(f"{subject}: layer {layer} attention head writes differ from c_proj")
        attention_rebuilt = projected.sum(axis=0) + block.attn.c_proj.bias.detach().float().cpu().numpy()
        if not np.allclose(attention_rebuilt, loaded["attention_output"][layer], rtol=3e-4, atol=3e-4):
            errors.append(f"{subject}: layer {layer} attention output differs from head writes")
        mlp_pre = (
            loaded["mlp_input"][layer]
            @ block.mlp.c_fc.weight.detach().float().cpu().numpy()
            + block.mlp.c_fc.bias.detach().float().cpu().numpy()
        )
        if not np.allclose(mlp_pre, loaded["mlp_preactivation"][layer], rtol=4e-4, atol=4e-4):
            errors.append(f"{subject}: layer {layer} MLP preactivation differs from c_fc")
        with torch.no_grad():
            mlp_act = block.mlp.act(torch.as_tensor(mlp_pre, device=device)).detach().float().cpu().numpy()
        if not np.allclose(mlp_act, loaded["mlp_activation"][layer], rtol=4e-4, atol=4e-4):
            errors.append(f"{subject}: layer {layer} MLP activation differs from GPT-2 GELU")
        mlp_rebuilt = (
            mlp_act
            @ block.mlp.c_proj.weight.detach().float().cpu().numpy()
            + block.mlp.c_proj.bias.detach().float().cpu().numpy()
        )
        if not np.allclose(mlp_rebuilt, loaded["mlp_output"][layer], rtol=4e-4, atol=4e-4):
            errors.append(f"{subject}: layer {layer} MLP output differs from c_proj")
        reconstructed_residual = residual_in + loaded["attention_output"][layer] + loaded["mlp_output"][layer]
        if not np.allclose(reconstructed_residual, residual_out, rtol=4e-4, atol=4e-4):
            errors.append(f"{subject}: layer {layer} residual addition does not reconstruct")
        if not _close(
            step["trace"]["layerTrace"][layer]["residualAddMaxAbsReconstructionError"],
            float(np.max(np.abs(reconstructed_residual - residual_out))),
        ):
            errors.append(f"{subject}: layer {layer} residual error summary is inconsistent")

    path_logits = loaded["selected_path_ablated_logits"]
    restore_logits = loaded["selected_path_restored_logits"]
    if not np.allclose(restore_logits, baseline_logits, rtol=1e-5, atol=1e-5):
        errors.append(f"{subject}: restoration did not recover the complete baseline logit vector")
    causal = step.get("causal", {})
    selected_remove = causal.get("selectedPath", {}).get("remove", {})
    target_drop = float(baseline_logits[greedy_id] - path_logits[greedy_id])
    target_probability_change = float(
        torch.softmax(torch.as_tensor(path_logits, dtype=torch.float64), dim=-1)[greedy_id]
        - probabilities[greedy_id]
    )
    if not _close(selected_remove.get("targetLogitDrop"), target_drop):
        errors.append(f"{subject}: selected-path logit effect is not reproducible from raw logits")
    if not _close(selected_remove.get("targetProbabilityChange"), target_probability_change):
        errors.append(f"{subject}: selected-path probability effect is not reproducible from raw logits")
    observed_distribution = selected_remove.get("fullDistributionChange", {})
    for name, measured in _distribution_metrics(baseline_logits, path_logits).items():
        if not _close(observed_distribution.get(name), measured, tolerance=5e-4):
            errors.append(f"{subject}: selected-path full-distribution {name} is inconsistent")

    individual_logits = [
        loaded[f"individual_{index}_ablated_logits"]
        for index in range(len(selected_nodes))
    ]
    interaction_vector_tensor = torch.as_tensor(
        (1 - len(individual_logits)) * baseline_logits - path_logits,
        dtype=torch.float32,
    )
    for changed_logits in individual_logits:
        interaction_vector_tensor += torch.as_tensor(changed_logits, dtype=torch.float32)
    interaction_vector = interaction_vector_tensor.numpy()
    if not np.allclose(
        interaction_vector,
        loaded["interaction_full_vocabulary_logit_delta"],
        rtol=1e-5,
        atol=1e-5,
    ):
        errors.append(f"{subject}: archived full-vocabulary interaction vector is inconsistent")
    interaction = causal.get("interaction", {})
    group_drop = (
        torch.as_tensor(baseline_logits[greedy_id], dtype=torch.float32)
        - torch.as_tensor(path_logits[greedy_id], dtype=torch.float32)
    )
    individual_drop_sum = torch.tensor(0.0, dtype=torch.float32)
    for row in individual_logits:
        individual_drop_sum += (
            torch.as_tensor(baseline_logits[greedy_id], dtype=torch.float32)
            - torch.as_tensor(row[greedy_id], dtype=torch.float32)
        )
    expected_interaction = float(group_drop - individual_drop_sum)
    if not _close(interaction.get("logitDropNonAdditivity"), expected_interaction):
        errors.append(f"{subject}: group-vs-individual interaction arithmetic is inconsistent")
    if not _close(interaction.get("fullVocabularyLogitInteractionL2"), float(np.linalg.norm(interaction_vector))):
        errors.append(f"{subject}: full-vocabulary interaction norm is inconsistent")

    individual_records = causal.get("individualComponentAblations", [])
    if len(individual_records) != len(individual_logits):
        errors.append(f"{subject}: individual component ablation count differs from source path")
    for index, (record, changed) in enumerate(zip(individual_records, individual_logits)):
        expected_drop = float(baseline_logits[greedy_id] - changed[greedy_id])
        if not _close(record.get("remove", {}).get("targetLogitDrop"), expected_drop):
            errors.append(f"{subject}: individual component {index} logit effect is inconsistent")

    controls = causal.get("matchedControls", [])
    sampling = causal.get("matchedControlSampling", {})
    count = int(report_control_count := len(controls))
    if count != 100 or sampling.get("requestedSampleCount") != 100:
        errors.append(f"{subject}: each generated prediction must have exactly 100 control draws")
    selected_attention = case["sourcePath"]["attention"]
    selected_mlp = case["sourcePath"]["mlp"]
    matched_count = 0
    control_effects = []
    control_group_keys = set()
    source_units = {
        ("attention", int(node["layer"]), int(node["head"]), tuple(node["ablateTokenPositions"]))
        for node in selected_attention
    } | {
        ("mlp", int(node["layer"]), int(node["neuron"]), tuple(node["ablateTokenPositions"]))
        for node in selected_mlp
    }
    for control in controls:
        control_effects.append(float(control.get("targetLogitDrop", math.nan)))
        control_group_keys.add(tuple(sorted(
            [
                ("attention", int(node.get("layer", -1)), int(node.get("head", -1)), tuple(node.get("ablateTokenPositions", [])))
                for node in control.get("attention", [])
            ] + [
                ("mlp", int(node.get("layer", -1)), int(node.get("neuron", -1)), tuple(node.get("ablateTokenPositions", [])))
                for node in control.get("mlp", [])
            ]
        )))
        for kind, selected_list, key, unit_key in (
            ("attention", selected_attention, "attention", "head"),
            ("mlp", selected_mlp, "mlp", "neuron"),
        ):
            nodes = control.get(key, [])
            if len(nodes) != len(selected_list):
                errors.append(f"{subject}: matched-control {kind} component count differs")
                continue
            for index, (node, selected) in enumerate(zip(nodes, selected_list)):
                if (
                    node.get("layer") != selected.get("layer")
                    or node.get("ablateTokenPositions") != selected.get("ablateTokenPositions")
                ):
                    errors.append(f"{subject}: matched-control {kind} layer/positions mismatch")
                identity = (
                    kind,
                    int(node.get("layer", -1)),
                    int(node.get(unit_key, -1)),
                    tuple(node.get("ablateTokenPositions", [])),
                )
                if identity in source_units:
                    errors.append(f"{subject}: matched control reuses a discovered component")
        matches = control.get("magnitudeMatches", [])
        if len(matches) != len(selected_attention) + len(selected_mlp):
            errors.append(f"{subject}: matched-control magnitude records are incomplete")
        if matches and all(bool(match.get("withinTolerance")) for match in matches):
            matched_count += 1
        for match in matches:
            selected_mag = float(match.get("selectedMagnitude", math.nan))
            control_mag = float(match.get("controlMagnitude", math.nan))
            relative = abs(control_mag - selected_mag) / max(abs(selected_mag), 1e-8)
            if not _close(match.get("relativeMagnitudeError"), relative, tolerance=1e-5):
                errors.append(f"{subject}: control activation-magnitude mismatch calculation is inconsistent")
        distribution = control.get("fullDistributionChange", {})
        if not _close(distribution.get("totalVariation"), float(distribution.get("l1Probability", math.nan)) * 0.5):
            errors.append(f"{subject}: control TV and L1 distribution changes disagree")
        if not (0 <= float(distribution.get("jensenShannonDivergence", -1)) <= math.log(2) + 1e-5):
            errors.append(f"{subject}: control Jensen-Shannon divergence is outside its valid range")
    specificity = causal.get("specificity", {})
    if specificity.get("magnitudeMatchedControlCount") != matched_count:
        errors.append(f"{subject}: reported magnitude-matched control count is inconsistent")
    if specificity.get("percentileMetric") != "signed-target-logit-drop":
        errors.append(f"{subject}: specificity percentile does not use the frozen signed-effect metric")
    if int(sampling.get("uniqueSampleCount", -1)) != len(control_group_keys):
        errors.append(f"{subject}: unique control count differs from the observed sampled groups")
    if sampling.get("samplingWithReplacement") is not (len(control_group_keys) < count):
        errors.append(f"{subject}: replacement status differs from observed duplicate control groups")
    percentile = 100.0 * (
        sum(value < target_drop for value in control_effects)
        + 0.5 * sum(value == target_drop for value in control_effects)
    ) / max(len(control_effects), 1)
    if not _close(specificity.get("empiricalPercentile"), percentile, tolerance=1e-5):
        errors.append(f"{subject}: signed-effect specificity percentile is inconsistent")
    absolute_control_effects = [abs(value) for value in control_effects]
    absolute_target_effect = abs(target_drop)
    absolute_percentile = 100.0 * (
        sum(value < absolute_target_effect for value in absolute_control_effects)
        + 0.5 * sum(value == absolute_target_effect for value in absolute_control_effects)
    ) / max(len(control_effects), 1)
    if not _close(specificity.get("absoluteEffectPercentile"), absolute_percentile, tolerance=1e-5):
        errors.append(f"{subject}: absolute-effect specificity percentile is inconsistent")
    if specificity.get("percentileMetric") != "signed-target-logit-drop":
        errors.append(f"{subject}: strict specificity does not use signed target-logit drop")
    expected_specificity_pass = (
        percentile >= 95.0
        and matched_count >= math.ceil(100 * 0.95)
        and int(sampling.get("uniqueSampleCount", 0)) >= 20
        and sampling.get("exactStructuralMatch") is True
    )
    if specificity.get("passesStrictSpecificity") is not expected_specificity_pass:
        errors.append(f"{subject}: specificity pass/fail does not follow the frozen threshold")

    return errors


def verify_autoregressive_report(
    report_path: Path,
    variant: str,
    research_report_path: Path | None = None,
) -> List[str]:
    errors = []
    report_path = report_path.resolve()
    if not report_path.is_file():
        return [f"report does not exist: {report_path}"]
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return [f"report JSON cannot be read: {error}"]
    if variant == "regression-five":
        return _verify_regression(report_path)
    if report.get("kind") != "gpt2-autoregressive-causal-computation-trace":
        errors.append("unexpected autoregressive trace report kind")
    if report.get("accessMode") != "instrumented-witness":
        errors.append("report does not identify its instrumented access mode")
    try:
        source_report = json.loads(SOURCE_REPORT.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return errors + [f"frozen source report cannot be read: {error}"]
    source_cases = _source_cases(source_report)
    source_hash = _sha256(SOURCE_REPORT)
    if source_hash != EXPECTED_SOURCE_SHA256 or report.get("sourceArtifact", {}).get("sha256") != source_hash:
        errors.append("source artifact hash differs from the frozen baseline")
    case_set = report.get("protocolConfig", {}).get("caseSet")
    if variant == "all-50":
        expected_cases = source_cases
        if case_set != "all-50" or len(source_cases) != 50:
            errors.append("all-50 verification requires every passed source case")
    elif variant == "pilot-five":
        wanted = ("France", "Germany", "Italy", "Portugal", "Greece")
        by_subject = {row["subject"]: row for row in source_cases}
        expected_cases = [by_subject[name] for name in wanted if name in by_subject]
        if case_set != "pilot-five" or len(expected_cases) != 5:
            errors.append("pilot-five verification requires the frozen five regression subjects")
    else:
        return errors + [f"unsupported experiment variant {variant!r}"]
    cases = report.get("cases", [])
    if [case.get("subject") for case in cases] != [case.get("subject") for case in expected_cases]:
        errors.append("report case order/count differs from the frozen source case selection")
    if report.get("summary", {}).get("caseCount") != len(expected_cases):
        errors.append("summary case count differs from source case selection")
    if report.get("protocolConfig", {}).get("controlCount") != 100:
        errors.append("report does not use exactly 100 matched controls per prediction")
    if report.get("protocolConfig", {}).get("controlMagnitudeRelativeTolerance") != 0.2:
        errors.append("activation magnitude tolerance differs from the frozen 20 percent protocol")
    if not str(report.get("runtime", {}).get("device", "")).startswith("cuda") or not report.get("runtime", {}).get("gpu"):
        errors.append("the experiment was not recorded on a verified CUDA device")

    raw_meta = report.get("rawTensorArchive", {})
    raw_path = (report_path.parent / str(raw_meta.get("file", ""))).resolve()
    if not raw_path.is_relative_to(report_path.parent.resolve()) or not raw_path.is_file():
        errors.append("raw tensor archive is missing or escapes the report directory")
        return errors
    if raw_meta.get("sha256") != _sha256(raw_path):
        errors.append("raw tensor archive SHA-256 mismatch")
        return errors
    try:
        arrays = np.load(raw_path, allow_pickle=False)
    except (OSError, ValueError) as error:
        return errors + [f"raw tensor archive cannot be opened safely: {error}"]
    if set(arrays.files) != set(raw_meta.get("arrays", {})):
        errors.append("raw tensor names do not match report inventory")
    for name, metadata in raw_meta.get("arrays", {}).items():
        if name not in arrays:
            continue
        if list(arrays[name].shape) != metadata.get("shape") or str(arrays[name].dtype) != metadata.get("dtype"):
            errors.append(f"raw array metadata mismatch: {name}")
        if arrays[name].dtype.hasobject or not np.isfinite(arrays[name]).all():
            errors.append(f"raw array is object-typed or non-finite: {name}")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_id = report.get("model", {}).get("id", "gpt2")
        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(model_id, local_files_only=True).eval()
        device = torch.device("cpu")
        model.to(device)
    except (ImportError, OSError, RuntimeError) as error:
        arrays.close()
        return errors + [f"local GPT-2 cannot be loaded for independent reconstruction: {error}"]

    if model.__class__.__name__ != "GPT2LMHeadModel":
        errors.append(f"unexpected local model class {model.__class__.__name__}")
    generated_count = 0
    criteria_count = {
        key: {"pass": 0, "fail": 0}
        for key in (
            "A_internalComputationalTrace",
            "B_correctTokenByTokenReconstruction",
            "C_causalInfluence",
            "D_matchedControlSpecificity",
            "E_interactionAndRedundancyMeasurement",
            "F_actualGreedyPredictionExplanation",
            "G_sourceCaseReproducibility",
        )
    }
    for case, source_case in zip(cases, expected_cases):
        errors.extend(_report_case_for_subject(case, source_case))
        try:
            tokenized = tokenizer(
                case["prompt"],
                return_offsets_mapping=True,
                add_special_tokens=False,
            )
            prompt_ids = [int(token) for token in tokenized["input_ids"]]
            offsets = [[int(start), int(end)] for start, end in tokenized["offset_mapping"]]
            if prompt_ids != case.get("tokenization", {}).get("promptTokenIds"):
                errors.append(f"{case['subject']}: exact prompt token IDs mismatch")
            if offsets != case.get("tokenization", {}).get("promptCharacterOffsets"):
                errors.append(f"{case['subject']}: prompt token offsets mismatch")
        except (KeyError, TypeError, ValueError) as error:
            errors.append(f"{case.get('subject')}: prompt tokenization failed: {error}")
            continue

        steps = case.get("generation", {}).get("steps", [])
        if not steps:
            errors.append(f"{case['subject']}: no generated prediction steps")
            continue
        prior_generated = []
        if steps[0].get("contextTokenIds") != prompt_ids:
            errors.append(f"{case['subject']}: first free-running context differs from original prompt")
        for step_index, step in enumerate(steps):
            expected_context = prompt_ids + prior_generated
            if step.get("contextTokenIds") != expected_context:
                errors.append(f"{case['subject']}: step {step_index} context is not the baseline autoregressive prefix")
            if step.get("generatedPrefixTokenIds") != prior_generated:
                errors.append(f"{case['subject']}: step {step_index} generated-prefix state diverges")
            if step.get("stepIndex") != step_index:
                errors.append(f"{case['subject']}: step indexes are not contiguous")
            errors.extend(_verify_step(
                case=case,
                step=step,
                source_case=source_case,
                arrays=arrays,
                model=model,
                tokenizer=tokenizer,
                device=device,
            ))
            generated_token = int(step.get("generatedToken", {}).get("tokenId", -1))
            if step_index > 0 and step.get("teacherForcedSourceTargetAtStepZero") is not None:
                errors.append(f"{case['subject']}: teacher-forced scoring leaked into a free-running later step")
            prior_generated.append(generated_token)
            generated_count += 1

        criteria = case.get("criteria", {})
        for key, entry in criteria_count.items():
            status = criteria.get(key, {}).get("status")
            if status not in ("pass", "fail"):
                errors.append(f"{case['subject']}: criterion {key} has no explicit pass/fail state")
            else:
                entry[status] += 1
        if criteria.get("causalSufficiency", {}).get("status") != "not-tested":
            errors.append(f"{case['subject']}: path sufficiency must not be conflated with ablation influence")
        if "WHAT:" not in case.get("explanationMarkdown", "") or "**WHY:**" not in case.get("explanationMarkdown", ""):
            errors.append(f"{case['subject']}: token-by-token explanation is incomplete")

    summary = report.get("summary", {})
    if summary.get("generatedPredictionCount") != generated_count:
        errors.append("summary generated token count differs from individual traces")
    if summary.get("criteria") != criteria_count:
        errors.append("summary criterion counts differ from explicit per-case states")
    if research_report_path is not None:
        if not research_report_path.is_file():
            errors.append("research report does not exist")
        else:
            text = research_report_path.read_text(encoding="utf-8").casefold()
            for phrase in (
                "teacher-forced",
                "free-running",
                "causal influence",
                "specificity",
                "sufficiency",
                "100",
                "not established",
            ):
                if phrase.casefold() not in text:
                    errors.append(f"research report omits required evidence boundary: {phrase}")
    arrays.close()
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Independently verify GPT-2 autoregressive traces and matched causal controls."
    )
    parser.add_argument("--variant", choices=("regression-five", "pilot-five", "all-50"), required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--research-report",
        type=Path,
        default=ROOT / "docs/research/2026-10-30-gpt2-autoregressive-causal-trace.md",
    )
    args = parser.parse_args()
    report_path = args.report or (
        ROOT
        / "benchmarks/release/nn-live-gpt2-autoregressive-causal-trace"
        / args.variant
        / "report.json"
    )
    errors = verify_autoregressive_report(report_path, args.variant, args.research_report)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
    print("GPT2_AUTOREGRESSIVE_CAUSAL_TRACE_VERIFIED")


if __name__ == "__main__":
    main()
