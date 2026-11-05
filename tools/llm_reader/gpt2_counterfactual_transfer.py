#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
SOURCE_REPORT = (
    ROOT
    / "benchmarks/release/nn-live-gpt2-ordered-residual-path-circuit-finder"
    / "gpt2-50-facts-ordered-residual-path/report.json"
)
BASELINE_DIR = ROOT / "benchmarks/release/nn-live-gpt2-autoregressive-causal-trace/all-50"
EXPECTED_SOURCE_SHA256 = "f24428b69b9b1fb2d750dfbf2ea55a065fc1d6cc740c415d01ac6ccb94e3715b"
BASELINE_HASHES = {
    "report.json": "820a2ae8a4ce642921405819169d425338b522c092c6ed16259c2e6fb01cda2e",
    "report.md": "9eabff525cb8d26338defc0a53d90201f5a0dfe805d8b8c1526fa58121af1ba0",
    "raw-tensors.npz": "244b33dcc553c8021bdd4cd0a5b5f8f33d97ba014b161b17495493523d5b9ebd",
    "manifest.json": "964188bc10e426cbf367a9fe8c356c74b45a91221bcf380c49e18e7d53066587",
}
CONTROL_COUNT = 100
CONTROL_MAGNITUDE_TOLERANCE = 0.2
BATCH_GROUPS = 16
BOOTSTRAP_REPLICATES = 5000
STATISTICAL_ALPHA = 0.05

sys.path.insert(0, str(ROOT / "tools"))
from llm_reader.gpt2_internal_trace import GPT2TraceCapture, conv1d_head_writes, full_distribution_change, summarize_logits  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test factual information transfer through frozen GPT-2 discovered paths."
    )
    parser.add_argument("--source-report", type=Path, default=SOURCE_REPORT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--variant", choices=("pilot", "all-pairs"), required=True)
    parser.add_argument("--pilot-pairs", type=int, default=10)
    parser.add_argument("--control-count", type=int, default=CONTROL_COUNT)
    parser.add_argument("--control-magnitude-tolerance", type=float, default=CONTROL_MAGNITUDE_TOLERANCE)
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_frozen_baselines() -> None:
    for filename, expected_hash in BASELINE_HASHES.items():
        path = BASELINE_DIR / filename
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError(f"Frozen autoregressive baseline artifact changed or is missing: {path}")


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _as_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().float().cpu().numpy()


def _tensor_scalar(value: torch.Tensor) -> float:
    return float(value.detach().float().cpu())


def _relation_slot(case: Dict[str, Any]) -> Tuple[str, str, Tuple[int, int]] | None:
    prompt = str(case.get("prompt", ""))
    rules = (
        ("country-capital", re.fullmatch(r"The capital of (?P<slot>.+) is", prompt)),
        (
            "planet-size-extreme",
            re.fullmatch(r"The (?P<slot>largest planet|smallest planet) in the solar system is", prompt),
        ),
        (
            "gendered-parent-term",
            re.fullmatch(r"A (?P<slot>female parent|male parent) is called a", prompt),
        ),
        (
            "occupation-description-to-profession",
            re.fullmatch(r"A person who (?P<slot>.+) is a", prompt),
        ),
    )
    for family, match in rules:
        if match is None:
            continue
        slot = match.group("slot")
        if family != "occupation-description-to-profession" and slot != str(case.get("subject", "")):
            return None
        return family, slot, match.span("slot")
    return None


def _tokenize_prompt(case: Dict[str, Any], tokenizer: Any) -> Dict[str, Any]:
    encoded = tokenizer(
        str(case["prompt"]),
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    ids = [int(value) for value in encoded["input_ids"]]
    offsets = [[int(start), int(end)] for start, end in encoded["offset_mapping"]]
    return {"ids": ids, "offsets": offsets}


def _validate_answer_sequence(case: Dict[str, Any], tokenizer: Any, prompt_ids: List[int]) -> Tuple[List[int] | None, str | None]:
    answer = case.get("answer")
    target = case.get("target", {})
    if not isinstance(answer, str) or not answer.strip():
        return None, "missing independently recorded answer string"
    target_id = target.get("tokenId") if isinstance(target, dict) else None
    if not isinstance(target_id, int):
        return None, "missing independently recorded target-token ID"
    answer_ids = [
        int(value)
        for value in tokenizer(" " + answer, add_special_tokens=False)["input_ids"]
    ]
    if not answer_ids:
        return None, "answer string tokenizes to an empty sequence"
    full_ids = [
        int(value)
        for value in tokenizer(str(case["prompt"]) + " " + answer, add_special_tokens=False)["input_ids"]
    ]
    if full_ids != prompt_ids + answer_ids:
        return None, "GPT-2 tokenization does not preserve the prompt/answer boundary"
    if answer_ids[0] != target_id:
        return None, "answer sequence first token differs from the source artifact target token"
    decoded_target = tokenizer.decode([target_id]).replace("\n", "\\n")
    if decoded_target != target.get("token"):
        return None, "source target-token text differs from the local tokenizer"
    return answer_ids, None


def _shared_token_mapping(
    source_tokenization: Dict[str, Any],
    source_slot: Tuple[int, int],
    destination_tokenization: Dict[str, Any],
    destination_slot: Tuple[int, int],
    source_position: int,
) -> Tuple[int | None, str | None]:
    source_ids = source_tokenization["ids"]
    destination_ids = destination_tokenization["ids"]
    source_offsets = source_tokenization["offsets"]
    destination_offsets = destination_tokenization["offsets"]
    source_start, source_end = source_slot
    destination_start, destination_end = destination_slot
    if source_position < 0 or source_position >= len(source_ids):
        return None, "source path position falls outside source prompt tokens"
    start, end = source_offsets[source_position]
    if end <= source_start:
        source_region = [index for index, (_, token_end) in enumerate(source_offsets) if token_end <= source_start]
        destination_region = [index for index, (_, token_end) in enumerate(destination_offsets) if token_end <= destination_start]
    elif start >= source_end:
        source_region = [index for index, (token_start, _) in enumerate(source_offsets) if token_start >= source_end]
        destination_region = [index for index, (token_start, _) in enumerate(destination_offsets) if token_start >= destination_end]
    else:
        return None, "source path token overlaps the variable fact slot"
    try:
        ordinal = source_region.index(source_position)
    except ValueError:
        return None, "source path token is not in an exactly alignable prompt region"
    if ordinal >= len(destination_region):
        return None, "destination prompt has no corresponding fixed-text token"
    destination_position = destination_region[ordinal]
    if source_ids[source_position] != destination_ids[destination_position]:
        return None, "corresponding fixed-text tokens have different token IDs"
    return destination_position, None


def _pair_alignment(
    case_a: Dict[str, Any],
    case_b: Dict[str, Any],
    tokenizer: Any,
    tokenized: Dict[str, Dict[str, Any]],
    slots: Dict[str, Tuple[str, str, Tuple[int, int]]],
) -> Tuple[Dict[str, Any] | None, str | None]:
    case_a_id = str(case_a["subject"])
    case_b_id = str(case_b["subject"])
    family_a, slot_a, span_a = slots[case_a_id]
    family_b, slot_b, span_b = slots[case_b_id]
    if family_a != family_b:
        return None, "relation-family mismatch"
    prompt_a = str(case_a["prompt"])
    prompt_b = str(case_b["prompt"])
    skeleton_a = prompt_a[:span_a[0]] + "{slot}" + prompt_a[span_a[1]:]
    skeleton_b = prompt_b[:span_b[0]] + "{slot}" + prompt_b[span_b[1]:]
    if skeleton_a != skeleton_b:
        return None, "prompt skeleton mismatch after replacing only the variable relation slot"
    read_a = int(case_a.get("readTokenIndex", -1))
    read_b = int(case_b.get("readTokenIndex", -1))
    if read_a != len(tokenized[case_a_id]["ids"]) - 1 or read_b != len(tokenized[case_b_id]["ids"]) - 1:
        return None, "source read index is not the final prompt token"
    nodes_a = case_a["orderedResidualPath"]["nodes"]
    nodes_b = case_b["orderedResidualPath"]["nodes"]
    position_maps = {"aToB": {}, "bToA": {}}
    for direction, donor, recipient, donor_slot, recipient_slot, donor_name, recipient_name in (
        ("aToB", case_a, case_b, span_a, span_b, case_a_id, case_b_id),
        ("bToA", case_b, case_a, span_b, span_a, case_b_id, case_a_id),
    ):
        for kind in ("attention", "mlp"):
            for node_index, node in enumerate(donor["orderedResidualPath"]["nodes"].get(kind, [])):
                for position in node.get("ablateTokenPositions", []):
                    mapped, error = _shared_token_mapping(
                        tokenized[donor_name],
                        donor_slot,
                        tokenized[recipient_name],
                        recipient_slot,
                        int(position),
                    )
                    if error:
                        return None, f"{direction} {kind} path position {position}: {error}"
                    position_maps[direction][f"{kind}:{node_index}:{position}"] = mapped
    return {
        "relationFamily": family_a,
        "promptSkeleton": skeleton_a,
        "aToBPositionMap": position_maps["aToB"],
        "bToAPositionMap": position_maps["bToA"],
        "slotA": {"text": slot_a, "charSpan": list(span_a)},
        "slotB": {"text": slot_b, "charSpan": list(span_b)},
        "readTokenIndexA": read_a,
        "readTokenIndexB": read_b,
        "fixedContextAlignment": "exact token-id alignment in shared prefix/suffix; variable-slot tokens are never patched",
    }, None


def build_pair_inventory(
    source_report: Dict[str, Any],
    tokenizer: Any,
) -> Dict[str, Any]:
    passed = [case for case in source_report.get("cases", []) if case.get("status") == "passed"]
    if len(passed) != int(source_report.get("config", {}).get("caseCount", -1)):
        raise ValueError("Frozen source case count does not equal its declared passed-case count.")
    valid_cases: List[Dict[str, Any]] = []
    exclusions: List[Dict[str, str]] = []
    tokenized: Dict[str, Dict[str, Any]] = {}
    slots: Dict[str, Tuple[str, str, Tuple[int, int]]] = {}
    answer_ids: Dict[str, List[int]] = {}
    for case in passed:
        subject = str(case.get("subject", ""))
        slot = _relation_slot(case)
        if slot is None:
            exclusions.append({"subject": subject, "reason": "no independently derivable repeated relation-prompt template"})
            continue
        tokenized[subject] = _tokenize_prompt(case, tokenizer)
        ids, error = _validate_answer_sequence(case, tokenizer, tokenized[subject]["ids"])
        if error:
            exclusions.append({"subject": subject, "reason": error})
            continue
        nodes = case.get("orderedResidualPath", {}).get("nodes", {})
        if not nodes.get("attention") or not nodes.get("mlp"):
            exclusions.append({"subject": subject, "reason": "source artifact has no complete attention-plus-MLP discovered path"})
            continue
        if not isinstance(case.get("readTokenIndex"), int):
            exclusions.append({"subject": subject, "reason": "source artifact has no integer read-token index"})
            continue
        slots[subject] = slot
        answer_ids[subject] = ids
        valid_cases.append(case)

    families: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for case in valid_cases:
        families[slots[str(case["subject"])][0]].append(case)
    eligible_pairs: List[Dict[str, Any]] = []
    rejected_pairs: List[Dict[str, Any]] = []
    for family, cases in families.items():
        for case_a, case_b in itertools.combinations(cases, 2):
            alignment, error = _pair_alignment(case_a, case_b, tokenizer, tokenized, slots)
            if error:
                rejected_pairs.append({
                    "relationFamily": family,
                    "subjectA": case_a["subject"],
                    "subjectB": case_b["subject"],
                    "reason": error,
                })
                continue
            eligible_pairs.append({
                "pairId": f"pair_{len(eligible_pairs):04d}",
                "relationFamily": family,
                "subjectA": str(case_a["subject"]),
                "subjectB": str(case_b["subject"]),
                "alignment": alignment,
            })
    eligible_pairs.sort(key=lambda pair: (
        list(case["subject"] for case in passed).index(pair["subjectA"]),
        list(case["subject"] for case in passed).index(pair["subjectB"]),
    ))
    for index, pair in enumerate(eligible_pairs):
        pair["pairId"] = f"pair_{index:04d}"
    family_summary = []
    for family, cases in sorted(families.items()):
        pair_count = sum(pair["relationFamily"] == family for pair in eligible_pairs)
        family_summary.append({
            "relationFamily": family,
            "promptPatternDerivation": "literal prompt regex with one explicit variable slot",
            "factCount": len(cases),
            "subjects": [str(case["subject"]) for case in cases],
            "eligibleUnorderedPairCount": pair_count,
        })
    grouped_subjects = {str(case["subject"]) for cases in families.values() for case in cases}
    for case in passed:
        subject = str(case["subject"])
        if subject not in grouped_subjects and not any(item["subject"] == subject for item in exclusions):
            exclusions.append({"subject": subject, "reason": "no eligible counterpart in a repeated relation family"})
        elif subject in grouped_subjects and not any(pair["subjectA"] == subject or pair["subjectB"] == subject for pair in eligible_pairs):
            exclusions.append({"subject": subject, "reason": "all candidate pairs failed exact prompt/path alignment"})
    return {
        "cases": valid_cases,
        "tokenizedPrompts": tokenized,
        "slots": slots,
        "answerTokenIds": answer_ids,
        "pairs": eligible_pairs,
        "inventory": {
            "passedSourceFactCount": len(passed),
            "pairEligibleFactCount": len(valid_cases),
            "relationFamilies": family_summary,
            "eligibleUnorderedPairCount": len(eligible_pairs),
            "excludedFacts": exclusions,
            "excludedFactCount": len(exclusions),
            "rejectedCandidatePairs": rejected_pairs,
            "rejectedCandidatePairCount": len(rejected_pairs),
            "excludedSourceRows": [
                {"subject": row.get("subject"), "status": row.get("status"), "reason": row.get("reason")}
                for row in source_report.get("cases", [])
                if row.get("status") != "passed"
            ],
        },
    }


def _model_input_ids(case: Dict[str, Any], answer_ids: Sequence[int], tokenizer: Any, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, int]:
    prompt_ids = [int(item) for item in tokenizer(str(case["prompt"]), add_special_tokens=False)["input_ids"]]
    input_ids = prompt_ids + [int(item) for item in answer_ids[:-1]]
    tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(tensor)
    return tensor, attention_mask, len(prompt_ids) - 1


def _score_sequence_logits(logits: torch.Tensor, prompt_length: int, answer_ids: Sequence[int]) -> Dict[str, Any]:
    if not answer_ids:
        raise ValueError("Cannot score an empty answer token sequence.")
    token_log_probs = []
    token_probs = []
    for index, target in enumerate(answer_ids):
        position = prompt_length - 1 + index
        if position >= logits.shape[0]:
            raise ValueError("Answer target position falls outside captured logits.")
        log_probabilities = torch.log_softmax(logits[position].float(), dim=-1)
        token_log_probs.append(float(log_probabilities[int(target)].detach().cpu()))
        token_probs.append(float(torch.exp(log_probabilities[int(target)]).detach().cpu()))
    greedy = summarize_logits(logits[prompt_length - 1], int(answer_ids[0]), _DecodeOnlyTokenizer(), count=5)
    return {
        "answerTokenIds": [int(item) for item in answer_ids],
        "tokenLogProbabilities": token_log_probs,
        "totalLogLikelihood": float(sum(token_log_probs)),
        "meanTokenLogLikelihood": float(np.mean(token_log_probs)),
        "sequenceProbability": float(math.exp(max(-745.0, sum(token_log_probs)))),
        "firstTokenProbability": token_probs[0],
        "firstTokenLogit": float(logits[prompt_length - 1, int(answer_ids[0])].detach().float().cpu()),
        "greedyNextTokenId": greedy["greedyNextToken"]["tokenId"],
        "greedyNextTokenLogit": greedy["greedyNextToken"]["logit"],
    }


class _DecodeOnlyTokenizer:
    def decode(self, token_ids: Sequence[int]) -> str:
        return str(int(token_ids[0]))


def capture_original_facts(
    cases: Sequence[Dict[str, Any]],
    tokenized_prompts: Dict[str, Dict[str, Any]],
    answer_token_ids: Dict[str, List[int]],
    model: Any,
    tokenizer: Any,
    device: torch.device,
    archive: Dict[str, np.ndarray],
) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    states: Dict[str, Dict[str, Any]] = {}
    records: List[Dict[str, Any]] = []
    for index, case in enumerate(cases):
        subject = str(case["subject"])
        prompt_ids = tokenized_prompts[subject]["ids"]
        answer_ids = answer_token_ids[subject]
        context_ids, attention_mask, read_index = _model_input_ids(case, answer_ids, tokenizer, device)
        prefix = f"fact_{index:02d}"
        with GPT2TraceCapture(model) as capture:
            with torch.no_grad():
                outputs = model(input_ids=context_ids, attention_mask=attention_mask, use_cache=False)
            logits = outputs.logits[0].detach().float()
            answer_score = _score_sequence_logits(logits, len(prompt_ids), answer_ids)
            next_summary = summarize_logits(logits[read_index], answer_ids[0], tokenizer, count=5)
            layer_trace = []
            max_residual_error = 0.0
            for layer in range(capture.layers):
                block_input = capture.values["block_input"][layer][0, read_index]
                block_output = capture.values["block_output"][layer][0, read_index]
                attention_output = capture.values[f"attention_output_{layer}"][0, read_index]
                mlp_output = capture.values[f"mlp_output_{layer}"][0, read_index]
                error = float((block_output - block_input - attention_output - mlp_output).abs().max().cpu())
                max_residual_error = max(max_residual_error, error)
                layer_trace.append({
                    "layer": layer,
                    "residualInputL2": _tensor_scalar(torch.linalg.vector_norm(block_input)),
                    "attentionOutputL2": _tensor_scalar(torch.linalg.vector_norm(attention_output)),
                    "mlpOutputL2": _tensor_scalar(torch.linalg.vector_norm(mlp_output)),
                    "residualOutputL2": _tensor_scalar(torch.linalg.vector_norm(block_output)),
                    "residualAddMaxAbsReconstructionError": error,
                })
            state = {
                "case": case,
                "promptIds": prompt_ids,
                "answerIds": answer_ids,
                "promptLength": len(prompt_ids),
                "readIndex": read_index,
                "inputIds": context_ids,
                "attentionMask": attention_mask,
                "baselineLogits": logits,
                "baselinePromptLogits": logits[read_index],
                "baselineAnswerScore": answer_score,
                "baselineSummary": next_summary,
                "captureValues": capture.values,
                "headCount": int(model.config.n_head),
                "prefix": prefix,
            }
            states[subject] = state
            archive[f"{prefix}_prompt_token_ids"] = np.asarray(prompt_ids, dtype=np.int64)
            archive[f"{prefix}_answer_token_ids"] = np.asarray(answer_ids, dtype=np.int64)
            archive[f"{prefix}_answer_teacher_forced_logits"] = _as_numpy(
                logits[len(prompt_ids) - 1 : len(prompt_ids) - 1 + len(answer_ids)]
            )
            archive[f"{prefix}_prompt_next_logits"] = _as_numpy(logits[read_index])
            archive[f"{prefix}_residual_input_at_read"] = _as_numpy(torch.stack([
                capture.values["block_input"][layer][0, read_index] for layer in range(capture.layers)
            ]))
            archive[f"{prefix}_residual_output_at_read"] = _as_numpy(torch.stack([
                capture.values["block_output"][layer][0, read_index] for layer in range(capture.layers)
            ]))
            archive[f"{prefix}_attention_output_at_read"] = _as_numpy(torch.stack([
                capture.values[f"attention_output_{layer}"][0, read_index] for layer in range(capture.layers)
            ]))
            archive[f"{prefix}_mlp_output_at_read"] = _as_numpy(torch.stack([
                capture.values[f"mlp_output_{layer}"][0, read_index] for layer in range(capture.layers)
            ]))
            for kind in ("attention", "mlp"):
                for node_index, node in enumerate(case["orderedResidualPath"]["nodes"].get(kind, [])):
                    unit = int(node["head"] if kind == "attention" else node["neuron"])
                    layer = int(node["layer"])
                    for position in node["ablateTokenPositions"]:
                        if kind == "attention":
                            head_dim = int(model.transformer.h[layer].attn.head_dim)
                            start = unit * head_dim
                            value = capture.values["attention_context"][layer][
                                0, int(position), start : start + head_dim
                            ]
                        else:
                            value = capture.values[f"mlp_activation_{layer}"][0, int(position), unit].reshape(1)
                        archive[f"{prefix}_path_{kind}_{node_index}_{position}"] = _as_numpy(value)
        records.append({
            "subject": subject,
            "prompt": str(case["prompt"]),
            "answer": str(case["answer"]),
            "promptTokenIds": prompt_ids,
            "answerTokenIds": answer_ids,
            "sourceReadTokenIndex": int(case["readTokenIndex"]),
            "greedyNextToken": next_summary["greedyNextToken"],
            "topFivePredictions": next_summary["topTokens"],
            "sourceAnswerFirstToken": next_summary["target"],
            "answerSequenceLikelihood": answer_score,
            "layerTrace": layer_trace,
            "maxResidualAddReconstructionError": max_residual_error,
            "sourcePath": case["orderedResidualPath"]["nodes"],
            "rawArchivePrefix": prefix,
        })
    return states, records


def _site_magnitude(
    state: Dict[str, Any],
    kind: str,
    layer: int,
    unit: int,
    positions: Sequence[int],
) -> float:
    values = state["captureValues"]
    if kind == "mlp":
        activation = values[f"mlp_activation_{layer}"][0]
        scalars = [activation[int(position), int(unit)].abs() for position in positions]
        return float(torch.stack(scalars).mean().detach().float().cpu())
    context = values["attention_context"][layer][0]
    head_dim = int(context.shape[-1]) // int(state["headCount"])
    start = int(unit) * head_dim
    end = start + head_dim
    magnitudes = [
        torch.linalg.vector_norm(context[int(position), start:end])
        for position in positions
    ]
    return float(torch.stack(magnitudes).mean().detach().float().cpu())


def _site_value(state: Dict[str, Any], kind: str, layer: int, unit: int, position: int) -> torch.Tensor:
    values = state["captureValues"]
    if kind == "mlp":
        return values[f"mlp_activation_{layer}"][0, int(position), int(unit)].detach()
    head_dim = int(values["attention_context"][layer].shape[-1]) // int(state["headCount"])
    start = int(unit) * head_dim
    return values["attention_context"][layer][0, int(position), start : start + head_dim].detach()


def _path_entries(case: Dict[str, Any]) -> List[Dict[str, Any]]:
    entries = []
    nodes = case["orderedResidualPath"]["nodes"]
    for kind in ("attention", "mlp"):
        for node in nodes.get(kind, []):
            entries.append({
                "kind": kind,
                "layer": int(node["layer"]),
                "unit": int(node["head"] if kind == "attention" else node["neuron"]),
                "sourcePositions": [int(value) for value in node["ablateTokenPositions"]],
            })
    return entries


def _recipient_position_map(
    donor: Dict[str, Any],
    recipient: Dict[str, Any],
    donor_slot: Tuple[int, int],
    recipient_slot: Tuple[int, int],
    tokenized: Dict[str, Dict[str, Any]],
) -> Dict[int, int]:
    result = {}
    for entry in _path_entries(donor["case"]):
        for source_position in entry["sourcePositions"]:
            mapped, error = _shared_token_mapping(
                tokenized[str(donor["case"]["subject"])],
                donor_slot,
                tokenized[str(recipient["case"]["subject"])],
                recipient_slot,
                source_position,
            )
            if error:
                raise ValueError(f"{donor['case']['subject']} -> {recipient['case']['subject']}: {error}")
            result[source_position] = int(mapped)
    return result


def _make_transfer_group(
    donor: Dict[str, Any],
    recipient: Dict[str, Any],
    entries: Sequence[Dict[str, Any]],
    position_map: Dict[int, int],
    model: Any,
    label: str,
    magnitude_matches: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    sites = []
    for entry in entries:
        positions = []
        for source_position in entry["sourcePositions"]:
            recipient_position = position_map[int(source_position)]
            positions.append({
                "sourcePosition": int(source_position),
                "recipientPosition": int(recipient_position),
                "donorValue": _site_value(
                    donor,
                    entry["kind"],
                    entry["layer"],
                    entry["unit"],
                    int(source_position),
                ),
                "recipientOriginalValue": _site_value(
                    recipient,
                    entry["kind"],
                    entry["layer"],
                    entry["unit"],
                    int(recipient_position),
                ),
            })
        sites.append({
            "kind": entry["kind"],
            "layer": entry["layer"],
            "unit": entry["unit"],
            "positions": positions,
        })
    return {
        "label": label,
        "sites": sites,
        "magnitudeMatches": magnitude_matches or [],
    }


def _candidate_unit_width(model: Any, kind: str, layer: int, state: Dict[str, Any]) -> int:
    if kind == "attention":
        return int(model.config.n_head)
    return int(state["captureValues"][f"mlp_activation_{layer}"].shape[-1])


def _sample_control_groups(
    donor: Dict[str, Any],
    model: Any,
    position_map: Dict[int, int],
    control_count: int,
    tolerance: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    entries = _path_entries(donor["case"])
    selected_by_layer: Dict[Tuple[str, int], set[int]] = defaultdict(set)
    for entry in entries:
        selected_by_layer[(entry["kind"], entry["layer"])].add(entry["unit"])
    candidate_lists = []
    selected_magnitudes = []
    entry_metadata = []
    for entry in entries:
        selected_magnitude = _site_magnitude(
            donor, entry["kind"], entry["layer"], entry["unit"], entry["sourcePositions"]
        )
        selected_magnitudes.append(selected_magnitude)
        candidates = []
        width = _candidate_unit_width(model, entry["kind"], entry["layer"], donor)
        for unit in range(width):
            if unit in selected_by_layer[(entry["kind"], entry["layer"])]:
                continue
            magnitude = _site_magnitude(
                donor, entry["kind"], entry["layer"], unit, entry["sourcePositions"]
            )
            relative = abs(magnitude - selected_magnitude) / max(abs(selected_magnitude), 1e-8)
            candidates.append({"unit": unit, "magnitude": magnitude, "relativeMagnitudeError": relative})
        if not candidates:
            raise ValueError(f"No structural control unit exists for {entry['kind']} layer {entry['layer']}.")
        candidates.sort(key=lambda item: (item["relativeMagnitudeError"], item["unit"]))
        matched = [item for item in candidates if item["relativeMagnitudeError"] <= tolerance]
        pool = matched[:] if matched else [candidates[0]]
        candidate_lists.append({"all": candidates, "pool": pool})
        entry_metadata.append({
            "componentType": entry["kind"],
            "layer": entry["layer"],
            "sourcePositions": entry["sourcePositions"],
            "recipientPositions": [position_map[position] for position in entry["sourcePositions"]],
            "selectedUnit": entry["unit"],
            "selectedDonorMagnitude": selected_magnitude,
            "candidateCount": len(candidates),
            "withinToleranceCandidateCount": len(matched),
            "relativeMagnitudeTolerance": tolerance,
            "matchingStatus": "within-tolerance" if matched else "nearest-fallback",
        })
    while math.prod(len(item["pool"]) for item in candidate_lists) < control_count:
        expandable = [
            index for index, item in enumerate(candidate_lists)
            if len(item["pool"]) < len(item["all"])
        ]
        if not expandable:
            break
        selected_index = min(
            expandable,
            key=lambda index: (
                candidate_lists[index]["all"][len(candidate_lists[index]["pool"])]["relativeMagnitudeError"],
                index,
            ),
        )
        candidate_lists[selected_index]["pool"].append(
            candidate_lists[selected_index]["all"][len(candidate_lists[selected_index]["pool"])]
        )
        metadata = entry_metadata[selected_index]
        metadata["matchingStatus"] = (
            "within-tolerance"
            if all(item["relativeMagnitudeError"] <= tolerance for item in candidate_lists[selected_index]["pool"])
            else "expanded-nearest-after-available-matches"
        )
    rng = random.Random(seed)
    controls = []
    unique = set()
    attempts = 0
    max_attempts = max(1000, control_count * 500)
    while len(controls) < control_count and attempts < max_attempts:
        attempts += 1
        chosen = [rng.choice(item["pool"]) for item in candidate_lists]
        units_seen: Dict[Tuple[str, int], set[int]] = defaultdict(set)
        valid = True
        for entry, choice in zip(entries, chosen):
            key = (entry["kind"], entry["layer"])
            if choice["unit"] in units_seen[key]:
                valid = False
                break
            units_seen[key].add(choice["unit"])
        if not valid:
            continue
        identity = tuple((entry["kind"], entry["layer"], choice["unit"]) for entry, choice in zip(entries, chosen))
        if identity in unique:
            continue
        unique.add(identity)
        control_entries = [
            {
                "kind": entry["kind"],
                "layer": entry["layer"],
                "unit": choice["unit"],
                "sourcePositions": list(entry["sourcePositions"]),
            }
            for entry, choice in zip(entries, chosen)
        ]
        matches = []
        for entry, choice, selected_magnitude in zip(entries, chosen, selected_magnitudes):
            matches.append({
                "componentType": entry["kind"],
                "layer": entry["layer"],
                "unit": choice["unit"],
                "selectedMagnitude": selected_magnitude,
                "controlMagnitude": choice["magnitude"],
                "relativeMagnitudeError": choice["relativeMagnitudeError"],
                "withinTolerance": choice["relativeMagnitudeError"] <= tolerance,
            })
        controls.append({
            "entries": control_entries,
            "magnitudeMatches": matches,
            "controlIndex": len(controls),
        })
    if len(controls) != control_count:
        raise RuntimeError(
            f"Only built {len(controls)} unique controls from the available structural control candidates."
        )
    metadata = {
        "seed": seed,
        "requestedCount": control_count,
        "uniqueCount": len(unique),
        "samplingWithReplacement": False,
        "exactStructuralMatch": True,
        "componentPools": entry_metadata,
        "magnitudeMatchedUniqueGroupCount": sum(
            all(item["withinTolerance"] for item in control["magnitudeMatches"])
            for control in controls
        ),
    }
    return controls, metadata


def _group_for_control(
    donor: Dict[str, Any],
    recipient: Dict[str, Any],
    control: Dict[str, Any],
    position_map: Dict[int, int],
    model: Any,
) -> Dict[str, Any]:
    return _make_transfer_group(
        donor,
        recipient,
        control["entries"],
        position_map,
        model,
        f"control-{control['controlIndex']}",
        control["magnitudeMatches"],
    )


def _group_for_transfer(
    donor: Dict[str, Any],
    recipient: Dict[str, Any],
    position_map: Dict[int, int],
    model: Any,
) -> Dict[str, Any]:
    return _make_transfer_group(
        donor,
        recipient,
        _path_entries(donor["case"]),
        position_map,
        model,
        "discovered-path-transplant",
    )


def _input_rows(
    recipient: Dict[str, Any],
    answer_options: Sequence[Sequence[int]],
    groups: Sequence[Dict[str, Any]],
    tokenizer: Any,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[int, int]]]:
    pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    rows = []
    row_index = []
    for group_index, _group in enumerate(groups):
        for option_index, answer_ids in enumerate(answer_options):
            prompt_ids = recipient["promptIds"]
            rows.append(prompt_ids + [int(value) for value in answer_ids[:-1]])
            row_index.append((group_index, option_index))
    max_length = max(len(row) for row in rows)
    ids = torch.full((len(rows), max_length), int(pad_id), dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for index, row in enumerate(rows):
        ids[index, :len(row)] = torch.tensor(row, dtype=torch.long, device=device)
        mask[index, :len(row)] = 1
    return ids, mask, row_index


def _install_patch_hooks(
    model: Any,
    groups: Sequence[Dict[str, Any]],
    row_index: Sequence[Tuple[int, int]],
) -> List[Any]:
    row_sites: Dict[int, List[Tuple[int, Dict[str, Any]]]] = defaultdict(list)
    for row, (group_index, _option_index) in enumerate(row_index):
        for site in groups[group_index]["sites"]:
            row_sites[row].append((group_index, site))
    sites_by_layer: Dict[Tuple[str, int], Dict[int, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row, values in row_sites.items():
        for _group_index, site in values:
            sites_by_layer[(site["kind"], site["layer"])][row].append(site)
    handles = []
    for (kind, layer), by_row in sites_by_layer.items():
        if kind == "mlp":
            module = model.transformer.h[layer].mlp.act

            def patch_mlp(
                _module: Any,
                _inputs: Any,
                output: torch.Tensor,
                row_nodes: Dict[int, List[Dict[str, Any]]] = by_row,
            ) -> torch.Tensor:
                patched = output.clone()
                for row, nodes in row_nodes.items():
                    for site in nodes:
                        unit = int(site["unit"])
                        for position in site["positions"]:
                            value = position.get("donorValue")
                            patched[row, int(position["recipientPosition"]), unit] = (
                                0.0 if value is None else value.to(device=patched.device, dtype=patched.dtype)
                            )
                return patched

            handles.append(module.register_forward_hook(patch_mlp))
        else:
            module = model.transformer.h[layer].attn.c_proj
            head_dim = int(model.transformer.h[layer].attn.head_dim)

            def patch_attention(
                _module: Any,
                inputs: Tuple[torch.Tensor, ...],
                row_nodes: Dict[int, List[Dict[str, Any]]] = by_row,
                dimension: int = head_dim,
            ) -> Tuple[torch.Tensor, ...]:
                patched = inputs[0].clone()
                for row, nodes in row_nodes.items():
                    for site in nodes:
                        first = int(site["unit"]) * dimension
                        last = first + dimension
                        for position in site["positions"]:
                            value = position.get("donorValue")
                            patched[row, int(position["recipientPosition"]), first:last] = (
                                0.0
                                if value is None
                                else value.to(device=patched.device, dtype=patched.dtype)
                            )
                return (patched,) + tuple(inputs[1:])

            handles.append(module.register_forward_pre_hook(patch_attention))
    return handles


def _evaluate_groups(
    model: Any,
    tokenizer: Any,
    recipient: Dict[str, Any],
    answer_options: Sequence[Sequence[int]],
    groups: Sequence[Dict[str, Any]],
    device: torch.device,
    archive: Dict[str, np.ndarray] | None = None,
    archive_prefix: str | None = None,
) -> List[Dict[str, Any]]:
    outputs_by_group: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for start in range(0, len(groups), BATCH_GROUPS):
        batch_groups = groups[start : start + BATCH_GROUPS]
        ids, mask, row_index = _input_rows(recipient, answer_options, batch_groups, tokenizer, device)
        handles = _install_patch_hooks(model, batch_groups, row_index)
        try:
            with torch.no_grad():
                batch_logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits.detach().float()
        finally:
            for handle in reversed(handles):
                handle.remove()
        for row, (group_local_index, option_index) in enumerate(row_index):
            group_global_index = start + group_local_index
            answer_ids = answer_options[option_index]
            row_input_length = recipient["promptLength"] + max(0, len(answer_ids) - 1)
            scored = _score_sequence_logits(
                batch_logits[row, :row_input_length],
                recipient["promptLength"],
                answer_ids,
            )
            scored["topFiveAtPrompt"] = [
                {
                    "rank": rank + 1,
                    "tokenId": int(token_id),
                    "logit": float(logit.detach().cpu()),
                    "probability": float(probability.detach().cpu()),
                }
                for rank, (logit, token_id, probability) in enumerate(zip(
                    torch.topk(batch_logits[row, recipient["readIndex"]], k=5).values,
                    torch.topk(batch_logits[row, recipient["readIndex"]], k=5).indices,
                    torch.softmax(batch_logits[row, recipient["readIndex"]], dim=-1)[
                        torch.topk(batch_logits[row, recipient["readIndex"]], k=5).indices
                    ],
                ))
            ]
            scored["promptLogits"] = batch_logits[row, recipient["readIndex"]].detach().cpu()
            scored["answerPositionLogits"] = batch_logits[
                row,
                recipient["promptLength"] - 1 : recipient["promptLength"] - 1 + len(answer_ids),
            ].detach().cpu()
            outputs_by_group[group_global_index].append(scored)
    results = []
    for index, group in enumerate(groups):
        option_scores = outputs_by_group[index]
        if len(option_scores) != len(answer_options):
            raise RuntimeError("Counterfactual output batch lost an answer-score row.")
        result = {
            "groupLabel": group["label"],
            "answerScores": [
                {
                    key: value
                    for key, value in score.items()
                    if key not in ("promptLogits", "answerPositionLogits")
                }
                for score in option_scores
            ],
            "_promptLogits": option_scores[0]["promptLogits"],
            "_answerPositionLogits": [score["answerPositionLogits"] for score in option_scores],
            "magnitudeMatches": group.get("magnitudeMatches", []),
            "patchSiteCount": len(group["sites"]),
            "patchedActivationCount": sum(len(site["positions"]) for site in group["sites"]),
        }
        results.append(result)
        if archive is not None and archive_prefix is not None:
            if group["label"] == "unmodified-recipient-baseline":
                archive[f"{archive_prefix}_baseline_prompt_logits"] = _as_numpy(result["_promptLogits"])
                for option_index, logits in enumerate(result["_answerPositionLogits"]):
                    archive[f"{archive_prefix}_baseline_answer_logits_{option_index}"] = _as_numpy(logits)
            elif group["label"] == "discovered-path-transplant":
                archive[f"{archive_prefix}_selected_prompt_logits"] = _as_numpy(result["_promptLogits"])
                for option_index, logits in enumerate(result["_answerPositionLogits"]):
                    archive[f"{archive_prefix}_selected_answer_logits_{option_index}"] = _as_numpy(logits)
            elif group["label"] in (
                "recipient-original-activation-identity-control",
                "recipient-original-activation-restoration",
            ):
                suffix = (
                    "identity"
                    if group["label"] == "recipient-original-activation-identity-control"
                    else "restored"
                )
                archive[f"{archive_prefix}_{suffix}_prompt_logits"] = _as_numpy(result["_promptLogits"])
                for option_index, logits in enumerate(result["_answerPositionLogits"]):
                    archive[f"{archive_prefix}_{suffix}_answer_logits_{option_index}"] = _as_numpy(logits)
    return results


def _seed_for_pair(source_sha: str, pair_id: str, direction: str, control_count: int) -> int:
    raw = hashlib.sha256(f"{source_sha}:{pair_id}:{direction}:{control_count}".encode()).digest()[:8]
    return int.from_bytes(raw, "big")


def _public_group_result(
    result: Dict[str, Any],
    baseline_result: Dict[str, Any],
    donor_option: int,
    recipient_option: int,
) -> Dict[str, Any]:
    scores = result["answerScores"]
    base_scores = baseline_result["answerScores"]
    baseline_contrast = (
        base_scores[donor_option]["meanTokenLogLikelihood"]
        - base_scores[recipient_option]["meanTokenLogLikelihood"]
    )
    changed_contrast = (
        scores[donor_option]["meanTokenLogLikelihood"]
        - scores[recipient_option]["meanTokenLogLikelihood"]
    )
    contrast_delta = changed_contrast - baseline_contrast
    baseline_prompt_logits = baseline_result["_promptLogits"]
    changed_prompt_logits = result["_promptLogits"]
    target_donor_first = int(scores[donor_option]["answerTokenIds"][0])
    target_recipient_first = int(scores[recipient_option]["answerTokenIds"][0])
    return {
        "groupLabel": result["groupLabel"],
        "donorAnswerMeanLogLikelihoodChange": (
            scores[donor_option]["meanTokenLogLikelihood"]
            - base_scores[donor_option]["meanTokenLogLikelihood"]
        ),
        "recipientAnswerMeanLogLikelihoodChange": (
            scores[recipient_option]["meanTokenLogLikelihood"]
            - base_scores[recipient_option]["meanTokenLogLikelihood"]
        ),
        "baselineDonorVsRecipientAnswerContrast": baseline_contrast,
        "transplantedDonorVsRecipientAnswerContrast": changed_contrast,
        "transferContrastDeltaTowardDonor": contrast_delta,
        "donorAnswerFirstTokenLogitChange": float(
            changed_prompt_logits[target_donor_first] - baseline_prompt_logits[target_donor_first]
        ),
        "recipientAnswerFirstTokenLogitChange": float(
            changed_prompt_logits[target_recipient_first] - baseline_prompt_logits[target_recipient_first]
        ),
        "donorAnswerFirstTokenProbabilityChange": float(
            torch.softmax(changed_prompt_logits, dim=-1)[target_donor_first]
            - torch.softmax(baseline_prompt_logits, dim=-1)[target_donor_first]
        ),
        "recipientAnswerFirstTokenProbabilityChange": float(
            torch.softmax(changed_prompt_logits, dim=-1)[target_recipient_first]
            - torch.softmax(baseline_prompt_logits, dim=-1)[target_recipient_first]
        ),
        "greedyTokenAfterTransferId": int(torch.argmax(changed_prompt_logits)),
        "greedyTokenAfterTransferChanged": int(torch.argmax(changed_prompt_logits))
        != int(torch.argmax(baseline_prompt_logits)),
        "fullDistributionChange": full_distribution_change(baseline_prompt_logits, changed_prompt_logits),
        "magnitudeMatches": result.get("magnitudeMatches", []),
        "patchSiteCount": result["patchSiteCount"],
        "patchedActivationCount": result["patchedActivationCount"],
    }


def _restore_group(
    donor: Dict[str, Any],
    recipient: Dict[str, Any],
    position_map: Dict[int, int],
    model: Any,
) -> Dict[str, Any]:
    entries = _path_entries(donor["case"])
    sites = []
    for entry in entries:
        positions = []
        for source_position in entry["sourcePositions"]:
            recipient_position = position_map[source_position]
            value = _site_value(
                recipient,
                entry["kind"],
                entry["layer"],
                entry["unit"],
                recipient_position,
            )
            positions.append({
                "sourcePosition": source_position,
                "recipientPosition": recipient_position,
                "donorValue": value,
                "recipientOriginalValue": value,
            })
        sites.append({
            "kind": entry["kind"],
            "layer": entry["layer"],
            "unit": entry["unit"],
            "positions": positions,
        })
    return {"label": "recipient-original-activation-restoration", "sites": sites, "magnitudeMatches": []}


def run_bidirectional_transfer(
    pair: Dict[str, Any],
    states: Dict[str, Dict[str, Any]],
    tokenized: Dict[str, Dict[str, Any]],
    slots: Dict[str, Tuple[str, str, Tuple[int, int]]],
    model: Any,
    tokenizer: Any,
    device: torch.device,
    source_sha: str,
    control_count: int,
    magnitude_tolerance: float,
    archive: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    case_a_id, case_b_id = pair["subjectA"], pair["subjectB"]
    directions = []
    for direction, donor_id, recipient_id, donor_slot, recipient_slot, donor_option, recipient_option in (
        ("a-to-b", case_a_id, case_b_id, slots[case_a_id][2], slots[case_b_id][2], 0, 1),
        ("b-to-a", case_b_id, case_a_id, slots[case_b_id][2], slots[case_a_id][2], 1, 0),
    ):
        donor = states[donor_id]
        recipient = states[recipient_id]
        position_map = _recipient_position_map(donor, recipient, donor_slot, recipient_slot, tokenized)
        donor_entries = _path_entries(donor["case"])
        seed = _seed_for_pair(source_sha, pair["pairId"], direction, control_count)
        control_groups, control_metadata = _sample_control_groups(
            donor,
            model,
            position_map,
            control_count,
            magnitude_tolerance,
            seed,
        )
        path_group = _group_for_transfer(donor, recipient, position_map, model)
        base_group = {"label": "unmodified-recipient-baseline", "sites": [], "magnitudeMatches": []}
        restoration_group = _restore_group(donor, recipient, position_map, model)
        identity_group = {
            **restoration_group,
            "label": "recipient-original-activation-identity-control",
        }
        groups = [base_group, identity_group, restoration_group, path_group] + [
            _group_for_control(donor, recipient, control, position_map, model)
            for control in control_groups
        ]
        pair_answer_ids = [
            states[case_a_id]["answerIds"],
            states[case_b_id]["answerIds"],
        ]
        archive_prefix = f"{pair['pairId']}_{direction}"
        evaluated = _evaluate_groups(
            model,
            tokenizer,
            recipient,
            pair_answer_ids,
            groups,
            device,
            archive=archive,
            archive_prefix=archive_prefix,
        )
        base_result = evaluated[0]
        identity_result = evaluated[1]
        restored = evaluated[2]
        selected_result = evaluated[3]
        selected_metrics = _public_group_result(
            selected_result, base_result, donor_option, recipient_option
        )
        controls = [
            _public_group_result(item, base_result, donor_option, recipient_option)
            for item in evaluated[4:]
        ]
        control_effects = [float(item["transferContrastDeltaTowardDonor"]) for item in controls]
        selected_effect = float(selected_metrics["transferContrastDeltaTowardDonor"])
        percentile = 100.0 * (
            sum(value < selected_effect for value in control_effects)
            + 0.5 * sum(value == selected_effect for value in control_effects)
        ) / max(len(control_effects), 1)
        matched_count = sum(
            bool(item["magnitudeMatches"])
            and all(bool(match["withinTolerance"]) for match in item["magnitudeMatches"])
            for item in controls
        )
        valid_control_set = (
            control_metadata["exactStructuralMatch"]
            and control_metadata["uniqueCount"] >= 20
            and matched_count >= math.ceil(control_count * 0.95)
        )
        empirical_p = (
            1 + sum(value >= selected_effect for value in control_effects)
        ) / (len(control_effects) + 1)

        baseline_prompt_logits = base_result["_promptLogits"]
        identity_logits = identity_result["_promptLogits"]
        restored_logits = restored["_promptLogits"]
        recovery_error = float((restored_logits - identity_logits).abs().max())
        identity_hook_error = float((identity_logits - baseline_prompt_logits).abs().max())
        restored_vs_unmodified_error = float((restored_logits - baseline_prompt_logits).abs().max())
        recovery_answer_deltas = [
            restored["answerScores"][index]["totalLogLikelihood"]
            - identity_result["answerScores"][index]["totalLogLikelihood"]
            for index in range(2)
        ]
        identity_answer_deltas = [
            identity_result["answerScores"][index]["totalLogLikelihood"]
            - base_result["answerScores"][index]["totalLogLikelihood"]
            for index in range(2)
        ]
        identity_numerical_floor = (
            restored_vs_unmodified_error <= identity_hook_error + 2e-6
            and all(
                abs(recovery_answer_deltas[index]) <= 2e-5
                and abs(
                    restored["answerScores"][index]["totalLogLikelihood"]
                    - base_result["answerScores"][index]["totalLogLikelihood"]
                )
                <= abs(identity_answer_deltas[index]) + 2e-6
                for index in range(2)
            )
        )
        recovered_original_greedy = (
            restored["answerScores"][0]["greedyNextTokenId"]
            == recipient["baselineSummary"]["greedyNextToken"]["tokenId"]
        )
        patch_values = []
        for site in path_group["sites"]:
            for position in site["positions"]:
                value = position["donorValue"]
                patch_values.append(value.reshape(-1))
        archive[f"{archive_prefix}_donor_patch_values"] = (
            _as_numpy(torch.cat(patch_values)) if patch_values else np.asarray([], dtype=np.float32)
        )
        recipient_patch_values = [
            position["recipientOriginalValue"].reshape(-1)
            for site in path_group["sites"]
            for position in site["positions"]
        ]
        archive[f"{archive_prefix}_recipient_original_patch_values"] = (
            _as_numpy(torch.cat(recipient_patch_values))
            if recipient_patch_values
            else np.asarray([], dtype=np.float32)
        )
        control_donor_values = []
        control_recipient_values = []
        for control_group in groups[4:]:
            control_donor_values.append(torch.cat([
                position["donorValue"].reshape(-1)
                for site in control_group["sites"]
                for position in site["positions"]
            ]))
            control_recipient_values.append(torch.cat([
                position["recipientOriginalValue"].reshape(-1)
                for site in control_group["sites"]
                for position in site["positions"]
            ]))
        archive[f"{archive_prefix}_control_donor_patch_values"] = _as_numpy(
            torch.stack(control_donor_values)
        )
        archive[f"{archive_prefix}_control_recipient_original_patch_values"] = _as_numpy(
            torch.stack(control_recipient_values)
        )
        archive[f"{archive_prefix}_control_transfer_contrast_delta"] = np.asarray(
            control_effects, dtype=np.float32
        )
        archive[f"{archive_prefix}_control_magnitude_match"] = np.asarray([
            all(match["withinTolerance"] for match in item["magnitudeMatches"])
            for item in controls
        ], dtype=np.bool_)
        directions.append({
            "direction": direction,
            "donorSubject": donor_id,
            "recipientSubject": recipient_id,
            "donorPath": donor["case"]["orderedResidualPath"]["nodes"],
            "recipientPath": recipient["case"]["orderedResidualPath"]["nodes"],
            "positionMap": {str(key): value for key, value in position_map.items()},
            "activationTransplant": {
                "method": "copy donor attention c_proj input head slices and donor MLP post-GELU neuron activations into matching component coordinates at exactly aligned fixed-context token positions",
                "patchedComponentCount": len(path_group["sites"]),
                "patchedActivationCount": sum(len(site["positions"]) for site in path_group["sites"]),
                "componentActivationChanges": [
                    {
                        "type": site["kind"],
                        "layer": site["layer"],
                        "unit": site["unit"],
                        "positions": [
                            {
                                "donorPosition": position["sourcePosition"],
                                "recipientPosition": position["recipientPosition"],
                                "donorActivationL2OrAbs": float(torch.linalg.vector_norm(position["donorValue"]).cpu())
                                if position["donorValue"].ndim > 0
                                else abs(float(position["donorValue"].cpu())),
                                "recipientOriginalActivationL2OrAbs": float(torch.linalg.vector_norm(position["recipientOriginalValue"]).cpu())
                                if position["recipientOriginalValue"].ndim > 0
                                else abs(float(position["recipientOriginalValue"].cpu())),
                            }
                            for position in site["positions"]
                        ],
                    }
                    for site in path_group["sites"]
                ],
                "rawDonorPatchValuesKey": f"{archive_prefix}_donor_patch_values",
                "rawRecipientOriginalPatchValuesKey": (
                    f"{archive_prefix}_recipient_original_patch_values"
                ),
                "successful": selected_result["patchedActivationCount"] > 0,
            },
            "baselineRecipientAnswerScores": base_result["answerScores"],
            "transplantedAnswerScores": selected_result["answerScores"],
            "pathTransfer": selected_metrics,
            "matchedControls": controls,
            "controlSampling": control_metadata,
            "specificity": {
                "empiricalPercentile": percentile,
                "controlRightTailPValue": empirical_p,
                "sampleCount": len(controls),
                "uniqueControlCount": control_metadata["uniqueCount"],
                "magnitudeMatchedControlCount": matched_count,
                "magnitudeMatchCoverage": matched_count / max(len(controls), 1),
                "validStructuralAndMagnitudeControls": valid_control_set,
                "passesPathSpecificityComparison": valid_control_set and percentile >= 95.0,
                "thresholds": {
                    "percentile": 95.0,
                    "minimumMagnitudeMatchedControls": math.ceil(control_count * 0.95),
                    "minimumUniqueGroups": 20,
                    "relativeMagnitudeTolerance": magnitude_tolerance,
                },
            },
            "recovery": {
                "method": "compare exact activation restoration against an independently installed identity-activation control at the same sites",
                "maxPromptLogitRecoveryError": recovery_error,
                "answerSequenceLogLikelihoodDeltas": recovery_answer_deltas,
                "maxIdentityHookVsUnmodifiedPromptLogitError": identity_hook_error,
                "maxRestoredVsUnmodifiedPromptLogitError": restored_vs_unmodified_error,
                "identityHookVsUnmodifiedAnswerSequenceLogLikelihoodDeltas": identity_answer_deltas,
                "withinIdentityInterventionNumericalFloor": identity_numerical_floor,
                "recoveredGreedyTokenId": restored["answerScores"][0]["greedyNextTokenId"],
                "baselineGreedyTokenId": recipient["baselineSummary"]["greedyNextToken"]["tokenId"],
                "recoveredOriginalGreedyPrediction": recovered_original_greedy,
                "recoveryVerified": (
                    recovery_error <= 2e-5
                    and identity_numerical_floor
                    and recovered_original_greedy
                ),
            },
            "rawControlContrastArrayKey": f"{archive_prefix}_control_transfer_contrast_delta",
            "rawControlMagnitudeMatchArrayKey": f"{archive_prefix}_control_magnitude_match",
            "rawControlDonorPatchValuesKey": f"{archive_prefix}_control_donor_patch_values",
            "rawControlRecipientOriginalPatchValuesKey": (
                f"{archive_prefix}_control_recipient_original_patch_values"
            ),
            "rawIdentityPromptLogitsKey": f"{archive_prefix}_identity_prompt_logits",
            "rawRecoveryPromptLogitsKey": f"{archive_prefix}_restored_prompt_logits",
            "rawBaselinePromptLogitsKey": f"{archive_prefix}_baseline_prompt_logits",
        })
    return {
        "pairId": pair["pairId"],
        "relationFamily": pair["relationFamily"],
        "subjectA": case_a_id,
        "subjectB": case_b_id,
        "alignment": pair["alignment"],
        "aToB": directions[0],
        "bToA": directions[1],
        "explanation": _explain_pair(pair, directions[0], directions[1]),
    }


def _explain_pair(pair: Dict[str, Any], a_to_b: Dict[str, Any], b_to_a: Dict[str, Any]) -> str:
    def sentence(direction: Dict[str, Any]) -> str:
        transfer = direction["pathTransfer"]
        specificity = direction["specificity"]
        recovery = direction["recovery"]
        return (
            f"{direction['donorSubject']} → {direction['recipientSubject']}: donor-vs-recipient "
            f"answer mean-log-likelihood contrast changed by "
            f"{transfer['transferContrastDeltaTowardDonor']:.6f}; donor answer likelihood changed "
            f"by {transfer['donorAnswerMeanLogLikelihoodChange']:.6f}; empirical control percentile "
            f"{specificity['empiricalPercentile']:.1f}, magnitude-valid controls "
            f"{specificity['magnitudeMatchedControlCount']}/{specificity['sampleCount']}; "
            f"recipient recovery error {recovery['maxPromptLogitRecoveryError']:.3g}."
        )
    return (
        f"## {pair['subjectA']} ↔ {pair['subjectB']} ({pair['relationFamily']})\n\n"
        f"**Alignment:** exact fixed-prefix/suffix token alignment; the variable fact slot is never patched. "
        f"**A→B:** {sentence(a_to_b)} "
        f"**B→A:** {sentence(b_to_a)} "
        "Transfer is measured as the change in donor-answer versus recipient-answer mean token log-likelihood. "
        "A positive effect is evidence of directional information transfer, not proof that the transplanted "
        "path is a complete sufficient circuit. If activation-magnitude control coverage is inadequate, "
        "specificity remains unresolved."
    )


def _fact_removal_groups(
    state: Dict[str, Any],
    model: Any,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    entries = _path_entries(state["case"])
    positions = {position: position for entry in entries for position in entry["sourcePositions"]}
    selected = _make_transfer_group(
        state,
        state,
        entries,
        positions,
        model,
        "selected-path-remove",
    )
    zero_group = {
        **selected,
        "label": "selected-path-group-zero-ablation",
        "sites": [
            {
                **site,
                "positions": [
                    {**position, "donorValue": None}
                    for position in site["positions"]
                ],
            }
            for site in selected["sites"]
        ],
    }
    singles = []
    for index in range(len(entries)):
        single = _make_transfer_group(
            state,
            state,
            [entries[index]],
            positions,
            model,
            f"component-{index}-zero-ablation",
        )
        single["sites"] = [
            {**site, "positions": [{**position, "donorValue": None} for position in site["positions"]]}
            for site in single["sites"]
        ]
        singles.append(single)
    restore = _restore_group(state, state, positions, model)
    return [zero_group, *singles, restore], entries


def run_fact_necessity(
    state: Dict[str, Any],
    model: Any,
    tokenizer: Any,
    device: torch.device,
    archive: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    case = state["case"]
    groups, entries = _fact_removal_groups(state, model)
    baseline = {"label": "unmodified", "sites": [], "magnitudeMatches": []}
    evaluated = _evaluate_groups(
        model,
        tokenizer,
        state,
        [state["answerIds"]],
        [baseline, *groups],
        device,
    )
    baseline_result = evaluated[0]
    group_result = evaluated[1]
    component_results = evaluated[2 : 2 + len(entries)]
    restored = evaluated[-1]
    baseline_score = baseline_result["answerScores"][0]
    group_score = group_result["answerScores"][0]
    component_scores = [result["answerScores"][0] for result in component_results]
    group_loglik_drop = baseline_score["totalLogLikelihood"] - group_score["totalLogLikelihood"]
    component_loglik_drops = [
        baseline_score["totalLogLikelihood"] - item["totalLogLikelihood"]
        for item in component_scores
    ]
    interaction = group_loglik_drop - sum(component_loglik_drops)
    baseline_logits = baseline_result["_promptLogits"]
    ablated_logits = group_result["_promptLogits"]
    restored_logits = restored["_promptLogits"]
    archive_prefix = state["prefix"]
    archive[f"{archive_prefix}_path_removed_prompt_logits"] = _as_numpy(ablated_logits)
    archive[f"{archive_prefix}_path_restored_prompt_logits"] = _as_numpy(restored_logits)
    for index, result in enumerate(component_results):
        archive[f"{archive_prefix}_component_{index}_removed_prompt_logits"] = _as_numpy(
            result["_promptLogits"]
        )
    return {
        "subject": str(case["subject"]),
        "sourceAnswer": str(case["answer"]),
        "baselineAnswerSequenceScore": baseline_score,
        "pathGroupAblation": {
            "answerMeanTokenLogLikelihoodDrop": (
                baseline_score["meanTokenLogLikelihood"] - group_score["meanTokenLogLikelihood"]
            ),
            "answerTotalLogLikelihoodDrop": group_loglik_drop,
            "greedyTokenAfterAblationId": group_score["greedyNextTokenId"],
            "baselineGreedyTokenId": baseline_score["greedyNextTokenId"],
            "baselineGreedyPredictionChanged": (
                group_score["greedyNextTokenId"] != baseline_score["greedyNextTokenId"]
            ),
            "sourceAnswerWasGreedyBeforeAblation": (
                int(case["target"]["tokenId"]) == int(baseline_score["greedyNextTokenId"])
            ),
            "sourceAnswerRemainsGreedyAfterAblation": (
                int(case["target"]["tokenId"]) == int(group_score["greedyNextTokenId"])
            ),
            "strictNecessityForBaselineCorrectGreedyPrediction": (
                int(case["target"]["tokenId"]) == int(baseline_score["greedyNextTokenId"])
                and int(case["target"]["tokenId"]) != int(group_score["greedyNextTokenId"])
            ),
            "sourceAnswerFirstTokenLogitDrop": float(
                baseline_logits[int(case["target"]["tokenId"])] - ablated_logits[int(case["target"]["tokenId"])]
            ),
            "measurablePositiveAnswerLikelihoodSupport": group_loglik_drop > 1e-6,
        },
        "individualComponentAblations": [
            {
                "componentType": entry["kind"],
                "layer": entry["layer"],
                "unit": entry["unit"],
                "sourcePositions": entry["sourcePositions"],
                "answerTotalLogLikelihoodDrop": component_loglik_drops[index],
                "sourceAnswerFirstTokenLogitDrop": float(
                    baseline_logits[int(case["target"]["tokenId"])]
                    - component_results[index]["_promptLogits"][int(case["target"]["tokenId"])]
                ),
                "greedyTokenAfterAblationId": component_scores[index]["greedyNextTokenId"],
            }
            for index, entry in enumerate(entries)
        ],
        "groupMinusSumIndividualAnswerLogLikelihoodDrop": interaction,
        "interactionInterpretation": (
            "approximately-additive"
            if abs(interaction) <= 0.1 * max(abs(sum(component_loglik_drops)), 1e-8)
            else "subadditive-overlap-consistent"
            if interaction < 0
            else "superadditive-synergy-consistent"
        ),
        "recipientOriginalActivationRestoration": {
            "maxPromptLogitRecoveryError": float((restored_logits - baseline_logits).abs().max()),
            "answerTotalLogLikelihoodDelta": (
                restored["answerScores"][0]["totalLogLikelihood"] - baseline_score["totalLogLikelihood"]
            ),
            "recoveryVerified": (
                float((restored_logits - baseline_logits).abs().max()) <= 2e-5
                and abs(
                    restored["answerScores"][0]["totalLogLikelihood"]
                    - baseline_score["totalLogLikelihood"]
                ) <= 2e-4
            ),
        },
    }


def _bootstrap_family_statistics(
    family: str,
    records: Sequence[Dict[str, Any]],
    replicate_count: int,
) -> Dict[str, Any]:
    if not records:
        return {"relationFamily": family, "factCount": 0, "directionCount": 0}
    all_subjects = sorted({
        record["donorSubject"] for record in records
    } | {
        record["recipientSubject"] for record in records
    })
    valid_records = [
        record for record in records if record["specificityControlsValid"]
    ]

    def estimate(
        selected_records: Sequence[Dict[str, Any]],
        values: np.ndarray,
        label: str,
    ) -> Dict[str, Any]:
        subjects = sorted({
            record["donorSubject"] for record in selected_records
        } | {
            record["recipientSubject"] for record in selected_records
        })
        mean = float(values.mean()) if len(values) else None
        if len(subjects) < 10 or not selected_records:
            return {
                "mean": mean,
                "ci": None,
                "p": None,
                "bootstrapReplicates": 0,
                "factCount": len(subjects),
                "directionCount": len(selected_records),
            }
        subject_index = {subject: index for index, subject in enumerate(subjects)}
        rng = np.random.default_rng(
            int.from_bytes(hashlib.sha256(f"{family}:{label}".encode()).digest()[:8], "big")
        )
        estimates = []
        for _ in range(replicate_count):
            sampled = rng.integers(0, len(subjects), size=len(subjects))
            weights = np.bincount(sampled, minlength=len(subjects))
            pair_weights = np.asarray([
                weights[subject_index[record["donorSubject"]]]
                * weights[subject_index[record["recipientSubject"]]]
                for record in selected_records
            ], dtype=np.float64)
            total_weight = float(pair_weights.sum())
            if total_weight:
                estimates.append(float(np.dot(pair_weights, values) / total_weight))
        if not estimates:
            return {
                "mean": mean,
                "ci": None,
                "p": None,
                "bootstrapReplicates": 0,
                "factCount": len(subjects),
                "directionCount": len(selected_records),
            }
        bootstrap = np.asarray(estimates)
        lower, upper = np.quantile(bootstrap, [0.025, 0.975])
        centered = bootstrap - mean
        p_value = min(
            1.0,
            2.0 * min(
                (int(np.sum(centered <= -abs(mean))) + 1) / (len(centered) + 1),
                (int(np.sum(centered >= abs(mean))) + 1) / (len(centered) + 1),
            ),
        )
        return {
            "mean": mean,
            "ci": [float(lower), float(upper)],
            "p": float(p_value),
            "bootstrapReplicates": len(bootstrap),
            "factCount": len(subjects),
            "directionCount": len(selected_records),
        }

    transfer_values = np.asarray([
        float(record["pathTransfer"]["transferContrastDeltaTowardDonor"])
        for record in records
    ], dtype=np.float64)
    donor_likelihood_values = np.asarray([
        float(record["pathTransfer"]["donorAnswerMeanLogLikelihoodChange"])
        for record in records
    ], dtype=np.float64)
    transfer = estimate(records, transfer_values, "path-transfer")
    donor_likelihood = estimate(records, donor_likelihood_values, "donor-answer-likelihood")
    control_differences = np.asarray([
        float(record["pathTransfer"]["transferContrastDeltaTowardDonor"])
        - float(record["meanControlTransferContrastDelta"])
        for record in valid_records
    ], dtype=np.float64)
    control = estimate(valid_records, control_differences, "path-minus-valid-controls")
    valid_control_count = len(valid_records)
    return {
        "relationFamily": family,
        "factCount": len(all_subjects),
        "directionCount": len(records),
        "validControlDirectionCount": valid_control_count,
        "invalidControlDirectionCount": len(records) - valid_control_count,
        "meanPathTransferContrastDelta": transfer["mean"],
        "pathTransferContrastDelta95PercentClusterBootstrapCI": transfer["ci"],
        "pathTransferTwoSidedCenteredClusterBootstrapPValue": transfer["p"],
        "pathTransferBootstrapReplicatesUsed": transfer["bootstrapReplicates"],
        "meanDonorAnswerMeanTokenLogLikelihoodChange": donor_likelihood["mean"],
        "donorAnswerMeanLogLikelihoodChange95PercentClusterBootstrapCI": donor_likelihood["ci"],
        "meanPathMinusMeanControlTransferContrastDelta": control["mean"],
        "clusterBootstrap95PercentCI": control["ci"],
        "twoSidedCenteredClusterBootstrapPValue": control["p"],
        "bootstrapReplicatesUsed": control["bootstrapReplicates"],
        "independentClusterUnit": "source fact endpoint; directed pair observations are weighted by donor- and recipient-cluster multiplicities",
        "inferentialStatus": (
            "estimated"
            if control["p"] is not None
            else "descriptive-only-invalid-controls"
            if not valid_records
            else "descriptive-only-too-few-valid-control-facts"
        ),
        "pathTransferInferentialStatus": (
            "estimated"
            if transfer["p"] is not None
            else "descriptive-only-too-few-independent-facts"
        ),
        "allDirectionsUnvalidatedPathMinusControlDescriptiveMean": float(np.mean([
            float(record["pathTransfer"]["transferContrastDeltaTowardDonor"])
            - float(record["meanControlTransferContrastDelta"])
            for record in records
        ])),
    }


def _benjamini_hochberg(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    order = np.argsort(np.asarray(values))
    adjusted = np.ones(len(values), dtype=np.float64)
    running = 1.0
    for reverse_rank, index in enumerate(order[::-1], start=1):
        rank = len(values) - reverse_rank + 1
        running = min(running, float(values[index]) * len(values) / rank)
        adjusted[index] = running
    return adjusted.tolist()


def _build_family_analysis(
    pair_records: Sequence[Dict[str, Any]],
    bootstrap_replicates: int,
) -> Dict[str, Any]:
    directions_by_family: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for pair in pair_records:
        for key in ("aToB", "bToA"):
            direction = pair[key]
            control_values = [
                float(item["transferContrastDeltaTowardDonor"])
                for item in direction["matchedControls"]
            ]
            mean_control = float(np.mean(control_values)) if control_values else 0.0
            directions_by_family[pair["relationFamily"]].append({
                "pairId": pair["pairId"],
                "donorSubject": direction["donorSubject"],
                "recipientSubject": direction["recipientSubject"],
                "pathTransfer": direction["pathTransfer"],
                "meanControlTransferContrastDelta": mean_control,
                "specificityControlsValid": direction["specificity"]["validStructuralAndMagnitudeControls"],
                "passesPathSpecificityComparison": direction["specificity"]["passesPathSpecificityComparison"],
            })
    family_results = [
        _bootstrap_family_statistics(family, records, bootstrap_replicates)
        for family, records in sorted(directions_by_family.items())
    ]
    eligible_indexes = [
        index for index, result in enumerate(family_results)
        if result.get("twoSidedCenteredClusterBootstrapPValue") is not None
    ]
    q_values = _benjamini_hochberg([
        float(family_results[index]["twoSidedCenteredClusterBootstrapPValue"])
        for index in eligible_indexes
    ])
    for index, q_value in zip(eligible_indexes, q_values):
        family_results[index]["benjaminiHochbergQValue"] = q_value
        family_results[index]["significantAfterMultipleComparisonCorrection"] = q_value < STATISTICAL_ALPHA
    transfer_eligible_indexes = [
        index for index, result in enumerate(family_results)
        if result.get("pathTransferTwoSidedCenteredClusterBootstrapPValue") is not None
    ]
    transfer_q_values = _benjamini_hochberg([
        float(family_results[index]["pathTransferTwoSidedCenteredClusterBootstrapPValue"])
        for index in transfer_eligible_indexes
    ])
    for index, q_value in zip(transfer_eligible_indexes, transfer_q_values):
        family_results[index]["pathTransferBenjaminiHochbergQValue"] = q_value
        family_results[index]["pathTransferSignificantAfterMultipleComparisonCorrection"] = (
            q_value < STATISTICAL_ALPHA
        )
    for result in family_results:
        result.setdefault("benjaminiHochbergQValue", None)
        result.setdefault("significantAfterMultipleComparisonCorrection", None)
        result.setdefault("pathTransferBenjaminiHochbergQValue", None)
        result.setdefault("pathTransferSignificantAfterMultipleComparisonCorrection", None)
    return {
        "relationFamilies": family_results,
        "multipleComparisonCorrection": {
            "method": "Benjamini-Hochberg",
            "controlContrastAdjustmentSet": [
                result["relationFamily"] for result in family_results
                if result["twoSidedCenteredClusterBootstrapPValue"] is not None
            ],
            "pathTransferAdjustmentSet": [
                result["relationFamily"] for result in family_results
                if result["pathTransferTwoSidedCenteredClusterBootstrapPValue"] is not None
            ],
            "alpha": STATISTICAL_ALPHA,
            "bootstrapReplicates": bootstrap_replicates,
            "dependentPairHandling": "cluster bootstrap resamples source facts and weights directed pair observations by endpoint multiplicities; pair directions are not assumed independent",
        },
    }


def _classify_computation(
    pair_records: Sequence[Dict[str, Any]],
    necessity_records: Sequence[Dict[str, Any]],
    statistics: Dict[str, Any],
) -> Dict[str, Any]:
    directions = [pair[key] for pair in pair_records for key in ("aToB", "bToA")]
    valid_specificity = sum(
        direction["specificity"]["passesPathSpecificityComparison"]
        for direction in directions
    )
    valid_control_directions = sum(
        direction["specificity"]["validStructuralAndMagnitudeControls"]
        for direction in directions
    )
    positive_transfer = sum(
        direction["pathTransfer"]["transferContrastDeltaTowardDonor"] > 0
        for direction in directions
    )
    necessity_support = sum(
        record["pathGroupAblation"]["measurablePositiveAnswerLikelihoodSupport"]
        for record in necessity_records
    )
    necessity_strict = sum(
        record["pathGroupAblation"]["strictNecessityForBaselineCorrectGreedyPrediction"]
        for record in necessity_records
    )
    large_family = [
        family for family in statistics["relationFamilies"]
        if family["inferentialStatus"] == "estimated"
    ]
    specific_inference = any(
        family.get("significantAfterMultipleComparisonCorrection") is True
        and family["meanPathMinusMeanControlTransferContrastDelta"] > 0
        for family in large_family
    )
    if specific_inference and valid_specificity:
        interpretation = "fact-specific-transfer-evidence-in-at-least-one-family"
    elif necessity_support and not specific_inference:
        interpretation = "generic-or-shared-predictive-influence-consistent; fact-specific-transfer-not-established"
    else:
        interpretation = "available-evidence-does-not-distinguish-shared-relation-from-generic-or-fact-specific-computation"
    return {
        "interpretation": interpretation,
        "factSpecificInformation": {
            "evidence": "path transfer changes donor-vs-recipient answer likelihood contrast relative to the recipient baseline and matched structural controls",
            "positiveDirectionalTransfers": positive_transfer,
            "directionCount": len(directions),
            "strictSpecificityPassingDirections": valid_specificity,
            "validMagnitudeAndStructuralControlDirections": valid_control_directions,
            "claim": "not established unless a relation-family effect survives the cluster-aware CI and multiple-comparison procedure with valid magnitude controls",
        },
        "sharedRelationInformation": {
            "evidenceAvailable": "repeated prompt families and component/group interventions",
            "interpretation": "shared processing is not semantically identified by unit overlap or activation correlation alone",
            "claim": "not distinguished from generic prediction without selective transfer against valid controls",
        },
        "generalPredictiveInfluence": {
            "factsWithPositiveAnswerLikelihoodDropUnderPathRemoval": necessity_support,
            "factsWithStrictCorrectGreedyPredictionNecessity": necessity_strict,
            "factCount": len(necessity_records),
            "claim": "ablation influence is separate from necessity and transfer sufficiency",
        },
        "distributedOrRedundantComputation": {
            "evidence": "per-component removal, group removal, and group-minus-sum-of-individual answer-log-likelihood effects",
            "perFactInteractionValuesRecorded": len(necessity_records),
            "claim": "interaction and redundancy are not equivalent; inspect raw component and group effects",
        },
        "unresolved": [
            "This instrumented experiment does not establish black-box decompilation.",
            "Counterfactual path transfer is not complete circuit sufficiency.",
            "Insufficiently magnitude-matched controls leave path specificity unresolved.",
            "Only source-represented relation families can be analyzed; the large capital family dominates statistical power.",
        ],
    }


def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    source_path = args.source_report.resolve()
    output_dir = args.output_dir.resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Frozen source report does not exist: {source_path}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty experiment output directory: {output_dir}")
    verify_frozen_baselines()
    if args.control_count != CONTROL_COUNT:
        raise ValueError(f"Control count is frozen at {CONTROL_COUNT}; got {args.control_count}.")
    if not math.isclose(args.control_magnitude_tolerance, CONTROL_MAGNITUDE_TOLERANCE):
        raise ValueError(
            f"Control magnitude tolerance is frozen at {CONTROL_MAGNITUDE_TOLERANCE}; "
            f"got {args.control_magnitude_tolerance}."
        )
    if args.pilot_pairs <= 0 or args.bootstrap_replicates < 100:
        raise ValueError("Pilot pair count must be positive and bootstrap replicates must be >= 100.")
    source_sha = sha256_file(source_path)
    if source_sha != EXPECTED_SOURCE_SHA256:
        raise ValueError("Frozen 50-fact source artifact hash differs from the registered baseline.")
    source_report = json.loads(source_path.read_text(encoding="utf-8"))
    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for the counterfactual experiment but is unavailable.")
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True).eval().to(device)
    model_load_seconds = time.perf_counter() - started
    if model.__class__.__name__ != "GPT2LMHeadModel":
        raise TypeError(f"Expected GPT2LMHeadModel, got {model.__class__.__name__}.")
    inventory = build_pair_inventory(source_report, tokenizer)
    all_pairs = inventory["pairs"]
    if len(all_pairs) < 10:
        selected_pairs = all_pairs
    elif args.variant == "pilot":
        selected_pairs = all_pairs[: min(args.pilot_pairs, len(all_pairs))]
    else:
        selected_pairs = all_pairs
    if args.variant == "all-pairs" and len(selected_pairs) != len(all_pairs):
        raise RuntimeError("all-pairs mode did not select the complete eligible pair inventory.")
    archive: Dict[str, np.ndarray] = {}
    states, fact_records = capture_original_facts(
        inventory["cases"],
        inventory["tokenizedPrompts"],
        inventory["answerTokenIds"],
        model,
        tokenizer,
        device,
        archive,
    )
    necessity_records = [
        run_fact_necessity(state, model, tokenizer, device, archive)
        for state in states.values()
    ]
    trace_started = time.perf_counter()
    pair_records = [
        run_bidirectional_transfer(
            pair,
            states,
            inventory["tokenizedPrompts"],
            inventory["slots"],
            model,
            tokenizer,
            device,
            source_sha,
            args.control_count,
            args.control_magnitude_tolerance,
            archive,
        )
        for pair in selected_pairs
    ]
    trace_seconds = time.perf_counter() - trace_started
    statistics = _build_family_analysis(pair_records, args.bootstrap_replicates)
    computation_analysis = _classify_computation(pair_records, necessity_records, statistics)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        gpu_info = {
            "name": torch.cuda.get_device_name(device),
            "totalMemoryBytes": int(torch.cuda.get_device_properties(device).total_memory),
            "peakAllocatedBytes": int(torch.cuda.max_memory_allocated(device)),
            "peakReservedBytes": int(torch.cuda.max_memory_reserved(device)),
        }
    else:
        gpu_info = None
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / "raw-counterfactual.npz"
    np.savez_compressed(archive_path, **archive)
    archive_metadata = {
        "file": archive_path.name,
        "sha256": sha256_file(archive_path),
        "arrayCount": len(archive),
        "arrays": {
            key: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for key, value in sorted(archive.items())
        },
    }
    criteria = {
        "A_counterfactualPromptAlignment": {
            "eligiblePairCount": len(all_pairs),
            "testedPairCount": len(selected_pairs),
            "allTestedPairsPreciselyAligned": all(
                pair["alignment"]["fixedContextAlignment"].startswith("exact")
                for pair in pair_records
            ),
            "excludedFactCount": inventory["inventory"]["excludedFactCount"],
            "rejectedCandidatePairCount": inventory["inventory"]["rejectedCandidatePairCount"],
        },
        "B_successfulActivationTransplantation": {
            "successfulDirections": sum(
                bool(pair[key]["activationTransplant"]["successful"])
                for pair in pair_records for key in ("aToB", "bToA")
            ),
            "directionCount": len(pair_records) * 2,
        },
        "C_directionalAToBTransfer": {
            "positiveDonorContrastDirections": sum(
                pair["aToB"]["pathTransfer"]["transferContrastDeltaTowardDonor"] > 0
                for pair in pair_records
            ),
            "directionCount": len(pair_records),
        },
        "D_reverseBToATransfer": {
            "positiveDonorContrastDirections": sum(
                pair["bToA"]["pathTransfer"]["transferContrastDeltaTowardDonor"] > 0
                for pair in pair_records
            ),
            "directionCount": len(pair_records),
        },
        "E_necessity": {
            "factsWithPositiveAnswerLikelihoodDrop": sum(
                record["pathGroupAblation"]["measurablePositiveAnswerLikelihoodSupport"]
                for record in necessity_records
            ),
            "factsWithStrictCorrectGreedyNecessity": sum(
                record["pathGroupAblation"]["strictNecessityForBaselineCorrectGreedyPrediction"]
                for record in necessity_records
            ),
            "factCount": len(necessity_records),
        },
        "F_counterfactualTransferSufficiency": {
            "directionsWithPositiveDonorAnswerLikelihoodChange": sum(
                pair[key]["pathTransfer"]["donorAnswerMeanLogLikelihoodChange"] > 0
                for pair in pair_records for key in ("aToB", "bToA")
            ),
            "directionCount": len(pair_records) * 2,
            "interpretation": "directional transfer evidence only; not a claim of a complete sufficient circuit",
        },
        "G_recovery": {
            "verifiedDirections": sum(
                bool(pair[key]["recovery"]["recoveryVerified"])
                for pair in pair_records for key in ("aToB", "bToA")
            ),
            "directionCount": len(pair_records) * 2,
        },
        "H_specificityAgainstStructuralControls": {
            "directionsPassingStrictSpecificity": sum(
                bool(pair[key]["specificity"]["passesPathSpecificityComparison"])
                for pair in pair_records for key in ("aToB", "bToA")
            ),
            "directionsWithValidMagnitudeControls": sum(
                bool(pair[key]["specificity"]["validStructuralAndMagnitudeControls"])
                for pair in pair_records for key in ("aToB", "bToA")
            ),
            "directionCount": len(pair_records) * 2,
        },
        "I_sharedVsFactSpecificComputation": {
            "relationFamilyAnalyses": len(statistics["relationFamilies"]),
            "familiesWithCorrectedPositiveSpecificityEvidence": sum(
                family.get("significantAfterMultipleComparisonCorrection") is True
                and family["meanPathMinusMeanControlTransferContrastDelta"] > 0
                for family in statistics["relationFamilies"]
            ),
            "interpretation": computation_analysis["interpretation"],
        },
    }
    report = {
        "schemaVersion": 1,
        "kind": "gpt2-counterfactual-circuit-transfer",
        "protocol": "paired-factual-path-activation-transplant-and-sufficiency-witness",
        "accessMode": "instrumented-witness",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "variant": args.variant,
        "model": {
            "id": args.model,
            "class": model.__class__.__name__,
            "device": str(device),
            "config": model.config.to_dict(),
            "tokenizerClass": tokenizer.__class__.__name__,
            "tokenizerNameOrPath": getattr(tokenizer, "name_or_path", None),
        },
        "sourceArtifact": {
            "path": str(source_path),
            "sha256": source_sha,
            "passedFactCount": inventory["inventory"]["passedSourceFactCount"],
        },
        "previousAutoregressiveArtifacts": {
            "directory": str(BASELINE_DIR),
            "expectedHashes": BASELINE_HASHES,
        },
        "pairInventory": inventory["inventory"],
        "protocolConfig": {
            "pilotPairLimit": args.pilot_pairs,
            "selectedPairCount": len(selected_pairs),
            "eligiblePairCount": len(all_pairs),
            "controlCountPerDirection": args.control_count,
            "controlMagnitudeRelativeTolerance": args.control_magnitude_tolerance,
            "controlFallback": "expand to nearest magnitude candidates only to create unique structural controls; mark mismatch and fail strict specificity when <95% of groups fully match",
            "activationMagnitude": "attention c_proj input head-slice L2 norm and MLP post-GELU neuron absolute activation, averaged across the source path positions",
            "fixedTextAlignment": "map only tokens in identical static prompt prefix/suffix with identical tokenizer IDs; never patch variable-slot tokens",
            "answerScoring": "teacher-forced full recorded answer token sequence; prompt/answer token boundary must exactly reproduce",
            "sufficiencyMeaning": "counterfactual directional information transfer, not complete circuit sufficiency",
            "statisticalAlpha": STATISTICAL_ALPHA,
            "bootstrapReplicates": args.bootstrap_replicates,
            "multipleComparisons": "Benjamini-Hochberg across relation-family cluster-bootstrap tests with at least 10 independent facts",
            "randomSeedDerivation": "first 64 bits SHA256(sourceArtifactSha256:pairId:direction:controlCount)",
        },
        "runtime": {
            "pythonVersion": sys.version.split()[0],
            "numpyVersion": np.__version__,
            "torchVersion": torch.__version__,
            "transformersVersion": _package_version("transformers"),
            "modelLoadSeconds": model_load_seconds,
            "traceAndInterventionSeconds": trace_seconds,
            "totalSeconds": time.perf_counter() - started,
            "cudaWasAvailable": bool(torch.cuda.is_available()),
            "gpu": gpu_info,
        },
        "rawArchive": archive_metadata,
        "originalFacts": fact_records,
        "necessityByFact": necessity_records,
        "pairs": pair_records,
        "pairedStatistics": statistics,
        "computationInterpretation": computation_analysis,
        "acceptanceCriteria": criteria,
        "claimBoundary": {
            "establishes": "For tested frozen GPT-2 prompts, records original activations and measures path removal, bidirectional exact-position activation transplantation, structural control transplantation, and recipient activation recovery.",
            "doesNotEstablish": "This hook-based local-weight experiment is not black-box decompilation. Transfer evidence is not complete circuit sufficiency. Invalid or amplitude-unmatched controls cannot establish specificity. Relation semantics are not assigned to individual neurons.",
        },
    }
    report_json = output_dir / "report.json"
    report_json.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    report_md = output_dir / "report.md"
    write_markdown(report, report_md)
    return report


def write_markdown(report: Dict[str, Any], path: Path) -> None:
    inventory = report["pairInventory"]
    lines = [
        "# GPT-2 counterfactual circuit transfer",
        "",
        f"- Variant: `{report['variant']}`",
        f"- Access mode: `{report['accessMode']}`",
        f"- Model/device: `{report['model']['class']}` on `{report['model']['device']}`",
        f"- Frozen source SHA-256: `{report['sourceArtifact']['sha256']}`",
        f"- Eligible factual pairs: {inventory['eligibleUnorderedPairCount']}",
        f"- Tested pairs: {report['protocolConfig']['selectedPairCount']}",
        f"- Control transplants per direction: {report['protocolConfig']['controlCountPerDirection']}",
        "",
        "## Pair eligibility",
        "",
        "| Relation family | Facts | Eligible unordered pairs |",
        "|---|---:|---:|",
    ]
    for family in inventory["relationFamilies"]:
        lines.append(
            f"| {family['relationFamily']} | {family['factCount']} | "
            f"{family['eligibleUnorderedPairCount']} |"
        )
    lines.extend([
        "",
        f"Excluded facts: {inventory['excludedFactCount']}; rejected candidate pairs: "
        f"{inventory['rejectedCandidatePairCount']}. Exclusion reasons are listed in `report.json`.",
        "",
        "## Acceptance evidence",
        "",
        "| Criterion | Measured result |",
        "|---|---|",
    ])
    for key, value in report["acceptanceCriteria"].items():
        lines.append(f"| {key} | `{json.dumps(value, sort_keys=True)}` |")
    lines.extend([
        "",
        "## Paired statistical analysis",
        "",
        f"Bootstrap unit: source factual endpoints; pair directions are not treated as independent. "
        f"Correction: {report['pairedStatistics']['multipleComparisonCorrection']['method']} "
        f"over control-contrast families "
        f"`{report['pairedStatistics']['multipleComparisonCorrection']['controlContrastAdjustmentSet']}` "
        f"and path-transfer families "
        f"`{report['pairedStatistics']['multipleComparisonCorrection']['pathTransferAdjustmentSet']}`.",
        "",
        "| Relation family | Facts | Transfer directions | Mean path-transfer contrast delta | 95% cluster CI | transfer q | Valid-control directions | Mean path−control delta | 95% cluster CI | control q |",
        "|---|---:|---:|---:|---|---:|---:|---:|---|---:|",
    ])
    for item in report["pairedStatistics"]["relationFamilies"]:
        lines.append(
            f"| {item['relationFamily']} | {item['factCount']} | "
            f"{item['directionCount']} | {item.get('meanPathTransferContrastDelta')} | "
            f"{item.get('pathTransferContrastDelta95PercentClusterBootstrapCI')} | "
            f"{item.get('pathTransferBenjaminiHochbergQValue')} | "
            f"{item.get('validControlDirectionCount')} | "
            f"{item.get('meanPathMinusMeanControlTransferContrastDelta')} | "
            f"{item.get('clusterBootstrap95PercentCI')} | "
            f"{item.get('benjaminiHochbergQValue')} |"
        )
    lines.extend([
        "",
        "## Per-pair explanations",
        "",
    ])
    lines.extend(pair["explanation"] + "\n" for pair in report["pairs"])
    lines.extend([
        "## Original factual states and necessity",
        "",
        "| Fact | Answer | Greedy next token | Answer mean log-likelihood | Group ablation answer LL drop | Strict necessity |",
        "|---|---|---|---:|---:|---|",
    ])
    necessity = {row["subject"]: row for row in report["necessityByFact"]}
    for fact in report["originalFacts"]:
        item = necessity[fact["subject"]]
        lines.append(
            f"| {fact['subject']} | {fact['answer']} | {fact['greedyNextToken']['token']} | "
            f"{fact['answerSequenceLikelihood']['meanTokenLogLikelihood']:.6f} | "
            f"{item['pathGroupAblation']['answerMeanTokenLogLikelihoodDrop']:.6f} | "
            f"{item['pathGroupAblation']['strictNecessityForBaselineCorrectGreedyPrediction']} |"
        )
    lines.extend([
        "",
        "## Computation interpretation and limitations",
        "",
        f"- Interpretation: `{report['computationInterpretation']['interpretation']}`",
        f"- Transfer: `{report['computationInterpretation']['factSpecificInformation']}`",
        f"- Shared relation: `{report['computationInterpretation']['sharedRelationInformation']}`",
        f"- General influence: `{report['computationInterpretation']['generalPredictiveInfluence']}`",
        f"- Distributed effects: `{report['computationInterpretation']['distributedOrRedundantComputation']}`",
        "",
        report["claimBoundary"]["doesNotEstablish"],
        "",
        "Raw activations, selected transfer logits, and control effect arrays are stored in the adjacent NPZ archive.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is required by the frozen counterfactual experiment protocol.")
    report = run_experiment(args)
    print(json.dumps({
        "marker": "GPT2_COUNTERFACTUAL_TRANSFER_EXPERIMENT_WRITTEN",
        "variant": report["variant"],
        "eligiblePairCount": report["pairInventory"]["eligibleUnorderedPairCount"],
        "testedPairCount": report["protocolConfig"]["selectedPairCount"],
        "directionCount": len(report["pairs"]) * 2,
        "device": report["runtime"]["gpu"]["name"] if report["runtime"]["gpu"] else "cpu",
        "report": str(args.output_dir.resolve() / "report.json"),
        "rawArchive": str(args.output_dir.resolve() / "raw-counterfactual.npz"),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
