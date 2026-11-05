#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import re
import sys
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
OUTPUT_ROOT = ROOT / "benchmarks/release/nn-live-gpt2-counterfactual-circuit-transfer"
CONTROL_COUNT = 100
MAGNITUDE_TOLERANCE = 0.2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Independently verify GPT-2 counterfactual path-transfer artifacts."
    )
    parser.add_argument("--variant", choices=("pilot", "all-pairs"))
    parser.add_argument("--report", type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--skip-model-replay", action="store_true")
    parser.add_argument("--baseline-only", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _close(actual: float, expected: float, atol: float = 2e-5, rtol: float = 2e-5) -> bool:
    return math.isfinite(float(actual)) and math.isclose(
        float(actual), float(expected), abs_tol=atol, rel_tol=rtol
    )


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - np.max(values)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum()


def _sequence_score(logits: np.ndarray, answer_ids: Sequence[int]) -> Dict[str, Any]:
    if logits.ndim != 2 or logits.shape[0] != len(answer_ids):
        raise ValueError("Archived answer logits do not align with the answer token sequence.")
    token_log_probabilities = []
    for row, token_id in zip(logits, answer_ids):
        maximum = float(np.max(row))
        log_normalizer = maximum + math.log(float(np.exp(row - maximum).sum()))
        token_log_probabilities.append(float(row[int(token_id)] - log_normalizer))
    return {
        "answerTokenIds": [int(item) for item in answer_ids],
        "tokenLogProbabilities": token_log_probabilities,
        "totalLogLikelihood": float(sum(token_log_probabilities)),
        "meanTokenLogLikelihood": float(np.mean(token_log_probabilities)),
    }


def _distribution_metrics(baseline_logits: np.ndarray, changed_logits: np.ndarray) -> Dict[str, float]:
    first = _softmax(baseline_logits)
    second = _softmax(changed_logits)
    absolute = np.abs(second - first)
    midpoint = (first + second) * 0.5
    epsilon = np.finfo(np.float64).tiny
    js = 0.5 * (
        np.sum(first * (np.log(np.maximum(first, epsilon)) - np.log(np.maximum(midpoint, epsilon))))
        + np.sum(second * (np.log(np.maximum(second, epsilon)) - np.log(np.maximum(midpoint, epsilon))))
    )
    return {
        "totalVariation": float(absolute.sum() * 0.5),
        "l1Probability": float(absolute.sum()),
        "jensenShannonDivergence": float(max(0.0, js)),
        "maxAbsoluteProbabilityChange": float(absolute.max()),
    }


def _source_family(case: Dict[str, Any]) -> Tuple[str, str, Tuple[int, int]] | None:
    prompt = str(case.get("prompt", ""))
    patterns = (
        ("country-capital", r"The capital of (?P<slot>.+) is"),
        ("planet-size-extreme", r"The (?P<slot>largest planet|smallest planet) in the solar system is"),
        ("gendered-parent-term", r"A (?P<slot>female parent|male parent) is called a"),
        ("occupation-description-to-profession", r"A person who (?P<slot>.+) is a"),
    )
    for family, pattern in patterns:
        match = re.fullmatch(pattern, prompt)
        if match:
            slot = match.group("slot")
            if family == "country-capital" and slot != str(case.get("subject", "")):
                return None
            return family, slot, match.span("slot")
    return None


def _tokenization(tokenizer: Any, prompt: str) -> Dict[str, List[Any]]:
    encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    return {
        "ids": [int(item) for item in encoded["input_ids"]],
        "offsets": [[int(start), int(end)] for start, end in encoded["offset_mapping"]],
    }


def _map_static_token(
    source: Dict[str, List[Any]],
    source_span: Tuple[int, int],
    destination: Dict[str, List[Any]],
    destination_span: Tuple[int, int],
    position: int,
) -> Tuple[int | None, str | None]:
    source_start, source_end = source_span
    destination_start, destination_end = destination_span
    if not 0 <= position < len(source["ids"]):
        return None, "path position is outside the token sequence"
    token_start, token_end = source["offsets"][position]
    if token_end <= source_start:
        source_region = [
            index for index, (_, end) in enumerate(source["offsets"]) if end <= source_start
        ]
        destination_region = [
            index for index, (_, end) in enumerate(destination["offsets"]) if end <= destination_start
        ]
    elif token_start >= source_end:
        source_region = [
            index for index, (start, _) in enumerate(source["offsets"]) if start >= source_end
        ]
        destination_region = [
            index for index, (start, _) in enumerate(destination["offsets"]) if start >= destination_end
        ]
    else:
        return None, "path position intersects a variable fact slot"
    try:
        ordinal = source_region.index(position)
    except ValueError:
        return None, "path position is not in a fixed-text region"
    if ordinal >= len(destination_region):
        return None, "destination has no corresponding fixed-text token"
    mapped = destination_region[ordinal]
    if source["ids"][position] != destination["ids"][mapped]:
        return None, "fixed-text token IDs differ"
    return mapped, None


def _expected_pairs(source: Dict[str, Any], tokenizer: Any) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
    passed = [item for item in source.get("cases", []) if item.get("status") == "passed"]
    if len(passed) != int(source.get("config", {}).get("caseCount", -1)):
        raise ValueError("Source passed-case count differs from the frozen source declaration.")
    groups: Dict[str, List[Dict[str, Any]]] = {}
    tokenized: Dict[str, Dict[str, List[Any]]] = {}
    spans: Dict[str, Tuple[str, str, Tuple[int, int]]] = {}
    excluded = []
    for case in passed:
        subject = str(case.get("subject", ""))
        slot = _source_family(case)
        if slot is None:
            excluded.append(subject)
            continue
        family, slot_text, span = slot
        encoded = _tokenization(tokenizer, str(case["prompt"]))
        answer = case.get("answer")
        target = case.get("target", {})
        if not isinstance(answer, str) or not answer.strip() or not isinstance(target, dict):
            excluded.append(subject)
            continue
        answer_ids = [int(item) for item in tokenizer(" " + answer, add_special_tokens=False)["input_ids"]]
        complete = [
            int(item)
            for item in tokenizer(
                str(case["prompt"]) + " " + answer, add_special_tokens=False
            )["input_ids"]
        ]
        if (
            not answer_ids
            or complete != encoded["ids"] + answer_ids
            or not isinstance(target.get("tokenId"), int)
            or answer_ids[0] != target["tokenId"]
            or tokenizer.decode([answer_ids[0]]).replace("\n", "\\n") != target.get("token")
            or not case.get("orderedResidualPath", {}).get("nodes", {}).get("attention")
            or not case.get("orderedResidualPath", {}).get("nodes", {}).get("mlp")
            or case.get("readTokenIndex") != len(encoded["ids"]) - 1
        ):
            excluded.append(subject)
            continue
        tokenized[subject] = encoded
        spans[subject] = (family, slot_text, span)
        groups.setdefault(family, []).append(case)

    pairs = []
    rejected = []
    for family, cases in groups.items():
        for case_a, case_b in itertools.combinations(cases, 2):
            subject_a, subject_b = str(case_a["subject"]), str(case_b["subject"])
            family_a, _, span_a = spans[subject_a]
            family_b, _, span_b = spans[subject_b]
            prompt_a, prompt_b = str(case_a["prompt"]), str(case_b["prompt"])
            skeleton_a = prompt_a[:span_a[0]] + "{slot}" + prompt_a[span_a[1]:]
            skeleton_b = prompt_b[:span_b[0]] + "{slot}" + prompt_b[span_b[1]:]
            reason = None
            if family_a != family_b or skeleton_a != skeleton_b:
                reason = "prompt skeleton mismatch"
            else:
                for donor, recipient, donor_span, recipient_span in (
                    (case_a, case_b, span_a, span_b),
                    (case_b, case_a, span_b, span_a),
                ):
                    donor_subject, recipient_subject = str(donor["subject"]), str(recipient["subject"])
                    for kind in ("attention", "mlp"):
                        for node in donor["orderedResidualPath"]["nodes"].get(kind, []):
                            for position in node.get("ablateTokenPositions", []):
                                _, error = _map_static_token(
                                    tokenized[donor_subject],
                                    donor_span,
                                    tokenized[recipient_subject],
                                    recipient_span,
                                    int(position),
                                )
                                if error:
                                    reason = error
                                    break
                            if reason:
                                break
                        if reason:
                            break
                    if reason:
                        break
            if reason:
                rejected.append(f"{subject_a}|{subject_b}")
            else:
                pairs.append({
                    "relationFamily": family,
                    "subjectA": subject_a,
                    "subjectB": subject_b,
                })
    order = {str(case["subject"]): index for index, case in enumerate(passed)}
    pairs.sort(key=lambda pair: (order[pair["subjectA"]], order[pair["subjectB"]]))
    return pairs, excluded, rejected


def _array(archive: Any, key: str, errors: List[str]) -> np.ndarray | None:
    if key not in archive:
        errors.append(f"raw archive is missing {key}")
        return None
    result = np.asarray(archive[key])
    if not np.isfinite(result).all():
        errors.append(f"raw archive array {key} contains non-finite values")
    return result


def validate_report_structure(report: Dict[str, Any], archive: Any) -> List[str]:
    errors: List[str] = []
    if report.get("kind") != "gpt2-counterfactual-circuit-transfer" or report.get("schemaVersion") != 1:
        return ["unsupported report kind or schema version"]
    pairs = report.get("pairs")
    if not isinstance(pairs, list):
        return ["report pairs field is not a list"]
    expected_directions = 2 * len(pairs)
    if report.get("rawArchive", {}).get("arrayCount") != len(archive.files):
        errors.append("raw archive array count differs from report metadata")
    arrays_metadata = report.get("rawArchive", {}).get("arrays", {})
    if set(arrays_metadata) != set(archive.files):
        errors.append("raw archive keys differ from report metadata")
    for key in archive.files:
        try:
            value = np.asarray(archive[key])
            metadata = arrays_metadata.get(key, {})
            if metadata.get("shape") != list(value.shape) or metadata.get("dtype") != str(value.dtype):
                errors.append(f"raw archive metadata does not match array {key}")
        except Exception as error:
            errors.append(f"cannot decode raw archive array {key}: {error}")
    if report.get("protocolConfig", {}).get("controlCountPerDirection") != CONTROL_COUNT:
        errors.append("control count differs from the frozen protocol")
    if not math.isclose(
        float(report.get("protocolConfig", {}).get("controlMagnitudeRelativeTolerance", -1)),
        MAGNITUDE_TOLERANCE,
    ):
        errors.append("control magnitude tolerance differs from the frozen protocol")
    if report.get("acceptanceCriteria", {}).get("G_recovery", {}).get("directionCount") != expected_directions:
        errors.append("recovery acceptance direction count differs from pair records")

    for pair in pairs:
        for direction_key in ("aToB", "bToA"):
            direction = pair.get(direction_key, {})
            controls = direction.get("matchedControls", [])
            if len(controls) != CONTROL_COUNT:
                errors.append(f"{pair.get('pairId')} {direction_key}: expected {CONTROL_COUNT} control records")
                continue
            sampling = direction.get("controlSampling", {})
            selected_entries = _path_entries_from_report(direction.get("donorPath", {}))
            selected_by_layer: Dict[Tuple[str, int], set[int]] = {}
            for entry in selected_entries:
                # The unit IDs are supplied by the source path and repeated in the transfer record.
                selected_by_layer.setdefault((entry["kind"], entry["layer"]), set())
            donor_nodes = direction.get("donorPath", {})
            for kind in ("attention", "mlp"):
                for node in donor_nodes.get(kind, []):
                    unit = int(node["head"] if kind == "attention" else node["neuron"])
                    selected_by_layer.setdefault((kind, int(node["layer"])), set()).add(unit)
            if sampling.get("uniqueCount") != len({
                tuple((item.get("componentType"), item.get("layer"), item.get("unit"))
                      for item in control.get("magnitudeMatches", []))
                for control in controls
            }):
                errors.append(f"{pair.get('pairId')} {direction_key}: control uniqueness count mismatch")
            if len(sampling.get("componentPools", [])) != len(selected_entries):
                errors.append(f"{pair.get('pairId')} {direction_key}: control pool count differs from path structure")
            else:
                for entry, pool in zip(selected_entries, sampling["componentPools"]):
                    if (
                        pool.get("componentType") != entry["kind"]
                        or pool.get("layer") != entry["layer"]
                        or pool.get("sourcePositions") != entry["positions"]
                    ):
                        errors.append(f"{pair.get('pairId')} {direction_key}: control positions/layers are not matched")
            for control in controls:
                matches_for_control = control.get("magnitudeMatches", [])
                if len(matches_for_control) != len(selected_entries):
                    errors.append(f"{pair.get('pairId')} {direction_key}: control component count differs from path")
                    continue
                for entry, match in zip(selected_entries, matches_for_control):
                    expected_relative_error = abs(
                        float(match.get("controlMagnitude", float("nan")))
                        - float(match.get("selectedMagnitude", float("nan")))
                    ) / max(abs(float(match.get("selectedMagnitude", float("nan")))), 1e-8)
                    if (
                        match.get("componentType") != entry["kind"]
                        or match.get("layer") != entry["layer"]
                        or int(match.get("unit", -1)) in selected_by_layer[(entry["kind"], entry["layer"])]
                        or not _close(
                            match.get("relativeMagnitudeError", float("nan")),
                            expected_relative_error,
                            atol=1e-7,
                            rtol=1e-6,
                        )
                        or bool(match.get("withinTolerance"))
                        != (expected_relative_error <= MAGNITUDE_TOLERANCE)
                    ):
                        errors.append(f"{pair.get('pairId')} {direction_key}: invalid structural/magnitude control")
                if (
                    control.get("patchSiteCount") != len(selected_entries)
                    or control.get("patchedActivationCount")
                    != direction.get("pathTransfer", {}).get("patchedActivationCount")
                ):
                    errors.append(f"{pair.get('pairId')} {direction_key}: control patch count differs from selected path")
            archive_keys = (
                direction.get("rawControlContrastArrayKey"),
                direction.get("rawControlMagnitudeMatchArrayKey"),
                direction.get("rawControlDonorPatchValuesKey"),
                direction.get("rawControlRecipientOriginalPatchValuesKey"),
                direction.get("activationTransplant", {}).get("rawDonorPatchValuesKey"),
                direction.get("activationTransplant", {}).get("rawRecipientOriginalPatchValuesKey"),
                direction.get("rawIdentityPromptLogitsKey"),
                direction.get("rawRecoveryPromptLogitsKey"),
                direction.get("rawBaselinePromptLogitsKey"),
            )
            for key in archive_keys:
                if not isinstance(key, str) or key not in archive:
                    errors.append(f"{pair.get('pairId')} {direction_key}: missing raw intervention array key {key}")
            prefix = f"{pair.get('pairId')}_{'a-to-b' if direction_key == 'aToB' else 'b-to-a'}"
            effects = _array(archive, f"{prefix}_control_transfer_contrast_delta", errors)
            matches = _array(archive, f"{prefix}_control_magnitude_match", errors)
            donor_values = _array(archive, f"{prefix}_control_donor_patch_values", errors)
            recipient_values = _array(archive, f"{prefix}_control_recipient_original_patch_values", errors)
            path_entries = _path_entries_from_report(direction.get("donorPath", {}))
            expected_patch_width = sum(
                (int(report.get("model", {}).get("config", {}).get("n_embd", 0))
                 // int(report.get("model", {}).get("config", {}).get("n_head", 1)))
                if entry["kind"] == "attention"
                else 1
                for entry in path_entries
                for _position in entry["positions"]
            )
            if effects is not None and effects.shape != (CONTROL_COUNT,):
                errors.append(f"{prefix}: raw control effects shape is invalid")
            if matches is not None and matches.shape != (CONTROL_COUNT,):
                errors.append(f"{prefix}: raw magnitude-match vector shape is invalid")
            if donor_values is not None and (donor_values.ndim != 2 or donor_values.shape[0] != CONTROL_COUNT):
                errors.append(f"{prefix}: raw control donor activations have invalid shape")
            elif donor_values is not None and donor_values.shape[1] != expected_patch_width:
                errors.append(f"{prefix}: raw control donor activation width differs from path structure")
            if recipient_values is not None and (
                recipient_values.ndim != 2 or recipient_values.shape[0] != CONTROL_COUNT
            ):
                errors.append(f"{prefix}: raw control recipient activations have invalid shape")
            elif recipient_values is not None and recipient_values.shape[1] != expected_patch_width:
                errors.append(f"{prefix}: raw control recipient activation width differs from path structure")
            if effects is not None:
                reported = np.asarray([
                    float(item.get("transferContrastDeltaTowardDonor", float("nan")))
                    for item in controls
                ])
                if reported.shape != effects.shape or not np.allclose(reported, effects, atol=1e-6, rtol=1e-6):
                    errors.append(f"{prefix}: raw control effects differ from report measurements")
            if matches is not None:
                reported_matches = np.asarray([
                    bool(item.get("magnitudeMatches"))
                    and all(bool(match.get("withinTolerance")) for match in item["magnitudeMatches"])
                    for item in controls
                ], dtype=np.bool_)
                if not np.array_equal(reported_matches, matches):
                    errors.append(f"{prefix}: raw control match flags differ from report measurements")
            donor_patch_key = f"{prefix}_donor_patch_values"
            recipient_patch_key = f"{prefix}_recipient_original_patch_values"
            donor_patch = _array(archive, donor_patch_key, errors)
            recipient_patch = _array(archive, recipient_patch_key, errors)
            if donor_patch is not None and donor_patch.ndim != 1:
                errors.append(f"{prefix}: donor path patch must be a flat activation vector")
            if recipient_patch is not None and (
                recipient_patch.ndim != 1
                or donor_patch is not None and recipient_patch.shape != donor_patch.shape
            ):
                errors.append(f"{prefix}: recipient path activation vector shape is invalid")
            baseline_logits = _array(archive, f"{prefix}_baseline_prompt_logits", errors)
            selected_logits = _array(archive, f"{prefix}_selected_prompt_logits", errors)
            restored_logits = _array(archive, f"{prefix}_restored_prompt_logits", errors)
            identity_logits = _array(archive, f"{prefix}_identity_prompt_logits", errors)
            if any(value is not None and value.ndim != 1 for value in (
                baseline_logits, selected_logits, restored_logits, identity_logits
            )):
                errors.append(f"{prefix}: prompt logit arrays must be one-dimensional")
            if all(value is not None for value in (
                baseline_logits, selected_logits, restored_logits, identity_logits
            )):
                if not np.allclose(restored_logits, identity_logits, atol=2e-5, rtol=2e-5):
                    errors.append(f"{prefix}: restoration differs from the identity activation control")
                metric = direction.get("pathTransfer", {}).get("fullDistributionChange", {})
                expected_metrics = _distribution_metrics(baseline_logits, selected_logits)
                for key, expected_value in expected_metrics.items():
                    if not _close(metric.get(key, float("nan")), expected_value, atol=1e-5, rtol=1e-4):
                        errors.append(f"{prefix}: path distribution metric {key} is inconsistent")
            recovery = direction.get("recovery", {})
            if recovery.get("recoveryVerified") and not recovery.get("withinIdentityInterventionNumericalFloor"):
                errors.append(f"{prefix}: recovery claim and identity-floor evidence conflict")
            if recovery.get("recoveryVerified") and not recovery.get("recoveredOriginalGreedyPrediction"):
                errors.append(f"{prefix}: recovery claim conflicts with the original greedy prediction")
            specificity = direction.get("specificity", {})
            if specificity.get("passesPathSpecificityComparison") and not specificity.get(
                "validStructuralAndMagnitudeControls"
            ):
                errors.append(f"{prefix}: specificity passed with invalid structural/magnitude controls")
    errors.extend(_verify_paired_statistics(report))
    return errors


def _independent_bootstrap(
    family: str,
    records: Sequence[Dict[str, Any]],
    values: np.ndarray,
    label: str,
    replicate_count: int,
) -> Dict[str, Any]:
    subjects = sorted({
        record["donorSubject"] for record in records
    } | {
        record["recipientSubject"] for record in records
    })
    mean = float(values.mean()) if len(values) else None
    if len(subjects) < 10 or not records:
        return {"mean": mean, "ci": None, "p": None, "replicates": 0}
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
            for record in records
        ], dtype=np.float64)
        total_weight = float(pair_weights.sum())
        if total_weight:
            estimates.append(float(np.dot(pair_weights, values) / total_weight))
    if not estimates:
        return {"mean": mean, "ci": None, "p": None, "replicates": 0}
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
        "replicates": len(bootstrap),
    }


def _independent_bh(values: Sequence[float]) -> List[float]:
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


def _verify_paired_statistics(report: Dict[str, Any]) -> List[str]:
    errors = []
    statistics = report.get("pairedStatistics", {})
    family_results = statistics.get("relationFamilies", [])
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for pair in report.get("pairs", []):
        for key in ("aToB", "bToA"):
            direction = pair[key]
            controls = direction.get("matchedControls", [])
            grouped.setdefault(pair["relationFamily"], []).append({
                "donorSubject": direction["donorSubject"],
                "recipientSubject": direction["recipientSubject"],
                "pathTransfer": direction["pathTransfer"],
                "meanControlTransferContrastDelta": float(np.mean([
                    item["transferContrastDeltaTowardDonor"] for item in controls
                ])) if controls else 0.0,
                "specificityControlsValid": bool(
                    direction["specificity"]["validStructuralAndMagnitudeControls"]
                ),
            })
    if set(grouped) != {item.get("relationFamily") for item in family_results}:
        return ["paired statistics relation families differ from raw pair families"]
    correction = statistics.get("multipleComparisonCorrection", {})
    replicate_count = int(correction.get("bootstrapReplicates", 0))
    if replicate_count < 100:
        return ["paired statistics omit the required bootstrap replicate count"]

    path_p_values = []
    control_p_values = []
    for result in family_results:
        family = result["relationFamily"]
        records = grouped[family]
        path_values = np.asarray([
            item["pathTransfer"]["transferContrastDeltaTowardDonor"] for item in records
        ], dtype=np.float64)
        donor_values = np.asarray([
            item["pathTransfer"]["donorAnswerMeanLogLikelihoodChange"] for item in records
        ], dtype=np.float64)
        valid_records = [item for item in records if item["specificityControlsValid"]]
        control_values = np.asarray([
            item["pathTransfer"]["transferContrastDeltaTowardDonor"]
            - item["meanControlTransferContrastDelta"] for item in valid_records
        ], dtype=np.float64)
        estimates = (
            _independent_bootstrap(family, records, path_values, "path-transfer", replicate_count),
            _independent_bootstrap(
                family, records, donor_values, "donor-answer-likelihood", replicate_count
            ),
            _independent_bootstrap(
                family, valid_records, control_values, "path-minus-valid-controls", replicate_count
            ),
        )
        mean_keys = (
            "meanPathTransferContrastDelta",
            "meanDonorAnswerMeanTokenLogLikelihoodChange",
            "meanPathMinusMeanControlTransferContrastDelta",
        )
        ci_keys = (
            "pathTransferContrastDelta95PercentClusterBootstrapCI",
            "donorAnswerMeanLogLikelihoodChange95PercentClusterBootstrapCI",
            "clusterBootstrap95PercentCI",
        )
        p_keys = (
            "pathTransferTwoSidedCenteredClusterBootstrapPValue",
            None,
            "twoSidedCenteredClusterBootstrapPValue",
        )
        for estimate, mean_key, ci_key, p_key in zip(estimates, mean_keys, ci_keys, p_keys):
            expected_mean = estimate["mean"]
            actual_mean = result.get(mean_key)
            if expected_mean is None:
                if actual_mean is not None:
                    errors.append(f"{family}: {mean_key} should be null without observations")
            elif not _close(actual_mean, expected_mean, atol=1e-8, rtol=1e-8):
                errors.append(f"{family}: {mean_key} differs from raw direction effects")
            expected_ci = estimate["ci"]
            actual_ci = result.get(ci_key)
            if (actual_ci is None) != (expected_ci is None):
                errors.append(f"{family}: {ci_key} availability differs from independent bootstrap")
            elif expected_ci is not None and not np.allclose(
                actual_ci, expected_ci, atol=1e-10, rtol=1e-10
            ):
                errors.append(f"{family}: {ci_key} differs from independent bootstrap")
            if p_key:
                expected_p = estimate["p"]
                actual_p = result.get(p_key)
                if (actual_p is None) != (expected_p is None):
                    errors.append(f"{family}: {p_key} availability differs from independent bootstrap")
                elif expected_p is not None and not _close(actual_p, expected_p, atol=1e-10, rtol=1e-10):
                    errors.append(f"{family}: {p_key} differs from independent bootstrap")
        if estimates[0]["p"] is not None:
            path_p_values.append((family, estimates[0]["p"]))
        if estimates[2]["p"] is not None:
            control_p_values.append((family, estimates[2]["p"]))

    for tests, q_key, significant_key in (
        (
            path_p_values,
            "pathTransferBenjaminiHochbergQValue",
            "pathTransferSignificantAfterMultipleComparisonCorrection",
        ),
        (
            control_p_values,
            "benjaminiHochbergQValue",
            "significantAfterMultipleComparisonCorrection",
        ),
    ):
        q_values = _independent_bh([p_value for _, p_value in tests])
        for (family, _), q_value in zip(tests, q_values):
            result = next(item for item in family_results if item["relationFamily"] == family)
            if not _close(result.get(q_key, float("nan")), q_value, atol=1e-10, rtol=1e-10):
                errors.append(f"{family}: {q_key} differs from independent Benjamini-Hochberg adjustment")
            if result.get(significant_key) is not (q_value < 0.05):
                errors.append(f"{family}: {significant_key} differs from the declared alpha")
    return errors


def _path_entries_from_report(path_nodes: Dict[str, Any]) -> List[Dict[str, Any]]:
    entries = []
    for kind in ("attention", "mlp"):
        for node in path_nodes.get(kind, []):
            entries.append({
                "kind": kind,
                "layer": int(node["layer"]),
                "unit": int(node["head"] if kind == "attention" else node["neuron"]),
                "positions": [int(item) for item in node.get("ablateTokenPositions", [])],
            })
    return entries


def _path_entries(case: Dict[str, Any]) -> List[Dict[str, Any]]:
    result = []
    nodes = case["orderedResidualPath"]["nodes"]
    for kind in ("attention", "mlp"):
        for node in nodes.get(kind, []):
            result.append({
                "kind": kind,
                "layer": int(node["layer"]),
                "unit": int(node["head"] if kind == "attention" else node["neuron"]),
                "positions": [int(item) for item in node["ablateTokenPositions"]],
            })
    return result


def _capture_activations(
    model: Any,
    input_ids: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[Tuple[str, int], torch.Tensor]]:
    values: Dict[Tuple[str, int], torch.Tensor] = {}
    handles = []
    for layer, block in enumerate(model.transformer.h):
        handles.append(block.attn.c_proj.register_forward_pre_hook(
            lambda _module, inputs, current=layer: values.__setitem__(
                ("attention", current), inputs[0].detach().cpu()
            )
        ))
        handles.append(block.mlp.act.register_forward_hook(
            lambda _module, _inputs, output, current=layer: values.__setitem__(
                ("mlp", current), output.detach().cpu()
            )
        ))
    try:
        with torch.no_grad():
            logits = model(input_ids=input_ids.to(device), use_cache=False).logits.detach().float().cpu()
    finally:
        for handle in reversed(handles):
            handle.remove()
    return logits, values


def _forward_with_patches(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    patches: Sequence[Dict[str, Any]],
    device: torch.device,
) -> torch.Tensor:
    by_layer: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for patch in patches:
        by_layer.setdefault((patch["kind"], patch["layer"]), []).append(patch)
    handles = []
    for (kind, layer), layer_patches in by_layer.items():
        if kind == "attention":
            module = model.transformer.h[layer].attn.c_proj

            def hook(_module: Any, inputs: Tuple[torch.Tensor, ...], current=layer_patches):
                patched = inputs[0].clone()
                for item in current:
                    first = item["unit"] * item["headDim"]
                    for position, value in item["positions"]:
                        patched[:, position, first : first + item["headDim"]] = value.to(
                            device=patched.device, dtype=patched.dtype
                        )
                return (patched,) + tuple(inputs[1:])

            handles.append(module.register_forward_pre_hook(hook))
        else:
            module = model.transformer.h[layer].mlp.act

            def hook(_module: Any, _inputs: Any, output: torch.Tensor, current=layer_patches):
                patched = output.clone()
                for item in current:
                    for position, value in item["positions"]:
                        patched[:, position, item["unit"]] = value.to(
                            device=patched.device, dtype=patched.dtype
                        )
                return patched

            handles.append(module.register_forward_hook(hook))
    try:
        with torch.no_grad():
            return model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                use_cache=False,
            ).logits.detach().float().cpu()
    finally:
        for handle in reversed(handles):
            handle.remove()


def _pack_patches(
    entries: Sequence[Dict[str, Any]],
    donor_values: Dict[Tuple[str, int], torch.Tensor],
    recipient_values: Dict[Tuple[str, int], torch.Tensor],
    mapping: Dict[int, int],
    model: Any,
    use_donor: bool,
) -> List[Dict[str, Any]]:
    patches = []
    for entry in entries:
        item = {
            "kind": entry["kind"],
            "layer": entry["layer"],
            "unit": entry["unit"],
            "positions": [],
        }
        for source_position in entry["positions"]:
            destination = mapping[int(source_position)]
            activation_map = donor_values if use_donor else recipient_values
            tensor = activation_map[(entry["kind"], entry["layer"])]
            if entry["kind"] == "attention":
                head_dim = int(model.transformer.h[entry["layer"]].attn.head_dim)
                start = entry["unit"] * head_dim
                value = tensor[0, int(source_position if use_donor else destination), start:start + head_dim]
                item["headDim"] = head_dim
            else:
                value = tensor[0, int(source_position if use_donor else destination), entry["unit"]]
            item["positions"].append((destination, value))
        patches.append(item)
    return patches


def _build_batch(prompt_ids: Sequence[int], answer_sequences: Sequence[Sequence[int]], pad_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    rows = [[int(item) for item in prompt_ids] + [int(item) for item in answer[:-1]] for answer in answer_sequences]
    max_length = max(len(row) for row in rows)
    ids = torch.full((len(rows), max_length), int(pad_id), dtype=torch.long)
    mask = torch.zeros_like(ids)
    for row_index, row in enumerate(rows):
        ids[row_index, :len(row)] = torch.tensor(row, dtype=torch.long)
        mask[row_index, :len(row)] = 1
    return ids, mask


def _verify_model_replay(
    report: Dict[str, Any],
    archive: Any,
    source: Dict[str, Any],
    tokenizer: Any,
    device: torch.device,
) -> List[str]:
    errors: List[str] = []
    model_id = report["model"]["id"]
    model = AutoModelForCausalLM.from_pretrained(model_id, local_files_only=True).eval().to(device)
    if model.__class__.__name__ != report["model"].get("class"):
        return [f"replayed model class differs: {model.__class__.__name__}"]
    source_cases = {
        str(case["subject"]): case for case in source.get("cases", []) if case.get("status") == "passed"
    }
    original_by_subject = {item["subject"]: item for item in report.get("originalFacts", [])}
    activations = {}
    baseline_logits = {}
    for subject, case in source_cases.items():
        if subject not in original_by_subject:
            continue
        row = original_by_subject[subject]
        prompt_ids = [int(item) for item in row["promptTokenIds"]]
        answer_ids = [int(item) for item in row["answerTokenIds"]]
        input_ids = torch.tensor([prompt_ids + answer_ids[:-1]], dtype=torch.long)
        logits, captured = _capture_activations(model, input_ids, device)
        activations[subject] = captured
        read_index = int(case["readTokenIndex"])
        baseline_logits[subject] = logits[0, read_index].numpy()
        raw_prefix = row["rawArchivePrefix"]
        archived_logits = _array(archive, f"{raw_prefix}_prompt_next_logits", errors)
        if archived_logits is not None and not np.allclose(
            archived_logits, baseline_logits[subject], atol=2e-3, rtol=2e-4
        ):
            errors.append(f"{subject}: replayed source logits differ from the raw archive")
        for kind in ("attention", "mlp"):
            nodes = case["orderedResidualPath"]["nodes"].get(kind, [])
            for node_index, node in enumerate(nodes):
                unit = int(node["head"] if kind == "attention" else node["neuron"])
                for position in node["ablateTokenPositions"]:
                    if kind == "attention":
                        head_dim = int(model.transformer.h[int(node["layer"])].attn.head_dim)
                        start = unit * head_dim
                        value = captured[(kind, int(node["layer"]))][0, int(position), start:start + head_dim]
                    else:
                        value = captured[(kind, int(node["layer"]))][0, int(position), unit].reshape(1)
                    key = f"{raw_prefix}_path_{kind}_{node_index}_{position}"
                    actual = _array(archive, key, errors)
                    if actual is not None and not np.allclose(
                        actual.reshape(-1), value.numpy().reshape(-1), atol=2e-5, rtol=2e-5
                    ):
                        errors.append(f"{subject}: captured {kind} activation differs from {key}")

    for pair in report.get("pairs", []):
        case_a = source_cases[pair["subjectA"]]
        case_b = source_cases[pair["subjectB"]]
        for direction_key, donor, recipient in (
            ("aToB", case_a, case_b),
            ("bToA", case_b, case_a),
        ):
            record = pair[direction_key]
            donor_id, recipient_id = str(donor["subject"]), str(recipient["subject"])
            donor_tokenization = _tokenization(tokenizer, str(donor["prompt"]))
            recipient_tokenization = _tokenization(tokenizer, str(recipient["prompt"]))
            donor_slot = _source_family(donor)[2]
            recipient_slot = _source_family(recipient)[2]
            position_map = {}
            for entry in _path_entries(donor):
                for source_position in entry["positions"]:
                    destination, error = _map_static_token(
                        donor_tokenization,
                        donor_slot,
                        recipient_tokenization,
                        recipient_slot,
                        source_position,
                    )
                    if error:
                        errors.append(f"{pair['pairId']} {direction_key}: {error}")
                    else:
                        position_map[source_position] = int(destination)
            if {str(key): value for key, value in position_map.items()} != record.get("positionMap"):
                errors.append(f"{pair['pairId']} {direction_key}: archived token alignment map is incorrect")
            entries = _path_entries(donor)
            donor_patch = _pack_patches(
                entries, activations[donor_id], activations[recipient_id],
                position_map, model, use_donor=True,
            )
            recipient_patch = _pack_patches(
                entries, activations[donor_id], activations[recipient_id],
                position_map, model, use_donor=False,
            )
            prefix = f"{pair['pairId']}_{record['direction']}"
            donor_vector = np.concatenate([
                value.numpy().reshape(-1)
                for entry in entries
                for source_position in entry["positions"]
                for value in [(
                    activations[donor_id][(entry["kind"], entry["layer"])][0, source_position,
                        entry["unit"] * int(model.transformer.h[entry["layer"]].attn.head_dim):
                        (entry["unit"] + 1) * int(model.transformer.h[entry["layer"]].attn.head_dim)]
                    if entry["kind"] == "attention"
                    else activations[donor_id][(entry["kind"], entry["layer"])][0, source_position, entry["unit"]].reshape(1)
                )]
            ])
            recipient_vector = np.concatenate([
                value.numpy().reshape(-1)
                for entry in entries
                for source_position in entry["positions"]
                for value in [(
                    activations[recipient_id][(entry["kind"], entry["layer"])][0, position_map[source_position],
                        entry["unit"] * int(model.transformer.h[entry["layer"]].attn.head_dim):
                        (entry["unit"] + 1) * int(model.transformer.h[entry["layer"]].attn.head_dim)]
                    if entry["kind"] == "attention"
                    else activations[recipient_id][(entry["kind"], entry["layer"])][0, position_map[source_position], entry["unit"]].reshape(1)
                )]
            ])
            actual_donor = _array(archive, f"{prefix}_donor_patch_values", errors)
            actual_recipient = _array(archive, f"{prefix}_recipient_original_patch_values", errors)
            if actual_donor is not None and not np.allclose(
                actual_donor, donor_vector, atol=2e-5, rtol=2e-5
            ):
                errors.append(f"{pair['pairId']} {direction_key}: donor patch vector failed model replay")
            if actual_recipient is not None and not np.allclose(
                actual_recipient, recipient_vector, atol=2e-5, rtol=2e-5
            ):
                errors.append(f"{pair['pairId']} {direction_key}: recipient-original vector failed model replay")

            options = [
                [int(item) for item in original_by_subject[pair["subjectA"]]["answerTokenIds"]],
                [int(item) for item in original_by_subject[pair["subjectB"]]["answerTokenIds"]],
            ]
            prompt_ids = [int(item) for item in original_by_subject[recipient_id]["promptTokenIds"]]
            pad_id = int(tokenizer.eos_token_id or 0)
            input_ids, mask = _build_batch(prompt_ids, options, pad_id)
            original_logits = _forward_with_patches(model, input_ids, mask, [], device)
            path_logits = _forward_with_patches(model, input_ids, mask, donor_patch, device)
            restored_logits = _forward_with_patches(model, input_ids, mask, recipient_patch, device)
            read_index = len(prompt_ids) - 1
            donor_option = 0 if direction_key == "aToB" else 1
            recipient_option = 1 - donor_option
            scores = [
                _sequence_score(
                    path_logits[index, read_index : read_index + len(answer)].numpy(),
                    answer,
                )
                for index, answer in enumerate(options)
            ]
            base_scores = [
                _sequence_score(
                    original_logits[index, read_index : read_index + len(answer)].numpy(),
                    answer,
                )
                for index, answer in enumerate(options)
            ]
            report_scores = record["transplantedAnswerScores"]
            for option_index in range(2):
                answer_count = len(options[option_index])
                for label, logits in (
                    ("baseline", original_logits),
                    ("selected", path_logits),
                    ("restored", restored_logits),
                ):
                    key = f"{prefix}_{label}_answer_logits_{option_index}"
                    actual_answer_logits = _array(archive, key, errors)
                    expected_answer_logits = logits[
                        option_index, read_index : read_index + answer_count
                    ].numpy()
                    if actual_answer_logits is not None and not np.allclose(
                        actual_answer_logits,
                        expected_answer_logits,
                        atol=3e-3,
                        rtol=3e-4,
                    ):
                        errors.append(f"{pair['pairId']} {direction_key}: {label} answer logits failed replay")
                if not _close(
                    scores[option_index]["totalLogLikelihood"],
                    report_scores[option_index]["totalLogLikelihood"],
                    atol=2e-3,
                    rtol=2e-4,
                ):
                    errors.append(f"{pair['pairId']} {direction_key}: replayed answer likelihood differs")
            contrast_delta = (
                scores[donor_option]["meanTokenLogLikelihood"]
                - scores[recipient_option]["meanTokenLogLikelihood"]
                - base_scores[donor_option]["meanTokenLogLikelihood"]
                + base_scores[recipient_option]["meanTokenLogLikelihood"]
            )
            if not _close(
                contrast_delta,
                record["pathTransfer"]["transferContrastDeltaTowardDonor"],
                atol=3e-3,
                rtol=3e-4,
            ):
                errors.append(f"{pair['pairId']} {direction_key}: replayed transfer contrast differs")
            recovered_token = int(torch.argmax(restored_logits[0, read_index]))
            expected_token = int(record["recovery"]["baselineGreedyTokenId"])
            if recovered_token != expected_token:
                errors.append(f"{pair['pairId']} {direction_key}: replayed restoration changed the greedy token")
            archived_selected = _array(archive, f"{prefix}_selected_prompt_logits", errors)
            if archived_selected is not None and not np.allclose(
                archived_selected, path_logits[0, read_index].numpy(), atol=3e-3, rtol=3e-4
            ):
                errors.append(f"{pair['pairId']} {direction_key}: selected transfer logits failed model replay")
            archived_restored = _array(archive, f"{prefix}_restored_prompt_logits", errors)
            if archived_restored is not None and not np.allclose(
                archived_restored, restored_logits[0, read_index].numpy(), atol=3e-3, rtol=3e-4
            ):
                errors.append(f"{pair['pairId']} {direction_key}: recovery logits failed model replay")
    return errors


def verify_report(report_path: Path, expected_variant: str, device_name: str, replay: bool) -> None:
    errors = []
    if not report_path.is_file():
        raise FileNotFoundError(f"Experiment report does not exist: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("variant") != expected_variant:
        errors.append(f"expected variant {expected_variant}, got {report.get('variant')}")
    source_path = SOURCE_REPORT.resolve()
    if not source_path.is_file() or sha256_file(source_path) != EXPECTED_SOURCE_SHA256:
        errors.append("frozen source artifact is missing or changed")
    for name, expected in BASELINE_HASHES.items():
        path = BASELINE_DIR / name
        if not path.is_file() or sha256_file(path) != expected:
            errors.append(f"previous autoregressive artifact changed or missing: {name}")
    if report.get("sourceArtifact", {}).get("sha256") != EXPECTED_SOURCE_SHA256:
        errors.append("report source artifact hash differs from frozen source")
    archive_path = report_path.parent / "raw-counterfactual.npz"
    if not archive_path.is_file():
        errors.append("raw counterfactual archive is missing")
        archive = None
    else:
        if sha256_file(archive_path) != report.get("rawArchive", {}).get("sha256"):
            errors.append("raw archive hash differs from report metadata")
        archive = np.load(archive_path, allow_pickle=False)
        errors.extend(validate_report_structure(report, archive))
    if source_path.is_file():
        tokenizer = AutoTokenizer.from_pretrained(report.get("model", {}).get("id", "gpt2"), local_files_only=True)
        source = json.loads(source_path.read_text(encoding="utf-8"))
        expected_pairs, excluded, rejected = _expected_pairs(source, tokenizer)
        inventory = report.get("pairInventory", {})
        if inventory.get("eligibleUnorderedPairCount") != len(expected_pairs):
            errors.append("eligible pair count differs from independent source-derived inventory")
        if inventory.get("excludedFactCount") != len(excluded):
            errors.append("excluded fact count differs from independent source-derived inventory")
        if inventory.get("rejectedCandidatePairCount") != len(rejected):
            errors.append("rejected pair count differs from independent source-derived inventory")
        expected_count = min(10, len(expected_pairs)) if expected_variant == "pilot" else len(expected_pairs)
        if len(report.get("pairs", [])) != expected_count:
            errors.append(f"{expected_variant} has {len(report.get('pairs', []))} pairs; expected {expected_count}")
        original_subjects = [item.get("subject") for item in report.get("originalFacts", [])]
        if len(original_subjects) != inventory.get("pairEligibleFactCount"):
            errors.append("original factual state count differs from pair-eligible fact count")
        source_by_subject = {
            str(item["subject"]): item for item in source.get("cases", []) if item.get("status") == "passed"
        }
        for index, pair in enumerate(report.get("pairs", [])):
            expected = expected_pairs[index]
            if any(pair.get(key) != expected[key] for key in ("relationFamily", "subjectA", "subjectB")):
                errors.append(f"pair at index {index} differs from independently derived eligible-pair order")
            for subject_key in ("subjectA", "subjectB"):
                if pair.get(subject_key) not in source_by_subject:
                    errors.append(f"pair references non-source subject: {pair.get(subject_key)}")
        if archive is not None and replay:
            if device_name == "cuda" and not torch.cuda.is_available():
                errors.append("CUDA replay requested but CUDA is unavailable")
            else:
                device = torch.device(device_name)
                errors.extend(_verify_model_replay(report, archive, source, tokenizer, device))
    if archive is not None:
        archive.close()
    if errors:
        raise AssertionError("Counterfactual transfer verification failed:\n- " + "\n- ".join(errors[:80]))
    print(json.dumps({
        "marker": "GPT2_COUNTERFACTUAL_TRANSFER_VERIFIED",
        "variant": expected_variant,
        "pairCount": len(report["pairs"]),
        "directionCount": 2 * len(report["pairs"]),
        "modelReplay": replay,
        "report": str(report_path.resolve()),
    }, sort_keys=True))


def verify_baseline_artifacts() -> None:
    errors = []
    if not SOURCE_REPORT.is_file() or sha256_file(SOURCE_REPORT) != EXPECTED_SOURCE_SHA256:
        errors.append("frozen ordered-path source report changed or is missing")
    for name, expected in BASELINE_HASHES.items():
        path = BASELINE_DIR / name
        if not path.is_file() or sha256_file(path) != expected:
            errors.append(f"previous autoregressive artifact changed or is missing: {name}")
    if errors:
        raise AssertionError("Baseline preservation verification failed:\n- " + "\n- ".join(errors))
    print(json.dumps({
        "marker": "GPT2_COUNTERFACTUAL_BASELINES_PRESERVED",
        "sourceSha256": EXPECTED_SOURCE_SHA256,
        "previousArtifactCount": len(BASELINE_HASHES),
    }, sort_keys=True))


def main() -> None:
    args = parse_args()
    if args.baseline_only:
        verify_baseline_artifacts()
        return
    if args.variant is None:
        raise SystemExit("--variant is required unless --baseline-only is selected")
    default_variant = "pilot-archive-corrected" if args.variant == "pilot" else "all-pairs"
    report_path = args.report or OUTPUT_ROOT / default_variant / "report.json"
    verify_report(report_path, args.variant, args.device, not args.skip_model_replay)


if __name__ == "__main__":
    main()
