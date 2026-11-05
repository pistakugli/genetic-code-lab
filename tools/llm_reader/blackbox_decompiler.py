from __future__ import annotations

import base64
import hashlib
import itertools
import json
import math
import os
import struct
import zlib

import numpy as np
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

from llm_reader.blackbox_decision import (
    _editable_context_units,
    _js_divergence,
    _logit_values,
    _scope_context_units,
    _softmax,
    trace_greedy_generation,
)


class OracleBudgetExceeded(RuntimeError):
    pass


def _single_sequence(
    value: Any,
    label: str,
    *,
    pair_sequence: bool = False,
) -> List[Any]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], (list, tuple)):
        is_batched_pairs = (
            pair_sequence
            and value[0]
            and isinstance(value[0][0], (list, tuple))
        )
        is_batched_values = not pair_sequence
        if is_batched_pairs or is_batched_values:
            if len(value) != 1:
                raise ValueError(f"{label} returned multiple prompts.")
            value = value[0]
    return list(value)


def _content_token_indices(
    tokenizer: Any,
    rendered_prompt: str,
    content_start: int,
    content_end: int,
    prompt: str,
    canonical_token_ids: Sequence[int],
) -> List[int]:
    if content_start == content_end:
        return []

    offset_error: Exception | None = None
    try:
        encoded = tokenizer(
            rendered_prompt,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
    except (NotImplementedError, TypeError, ValueError) as error:
        offset_error = error
        encoded = None

    if isinstance(encoded, Mapping) and "offset_mapping" in encoded:
        encoded_ids = [
            int(value) for value in _single_sequence(
                encoded["input_ids"],
                "Rendered chat-template tokenization",
            )
        ]
        offsets = _single_sequence(
            encoded["offset_mapping"],
            "Rendered chat-template offsets",
            pair_sequence=True,
        )
        if encoded_ids == list(canonical_token_ids) and len(offsets) == len(encoded_ids):
            editable_indices = [
                index
                for index, pair in enumerate(offsets)
                if len(pair) == 2
                and int(pair[0]) >= content_start
                and int(pair[1]) <= content_end
                and int(pair[1]) > int(pair[0])
            ]
            if editable_indices:
                return editable_indices
            raise ValueError(
                "Tokenizer offsets did not identify any complete user-content tokens."
            )

    content_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    content_ids = [
        int(value)
        for value in _single_sequence(content_ids, "Prompt content tokenization")
    ]
    matches = [
        start
        for start in range(len(canonical_token_ids) - len(content_ids) + 1)
        if content_ids
        and list(canonical_token_ids[start:start + len(content_ids)]) == content_ids
    ]
    if len(matches) == 1:
        return list(range(matches[0], matches[0] + len(content_ids)))
    if len(matches) > 1:
        prefix_ids = tokenizer(
            rendered_prompt[:content_start],
            add_special_tokens=False,
        )["input_ids"]
        prefix_count = len(_single_sequence(prefix_ids, "Prompt prefix tokenization"))
        nearest_distance = min(abs(start - prefix_count) for start in matches)
        nearest = [start for start in matches if abs(start - prefix_count) == nearest_distance]
        if len(nearest) == 1:
            return list(range(nearest[0], nearest[0] + len(content_ids)))
    error = ValueError(
        "Could not map chat-template user content to encoded token positions; "
        "use a tokenizer with offset mappings or an unambiguous content token span."
    )
    if offset_error is not None:
        raise error from offset_error
    raise error


def encode_prompt_with_scope(
    tokenizer: Any,
    prompt: str,
    prompt_format: str = "auto",
) -> Tuple[List[int], str, List[int]]:
    token_ids, encoded_format = encode_prompt(tokenizer, prompt, prompt_format)
    if encoded_format == "raw":
        return token_ids, encoded_format, list(range(len(token_ids)))

    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_chat_template):
        raise ValueError("The tokenizer does not provide a usable chat template.")
    rendered_prompt = apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    if not isinstance(rendered_prompt, str):
        raise ValueError("Chat-template rendering did not return text.")

    marker_suffix = ""
    start_marker = "__GCL_USER_CONTENT_START__"
    end_marker = "__GCL_USER_CONTENT_END__"
    while start_marker in prompt or end_marker in prompt:
        marker_suffix += "_"
        start_marker = f"__GCL_USER_CONTENT_START{marker_suffix}__"
        end_marker = f"__GCL_USER_CONTENT_END{marker_suffix}__"
    marked_prompt = apply_chat_template(
        [{
            "role": "user",
            "content": f"{start_marker}{prompt}{end_marker}",
        }],
        tokenize=False,
        add_generation_prompt=True,
    )
    if not isinstance(marked_prompt, str):
        raise ValueError("Marked chat-template rendering did not return text.")
    marker_start = marked_prompt.find(start_marker)
    marker_end = marked_prompt.find(end_marker, marker_start + len(start_marker))
    if (
        marker_start < 0
        or marker_end < 0
        or marked_prompt.count(start_marker) != 1
        or marked_prompt.count(end_marker) != 1
    ):
        raise ValueError(
            "Could not identify the user-message content span in the chat template."
        )
    content_start = marker_start
    content_end = marker_end - len(start_marker)
    if rendered_prompt[content_start:content_end] != prompt:
        raise ValueError(
            "Chat-template rendering changed user content and its token span "
            "cannot be mapped safely."
        )
    editable_token_indices = _content_token_indices(
        tokenizer,
        rendered_prompt,
        content_start,
        content_end,
        prompt,
        token_ids,
    )
    if not token_ids:
        raise ValueError("Prompt encoding produced no input tokens.")
    return token_ids, encoded_format, editable_token_indices


def encode_prompt(
    tokenizer: Any,
    prompt: str,
    prompt_format: str = "auto",
) -> Tuple[List[int], str]:
    if prompt_format not in {"auto", "raw", "chat-template"}:
        raise ValueError(
            "prompt_format must be 'auto', 'raw', or 'chat-template'."
        )
    has_chat_template = bool(getattr(tokenizer, "chat_template", None))
    use_chat_template = prompt_format == "chat-template" or (
        prompt_format == "auto" and has_chat_template
    )
    if use_chat_template:
        apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
        if not has_chat_template or not callable(apply_chat_template):
            raise ValueError(
                "The tokenizer does not provide a usable chat template."
            )
        input_ids = apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
        )
        if isinstance(input_ids, Mapping):
            input_ids = input_ids["input_ids"]
        input_ids = _single_sequence(input_ids, "Chat-template encoding")
        encoded_format = "chat-template"
    else:
        input_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        input_ids = _single_sequence(input_ids, "Raw prompt encoding")
        encoded_format = "raw"
    token_ids = [int(token_id) for token_id in input_ids]
    if not token_ids:
        raise ValueError("Prompt encoding produced no input tokens.")
    return token_ids, encoded_format


def _atomic_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


class PersistentOracleSession:
    """Durably cache oracle-only observations so a decompilation can resume."""

    def __init__(
        self,
        oracle: Any,
        directory: str | Path,
        *,
        identity: Dict[str, Any],
        prompt: str,
        parameters: Dict[str, Any],
        query_budget: int,
        resume: bool,
    ) -> None:
        if query_budget < 1:
            raise ValueError("query_budget must be positive.")
        self.oracle = oracle
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.log_path = self.directory / "oracle-observations.jsonl"
        self.checkpoint_path = self.directory / "checkpoint.json"
        self.request = {
            "model": identity,
            "promptSha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "parameters": parameters,
        }
        self.query_budget = query_budget
        self.cache: Dict[Tuple[int, ...], List[float]] = {}
        self.records: List[Dict[str, Any]] = []
        self.logical_calls = 0
        self.training_query_ids: List[str] = []
        if resume:
            if not self.checkpoint_path.exists():
                raise FileNotFoundError(f"No resumable session at {self.directory}")
            checkpoint = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
            if checkpoint.get("request") != self.request:
                raise ValueError("Resume request does not match the saved model, prompt, and parameters.")
            self.training_query_ids = list(checkpoint.get("trainingQueryIds", []))
            if not self.log_path.exists():
                raise ValueError("Session checkpoint exists but its oracle observation log is missing.")
            self._load_records()
        elif self.checkpoint_path.exists() or self.log_path.exists():
            raise FileExistsError(
                f"Session already exists at {self.directory}; use --resume to continue it."
            )
        self._checkpoint("running")

    def _load_records(self) -> None:
        for line_number, line in enumerate(
            self.log_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            try:
                record = json.loads(line)
                raw = zlib.decompress(base64.b64decode(record["logitsZlibF32"]))
                size = int(record["vocabularySize"])
                if len(raw) != size * 4:
                    raise ValueError("compressed logits length does not match vocabulary size")
                logits = list(struct.unpack(f"<{size}f", raw))
                ids = tuple(int(value) for value in record["inputTokenIds"])
            except Exception as error:
                raise ValueError(
                    f"Invalid oracle observation at {self.log_path}:{line_number}: {error}"
                ) from error
            if ids in self.cache:
                raise ValueError(f"Duplicate oracle context in session log at line {line_number}.")
            self.cache[ids] = logits
            self.records.append(record)

    def _checkpoint(self, status: str) -> None:
        _atomic_json(self.checkpoint_path, {
            "schemaVersion": 1,
            "status": status,
            "updatedAt": datetime.now(timezone.utc).isoformat(),
            "request": self.request,
            "queryBudget": self.query_budget,
            "oracleQueriesCompleted": len(self.records),
            "logicalOracleCallsThisRun": self.logical_calls,
            "trainingQueryIds": self.training_query_ids,
            "observationLog": self.log_path.name,
        })

    def set_training_boundary(self, query_ids: Sequence[str]) -> None:
        self.training_query_ids = list(query_ids)
        self._checkpoint("training-complete")

    def query_token_ids(self, token_ids: Sequence[int]) -> List[float]:
        context = tuple(int(value) for value in token_ids)
        if not context:
            raise ValueError("At least one input token is required.")
        self.logical_calls += 1
        cached = self.cache.get(context)
        if cached is not None:
            return list(cached)
        if len(self.records) >= self.query_budget:
            self._checkpoint("budget-exhausted")
            raise OracleBudgetExceeded(
                f"Oracle query budget {self.query_budget} exhausted after "
                f"{len(self.records)} unique calls. Resume this session with a larger --query-budget."
            )
        logits = _logit_values(self.oracle.query_token_ids(list(context)))
        packed = struct.pack(f"<{len(logits)}f", *logits)
        record = {
            "queryId": f"q{len(self.records) + 1:06d}",
            "inputTokenIds": list(context),
            "vocabularySize": len(logits),
            "logitsZlibF32": base64.b64encode(zlib.compress(packed, level=6)).decode("ascii"),
        }
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.cache[context] = logits
        self.records.append(record)
        self._checkpoint("running")
        return list(logits)

    def finish(self, status: str = "complete") -> None:
        self._checkpoint(status)


def _top_candidates(logits: Sequence[float], count: int) -> List[Dict[str, Any]]:
    probabilities = _softmax(logits)
    indices = sorted(range(len(logits)), key=logits.__getitem__, reverse=True)[:count]
    return [
        {
            "tokenId": index,
            "logit": float(logits[index]),
            "probability": float(probabilities[index]),
        }
        for index in indices
    ]


def _sequence_edit_distance(left: Sequence[int], right: Sequence[int]) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_token in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_token in enumerate(right, start=1):
            current.append(min(
                current[-1] + 1,
                previous[right_index] + 1,
                previous[right_index - 1] + (left_token != right_token),
            ))
        previous = current
    return previous[-1]


def _query_id_map(records: Sequence[Dict[str, Any]]) -> Dict[Tuple[int, ...], str]:
    return {
        tuple(int(value) for value in row["inputTokenIds"]): row["queryId"]
        for row in records
    }


def _prompt_formulation_effects(
    *,
    baseline_prompt_token_ids: Sequence[int],
    baseline_generated_token_ids: Sequence[int],
    formulation_paths: Sequence[Dict[str, Any]],
    records: Sequence[Dict[str, Any]],
    cache: Dict[Tuple[int, ...], Sequence[float]],
) -> List[Dict[str, Any]]:
    query_ids = _query_id_map(records)
    baseline_prompt = [int(value) for value in baseline_prompt_token_ids]
    baseline_generated = [int(value) for value in baseline_generated_token_ids]
    effects = []
    for path in formulation_paths:
        variant_prompt = [int(value) for value in path["inputTokenIds"]]
        variant_generated = [int(value) for value in path["generatedTokenIds"]]
        shared = 0
        while (
            shared < len(baseline_generated)
            and shared < len(variant_generated)
            and baseline_generated[shared] == variant_generated[shared]
        ):
            shared += 1
        baseline_context = tuple([*baseline_prompt, *baseline_generated[:shared]])
        variant_context = tuple([*variant_prompt, *variant_generated[:shared]])
        baseline_query_id = query_ids.get(baseline_context)
        variant_query_id = query_ids.get(variant_context)
        evidence_ids = list(dict.fromkeys(
            value for value in (baseline_query_id, variant_query_id) if value
        ))
        divergence = None
        if shared < len(baseline_generated) and shared < len(variant_generated):
            baseline_token = baseline_generated[shared]
            variant_token = variant_generated[shared]
            baseline_logits = cache[baseline_context]
            variant_logits = cache[variant_context]
            baseline_margin = float(
                baseline_logits[baseline_token] - baseline_logits[variant_token]
            )
            variant_margin = float(
                variant_logits[variant_token] - variant_logits[baseline_token]
            )
            divergence = {
                "stepIndex": shared,
                "baselineTokenId": baseline_token,
                "formulationTokenId": variant_token,
                "baselinePairwiseMargin": baseline_margin,
                "formulationPairwiseMargin": variant_margin,
                "pairwiseMarginShift": variant_margin - baseline_margin,
                "baselineQueryId": baseline_query_id,
                "formulationQueryId": variant_query_id,
            }
        effects.append({
            "formulationId": path["id"],
            "sharedGeneratedPrefixLength": shared,
            "firstDivergence": divergence,
            "evidenceQueryIds": evidence_ids,
            "interpretation": (
                "observed_prompt_formulation_contrast_not_isolated_token_causality"
            ),
        })
    return effects


def _measured_decision_models(
    *,
    trace: Dict[str, Any],
    input_token_ids: Sequence[int],
    cache: Dict[Tuple[int, ...], Sequence[float]],
    records: Sequence[Dict[str, Any]],
    candidate_count: int,
) -> List[Dict[str, Any]]:
    query_ids = _query_id_map(records)
    models: List[Dict[str, Any]] = []
    for step in trace["steps"]:
        base_context = tuple(int(value) for value in step["contextTokenIds"])
        baseline = cache[base_context]
        interventions: List[Dict[str, Any]] = []
        token_ids = {
            item["tokenId"]
            for item in _top_candidates(baseline, candidate_count)
        }
        for item in step["contextTokenInterventions"]:
            if item["source"] != "input":
                continue
            context = tuple(int(value) for value in item["counterfactualContextTokenIds"])
            changed = cache.get(context)
            if changed is None:
                continue
            token_ids.update(
                candidate["tokenId"]
                for candidate in _top_candidates(changed, candidate_count)
            )
            interventions.append({
                "deletedTokenIndices": [
                    int(position) for position in item["contextTokenIndices"]
                ],
                "deltaByTokenId": {},
                "queryId": query_ids.get(context),
            })
            for candidate_id in token_ids:
                interventions[-1]["deltaByTokenId"][str(candidate_id)] = (
                    float(changed[candidate_id]) - float(baseline[candidate_id])
                )
        base_candidate_ids = {
            item["tokenId"] for item in _top_candidates(baseline, candidate_count)
        }
        for intervention in interventions:
            changed_context = next(
                (
                    tuple(int(value) for value in item["counterfactualContextTokenIds"])
                    for item in step["contextTokenInterventions"]
                    if item["source"] == "input"
                    and [int(value) for value in item["contextTokenIndices"]]
                    == intervention["deletedTokenIndices"]
                ),
                None,
            )
            if changed_context is None:
                continue
            changed = cache[changed_context]
            for candidate_id in token_ids:
                intervention["deltaByTokenId"][str(candidate_id)] = (
                    float(changed[candidate_id]) - float(baseline[candidate_id])
                )
        interactions: List[Dict[str, Any]] = []
        for pair in step["pairwiseInterventions"]:
            units = pair["deletedUnits"]
            if len(units) != 2 or any(unit["source"] != "input" for unit in units):
                continue
            groups = [
                [int(position) for position in unit["contextTokenIndices"]]
                for unit in units
            ]
            effects = []
            for group in groups:
                effect = next(
                    (
                        item for item in interventions
                        if item["deletedTokenIndices"] == group
                    ),
                    None,
                )
                if effect is None:
                    effects = []
                    break
                effects.append(effect)
            if len(effects) != 2:
                continue
            pair_context = tuple(int(value) for value in pair["counterfactualContextTokenIds"])
            joint_logits = cache.get(pair_context)
            if joint_logits is None:
                continue
            token_ids.update(
                candidate["tokenId"]
                for candidate in _top_candidates(joint_logits, candidate_count)
            )
            interactions.append({
                "deletedTokenIndices": groups,
                "residualByTokenId": {
                    str(candidate_id): (
                        float(joint_logits[candidate_id])
                        - float(baseline[candidate_id])
                        - float(effects[0]["deltaByTokenId"].get(str(candidate_id), 0.0))
                        - float(effects[1]["deltaByTokenId"].get(str(candidate_id), 0.0))
                    )
                    for candidate_id in token_ids
                },
                "queryId": query_ids.get(pair_context),
            })
        models.append({
            "stepIndex": step["stepIndex"],
            "generatedPrefixTokenIds": [
                int(value) for value in trace["generatedTokenIds"][:step["stepIndex"]]
            ],
            "baselineContextTokenIds": list(base_context),
            "baselinePromptTokenIds": [int(value) for value in input_token_ids],
            "baselineLogitsByTokenId": {
                str(candidate_id): float(baseline[candidate_id])
                for candidate_id in token_ids
            },
            "baselineTopTokenId": int(step["chosenTokenId"]),
            "candidateTokenIds": sorted(int(value) for value in token_ids),
            "inputDeletionEffects": interventions,
            "measuredPairCorrections": interactions,
            "baselineQueryId": query_ids.get(base_context),
        })
    return models


def _deleted_indices(
    baseline: Sequence[int],
    candidate: Sequence[int],
) -> List[int] | None:
    removed: List[int] = []
    candidate_index = 0
    for baseline_index, token_id in enumerate(baseline):
        if candidate_index < len(candidate) and int(candidate[candidate_index]) == int(token_id):
            candidate_index += 1
        else:
            removed.append(baseline_index)
    if candidate_index != len(candidate):
        return None
    return removed


def _candidate_log_scores(
    prediction: Dict[str, Any],
) -> Dict[int, float]:
    stored_scores = prediction.get("candidateLogScoresByTokenId")
    if isinstance(stored_scores, dict):
        return {
            int(token_id): float(score)
            for token_id, score in stored_scores.items()
        }
    scores = {}
    for candidate in prediction.get("candidates", []):
        token_id = int(candidate["tokenId"])
        if "relativeLogit" in candidate:
            scores[token_id] = float(candidate["relativeLogit"])
        elif "logit" in candidate:
            scores[token_id] = float(candidate["logit"])
        else:
            probability = candidate.get(
                "weightedProbability",
                candidate.get("probability"),
            )
            if probability is not None and float(probability) > 0.0:
                scores[token_id] = math.log(float(probability))
    return scores


def _predicted_logit_margin(
    prediction: Dict[str, Any],
    target_token_id: int,
    competitor_token_id: int,
) -> float | None:
    scores = _candidate_log_scores(prediction)
    if (
        target_token_id not in scores
        or competitor_token_id not in scores
    ):
        return None
    return scores[target_token_id] - scores[competitor_token_id]


def _contiguous_index_groups(indices: Sequence[int]) -> List[List[int]]:
    groups: List[List[int]] = []
    for index in sorted({int(value) for value in indices}):
        if not groups or index != groups[-1][-1] + 1:
            groups.append([index])
        else:
            groups[-1].append(index)
    return groups


def _calibrated_local_deletion_effects(
    decision_model: Dict[str, Any],
    missing_groups: Sequence[Sequence[int]],
) -> Dict[str, Any] | None:
    effects = [
        effect
        for effect in decision_model.get("inputDeletionEffects", [])
        if effect.get("deletedTokenIndices") and effect.get("deltaByTokenId")
    ]
    if len(effects) < 3 or not missing_groups:
        return None

    spans = [
        sorted({int(value) for value in effect["deletedTokenIndices"]})
        for effect in effects
    ]
    deltas = [
        {
            int(token_id): float(delta)
            for token_id, delta in effect["deltaByTokenId"].items()
        }
        for effect in effects
    ]
    baseline_scores = {
        int(token_id): float(score)
        for token_id, score in decision_model["baselineLogitsByTokenId"].items()
    }
    candidate_ids = sorted(
        set(baseline_scores).union(
            token_id for effect_delta in deltas for token_id in effect_delta
        )
    )
    powers = (1, 2, 3)
    shrinkages = (0.0, 0.25, 0.5, 0.75, 1.0)

    def span_gap(left: Sequence[int], right: Sequence[int]) -> int:
        return max(
            0,
            max(left[-1], right[-1]) - min(left[0], right[0]) - 1,
        )

    def estimate(
        target_span: Sequence[int],
        training_indices: Sequence[int],
        power: int,
    ) -> Dict[int, float] | None:
        weighted_effects = []
        for index in training_indices:
            training_span = spans[index]
            if set(target_span).intersection(training_span):
                continue
            weight = 1.0 / ((span_gap(target_span, training_span) + 1) ** power)
            weighted_effects.append((weight, deltas[index]))
        total_weight = sum(weight for weight, _ in weighted_effects)
        if total_weight <= 0.0:
            return None
        return {
            token_id: sum(
                weight * effect_delta.get(token_id, 0.0)
                for weight, effect_delta in weighted_effects
            ) / total_weight
            for token_id in candidate_ids
        }

    baseline_error = 0.0
    validation_targets = 0
    for target_delta in deltas:
        baseline_error += sum(
            target_delta.get(token_id, 0.0) ** 2
            for token_id in candidate_ids
        ) / max(len(candidate_ids), 1)
        validation_targets += 1
    if not validation_targets or baseline_error <= 0.0:
        return None
    baseline_error /= validation_targets

    best: Tuple[float, float, int] | None = None
    for power in powers:
        for shrinkage in shrinkages:
            total_error = 0.0
            evaluated = 0
            for target_index, target_span in enumerate(spans):
                training_indices = [
                    index for index in range(len(effects))
                    if index != target_index
                ]
                predicted_delta = estimate(
                    target_span,
                    training_indices,
                    power,
                )
                if predicted_delta is None:
                    continue
                total_error += sum(
                    (
                        shrinkage * predicted_delta[token_id]
                        - deltas[target_index].get(token_id, 0.0)
                    ) ** 2
                    for token_id in candidate_ids
                ) / max(len(candidate_ids), 1)
                evaluated += 1
            if evaluated != validation_targets:
                continue
            mean_error = total_error / evaluated
            if best is None or (mean_error, -shrinkage, power) < (
                best[0],
                -best[1],
                best[2],
            ):
                best = (mean_error, shrinkage, power)

    if best is None or best[0] >= baseline_error * 0.95:
        return None
    calibrated_error, shrinkage, power = best
    transferred: Dict[int, float] = {}
    evidence_query_ids = set()
    estimated_spans = []
    estimated_effects_by_span = []
    for target_span in missing_groups:
        target_span = sorted({int(value) for value in target_span})
        predicted_delta = estimate(target_span, range(len(effects)), power)
        if predicted_delta is None:
            return None
        transferred_effect = {}
        span_evidence_query_ids = set()
        for token_id, delta in predicted_delta.items():
            scaled_delta = shrinkage * delta
            transferred_effect[str(token_id)] = scaled_delta
            transferred[token_id] = (
                transferred.get(token_id, 0.0) + scaled_delta
            )
        for effect, training_span in zip(effects, spans):
            if not set(target_span).intersection(training_span):
                query_id = effect.get("queryId")
                if query_id:
                    evidence_query_ids.add(str(query_id))
                    span_evidence_query_ids.add(str(query_id))
        estimated_spans.append(target_span)
        estimated_effects_by_span.append({
            "deletedTokenIndices": target_span,
            "deltaByTokenId": transferred_effect,
            "evidenceQueryIds": sorted(span_evidence_query_ids),
        })

    return {
        "method": "leave-one-intervention-out-calibrated-local-transfer",
        "kernel": "inverse-token-span-distance",
        "distancePower": power,
        "shrinkage": shrinkage,
        "crossValidationBaselineMSE": baseline_error,
        "crossValidationTransferredMSE": calibrated_error,
        "crossValidationRelativeMSEImprovement": (
            (baseline_error - calibrated_error) / baseline_error
        ),
        "crossValidationInterventionCount": validation_targets,
        "estimatedDeletedTokenIndices": estimated_spans,
        "estimatedEffectsBySpan": estimated_effects_by_span,
        "estimatedDeltaByTokenId": transferred,
        "evidenceQueryIds": sorted(evidence_query_ids),
        "interactionEffectsModeled": False,
    }


def _effect_signature_similarity(
    left: Dict[str, Any],
    right: Dict[str, Any],
    baseline_prompt_token_ids: Sequence[int],
) -> float:
    left_indices = tuple(int(value) for value in left["deletedTokenIndices"])
    right_indices = tuple(int(value) for value in right["deletedTokenIndices"])
    if left_indices == right_indices:
        return 1.0
    left_deltas = left["deltaByTokenId"]
    right_deltas = right["deltaByTokenId"]
    token_ids = set(left_deltas).union(right_deltas)
    left_vector = [float(left_deltas.get(token_id, 0.0)) for token_id in token_ids]
    right_vector = [float(right_deltas.get(token_id, 0.0)) for token_id in token_ids]
    left_norm = math.sqrt(sum(value * value for value in left_vector))
    right_norm = math.sqrt(sum(value * value for value in right_vector))
    if left_norm and right_norm:
        effect_similarity = max(
            0.0,
            sum(a * b for a, b in zip(left_vector, right_vector))
            / (left_norm * right_norm),
        )
    else:
        effect_similarity = 0.0

    left_tokens = {
        int(baseline_prompt_token_ids[index])
        for index in left_indices
        if 0 <= index < len(baseline_prompt_token_ids)
    }
    right_tokens = {
        int(baseline_prompt_token_ids[index])
        for index in right_indices
        if 0 <= index < len(baseline_prompt_token_ids)
    }
    token_union = left_tokens.union(right_tokens)
    token_similarity = (
        len(left_tokens.intersection(right_tokens)) / len(token_union)
        if token_union else 0.0
    )
    left_position = sum(left_indices) / len(left_indices) if left_indices else 0.0
    right_position = sum(right_indices) / len(right_indices) if right_indices else 0.0
    position_scale = max(1.0, len(baseline_prompt_token_ids) * 0.2)
    position_similarity = math.exp(
        -abs(left_position - right_position) / position_scale
    )
    return (
        0.55 * effect_similarity
        + 0.25 * token_similarity
        + 0.20 * position_similarity
    )


def _pair_effect_similarity(
    left_groups: Sequence[Sequence[int]],
    right_groups: Sequence[Sequence[int]],
    effects_by_group: Dict[Tuple[int, ...], Dict[str, Any]],
    baseline_prompt_token_ids: Sequence[int],
) -> float:
    left = [tuple(sorted(int(value) for value in group)) for group in left_groups]
    right = [tuple(sorted(int(value) for value in group)) for group in right_groups]
    if len(left) != 2 or len(right) != 2 or any(group not in effects_by_group for group in (*left, *right)):
        return 0.0

    def assigned_similarity(order: Sequence[int]) -> float:
        first = _effect_signature_similarity(
            effects_by_group[left[0]],
            effects_by_group[right[order[0]]],
            baseline_prompt_token_ids,
        )
        second = _effect_signature_similarity(
            effects_by_group[left[1]],
            effects_by_group[right[order[1]]],
            baseline_prompt_token_ids,
        )
        pair_distance = abs(
            abs(left[0][0] - left[1][0])
            - abs(right[order[0]][0] - right[order[1]][0])
        )
        distance_similarity = math.exp(
            -pair_distance / max(1.0, len(baseline_prompt_token_ids) * 0.2)
        )
        return (
            0.85 * math.sqrt(first * second)
            + 0.15 * distance_similarity
        )

    return max(assigned_similarity((0, 1)), assigned_similarity((1, 0)))


def _infer_pair_residuals(
    decision_model: Dict[str, Any],
    applied: Sequence[Dict[str, Any]],
    *,
    additional_effects: Sequence[Dict[str, Any]] = (),
    protected_token_ids: Sequence[int] = (),
) -> Dict[str, Any] | None:
    target_groups = [
        tuple(sorted(int(value) for value in effect["deletedTokenIndices"]))
        for effect in applied
    ]
    if len(target_groups) != 2:
        return None
    effects_by_group = {
        tuple(sorted(int(value) for value in effect["deletedTokenIndices"])): effect
        for effect in decision_model["inputDeletionEffects"]
    }
    for effect in additional_effects:
        group = tuple(sorted(int(value) for value in effect["deletedTokenIndices"]))
        if not group or group in effects_by_group:
            return None
        effects_by_group[group] = effect
    all_corrections = decision_model["measuredPairCorrections"]
    if not all_corrections:
        return None

    def kernel_estimate(
        pair_groups: Sequence[Sequence[int]],
        pair_corrections: Sequence[Dict[str, Any]],
    ) -> Tuple[Dict[str, float], float, float, List[Tuple[float, Dict[str, Any]]]]:
        weighted_residuals: Dict[str, float] = {}
        total_weight = 0.0
        ranked_corrections = []
        for correction in pair_corrections:
            similarity = _pair_effect_similarity(
                pair_groups,
                correction["deletedTokenIndices"],
                effects_by_group,
                decision_model["baselinePromptTokenIds"],
            )
            if similarity < 0.25:
                continue
            ranked_corrections.append((similarity, correction))
        ranked_corrections.sort(key=lambda item: item[0], reverse=True)
        for similarity, correction in ranked_corrections:
            weight = similarity ** 4
            total_weight += weight
            for token_id, residual in correction["residualByTokenId"].items():
                weighted_residuals[token_id] = (
                    weighted_residuals.get(token_id, 0.0)
                    + weight * float(residual)
                )
        if total_weight <= 0.0:
            return {}, 0.0, 0.0, ranked_corrections
        shrinkage = total_weight / (1.0 + total_weight)
        return (
            {
                token_id: (value / total_weight) * shrinkage
                for token_id, value in weighted_residuals.items()
            },
            shrinkage,
            total_weight,
            ranked_corrections,
        )

    candidate_residuals, shrinkage, total_weight, candidates = kernel_estimate(
        target_groups,
        all_corrections,
    )
    if total_weight <= 0.0:
        return None

    baseline_logits = {
        str(token_id): float(logit)
        for token_id, logit in decision_model["baselineLogitsByTokenId"].items()
    }
    frontier_logits = dict(baseline_logits)
    for effect in applied:
        for token_id, delta in effect["deltaByTokenId"].items():
            token_key = str(token_id)
            frontier_logits[token_key] = (
                frontier_logits.get(token_key, 0.0) + float(delta)
            )
    ranked_decisions = sorted(
        frontier_logits,
        key=frontier_logits.__getitem__,
        reverse=True,
    )
    anchor_token_id = ranked_decisions[0] if ranked_decisions else ""
    if not anchor_token_id:
        return None
    alternatives = [
        token_id for token_id in ranked_decisions
        if token_id != anchor_token_id
    ][:3]
    alternatives.extend(
        str(int(token_id))
        for token_id in protected_token_ids
        if str(int(token_id)) in frontier_logits
        and str(int(token_id)) != anchor_token_id
        and str(int(token_id)) not in alternatives
    )
    if not alternatives:
        return None

    loo_pair_estimates = []
    if len(decision_model["measuredPairCorrections"]) >= 3:
        for heldout_correction in decision_model["measuredPairCorrections"]:
            heldout_estimate, _, _, _ = kernel_estimate(
                heldout_correction["deletedTokenIndices"],
                [
                    correction
                    for correction in decision_model["measuredPairCorrections"]
                    if correction.get("queryId") != heldout_correction.get("queryId")
                ],
            )
            if not heldout_estimate:
                continue
            loo_pair_estimates.append((
                heldout_correction["residualByTokenId"],
                heldout_estimate,
            ))

    margin_calibrations = []
    loo_absolute_error = 0.0
    loo_zero_error = 0.0
    loo_margin_count = 0
    for alternative_id in alternatives:
        margin_error = 0.0
        zero_margin_error = 0.0
        margin_count = 0
        for actual, estimate in loo_pair_estimates:
            actual_margin_residual = (
                float(actual.get(anchor_token_id, 0.0))
                - float(actual.get(alternative_id, 0.0))
            )
            predicted_margin_residual = (
                float(estimate.get(anchor_token_id, 0.0))
                - float(estimate.get(alternative_id, 0.0))
            )
            margin_error += abs(
                actual_margin_residual - predicted_margin_residual
            )
            zero_margin_error += abs(actual_margin_residual)
            margin_count += 1
        margin_mae = margin_error / margin_count if margin_count else None
        zero_margin_mae = (
            zero_margin_error / margin_count if margin_count else None
        )
        accepted_margin = bool(
            margin_count >= 3
            and margin_mae is not None
            and zero_margin_mae is not None
            and margin_mae < zero_margin_mae * 0.9
        )
        margin_calibrations.append({
            "alternativeTokenId": int(alternative_id),
            "evaluatedPairs": margin_count,
            "kernelMeanAbsoluteError": margin_mae,
            "zeroResidualMeanAbsoluteError": zero_margin_mae,
            "accepted": accepted_margin,
        })
        loo_absolute_error += margin_error
        loo_zero_error += zero_margin_error
        loo_margin_count += margin_count
    loo_mae = loo_absolute_error / loo_margin_count if loo_margin_count else None
    zero_mae = loo_zero_error / loo_margin_count if loo_margin_count else None
    accepted_alternatives = {
        str(item["alternativeTokenId"])
        for item in margin_calibrations
        if item["accepted"]
    }
    calibrated = bool(accepted_alternatives)
    top_similarity = candidates[0][0]
    anchor_residual = float(candidate_residuals.get(anchor_token_id, 0.0))
    applied_residuals = (
        {
            token_id: (
                float(candidate_residuals.get(token_id, 0.0))
                if token_id == anchor_token_id
                or token_id in accepted_alternatives
                else anchor_residual
            )
            for token_id in frontier_logits
        }
        if calibrated
        else {token_id: 0.0 for token_id in frontier_logits}
    )
    transferred_effect_groups = {
        tuple(sorted(int(value) for value in effect["deletedTokenIndices"])): effect
        for effect in additional_effects
    }
    return {
        "residualByTokenId": applied_residuals,
        "candidateResidualByTokenId": candidate_residuals,
        "applied": calibrated,
        "calibration": {
            "method": (
                "leave-one-pair-out-target-additive-decision-margin-"
                "versus-zero-residual-baseline"
            ),
            "decisionMarginAnchorTokenId": int(anchor_token_id),
            "decisionFrontierSource": (
                "baseline_plus_applied_single_deletion_effects"
            ),
            "alternativeTokenIds": [int(token_id) for token_id in alternatives],
            "evaluatedDecisionMargins": loo_margin_count,
            "kernelMeanAbsoluteError": loo_mae,
            "zeroResidualMeanAbsoluteError": zero_mae,
            "decisionFrontier": margin_calibrations,
            "accepted": calibrated,
            "applicationPolicy": "apply_only_individually_calibrated_frontier_margins",
            "acceptedMarginCount": len(accepted_alternatives),
            "rejectedMarginCount": (
                len(margin_calibrations) - len(accepted_alternatives)
            ),
        },
        "confidence": shrinkage * top_similarity if calibrated else 0.0,
        "effectiveSupport": total_weight,
        "targetComponents": [
            {
                "deletedTokenIndices": list(group),
                "effectSource": (
                    "calibrated_local_intervention_transfer"
                    if group in transferred_effect_groups
                    else "measured_single_intervention"
                ),
                "evidenceQueryIds": (
                    list(transferred_effect_groups[group].get(
                        "evidenceQueryIds",
                        [],
                    ))
                    if group in transferred_effect_groups
                    else [
                        str(effects_by_group[group]["queryId"])
                    ]
                    if effects_by_group[group].get("queryId")
                    else []
                ),
            }
            for group in target_groups
        ],
        "evidence": [
            {
                "queryId": correction.get("queryId"),
                "similarity": similarity,
                "weight": similarity ** 4,
                "deletedTokenIndices": correction["deletedTokenIndices"],
            }
            for similarity, correction in candidates[:4]
        ],
    }


def _prompt_wrapper_regions(
    formatted_prompt: str,
    core_prompt: str,
) -> Tuple[str, str] | None:
    location = formatted_prompt.find(core_prompt)
    if location < 0:
        return None
    return (
        formatted_prompt[:location],
        formatted_prompt[location + len(core_prompt):],
    )


def _character_ngram_similarity(left: str, right: str) -> float:
    def grams(value: str) -> Dict[str, int]:
        normalized = " ".join(value.casefold().split())
        if not normalized:
            return {}
        padded = f"  {normalized}  "
        result: Dict[str, int] = {}
        for index in range(max(1, len(padded) - 2)):
            gram = padded[index:index + 3]
            result[gram] = result.get(gram, 0) + 1
        return result

    left_grams = grams(left)
    right_grams = grams(right)
    if not left_grams or not right_grams:
        return 0.0
    dot = sum(
        count * right_grams.get(gram, 0)
        for gram, count in left_grams.items()
    )
    left_norm = math.sqrt(sum(count * count for count in left_grams.values()))
    right_norm = math.sqrt(sum(count * count for count in right_grams.values()))
    return dot / (left_norm * right_norm)


def _prompt_wrapper_similarity(
    left_prefix: str,
    left_suffix: str,
    right_prefix: str,
    right_suffix: str,
) -> float:
    similarities = []
    if left_prefix and right_prefix:
        similarities.append(
            _character_ngram_similarity(left_prefix, right_prefix)
        )
    if left_suffix and right_suffix:
        similarities.append(
            _character_ngram_similarity(left_suffix, right_suffix)
        )
    return sum(similarities) / len(similarities) if similarities else 0.0


def build_reconstruction(
    *,
    records: Sequence[Dict[str, Any]],
    cache: Dict[Tuple[int, ...], Sequence[float]],
    tokenizer: Any,
    model: Dict[str, Any],
    input_token_ids: Sequence[int],
    input_prompt_text: str,
    input_prompt_format: str,
    top_token_count: int,
    trace: Dict[str, Any],
    decision_models: Sequence[Dict[str, Any]],
    editable_input_token_indices: Sequence[int] | None = None,
    additional_observed_paths: Sequence[Dict[str, Any]] = (),
) -> Dict[str, Any]:
    generated_prefixes: Dict[Tuple[int, ...], List[int]] = {}
    baseline_generated = [int(value) for value in trace["generatedTokenIds"]]
    for step in trace["steps"]:
        prefix = baseline_generated[:step["stepIndex"]]
        generated_prefixes[tuple(step["contextTokenIds"])] = prefix
        for intervention in step["contextTokenInterventions"]:
            context = tuple(intervention["counterfactualContextTokenIds"])
            if not prefix or list(context[-len(prefix):]) == prefix:
                generated_prefixes[context] = prefix
        for interaction in step["pairwiseInterventions"]:
            context = tuple(interaction["counterfactualContextTokenIds"])
            if not prefix or list(context[-len(prefix):]) == prefix:
                generated_prefixes[context] = prefix
        for branch in step["alternativeBranches"]:
            for continuation in branch["continuations"]:
                path = [int(value) for value in continuation["tokenIds"]]
                for length in range(step["stepIndex"] + 1, len(path) + 1):
                    context = tuple([*input_token_ids, *path[:length]])
                    generated_prefixes[context] = path[:length]
    for observed_path in additional_observed_paths:
        path_prompt = [
            int(value) for value in observed_path["inputTokenIds"]
        ]
        generated = [
            int(value) for value in observed_path["generatedTokenIds"]
        ]
        for length in range(len(generated) + 1):
            context = tuple([*path_prompt, *generated[:length]])
            generated_prefixes[context] = generated[:length]
    observations: List[Dict[str, Any]] = []
    seen = set()
    for record in records:
        context = tuple(int(value) for value in record["inputTokenIds"])
        if context in seen:
            continue
        seen.add(context)
        logits = cache[context]
        observations.append({
            "queryId": record["queryId"],
            "contextTokenIds": list(context),
            "generatedPrefixTokenIds": generated_prefixes.get(context, []),
            "topCandidates": _top_candidates(logits, top_token_count),
            "vocabularySize": len(logits),
        })
    query_ids = _query_id_map(records)
    paths = [{
        "id": "baseline",
        "prompt": input_prompt_text,
        "inputTokenIds": [int(value) for value in input_token_ids],
        "generatedTokenIds": baseline_generated,
    }, *additional_observed_paths]
    formulation_experts = []
    for path in paths:
        path_prompt = [int(value) for value in path["inputTokenIds"]]
        path_generated = [int(value) for value in path["generatedTokenIds"]]
        wrapper_regions = _prompt_wrapper_regions(
            path["prompt"],
            input_prompt_text,
        )
        if wrapper_regions is None:
            continue
        prefix_text, suffix_text = wrapper_regions
        expert_observations = []
        for prefix_length in range(len(path_generated) + 1):
            context = tuple([*path_prompt, *path_generated[:prefix_length]])
            query_id = query_ids.get(context)
            if query_id is None or context not in cache:
                continue
            expert_observations.append({
                "generatedPrefixTokenIds": path_generated[:prefix_length],
                "queryId": query_id,
                "topCandidates": _top_candidates(
                    cache[context],
                    max(16, top_token_count),
                ),
            })
        formulation_experts.append({
            "id": path["id"],
            "promptPrefix": prefix_text,
            "promptSuffix": suffix_text,
            "observations": expert_observations,
        })
    return {
        "schemaVersion": 1,
        "kind": "gcl-blackbox-executable-behavior-reconstruction",
        "method": "formulation-conditioned-observation-kernel",
        "model": model,
        "inputTokenIds": [int(value) for value in input_token_ids],
        "inputPromptText": input_prompt_text,
        "inputPromptFormat": input_prompt_format,
        "eosTokenId": getattr(tokenizer, "eos_token_id", None),
        "editableInputScope": trace["editableInputScope"],
        "editableInputTokenIndices": (
            list(range(len(input_token_ids)))
            if editable_input_token_indices is None
            else sorted({int(value) for value in editable_input_token_indices})
        ),
        "tokenizerModelId": model["id"],
        "decodePolicy": "greedy_argmax",
        "observations": observations,
        "promptFormulationExperts": formulation_experts,
        "decisionModels": list(decision_models),
        "domain": {
            "description": (
                "The investigated prompt and nearby token-sequence edits represented "
                "by the recorded oracle contexts. No guarantee is made outside this measured domain."
            ),
            "trainingObservationCount": len(observations),
            "generalizationIsEmpirical": True,
        },
    }


def _cosine_token_similarity(left: Sequence[int], right: Sequence[int]) -> float:
    left_counts: Dict[int, int] = {}
    right_counts: Dict[int, int] = {}
    for token in left:
        left_counts[int(token)] = left_counts.get(int(token), 0) + 1
    for token in right:
        right_counts[int(token)] = right_counts.get(int(token), 0) + 1
    shared = set(left_counts).intersection(right_counts)
    dot = sum(left_counts[token] * right_counts[token] for token in shared)
    left_norm = math.sqrt(sum(value * value for value in left_counts.values()))
    right_norm = math.sqrt(sum(value * value for value in right_counts.values()))
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


def _autoregressive_state_signature(
    context_token_ids: Sequence[int],
    generated_prefix_token_ids: Sequence[int] = (),
    *,
    state_dim: int = 24,
) -> np.ndarray:
    tokens = [int(value) for value in context_token_ids]
    tokens.extend(int(value) for value in generated_prefix_token_ids)
    if not tokens:
        return np.zeros(state_dim, dtype=np.float64)

    state = np.zeros(state_dim, dtype=np.float64)
    for position, token_id in enumerate(tokens):
        token_weight = 1.0 / (position + 1)
        token_vector = np.array(
            [
                math.sin((int(token_id) + 1) * (index + 1) * 0.6180339887498949 + position * 0.37)
                for index in range(state_dim)
            ],
            dtype=np.float64,
        )
        state = np.tanh(state * 0.75 + token_vector * token_weight)
    return state


def _autoregressive_state_kernel_prediction(
    reconstruction: Dict[str, Any],
    context_token_ids: Sequence[int],
    generated_prefix_token_ids: Sequence[int] = (),
    *,
    top_k: int = 8,
) -> Dict[str, Any] | None:
    observations = reconstruction.get("observations", [])
    if not observations:
        return None

    context = [int(value) for value in context_token_ids]
    prefix = [int(value) for value in generated_prefix_token_ids]
    query_state = _autoregressive_state_signature(context, prefix)
    query_norm = float(np.linalg.norm(query_state))
    if query_norm <= 1e-12:
        return None

    candidate_scores: Dict[int, float] = {}
    similarity_rows: List[Tuple[float, str, List[Dict[str, Any]]]] = []
    for item in observations:
        if item.get("contextTokenIds") is None:
            continue
        item_context = [int(value) for value in item["contextTokenIds"]]
        item_prefix = [int(value) for value in item.get("generatedPrefixTokenIds", [])]
        if item_prefix != prefix:
            continue
        item_state = _autoregressive_state_signature(item_context, item_prefix)
        item_norm = float(np.linalg.norm(item_state))
        if item_norm <= 1e-12:
            continue
        similarity = float(np.dot(query_state, item_state) / (query_norm * item_norm))
        if similarity <= 0.0:
            continue
        for candidate in item.get("topCandidates", []):
            token_id = int(candidate["tokenId"])
            candidate_scores[token_id] = candidate_scores.get(token_id, 0.0) + (
                similarity ** 2 * float(candidate.get("probability", 0.0))
            )
        similarity_rows.append((similarity, item.get("queryId", ""), item.get("topCandidates", [])))

    if not candidate_scores:
        return None

    ranked = sorted(candidate_scores.items(), key=lambda pair: pair[1], reverse=True)
    best_token_id, best_score = ranked[0]
    total_weight = sum(score for _, score in ranked[:top_k])
    best_similarity = max(similarity for similarity, _, _ in similarity_rows) if similarity_rows else 0.0
    ordered_candidates = [
        {
            "tokenId": int(token_id),
            "weightedProbability": float(score / total_weight) if total_weight > 0.0 else 0.0,
        }
        for token_id, score in ranked[:10]
    ]
    candidate_log_scores = {
        str(token_id): float(math.log(max(score / total_weight, 1e-12))) if total_weight > 0.0 else 0.0
        for token_id, score in ranked[:top_k]
    }
    return {
        "tokenId": int(best_token_id),
        "confidence": float(best_score / total_weight) if total_weight > 0.0 else 0.0,
        "method": "autoregressive_state_kernel",
        "nearestSimilarity": float(best_similarity),
        "stateSignature": query_state.astype(float).tolist(),
        "evidenceQueryIds": [
            query_id
            for _, query_id, _ in similarity_rows[:max(1, top_k)]
            if query_id
        ],
        "candidateLogScoresByTokenId": candidate_log_scores,
        "candidates": ordered_candidates,
    }


def predict_next_token(
    reconstruction: Dict[str, Any],
    context_token_ids: Sequence[int],
    *,
    prompt_token_ids: Sequence[int] | None = None,
    prompt_surface: str | None = None,
    generated_prefix_token_ids: Sequence[int] | None = None,
    neighbor_count: int = 8,
) -> Dict[str, Any]:
    context = [int(value) for value in context_token_ids]
    if not context:
        raise ValueError("Prediction context must not be empty.")
    observations = reconstruction["observations"]
    candidates = observations
    if generated_prefix_token_ids is not None:
        expected_prefix = [int(value) for value in generated_prefix_token_ids]
        prefix_matches = [
            item for item in observations
            if item.get("generatedPrefixTokenIds", []) == expected_prefix
        ]
        if prefix_matches:
            candidates = prefix_matches
    exact = next(
        (item for item in candidates if item["contextTokenIds"] == context),
        None,
    )
    if exact is not None:
        return {
            "tokenId": exact["topCandidates"][0]["tokenId"],
            "confidence": 1.0,
            "method": "exact_observed_context",
            "nearestSimilarity": 1.0,
            "evidenceQueryIds": [exact["queryId"]],
            "candidates": exact["topCandidates"],
            "candidateLogScoresByTokenId": {
                str(token_id): score
                for token_id, score in _candidate_log_scores({
                    "candidates": exact["topCandidates"],
                }).items()
            },
        }
    if (
        prompt_surface is not None
        and prompt_token_ids is not None
        and generated_prefix_token_ids is not None
    ):
        base_prompt = reconstruction.get("inputPromptText", "")
        query_regions = _prompt_wrapper_regions(
            prompt_surface,
            base_prompt,
        )
        expert_rows = []
        if query_regions is not None:
            query_prefix, query_suffix = query_regions
            prefix = [int(value) for value in generated_prefix_token_ids]
            for expert in reconstruction.get("promptFormulationExperts", []):
                similarity = _prompt_wrapper_similarity(
                    query_prefix,
                    query_suffix,
                    expert["promptPrefix"],
                    expert["promptSuffix"],
                )
                if similarity <= 0.0:
                    continue
                observed = next(
                    (
                        item for item in expert["observations"]
                        if item["generatedPrefixTokenIds"] == prefix
                    ),
                    None,
                )
                if observed is not None:
                    expert_rows.append((similarity, expert, observed))
        if expert_rows:
            expert_rows.sort(key=lambda item: item[0], reverse=True)
            candidate_scores: Dict[int, float] = {}
            total_weight = 0.0
            for similarity, _, observed in expert_rows:
                weight = similarity ** 4
                total_weight += weight
                for candidate in observed["topCandidates"]:
                    token_id = int(candidate["tokenId"])
                    candidate_scores[token_id] = (
                        candidate_scores.get(token_id, 0.0)
                        + weight * float(candidate["probability"])
                    )
            ranked = sorted(
                candidate_scores.items(),
                key=lambda pair: pair[1],
                reverse=True,
            )
            if ranked:
                best_token, best_score = ranked[0]
                return {
                    "tokenId": best_token,
                    "confidence": best_score / total_weight if total_weight else 0.0,
                    "method": "prompt_formulation_kernel_vote",
                    "nearestSimilarity": expert_rows[0][0],
                    "formulationEvidence": [
                        {
                            "formulationId": expert["id"],
                            "similarity": similarity,
                            "queryId": observed["queryId"],
                        }
                        for similarity, expert, observed in expert_rows
                    ],
                    "evidenceQueryIds": [
                        observed["queryId"] for _, _, observed in expert_rows
                    ],
                    "candidateLogScoresByTokenId": {
                        str(token_id): math.log(score / total_weight)
                        for token_id, score in candidate_scores.items()
                        if score > 0.0 and total_weight > 0.0
                    },
                    "candidates": [
                        {
                            "tokenId": token_id,
                            "weightedProbability": score / total_weight,
                        }
                        for token_id, score in ranked[:10]
                    ],
                }
    if prompt_token_ids is not None and generated_prefix_token_ids is not None:
        expected_prefix = [int(value) for value in generated_prefix_token_ids]
        decision_model = next(
            (
                item for item in reconstruction.get("decisionModels", [])
                if item["generatedPrefixTokenIds"] == expected_prefix
            ),
            None,
        )
        if decision_model is not None:
            removed = _deleted_indices(
                decision_model["baselinePromptTokenIds"],
                [int(value) for value in prompt_token_ids],
            )
            if removed is not None:
                applied: List[Dict[str, Any]] = []
                covered = set()
                partial_overlap = False
                for effect in decision_model["inputDeletionEffects"]:
                    group = set(effect["deletedTokenIndices"])
                    overlap = group.intersection(removed)
                    if overlap and not group.issubset(removed):
                        partial_overlap = True
                        continue
                    if group and group.issubset(removed):
                        applied.append(effect)
                        covered.update(group)
                missing_groups = _contiguous_index_groups(
                    sorted(set(removed) - covered)
                )
                editable_positions = {
                    int(value)
                    for value in reconstruction.get(
                        "editableInputTokenIndices",
                        range(len(decision_model["baselinePromptTokenIds"])),
                    )
                }
                transfer = (
                    _calibrated_local_deletion_effects(
                        decision_model,
                        missing_groups,
                    )
                    if set(removed).issubset(editable_positions)
                    else None
                )
                supported = not partial_overlap and not missing_groups
                if supported or transfer is not None:
                    scores = {
                        int(token_id): float(score)
                        for token_id, score in decision_model[
                            "baselineLogitsByTokenId"
                        ].items()
                    }
                    evidence_ids = [decision_model["baselineQueryId"]]
                    for effect in applied:
                        evidence_ids.append(effect["queryId"])
                        for token_id, delta in effect["deltaByTokenId"].items():
                            token_id = int(token_id)
                            scores[token_id] = (
                                scores.get(token_id, 0.0) + float(delta)
                            )
                    measured_correction = False
                    applied_groups = {
                        tuple(sorted(effect["deletedTokenIndices"]))
                        for effect in applied
                    }
                    if not missing_groups and not partial_overlap:
                        for correction in decision_model["measuredPairCorrections"]:
                            if all(
                                tuple(sorted(group)) in applied_groups
                                for group in correction["deletedTokenIndices"]
                            ):
                                evidence_ids.append(correction["queryId"])
                                for token_id, residual in correction[
                                    "residualByTokenId"
                                ].items():
                                    token_id = int(token_id)
                                    scores[token_id] = (
                                        scores.get(token_id, 0.0)
                                        + float(residual)
                                    )
                                measured_correction = True
                    inferred_interaction = None
                    if not measured_correction and not partial_overlap:
                        interaction_effects = list(applied)
                        transferred_components = []
                        if missing_groups:
                            transferred_components = list(
                                (transfer or {}).get(
                                    "estimatedEffectsBySpan",
                                    [],
                                )
                            )
                            if len(transferred_components) != len(missing_groups):
                                transferred_components = []
                            interaction_effects.extend(transferred_components)
                        if len(interaction_effects) != 2:
                            interaction_effects = []
                    else:
                        interaction_effects = []
                        transferred_components = []
                    if interaction_effects:
                        inferred_interaction = _infer_pair_residuals(
                            decision_model,
                            interaction_effects,
                            additional_effects=transferred_components,
                            protected_token_ids=(
                                [int(reconstruction["eosTokenId"])]
                                if reconstruction.get("eosTokenId") is not None
                                else []
                            ),
                        )
                        if inferred_interaction is not None and inferred_interaction[
                            "applied"
                        ]:
                            for token_id, residual in inferred_interaction[
                                "residualByTokenId"
                            ].items():
                                token_id = int(token_id)
                                scores[token_id] = (
                                    scores.get(token_id, 0.0) + float(residual)
                                )
                            evidence_ids.extend(
                                item["queryId"]
                                for item in inferred_interaction["evidence"]
                                if item["queryId"]
                            )
                    if transfer is not None:
                        evidence_ids.extend(transfer["evidenceQueryIds"])
                        for token_id, delta in transfer[
                            "estimatedDeltaByTokenId"
                        ].items():
                            scores[int(token_id)] = (
                                scores.get(int(token_id), 0.0) + float(delta)
                            )
                    if scores:
                        ranked = sorted(
                            scores.items(),
                            key=lambda pair: pair[1],
                            reverse=True,
                        )
                        return {
                            "tokenId": int(ranked[0][0]),
                            "confidence": (
                                float(
                                    transfer[
                                        "crossValidationRelativeMSEImprovement"
                                    ]
                                )
                                if transfer is not None
                                else
                                1.0
                                if measured_correction
                                else float(inferred_interaction["confidence"])
                                if inferred_interaction is not None
                                and inferred_interaction["applied"]
                                else 0.65
                            ),
                            "confidenceKind": (
                                "cross_validation_relative_mse_improvement"
                                if transfer is not None
                                else "predictor_default"
                            ),
                            "method": (
                                "calibrated_local_intervention_and_pair_interaction_transfer"
                                if transfer is not None
                                and inferred_interaction is not None
                                and inferred_interaction["applied"]
                                else "calibrated_local_intervention_transfer"
                                if transfer is not None
                                else
                                "measured_pair_intervention_model"
                                if measured_correction
                                else "similarity_weighted_pair_interaction_model"
                                if inferred_interaction is not None
                                and inferred_interaction["applied"]
                                else "additive_single_intervention_model"
                            ),
                            "interactionEstimate": (
                                {
                                    "method": "cross-pair-shrunk-kernel",
                                    **inferred_interaction,
                                }
                                if inferred_interaction is not None else None
                            ),
                            "interventionTransferEstimate": transfer,
                            "nearestSimilarity": 1.0,
                            "evidenceQueryIds": [
                                value for value in evidence_ids if value
                            ],
                            "candidateLogScoresByTokenId": {
                                str(token_id): float(score)
                                for token_id, score in scores.items()
                            },
                            "candidates": [
                                {"tokenId": int(token_id), "relativeLogit": score}
                                for token_id, score in ranked[:10]
                            ],
                        }
    if generated_prefix_token_ids:
        state_kernel_prediction = _autoregressive_state_kernel_prediction(
            reconstruction,
            context,
            generated_prefix_token_ids,
        )
        if state_kernel_prediction is not None:
            return state_kernel_prediction
    neighbors = sorted(
        (
            (_cosine_token_similarity(context, item["contextTokenIds"]), item)
            for item in candidates
        ),
        key=lambda pair: pair[0],
        reverse=True,
    )[:max(1, neighbor_count)]
    neighbors = [(score, item) for score, item in neighbors if score > 0.0]
    if not neighbors:
        raise ValueError("Reconstruction has no similar observed context for this input.")
    candidate_scores: Dict[int, float] = {}
    total_weight = 0.0
    for similarity, item in neighbors:
        weight = similarity ** 4
        total_weight += weight
        for candidate in item["topCandidates"]:
            token_id = int(candidate["tokenId"])
            candidate_scores[token_id] = candidate_scores.get(token_id, 0.0) + (
                weight * float(candidate["probability"])
            )
    ranked = sorted(candidate_scores.items(), key=lambda pair: pair[1], reverse=True)
    best_token, best_score = ranked[0]
    return {
        "tokenId": best_token,
        "confidence": best_score / total_weight if total_weight else 0.0,
        "method": "similarity_weighted_observation_vote",
        "nearestSimilarity": neighbors[0][0],
        "evidenceQueryIds": [item["queryId"] for _, item in neighbors],
        "candidateLogScoresByTokenId": {
            str(token_id): math.log(score / total_weight)
            for token_id, score in candidate_scores.items()
            if score > 0.0 and total_weight > 0.0
        },
        "candidates": [
            {"tokenId": token_id, "weightedProbability": score / total_weight}
            for token_id, score in ranked[:10]
        ],
    }


def generate_reconstruction(
    reconstruction: Dict[str, Any],
    input_token_ids: Sequence[int],
    tokenizer: Any,
    *,
    max_new_tokens: int,
    prompt_text: str | None = None,
    beam_width: int = 1,
) -> Dict[str, Any]:
    if beam_width < 1:
        raise ValueError("beam_width must be positive.")
    if beam_width > 1:
        return _generate_reconstruction_beam(
            reconstruction,
            input_token_ids,
            tokenizer,
            max_new_tokens=max_new_tokens,
            prompt_text=prompt_text,
            beam_width=beam_width,
        )
    context = [int(value) for value in input_token_ids]
    generated: List[int] = []
    predictions: List[Dict[str, Any]] = []
    prompt_surface = prompt_text
    if prompt_surface is None:
        prompt_surface = tokenizer.decode(
            context,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    eos_id = getattr(tokenizer, "eos_token_id", None)
    for _ in range(max_new_tokens):
        prediction = predict_next_token(
            reconstruction,
            context,
            prompt_token_ids=input_token_ids,
            prompt_surface=prompt_surface,
            generated_prefix_token_ids=generated,
        )
        token_id = int(prediction["tokenId"])
        predictions.append(prediction)
        generated.append(token_id)
        context.append(token_id)
        if eos_id is not None and token_id == int(eos_id):
            break
    terminated_by_eos = bool(
        eos_id is not None
        and generated
        and generated[-1] == int(eos_id)
    )
    return {
        "tokenIds": generated,
        "text": tokenizer.decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ),
        "predictions": predictions,
        "terminatedByEos": terminated_by_eos,
        "terminationReason": "eos" if terminated_by_eos else "max_new_tokens",
        "outputComplete": terminated_by_eos,
    }


def _prediction_distribution(
    prediction: Dict[str, Any],
) -> List[Tuple[int, float]]:
    candidates = prediction.get("candidates", [])
    if not candidates:
        return [(int(prediction["tokenId"]), 1.0)]
    if any("relativeLogit" in item for item in candidates):
        scores = [
            float(item.get("relativeLogit", -1e9))
            for item in candidates
        ]
        maximum = max(scores)
        values = [math.exp(score - maximum) for score in scores]
    else:
        values = [
            max(
                0.0,
                float(item.get(
                    "weightedProbability",
                    item.get("probability", 0.0),
                )),
            )
            for item in candidates
        ]
    total = sum(values)
    if total <= 0.0:
        return [(int(prediction["tokenId"]), 1.0)]
    distribution = [
        (int(item["tokenId"]), value / total)
        for item, value in zip(candidates, values)
        if value > 0.0
    ]
    if not any(token_id == int(prediction["tokenId"]) for token_id, _ in distribution):
        distribution.insert(0, (int(prediction["tokenId"]), 0.0))
    return distribution


def _causal_state_estimator(
    decision_model: Dict[str, Any],
    removed: Sequence[int],
) -> Dict[int, float]:
    removed_set = {int(index) for index in removed}
    baseline = {
        int(token_id): float(logit)
        for token_id, logit in decision_model["baselineLogitsByTokenId"].items()
    }
    token_ids = sorted(baseline)
    predicted = dict(baseline)
    for token_id in token_ids:
        single_effect = 0.0
        for effect in decision_model.get("inputDeletionEffects", []):
            effect_groups = {
                int(index)
                for index in effect.get("deletedTokenIndices", [])
            }
            if not effect_groups.issubset(removed_set):
                continue
            delta = float(effect.get("deltaByTokenId", {}).get(str(token_id), 0.0))
            single_effect += delta / max(1, len(effect_groups))
        pair_effect = 0.0
        for correction in decision_model.get("measuredPairCorrections", []):
            groups = [
                {int(index) for index in group}
                for group in correction.get("deletedTokenIndices", [])
            ]
            if not groups or not all(group.issubset(removed_set) for group in groups):
                continue
            residual = float(
                correction.get("residualByTokenId", {}).get(str(token_id), 0.0)
            )
            pair_effect += residual / max(1, len(groups))
        predicted[token_id] = baseline[token_id] + single_effect + 0.5 * pair_effect
    return predicted


def _context_group_similarity(
    deleted_token_indices: Sequence[int],
    candidate_context: Sequence[int],
) -> float:
    deleted = {int(value) for value in deleted_token_indices}
    context = {int(value) for value in candidate_context}
    if not deleted:
        return 0.0
    union = deleted.union(context)
    if not union:
        return 0.0
    return len(deleted.intersection(context)) / len(union)


def _causal_transition_prediction(
    decision_model: Dict[str, Any],
    context_token_ids: Sequence[int],
    generated_prefix_token_ids: Sequence[int],
) -> Dict[str, Any] | None:
    if not decision_model or not decision_model.get("baselineLogitsByTokenId"):
        return None
    baseline_logits = {
        int(token_id): float(logit)
        for token_id, logit in decision_model["baselineLogitsByTokenId"].items()
    }
    if not baseline_logits:
        return None
    context_view = {int(value) for value in context_token_ids}
    prefix_view = {int(value) for value in generated_prefix_token_ids}
    ranked_scores: Dict[int, float] = {}
    evidence_query_ids: List[str] = []
    for candidate_token_id, baseline_logit in baseline_logits.items():
        additive = 0.0
        interaction = 0.0
        for effect in decision_model.get("inputDeletionEffects", []):
            deleted = [int(value) for value in effect.get("deletedTokenIndices", [])]
            if not deleted:
                continue
            similarity = _context_group_similarity(deleted, context_view)
            if similarity <= 0.0:
                continue
            additive += similarity * float(
                effect.get("deltaByTokenId", {}).get(str(candidate_token_id), 0.0)
            )
            evidence_query_ids.append(str(effect.get("queryId", "")))
        for correction in decision_model.get("measuredPairCorrections", []):
            groups = correction.get("deletedTokenIndices", [])
            if not groups:
                continue
            pair_similarity = 0.0
            for group in groups:
                pair_similarity += _context_group_similarity(group, context_view)
            pair_similarity /= max(1, len(groups))
            if pair_similarity <= 0.0:
                continue
            interaction += pair_similarity * float(
                correction.get("residualByTokenId", {}).get(str(candidate_token_id), 0.0)
            )
            evidence_query_ids.append(str(correction.get("queryId", "")))
        prefix_signal = 0.0
        if candidate_token_id in prefix_view:
            prefix_signal += 0.25
        ranked_scores[candidate_token_id] = (
            baseline_logit + additive + 0.5 * interaction + prefix_signal
        )
    if not ranked_scores:
        return None
    ordered = sorted(ranked_scores.items(), key=lambda pair: pair[1], reverse=True)
    best_token_id, best_score = ordered[0]
    normalized = [
        {
            "tokenId": int(token_id),
            "relativeLogit": score - ordered[0][1],
        }
        for token_id, score in ordered[:10]
    ]
    return {
        "tokenId": int(best_token_id),
        "confidence": 0.7 + min(0.29, max(0.0, (best_score - ordered[-1][1]) / max(1.0, abs(best_score) + 1.0))),
        "method": "causal_state_transition_model",
        "nearestSimilarity": 1.0,
        "evidenceQueryIds": [
            query_id for query_id in dict.fromkeys(evidence_query_ids) if query_id
        ],
        "candidateLogScoresByTokenId": {
            str(token_id): float(score - ordered[0][1])
            for token_id, score in ordered[:10]
        },
        "candidates": normalized,
    }


def _generate_reconstruction_beam(
    reconstruction: Dict[str, Any],
    input_token_ids: Sequence[int],
    tokenizer: Any,
    *,
    max_new_tokens: int,
    prompt_text: str | None,
    beam_width: int,
) -> Dict[str, Any]:
    prompt_ids = [int(value) for value in input_token_ids]
    prompt_surface = prompt_text
    if prompt_surface is None:
        prompt_surface = tokenizer.decode(
            prompt_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    eos_id = getattr(tokenizer, "eos_token_id", None)
    beams = [{
        "tokenIds": [],
        "score": 0.0,
        "predictions": [],
        "finished": False,
    }]
    for _ in range(max_new_tokens):
        expanded = []
        for beam in beams:
            if beam["finished"]:
                expanded.append(beam)
                continue
            context = [*prompt_ids, *beam["tokenIds"]]
            prediction = predict_next_token(
                reconstruction,
                context,
                prompt_token_ids=prompt_ids,
                prompt_surface=prompt_surface,
                generated_prefix_token_ids=beam["tokenIds"],
            )
            for token_id, probability in _prediction_distribution(prediction)[:beam_width]:
                probability = max(probability, 1e-12)
                token_ids = [*beam["tokenIds"], token_id]
                expanded.append({
                    "tokenIds": token_ids,
                    "score": beam["score"] + math.log(probability),
                    "predictions": [*beam["predictions"], prediction],
                    "finished": eos_id is not None and token_id == int(eos_id),
                })
        beams = sorted(
            expanded,
            key=lambda item: item["score"],
            reverse=True,
        )[:beam_width]
        if beams and all(item["finished"] for item in beams):
            break
    candidates = [
        {
            "tokenIds": item["tokenIds"],
            "text": tokenizer.decode(
                item["tokenIds"],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ),
            "cumulativeLogCandidateScore": item["score"],
            "predictions": item["predictions"],
            "terminatedByEos": item["finished"],
            "terminationReason": (
                "eos" if item["finished"] else "max_new_tokens"
            ),
        }
        for item in beams
    ]
    best = candidates[0] if candidates else {
        "tokenIds": [],
        "text": "",
        "cumulativeLogCandidateScore": 0.0,
        "predictions": [],
    }
    return {
        "tokenIds": best["tokenIds"],
        "text": best["text"],
        "predictions": best["predictions"],
        "terminatedByEos": best.get("terminatedByEos", False),
        "terminationReason": best.get("terminationReason", "max_new_tokens"),
        "outputComplete": best.get("terminatedByEos", False),
        "beamWidth": beam_width,
        "candidateResponses": candidates,
    }


def _query_id_for(
    context: Sequence[int],
    records_by_context: Dict[Tuple[int, ...], str],
) -> List[str]:
    query_id = records_by_context.get(tuple(int(value) for value in context))
    return [query_id] if query_id else []


def build_behavior_graph(
    trace: Dict[str, Any],
    *,
    input_token_ids: Sequence[int],
    tokenizer: Any,
    records: Sequence[Dict[str, Any]],
    editable_input_token_indices: Sequence[int] | None = None,
) -> Dict[str, Any]:
    query_ids = {
        tuple(int(value) for value in row["inputTokenIds"]): row["queryId"]
        for row in records
    }
    editable_positions = (
        set(range(len(input_token_ids)))
        if editable_input_token_indices is None
        else {int(position) for position in editable_input_token_indices}
    )
    variables = [
        {
            "id": f"input-{index}",
            "tokenIndex": index,
            "tokenId": int(token_id),
            "surface": tokenizer.decode(
                [int(token_id)],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            ),
        }
        for index, token_id in enumerate(input_token_ids)
        if index in editable_positions
    ]
    decisions: List[Dict[str, Any]] = []
    dependencies: List[Dict[str, Any]] = []
    interactions: List[Dict[str, Any]] = []
    alternatives: List[Dict[str, Any]] = []
    for step in trace["steps"]:
        context = step["contextTokenIds"]
        node_id = f"token-{step['stepIndex']}"
        baseline_queries = _query_id_for(context, query_ids)
        decisions.append({
            "id": node_id,
            "stepIndex": step["stepIndex"],
            "contextTokenIds": context,
            "chosenTokenId": step["chosenTokenId"],
            "chosenToken": step["chosenToken"],
            "probability": step["chosenTokenProbability"],
            "marginVsTopAlternatives": step["marginVsTopAlternatives"],
            "alternatives": step["topAlternatives"],
            "evidenceQueryIds": baseline_queries,
        })
        for index, intervention in enumerate(step["contextTokenInterventions"]):
            counterfactual = intervention["counterfactualContextTokenIds"]
            dependencies.append({
                "id": f"dependency-{step['stepIndex']}-{index}",
                "source": intervention["source"],
                "fromContextTokenIndices": intervention["contextTokenIndices"],
                "fromInputVariables": [
                    f"input-{position}"
                    for position in intervention["contextTokenIndices"]
                    if intervention["source"] == "input"
                    and position < len(input_token_ids)
                ],
                "toDecision": node_id,
                "effect": "supports" if intervention["marginSupport"] > 0
                else "suppresses" if intervention["marginSupport"] < 0
                else "no_measured_margin_change",
                "marginSupport": intervention["marginSupport"],
                "measurement": "single_deletion_at_fixed_autoregressive_prefix",
                "evidenceQueryIds": _query_id_for(counterfactual, query_ids),
                "confidence": {
                    "level": "direct_measurement",
                    "scope": "exact_token_deletion_and_exact_prefix",
                },
            })
        for index, interaction in enumerate(step["pairwiseInterventions"]):
            interactions.append({
                "id": f"interaction-{step['stepIndex']}-{index}",
                "deletedUnits": interaction["deletedUnits"],
                "toDecision": node_id,
                "interactionResidual": interaction["interactionResidual"],
                "measurement": "joint_deletion_minus_additive_single_deletion_margin",
                "evidenceQueryIds": _query_id_for(
                    interaction["counterfactualContextTokenIds"], query_ids
                ),
                "confidence": {
                    "level": "direct_measurement",
                    "scope": "exact_pair_and_exact_prefix",
                },
            })
        for branch in step["alternativeBranches"]:
            alternatives.append({
                "atDecision": node_id,
                "forcedTokenId": branch["forcedTokenId"],
                "forcedTokenRank": branch["forcedTokenRank"],
                "continuations": branch["continuations"],
                "evidenceQueryIds": [
                    query
                    for continuation in branch["continuations"]
                    for length in range(
                        step["stepIndex"] + 1,
                        len(continuation["tokenIds"]) + 1,
                    )
                    for query in _query_id_for(
                        [
                            *input_token_ids,
                            *continuation["tokenIds"][:length],
                        ],
                        query_ids,
                    )
                ],
            })
    return {
        "inputVariables": variables,
        "editableInputTokenIndices": sorted(editable_positions),
        "editableInputScope": trace["editableInputScope"],
        "fixedPromptContextTokenIndices": sorted(
            set(range(len(input_token_ids))) - editable_positions
        ),
        "tokenPredictions": decisions,
        "conditionalDependencies": dependencies,
        "interactions": interactions,
        "alternativeRoutes": alternatives,
    }


def explain_decompilation(graph: Dict[str, Any], trace: Dict[str, Any]) -> Dict[str, Any]:
    what = []
    how = []
    why = []
    for index, decision in enumerate(graph["tokenPredictions"]):
        cited = decision["evidenceQueryIds"]
        how.append({
            "decision": decision["id"],
            "token": decision["chosenToken"],
            "tokenId": decision["chosenTokenId"],
            "probability": decision["probability"],
            "prefixTokenIds": decision["contextTokenIds"],
            "queryIds": cited,
        })
        for edge in graph["conditionalDependencies"]:
            if edge["toDecision"] != decision["id"]:
                continue
            why.append({
                "decision": decision["id"],
                "source": edge["source"],
                "contextTokenIndices": edge["fromContextTokenIndices"],
                "effect": edge["effect"],
                "marginSupport": edge["marginSupport"],
                "queryIds": edge["evidenceQueryIds"],
                "status": "measured",
            })
        if index < len(trace["steps"]):
            sufficient = trace["steps"][index]["minimalSufficientContext"]
            what.append({
                "decision": decision["id"],
                "retainedContextTokenIds": sufficient["retainedContextTokenIds"],
                "oneMinimalAmongEditableUnits": (
                    sufficient["minimality"]["oneMinimalAmongEditableUnits"]
                ),
                "globalMinimumProven": False,
                "queryIds": cited,
            })
    return {
        "what": what,
        "how": how,
        "why": why,
        "interactions": graph["interactions"],
        "alternatives": graph["alternativeRoutes"],
        "counterfactuals": trace["responseCounterfactuals"],
        "uncertainty": [
            {
                "status": "unresolved",
                "claim": "This evidence does not expose or uniquely identify internal reasoning.",
            },
            {
                "status": "measured_scope_only",
                "claim": "Dependencies apply only to tested token edits and autoregressive prefixes.",
            },
            {
                "status": "empirical",
                "claim": "Executable reconstruction fidelity is limited to the validation results.",
            },
        ],
    }


def _generate_oracle(
    query: Callable[[Sequence[int]], Any],
    prompt_ids: Sequence[int],
    tokenizer: Any,
    max_new_tokens: int,
) -> Tuple[List[int], List[int]]:
    context = [int(value) for value in prompt_ids]
    generated: List[int] = []
    winners: List[int] = []
    eos_id = getattr(tokenizer, "eos_token_id", None)
    for _ in range(max_new_tokens):
        logits = _logit_values(query(context))
        winner = max(range(len(logits)), key=logits.__getitem__)
        winners.append(winner)
        generated.append(winner)
        context.append(winner)
        if eos_id is not None and winner == int(eos_id):
            break
    return generated, winners


def _training_formulations(prompt: str) -> List[Dict[str, str]]:
    return [
        {
            "id": "training-formulation-request-wrapper",
            "prompt": (
                f"Please answer the following request:\n\n{prompt}"
                "\n\nAnswer concisely."
            ),
        },
        {
            "id": "training-formulation-task-label",
            "prompt": f"Request:\n{prompt}\n\nResponse:",
        },
    ]


def _heldout_formulation(prompt: str) -> str:
    return f"Instruction: {prompt}\n\nProvide a concise answer."


def _dependency_prediction_metrics(
    case_reports: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    assessments = [
        assessment
        for case in case_reports
        for assessment in case.get("dependencyAssessments", [])
    ]
    method_counts: Dict[str, int] = {}
    for assessment in assessments:
        method = str(assessment.get("reconstructionMethod") or "unknown")
        method_counts[method] = method_counts.get(method, 0) + 1
    intervention_methods = {
        "additive_single_intervention_model",
        "measured_pair_intervention_model",
        "similarity_weighted_pair_interaction_model",
        "calibrated_local_intervention_transfer",
        "calibrated_local_intervention_and_pair_interaction_transfer",
    }
    intervention_predictions = sum(
        count
        for method, count in method_counts.items()
        if method in intervention_methods
    )
    return {
        "methodCounts": method_counts,
        "interventionModelCoverage": (
            intervention_predictions / len(assessments) if assessments else None
        ),
    }


def _validation_metrics(
    reconstruction: Dict[str, Any],
    *,
    prompt_cases: Sequence[Dict[str, Any]],
    query: Callable[[Sequence[int]], Any],
    oracle_records: Sequence[Dict[str, Any]],
    tokenizer: Any,
    max_new_tokens: int,
    baseline_token_id: int,
) -> Dict[str, Any]:
    case_reports: List[Dict[str, Any]] = []
    token_matches = 0
    token_total = 0
    rollout_token_matches = 0
    top_five_agreement_sum = 0.0
    top_five_evaluation_total = 0
    top_five_oracle_top_token_covered = 0
    teacher_forced_distribution_fidelity_sum = 0.0
    teacher_forced_distribution_total = 0
    eos_prediction_matches = 0
    eos_prediction_total = 0
    sequence_edit_distances: List[int] = []
    sequence_edit_similarities: List[float] = []
    response_exact = 0
    complete_response_total = 0
    beam_candidate_response_matches = 0
    beam_complete_response_matches = 0
    counterfactual_effect_matches = 0
    counterfactual_effect_total = 0
    dependency_matches = 0
    dependency_total = 0
    autoregressive_dependency_matches = 0
    autoregressive_dependency_total = 0
    interaction_reconstruction_matches = 0
    interaction_reconstruction_total = 0
    interaction_sign_matches = 0
    interaction_count = 0
    interaction_absolute_errors: List[float] = []
    validation_query_count = 0
    for case in prompt_cases:
        prompt_ids = [int(value) for value in case["inputTokenIds"]]
        actual_generated: List[int] = []
        context = list(prompt_ids)
        predictions = []
        actual_winners = []
        case_top_five_agreement_sum = 0.0
        case_top_five_evaluation_total = 0
        case_top_five_oracle_top_token_covered = 0
        case_distribution_fidelity_sum = 0.0
        case_distribution_total = 0
        case_eos_prediction_matches = 0
        case_eos_prediction_total = 0
        dependency_assessments: List[Dict[str, Any]] = []
        autoregressive_dependency_assessments: List[Dict[str, Any]] = []
        interaction_assessment = None
        evidence_contexts: List[List[int]] = []
        for _ in range(max_new_tokens):
            evidence_contexts.append(list(context))
            logits = _logit_values(query(context))
            validation_query_count += 1
            actual_winner = max(range(len(logits)), key=logits.__getitem__)
            conditional_prediction = predict_next_token(
                reconstruction,
                context,
                prompt_token_ids=prompt_ids,
                prompt_surface=case.get("prompt"),
                generated_prefix_token_ids=actual_generated,
            )
            predicted_winner = int(conditional_prediction["tokenId"])
            actual_top_five = sorted(
                range(len(logits)),
                key=logits.__getitem__,
                reverse=True,
            )[:5]
            candidate_scores = _candidate_log_scores(conditional_prediction)
            predicted_top_five = sorted(
                candidate_scores,
                key=candidate_scores.__getitem__,
                reverse=True,
            )[:5]
            top_five_agreement = (
                len(set(actual_top_five) & set(predicted_top_five))
                / max(len(actual_top_five), 1)
            )
            top_five_agreement_sum += top_five_agreement
            top_five_evaluation_total += 1
            case_top_five_agreement_sum += top_five_agreement
            case_top_five_evaluation_total += 1
            oracle_top_token_covered = bool(
                actual_top_five
                and actual_top_five[0] in predicted_top_five
            )
            top_five_oracle_top_token_covered += int(oracle_top_token_covered)
            case_top_five_oracle_top_token_covered += int(
                oracle_top_token_covered
            )
            union_top_five = sorted(set(actual_top_five) | set(predicted_top_five))
            distribution_js_bits = None
            if predicted_top_five and union_top_five:
                distribution_js_bits = _js_divergence(
                    [float(logits[token_id]) for token_id in union_top_five],
                    [
                        float(candidate_scores.get(token_id, -math.inf))
                        for token_id in union_top_five
                    ],
                )
                fidelity = 1.0 - distribution_js_bits
                teacher_forced_distribution_fidelity_sum += fidelity
                teacher_forced_distribution_total += 1
                case_distribution_fidelity_sum += fidelity
                case_distribution_total += 1
            eos_id = getattr(tokenizer, "eos_token_id", None)
            if eos_id is not None:
                eos_prediction_matches += int(
                    (actual_winner == int(eos_id))
                    == (predicted_winner == int(eos_id))
                )
                eos_prediction_total += 1
                case_eos_prediction_matches += int(
                    (actual_winner == int(eos_id))
                    == (predicted_winner == int(eos_id))
                )
                case_eos_prediction_total += 1
            actual_winners.append(actual_winner)
            predictions.append({
                "conditional": conditional_prediction,
                "rollout": None,
                "oracleTopFiveTokenIds": actual_top_five,
                "predictedTopFiveTokenIds": predicted_top_five,
                "topFiveAgreement": top_five_agreement,
                "oracleTopTokenCoveredByTopFive": oracle_top_token_covered,
                "teacherForcedTopFiveJensenShannonBits": distribution_js_bits,
            })
            token_total += 1
            token_matches += int(actual_winner == predicted_winner)
            actual_generated.append(actual_winner)
            context.append(actual_winner)
            if eos_id is not None and actual_winner == int(eos_id):
                break
        rollout_generation = generate_reconstruction(
            reconstruction,
            prompt_ids,
            tokenizer,
            max_new_tokens=max_new_tokens,
            prompt_text=case.get("prompt"),
        )
        surrogate_generated = [
            int(token_id) for token_id in rollout_generation["tokenIds"]
        ]
        for index, rollout_prediction in enumerate(
            rollout_generation["predictions"]
        ):
            if index < len(predictions):
                predictions[index]["rollout"] = rollout_prediction
        rollout_token_matches += sum(
            int(index < len(surrogate_generated)
                and actual_token == surrogate_generated[index])
            for index, actual_token in enumerate(actual_generated)
        )
        sequence_edit_distance = _sequence_edit_distance(
            actual_generated,
            surrogate_generated,
        )
        sequence_edit_distances.append(sequence_edit_distance)
        sequence_edit_similarities.append(
            1.0 - sequence_edit_distance
            / max(len(actual_generated), len(surrogate_generated), 1)
        )
        eos_id = getattr(tokenizer, "eos_token_id", None)
        actual_terminated_by_eos = bool(
            eos_id is not None
            and actual_generated
            and actual_generated[-1] == int(eos_id)
        )
        predicted_terminated_by_eos = bool(
            rollout_generation.get("terminatedByEos", False)
        )
        full_response_match = None
        if actual_terminated_by_eos:
            complete_response_total += 1
            full_response_match = (
                actual_generated == surrogate_generated
                and predicted_terminated_by_eos
            )
            response_exact += int(full_response_match)
        if case.get("role") in {
            "heldout_counterfactual",
            "heldout_interaction",
        } and actual_winners:
            if case["role"] == "heldout_counterfactual":
                actual_changed = actual_winners[0] != baseline_token_id
                surrogate_baseline_prediction = predict_next_token(
                    reconstruction,
                    case["baselineTokenIds"],
                    prompt_token_ids=case["baselineTokenIds"],
                    generated_prefix_token_ids=[],
                )
                surrogate_baseline = int(
                    surrogate_baseline_prediction["tokenId"]
                )
                surrogate_changed = (
                    int(predictions[0]["conditional"]["tokenId"])
                    != int(surrogate_baseline)
                )
                counterfactual_effect_total += 1
                counterfactual_effect_matches += int(
                    actual_changed == surrogate_changed
                )
                baseline_prompt_ids = [
                    int(value) for value in case["baselineTokenIds"]
                ]
                for token_index, actual_counterfactual_winner in enumerate(
                    actual_winners
                ):
                    generated_prefix = actual_generated[:token_index]
                    counterfactual_context = prompt_ids + generated_prefix
                    baseline_context = baseline_prompt_ids + generated_prefix
                    evidence_contexts.append(list(baseline_context))
                    baseline_logits = _logit_values(query(baseline_context))
                    validation_query_count += 1
                    actual_baseline_winner = max(
                        range(len(baseline_logits)),
                        key=baseline_logits.__getitem__,
                    )
                    counterfactual_prediction = predictions[token_index][
                        "conditional"
                    ]
                    baseline_prediction = predict_next_token(
                        reconstruction,
                        baseline_context,
                        prompt_token_ids=baseline_prompt_ids,
                        prompt_surface=case.get("baselinePrompt"),
                        generated_prefix_token_ids=generated_prefix,
                    )
                    actual_dependency = (
                        actual_counterfactual_winner != actual_baseline_winner
                    )
                    reconstructed_dependency = (
                        int(counterfactual_prediction["tokenId"])
                        != int(baseline_prediction["tokenId"])
                    )
                    autoregressive_dependency_total += 1
                    autoregressive_dependency_matches += int(
                        actual_dependency == reconstructed_dependency
                    )
                    autoregressive_dependency_assessments.append({
                        "tokenIndex": token_index,
                        "generatedPrefixTokenIds": generated_prefix,
                        "actualCounterfactualTokenId": (
                            actual_counterfactual_winner
                        ),
                        "actualBaselineTokenId": actual_baseline_winner,
                        "actualDependency": actual_dependency,
                        "predictedCounterfactualTokenId": int(
                            counterfactual_prediction["tokenId"]
                        ),
                        "predictedBaselineTokenId": int(
                            baseline_prediction["tokenId"]
                        ),
                        "reconstructedDependency": reconstructed_dependency,
                        "matches": actual_dependency == reconstructed_dependency,
                        "counterfactualPredictionMethod": (
                            counterfactual_prediction.get("method")
                        ),
                        "counterfactualPredictionConfidence": (
                            counterfactual_prediction.get("confidence")
                        ),
                        "counterfactualPredictionEvidenceQueryIds": list(
                            counterfactual_prediction.get(
                                "evidenceQueryIds", []
                            )
                        ),
                        "baselinePredictionMethod": baseline_prediction.get(
                            "method"
                        ),
                        "baselinePredictionConfidence": baseline_prediction.get(
                            "confidence"
                        ),
                        "baselinePredictionEvidenceQueryIds": list(
                            baseline_prediction.get("evidenceQueryIds", [])
                        ),
                        "baselineContextTokenIds": baseline_context,
                        "counterfactualContextTokenIds": counterfactual_context,
                    })
            interaction_reconstruction_total += 1
            interaction_reconstruction_matches += int(
                actual_winners[0]
                == int(predictions[0]["conditional"]["tokenId"])
            )
            if case.get("singleDeletionTokenIds"):
                base_observation = next(
                    item for item in reconstruction["observations"]
                    if item["contextTokenIds"] == case["baselineTokenIds"]
                )
                target_id = baseline_token_id
                competitor_id = int(base_observation["topCandidates"][1]["tokenId"])
                baseline_logits = _logit_values(
                    query(case["baselineTokenIds"])
                )
                evidence_contexts.append(list(case["baselineTokenIds"]))
                validation_query_count += 1
                base_margin = (
                    baseline_logits[target_id] - baseline_logits[competitor_id]
                )
                baseline_prediction = predict_next_token(
                    reconstruction,
                    case["baselineTokenIds"],
                    prompt_token_ids=case["baselineTokenIds"],
                    generated_prefix_token_ids=[],
                )
                predicted_base_margin = _predicted_logit_margin(
                    baseline_prediction,
                    target_id,
                    competitor_id,
                )
                actual_single_margins = []
                predicted_single_margins = []
                for single_context in case["singleDeletionTokenIds"]:
                    evidence_contexts.append(list(single_context))
                    single_logits = _logit_values(query(single_context))
                    validation_query_count += 1
                    single_actual_winner = max(
                        range(len(single_logits)), key=single_logits.__getitem__
                    )
                    actual_single_margins.append(
                        single_logits[target_id] - single_logits[competitor_id]
                    )
                    single_prediction = predict_next_token(
                        reconstruction,
                        single_context,
                        prompt_token_ids=single_context,
                        generated_prefix_token_ids=[],
                    )
                    predicted_single_margins.append(
                        _predicted_logit_margin(
                            single_prediction,
                            target_id,
                            competitor_id,
                        )
                    )
                    if case["role"] == "heldout_counterfactual":
                        single_predicted_winner = int(single_prediction["tokenId"])
                        actual_dependency = (
                            single_actual_winner != baseline_token_id
                        )
                        reconstructed_dependency = (
                            single_predicted_winner != int(surrogate_baseline)
                        )
                        dependency_total += 1
                        dependency_matches += int(
                            actual_dependency == reconstructed_dependency
                        )
                        dependency_assessments.append({
                            "singleDeletionTokenIds": list(single_context),
                            "actualDependency": actual_dependency,
                            "reconstructedDependency": reconstructed_dependency,
                            "matches": actual_dependency == reconstructed_dependency,
                            "actualWinnerTokenId": single_actual_winner,
                            "reconstructedWinnerTokenId": single_predicted_winner,
                            "reconstructionMethod": single_prediction.get("method"),
                            "reconstructionConfidence": single_prediction.get(
                                "confidence"
                            ),
                            "reconstructionEvidenceQueryIds": list(
                                single_prediction.get("evidenceQueryIds", [])
                            ),
                            "reconstructionInterventionTransferEstimate": (
                                single_prediction.get(
                                    "interventionTransferEstimate"
                                )
                            ),
                            "baselineReconstructionMethod": (
                                surrogate_baseline_prediction.get("method")
                            ),
                            "baselineReconstructionEvidenceQueryIds": list(
                                surrogate_baseline_prediction.get(
                                    "evidenceQueryIds", []
                                )
                            ),
                            "contextTokenIds": list(single_context),
                        })
                evidence_contexts.append(list(prompt_ids))
                joint_logits = _logit_values(query(prompt_ids))
                validation_query_count += 1
                actual_joint_margin = joint_logits[target_id] - joint_logits[competitor_id]
                joint_prediction = predict_next_token(
                    reconstruction,
                    prompt_ids,
                    prompt_token_ids=prompt_ids,
                    generated_prefix_token_ids=[],
                )
                predicted_joint_margin = _predicted_logit_margin(
                    joint_prediction,
                    target_id,
                    competitor_id,
                )
                if case["role"] == "heldout_interaction":
                    actual_residual = (
                        actual_joint_margin
                        - actual_single_margins[0]
                        - actual_single_margins[1]
                        + base_margin
                    )
                    predicted_margins = [
                        predicted_base_margin,
                        *predicted_single_margins,
                        predicted_joint_margin,
                    ]
                    score_available = all(
                        margin is not None for margin in predicted_margins
                    )
                    predicted_residual = None
                    sign_matches = None
                    absolute_error = None
                    if score_available:
                        predicted_residual = (
                            float(predicted_joint_margin)
                            - float(predicted_single_margins[0])
                            - float(predicted_single_margins[1])
                            + float(predicted_base_margin)
                        )
                        actual_sign = (actual_residual > 0) - (
                            actual_residual < 0
                        )
                        predicted_sign = (predicted_residual > 0) - (
                            predicted_residual < 0
                        )
                        sign_matches = actual_sign == predicted_sign
                        absolute_error = abs(
                            actual_residual - predicted_residual
                        )
                        interaction_sign_matches += int(sign_matches)
                        interaction_absolute_errors.append(absolute_error)
                        interaction_count += 1
                    interaction_assessment = {
                        "targetTokenId": target_id,
                        "competitorTokenId": competitor_id,
                        "actualBaselineMargin": base_margin,
                        "predictedBaselineMargin": predicted_base_margin,
                        "actualSingleDeletionMargins": actual_single_margins,
                        "predictedSingleDeletionMargins": predicted_single_margins,
                        "actualJointMargin": actual_joint_margin,
                        "predictedJointMargin": predicted_joint_margin,
                        "actualResidual": actual_residual,
                        "predictedResidual": predicted_residual,
                        "residualScoreAvailable": score_available,
                        "signMatches": sign_matches,
                        "absoluteError": absolute_error,
                        "evidenceContexts": [
                            list(case["baselineTokenIds"]),
                            *[
                                list(value)
                                for value in case["singleDeletionTokenIds"]
                            ],
                            list(prompt_ids),
                        ],
                    }
        oracle_records_by_context = _query_id_map(oracle_records)
        oracle_evidence = []
        seen_query_ids = set()
        for evidence_context in evidence_contexts:
            query_id = oracle_records_by_context.get(tuple(evidence_context))
            if query_id and query_id not in seen_query_ids:
                oracle_evidence.append({
                    "queryId": query_id,
                    "contextTokenIds": evidence_context,
                })
                seen_query_ids.add(query_id)
        for assessment in dependency_assessments:
            assessment["evidenceQueryIds"] = [
                query_id
                for evidence in oracle_evidence
                if evidence["contextTokenIds"] == assessment["contextTokenIds"]
                for query_id in [evidence["queryId"]]
            ]
            del assessment["contextTokenIds"]
        for assessment in autoregressive_dependency_assessments:
            assessment["counterfactualOracleQueryIds"] = [
                query_id
                for evidence in oracle_evidence
                if evidence["contextTokenIds"]
                == assessment["counterfactualContextTokenIds"]
                for query_id in [evidence["queryId"]]
            ]
            assessment["baselineOracleQueryIds"] = [
                query_id
                for evidence in oracle_evidence
                if evidence["contextTokenIds"]
                == assessment["baselineContextTokenIds"]
                for query_id in [evidence["queryId"]]
            ]
            del assessment["baselineContextTokenIds"]
            del assessment["counterfactualContextTokenIds"]
        if interaction_assessment is not None:
            interaction_assessment["evidenceQueryIds"] = [
                evidence["queryId"]
                for evidence in oracle_evidence
                if evidence["contextTokenIds"]
                in interaction_assessment["evidenceContexts"]
            ]
            del interaction_assessment["evidenceContexts"]
        beam_prediction = generate_reconstruction(
            reconstruction,
            prompt_ids,
            tokenizer,
            max_new_tokens=max_new_tokens,
            prompt_text=case.get("prompt"),
            beam_width=2,
        )
        beam_candidate_match = any(
            item["tokenIds"] == actual_generated
            for item in beam_prediction["candidateResponses"]
        )
        beam_candidate_response_matches += int(beam_candidate_match)
        if actual_terminated_by_eos:
            beam_complete_response_matches += int(beam_candidate_match)
        case_reports.append({
            "id": case["id"],
            "role": case["role"],
            "inputTokenIds": prompt_ids,
            "removedTokenIndices": case.get("removedTokenIndices"),
            "editableInputTokenIndices": case.get("editableInputTokenIndices"),
            "editableInputScope": case.get("editableInputScope"),
            "oracleEvidence": oracle_evidence,
            "actualTokenIds": actual_generated,
            "predictedTokenIds": surrogate_generated,
            "actualTerminatedByEos": actual_terminated_by_eos,
            "predictedTerminatedByEos": predicted_terminated_by_eos,
            "actualTerminationReason": (
                "eos" if actual_terminated_by_eos else "max_new_tokens"
            ),
            "predictedTerminationReason": (
                rollout_generation["terminationReason"]
            ),
            "rolloutOutputComplete": rollout_generation["outputComplete"],
            "actualText": tokenizer.decode(
                actual_generated, skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ),
            "predictedText": tokenizer.decode(
                surrogate_generated, skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ),
            "dependencyAssessments": dependency_assessments,
            "autoregressiveDependencyAssessments": (
                autoregressive_dependency_assessments
            ),
            "interactionAssessment": interaction_assessment,
            "exactResponseMatch": actual_generated == surrogate_generated,
            "fullResponseExactMatch": full_response_match,
            "beamCandidateResponseComplete": (
                actual_terminated_by_eos and beam_candidate_match
            ),
            "beamCandidateResponseMatch": beam_candidate_match,
            "beamCandidateResponses": [
                {
                    "tokenIds": item["tokenIds"],
                    "text": item["text"],
                    "terminatedByEos": item["terminatedByEos"],
                    "terminationReason": item["terminationReason"],
                    "cumulativeLogCandidateScore": item[
                        "cumulativeLogCandidateScore"
                    ],
                }
                for item in beam_prediction["candidateResponses"]
            ],
            "tokenAccuracy": (
                sum(a == b for a, b in zip(actual_generated, surrogate_generated))
                / max(len(actual_generated), len(surrogate_generated), 1)
            ),
            "sequenceEditDistance": sequence_edit_distance,
            "normalizedSequenceEditSimilarity": sequence_edit_similarities[-1],
            "topFivePredictionAgreement": (
                case_top_five_agreement_sum
                / case_top_five_evaluation_total
                if case_top_five_evaluation_total else None
            ),
            "topFiveOracleTopTokenCoverage": (
                case_top_five_oracle_top_token_covered
                / case_top_five_evaluation_total
                if case_top_five_evaluation_total else None
            ),
            "teacherForcedTopFiveDistributionFidelity": (
                case_distribution_fidelity_sum / case_distribution_total
                if case_distribution_total else None
            ),
            "teacherForcedTopFiveDistributionEvaluations": (
                case_distribution_total
            ),
            "eosPredictionAccuracy": (
                case_eos_prediction_matches / case_eos_prediction_total
                if case_eos_prediction_total else None
            ),
            "eosPredictionEvaluations": case_eos_prediction_total,
            "predictionEvidence": [
                {
                    "conditional": {
                        "method": item["conditional"]["method"],
                        "confidence": item["conditional"]["confidence"],
                        "queryIds": item["conditional"]["evidenceQueryIds"],
                        "formulationEvidence": item["conditional"].get(
                            "formulationEvidence", []
                        ),
                        "interactionEstimate": item["conditional"].get(
                            "interactionEstimate"
                        ),
                        "oracleTopFiveTokenIds": item[
                            "oracleTopFiveTokenIds"
                        ],
                        "predictedTopFiveTokenIds": item[
                            "predictedTopFiveTokenIds"
                        ],
                        "topFiveAgreement": item["topFiveAgreement"],
                        "oracleTopTokenCoveredByTopFive": item[
                            "oracleTopTokenCoveredByTopFive"
                        ],
                        "teacherForcedTopFiveJensenShannonBits": item[
                            "teacherForcedTopFiveJensenShannonBits"
                        ],
                    },
                    "rollout": (
                        {
                            "method": item["rollout"]["method"],
                            "confidence": item["rollout"]["confidence"],
                            "queryIds": item["rollout"]["evidenceQueryIds"],
                            "formulationEvidence": item["rollout"].get(
                                "formulationEvidence", []
                            ),
                        }
                        if item["rollout"] is not None
                        else None
                    ),
                }
                for item in predictions
            ],
        })
    dependency_prediction_metrics = _dependency_prediction_metrics(case_reports)
    dependency_prediction_method_counts = dependency_prediction_metrics[
        "methodCounts"
    ]
    return {
        "tokenPredictionFidelity": token_matches / token_total if token_total else 0.0,
        "teacherForcedNextTokenAccuracy": (
            token_matches / token_total if token_total else 0.0
        ),
        "teacherForcedNextTokenEvaluations": token_total,
        "topFivePredictionAgreement": (
            top_five_agreement_sum / top_five_evaluation_total
            if top_five_evaluation_total else None
        ),
        "topFivePredictionEvaluations": top_five_evaluation_total,
        "topFiveOracleTopTokenCoverage": (
            top_five_oracle_top_token_covered / top_five_evaluation_total
            if top_five_evaluation_total else None
        ),
        "teacherForcedTopFiveDistributionFidelity": (
            teacher_forced_distribution_fidelity_sum
            / teacher_forced_distribution_total
            if teacher_forced_distribution_total else None
        ),
        "teacherForcedTopFiveDistributionEvaluations": (
            teacher_forced_distribution_total
        ),
        "rolloutTokenFidelity": (
            rollout_token_matches / token_total if token_total else 0.0
        ),
        "freeRunningRolloutEvaluations": token_total,
        "freeRunningRolloutFidelity": (
            rollout_token_matches / token_total if token_total else 0.0
        ),
        "meanSequenceEditDistance": (
            sum(sequence_edit_distances) / len(sequence_edit_distances)
            if sequence_edit_distances else None
        ),
        "meanNormalizedSequenceEditSimilarity": (
            sum(sequence_edit_similarities) / len(sequence_edit_similarities)
            if sequence_edit_similarities else None
        ),
        "sequenceEditEvaluations": len(sequence_edit_distances),
        "eosPredictionAccuracy": (
            eos_prediction_matches / eos_prediction_total
            if eos_prediction_total else None
        ),
        "eosPredictionEvaluations": eos_prediction_total,
        "fullResponseExactMatchRate": (
            response_exact / complete_response_total
            if complete_response_total else None
        ),
        "completeOracleResponseCount": complete_response_total,
        "beamCandidateResponseCoverage": (
            beam_candidate_response_matches / len(case_reports)
            if case_reports else 0.0
        ),
        "beamCandidateCompleteResponseCoverage": (
            beam_complete_response_matches / complete_response_total
            if complete_response_total else None
        ),
        "counterfactualEffectAccuracy": (
            counterfactual_effect_matches / counterfactual_effect_total
            if counterfactual_effect_total else None
        ),
        "counterfactualEffectEvaluations": counterfactual_effect_total,
        "dependencyReconstructionAccuracy": (
            dependency_matches / dependency_total if dependency_total else None
        ),
        "dependencyEvaluations": dependency_total,
        "autoregressiveCounterfactualDependencyAccuracy": (
            autoregressive_dependency_matches / autoregressive_dependency_total
            if autoregressive_dependency_total else None
        ),
        "autoregressiveCounterfactualDependencyEvaluations": (
            autoregressive_dependency_total
        ),
        "dependencyPredictionMethodCounts": dependency_prediction_method_counts,
        "dependencyInterventionModelCoverage": dependency_prediction_metrics[
            "interventionModelCoverage"
        ],
        "metricDefinitions": {
            "teacherForcedNextTokenAccuracy": (
                "Fraction of oracle next-token decisions matched by the "
                "reconstruction when both are given the same actual prefix."
            ),
            "topFivePredictionAgreement": (
                "Mean fraction of the oracle's top-five tokens present in the "
                "reconstruction's top-five candidate scores at teacher-forced contexts."
            ),
            "topFiveOracleTopTokenCoverage": (
                "Fraction of teacher-forced contexts where the oracle's top token "
                "appears in the reconstruction's top-five candidates."
            ),
            "teacherForcedTopFiveDistributionFidelity": (
                "One minus Jensen-Shannon divergence in bits, averaged over the "
                "union of each side's top-five tokens; this is a truncated "
                "distribution comparison, not full-vocabulary probability fidelity."
            ),
            "freeRunningRolloutFidelity": (
                "Fraction of oracle token positions matched by a reconstruction "
                "generated from its own preceding tokens; EOS stops that rollout, "
                "and missing later positions count as mismatches."
            ),
            "meanSequenceEditDistance": (
                "Mean token-level Levenshtein distance between complete bounded "
                "oracle and free-running reconstruction sequences, including EOS "
                "when emitted."
            ),
            "eosPredictionAccuracy": (
                "Binary EOS-versus-non-EOS decision accuracy at teacher-forced "
                "oracle token positions."
            ),
            "beamCandidateResponseCoverage": (
                "Fraction of validation responses present in the two highest "
                "scoring executable reconstruction beam candidates. This is "
                "set-valued coverage, not top-response fidelity."
            ),
            "counterfactualEffectAccuracy": (
                "Fraction of paired input-deletion cases where the predicted "
                "top-token change/no-change matches the oracle."
            ),
            "autoregressiveCounterfactualDependencyAccuracy": (
                "Fraction of held-out counterfactual token positions where the "
                "reconstructed baseline-vs-counterfactual dependency agrees "
                "with oracle decisions at the same actual generated prefix."
            ),
            "dependencyReconstructionAccuracy": (
                "Fraction of individually deleted, previously unqueried input "
                "units where the predicted top-token dependency matches the oracle."
            ),
            "dependencyInterventionModelCoverage": (
                "Fraction of held-out dependency predictions produced by a measured "
                "or inferred input-intervention effect model rather than exact-context "
                "or generic observation/formulation voting."
            ),
            "interactionReconstructionAccuracy": (
                "Fraction of held-out paired-deletion contexts where the "
                "reconstructed next token matches the oracle."
            ),
            "interactionResidualSignAccuracy": (
                "Fraction of held-out pairwise logit-margin interaction residuals "
                "whose positive/zero/negative sign matches the oracle, among cases "
                "where the reconstruction has scores for both compared tokens at "
                "the baseline, both singleton contexts, and joint context."
            ),
            "interactionResidualEvaluations": (
                "Number of held-out interaction cases with complete reconstructed "
                "scores for both compared tokens across all four contexts."
            ),
        },
        "interactionReconstructionAccuracy": (
            interaction_reconstruction_matches / interaction_reconstruction_total
            if interaction_reconstruction_total else None
        ),
        "interactionEvaluations": interaction_reconstruction_total,
        "interactionResidualSignAccuracy": (
            interaction_sign_matches / interaction_count if interaction_count else None
        ),
        "interactionResidualMeanAbsoluteError": (
            sum(interaction_absolute_errors) / interaction_count
            if interaction_count else None
        ),
        "interactionResidualEvaluations": interaction_count,
        "oracleQueryCount": validation_query_count,
        "generalizationAcrossPromptFormulations": {
            "tested": sum(case["role"] == "heldout_formulation" for case in case_reports),
            "completeOracleResponses": sum(
                case["role"] == "heldout_formulation"
                and case["actualTerminatedByEos"]
                for case in case_reports
            ),
            "exactResponseMatches": sum(
                case["role"] == "heldout_formulation"
                and case["fullResponseExactMatch"] is True
                for case in case_reports
            ),
        },
        "explanationEvidenceCoverage": None,
        "cases": case_reports,
        "limitations": [
            "Metrics are empirical for the listed held-out cases only.",
            "Dependency and interaction generalization require more independent intervention cases.",
        ],
    }


def _heldout_cases(
    prompt: str,
    prompt_ids: Sequence[int],
    tokenizer: Any,
    training_contexts: Iterable[Tuple[int, ...]],
    decision_models: Sequence[Dict[str, Any]],
    heldout_formulation: str,
    prompt_format: str,
    editable_input_token_indices: Sequence[int] | None = None,
    preferred_interaction_token_indices: Sequence[int] = (),
) -> List[Dict[str, Any]]:
    heldout_ids, _, heldout_editable_indices = encode_prompt_with_scope(
        tokenizer,
        heldout_formulation,
        prompt_format,
    )
    heldout_units, _ = _scope_context_units(
        *_editable_context_units(heldout_ids, tokenizer),
        heldout_editable_indices,
    )
    heldout_editable_indices = sorted({
        position for unit in heldout_units for position in unit
    })
    cases = [{
        "id": "heldout-formulation-1",
        "role": "heldout_formulation",
        "prompt": heldout_formulation,
        "inputTokenIds": heldout_ids,
        "editableInputTokenIndices": heldout_editable_indices,
        "editableInputScope": (
            "user_message_content"
            if prompt_format == "chat-template"
            else "full_prompt"
        ),
    }]
    units, _ = _scope_context_units(
        *_editable_context_units(prompt_ids, tokenizer),
        editable_input_token_indices,
    )
    editable_positions = (
        set(range(len(prompt_ids)))
        if editable_input_token_indices is None
        else {int(position) for position in editable_input_token_indices}
    )
    training = set(training_contexts)
    baseline_model = next(
        (item for item in decision_models if item["stepIndex"] == 0),
        None,
    )
    measured_deletions = {
        tuple(sorted(int(value) for value in effect["deletedTokenIndices"]))
        for effect in (
            baseline_model["inputDeletionEffects"] if baseline_model else []
        )
    }
    pair_indices = list(itertools.combinations(range(len(units)), 2))
    preferred_positions = {
        int(position) for position in preferred_interaction_token_indices
    }
    if preferred_positions:
        preferred_pair = next(
            (
                pair_index
                for pair_index in pair_indices
                if {
                    position
                    for unit_index in pair_index
                    for position in units[unit_index]
                } == preferred_positions
            ),
            None,
        )
        if preferred_pair is not None:
            pair_indices.remove(preferred_pair)
            pair_indices.insert(0, preferred_pair)
    elif len(units) >= 2:
        preferred = (0, len(units) - 1)
        pair_indices.remove(preferred)
        pair_indices.insert(0, preferred)
    heldout_pairs = []
    for pair_index in pair_indices:
        pair = (units[pair_index[0]], units[pair_index[1]])
        removed = {position for unit in pair for position in unit}
        counterfactual = [
            int(token) for index, token in enumerate(prompt_ids) if index not in removed
        ]
        if not counterfactual or tuple(counterfactual) in training:
            continue
        singles = []
        for unit in pair:
            single_removed = set(unit)
            single_context = [
                int(token)
                for index, token in enumerate(prompt_ids)
                if index not in single_removed
            ]
            if single_context:
                singles.append(single_context)
        singles_are_heldout = (
            len(singles) == 2
            and singles[0] != singles[1]
            and all(tuple(single) not in training for single in singles)
        )
        singles_are_observed = (
            len(singles) == 2
            and singles[0] != singles[1]
            and all(
                tuple(sorted(int(value) for value in unit)) in measured_deletions
                and tuple(single) in training
                for unit, single in zip(pair, singles)
            )
        )
        heldout_pairs.append({
            "pairIndices": pair_index,
            "inputTokenIds": counterfactual,
            "baselineTokenIds": [int(value) for value in prompt_ids],
            "removedTokenIndices": sorted(removed),
            "editableInputTokenIndices": sorted(
                position - sum(deleted < position for deleted in removed)
                for position in editable_positions
                if position not in removed
            ),
            "singleDeletionTokenIds": singles,
            "singlesAreHeldout": singles_are_heldout,
            "singlesAreObserved": singles_are_observed,
        })

    def append_pair(candidate: Dict[str, Any], role: str) -> None:
        cases.append({
            "id": f"{role}-{len(cases)}",
            "role": role,
            "inputTokenIds": candidate["inputTokenIds"],
            "baselineTokenIds": candidate["baselineTokenIds"],
            "removedTokenIndices": candidate["removedTokenIndices"],
            "editableInputTokenIndices": candidate["editableInputTokenIndices"],
            "editableInputScope": (
                "user_message_content"
                if prompt_format == "chat-template"
                else "full_prompt"
            ),
            "singleDeletionTokenIds": candidate["singleDeletionTokenIds"],
        })

    dependency_pair = next(
        (pair for pair in heldout_pairs if pair["singlesAreHeldout"]),
        None,
    )
    if dependency_pair:
        append_pair(dependency_pair, "heldout_counterfactual")
    interaction_pair = next(
        (
            pair for pair in heldout_pairs
            if pair["singlesAreObserved"]
            and (
                dependency_pair is None
                or pair["pairIndices"] != dependency_pair["pairIndices"]
            )
        ),
        None,
    )
    if interaction_pair:
        append_pair(interaction_pair, "heldout_interaction")
    return cases


def allocate_query_budget(
    *,
    query_budget: int,
    max_new_tokens: int,
    probes_per_token: int,
    explanation_probes: int,
    interaction_probes: int,
    branch_probes: int,
    branch_width: int,
    response_probes: int,
    formulation_probes: int = 1,
) -> Dict[str, Any]:
    if query_budget < 1:
        raise ValueError("query_budget must be positive.")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive.")
    if any(value < 0 for value in (
        probes_per_token,
        explanation_probes,
        interaction_probes,
        branch_probes,
        response_probes,
        formulation_probes,
    )):
        raise ValueError("Probe budgets must be non-negative.")
    available_formulations = len(_training_formulations(""))
    if formulation_probes > available_formulations:
        raise ValueError(
            "formulation_probes cannot exceed the "
            f"{available_formulations} available training formulations."
        )
    if branch_width < 1:
        raise ValueError("branch_width must be positive.")
    requested = {
        "maxNewTokens": max_new_tokens,
        "probesPerToken": probes_per_token,
        "explanationProbes": explanation_probes,
        "interactionProbes": interaction_probes,
        "branchProbes": branch_probes,
        "branchWidth": branch_width,
        "responseProbes": response_probes,
        "formulationProbes": formulation_probes,
    }
    effective = dict(requested)

    def estimate() -> int:
        steps = effective["maxNewTokens"]
        trace_per_step = (
            1
            + effective["probesPerToken"]
            + effective["explanationProbes"]
            + effective["interactionProbes"]
        )
        branch_cost = (
            effective["branchProbes"]
            * effective["branchWidth"]
            * steps
            * max(steps - 1, 0)
            // 2
        )
        response_cost = effective["responseProbes"] * steps
        formulation_cost = effective["formulationProbes"] * steps
        validation_cost = 3 * steps + 8
        return (
            steps * trace_per_step
            + branch_cost
            + response_cost
            + formulation_cost
            + validation_cost
        )

    reduction_order = (
        "responseProbes",
        "formulationProbes",
        "branchProbes",
        "interactionProbes",
        "explanationProbes",
        "probesPerToken",
    )
    while estimate() > query_budget:
        reduced = False
        for key in reduction_order:
            if effective[key] > 0:
                effective[key] -= 1
                reduced = True
                break
        if reduced:
            continue
        if effective["maxNewTokens"] > 1:
            effective["maxNewTokens"] -= 1
            continue
        raise ValueError(
            "query budget is too small for one baseline token and independent validation."
        )
    return {
        "requested": requested,
        "effective": effective,
        "estimatedUniqueOracleQueries": estimate(),
        "queryBudget": query_budget,
    }


def _adaptive_interaction_probe_target(
    trace: Dict[str, Any],
    current_probe_count: int,
    available_unique_queries: int,
) -> int:
    if current_probe_count < 1 or available_unique_queries < 1:
        return current_probe_count
    per_step_maxima = [
        max(
            0,
            len(step["contextTokenInterventions"])
            * (len(step["contextTokenInterventions"]) - 1)
            // 2
            - 1,
        )
        for step in trace["steps"]
        if (
            len(step["contextTokenInterventions"])
            * (len(step["contextTokenInterventions"]) - 1)
            // 2
            - 1
        ) > 0
    ]
    maximum = max(
        current_probe_count,
        min(per_step_maxima, default=current_probe_count),
    )
    target = current_probe_count
    while target < maximum:
        next_target = target + 1
        additional_queries = sum(
            max(
                0,
                min(
                    next_target,
                    max(
                        0,
                        len(step["contextTokenInterventions"])
                        * (len(step["contextTokenInterventions"]) - 1)
                        // 2
                        - 1,
                    ),
                )
                - len(step["pairwiseInterventions"]),
            )
            for step in trace["steps"]
        )
        if additional_queries > available_unique_queries:
            break
        target = next_target
    return target


def _adaptive_context_probe_target(
    current_probe_count: int,
    maximum_probe_count: int,
    step_count: int,
    available_unique_queries: int,
) -> int:
    if step_count < 1 or available_unique_queries < 2:
        return current_probe_count
    target = current_probe_count
    remaining_probe_budget = available_unique_queries // 2
    while (
        target < maximum_probe_count
        and remaining_probe_budget >= step_count
    ):
        target += 1
        remaining_probe_budget -= step_count
    return target


def _select_heldout_input_token_indices(
    prompt_token_ids: Sequence[int],
    input_units: Sequence[Sequence[int]],
    *,
    minimum_trainable_units: int = 3,
    excluded_input_token_indices: Sequence[int] = (),
) -> List[int]:
    if len(input_units) <= minimum_trainable_units + 1:
        return []
    excluded_positions = {
        int(position) for position in excluded_input_token_indices
    }
    eligible_unit_indices = [
        index
        for index, unit in enumerate(input_units)
        if not excluded_positions.intersection(int(position) for position in unit)
    ]
    if len(eligible_unit_indices) <= minimum_trainable_units + 1:
        return []
    singleton_contexts = []
    for unit in input_units:
        removed = set(int(position) for position in unit)
        singleton_contexts.append(tuple(
            int(token_id)
            for position, token_id in enumerate(prompt_token_ids)
            if position not in removed
        ))
    pair_contexts = {}
    for pair in itertools.combinations(range(len(input_units)), 2):
        removed = {
            int(position)
            for unit_index in pair
            for position in input_units[unit_index]
        }
        pair_contexts[pair] = tuple(
            int(token_id)
            for position, token_id in enumerate(prompt_token_ids)
            if position not in removed
        )
    pair_context_counts: Dict[Tuple[int, ...], int] = {}
    for context in pair_contexts.values():
        pair_context_counts[context] = pair_context_counts.get(context, 0) + 1
    center = (len(input_units) - 1) / 2
    candidate_pairs = sorted(
        itertools.combinations(eligible_unit_indices, 2),
        key=lambda pair: (
            abs((pair[0] + pair[1]) / 2 - center),
            abs(pair[1] - pair[0]),
            -pair[0],
        ),
    )
    for pair in candidate_pairs:
        pair_positions = {
            int(position)
            for unit_index in pair
            for position in input_units[unit_index]
        }
        if pair_positions.intersection(excluded_positions):
            continue
        single_contexts = [singleton_contexts[index] for index in pair]
        joint_context = pair_contexts[pair]
        if (
            not all(single_contexts)
            or single_contexts[0] == single_contexts[1]
            or not joint_context
            or joint_context in singleton_contexts
            or pair_context_counts[joint_context] != 1
        ):
            continue
        return sorted({
            int(position)
            for unit_index in pair
            for position in input_units[unit_index]
        })
    return []


def run_decompilation(
    *,
    oracle: Any,
    tokenizer: Any,
    prompt: str,
    model_identity: Dict[str, Any],
    session_dir: str | Path,
    query_budget: int,
    max_new_tokens: int,
    probes_per_token: int,
    explanation_probes: int,
    interaction_probes: int,
    branch_probes: int,
    branch_width: int,
    top_token_count: int,
    response_probes: int,
    resume: bool,
    formulation_probes: int = 1,
    prompt_format: str = "auto",
) -> Dict[str, Any]:
    prompt_ids, effective_prompt_format, editable_input_token_indices = encode_prompt_with_scope(
        tokenizer,
        prompt,
        prompt_format,
    )
    allocation = allocate_query_budget(
        query_budget=query_budget,
        max_new_tokens=max_new_tokens,
        probes_per_token=probes_per_token,
        explanation_probes=explanation_probes,
        interaction_probes=interaction_probes,
        branch_probes=branch_probes,
        branch_width=branch_width,
        response_probes=response_probes,
        formulation_probes=formulation_probes,
    )
    requested_parameters = allocation["requested"]
    effective = allocation["effective"]
    max_new_tokens = effective["maxNewTokens"]
    probes_per_token = effective["probesPerToken"]
    initial_effective_probe_count = probes_per_token
    explanation_probes = effective["explanationProbes"]
    interaction_probes = effective["interactionProbes"]
    branch_probes = effective["branchProbes"]
    branch_width = effective["branchWidth"]
    response_probes = effective["responseProbes"]
    formulation_probes = effective["formulationProbes"]
    parameters = {
        **requested_parameters,
        "topTokenCount": top_token_count,
    }
    if effective_prompt_format != "raw":
        parameters["promptFormat"] = effective_prompt_format
    input_units, _ = _scope_context_units(
        *_editable_context_units(prompt_ids, tokenizer),
        editable_input_token_indices,
    )
    central_input_pair = _select_heldout_input_token_indices(
        prompt_ids,
        input_units,
    )
    disjoint_input_pair = _select_heldout_input_token_indices(
        prompt_ids,
        input_units,
        excluded_input_token_indices=central_input_pair,
    )
    if disjoint_input_pair:
        heldout_interaction_token_indices = disjoint_input_pair
    else:
        heldout_interaction_token_indices = []
    heldout_input_token_indices = central_input_pair
    reserved_input_positions = set(heldout_input_token_indices)
    probeable_input_units = [
        unit for unit in input_units
        if not reserved_input_positions.intersection(unit)
    ]
    worker_queries_before = int(getattr(oracle, "query_count", 0))
    records_before = 0
    session = PersistentOracleSession(
        oracle,
        session_dir,
        identity=model_identity,
        prompt=prompt,
        parameters=parameters,
        query_budget=query_budget,
        resume=resume,
    )
    records_before = len(session.records)
    trace = trace_greedy_generation(
        oracle_query_token_ids=session.query_token_ids,
        tokenizer=tokenizer,
        input_token_ids=prompt_ids,
        max_new_tokens=max_new_tokens,
        probes_per_token=probes_per_token,
        top_token_count=top_token_count,
        response_probe_count=response_probes,
        explanation_probes_per_token=explanation_probes,
        interaction_probe_count=interaction_probes,
        branch_probe_count=branch_probes,
        branch_width=branch_width,
        editable_input_token_indices=editable_input_token_indices,
        editable_input_scope=(
            "user_message_content"
            if effective_prompt_format == "chat-template"
            else "full_prompt"
        ),
        reserved_input_token_indices=heldout_input_token_indices,
        priority_input_token_indices=heldout_interaction_token_indices,
        reserved_pairwise_input_token_indices=heldout_interaction_token_indices,
    )
    initial_trace_query_count = int(trace["queryCount"])
    initial_pair_counts = [
        len(step["pairwiseInterventions"])
        for step in trace["steps"]
    ]
    adaptive_interaction_reserve = (
        3 * len(trace["steps"]) + 8
        + formulation_probes * len(trace["steps"])
    )
    available_interaction_queries = max(
        0,
        query_budget
        - len(session.records)
        - adaptive_interaction_reserve,
    )
    generated_unit_count = max(
        (
            len(_editable_context_units(
                trace["generatedTokenIds"][:int(step["stepIndex"])],
                tokenizer,
            )[0])
            for step in trace["steps"]
        ),
        default=0,
    )
    maximum_probe_ceiling = max(
        probes_per_token,
        len(probeable_input_units) + generated_unit_count,
    )
    maximum_probe_count = maximum_probe_ceiling
    heldout_input_unit_count = len(input_units) - len(probeable_input_units)
    adaptive_context_probes = _adaptive_context_probe_target(
        probes_per_token,
        maximum_probe_count,
        len(trace["steps"]),
        available_interaction_queries,
    )
    adaptive_context_unique_queries = 0
    adaptive_context_logical_queries = 0
    if adaptive_context_probes > probes_per_token:
        records_before_context_expansion = len(session.records)
        expanded_trace = trace_greedy_generation(
            oracle_query_token_ids=session.query_token_ids,
            tokenizer=tokenizer,
            input_token_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            probes_per_token=adaptive_context_probes,
            top_token_count=top_token_count,
            response_probe_count=response_probes,
            explanation_probes_per_token=explanation_probes,
            interaction_probe_count=0,
            branch_probe_count=branch_probes,
            branch_width=branch_width,
            editable_input_token_indices=editable_input_token_indices,
            editable_input_scope=(
                "user_message_content"
                if effective_prompt_format == "chat-template"
                else "full_prompt"
            ),
            reserved_input_token_indices=heldout_input_token_indices,
            priority_input_token_indices=heldout_interaction_token_indices,
            reserved_pairwise_input_token_indices=heldout_interaction_token_indices,
        )
        adaptive_context_logical_queries = int(expanded_trace["queryCount"])
        adaptive_context_unique_queries = (
            len(session.records) - records_before_context_expansion
        )
        trace = expanded_trace
        probes_per_token = adaptive_context_probes
        effective["probesPerToken"] = adaptive_context_probes
        available_interaction_queries = max(
            0,
            query_budget
            - len(session.records)
            - adaptive_interaction_reserve,
        )
    allocation["adaptiveContextProbing"] = {
        "requestedPerToken": int(requested_parameters["probesPerToken"]),
        "initialEffectivePerToken": initial_effective_probe_count,
        "effectivePerToken": adaptive_context_probes,
        "maximumPerToken": maximum_probe_count,
        "heldOutInputUnitsReserved": heldout_input_unit_count,
        "allocatedQueryUpperBound": (
            max(0, adaptive_context_probes - initial_effective_probe_count)
            * len(trace["steps"])
        ),
        "additionalLogicalQueries": adaptive_context_logical_queries,
        "additionalUniqueOracleQueries": adaptive_context_unique_queries,
    }
    adaptive_interaction_probes = _adaptive_interaction_probe_target(
        trace,
        interaction_probes,
        available_interaction_queries,
    )
    effective["interactionProbes"] = adaptive_interaction_probes
    adaptive_interaction_queries = 0
    adaptive_interaction_unique_queries = 0
    if (
        adaptive_interaction_probes > interaction_probes
        or adaptive_context_probes > initial_effective_probe_count
    ):
        records_before_adaptive = len(session.records)
        expanded_trace = trace_greedy_generation(
            oracle_query_token_ids=session.query_token_ids,
            tokenizer=tokenizer,
            input_token_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            probes_per_token=probes_per_token,
            top_token_count=top_token_count,
            response_probe_count=response_probes,
            explanation_probes_per_token=explanation_probes,
            interaction_probe_count=adaptive_interaction_probes,
            branch_probe_count=branch_probes,
            branch_width=branch_width,
            editable_input_token_indices=editable_input_token_indices,
            editable_input_scope=(
                "user_message_content"
                if effective_prompt_format == "chat-template"
                else "full_prompt"
            ),
            reserved_input_token_indices=heldout_input_token_indices,
            priority_input_token_indices=heldout_interaction_token_indices,
            reserved_pairwise_input_token_indices=heldout_interaction_token_indices,
        )
        adaptive_interaction_queries = int(expanded_trace["interactionQueryCount"])
        expanded_trace["queryCount"] += (
            initial_trace_query_count + adaptive_context_logical_queries
        )
        trace = expanded_trace
        adaptive_interaction_unique_queries = (
            len(session.records) - records_before_adaptive
        )
        allocation["adaptiveInteractionProbing"] = {
            "requestedPerToken": interaction_probes,
            "effectivePerToken": adaptive_interaction_probes,
            "availableQueriesBeforePairExpansion": available_interaction_queries,
            "reservedForValidationAndFormulations": adaptive_interaction_reserve,
            "additionalPairInterventions": sum(
                len(step["pairwiseInterventions"]) - previous
                for step, previous in zip(trace["steps"], initial_pair_counts)
            ),
            "additionalLogicalQueries": adaptive_interaction_queries,
            "additionalUniqueOracleQueries": adaptive_interaction_unique_queries,
        }
    observed_formulation_paths = []
    formulation_logical_calls = 0
    for formulation in _training_formulations(prompt)[:formulation_probes]:
        variant_ids, _ = encode_prompt(
            tokenizer,
            formulation["prompt"],
            effective_prompt_format,
        )
        calls_before = session.logical_calls
        generated, _ = _generate_oracle(
            session.query_token_ids,
            variant_ids,
            tokenizer,
            max_new_tokens,
        )
        formulation_logical_calls += session.logical_calls - calls_before
        eos_id = getattr(tokenizer, "eos_token_id", None)
        terminated_by_eos = bool(
            eos_id is not None
            and generated
            and generated[-1] == int(eos_id)
        )
        observed_formulation_paths.append({
            "id": formulation["id"],
            "prompt": formulation["prompt"],
            "inputTokenIds": variant_ids,
            "generatedTokenIds": generated,
            "terminatedByEos": terminated_by_eos,
            "terminationReason": (
                "eos" if terminated_by_eos else "max_new_tokens"
            ),
        })
    if session.training_query_ids:
        training_ids = set(session.training_query_ids)
        training_records = [
            row for row in session.records if row["queryId"] in training_ids
        ]
    else:
        training_records = list(session.records)
        session.set_training_boundary([row["queryId"] for row in training_records])
    training_query_ids = {row["queryId"] for row in training_records}
    formulation_effects = _prompt_formulation_effects(
        baseline_prompt_token_ids=prompt_ids,
        baseline_generated_token_ids=trace["generatedTokenIds"],
        formulation_paths=observed_formulation_paths,
        records=training_records,
        cache=session.cache,
    )
    reconstruction = build_reconstruction(
        records=training_records,
        cache=session.cache,
        tokenizer=tokenizer,
        model=model_identity,
        input_token_ids=prompt_ids,
        input_prompt_text=prompt,
        input_prompt_format=effective_prompt_format,
        top_token_count=max(16, top_token_count),
        trace=trace,
        decision_models=_measured_decision_models(
            trace=trace,
            input_token_ids=prompt_ids,
            cache=session.cache,
            records=training_records,
            candidate_count=max(16, top_token_count),
        ),
        editable_input_token_indices=trace["editableInputTokenIndices"],
        additional_observed_paths=observed_formulation_paths,
    )
    graph = build_behavior_graph(
        trace,
        input_token_ids=prompt_ids,
        tokenizer=tokenizer,
        records=training_records,
        editable_input_token_indices=trace["editableInputTokenIndices"],
    )
    explanation = explain_decompilation(graph, trace)
    explanation["formulationEffects"] = formulation_effects
    validation_cases = _heldout_cases(
        prompt,
        prompt_ids,
        tokenizer,
        (tuple(int(value) for value in row["inputTokenIds"]) for row in training_records),
        reconstruction["decisionModels"],
        heldout_formulation=_heldout_formulation(prompt),
        prompt_format=effective_prompt_format,
        editable_input_token_indices=trace["editableInputTokenIndices"],
        preferred_interaction_token_indices=heldout_interaction_token_indices,
    )
    validation = _validation_metrics(
        reconstruction,
        prompt_cases=validation_cases,
        query=session.query_token_ids,
        oracle_records=session.records,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        baseline_token_id=int(trace["steps"][0]["chosenTokenId"]),
    )
    evidence_objects = [
        *graph["tokenPredictions"],
        *graph["conditionalDependencies"],
        *graph["interactions"],
        *graph["alternativeRoutes"],
    ]
    validation["explanationEvidenceCoverage"] = (
        sum(bool(item.get("evidenceQueryIds")) for item in evidence_objects)
        / len(evidence_objects)
        if evidence_objects else 0.0
    )
    validation["explanationEvidenceEvaluationCount"] = len(evidence_objects)
    validation_query_ids = [
        record["queryId"] for record in session.records
        if record["queryId"] not in training_query_ids
    ]
    session.finish("complete")
    return {
        "schemaVersion": 1,
        "kind": "gcl-blackbox-neural-decompilation",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "model": model_identity,
        "access": {
            "solverAccess": "process-separated-token-ids-to-next-token-logits",
            "weightsUsedBySolver": False,
            "hooksUsedBySolver": False,
            "activationsUsedBySolver": False,
            "postHocWitnessOnly": True,
        },
        "input": {
            "prompt": prompt,
            "tokenIds": prompt_ids,
            "promptFormat": effective_prompt_format,
            "editableTokenIndices": trace["editableInputTokenIndices"],
            "editableTokenScope": trace["editableInputScope"],
            "fixedContextTokenIndices": sorted(
                set(range(len(prompt_ids)))
                - set(trace["editableInputTokenIndices"])
            ),
        },
        "trace": trace,
        "trainingPromptFormulations": [
            {
                "id": item["id"],
                "prompt": item["prompt"],
                "inputTokenIds": item["inputTokenIds"],
                "generatedTokenIds": item["generatedTokenIds"],
                "terminatedByEos": item["terminatedByEos"],
                "terminationReason": item["terminationReason"],
            }
            for item in observed_formulation_paths
        ],
        "promptFormulationEffects": formulation_effects,
        "queryAllocation": allocation,
        "behaviorGraph": graph,
        "explanation": explanation,
        "executableReconstruction": reconstruction,
        "validation": validation,
        "provenance": {
            "sessionDirectory": str(Path(session_dir).resolve()),
            "trainingQueryIds": sorted(training_query_ids),
            "independentValidationQueryIds": validation_query_ids,
            "uniqueOracleQueries": len(session.records),
            "newUniqueOracleQueriesThisRun": len(session.records) - records_before,
            "workerOracleQueriesThisRun": (
                int(getattr(oracle, "query_count", 0)) - worker_queries_before
            ),
            "logicalOracleCalls": session.logical_calls,
            "traceLogicalQueryCount": trace["queryCount"],
            "formulationLogicalQueryCount": formulation_logical_calls,
            "queryAccountingMatches": session.logical_calls
            == (
                trace["queryCount"]
                + formulation_logical_calls
                + validation["oracleQueryCount"]
            ),
            "queryBudget": query_budget,
            "workerAccountingMatches": (
                int(getattr(oracle, "query_count", 0)) - worker_queries_before
                == len(session.records) - records_before
            ),
            "requestedProbeSettings": requested_parameters,
            "effectiveProbeSettings": effective,
        },
        "claimBoundary": (
            "This artifact is an executable, nearest-observed-context behavioral "
            "reconstruction for the measured prompt neighborhood. It is not a "
            "recovered neural architecture or a complete account of hidden reasoning. "
            "Independent validation scores, rather than the original response alone, "
            "determine its measured fidelity."
        ),
    }


def write_explanation_markdown(path: str | Path, report: Dict[str, Any]) -> None:
    explanation = report["explanation"]
    validation = report["validation"]
    response_rate = validation["fullResponseExactMatchRate"]
    response_rate_text = (
        f"{response_rate:.4f}"
        if response_rate is not None
        else f"not measured ({validation['completeOracleResponseCount']} complete oracle responses)"
    )
    autoregressive_dependency_accuracy = validation[
        "autoregressiveCounterfactualDependencyAccuracy"
    ]
    autoregressive_dependency_evaluations = validation[
        "autoregressiveCounterfactualDependencyEvaluations"
    ]
    autoregressive_dependency_text = (
        f"{autoregressive_dependency_accuracy:.4f}"
        if autoregressive_dependency_accuracy is not None
        else f"not measured ({autoregressive_dependency_evaluations} token positions)"
    )
    interaction_sign = validation["interactionResidualSignAccuracy"]
    interaction_mae = validation["interactionResidualMeanAbsoluteError"]
    interaction_probes = report["queryAllocation"].get(
        "adaptiveInteractionProbing",
        {},
    )
    lines = [
        "# Black-box behavioral decompilation",
        "",
        f"- model: `{report['model']['id']}`",
        f"- prompt format: `{report['input']['promptFormat']}`",
        f"- unique oracle queries: `{report['provenance']['uniqueOracleQueries']}`",
        f"- token prediction fidelity: `{validation['tokenPredictionFidelity']:.4f}`",
        f"- held-out autoregressive counterfactual dependency accuracy: "
        f"`{autoregressive_dependency_text}`",
        f"- complete oracle responses: `{validation['completeOracleResponseCount']}`",
        f"- held-out full-response exact-match rate: `{response_rate_text}`",
        f"- interaction residual sign accuracy: "
        f"`{interaction_sign if interaction_sign is not None else 'not measured'}`",
        f"- interaction residual mean absolute error: "
        f"`{interaction_mae if interaction_mae is not None else 'not measured'}`",
        "",
        "## WHAT: measured information dependencies",
        "",
    ]
    for item in explanation["what"]:
        lines.append(
            f"- `{item['decision']}` retained context token IDs "
            f"`{item['retainedContextTokenIds']}`; one-minimal among tested units: "
            f"`{item['oneMinimalAmongEditableUnits']}`; evidence: `{item['queryIds']}`."
        )
    lines.extend(["", "## HOW: observed token decisions", ""])
    for item in explanation["how"]:
        lines.append(
            f"- `{item['decision']}` selected `{item['token']}` "
            f"(id `{item['tokenId']}`, probability `{item['probability']:.6g}`) "
            f"after prefix `{item['prefixTokenIds']}`; evidence: `{item['queryIds']}`."
        )
    lines.extend(["", "## WHY: experimentally measured effects", ""])
    for item in explanation["why"]:
        lines.append(
            f"- `{item['source']}` context indices `{item['contextTokenIndices']}` "
            f"`{item['effect']}` the selected token margin by "
            f"`{item['marginSupport']:.6f}` at `{item['decision']}`; "
            f"evidence: `{item['queryIds']}`."
        )
    lines.extend([
        "",
        "## INTERACTIONS",
        "",
        f"Measured pairwise interaction nodes: `{len(explanation['interactions'])}`.",
        *(
            [
                f"Adaptive probing requested `{interaction_probes['requestedPerToken']}` "
                f"and used `{interaction_probes['effectivePerToken']}` pairs per token; "
                f"`{interaction_probes['additionalUniqueOracleQueries']}` additional "
                "unique oracle queries were used."
            ]
            if interaction_probes else []
        ),
        "",
        "## ALTERNATIVES",
        "",
        f"Measured alternative routes: `{len(explanation['alternatives'])}`.",
        "",
        "## COUNTERFACTUALS",
        "",
        f"Full-response prompt deletion counterfactuals: `{len(explanation['counterfactuals'])}`.",
        "",
        "## PROMPT FORMULATION CONTRASTS",
        "",
    ])
    for effect in explanation["formulationEffects"]:
        divergence = effect["firstDivergence"]
        if divergence is None:
            lines.append(
                f"- `{effect['formulationId']}` shared "
                f"`{effect['sharedGeneratedPrefixLength']}` generated tokens "
                f"with the baseline; evidence: `{effect['evidenceQueryIds']}`."
            )
        else:
            lines.append(
                f"- `{effect['formulationId']}` first diverged at token step "
                f"`{divergence['stepIndex']}`: baseline "
                f"`{divergence['baselineTokenId']}` vs formulation "
                f"`{divergence['formulationTokenId']}`; pairwise-margin shift "
                f"`{divergence['pairwiseMarginShift']:.6f}`; evidence: "
                f"`{effect['evidenceQueryIds']}`. This is a whole-formulation "
                "contrast, not isolated token causality."
            )
    lines.extend([
        "",
        "## Independent executable-reconstruction validation",
        "",
        "| case | role | token accuracy | full response exact | completion | actual | predicted |",
        "| --- | --- | ---: | --- | --- | --- | --- |",
    ])
    for case in validation["cases"]:
        lines.append(
            f"| {case['id']} | {case['role']} | {case['tokenAccuracy']:.4f} "
            f"| {case['fullResponseExactMatch']} "
            f"| {case['actualTerminationReason']} | {case['actualText']!r} "
            f"| {case['predictedText']!r} |"
        )
        for prediction in case["predictionEvidence"]:
            estimate = prediction["conditional"].get("interactionEstimate")
            if estimate is None:
                continue
            calibration = estimate["calibration"]
            lines.append(
                f"|  | interaction estimate |  | applied: `{estimate['applied']}` "
                f"| LOO MAE `{calibration['kernelMeanAbsoluteError']}` vs zero "
                f"`{calibration['zeroResidualMeanAbsoluteError']}` "
                f"| evidence `{[item['queryId'] for item in estimate['evidence']]}` "
                "|  |"
            )
    autoregressive_assessments = [
        (case, assessment)
        for case in validation["cases"]
        for assessment in case.get("autoregressiveDependencyAssessments", [])
    ]
    if autoregressive_assessments:
        lines.extend([
            "",
            "## Autoregressive counterfactual dependency decisions",
            "",
            "Baseline and counterfactual decisions are compared at the same actual "
            "generated prefix. Oracle observations and reconstruction evidence "
            "are listed separately.",
            "",
            "| case | token index | actual prefix | oracle baseline / counterfactual | reconstructed baseline / counterfactual | oracle dependency -> reconstructed | methods | oracle evidence |",
            "| --- | ---: | --- | --- | --- | --- | --- | --- |",
        ])
        for case, assessment in autoregressive_assessments:
            lines.append(
                f"| {case['id']} | {assessment['tokenIndex']} "
                f"| `{assessment['generatedPrefixTokenIds']}` "
                f"| `{assessment['actualBaselineTokenId']}` / "
                f"`{assessment['actualCounterfactualTokenId']}` "
                f"| `{assessment['predictedBaselineTokenId']}` / "
                f"`{assessment['predictedCounterfactualTokenId']}` "
                f"| `{assessment['actualDependency']}` -> "
                f"`{assessment['reconstructedDependency']}` "
                f"| `{assessment['baselinePredictionMethod']}` / "
                f"`{assessment['counterfactualPredictionMethod']}` "
                f"| baseline `{assessment['baselineOracleQueryIds']}`, "
                f"counterfactual `{assessment['counterfactualOracleQueryIds']}` |"
            )
    lines.extend([
        "",
        "## UNCERTAINTY",
        "",
        *[f"- {item['status']}: {item['claim']}" for item in explanation["uncertainty"]],
        "",
        report["claimBoundary"],
        "",
    ])
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines), encoding="utf-8")


def load_executable_reconstruction(path: str | Path) -> Dict[str, Any]:
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        artifact.get("schemaVersion") != 1
        or artifact.get("kind") != "gcl-blackbox-executable-behavior-reconstruction"
        or not artifact.get("observations")
    ):
        raise ValueError("Unsupported or empty executable reconstruction artifact.")
    return artifact
