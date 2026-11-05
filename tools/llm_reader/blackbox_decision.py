from __future__ import annotations

import math
import re
from itertools import combinations
from typing import Any, Callable, Dict, List, Sequence, Tuple


def _single_token_id(tokenizer: Any, text: str, role: str) -> Tuple[int, str]:
    if not text:
        raise ValueError(f"{role} token must not be empty.")
    encoded = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(encoded) != 1:
        raise ValueError(
            f"{role} must be exactly one tokenizer token in the supplied form: {text!r}. "
            "Include any required leading whitespace in the token text."
        )
    token_id = int(encoded[0])
    return token_id, tokenizer.decode([token_id]).replace("\n", "\\n")


def _logit_values(logits: Any) -> List[float]:
    values = logits.detach().cpu().tolist() if hasattr(logits, "detach") else logits
    result = [float(value) for value in values]
    if not result or any(not math.isfinite(value) for value in result):
        raise ValueError("Oracle returned empty or non-finite next-token logits.")
    return result


def _softmax(logits: Sequence[float]) -> List[float]:
    maximum = max(logits)
    weights = [math.exp(value - maximum) for value in logits]
    normalizer = sum(weights)
    return [weight / normalizer for weight in weights]


def _log_softmax(logits: Sequence[float]) -> List[float]:
    maximum = max(logits)
    log_normalizer = maximum + math.log(
        sum(math.exp(value - maximum) for value in logits)
    )
    return [value - log_normalizer for value in logits]


def _js_divergence(left: Sequence[float], right: Sequence[float]) -> float:
    left_probabilities = _softmax(left)
    right_probabilities = _softmax(right)
    divergence = 0.0
    for left_probability, right_probability in zip(left_probabilities, right_probabilities):
        midpoint = (left_probability + right_probability) / 2.0
        if left_probability:
            divergence += 0.5 * left_probability * math.log(left_probability / midpoint)
        if right_probability:
            divergence += 0.5 * right_probability * math.log(right_probability / midpoint)
    return divergence / math.log(2.0)


def _rank(logits: Sequence[float], token_id: int) -> int:
    target = logits[token_id]
    return 1 + sum(value > target for value in logits)


def _raw_decode_token(tokenizer: Any, token_id: int) -> str:
    return tokenizer.decode(
        [token_id],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def _decode_token(tokenizer: Any, token_id: int) -> str:
    return _raw_decode_token(tokenizer, token_id).replace("\n", "\\n")


def _top_tokens(tokenizer: Any, logits: Sequence[float], count: int) -> List[Dict[str, Any]]:
    indices = sorted(range(len(logits)), key=logits.__getitem__, reverse=True)[:count]
    return [
        {
            "rank": rank,
            "tokenId": token_id,
            "token": _decode_token(tokenizer, token_id),
            "logit": logits[token_id],
        }
        for rank, token_id in enumerate(indices, start=1)
    ]


def _without_span(prompt: str, start: int, end: int) -> str:
    if end < len(prompt) and prompt[end].isspace():
        end += 1
    elif start > 0 and prompt[start - 1].isspace():
        start -= 1
    return prompt[:start] + prompt[end:]


def _bounded_spans(
    prompt: str,
    budget: int,
) -> Tuple[List[Tuple[int, int, str, int]], List[str], int]:
    all_spans = [(match.start(), match.end(), match.group(), index)
                 for index, match in enumerate(re.finditer(r"\S+", prompt))]
    excluded = [span[2] for span in all_spans if "<|" in span[2] or "|>" in span[2]]
    spans = [span for span in all_spans if "<|" not in span[2] and "|>" not in span[2]]
    if budget <= 0:
        return [], excluded, len(spans)
    if len(spans) <= budget:
        return spans, excluded, len(spans)
    if budget == 1:
        return [spans[len(spans) // 2]], excluded, len(spans)
    indices = sorted({round(index * (len(spans) - 1) / (budget - 1)) for index in range(budget)})
    return [spans[index] for index in indices], excluded, len(spans)


def analyze_next_token_decision(
    *,
    oracle_query: Callable[[str], Any],
    tokenizer: Any,
    prompt: str,
    target_token: str | None = None,
    target_token_id: int | None = None,
    contrast_tokens: Sequence[str] = (),
    max_probes: int = 24,
    top_token_count: int = 10,
    baseline_logits: Any = None,
    editable_prefix: str | None = None,
) -> Dict[str, Any]:
    if not prompt.strip():
        raise ValueError("Prompt must not be empty.")
    if max_probes < 0:
        raise ValueError("max_probes must be non-negative.")
    if top_token_count < 1:
        raise ValueError("top_token_count must be positive.")
    if (target_token is None) == (target_token_id is None):
        raise ValueError("Provide exactly one of target_token or target_token_id.")
    if editable_prefix is not None and not prompt.startswith(editable_prefix):
        raise ValueError("editable_prefix must be an exact prefix of prompt.")

    if target_token_id is None:
        target_id, target_text = _single_token_id(tokenizer, target_token, "Target")
        requested_target = target_token
    else:
        if target_token_id < 0:
            raise ValueError("target_token_id must be non-negative.")
        target_id = target_token_id
        requested_target = tokenizer.decode(
            [target_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ).replace("\n", "\\n")
        target_text = requested_target
    explicit_contrasts = [
        _single_token_id(tokenizer, token, "Contrast")[0]
        for token in contrast_tokens
    ]
    if target_id in explicit_contrasts:
        raise ValueError("Target token cannot also be a contrast token.")
    contrast_ids = list(dict.fromkeys(explicit_contrasts))

    baseline_was_queried = baseline_logits is None
    if baseline_was_queried:
        baseline_values = _logit_values(oracle_query(prompt))
    else:
        baseline_values = _logit_values(baseline_logits)
    if target_id >= len(baseline_values) or any(token_id >= len(baseline_values) for token_id in contrast_ids):
        raise ValueError("Tokenizer returned a token id outside the oracle logit vocabulary.")

    if not contrast_ids:
        contrast_ids = [
            item["tokenId"]
            for item in _top_tokens(tokenizer, baseline_values, max(2, top_token_count + 1))
            if item["tokenId"] != target_id
        ][:top_token_count]
    if not contrast_ids:
        raise ValueError("No competitor token is available for the target.")

    def margin(logits: Sequence[float]) -> float:
        return logits[target_id] - max(logits[token_id] for token_id in contrast_ids)

    baseline_margin = margin(baseline_values)
    baseline_probability = _softmax(baseline_values)[target_id]
    probes: List[Dict[str, Any]] = []
    skipped_probes: List[Dict[str, Any]] = []
    query_count = int(baseline_was_queried)

    probe_source = editable_prefix if editable_prefix is not None else prompt
    generated_suffix = prompt[len(probe_source):]
    selected_spans, excluded_control_tokens, eligible_span_count = _bounded_spans(probe_source, max_probes)
    for start, end, surface, span_index in selected_spans:
        counterfactual_prompt = _without_span(probe_source, start, end) + generated_suffix
        if not counterfactual_prompt.strip():
            skipped_probes.append({
                "spanIndex": span_index,
                "surface": surface,
                "reason": "deletion_would_make_prompt_empty",
            })
            continue

        logits = _logit_values(oracle_query(counterfactual_prompt))
        query_count += 1
        if len(logits) != len(baseline_values):
            raise ValueError("Oracle logit vocabulary changed between prompt interventions.")
        changed_margin = margin(logits)
        margin_support = baseline_margin - changed_margin
        probes.append({
            "spanIndex": span_index,
            "surface": surface,
            "characterSpan": [start, end],
            "prompt": counterfactual_prompt,
            "targetLogit": logits[target_id],
            "targetProbability": _softmax(logits)[target_id],
            "targetRank": _rank(logits, target_id),
            "targetVsContrastMargin": changed_margin,
            "marginSupport": margin_support,
            "distributionJensenShannonBits": _js_divergence(baseline_values, logits),
            "topTokens": _top_tokens(tokenizer, logits, top_token_count),
            "interpretation": (
                "supports_target" if margin_support > 0.0
                else "suppresses_target" if margin_support < 0.0
                else "no_margin_change"
            ),
        })

    probes.sort(key=lambda item: (-abs(float(item["marginSupport"])), item["spanIndex"]))
    supporters = [item for item in probes if item["marginSupport"] > 0.0]
    suppressors = [item for item in probes if item["marginSupport"] < 0.0]
    return {
        "schemaVersion": 1,
        "mode": "blackbox_next_token_decision",
        "method": "whitespace-delimited-span deletion counterfactuals over next-token logits",
        "prompt": prompt,
        "editablePrefix": probe_source,
        "target": {"text": requested_target, "token": target_text, "tokenId": target_id},
        "contrastTokens": [
            {
                "tokenId": token_id,
                "token": tokenizer.decode([token_id]).replace("\n", "\\n"),
            }
            for token_id in contrast_ids
        ],
        "baseline": {
            "targetLogit": baseline_values[target_id],
            "targetProbability": baseline_probability,
            "targetRank": _rank(baseline_values, target_id),
            "targetVsContrastMargin": baseline_margin,
            "topTokens": _top_tokens(tokenizer, baseline_values, top_token_count),
        },
        "counterfactuals": probes,
        "skippedCounterfactuals": skipped_probes,
        "excludedControlTokens": excluded_control_tokens,
        "queryCount": query_count,
        "summary": {
            "attemptedCounterfactuals": len(probes) + len(skipped_probes),
            "measuredCounterfactuals": len(probes),
            "eligiblePromptSpanCount": eligible_span_count,
            "selectedPromptSpanCount": len(selected_spans),
            "unprobedPromptSpanCount": eligible_span_count - len(selected_spans),
            "excludedControlTokenCount": len(excluded_control_tokens),
            "supportingPromptSpans": len(supporters),
            "suppressingPromptSpans": len(suppressors),
            "strongestSupportingSpan": supporters[0] if supporters else None,
            "strongestSuppressingSpan": suppressors[0] if suppressors else None,
            "meanDistributionJensenShannonBits": (
                sum(float(item["distributionJensenShannonBits"]) for item in probes) / len(probes)
                if probes else 0.0
            ),
        },
        "claimBoundary": (
            "These are measured prompt-to-logit counterfactual effects. They do not identify "
            "hidden neurons, internal reasoning steps, or a unique causal mechanism."
        ),
    }


def _bounded_indices(indices: Sequence[int], budget: int) -> List[int]:
    if budget <= 0 or not indices:
        return []
    if len(indices) <= budget:
        return list(indices)
    if budget == 1:
        return [indices[len(indices) // 2]]
    selected = sorted({
        round(index * (len(indices) - 1) / (budget - 1))
        for index in range(budget)
    })
    return [indices[index] for index in selected]


def _balanced_indices(
    item_count: int,
    budget: int,
    observation_counts: Sequence[int],
    cursor: int,
) -> List[int]:
    if item_count <= 0 or budget <= 0:
        return []
    start = cursor % item_count
    order = sorted(
        range(item_count),
        key=lambda index: (
            observation_counts[index],
            (index - start) % item_count,
        ),
    )
    return sorted(order[:budget])


def _partition_positions(positions: Sequence[int], part_count: int) -> List[List[int]]:
    chunk_size = max(1, math.ceil(len(positions) / part_count))
    return [
        list(positions[start:start + chunk_size])
        for start in range(0, len(positions), chunk_size)
    ]


def _editable_context_units(
    token_ids: Sequence[int],
    tokenizer: Any,
) -> Tuple[List[List[int]], set[int]]:
    special_ids = set(getattr(tokenizer, "all_special_ids", []))
    units: List[List[int]] = []
    protected_positions: set[int] = set()
    current: List[int] = []
    pending_whitespace: List[int] = []
    in_header = False

    def finish_current() -> None:
        nonlocal current
        if current:
            units.append(current)
            current = []

    for index, token_id in enumerate(token_ids):
        if token_id in special_ids:
            finish_current()
            protected_positions.update(pending_whitespace)
            pending_whitespace.clear()
            protected_positions.add(index)
            surface = _raw_decode_token(tokenizer, token_id)
            marker = (
                surface[2:-2].lower()
                if surface.startswith("<|") and surface.endswith("|>")
                else ""
            )
            if in_header:
                if "end_header" in marker or "meta_sep" in marker:
                    in_header = False
            elif "start" in marker:
                in_header = True
            continue

        surface = _raw_decode_token(tokenizer, token_id)
        if in_header:
            finish_current()
            protected_positions.update(pending_whitespace)
            pending_whitespace.clear()
            protected_positions.add(index)
            if "\n" in surface:
                in_header = False
            continue
        if not surface.strip():
            finish_current()
            pending_whitespace.append(index)
            continue

        if surface[0].isspace():
            finish_current()
            current = pending_whitespace + [index]
            pending_whitespace = []
        else:
            current.extend(pending_whitespace)
            pending_whitespace = []
            current.append(index)

        if surface[-1].isspace():
            finish_current()

    finish_current()
    protected_positions.update(pending_whitespace)
    return units, protected_positions


def _scope_context_units(
    units: Sequence[Sequence[int]],
    protected_positions: set[int],
    editable_positions: Sequence[int] | None,
) -> Tuple[List[List[int]], set[int]]:
    if editable_positions is None:
        return [list(unit) for unit in units], set(protected_positions)

    allowed = {int(position) for position in editable_positions}
    scoped_units: List[List[int]] = []
    scoped_protected = set(protected_positions)
    for unit in units:
        positions = list(unit)
        if all(position in allowed for position in positions):
            scoped_units.append(positions)
        else:
            scoped_protected.update(positions)
    return scoped_units, scoped_protected


def _minimal_sufficient_context(
    *,
    oracle_query_token_ids: Callable[[Sequence[int]], Any],
    tokenizer: Any,
    context_token_ids: Sequence[int],
    target_token_id: int,
    baseline_logits: Sequence[float],
    max_queries: int,
    editable_context_token_indices: Sequence[int] | None = None,
) -> Dict[str, Any]:
    if max_queries < 0:
        raise ValueError("max_queries must be non-negative.")
    if not context_token_ids:
        raise ValueError("Context must contain at least one token.")

    original_ids = [int(token_id) for token_id in context_token_ids]
    editable_units, protected_positions = _scope_context_units(
        *_editable_context_units(original_ids, tokenizer),
        editable_context_token_indices,
    )
    unit_positions = {
        unit_index: set(positions)
        for unit_index, positions in enumerate(editable_units)
    }
    retained_units = set(unit_positions)
    cached_logits: Dict[Tuple[int, ...], List[float]] = {
        tuple(original_ids): list(baseline_logits)
    }
    query_count = 0
    budget_exhausted = False
    empty_context_blocked = False
    singleton_sweep_complete = not editable_units
    deletion_witnesses: Dict[int, Dict[str, Any]] = {}

    def candidate_ids(units: set[int]) -> List[int]:
        retained_positions = set(protected_positions)
        for unit_index in units:
            retained_positions.update(unit_positions[unit_index])
        return [
            token_id for index, token_id in enumerate(original_ids)
            if index in retained_positions
        ]

    def evaluate(units: set[int]) -> Tuple[List[int], List[float]] | None:
        nonlocal query_count, budget_exhausted, empty_context_blocked
        ids = candidate_ids(units)
        if not ids:
            empty_context_blocked = True
            return None
        key = tuple(ids)
        logits = cached_logits.get(key)
        if logits is None:
            if query_count >= max_queries:
                budget_exhausted = True
                return None
            logits = _logit_values(oracle_query_token_ids(ids))
            query_count += 1
            if len(logits) != len(baseline_logits):
                raise ValueError("Oracle vocabulary changed during context minimization.")
            cached_logits[key] = logits
        return ids, logits

    granularity = 2
    while retained_units:
        ordered_units = sorted(retained_units)
        parts = min(granularity, len(ordered_units))
        removed = False
        deletion_witnesses = {}
        for chunk in _partition_positions(ordered_units, parts):
            candidate_units = retained_units.difference(chunk)
            evaluated = evaluate(candidate_units)
            if evaluated is None:
                if budget_exhausted:
                    break
                continue
            candidate_context, logits = evaluated
            winner_id = max(range(len(logits)), key=logits.__getitem__)
            if winner_id == target_token_id:
                retained_units = candidate_units
                granularity = max(2, parts - 1)
                removed = True
                break
            if len(chunk) == 1 and parts == len(ordered_units):
                unit_index = chunk[0]
                removed_positions = sorted(unit_positions[unit_index])
                competitors = [
                    index for index in range(len(logits))
                    if index != target_token_id
                ]
                deletion_witnesses[unit_index] = {
                    "removedContextUnitIndex": unit_index,
                    "removedContextTokenIndices": removed_positions,
                    "removedContextTokenIds": [original_ids[index] for index in removed_positions],
                    "removedContextTokens": [
                        _decode_token(tokenizer, original_ids[index])
                        for index in removed_positions
                    ],
                    "counterfactualWinnerTokenId": winner_id,
                    "counterfactualWinnerToken": _decode_token(tokenizer, winner_id),
                    "targetVsBestAlternativeMargin": (
                        logits[target_token_id]
                        - max(logits[index] for index in competitors)
                    ),
                    "counterfactualContextTokenIds": candidate_context,
                }
        if budget_exhausted:
            break
        if removed:
            continue
        if parts == len(ordered_units):
            singleton_sweep_complete = True
            break
        granularity = min(len(ordered_units), parts * 2)

    retained_context_ids = candidate_ids(retained_units)
    retained_logits = cached_logits.get(tuple(retained_context_ids), list(baseline_logits))
    one_minimal = (
        (not retained_units or singleton_sweep_complete)
        and not empty_context_blocked
    )
    return {
        "targetTokenId": target_token_id,
        "targetToken": _decode_token(tokenizer, target_token_id),
        "retainedContextTokenIds": retained_context_ids,
        "retainedContextTokens": [
            {
                "contextTokenIndex": index,
                "tokenId": token_id,
                "token": _decode_token(tokenizer, token_id),
                "protected": index in protected_positions,
            }
            for index, token_id in enumerate(original_ids)
            if index in protected_positions
            or any(index in unit_positions[unit_index] for unit_index in retained_units)
        ],
        "retainedEditableUnitCount": len(retained_units),
        "originalEditableUnitCount": len(editable_units),
        "retainedEditableTokenCount": sum(
            len(unit_positions[unit_index]) for unit_index in retained_units
        ),
        "originalEditableTokenCount": sum(len(positions) for positions in editable_units),
        "protectedTokenCount": len(protected_positions),
        "targetRemainsTop1": (
            max(
                range(len(retained_logits)),
                key=retained_logits.__getitem__,
            ) == target_token_id
        ),
        "minimality": {
            "oneMinimalAmongEditableUnits": one_minimal,
            "globalMinimumProven": False,
            "budgetExhausted": budget_exhausted,
            "blockedByEmptyContext": empty_context_blocked,
            "meaning": (
                "Every retained editable unit was individually tested and removing it "
                "changed the top-1 token."
                if one_minimal else
                "This is a budget-limited sufficient context; unit-level minimality was not established."
            ),
        },
        "singleTokenDeletionWitnesses": [
            deletion_witnesses[index]
            for index in sorted(deletion_witnesses)
        ],
        "oracleQueryCount": query_count,
        "oracleQueryBudget": max_queries,
    }


def trace_greedy_generation(
    *,
    oracle_query_token_ids: Callable[[Sequence[int]], Any],
    tokenizer: Any,
    input_token_ids: Sequence[int],
    max_new_tokens: int = 32,
    probes_per_token: int = 4,
    top_token_count: int = 8,
    response_probe_count: int = 0,
    explanation_probes_per_token: int = 0,
    interaction_probe_count: int = 0,
    branch_probe_count: int = 0,
    branch_width: int = 3,
    editable_input_token_indices: Sequence[int] | None = None,
    editable_input_scope: str = "full_prompt",
    reserved_input_token_indices: Sequence[int] | None = None,
    priority_input_token_indices: Sequence[int] | None = None,
    reserved_pairwise_input_token_indices: Sequence[int] | None = None,
) -> Dict[str, Any]:
    if not input_token_ids:
        raise ValueError("At least one input token is required.")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive.")
    if probes_per_token < 0:
        raise ValueError("probes_per_token must be non-negative.")
    if top_token_count < 2:
        raise ValueError("top_token_count must be at least two.")
    if response_probe_count < 0:
        raise ValueError("response_probe_count must be non-negative.")
    if explanation_probes_per_token < 0:
        raise ValueError("explanation_probes_per_token must be non-negative.")
    if interaction_probe_count < 0:
        raise ValueError("interaction_probe_count must be non-negative.")
    if branch_probe_count < 0:
        raise ValueError("branch_probe_count must be non-negative.")
    if branch_width < 1:
        raise ValueError("branch_width must be positive.")
    if editable_input_scope not in {"full_prompt", "user_message_content"}:
        raise ValueError(
            "editable_input_scope must be 'full_prompt' or 'user_message_content'."
        )
    if (
        editable_input_scope == "user_message_content"
        and editable_input_token_indices is None
    ):
        raise ValueError(
            "user_message_content scope requires editable_input_token_indices."
        )

    original_input = [int(token_id) for token_id in input_token_ids]
    context = list(original_input)
    special_ids = set(getattr(tokenizer, "all_special_ids", []))
    if editable_input_token_indices is not None and any(
        int(position) < 0 or int(position) >= len(original_input)
        for position in editable_input_token_indices
    ):
        raise ValueError("editable_input_token_indices contains an out-of-range position.")
    editable_input_units, protected_input_positions = _scope_context_units(
        *_editable_context_units(original_input, tokenizer),
        editable_input_token_indices,
    )
    measured_editable_input_indices = sorted({
        position
        for unit in editable_input_units
        for position in unit
    })
    reserved_input_positions = {
        int(position) for position in (reserved_input_token_indices or [])
    }
    if not reserved_input_positions.issubset(measured_editable_input_indices):
        raise ValueError(
            "reserved_input_token_indices must be inside the editable input scope."
        )
    probeable_input_units = [
        unit for unit in editable_input_units
        if not reserved_input_positions.intersection(unit)
    ]
    covered_reserved_positions = {
        position
        for unit in editable_input_units
        if reserved_input_positions.intersection(unit)
        for position in unit
    }
    if covered_reserved_positions != reserved_input_positions:
        raise ValueError(
            "reserved_input_token_indices must contain complete editable units."
        )
    priority_input_positions = {
        int(position) for position in (priority_input_token_indices or [])
    }
    if not priority_input_positions.issubset(measured_editable_input_indices):
        raise ValueError(
            "priority_input_token_indices must be inside the editable input scope."
        )
    covered_priority_positions = {
        position
        for unit in editable_input_units
        if priority_input_positions.intersection(unit)
        for position in unit
    }
    if covered_priority_positions != priority_input_positions:
        raise ValueError(
            "priority_input_token_indices must contain complete editable units."
        )
    if priority_input_positions.intersection(reserved_input_positions):
        raise ValueError(
            "priority_input_token_indices cannot overlap reserved input units."
        )
    reserved_pairwise_positions = {
        int(position)
        for position in (reserved_pairwise_input_token_indices or [])
    }
    if not reserved_pairwise_positions.issubset(measured_editable_input_indices):
        raise ValueError(
            "reserved_pairwise_input_token_indices must be inside the editable input scope."
        )
    covered_pairwise_positions = {
        position
        for unit in editable_input_units
        if reserved_pairwise_positions.intersection(unit)
        for position in unit
    }
    if covered_pairwise_positions != reserved_pairwise_positions:
        raise ValueError(
            "reserved_pairwise_input_token_indices must contain complete editable units."
        )
    probeable_input_indices = sorted({
        position
        for unit in probeable_input_units
        for position in unit
    })
    special_input_count = sum(token_id in special_ids for token_id in original_input)
    whitespace_input_count = sum(
        token_id not in special_ids
        and not _raw_decode_token(tokenizer, token_id).strip()
        for token_id in original_input
    )
    eos_id = getattr(tokenizer, "eos_token_id", None)
    steps: List[Dict[str, Any]] = []
    query_count = 0
    explanation_query_count = 0
    interaction_query_count = 0
    branch_query_count = 0
    input_token_evidence: Dict[Tuple[int, ...], Dict[str, Any]] = {}

    for step_index in range(max_new_tokens):
        baseline_logits = _logit_values(oracle_query_token_ids(context))
        query_count += 1
        if len(baseline_logits) <= top_token_count:
            raise ValueError("top_token_count must be smaller than the oracle vocabulary size.")

        target_id = max(range(len(baseline_logits)), key=baseline_logits.__getitem__)
        target_logit = baseline_logits[target_id]
        alternatives = _top_tokens(
            tokenizer,
            baseline_logits,
            min(len(baseline_logits), top_token_count + 1),
        )
        contrast_ids = [item["tokenId"] for item in alternatives if item["tokenId"] != target_id][:top_token_count]
        baseline_margin = target_logit - max(baseline_logits[token_id] for token_id in contrast_ids)
        baseline_probability = _softmax(baseline_logits)[target_id]
        probes: List[Dict[str, Any]] = []

        generated_units, _ = _editable_context_units(
            context[len(original_input):],
            tokenizer,
        )
        generated_units = [
            [position + len(original_input) for position in unit]
            for unit in generated_units
        ]
        generated_probe_budget = min(len(generated_units), probes_per_token // 2)
        if generated_units and probes_per_token and generated_probe_budget == 0:
            generated_probe_budget = 1
        input_probe_budget = probes_per_token - generated_probe_budget
        input_observation_counts = [
            int(input_token_evidence.get(tuple(unit), {}).get("observedDecisionCount", 0))
            for unit in probeable_input_units
        ]
        priority_unit_indexes = [
            index
            for index, unit in enumerate(probeable_input_units)
            if priority_input_positions.intersection(unit)
        ][:input_probe_budget]
        remaining_unit_indexes = [
            index
            for index in range(len(probeable_input_units))
            if index not in priority_unit_indexes
        ]
        balanced_remaining_indexes = _balanced_indices(
            len(remaining_unit_indexes),
            max(0, input_probe_budget - len(priority_unit_indexes)),
            [
                input_observation_counts[index]
                for index in remaining_unit_indexes
            ],
            step_index,
        )
        selected_input_unit_indexes = sorted([
            *priority_unit_indexes,
            *[
                remaining_unit_indexes[index]
                for index in balanced_remaining_indexes
            ],
        ])
        selected_generated_unit_indexes = _bounded_indices(
            list(range(len(generated_units))),
            generated_probe_budget,
        )
        intervention_units = [
            (probeable_input_units[index], "input")
            for index in selected_input_unit_indexes
        ] + [
            (generated_units[index], "generated_prefix")
            for index in selected_generated_unit_indexes
        ]
        intervention_units.sort(key=lambda item: item[0][0])

        for unit_positions, source in intervention_units:
            removed_positions = set(unit_positions)
            counterfactual_context = [
                token_id for index, token_id in enumerate(context)
                if index not in removed_positions
            ]
            if not counterfactual_context:
                continue
            changed_logits = _logit_values(oracle_query_token_ids(counterfactual_context))
            query_count += 1
            if len(changed_logits) != len(baseline_logits):
                raise ValueError("Oracle vocabulary changed during a token intervention.")
            changed_margin = changed_logits[target_id] - max(
                changed_logits[token_id] for token_id in contrast_ids
            )
            margin_support = baseline_margin - changed_margin
            token_surface = "".join(
                _decode_token(tokenizer, context[index])
                for index in unit_positions
            )
            token_ids = [context[index] for index in unit_positions]
            context_position = unit_positions[0]
            probes.append({
                "contextTokenIndex": context_position,
                "contextTokenIndices": unit_positions,
                "source": source,
                "inputTokenIndex": context_position if context_position < len(original_input) else None,
                "generatedTokenIndex": (
                    context_position - len(original_input)
                    if context_position >= len(original_input) else None
                ),
                "inputTokenId": token_ids[0],
                "inputTokenIds": token_ids,
                "inputToken": token_surface,
                "counterfactualContextTokenIds": counterfactual_context,
                "targetLogit": changed_logits[target_id],
                "targetRank": _rank(changed_logits, target_id),
                "targetVsContrastMargin": changed_margin,
                "marginSupport": margin_support,
                "distributionJensenShannonBits": _js_divergence(baseline_logits, changed_logits),
                "topTokens": _top_tokens(tokenizer, changed_logits, min(top_token_count, len(changed_logits))),
                "interpretation": (
                    "supports_target" if margin_support > 0.0
                    else "suppresses_target" if margin_support < 0.0
                    else "no_margin_change"
                ),
            })
            if source == "input":
                evidence = input_token_evidence.setdefault(
                    tuple(unit_positions),
                    {
                        "inputTokenIds": token_ids,
                        "inputToken": token_surface,
                        "maxAbsMarginSupport": 0.0,
                        "observedDecisionCount": 0,
                    },
                )
                evidence["maxAbsMarginSupport"] = max(
                    evidence["maxAbsMarginSupport"],
                    abs(margin_support),
                )
                evidence["observedDecisionCount"] += 1

        probes.sort(key=lambda item: (-abs(float(item["marginSupport"])), item["contextTokenIndex"]))
        interactions: List[Dict[str, Any]] = []
        if interaction_probe_count and len(probes) > 1:
            candidate_units = probes
            candidate_pairs = list(combinations(candidate_units, 2))
            candidate_pairs.sort(key=lambda pair: (
                -(
                    abs(float(pair[0]["marginSupport"]))
                    + abs(float(pair[1]["marginSupport"]))
                ),
                pair[0]["contextTokenIndex"],
                pair[1]["contextTokenIndex"],
            ))
            candidate_pairs = [
                (left, right)
                for left, right in candidate_pairs
                if not (
                    left["source"] == "input"
                    and right["source"] == "input"
                    and set(left["contextTokenIndices"])
                    .union(right["contextTokenIndices"])
                    == reserved_pairwise_positions
                )
            ]
            for left, right in candidate_pairs[:interaction_probe_count]:
                removed_positions = set(left["contextTokenIndices"])
                removed_positions.update(right["contextTokenIndices"])
                counterfactual_context = [
                    token_id for index, token_id in enumerate(context)
                    if index not in removed_positions
                ]
                if not counterfactual_context:
                    continue
                changed_logits = _logit_values(
                    oracle_query_token_ids(counterfactual_context)
                )
                query_count += 1
                interaction_query_count += 1
                if len(changed_logits) != len(baseline_logits):
                    raise ValueError(
                        "Oracle vocabulary changed during a pairwise intervention."
                    )
                changed_margin = changed_logits[target_id] - max(
                    changed_logits[token_id] for token_id in contrast_ids
                )
                additive_margin = (
                    baseline_margin
                    - float(left["marginSupport"])
                    - float(right["marginSupport"])
                )
                interactions.append({
                    "deletedUnits": [
                        {
                            "source": item["source"],
                            "contextTokenIndices": item["contextTokenIndices"],
                            "tokenIds": item["inputTokenIds"],
                            "text": item["inputToken"],
                        }
                        for item in (left, right)
                    ],
                    "counterfactualContextTokenIds": counterfactual_context,
                    "targetVsContrastMargin": changed_margin,
                    "additivePredictedMargin": additive_margin,
                    "interactionResidual": changed_margin - additive_margin,
                    "singleDeletionMargins": [
                        float(baseline_margin - item["marginSupport"])
                        for item in (left, right)
                    ],
                    "interpretation": (
                        "joint_deletion_suppresses_target_more_than_additive"
                        if changed_margin < additive_margin
                        else "joint_deletion_suppresses_target_less_than_additive"
                        if changed_margin > additive_margin
                        else "consistent_with_additive_margin_effects"
                    ),
                    "topTokens": _top_tokens(
                        tokenizer,
                        changed_logits,
                        min(top_token_count, len(changed_logits)),
                    ),
                })
        branches: List[Dict[str, Any]] = []
        if branch_probe_count and (
            eos_id is None or target_id != int(eos_id)
        ):
            branch_candidates = alternatives[1:1 + branch_probe_count]
            baseline_log_probabilities = _log_softmax(baseline_logits)
            for candidate in branch_candidates:
                candidate_id = int(candidate["tokenId"])
                beams: List[Tuple[List[int], float]] = [
                    ([candidate_id], baseline_log_probabilities[candidate_id])
                ]
                branch_queries = 0
                branch_budget = max_new_tokens - step_index
                while any(
                    len(tokens) < branch_budget
                    and (eos_id is None or tokens[-1] != int(eos_id))
                    for tokens, _ in beams
                ):
                    expanded: List[Tuple[List[int], float]] = []
                    for branch_suffix, cumulative_log_probability in beams:
                        if (
                            len(branch_suffix) >= branch_budget
                            or (eos_id is not None and branch_suffix[-1] == int(eos_id))
                        ):
                            expanded.append((branch_suffix, cumulative_log_probability))
                            continue
                        branch_context = [
                            *context,
                            *branch_suffix[len(context) - len(original_input):],
                        ]
                        branch_logits = _logit_values(
                            oracle_query_token_ids(branch_context)
                        )
                        query_count += 1
                        branch_query_count += 1
                        branch_queries += 1
                        if len(branch_logits) != len(baseline_logits):
                            raise ValueError(
                                "Oracle vocabulary changed during an alternative branch."
                            )
                        branch_log_probabilities = _log_softmax(branch_logits)
                        candidate_ids = sorted(
                            range(len(branch_logits)),
                            key=branch_logits.__getitem__,
                            reverse=True,
                        )[:min(branch_width, len(branch_logits))]
                        expanded.extend(
                            (
                                [*branch_suffix, next_id],
                                cumulative_log_probability
                                + branch_log_probabilities[next_id],
                            )
                            for next_id in candidate_ids
                        )
                    beams = sorted(
                        expanded,
                        key=lambda item: item[1],
                        reverse=True,
                    )[:branch_width]

                continuations: List[Dict[str, Any]] = []
                baseline_generated_through_step = (
                    context[len(original_input):] + [target_id]
                )
                for branch_suffix, cumulative_log_probability in beams:
                    branch_generated = [
                        *context[len(original_input):],
                        *branch_suffix,
                    ]
                    common_prefix_length = 0
                    for baseline_id, branch_id in zip(
                        baseline_generated_through_step,
                        branch_generated,
                    ):
                        if baseline_id != branch_id:
                            break
                        common_prefix_length += 1
                    continuations.append({
                        "tokenIds": branch_generated,
                        "text": tokenizer.decode(
                            branch_generated,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False,
                        ),
                        "cumulativeLogProbability": cumulative_log_probability,
                        "meanTokenLogProbability": (
                            cumulative_log_probability / len(branch_suffix)
                        ),
                        "generatedTokenCount": len(branch_suffix),
                        "commonPrefixTokenCount": common_prefix_length,
                        "firstDivergentTokenIndex": (
                            common_prefix_length
                            if common_prefix_length < min(
                                len(baseline_generated_through_step),
                                len(branch_generated),
                            )
                            else None
                        ),
                    })
                best_continuation = continuations[0]
                branches.append({
                    "forcedTokenId": candidate_id,
                    "forcedToken": candidate["token"],
                    "forcedTokenRank": candidate["rank"],
                    "forcedTokenLogit": candidate["logit"],
                    "greedyVsForcedLogitMargin": target_logit - float(candidate["logit"]),
                    "requestedBeamWidth": branch_width,
                    "observedContinuationCount": len(continuations),
                    "continuations": continuations,
                    "branchTokenIds": best_continuation["tokenIds"],
                    "branchContinuationTokenIds": best_continuation["tokenIds"][
                        len(context) - len(original_input):
                    ],
                    "branchText": best_continuation["text"],
                    "commonPrefixTokenCount": best_continuation[
                        "commonPrefixTokenCount"
                    ],
                    "firstDivergentTokenIndex": best_continuation[
                        "firstDivergentTokenIndex"
                    ],
                    "oracleQueryCount": branch_queries,
                    "decodePolicyAfterForcedToken": "beam_log_probability",
                })
        sufficient_context = _minimal_sufficient_context(
            oracle_query_token_ids=oracle_query_token_ids,
            tokenizer=tokenizer,
            context_token_ids=context,
            target_token_id=target_id,
            baseline_logits=baseline_logits,
            max_queries=explanation_probes_per_token,
            editable_context_token_indices=(
                *probeable_input_indices,
                *range(len(original_input), len(context)),
            ),
        )
        sufficient_context["minimality"]["scope"] = (
            "editable_context_units_excluding_reserved_input_holdout"
            if reserved_input_positions
            else "all_editable_context_units"
        )
        sufficient_context["excludedHeldOutInputTokenIndices"] = sorted(
            reserved_input_positions
        )
        if reserved_input_positions:
            sufficient_context["minimality"]["meaning"] += (
                " Reserved held-out input units were excluded from this search."
            )
        query_count += sufficient_context["oracleQueryCount"]
        explanation_query_count += sufficient_context["oracleQueryCount"]
        token_text = _decode_token(tokenizer, target_id)
        steps.append({
            "stepIndex": step_index,
            "contextTokenIds": list(context),
            "chosenTokenId": target_id,
            "chosenToken": token_text,
            "chosenTokenLogit": target_logit,
            "chosenTokenProbability": baseline_probability,
            "chosenTokenRank": _rank(baseline_logits, target_id),
            "marginVsTopAlternatives": baseline_margin,
            "topAlternatives": alternatives,
            "contextTokenInterventions": probes,
            "pairwiseInterventions": interactions,
            "alternativeBranches": branches,
            "minimalSufficientContext": sufficient_context,
            "probedGeneratedPrefixUnitCount": len(selected_generated_unit_indexes),
            "eligibleGeneratedPrefixUnitCount": len(generated_units),
            "probedInputUnitCount": len(selected_input_unit_indexes),
            "eligibleInputUnitCount": len(editable_input_units),
            "probeableInputUnitCount": len(probeable_input_units),
            "reservedInputUnitCount": (
                len(editable_input_units) - len(probeable_input_units)
            ),
            "unprobedInputUnitCount": (
                len(probeable_input_units) - len(selected_input_unit_indexes)
            ),
            "strongestSupportingToken": next(
                (item for item in probes if item["marginSupport"] > 0.0), None
            ),
            "strongestSuppressingToken": next(
                (item for item in probes if item["marginSupport"] < 0.0), None
            ),
        })
        context.append(target_id)
        if eos_id is not None and target_id == int(eos_id):
            break

    generated_ids = context[len(original_input):]
    terminated_by_eos = bool(
        eos_id is not None
        and generated_ids
        and generated_ids[-1] == int(eos_id)
    )
    response_counterfactuals: List[Dict[str, Any]] = []
    response_query_count = 0
    selected_response_probes = sorted(
        input_token_evidence.items(),
        key=lambda item: (-float(item[1]["maxAbsMarginSupport"]), item[0]),
    )[:response_probe_count]
    for input_positions, evidence in selected_response_probes:
        removed_positions = set(input_positions)
        counterfactual_input = [
            token_id for index, token_id in enumerate(original_input)
            if index not in removed_positions
        ]
        if not counterfactual_input:
            continue
        counterfactual_context = list(counterfactual_input)
        counterfactual_generated: List[int] = []
        for _ in range(max_new_tokens):
            counterfactual_logits = _logit_values(
                oracle_query_token_ids(counterfactual_context)
            )
            query_count += 1
            response_query_count += 1
            counterfactual_id = max(
                range(len(counterfactual_logits)),
                key=counterfactual_logits.__getitem__,
            )
            counterfactual_generated.append(counterfactual_id)
            counterfactual_context.append(counterfactual_id)
            if eos_id is not None and counterfactual_id == int(eos_id):
                break
        common_prefix_length = 0
        for baseline_id, counterfactual_id in zip(generated_ids, counterfactual_generated):
            if baseline_id != counterfactual_id:
                break
            common_prefix_length += 1
        response_counterfactuals.append({
            "deletedInputTokenIndex": input_positions[0],
            "deletedInputTokenIndices": list(input_positions),
            "deletedInputTokenId": evidence["inputTokenIds"][0],
            "deletedInputTokenIds": evidence["inputTokenIds"],
            "deletedInputToken": evidence["inputToken"],
            "counterfactualInputTokenIds": counterfactual_input,
            "selectionEvidence": {
                "maxAbsNextTokenMarginChange": evidence["maxAbsMarginSupport"],
                "observedDecisionCount": evidence["observedDecisionCount"],
            },
            "baselineGeneratedTokenIds": generated_ids,
            "counterfactualGeneratedTokenIds": counterfactual_generated,
            "counterfactualGeneratedText": tokenizer.decode(
                counterfactual_generated,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ),
            "counterfactualTerminatedByEos": bool(
                eos_id is not None
                and counterfactual_generated
                and counterfactual_generated[-1] == int(eos_id)
            ),
            "commonPrefixTokenCount": common_prefix_length,
            "firstDivergentTokenIndex": (
                common_prefix_length
                if common_prefix_length < min(len(generated_ids), len(counterfactual_generated))
                else None
            ),
            "responseChanged": generated_ids != counterfactual_generated,
        })
    return {
        "mode": "autoregressive_blackbox_decision_trace",
        "decodePolicy": "greedy_argmax",
        "inputTokenIds": original_input,
        "generatedTokenIds": generated_ids,
        "generatedText": tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ),
        "maxNewTokens": max_new_tokens,
        "terminatedByEos": terminated_by_eos,
        "terminationReason": "eos" if terminated_by_eos else "max_new_tokens",
        "outputComplete": terminated_by_eos,
        "probesPerToken": probes_per_token,
        "explanationProbesPerToken": explanation_probes_per_token,
        "interactionProbesPerToken": interaction_probe_count,
        "branchProbesPerToken": branch_probe_count,
        "branchWidth": branch_width,
        "inputTokenCount": len(original_input),
        "editableInputTokenIndices": measured_editable_input_indices,
        "heldOutInputTokenIndices": sorted(reserved_input_positions),
        "heldOutPairwiseInputTokenIndices": sorted(
            reserved_pairwise_positions
        ),
        "editableInputScope": editable_input_scope,
        "protectedInputTokenCount": len(protected_input_positions),
        "specialInputTokenCount": special_input_count,
        "whitespaceInputTokenCount": whitespace_input_count,
        "steps": steps,
        "responseCounterfactuals": response_counterfactuals,
        "responseProbeSummary": {
            "requested": response_probe_count,
            "selected": len(response_counterfactuals),
            "eligibleMeasuredInputUnits": len(input_token_evidence),
            "eligibleMeasuredInputTokens": len(input_token_evidence),
            "oracleQueryCount": response_query_count,
        },
        "explanationQueryCount": explanation_query_count,
        "interactionQueryCount": interaction_query_count,
        "branchQueryCount": branch_query_count,
        "queryCount": query_count,
        "claimBoundary": (
            "This is a greedy output-behavior trace with exact single- and pairwise "
            "token-deletion interventions, "
            "including full response regeneration for selected measured prompt-token deletions. "
            "Selected alternative-token branches force one observed candidate and retain "
            "bounded high-probability continuations using beam search. "
            "Pairwise interaction residuals describe non-additivity of measured logit margins "
            "for the tested deletions only. "
            "It does not expose hidden activations, weights, or a unique internal reasoning mechanism."
        ),
    }
