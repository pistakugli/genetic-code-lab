"""Why-report layer for the live ordered-path stability lane.

Given a recorded ordered-path stability/OOD-stress artifact, this module
explains, per broken paraphrase, why the original path broke and where the
measured causal effect went:

- break classification (sign flip vs magnitude collapse vs relative falloff);
- layer-shift rescue outcome (same components, shifted layers);
- replacement-path comparison against the source circuit (layer overlap,
  component overlap, position movement, earlier/later direction);
- held-out family transfer diagnosis for the replacement path;
- a number-backed narrative statement and an explicit claim boundary.

The analysis is deterministic post-processing of numbers already recorded by
`tools/llm-reader-ordered-path-stability.py`. It performs no model queries,
adds no measurements, and invents no values: every sentence is assembled from
fields present in the artifact. The independent verifier recomputes the same
comparisons from the raw fields so a fabricated why block fails verification.
"""

from __future__ import annotations

from typing import Any, Dict, List, Set, Tuple


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _layer_component_ids(interventions: List[Dict[str, Any]], id_field: str) -> Set[Tuple[int, int]]:
    ids: Set[Tuple[int, int]] = set()
    for intervention in interventions or []:
        layer = intervention.get("layer")
        component = intervention.get(id_field)
        if isinstance(layer, int) and isinstance(component, int):
            ids.add((layer, component))
    return ids


def _ablate_positions(interventions: List[Dict[str, Any]]) -> Set[int]:
    positions: Set[int] = set()
    for intervention in interventions or []:
        for position in intervention.get("ablateTokenPositions", []) or []:
            if isinstance(position, int):
                positions.add(position)
    return positions


def classify_break(
    *,
    broken_drop: float,
    original_drop: float,
    min_original_drop: float,
    min_ratio: float,
) -> Dict[str, Any]:
    """Classify why a paraphrase replay counted as broken.

    A break is either a sign flip (ablating the original components raises the
    target logit, so the original path is counterproductive under the new
    phrasing), a magnitude collapse (the drop is positive but below the
    absolute gate), or a relative falloff (the drop clears the absolute gate
    but falls below the configured ratio to the original replay).
    """

    ratio = broken_drop / max(abs(original_drop), 1e-9) if original_drop >= 0 else None
    if broken_drop < 0:
        kind = "sign-flip"
        detail = (
            "Ablating the original components raises the target logit under this "
            "phrasing, so the original path is not carrying the behaviour; the "
            "recorded components are counterproductive here."
        )
    elif broken_drop < min_original_drop:
        kind = "magnitude-collapse"
        detail = (
            "Ablating the original components leaves the target logit almost "
            "unchanged under this phrasing, so the original path has lost its "
            "measured causal effect."
        )
    else:
        kind = "relative-falloff"
        detail = (
            "The replay drop clears the absolute gate but falls below the "
            "configured ratio to the original replay, so the path retains only a "
            "fraction of its source-prompt effect."
        )
    return {
        "kind": kind,
        "brokenTargetLogitDrop": broken_drop,
        "originalTargetLogitDrop": original_drop,
        "ratioToOriginal": ratio,
        "minOriginalDrop": min_original_drop,
        "minParaphraseDropRatio": min_ratio,
        "interpretation": detail,
    }


def compare_paths(
    *,
    source_nodes: Dict[str, Any],
    replacement: Dict[str, Any],
) -> Dict[str, Any]:
    """Compare the source circuit with a rediscovered replacement path."""

    source_mlp = _layer_component_ids(source_nodes.get("mlp", []), "neuron")
    source_attention = _layer_component_ids(source_nodes.get("attention", []), "head")
    replacement_mlp = _layer_component_ids(replacement.get("mlpInterventions", []), "neuron")
    replacement_attention = _layer_component_ids(replacement.get("attentionInterventions", []), "head")
    source_layers = {layer for layer, _ in source_mlp | source_attention}
    replacement_layers = {layer for layer, _ in replacement_mlp | replacement_attention}
    source_positions = _ablate_positions(source_nodes.get("mlp", [])) | _ablate_positions(
        source_nodes.get("attention", [])
    )
    replacement_positions = _ablate_positions(replacement.get("mlpInterventions", [])) | _ablate_positions(
        replacement.get("attentionInterventions", [])
    )
    shared_layers = sorted(source_layers & replacement_layers)
    shared_mlp = sorted(source_mlp & replacement_mlp)
    shared_attention = sorted(source_attention & replacement_attention)
    component_overlap = len(shared_mlp) + len(shared_attention)
    source_component_count = len(source_mlp) + len(source_attention)
    if source_layers and replacement_layers:
        layer_delta = sum(replacement_layers) / len(replacement_layers) - sum(source_layers) / len(source_layers)
    else:
        layer_delta = None
    if component_overlap and shared_layers:
        change_kind = "component-retention"
        change_detail = "The replacement path reuses measured source components in overlapping layers."
    elif component_overlap:
        change_kind = "position-rebinding"
        change_detail = "The replacement path reuses source components at different layers."
    elif replacement_layers and source_layers and layer_delta is not None and abs(layer_delta) < 1e-12:
        change_kind = "layer-local-rebinding"
        change_detail = "The replacement path uses different components in the same layers."
    else:
        change_kind = "component-rebinding"
        change_detail = "The replacement path uses components that do not overlap the source circuit."
    return {
        "sourceLayers": sorted(source_layers),
        "replacementLayers": sorted(replacement_layers),
        "layerOverlap": shared_layers,
        "meanLayerDelta": layer_delta,
        "direction": (
            "earlier" if layer_delta is not None and layer_delta < 0
            else "later" if layer_delta is not None and layer_delta > 0
            else "same"
        ),
        "sourceMlpComponents": sorted(source_mlp),
        "replacementMlpComponents": sorted(replacement_mlp),
        "sourceAttentionComponents": sorted(source_attention),
        "replacementAttentionComponents": sorted(replacement_attention),
        "mlpOverlap": shared_mlp,
        "attentionOverlap": shared_attention,
        "componentOverlapCount": component_overlap,
        "sourceComponentCount": source_component_count,
        "sourceAblatePositions": sorted(source_positions),
        "replacementAblatePositions": sorted(replacement_positions),
        "changeKind": change_kind,
        "interpretation": change_detail,
    }


def _transfer_diagnosis(transfer: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(transfer, dict):
        return {"status": "absent", "sourceFamily": None, "attempted": None, "passed": None, "passRate": None, "failingFamilies": []}
    failing: List[Dict[str, Any]] = []
    for item in transfer.get("items", []) or []:
        if item.get("pass") is False:
            drop = _number(item.get("targetLogitDrop"))
            ratio = _number(item.get("ratioToDiscovery"))
            min_drop = _number(transfer.get("minTargetLogitDrop"))
            min_ratio = _number(transfer.get("minDropRatio"))
            reasons = []
            if drop is not None and min_drop is not None and drop < min_drop:
                reasons.append("absolute-drop-below-threshold")
            if ratio is not None and min_ratio is not None and ratio < min_ratio:
                reasons.append("relative-ratio-below-threshold")
            failing.append({
                "family": item.get("family"),
                "targetLogitDrop": drop,
                "ratioToDiscovery": ratio,
                "belowThresholds": reasons or ["unrecorded"],
            })
    summary = transfer.get("summary", {}) or {}
    return {
        "status": transfer.get("status"),
        "sourceFamily": transfer.get("sourceFamily"),
        "attempted": summary.get("attempted"),
        "passed": summary.get("passed"),
        "passRate": summary.get("passRate"),
        "failingFamilies": failing,
    }


def _fmt(value: Any, digits: int = 4) -> str:
    number = _number(value)
    return "n/a" if number is None else f"{number:.{digits}f}"


def _components_phrase(nodes: Dict[str, Any]) -> str:
    mlp = _layer_component_ids(nodes.get("mlp", []), "neuron")
    attention = _layer_component_ids(nodes.get("attention", []), "head")
    parts = []
    if mlp:
        parts.append("MLP " + ", ".join(f"L{layer} n{neuron}" for layer, neuron in sorted(mlp)))
    if attention:
        parts.append("attention " + ", ".join(f"L{layer} h{head}" for layer, head in sorted(attention)))
    return " + ".join(parts) if parts else "no recorded components"


def _replacement_phrase(replacement: Dict[str, Any]) -> str:
    nodes = {
        "mlp": replacement.get("mlpInterventions", []),
        "attention": replacement.get("attentionInterventions", []),
    }
    return _components_phrase(nodes)


def build_rescue_why(
    *,
    case: Dict[str, Any],
    rescue_item: Dict[str, Any],
    config: Dict[str, Any],
    family: str = "",
) -> Dict[str, Any]:
    """Build the why-report entry for one broken-paraphrase rescue record.

    ``family`` is the caller-resolved prompt family: historical rescue items
    recorded before families were stored on the item are joined back to the
    case's paraphrase list by exact prompt match.
    """

    original = case.get("originalReplay", {}) or {}
    original_drop = _number(original.get("targetLogitDrop"))
    broken_drop = _number(rescue_item.get("brokenDrop"))
    min_original_drop = _number(config.get("minOriginalDrop")) or 0.0
    min_ratio = _number(config.get("minParaphraseDropRatio")) or 0.0
    if broken_drop is None or original_drop is None:
        raise ValueError(
            "why-report requires recorded target-logit drops for the broken replay "
            "and the original replay"
        )
    break_report = classify_break(
        broken_drop=broken_drop,
        original_drop=original_drop,
        min_original_drop=min_original_drop,
        min_ratio=min_ratio,
    )
    attempts = rescue_item.get("attempts", []) or []
    best_drop = _number(rescue_item.get("bestDrop"))
    layer_shift = {
        "attemptedShifts": [attempt.get("layerShift") for attempt in attempts],
        "bestLayerShift": rescue_item.get("bestLayerShift"),
        "bestShiftedTargetLogitDrop": best_drop,
        "recovered": bool(rescue_item.get("found")),
        "interpretation": (
            "Shifting the same components across neighbouring layers recovers the effect, "
            "so the break is a layer misplacement."
            if rescue_item.get("found")
            else "Shifting the same components across the configured layers never recovers "
            "the effect, so the break is not a layer misplacement of the same components."
        ),
    }
    replacement = rescue_item.get("replacementPathSearch", {}) or {}
    replacement_attempted = "replacementPathSearch" in rescue_item
    replacement_found = bool(replacement.get("found")) if replacement_attempted else False
    path_change = compare_paths(
        source_nodes=case.get("sourceCircuit", {}) or {},
        replacement=replacement,
    )
    transfer = _transfer_diagnosis(replacement.get("crossFamilyTransfer", {}))
    replacement_drop = _number(replacement.get("targetLogitDrop"))
    subject = str(case.get("subject", ""))
    if not family:
        family = str(rescue_item.get("family", ""))

    sentences: List[str] = []
    sentences.append(
        f"Under the {family} phrasing of the {subject} prompt the original path "
        f"({_components_phrase(case.get('sourceCircuit', {}) or {})}) broke as a "
        f"{break_report['kind']}: the remapped replay drop was {_fmt(broken_drop)} "
        f"against {_fmt(original_drop)} on the source prompt."
    )
    if layer_shift["attemptedShifts"]:
        if rescue_item.get("found"):
            sentences.append(
                f"A layer shift of {layer_shift['bestLayerShift']} alone recovered the effect."
            )
        else:
            sentences.append(
                f"Replaying the same components shifted across layers "
                f"({', '.join(str(shift) for shift in layer_shift['attemptedShifts'])}) reached at best "
                f"{_fmt(best_drop)}, so the effect did not simply move with those components."
            )
    if replacement_found:
        sentences.append(
            f"The replacement search found a path ({_replacement_phrase(replacement)}) with "
            f"target-logit drop {_fmt(replacement_drop)}; relative to the source circuit this is a "
            f"{path_change['changeKind']}: layers {path_change['sourceLayers']} -> "
            f"{path_change['replacementLayers']} ({path_change['direction']}), "
            f"{path_change['componentOverlapCount']}/{path_change['sourceComponentCount']} "
            f"components retained, ablate positions {path_change['sourceAblatePositions']} -> "
            f"{path_change['replacementAblatePositions']}."
        )
    else:
        if replacement_attempted:
            reason = replacement.get("failureReason")
            sentences.append(
                "The replacement search did not find a passing path"
                + (f" ({reason})." if reason else ".")
            )
        else:
            sentences.append(
                "The replacement-path search was not run for this break in the recorded "
                "run, so the artifact says nothing about where a replacement would be found."
            )
    if transfer.get("status") in ("passed", "failed"):
        failing = [str(item["family"]) for item in transfer["failingFamilies"]]
        sentences.append(
            f"On held-out families the replacement path passed {transfer['passed']}/"
            f"{transfer['attempted']}"
            + (f"; failing families: {', '.join(failing)}." if failing else ".")
        )
    why_statement = " ".join(sentences)

    return {
        "family": family,
        "prompt": rescue_item.get("prompt"),
        "break": break_report,
        "layerShiftRescue": layer_shift,
        "replacementPath": {
            "found": replacement_found,
            "attempted": replacement_attempted,
            "selectionMode": replacement.get("selectionMode"),
            "targetLogitDrop": replacement_drop,
            "pathChange": path_change,
        },
        "transfer": transfer,
        "whyStatement": why_statement,
        "claimBoundary": (
            "This entry explains where the measured target-logit causal effect moved "
            "between two recorded prompts of one subject. It is not semantic proof of "
            "what the components compute, not proof of a unique circuit, and not "
            "evidence about families, subjects, or models beyond those measured here."
        ),
    }


def _case_gates(case: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, float]:
    original = case.get("originalReplay", {}) or {}
    return {
        "original_drop": _number(original.get("targetLogitDrop")),
        "min_original_drop": _number(config.get("minOriginalDrop")) or 0.0,
        "min_ratio": _number(config.get("minParaphraseDropRatio")) or 0.0,
    }


def build_paraphrase_break(
    *,
    case: Dict[str, Any],
    paraphrase: Dict[str, Any],
    config: Dict[str, Any],
    rescue_by_prompt: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Classify one broken paraphrase from the recorded replay numbers.

    This applies the same break taxonomy as the rescue entries to *every*
    failed paraphrase, including the ones the recorded run never selected for
    a rescue search, so the artifact states why each measured replay failed
    rather than only the searched subset. ``rescueSought`` records honestly
    whether that prompt was rescue-searched at all.
    """

    gates = _case_gates(case, config)
    drop = _number(paraphrase.get("targetLogitDrop"))
    if drop is None or gates["original_drop"] is None:
        raise ValueError(
            "why-report requires recorded target-logit drops for every broken "
            "paraphrase and the original replay"
        )
    break_report = classify_break(
        broken_drop=drop,
        original_drop=gates["original_drop"],
        min_original_drop=gates["min_original_drop"],
        min_ratio=gates["min_ratio"],
    )
    rescue_item = rescue_by_prompt.get(str(paraphrase.get("prompt")))
    rescue_sought = rescue_item is not None
    layer_shift = None
    replacement_sought = False
    if rescue_item is not None:
        replacement_sought = "replacementPathSearch" in rescue_item
        layer_shift = {
            "recovered": bool(rescue_item.get("found")),
            "bestLayerShift": rescue_item.get("bestLayerShift"),
            "bestShiftedTargetLogitDrop": _number(rescue_item.get("bestDrop")),
        }
    return {
        "family": paraphrase.get("family"),
        "prompt": paraphrase.get("prompt"),
        "break": break_report,
        "rescueSought": rescue_sought,
        "layerShiftRescue": layer_shift,
        "replacementSought": replacement_sought,
        "claimBoundary": (
            "This entry classifies one measured replay from the recorded drops. "
            "It is not a semantic claim about the family, and it says nothing "
            "about prompts, subjects, or models that were not measured."
        ),
    }


def build_case_why(case: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    rescue = case.get("paraphraseRescue", {}) or {}
    items = rescue.get("items", []) or []
    paraphrases = [
        paraphrase
        for paraphrase in case.get("paraphrases", []) or []
        if isinstance(paraphrase, dict)
    ]
    family_by_prompt = {
        str(paraphrase.get("prompt")): str(paraphrase.get("family"))
        for paraphrase in paraphrases
    }
    rescue_by_prompt: Dict[str, Dict[str, Any]] = {}
    for rescue_item in items:
        if isinstance(rescue_item, dict):
            rescue_by_prompt[str(rescue_item.get("prompt"))] = rescue_item
    why_items = [
        build_rescue_why(
            case=case,
            rescue_item=rescue_item,
            config=config,
            family=family_by_prompt.get(str(rescue_item.get("prompt")), ""),
        )
        for rescue_item in items
        if isinstance(rescue_item, dict)
    ]
    paraphrase_breaks = [
        build_paraphrase_break(
            case=case,
            paraphrase=paraphrase,
            config=config,
            rescue_by_prompt=rescue_by_prompt,
        )
        for paraphrase in paraphrases
        if paraphrase.get("pass") is False
    ]
    break_kinds: Dict[str, int] = {}
    for entry in why_items:
        kind = entry["break"]["kind"]
        break_kinds[kind] = break_kinds.get(kind, 0) + 1
    paraphrase_break_kinds: Dict[str, int] = {}
    for entry in paraphrase_breaks:
        kind = entry["break"]["kind"]
        paraphrase_break_kinds[kind] = paraphrase_break_kinds.get(kind, 0) + 1
    replacements = [entry["replacementPath"] for entry in why_items]
    change_kinds: Dict[str, int] = {}
    for replacement in replacements:
        if replacement["found"]:
            change = replacement["pathChange"]["changeKind"]
            change_kinds[change] = change_kinds.get(change, 0) + 1
    directions = [
        replacement["pathChange"]["direction"]
        for replacement in replacements
        if replacement["found"]
    ]
    return {
        "subject": case.get("subject"),
        "rescueCount": len(why_items),
        "layerShiftOnlyRecoveries": sum(
            1 for entry in why_items if entry["layerShiftRescue"]["recovered"]
        ),
        "replacementFoundCount": sum(1 for replacement in replacements if replacement["found"]),
        "breakKinds": break_kinds,
        "replacementChangeKinds": change_kinds,
        "paraphraseBreakCount": len(paraphrase_breaks),
        "paraphraseBreakKinds": paraphrase_break_kinds,
        "rescueSoughtCount": sum(1 for entry in paraphrase_breaks if entry["rescueSought"]),
        "whereEffectWent": {
            "layerSets": [
                replacement["pathChange"]["replacementLayers"]
                for replacement in replacements
                if replacement["found"]
            ],
            "directions": sorted(set(directions)),
        },
        "items": why_items,
        "paraphraseBreaks": paraphrase_breaks,
    }


def _merge_counts(count_dicts) -> Dict[str, int]:
    merged: Dict[str, int] = {}
    for counts in count_dicts:
        for key, value in counts.items():
            merged[key] = merged.get(key, 0) + value
    return merged


def _case_has_why_input(case: Dict[str, Any]) -> bool:
    rescue = case.get("paraphraseRescue")
    if isinstance(rescue, dict) and rescue.get("items"):
        return True
    return any(
        isinstance(paraphrase, dict) and paraphrase.get("pass") is False
        for paraphrase in case.get("paraphrases", []) or []
    )


def attach_why_analysis(report: Dict[str, Any]) -> Dict[str, Any]:
    """Attach why-report blocks to a recorded stability/OOD-stress artifact.

    Mutates the given report in place by adding ``whyReport`` to every case
    that has rescue measurements or at least one broken paraphrase, a
    ``whyReport`` aggregate under ``summary``, and the ``whyReportEnabled``
    config marker. Returns the report.
    """

    config = report.get("config", {}) or {}
    for case in report.get("cases", []) or []:
        if not isinstance(case, dict):
            continue
        if not _case_has_why_input(case):
            continue
        case["whyReport"] = build_case_why(case, config)
    case_whys = [
        case["whyReport"]
        for case in report.get("cases", []) or []
        if isinstance(case, dict) and isinstance(case.get("whyReport"), dict)
    ]
    report.setdefault("summary", {})["whyReport"] = {
        "caseCount": len(case_whys),
        "rescueCount": sum(entry["rescueCount"] for entry in case_whys),
        "layerShiftOnlyRecoveries": sum(entry["layerShiftOnlyRecoveries"] for entry in case_whys),
        "replacementFoundCount": sum(entry["replacementFoundCount"] for entry in case_whys),
        "breakKinds": _merge_counts(entry["breakKinds"] for entry in case_whys),
        "replacementChangeKinds": _merge_counts(entry["replacementChangeKinds"] for entry in case_whys),
        "paraphraseBreakCount": sum(entry["paraphraseBreakCount"] for entry in case_whys),
        "paraphraseBreakKinds": _merge_counts(entry["paraphraseBreakKinds"] for entry in case_whys),
        "rescueSoughtCount": sum(entry["rescueSoughtCount"] for entry in case_whys),
    }
    report.setdefault("config", {})["whyReportEnabled"] = True
    return report


def _drop_cell(value: Any) -> str:
    number = _number(value)
    return "n/a" if number is None else f"{number:.6f}"


def render_why_markdown(report: Dict[str, Any]) -> List[str]:
    """Render the why-report markdown section for a stability artifact."""

    lines: List[str] = []
    case_whys = [
        (case.get("subject"), case["whyReport"])
        for case in report.get("cases", []) or []
        if isinstance(case, dict) and isinstance(case.get("whyReport"), dict) and case["whyReport"]
    ]
    if not case_whys:
        return lines
    lines.extend([
        "",
        "## Why the path broke",
        "",
        "Per broken paraphrase: why the replay failed, where the measured effect "
        "went, and whether the replacement path carries it. All values come from "
        "the recorded artifact; this is a comparative diagnosis, not a semantic claim.",
        "",
    ])
    for subject, why in case_whys:
        lines.append(f"### {subject}")
        lines.append("")
        breaks = why.get("paraphraseBreaks", [])
        if breaks:
            lines.append("Measured paraphrase breaks for this subject:")
            lines.append("")
            lines.append("| family | break kind | broken drop | ratio to original | rescue sought | layer-shift outcome |")
            lines.append("|---|---|---|---|---|---|")
            for entry in breaks:
                shift = entry.get("layerShiftRescue")
                if shift is None:
                    shift_cell = "not sought"
                elif shift.get("recovered"):
                    shift_cell = f"recovered (shift {shift.get('bestLayerShift')})"
                else:
                    shift_cell = f"not recovered (best shift {shift.get('bestLayerShift')})"
                lines.append(
                    f"| {entry.get('family')} | {entry['break']['kind']} | "
                    f"{_drop_cell(entry['break']['brokenTargetLogitDrop'])} | "
                    f"{_drop_cell(entry['break']['ratioToOriginal'])} | "
                    f"{'yes' if entry.get('rescueSought') else 'no'} | {shift_cell} |"
                )
            lines.append("")
        for entry in why.get("items", []):
            lines.append(f"**{entry.get('family')}** - {entry.get('prompt')}")
            lines.append("")
            lines.append(str(entry.get("whyStatement")))
            lines.append("")
            path_change = entry["replacementPath"]["pathChange"]
            lines.append("| field | source circuit | replacement path |")
            lines.append("|---|---|---|")
            lines.append(
                f"| layers | {path_change['sourceLayers']} | {path_change['replacementLayers']} |"
            )
            lines.append(
                f"| MLP components | {path_change['sourceMlpComponents']} | {path_change['replacementMlpComponents']} |"
            )
            lines.append(
                f"| attention components | {path_change['sourceAttentionComponents']} | {path_change['replacementAttentionComponents']} |"
            )
            lines.append(
                f"| ablate positions | {path_change['sourceAblatePositions']} | {path_change['replacementAblatePositions']} |"
            )
            lines.append(
                f"| target-logit drop | {_drop_cell(entry['break']['originalTargetLogitDrop'])} | "
                f"{_drop_cell(entry['replacementPath']['targetLogitDrop'])} |"
            )
            lines.append("")
    return lines
