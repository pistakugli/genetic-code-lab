#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List


EXPECTED_FAMILIES = {
    "service-question",
    "possessive-question",
    "imperative",
    "embedded-question",
    "sentence-completion",
    "context-prefixed-question",
}


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _close(value: Any, expected: float) -> bool:
    return _is_number(value) and math.isclose(float(value), expected, rel_tol=1e-9, abs_tol=1e-9)


def _why_component_ids(interventions: Any, id_field: str) -> set:
    ids = set()
    for intervention in interventions or []:
        if not isinstance(intervention, dict):
            continue
        layer = intervention.get("layer")
        component = intervention.get(id_field)
        if isinstance(layer, int) and isinstance(component, int):
            ids.add((layer, component))
    return ids


def _why_positions(interventions: Any) -> set:
    positions = set()
    for intervention in interventions or []:
        if not isinstance(intervention, dict):
            continue
        for position in intervention.get("ablateTokenPositions", []) or []:
            if isinstance(position, int):
                positions.add(position)
    return positions


def _expected_path_change(source_nodes: Any, replacement: Any) -> Dict[str, Any] | None:
    if not isinstance(source_nodes, dict) or not isinstance(replacement, dict):
        return None
    source_mlp = _why_component_ids(source_nodes.get("mlp", []), "neuron")
    source_attention = _why_component_ids(source_nodes.get("attention", []), "head")
    replacement_mlp = _why_component_ids(replacement.get("mlpInterventions", []), "neuron")
    replacement_attention = _why_component_ids(replacement.get("attentionInterventions", []), "head")
    source_layers = {layer for layer, _ in source_mlp | source_attention}
    replacement_layers = {layer for layer, _ in replacement_mlp | replacement_attention}
    source_positions = _why_positions(source_nodes.get("mlp", [])) | _why_positions(
        source_nodes.get("attention", [])
    )
    replacement_positions = _why_positions(replacement.get("mlpInterventions", [])) | _why_positions(
        replacement.get("attentionInterventions", [])
    )
    overlap = len(source_mlp & replacement_mlp) + len(source_attention & replacement_attention)
    shared_layers = sorted(source_layers & replacement_layers)
    if overlap and shared_layers:
        change_kind = "component-retention"
    elif overlap:
        change_kind = "position-rebinding"
    elif source_layers and replacement_layers and abs(
        sum(replacement_layers) / len(replacement_layers) - sum(source_layers) / len(source_layers)
    ) < 1e-12:
        change_kind = "layer-local-rebinding"
    else:
        change_kind = "component-rebinding"
    if source_layers and replacement_layers:
        delta = (
            sum(replacement_layers) / len(replacement_layers)
            - sum(source_layers) / len(source_layers)
        )
        direction = "earlier" if delta < 0 else "later" if delta > 0 else "same"
    else:
        direction = "same"
    return {
        "sourceLayers": sorted(source_layers),
        "replacementLayers": sorted(replacement_layers),
        "layerOverlap": shared_layers,
        "sourceAblatePositions": sorted(source_positions),
        "replacementAblatePositions": sorted(replacement_positions),
        "componentOverlapCount": overlap,
        "changeKind": change_kind,
        "direction": direction,
    }


def _verify_case_why(case: Dict[str, Any], case_prefix: str, config: Dict[str, Any]) -> List[str]:
    """Check the why-report blocks against the raw rescue measurements.

    Every classification, overlap count, direction, and transfer diagnosis is
    recomputed here from the raw artifact fields, so a fabricated or stale
    why-report entry fails verification. Artifacts recorded before the
    why-report layer (no ``whyReportEnabled`` marker) keep verifying under the
    previous rules.
    """

    errors: List[str] = []
    if config.get("whyReportEnabled") is not True:
        return errors
    rescue = case.get("paraphraseRescue", {})
    rescue_items = rescue.get("items", []) if isinstance(rescue, dict) else []
    paraphrases = [
        paraphrase
        for paraphrase in case.get("paraphrases", []) or []
        if isinstance(paraphrase, dict)
    ]
    broken_paraphrases = [paraphrase for paraphrase in paraphrases if paraphrase.get("pass") is False]
    why = case.get("whyReport")
    if not isinstance(why, dict):
        if rescue_items:
            errors.append(case_prefix + "why-report is missing for a case with rescue measurements")
        elif broken_paraphrases:
            errors.append(case_prefix + "why-report is missing for a case with broken paraphrases")
        return errors
    if not rescue_items and not broken_paraphrases:
        errors.append(case_prefix + "why-report exists without rescue measurements or broken paraphrases")
        return errors
    if why.get("subject") != case.get("subject"):
        errors.append(case_prefix + "why-report subject does not match the case")
    why_items = why.get("items")
    if not isinstance(why_items, list) or len(why_items) != len(rescue_items):
        errors.append(case_prefix + "why-report does not cover every rescue measurement")
        return errors
    original_drop = case.get("originalReplay", {}).get("targetLogitDrop")
    min_original_drop = config.get("minOriginalDrop")
    rescue_by_prompt = {
        str(rescue_item.get("prompt")): rescue_item
        for rescue_item in rescue_items
        if isinstance(rescue_item, dict)
    }
    paraphrase_breaks = why.get("paraphraseBreaks")
    if broken_paraphrases:
        if not isinstance(paraphrase_breaks, list) or len(paraphrase_breaks) != len(broken_paraphrases):
            errors.append(case_prefix + "why-report does not classify every broken paraphrase")
        else:
            breaks_by_prompt = {
                str(entry.get("prompt")): entry
                for entry in paraphrase_breaks
                if isinstance(entry, dict)
            }
            for paraphrase in broken_paraphrases:
                entry = breaks_by_prompt.get(str(paraphrase.get("prompt")))
                break_prefix = f"{case_prefix}paraphrase break {paraphrase.get('family')}: "
                if entry is None:
                    errors.append(break_prefix + "classification is missing")
                    continue
                drop = paraphrase.get("targetLogitDrop")
                if _is_number(drop) and _is_number(min_original_drop):
                    expected_kind = (
                        "sign-flip"
                        if drop < 0
                        else "magnitude-collapse"
                        if drop < min_original_drop
                        else "relative-falloff"
                    )
                    break_block = entry.get("break")
                    if not isinstance(break_block, dict) or break_block.get("kind") != expected_kind:
                        errors.append(break_prefix + "break kind disagrees with the recorded drop")
                    elif not _close(break_block.get("brokenTargetLogitDrop"), float(drop)):
                        errors.append(break_prefix + "broken drop differs from the raw paraphrase measurement")
                rescue_item = rescue_by_prompt.get(str(paraphrase.get("prompt")))
                if entry.get("rescueSought") is not (rescue_item is not None):
                    errors.append(break_prefix + "rescue-sought flag disagrees with the recorded rescue searches")
                elif rescue_item is not None:
                    shift_block = entry.get("layerShiftRescue")
                    if not isinstance(shift_block, dict):
                        errors.append(break_prefix + "layer-shift outcome is missing")
                    else:
                        if shift_block.get("recovered") is not bool(rescue_item.get("found")):
                            errors.append(break_prefix + "layer-shift recovery flag disagrees with the rescue measurement")
                        if shift_block.get("bestLayerShift") != rescue_item.get("bestLayerShift"):
                            errors.append(break_prefix + "best layer shift differs from the rescue measurement")
                if entry.get("replacementSought") is not (
                    rescue_item is not None and "replacementPathSearch" in rescue_item
                ):
                    errors.append(break_prefix + "replacement-sought flag disagrees with the recorded path searches")
            expected_break_kinds_all: Dict[str, int] = {}
            for paraphrase in broken_paraphrases:
                drop = paraphrase.get("targetLogitDrop")
                if _is_number(drop) and _is_number(min_original_drop):
                    kind = (
                        "sign-flip"
                        if drop < 0
                        else "magnitude-collapse"
                        if drop < min_original_drop
                        else "relative-falloff"
                    )
                    expected_break_kinds_all[kind] = expected_break_kinds_all.get(kind, 0) + 1
            if why.get("paraphraseBreakKinds") != expected_break_kinds_all:
                errors.append(case_prefix + "why-report paraphrase break kinds do not match raw measurements")
            if why.get("paraphraseBreakCount") != len(broken_paraphrases):
                errors.append(case_prefix + "why-report paraphrase break count does not match raw measurements")
            expected_sought = sum(
                1
                for paraphrase in broken_paraphrases
                if str(paraphrase.get("prompt")) in rescue_by_prompt
            )
            if why.get("rescueSoughtCount") != expected_sought:
                errors.append(case_prefix + "why-report rescue-sought count does not match raw measurements")
    elif paraphrase_breaks:
        errors.append(case_prefix + "paraphrase break classifications exist without broken paraphrases")
    for why_index, (rescue_item, why_item) in enumerate(zip(rescue_items, why_items)):
        if not isinstance(rescue_item, dict) or not isinstance(why_item, dict):
            errors.append(case_prefix + f"why entry {why_index} is malformed")
            continue
        why_prefix = f"{case_prefix}why entry {why_index}: "
        broken_drop = rescue_item.get("brokenDrop")
        replacement = rescue_item.get("replacementPathSearch", {})
        break_block = why_item.get("break")
        if _is_number(broken_drop) and _is_number(min_original_drop):
            expected_kind = (
                "sign-flip"
                if broken_drop < 0
                else "magnitude-collapse"
                if broken_drop < min_original_drop
                else "relative-falloff"
            )
            if not isinstance(break_block, dict):
                errors.append(why_prefix + "break classification is missing")
            else:
                if break_block.get("kind") != expected_kind:
                    errors.append(why_prefix + "break kind disagrees with the recorded drops")
                if not _close(break_block.get("brokenTargetLogitDrop"), float(broken_drop)):
                    errors.append(why_prefix + "broken drop differs from the raw rescue measurement")
                if _is_number(original_drop) and not _close(
                    break_block.get("originalTargetLogitDrop"), float(original_drop)
                ):
                    errors.append(why_prefix + "original drop differs from the raw replay measurement")
        shift_block = why_item.get("layerShiftRescue")
        if not isinstance(shift_block, dict):
            errors.append(why_prefix + "layer-shift rescue block is missing")
        else:
            if shift_block.get("recovered") is not bool(rescue_item.get("found")):
                errors.append(why_prefix + "layer-shift recovery flag disagrees with the rescue measurement")
            if shift_block.get("bestLayerShift") != rescue_item.get("bestLayerShift"):
                errors.append(why_prefix + "best layer shift differs from the rescue measurement")
        replacement_block = why_item.get("replacementPath")
        if not isinstance(replacement_block, dict):
            errors.append(why_prefix + "replacement comparison is missing")
        else:
            if replacement_block.get("found") is not bool(replacement.get("found")):
                errors.append(why_prefix + "replacement found flag disagrees with the path-search result")
            if replacement_block.get("attempted") is not ("replacementPathSearch" in rescue_item):
                errors.append(why_prefix + "replacement attempted flag disagrees with the recorded run")
            if _is_number(replacement.get("targetLogitDrop")) and not _close(
                replacement_block.get("targetLogitDrop"), float(replacement["targetLogitDrop"])
            ):
                errors.append(why_prefix + "replacement drop differs from the raw path-search result")
            expected_change = _expected_path_change(case.get("sourceCircuit", {}), replacement)
            path_change = replacement_block.get("pathChange")
            if not isinstance(path_change, dict):
                errors.append(why_prefix + "path comparison is missing")
            elif expected_change is not None:
                for key in (
                    "sourceLayers",
                    "replacementLayers",
                    "layerOverlap",
                    "sourceAblatePositions",
                    "replacementAblatePositions",
                ):
                    if path_change.get(key) != expected_change[key]:
                        errors.append(why_prefix + f"{key} does not match the recorded interventions")
                if path_change.get("componentOverlapCount") != expected_change["componentOverlapCount"]:
                    errors.append(why_prefix + "component overlap count does not match the recorded interventions")
                if path_change.get("changeKind") != expected_change["changeKind"]:
                    errors.append(why_prefix + "change kind disagrees with the recorded interventions")
                if path_change.get("direction") != expected_change["direction"]:
                    errors.append(why_prefix + "movement direction disagrees with the recorded interventions")
        transfer_block = why_item.get("transfer")
        transfer = replacement.get("crossFamilyTransfer", {}) if isinstance(replacement, dict) else {}
        if isinstance(transfer, dict) and transfer:
            if not isinstance(transfer_block, dict):
                errors.append(why_prefix + "transfer diagnosis is missing")
            else:
                raw_failing = sorted(
                    str(item.get("family"))
                    for item in transfer.get("items", [])
                    if isinstance(item, dict) and item.get("pass") is False
                )
                why_failing = sorted(
                    str(item.get("family"))
                    for item in transfer_block.get("failingFamilies", [])
                    if isinstance(item, dict)
                )
                if why_failing != raw_failing:
                    errors.append(why_prefix + "failing family diagnosis differs from raw transfer measurements")
                raw_transfer_summary = transfer.get("summary", {})
                if isinstance(raw_transfer_summary, dict) and (
                    transfer_block.get("attempted") != raw_transfer_summary.get("attempted")
                    or transfer_block.get("passed") != raw_transfer_summary.get("passed")
                ):
                    errors.append(why_prefix + "transfer counts differ from the raw transfer summary")
                if transfer_block.get("status") != transfer.get("status"):
                    errors.append(why_prefix + "transfer status differs from the raw transfer report")
        if not isinstance(why_item.get("whyStatement"), str) or not why_item.get("whyStatement", "").strip():
            errors.append(why_prefix + "why statement is missing")
        if not isinstance(why_item.get("claimBoundary"), str) or not why_item.get("claimBoundary", "").strip():
            errors.append(why_prefix + "claim boundary is missing")
    if why.get("rescueCount") != len(rescue_items):
        errors.append(case_prefix + "why-report rescue count does not match raw measurements")
    expected_replacements = sum(
        1
        for rescue_item in rescue_items
        if isinstance(rescue_item, dict)
        and bool(rescue_item.get("replacementPathSearch", {}).get("found"))
    )
    if why.get("replacementFoundCount") != expected_replacements:
        errors.append(case_prefix + "why-report replacement count does not match raw measurements")
    expected_shift_recoveries = sum(
        1 for rescue_item in rescue_items if isinstance(rescue_item, dict) and bool(rescue_item.get("found"))
    )
    if why.get("layerShiftOnlyRecoveries") != expected_shift_recoveries:
        errors.append(case_prefix + "why-report layer-shift recovery count does not match raw measurements")
    expected_break_kinds: Dict[str, int] = {}
    for rescue_item in rescue_items:
        if (
            isinstance(rescue_item, dict)
            and _is_number(rescue_item.get("brokenDrop"))
            and _is_number(min_original_drop)
        ):
            kind = (
                "sign-flip"
                if rescue_item["brokenDrop"] < 0
                else "magnitude-collapse"
                if rescue_item["brokenDrop"] < min_original_drop
                else "relative-falloff"
            )
            expected_break_kinds[kind] = expected_break_kinds.get(kind, 0) + 1
    if why.get("breakKinds") != expected_break_kinds:
        errors.append(case_prefix + "why-report break-kind aggregate does not match raw measurements")
    return errors


def verify_case_report(report: Dict[str, Any], expected_model_id: str, expected_adapter: str) -> List[str]:
    errors: List[str] = []
    prefix = f"{expected_model_id} ({expected_adapter}): "
    if report.get("kind") != f"{expected_adapter}-live-ordered-path-ood-stress":
        errors.append(prefix + "case report kind is not the OOD stress lane")
    if report.get("synthetic") is not False or report.get("surrogate") is not False:
        errors.append(prefix + "report must identify a real, non-surrogate model run")

    model = report.get("model")
    if not isinstance(model, dict):
        errors.append(prefix + "model identity is missing")
        model = {}
    if model.get("id") != expected_model_id:
        errors.append(prefix + "model identity does not match the selected suite case")
    if model.get("adapter") != expected_adapter:
        errors.append(prefix + "adapter identity does not match the selected suite case")
    if model.get("device") not in ("cuda", "cpu"):
        errors.append(prefix + "pilot must record a supported live device (cuda or cpu)")

    config = report.get("config")
    summary = report.get("summary")
    if not isinstance(config, dict):
        errors.append(prefix + "run configuration is missing")
        config = {}
    if not isinstance(summary, dict):
        errors.append(prefix + "run summary is missing")
        summary = {}
    if config.get("paraphraseProfile") != "ood" or summary.get("paraphraseProfile") != "ood":
        errors.append(prefix + "OOD profile is not recorded consistently")
    cases = report.get("cases")
    if not isinstance(cases, list) or not cases:
        errors.append(prefix + "report has no measured cases")
        return errors
    if any(not isinstance(case, dict) for case in cases):
        errors.append(prefix + "case measurements are malformed")
        return errors
    if len(cases) != config.get("caseCount"):
        errors.append(prefix + "measured case count does not match the configured count")
    if summary.get("caseCount") != len(cases):
        errors.append(prefix + "summary case count does not match the measured cases")
    if not _is_number(summary.get("originalReplayPassRate")):
        errors.append(prefix + "original causal replay rate is missing")
    if not _is_number(summary.get("paraphrasePassRate")):
        errors.append(prefix + "paraphrase causal replay rate is missing")
    if not _is_number(summary.get("layerSpecificityRate")):
        errors.append(prefix + "neighboring-layer control rate is missing")
    if not _is_number(summary.get("maxReplayError")):
        errors.append(prefix + "maximum replay error is missing")

    seen_families = set()
    total_prompts = 0
    passed_prompts = 0
    original_passes = []
    layer_passes = []
    replay_errors = []
    replacement_searches = []
    replacement_transfer_records = []
    for case_index, case in enumerate(cases):
        case_prefix = f"{prefix}case {case_index}: "
        original = case.get("originalReplay")
        if not isinstance(original, dict):
            errors.append(case_prefix + "original causal replay measurements are missing")
            original = {}
        if not _is_number(original.get("targetLogitDrop")) or not _is_number(original.get("absoluteReplayError")):
            errors.append(case_prefix + "original causal replay measurements are missing")
        else:
            replay_errors.append(float(original["absoluteReplayError"]))
        if isinstance(original.get("pass"), bool):
            original_passes.append(original["pass"])
        else:
            errors.append(case_prefix + "original replay pass/fail measurement is missing")
        controls = case.get("layerControls")
        if not isinstance(controls, dict):
            errors.append(case_prefix + "neighboring-layer control measurements are missing")
            controls = {}
        if not isinstance(controls.get("items"), list) or not controls["items"]:
            errors.append(case_prefix + "neighboring-layer control measurements are missing")
        if isinstance(controls.get("pass"), bool):
            layer_passes.append(controls["pass"])
        else:
            errors.append(case_prefix + "neighboring-layer pass/fail measurement is missing")

        paraphrases = case.get("paraphrases")
        rescue = case.get("paraphraseRescue", {})
        rescue_items = rescue.get("items", []) if isinstance(rescue, dict) else []
        if not isinstance(rescue_items, list):
            errors.append(case_prefix + "paraphrase rescue measurements are malformed")
            rescue_items = []
        for rescue_index, rescue_item in enumerate(rescue_items):
            if not isinstance(rescue_item, dict):
                errors.append(case_prefix + "paraphrase rescue item is malformed")
                continue
            replacement = rescue_item.get("replacementPathSearch")
            if replacement is None:
                if config.get("replacementPathSearchEnabled") is True:
                    errors.append(case_prefix + "enabled replacement search is missing a rescue result")
                continue
            replacement_prefix = f"{case_prefix}replacement search {rescue_index}: "
            if not isinstance(replacement, dict):
                errors.append(replacement_prefix + "result is malformed")
                continue
            replacement_searches.append(replacement)
            if replacement.get("prompt") != rescue_item.get("prompt"):
                errors.append(replacement_prefix + "searched prompt does not match the failed paraphrase")
            status = replacement.get("status")
            if status not in {"passed", "failed", "skipped"}:
                errors.append(replacement_prefix + "status is invalid")
            if replacement.get("found") is not (status == "passed"):
                errors.append(replacement_prefix + "found flag disagrees with the path gate")
            if not isinstance(replacement.get("searchLayers"), list) or not replacement["searchLayers"]:
                errors.append(replacement_prefix + "searched layer set is missing")
            for key in ("mlpInterventions", "attentionInterventions"):
                if not isinstance(replacement.get(key), list):
                    errors.append(replacement_prefix + f"{key} must be a list")
            if not isinstance(replacement.get("searchScope"), str) or not replacement["searchScope"].strip():
                errors.append(replacement_prefix + "search scope is missing")
            if status == "passed":
                if not replacement.get("mlpInterventions") or not replacement.get("attentionInterventions"):
                    errors.append(replacement_prefix + "passed path has no MLP or attention nodes")
                for key in ("targetLogitDrop", "pathActivationDelta", "pathMarginOverControlP95"):
                    if not _is_number(replacement.get(key)):
                        errors.append(replacement_prefix + f"{key} is missing for a passed path")
            if config.get("replacementPathTransferEnabled") is True:
                replacement_transfer_records.append(
                    (replacement, rescue_item, replacement_prefix)
                )
        if not isinstance(paraphrases, list) or not paraphrases:
            errors.append(case_prefix + "no paraphrase causal replays were measured")
            continue
        if len(paraphrases) != config.get("paraphraseCount"):
            errors.append(case_prefix + "measured paraphrase count does not match configuration")
        valid_case_prompts = []
        for item in paraphrases:
            if not isinstance(item, dict):
                errors.append(case_prefix + "paraphrase measurement is malformed")
                continue
            family = item.get("family")
            prompt = item.get("prompt")
            if item.get("profile") != "ood" or not isinstance(family, str) or family not in EXPECTED_FAMILIES:
                errors.append(case_prefix + "paraphrase has an unknown profile or family")
                continue
            seen_families.add(family)
            answer = str(case.get("answer") or "").strip()
            if not answer:
                errors.append(case_prefix + "answer target is missing")
            if not isinstance(prompt, str) or not prompt.strip() or (answer and answer.casefold() in prompt.casefold()):
                errors.append(case_prefix + f"{family} prompt is missing or leaks the answer")
            if not _is_number(item.get("tokenCount")) or item["tokenCount"] < 1:
                errors.append(case_prefix + f"{family} token count is missing")
            if not _is_number(item.get("targetLogitDrop")) or not _is_number(item.get("ratioToOriginal")):
                errors.append(case_prefix + f"{family} causal replay measurements are missing")
            if not isinstance(item.get("pass"), bool):
                errors.append(case_prefix + f"{family} pass/fail measurement is missing")
            else:
                total_prompts += 1
                passed_prompts += int(item["pass"])
                valid_case_prompts.append(item)
        broken_prompts = {
            item.get("prompt")
            for item in valid_case_prompts
            if item.get("pass") is False
        }
        for rescue_index, rescue_item in enumerate(rescue_items):
            if not isinstance(rescue_item, dict) or "replacementPathSearch" not in rescue_item:
                continue
            if rescue_item.get("prompt") not in broken_prompts:
                errors.append(
                    f"{case_prefix}replacement search {rescue_index}: prompt was not a failed replay"
                )
        for replacement, rescue_item, replacement_prefix in replacement_transfer_records:
            transfer = replacement.get("crossFamilyTransfer")
            if not isinstance(transfer, dict):
                errors.append(replacement_prefix + "held-out family transfer report is missing")
                continue
            source_family = rescue_item.get("family")
            if transfer.get("sourceFamily") != source_family:
                errors.append(replacement_prefix + "held-out transfer source family is inconsistent")
            if replacement.get("status") != "passed":
                if transfer.get("status") != "not_applicable" or transfer.get("items") != []:
                    errors.append(replacement_prefix + "transfer must be not_applicable when no replacement path was found")
                continue

            expected_by_family = {
                item["family"]: item
                for item in valid_case_prompts
                if item["family"] != source_family
            }
            transfer_items = transfer.get("items")
            if not isinstance(transfer_items, list):
                errors.append(replacement_prefix + "held-out transfer measurements are malformed")
                continue
            if not expected_by_family:
                if transfer.get("status") != "not_applicable" or transfer_items:
                    errors.append(replacement_prefix + "transfer must be not_applicable without a held-out family")
                continue
            discovery_drop = replacement.get("targetLogitDrop")
            min_drop = config.get("minOriginalDrop")
            min_ratio = config.get("minParaphraseDropRatio")
            if transfer.get("sourcePrompt") != replacement.get("prompt"):
                errors.append(replacement_prefix + "transfer source prompt differs from the discovered-path prompt")
            if (
                not _is_number(transfer.get("discoveryTargetLogitDrop"))
                or not _close(
                    transfer.get("discoveryTargetLogitDrop"),
                    float(discovery_drop) if _is_number(discovery_drop) else math.nan,
                )
            ):
                errors.append(replacement_prefix + "transfer discovery drop differs from the raw path-search result")
            if not _close(transfer.get("minTargetLogitDrop"), float(min_drop) if _is_number(min_drop) else math.nan):
                errors.append(replacement_prefix + "transfer target-drop threshold differs from configuration")
            if not _close(transfer.get("minDropRatio"), float(min_ratio) if _is_number(min_ratio) else math.nan):
                errors.append(replacement_prefix + "transfer ratio threshold differs from configuration")
            if transfer.get("expectedHeldoutFamilyCount") != len(expected_by_family):
                errors.append(replacement_prefix + "transfer held-out family count is inconsistent")
            observed_families = set()
            transfer_passes = 0
            for item in transfer_items:
                if not isinstance(item, dict):
                    errors.append(replacement_prefix + "held-out transfer item is malformed")
                    continue
                family = item.get("family")
                source_measurement = expected_by_family.get(family)
                if source_measurement is None:
                    errors.append(replacement_prefix + "transfer family is not held out from path selection")
                    continue
                if family in observed_families:
                    errors.append(replacement_prefix + "transfer family is measured more than once")
                observed_families.add(family)
                if item.get("prompt") != source_measurement.get("prompt"):
                    errors.append(replacement_prefix + "transfer prompt does not match the held-out family input")
                if (
                    not _is_number(item.get("tokenCount"))
                    or item["tokenCount"] < 1
                ):
                    errors.append(replacement_prefix + "transfer token count is missing")
                drop = item.get("targetLogitDrop")
                ratio = item.get("ratioToDiscovery")
                if not _is_number(drop) or not _is_number(ratio) or not _is_number(discovery_drop):
                    errors.append(replacement_prefix + "transfer causal measurements are missing")
                    continue
                expected_ratio = float(drop) / max(float(discovery_drop), 1e-9)
                if not _close(ratio, expected_ratio):
                    errors.append(replacement_prefix + "transfer ratio does not match raw target-logit drops")
                expected_item_pass = (
                    _is_number(min_drop)
                    and _is_number(min_ratio)
                    and float(drop) >= float(min_drop)
                    and expected_ratio >= float(min_ratio)
                )
                if item.get("pass") is not expected_item_pass:
                    errors.append(replacement_prefix + "transfer pass/fail disagrees with configured thresholds")
                transfer_passes += int(expected_item_pass)

            expected_families = set(expected_by_family)
            if observed_families != expected_families:
                errors.append(replacement_prefix + "held-out transfer does not cover every other measured OOD family")
            transfer_summary = transfer.get("summary")
            if not isinstance(transfer_summary, dict):
                errors.append(replacement_prefix + "held-out transfer aggregate is missing")
                continue
            expected_attempted = len(expected_families)
            expected_status = (
                "passed"
                if expected_attempted and transfer_passes == expected_attempted
                else "failed"
            )
            if transfer.get("status") != expected_status:
                errors.append(replacement_prefix + "held-out transfer status disagrees with raw measurements")
            if (
                transfer_summary.get("attempted") != expected_attempted
                or transfer_summary.get("passed") != transfer_passes
                or not _close(
                    transfer_summary.get("passRate"),
                    transfer_passes / expected_attempted if expected_attempted else 0.0,
                )
            ):
                errors.append(replacement_prefix + "held-out transfer aggregate does not match raw measurements")
        errors.extend(_verify_case_why(case, case_prefix, config))
        if paraphrases and isinstance(original.get("pass"), bool) and isinstance(controls.get("pass"), bool):
            case_paraphrase_rate = sum(bool(item.get("pass")) for item in valid_case_prompts) / max(1, len(valid_case_prompts))
            min_case_rate = config.get("minParaphrasePassRate", 1.0)
            if not _is_number(min_case_rate):
                errors.append(case_prefix + "configured paraphrase threshold is invalid")
                min_case_rate = 1.0
            expected_case_pass = (
                original["pass"]
                and case_paraphrase_rate >= float(min_case_rate)
                and controls["pass"]
            )
            if case.get("status") != ("passed" if expected_case_pass else "failed"):
                errors.append(case_prefix + "case status disagrees with its measured gates")

    if seen_families != EXPECTED_FAMILIES:
        errors.append(prefix + "not all six required OOD prompt families are represented")
    family_summary = summary.get("paraphraseFamilySummary")
    if not isinstance(family_summary, dict) or set(family_summary) != EXPECTED_FAMILIES:
        errors.append(prefix + "family-level aggregate is missing or incomplete")
    else:
        for family in EXPECTED_FAMILIES:
            rows = [
                item
                for case in cases
                for item in case.get("paraphrases", [])
                if isinstance(item, dict) and item.get("family") == family
            ]
            aggregate = family_summary[family]
            if not isinstance(aggregate, dict):
                errors.append(prefix + f"{family} aggregate is malformed")
                continue
            expected_passed = sum(bool(item.get("pass")) for item in rows)
            if aggregate.get("count") != len(rows) or aggregate.get("passed") != expected_passed:
                errors.append(prefix + f"{family} aggregate does not match raw measurements")
            elif rows and all(
                _is_number(item.get("ratioToOriginal")) and _is_number(item.get("tokenCount"))
                for item in rows
            ):
                expected_ratio = sum(float(item["ratioToOriginal"]) for item in rows) / len(rows)
                expected_tokens = sum(int(item["tokenCount"]) for item in rows) / len(rows)
                if not _close(aggregate.get("meanDropRatio"), expected_ratio):
                    errors.append(prefix + f"{family} mean drop ratio does not match raw measurements")
                if not _close(aggregate.get("meanTokenCount"), expected_tokens):
                    errors.append(prefix + f"{family} mean token count does not match raw measurements")
                if not _close(aggregate.get("passRate"), expected_passed / len(rows)):
                    errors.append(prefix + f"{family} pass rate does not match raw measurements")
    if total_prompts and _is_number(summary.get("paraphrasePassRate")) and not _close(
        summary["paraphrasePassRate"], passed_prompts / total_prompts
    ):
        errors.append(prefix + "overall paraphrase pass rate does not match raw measurements")
    if original_passes and _is_number(summary.get("originalReplayPassRate")) and not _close(
        summary["originalReplayPassRate"], sum(original_passes) / len(original_passes)
    ):
        errors.append(prefix + "original replay rate does not match raw measurements")
    if layer_passes and _is_number(summary.get("layerSpecificityRate")) and not _close(
        summary["layerSpecificityRate"], sum(layer_passes) / len(layer_passes)
    ):
        errors.append(prefix + "neighbor-layer rate does not match raw measurements")
    if replay_errors and _is_number(summary.get("maxReplayError")) and not _close(
        summary["maxReplayError"], max(replay_errors)
    ):
        errors.append(prefix + "maximum replay error does not match raw measurements")
    if "replacementPathSearchEnabled" in config:
        replacement_summary = summary.get("replacementPathSearch")
        if not isinstance(replacement_summary, dict):
            errors.append(prefix + "replacement-path search aggregate is missing")
        else:
            expected_attempted = len(replacement_searches)
            expected_passed = sum(bool(item.get("found")) for item in replacement_searches)
            if replacement_summary.get("attempted") != expected_attempted:
                errors.append(prefix + "replacement-path attempt count does not match raw measurements")
            if replacement_summary.get("passed") != expected_passed:
                errors.append(prefix + "replacement-path pass count does not match raw measurements")
    if "replacementPathTransferEnabled" in config:
        transfer_summary = summary.get("replacementPathTransfer")
        if not isinstance(transfer_summary, dict):
            errors.append(prefix + "replacement-path transfer aggregate is missing")
        else:
            transfer_items = [
                transfer
                for case in cases
                for rescue in case.get("paraphraseRescue", {}).get("items", [])
                for replacement in [rescue.get("replacementPathSearch", {})]
                for transfer in [replacement.get("crossFamilyTransfer", {})]
                if isinstance(transfer, dict)
            ]
            expected_attempted = sum(
                int(item.get("summary", {}).get("attempted", 0))
                for item in transfer_items
                if isinstance(item.get("summary"), dict)
            )
            expected_passed = sum(
                int(item.get("summary", {}).get("passed", 0))
                for item in transfer_items
                if isinstance(item.get("summary"), dict)
            )
            if transfer_summary.get("attempted") != expected_attempted:
                errors.append(prefix + "replacement-path transfer attempt count does not match raw measurements")
            if transfer_summary.get("passed") != expected_passed:
                errors.append(prefix + "replacement-path transfer pass count does not match raw measurements")
    if config.get("whyReportEnabled") is True:
        why_summary = summary.get("whyReport")
        expected_why_cases = [
            case
            for case in cases
            if isinstance(case, dict)
            and isinstance(case.get("paraphraseRescue", {}), dict)
            and case.get("paraphraseRescue", {}).get("items")
        ]
        expected_rescues = sum(len(case["paraphraseRescue"]["items"]) for case in expected_why_cases)
        expected_replacement_found = sum(
            1
            for case in expected_why_cases
            for rescue_item in case["paraphraseRescue"]["items"]
            if isinstance(rescue_item, dict)
            and bool(rescue_item.get("replacementPathSearch", {}).get("found"))
        )
        expected_shift_recoveries = sum(
            1
            for case in expected_why_cases
            for rescue_item in case["paraphraseRescue"]["items"]
            if isinstance(rescue_item, dict) and bool(rescue_item.get("found"))
        )
        expected_break_kinds: Dict[str, int] = {}
        expected_change_kinds: Dict[str, int] = {}
        for case in expected_why_cases:
            for rescue_item in case["paraphraseRescue"]["items"]:
                if not isinstance(rescue_item, dict):
                    continue
                if (
                    _is_number(rescue_item.get("brokenDrop"))
                    and _is_number(config.get("minOriginalDrop"))
                ):
                    kind = (
                        "sign-flip"
                        if rescue_item["brokenDrop"] < 0
                        else "magnitude-collapse"
                        if rescue_item["brokenDrop"] < config["minOriginalDrop"]
                        else "relative-falloff"
                    )
                    expected_break_kinds[kind] = expected_break_kinds.get(kind, 0) + 1
                replacement = rescue_item.get("replacementPathSearch", {})
                if isinstance(replacement, dict) and replacement.get("found") is True:
                    change = _expected_path_change(case.get("sourceCircuit", {}), replacement)
                    if change:
                        expected_change_kinds[change["changeKind"]] = (
                            expected_change_kinds.get(change["changeKind"], 0) + 1
                        )
        if not isinstance(why_summary, dict):
            errors.append(prefix + "why-report aggregate is missing")
        else:
            if why_summary.get("caseCount") != len(expected_why_cases):
                errors.append(prefix + "why-report case count does not match raw measurements")
            if why_summary.get("rescueCount") != expected_rescues:
                errors.append(prefix + "why-report rescue count does not match raw measurements")
            if why_summary.get("replacementFoundCount") != expected_replacement_found:
                errors.append(prefix + "why-report replacement count does not match raw measurements")
            if why_summary.get("layerShiftOnlyRecoveries") != expected_shift_recoveries:
                errors.append(prefix + "why-report layer-shift recovery count does not match raw measurements")
            if why_summary.get("breakKinds") != expected_break_kinds:
                errors.append(prefix + "why-report break-kind aggregate does not match raw measurements")
            if why_summary.get("replacementChangeKinds") != expected_change_kinds:
                errors.append(prefix + "why-report change-kind aggregate does not match raw measurements")
    if (
        len(original_passes) == len(cases)
        and len(layer_passes) == len(cases)
        and total_prompts > 0
        and _is_number(config.get("minParaphrasePassRate"))
        and _is_number(config.get("minLayerSpecificityRate"))
    ):
        expected_summary_pass = (
            all(case.get("status") == "passed" for case in cases)
            and all(original_passes)
            and passed_prompts / total_prompts >= float(config["minParaphrasePassRate"])
            and sum(layer_passes) / len(layer_passes) >= float(config["minLayerSpecificityRate"])
        )
        if summary.get("pass") is not expected_summary_pass:
            errors.append(prefix + "summary pass/fail does not match the recorded gates")

    claim = report.get("claimBoundary", {})
    if not isinstance(claim.get("provesIfPasses"), str) or not claim["provesIfPasses"].strip():
        errors.append(prefix + "positive claim boundary is missing")
    if not isinstance(claim.get("doesNotProve"), str) or not claim["doesNotProve"].strip():
        errors.append(prefix + "limitation claim boundary is missing")
    return errors


def verify_suite(suite: Dict[str, Any], report_dir: Path) -> List[str]:
    errors: List[str] = []
    if suite.get("kind") != "nn-live-ordered-path-ood-stress-suite":
        errors.append("suite report kind is not the OOD stress lane")
    if suite.get("paraphraseProfile") != "ood":
        errors.append("suite does not identify the OOD paraphrase profile")
    if suite.get("synthetic") is not False or suite.get("surrogate") is not False:
        errors.append("suite must identify real-model, non-surrogate evidence")
    results = suite.get("results")
    summary = suite.get("summary", {})
    if not isinstance(results, list) or not results:
        errors.append("suite has no model results")
        return errors
    if any(not isinstance(result, dict) for result in results):
        errors.append("suite model results are malformed")
        return errors
    if summary.get("cases") != len(results):
        errors.append("suite result count does not match its summary")
    observed_passed = sum(result.get("status") == "pass" for result in results)
    if summary.get("passed") != observed_passed:
        errors.append("suite passed count does not match model statuses")
    if summary.get("pass") is not (observed_passed == len(results)):
        errors.append("suite pass/fail does not match model statuses")

    for result in results:
        model_id = result.get("id")
        model_name = result.get("model")
        adapter = result.get("adapter")
        relative_report = result.get("artifactPath")
        if not all(isinstance(value, str) and value for value in (model_id, model_name, adapter, relative_report)):
            errors.append("suite result is missing its model, adapter, or case report path")
            continue
        case_path = (report_dir / relative_report).resolve()
        if not case_path.is_relative_to(report_dir.resolve()):
            errors.append(f"{model_id}: case report path escapes the suite directory")
            continue
        if not case_path.is_file():
            errors.append(f"{model_id}: case report is missing at {relative_report}")
            continue
        try:
            case_report = json.loads(case_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"{model_id}: cannot read case report: {error}")
            continue
        errors.extend(verify_case_report(case_report, model_name, adapter))
        case_pass = case_report.get("summary", {}).get("pass")
        if result.get("status") != ("pass" if case_pass is True else "fail"):
            errors.append(f"{model_id}: suite status disagrees with case report")
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Independently verify a live-model ordered-path OOD stress artifact.")
    default_variant = os.environ.get("GCL_BENCHMARK_VARIANT", "ood-paraphrase-stress-gpt2-5case")
    default_root = Path(os.environ.get("GCL_BENCHMARK_ROOT", "benchmarks/release"))
    if not default_root.is_absolute():
        default_root = Path(__file__).resolve().parents[2] / default_root
    default_report = default_root / "nn-live-ordered-path-stability" / default_variant / "report.json"
    parser.add_argument("--report", type=Path, default=default_report)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_path = args.report.resolve()
    if not report_path.is_file():
        raise SystemExit(f"Missing OOD stress suite report: {report_path}")
    try:
        suite = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SystemExit(f"Invalid OOD stress suite JSON: {error}") from error
    errors = verify_suite(suite, report_path.parent)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    print("ORDERED_PATH_OOD_STRESS_ARTIFACT_VERIFIED")


if __name__ == "__main__":
    main()
