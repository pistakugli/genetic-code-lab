"""Family-level break atlas for recorded ordered-path stability/OOD artifacts.

The per-case why-report explains one break at a time. This module aggregates
every recorded paraphrase measurement across artifacts into a family-level
break taxonomy: how many measured replays broke per prompt family, which
break kinds they were, where the effect went when a rescue search was run,
and how much of the broken population was actually rescue-searched.

Like the why-report layer, the atlas is deterministic post-processing of
numbers already recorded by ``tools/llm-reader-ordered-path-stability.py``.
It performs no model queries and invents no values. ``verify_break_atlas``
recomputes every aggregate from the raw paraphrase/rescue fields, so a
fabricated or stale atlas fails verification.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

from llm_reader.why_report import compare_paths


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _expected_break_kind(drop: float, min_original_drop: float) -> str:
    if drop < 0:
        return "sign-flip"
    if drop < min_original_drop:
        return "magnitude-collapse"
    return "relative-falloff"


def collect_break_records(report: Dict[str, Any], source_label: str) -> List[Dict[str, Any]]:
    """Collect one row per measured paraphrase, with recomputed break kind.

    Rows are derived only from raw artifact fields (paraphrase pass flags and
    drops, original replay drop, config gates, rescue item joins by exact
    prompt). Why-report blocks are deliberately not read here so the atlas
    verifier stays independent of the layer it verifies.
    """

    config = report.get("config", {}) or {}
    min_original_drop = _number(config.get("minOriginalDrop")) or 0.0
    rows: List[Dict[str, Any]] = []
    for case in report.get("cases", []) or []:
        if not isinstance(case, dict):
            continue
        original_drop = _number((case.get("originalReplay", {}) or {}).get("targetLogitDrop"))
        rescue_by_prompt: Dict[str, Dict[str, Any]] = {}
        rescue = case.get("paraphraseRescue")
        if isinstance(rescue, dict):
            for rescue_item in rescue.get("items", []) or []:
                if isinstance(rescue_item, dict):
                    rescue_by_prompt[str(rescue_item.get("prompt"))] = rescue_item
        for paraphrase in case.get("paraphrases", []) or []:
            if not isinstance(paraphrase, dict):
                continue
            drop = _number(paraphrase.get("targetLogitDrop"))
            passed = paraphrase.get("pass") is True
            row: Dict[str, Any] = {
                "source": source_label,
                "subject": case.get("subject"),
                "family": paraphrase.get("family"),
                "prompt": paraphrase.get("prompt"),
                "pass": passed,
                "targetLogitDrop": drop,
                "ratioToOriginal": _number(paraphrase.get("ratioToOriginal")),
            }
            if not passed and drop is not None and original_drop is not None:
                row["breakKind"] = _expected_break_kind(drop, min_original_drop)
            else:
                row["breakKind"] = None
            rescue_item = rescue_by_prompt.get(str(paraphrase.get("prompt")))
            row["rescueSought"] = rescue_item is not None
            if rescue_item is not None:
                row["layerShiftRecovered"] = bool(rescue_item.get("found"))
                replacement = rescue_item.get("replacementPathSearch")
                row["replacementSought"] = isinstance(replacement, dict)
                if isinstance(replacement, dict) and replacement.get("found"):
                    row["replacementFound"] = True
                    row["replacementChangeKind"] = compare_paths(
                        source_nodes=case.get("sourceCircuit", {}) or {},
                        replacement=replacement,
                    )["changeKind"]
                else:
                    row["replacementFound"] = False
                    row["replacementChangeKind"] = None
            else:
                row["layerShiftRecovered"] = None
                row["replacementSought"] = False
                row["replacementFound"] = False
                row["replacementChangeKind"] = None
            rows.append(row)
    return rows


def _empty_bucket() -> Dict[str, Any]:
    return {
        "measured": 0,
        "passed": 0,
        "broken": 0,
        "breakKinds": {},
        "meanBrokenRatioToOriginal": None,
        "rescueSought": 0,
        "layerShiftRecovered": 0,
        "replacementSought": 0,
        "replacementFound": 0,
        "replacementChangeKinds": {},
    }


def build_break_atlas(sources: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate break records from labeled stability/OOD artifacts.

    Each source is ``{"label": str, "path": str, "report": dict}``. The atlas
    keeps per-source and per-family aggregates plus overall totals and rescue
    coverage, so the artifact states both what broke and how much of the
    broken population was ever rescue-searched.
    """

    families: Dict[str, Dict[str, Any]] = {}
    source_summaries: List[Dict[str, Any]] = []
    totals = _empty_bucket()
    for source in sources:
        rows = collect_break_records(source["report"], source["label"])
        source_bucket = _empty_bucket()
        source_broken_ratios: List[float] = []
        for row in rows:
            family = str(row["family"])
            bucket = families.setdefault(family, _empty_bucket())
            for target in (bucket, source_bucket, totals):
                target["measured"] += 1
            if row["pass"]:
                for target in (bucket, source_bucket, totals):
                    target["passed"] += 1
            else:
                for target in (bucket, source_bucket, totals):
                    target["broken"] += 1
                kind = row["breakKind"]
                if kind is not None:
                    for target in (bucket, source_bucket, totals):
                        target["breakKinds"][kind] = target["breakKinds"].get(kind, 0) + 1
                if row["ratioToOriginal"] is not None:
                    source_broken_ratios.append(row["ratioToOriginal"])
                if row["rescueSought"]:
                    for target in (bucket, source_bucket, totals):
                        target["rescueSought"] += 1
                    if row["layerShiftRecovered"]:
                        for target in (bucket, source_bucket, totals):
                            target["layerShiftRecovered"] += 1
                if row["replacementSought"]:
                    for target in (bucket, source_bucket, totals):
                        target["replacementSought"] += 1
                    if row["replacementFound"]:
                        for target in (bucket, source_bucket, totals):
                            target["replacementFound"] += 1
                        change_kind = row["replacementChangeKind"]
                        if change_kind is not None:
                            for target in (bucket, source_bucket, totals):
                                target["replacementChangeKinds"][change_kind] = (
                                    target["replacementChangeKinds"].get(change_kind, 0) + 1
                                )
        source_ratio = (
            sum(source_broken_ratios) / len(source_broken_ratios) if source_broken_ratios else None
        )
        report = source["report"]
        source_summaries.append({
            "label": source["label"],
            "path": source.get("path"),
            "profile": (report.get("config", {}) or {}).get("paraphraseProfile"),
            "modelId": (report.get("model", {}) or {}).get("id"),
            "adapter": (report.get("model", {}) or {}).get("adapter"),
            "measured": source_bucket["measured"],
            "passed": source_bucket["passed"],
            "broken": source_bucket["broken"],
            "breakKinds": source_bucket["breakKinds"],
            "meanBrokenRatioToOriginal": source_ratio,
        })
    # attach family-level mean ratios over all sources
    family_ratios: Dict[str, List[float]] = {}
    for source in sources:
        for row in collect_break_records(source["report"], source["label"]):
            if not row["pass"] and row["ratioToOriginal"] is not None:
                family_ratios.setdefault(str(row["family"]), []).append(row["ratioToOriginal"])
    for family, ratios in family_ratios.items():
        if family in families:
            families[family]["meanBrokenRatioToOriginal"] = sum(ratios) / len(ratios)
    return {
        "kind": "ordered-path-ood-break-atlas",
        "sources": source_summaries,
        "families": families,
        "totals": totals,
        "rescueCoverage": {
            "brokenTotal": totals["broken"],
            "brokenWithRescueSearch": totals["rescueSought"],
        },
        "claimBoundary": (
            "This atlas aggregates recorded paraphrase replays from the listed "
            "artifacts only. It is a population-level restatement of measured "
            "target-logit drops and rescue searches, not a semantic claim about "
            "prompt families, not proof of circuit uniqueness, and not evidence "
            "about subjects, families, or models that were not measured."
        ),
    }


def _same_number(left: Any, right: Any) -> bool:
    left_number = _number(left)
    right_number = _number(right)
    if left_number is None or right_number is None:
        return left_number is None and right_number is None
    return math.isclose(left_number, right_number, rel_tol=1e-9, abs_tol=1e-9)


def _same_counts(left: Any, right: Dict[str, int]) -> bool:
    return isinstance(left, dict) and left == right


def verify_break_atlas(atlas: Dict[str, Any], sources: List[Dict[str, Any]]) -> List[str]:
    """Recompute the atlas from raw artifact fields and compare.

    Every count, break-kind tally, mean ratio, rescue outcome, and replacement
    change kind is recomputed here from the paraphrase records and rescue
    items of the source reports, so a fabricated or edited atlas fails.
    """

    errors: List[str] = []
    if not isinstance(atlas, dict) or atlas.get("kind") != "ordered-path-ood-break-atlas":
        return ["atlas kind is not the ordered-path break atlas"]
    recomputed = build_break_atlas(sources)
    for source_index, (recorded_source, expected_source) in enumerate(
        zip(atlas.get("sources", []) or [], recomputed["sources"])
    ):
        prefix = f"atlas source {source_index}: "
        if not isinstance(recorded_source, dict):
            errors.append(prefix + "source summary is malformed")
            continue
        if recorded_source.get("label") != expected_source["label"]:
            errors.append(prefix + "source label does not match the provided artifacts")
        if recorded_source.get("path") != expected_source["path"]:
            errors.append(prefix + "source path does not match the provided artifacts")
        for key in ("measured", "passed", "broken"):
            if recorded_source.get(key) != expected_source[key]:
                errors.append(prefix + f"{key} count differs from the raw measurements")
        if not _same_counts(recorded_source.get("breakKinds"), expected_source["breakKinds"]):
            errors.append(prefix + "break-kind tally differs from the raw measurements")
        if not _same_number(
            recorded_source.get("meanBrokenRatioToOriginal"),
            expected_source["meanBrokenRatioToOriginal"],
        ):
            errors.append(prefix + "mean broken ratio differs from the raw measurements")
    if len(atlas.get("sources", []) or []) != len(recomputed["sources"]):
        errors.append("atlas source count differs from the provided artifacts")
    recorded_families = atlas.get("families")
    if not isinstance(recorded_families, dict):
        return errors + ["atlas family table is missing"]
    expected_families = recomputed["families"]
    for family in sorted(set(recorded_families) | set(expected_families)):
        recorded = recorded_families.get(family)
        expected = expected_families.get(family)
        prefix = f"atlas family {family}: "
        if not isinstance(recorded, dict):
            errors.append(prefix + "family bucket is missing")
            continue
        for key in (
            "measured",
            "passed",
            "broken",
            "rescueSought",
            "layerShiftRecovered",
            "replacementSought",
            "replacementFound",
        ):
            if recorded.get(key) != expected[key]:
                errors.append(prefix + f"{key} count differs from the raw measurements")
        if not _same_counts(recorded.get("breakKinds"), expected["breakKinds"]):
            errors.append(prefix + "break-kind tally differs from the raw measurements")
        if not _same_counts(
            recorded.get("replacementChangeKinds"), expected["replacementChangeKinds"]
        ):
            errors.append(prefix + "replacement change-kind tally differs from the raw measurements")
        if not _same_number(
            recorded.get("meanBrokenRatioToOriginal"),
            expected["meanBrokenRatioToOriginal"],
        ):
            errors.append(prefix + "mean broken ratio differs from the raw measurements")
    recorded_totals = atlas.get("totals")
    if not isinstance(recorded_totals, dict):
        errors.append("atlas totals are missing")
    else:
        for key in (
            "measured",
            "passed",
            "broken",
            "rescueSought",
            "layerShiftRecovered",
            "replacementSought",
            "replacementFound",
        ):
            if recorded_totals.get(key) != recomputed["totals"][key]:
                errors.append(f"atlas totals: {key} differs from the raw measurements")
        if not _same_counts(recorded_totals.get("breakKinds"), recomputed["totals"]["breakKinds"]):
            errors.append("atlas totals: break-kind tally differs from the raw measurements")
    coverage = atlas.get("rescueCoverage")
    if not isinstance(coverage, dict):
        errors.append("atlas rescue coverage is missing")
    else:
        if coverage.get("brokenTotal") != recomputed["rescueCoverage"]["brokenTotal"]:
            errors.append("atlas rescue coverage: broken total differs from the raw measurements")
        if coverage.get("brokenWithRescueSearch") != recomputed["rescueCoverage"]["brokenWithRescueSearch"]:
            errors.append("atlas rescue coverage: searched count differs from the raw measurements")
    return errors


def _fmt(value: Any, digits: int = 4) -> str:
    number = _number(value)
    return "n/a" if number is None else f"{number:.{digits}f}"


def _counts_cell(counts: Dict[str, int]) -> str:
    if not counts:
        return "none"
    return ", ".join(f"{kind} {count}" for kind, count in sorted(counts.items()))


def render_atlas_markdown(atlas: Dict[str, Any]) -> List[str]:
    """Render the break-atlas markdown section."""

    lines: List[str] = []
    lines.append("# Ordered-Path OOD Break Atlas")
    lines.append("")
    lines.append(
        "Population-level restatement of every recorded paraphrase replay in the "
        "listed artifacts: what broke, per prompt family, and where the measured "
        "effect went when a rescue search was run. All values are recomputed from "
        "the raw artifact fields; this is not a semantic claim about families."
    )
    lines.append("")
    lines.append("## Sources")
    lines.append("")
    lines.append("| source | profile | model | measured | passed | broken | mean broken ratio |")
    lines.append("|---|---|---|---:|---:|---:|---:|")
    for source in atlas.get("sources", []):
        lines.append(
            f"| {source.get('label')} | {source.get('profile')} | {source.get('modelId')} | "
            f"{source.get('measured')} | {source.get('passed')} | {source.get('broken')} | "
            f"{_fmt(source.get('meanBrokenRatioToOriginal'))} |"
        )
    lines.append("")
    lines.append("## Break taxonomy by prompt family")
    lines.append("")
    lines.append(
        "| family | measured | passed | broken | break kinds | mean broken ratio | "
        "rescue searched | layer-shift recovered | replacement found |"
    )
    lines.append("|---|---:|---:|---:|---|---:|---:|---:|---:|")
    for family in sorted(atlas.get("families", {})):
        bucket = atlas["families"][family]
        lines.append(
            f"| {family} | {bucket['measured']} | {bucket['passed']} | {bucket['broken']} | "
            f"{_counts_cell(bucket['breakKinds'])} | {_fmt(bucket['meanBrokenRatioToOriginal'])} | "
            f"{bucket['rescueSought']} | {bucket['layerShiftRecovered']} | {bucket['replacementFound']} |"
        )
    totals = atlas.get("totals", {})
    lines.append("")
    lines.append("## Totals")
    lines.append("")
    lines.append(
        f"- measured paraphrase replays: {totals.get('measured')}; "
        f"passed {totals.get('passed')}; broken {totals.get('broken')}"
    )
    lines.append(f"- break kinds: {_counts_cell(totals.get('breakKinds', {}))}")
    coverage = atlas.get("rescueCoverage", {})
    lines.append(
        f"- rescue coverage: {coverage.get('brokenWithRescueSearch')} of "
        f"{coverage.get('brokenTotal')} broken paraphrases were rescue-searched in the "
        "recorded runs; the atlas says nothing about the unsearched remainder beyond "
        "their break classification"
    )
    lines.append(
        f"- layer-shift recoveries: {totals.get('layerShiftRecovered')}; "
        f"replacement paths found: {totals.get('replacementFound')} "
        f"({_counts_cell(totals.get('replacementChangeKinds', {}))})"
    )
    lines.append("")
    lines.append(str(atlas.get("claimBoundary")))
    lines.append("")
    return lines
