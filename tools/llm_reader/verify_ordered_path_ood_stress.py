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
    if model.get("device") != "cuda":
        errors.append(prefix + "pilot must record CUDA as the model device")

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
