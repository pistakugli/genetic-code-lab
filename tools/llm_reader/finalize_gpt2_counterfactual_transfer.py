#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

from llm_reader.gpt2_counterfactual_transfer import (  # noqa: E402
    BOOTSTRAP_REPLICATES,
    _build_family_analysis,
    _classify_computation,
    write_markdown,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finalize paired uncertainty estimates and per-pair necessity explanations."
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    return parser.parse_args()


def finalize_report(report_path: Path, bootstrap_replicates: int) -> Dict[str, Any]:
    if bootstrap_replicates < 100:
        raise ValueError("At least 100 bootstrap replicates are required.")
    report_path = report_path.resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("kind") != "gpt2-counterfactual-circuit-transfer":
        raise ValueError("Cannot finalize a report with an unexpected experiment kind.")
    necessity_by_subject = {
        item["subject"]: item for item in report.get("necessityByFact", [])
    }
    for pair in report.get("pairs", []):
        pair["necessityInOriginalContexts"] = {
            subject: necessity_by_subject[subject]
            for subject in (pair["subjectA"], pair["subjectB"])
        }
        details = []
        for subject in (pair["subjectA"], pair["subjectB"]):
            item = necessity_by_subject[subject]
            ablation = item["pathGroupAblation"]
            details.append(
                f"{subject}: answer-sequence log-likelihood drop "
                f"{ablation['answerTotalLogLikelihoodDrop']:.6f}; strict correct-greedy "
                f"necessity={ablation['strictNecessityForBaselineCorrectGreedyPrediction']}."
            )
        pair["explanation"] += (
            "\n\n**Original-context path removal:** " + " ".join(details)
            + " These necessity measurements are separate from counterfactual transfer."
        )
    statistics = _build_family_analysis(report.get("pairs", []), bootstrap_replicates)
    computation = _classify_computation(
        report.get("pairs", []),
        report.get("necessityByFact", []),
        statistics,
    )
    report["pairedStatistics"] = statistics
    report["computationInterpretation"] = computation
    report["acceptanceCriteria"]["I_sharedVsFactSpecificComputation"] = {
        "relationFamilyAnalyses": len(statistics["relationFamilies"]),
        "familiesWithCorrectedPositiveSpecificityEvidence": sum(
            item.get("significantAfterMultipleComparisonCorrection") is True
            and item.get("meanPathMinusMeanControlTransferContrastDelta") is not None
            and item["meanPathMinusMeanControlTransferContrastDelta"] > 0
            for item in statistics["relationFamilies"]
        ),
        "familiesWithCorrectedPositivePathTransferEvidence": sum(
            item.get("pathTransferSignificantAfterMultipleComparisonCorrection") is True
            and item.get("meanPathTransferContrastDelta") is not None
            and item["meanPathTransferContrastDelta"] > 0
            for item in statistics["relationFamilies"]
        ),
        "interpretation": computation["interpretation"],
    }
    report["statisticsFinalization"] = {
        "method": "fact-endpoint cluster bootstrap for path transfer and valid-control contrasts; independent Benjamini-Hochberg adjustment sets",
        "bootstrapReplicates": bootstrap_replicates,
        "reportUpdatedFromPairAndDirectionMeasurements": True,
    }
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_markdown(report, report_path.with_name("report.md"))
    manifest_path = report_path.with_name("manifest.json")
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        summary = manifest.setdefault("summary", {})
        summary["pairedStatistics"] = statistics
        summary["acceptanceCriteria"] = report["acceptanceCriteria"]
        manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    report = finalize_report(args.report, args.bootstrap_replicates)
    print(json.dumps({
        "marker": "GPT2_COUNTERFACTUAL_TRANSFER_REPORT_FINALIZED",
        "variant": report["variant"],
        "relationFamilyCount": len(report["pairedStatistics"]["relationFamilies"]),
        "bootstrapReplicates": args.bootstrap_replicates,
        "report": str(args.report.resolve()),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
