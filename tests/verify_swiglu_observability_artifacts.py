from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.llm_reader.swiglu_observability_experiments import (  # noqa: E402
    DEFAULT_QUERY_BUDGET,
    DEFAULT_SEEDS,
    HYPOTHESES,
    PROTOCOLS,
    TEACHER_CASES,
    make_synthetic_teacher,
)


DEFAULT_ARTIFACT_DIR = (
    ROOT / "benchmarks/research/swiglu-hidden-structure-observability"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _independent_forward(
    inputs: np.ndarray,
    gate: np.ndarray,
    up: np.ndarray,
    down: np.ndarray,
) -> np.ndarray:
    points = np.asarray(inputs, dtype=np.float64)
    gate_pre = points @ gate.T
    up_pre = points @ up.T
    sigmoid = np.empty_like(gate_pre)
    positive = gate_pre >= 0.0
    sigmoid[positive] = 1.0 / (1.0 + np.exp(-gate_pre[positive]))
    exp_values = np.exp(gate_pre[~positive])
    sigmoid[~positive] = exp_values / (1.0 + exp_values)
    return ((gate_pre * sigmoid) * up_pre) @ down.T


def _load_report(artifact_dir: Path) -> dict[str, Any]:
    path = artifact_dir / "study-report.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing synthetic study report: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def verify_report(artifact_dir: Path) -> None:
    report = _load_report(artifact_dir)
    mechanisms = report.get("queryMechanisms", {})
    if set(mechanisms) != set(HYPOTHESES):
        raise AssertionError("report does not contain exactly three hypotheses")
    for name in HYPOTHESES:
        entry = mechanisms[name]
        for field in (
            "mathematical_reason",
            "expected_observable_signature",
            "failure_conditions",
            "artifact_criterion",
        ):
            if not entry.get(field):
                raise AssertionError(f"{name} is missing {field}")
    controls = report.get("negativeControls", {})
    if not controls.get("outputReadoutPostprocessing"):
        raise AssertionError("output-readout information control is missing")
    equivalences = controls.get("exactParameterEquivalences", {})
    for key in ("permutation", "reciprocalUpDownScaling", "duplicateGateGL2", "zeroDownColumn"):
        if not equivalences.get(key):
            raise AssertionError(f"identifiability control {key} is missing")
    if report.get("scope") != "synthetic_only":
        raise AssertionError("study scope must remain synthetic-only")
    if report.get("experimentDesign", {}).get("newQwenOraclePointsSpent") != 0:
        raise AssertionError("synthetic report claims a Qwen oracle query")


def verify_artifacts(artifact_dir: Path) -> None:
    report = _load_report(artifact_dir)
    expected_cases = len(DEFAULT_SEEDS) * len(TEACHER_CASES)
    if len(report.get("cases", [])) != expected_cases:
        raise AssertionError("unexpected number of teacher cases")
    expected_total = expected_cases * len(PROTOCOLS) * report[
        "queryAccounting"
    ]["pointsPerProtocol"]
    if report["queryAccounting"]["oraclePoints"] != expected_total:
        raise AssertionError("reported oracle call total is inconsistent")
    if report["queryAccounting"]["expectedOraclePoints"] != expected_total:
        raise AssertionError("expected oracle call total is inconsistent")
    if report["queryAccounting"]["pointsPerProtocol"] != DEFAULT_QUERY_BUDGET:
        raise AssertionError("default study did not use the preregistered budget")
    if report["experimentDesign"]["teacherSeeds"] != list(DEFAULT_SEEDS):
        raise AssertionError("study used different teacher seeds")

    for row in report["cases"]:
        dims = row["dimensions"]
        teacher = make_synthetic_teacher(
            row["case"],
            input_dim=dims["inputDim"],
            output_dim=dims["outputDim"],
            width=dims["teacherHiddenWidthPosthocOnly"],
            seed=row["teacherSeedForEvaluator"],
        )
        protocol_rows = {entry["protocol"]: entry for entry in row["protocols"]}
        if set(protocol_rows) != set(PROTOCOLS):
            raise AssertionError("case is missing a measurement protocol")
        for protocol in PROTOCOLS:
            record = protocol_rows[protocol]
            accounting = record["queryAccounting"]
            if accounting["oraclePoints"] != DEFAULT_QUERY_BUDGET:
                raise AssertionError("per-protocol oracle budget changed")
            if accounting["trainingPoints"] + accounting["heldoutPoints"] != DEFAULT_QUERY_BUDGET:
                raise AssertionError("training/held-out accounting does not sum")
            raw_record = row["rawArtifacts"][protocol]
            raw_path = artifact_dir / raw_record["file"]
            if not raw_path.is_file():
                raise FileNotFoundError(f"missing raw oracle data: {raw_path}")
            if _sha256(raw_path) != raw_record["sha256"]:
                raise AssertionError(f"raw artifact hash mismatch: {raw_path}")
            with np.load(raw_path, allow_pickle=False) as raw:
                training_inputs = raw["training_inputs"]
                training_outputs = raw["training_outputs"]
                heldout_inputs = raw["heldout_inputs"]
                heldout_outputs = raw["heldout_outputs"]
                readout = raw["output_readout_matrix"]
                saved_training_readouts = raw["training_readout_outputs"]
                saved_heldout_readouts = raw["heldout_readout_outputs"]

            if training_inputs.shape != (
                accounting["trainingPoints"],
                dims["inputDim"],
            ):
                raise AssertionError("invalid training-input shape")
            if training_outputs.shape != (
                accounting["trainingPoints"],
                dims["outputDim"],
            ):
                raise AssertionError("invalid training-output shape")
            if heldout_inputs.shape != (
                accounting["heldoutPoints"],
                dims["inputDim"],
            ):
                raise AssertionError("invalid held-out-input shape")
            if heldout_outputs.shape != (
                accounting["heldoutPoints"],
                dims["outputDim"],
            ):
                raise AssertionError("invalid held-out-output shape")
            if not all(
                np.isfinite(array).all()
                for array in (
                    training_inputs,
                    training_outputs,
                    heldout_inputs,
                    heldout_outputs,
                    readout,
                    saved_training_readouts,
                    saved_heldout_readouts,
                )
            ):
                raise AssertionError("raw oracle artifact contains non-finite data")

            expected_training = _independent_forward(
                training_inputs, teacher.gate, teacher.up, teacher.down
            )
            expected_heldout = _independent_forward(
                heldout_inputs, teacher.gate, teacher.up, teacher.down
            )
            np.testing.assert_allclose(
                training_outputs, expected_training, rtol=2e-13, atol=2e-13
            )
            np.testing.assert_allclose(
                heldout_outputs, expected_heldout, rtol=2e-13, atol=2e-13
            )
            projected_training = training_outputs @ readout.T
            projected_heldout = heldout_outputs @ readout.T
            np.testing.assert_allclose(
                saved_training_readouts,
                projected_training,
                rtol=2e-13,
                atol=2e-13,
            )
            np.testing.assert_allclose(
                saved_heldout_readouts,
                projected_heldout,
                rtol=2e-13,
                atol=2e-13,
            )
            if record["outputReadoutPostprocessingControl"][
                "reconstructionMaximumAbsoluteError"
            ] > 2e-13:
                raise AssertionError("output readout postprocessing did not replay")

        environment_protocol = protocol_rows["activation_environments"]
        subcohorts = environment_protocol[
            "activationEnvironmentSubcohortsPosthoc"
        ]
        if set(subcohorts) != {"sparse", "correlated"}:
            raise AssertionError("activation-environment subcohorts are incomplete")
        for metrics in subcohorts.values():
            if metrics["heldoutPoints"] != metrics["iidMatchedHeldoutPoints"]:
                raise AssertionError("activation cohort comparison is not sample-matched")
            expected_improvement = (
                metrics["iidMedianNearestAbsoluteCosine"]
                - metrics["medianNearestAbsoluteCosine"]
            )
            if abs(metrics["improvementVsIid"] - expected_improvement) > 1e-14:
                raise AssertionError("activation cohort improvement is inconsistent")

        for name, metrics in row[
            "equivalentParameterizationUnseenInputMetricsPosthoc"
        ].items():
            if metrics["relativeError"] > 1e-13:
                raise AssertionError(f"parameter equivalence failed: {name}")


def verify_decision(artifact_dir: Path) -> None:
    report = _load_report(artifact_dir)
    comparisons = report["posthocComparisons"]
    recomputed = {}
    for protocol, comparison in comparisons.items():
        seed_results = comparison["seedResults"]
        if len(seed_results) != 3:
            raise AssertionError(f"{protocol} criterion must use three seeds")
        for item in seed_results:
            expected = (
                item["heldoutMedianPostprojectionCosineImprovement"] is not None
                and item["heldoutP95PostprojectionCosineImprovement"] is not None
                and item["heldoutMedianPostprojectionCosineImprovement"] >= 0.10
                and item["heldoutP95PostprojectionCosineImprovement"] >= 0.05
                and item["protocolSpecificCriterionMet"]
            )
            if item["criterionMet"] is not bool(expected):
                raise AssertionError(f"{protocol} per-seed criterion is inconsistent")
        passing = sum(bool(item["criterionMet"]) for item in seed_results)
        if comparison["passingTeacherSeeds"] != passing:
            raise AssertionError(f"{protocol} passing-seed count is inconsistent")
        criterion = passing >= 2
        if comparison["syntheticInformationCriterionMet"] is not criterion:
            raise AssertionError(f"{protocol} aggregate criterion is inconsistent")
        recomputed[protocol] = criterion

    any_posthoc_candidate = any(recomputed.values())
    decision = report["qwenPilotDecision"]
    if decision["syntheticInformationCriterionMet"] is not any_posthoc_candidate:
        raise AssertionError("post-hoc screening decision disagrees with comparisons")
    if decision["oracleOnlyRecoveryDemonstrated"] is not False:
        raise AssertionError("study has no oracle-only recovery witness")
    if decision["runQwenPilot"] is not False:
        raise AssertionError(
            "post-hoc component signatures alone must not authorize a Qwen pilot"
        )
    if report["identifiabilityResultsPosthoc"]["oracleOnlyRecoveryImplemented"]:
        raise AssertionError("no oracle-only recovery solver was implemented")
    if report["experimentDesign"]["newQwenOraclePointsSpent"] != 0:
        raise AssertionError("the synthetic study itself must not query Qwen")


def verify_documentation(artifact_dir: Path) -> None:
    report_path = ROOT / "docs/research/2026-11-01-qwen2.5-0.5b-swiglu-oracle-recovery.md"
    roadmap_path = ROOT / "docs/nn-decompiler-roadmap.md"
    for path in (report_path, roadmap_path):
        if not path.is_file():
            raise FileNotFoundError(f"missing research documentation: {path}")
    research = report_path.read_text(encoding="utf-8").lower()
    roadmap = roadmap_path.read_text(encoding="utf-8").lower()
    if "synthetic hidden-structure observability study (2026-11-01)" not in research:
        raise AssertionError("chronological research report omits the synthetic study")
    required_terms = (
        "coordinate erasure",
        "complementary blocks",
        "activation environments",
        "oracle-only",
        "identifiability",
        "next gate",
        "qwen",
    )
    for term in required_terms:
        if term not in research:
            raise AssertionError(f"research report omits {term!r}")
    if "swiglu hidden-structure observability" not in roadmap:
        raise AssertionError("roadmap does not link the observability result")
    report = _load_report(artifact_dir)
    formatted_points = f'{report["queryAccounting"]["oraclePoints"]:,}'
    if formatted_points not in research:
        raise AssertionError("research report does not include measured oracle count")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("report", "artifacts", "decision", "documentation"),
        required=True,
    )
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    args = parser.parse_args()
    if args.stage == "report":
        verify_report(args.artifact_dir)
        print("three measurement hypotheses and two identifiability controls recorded")
    elif args.stage == "artifacts":
        verify_artifacts(args.artifact_dir)
        print("synthetic observability artifacts verified")
    elif args.stage == "decision":
        verify_decision(args.artifact_dir)
        print("Qwen campaign decision verified")
    else:
        verify_documentation(args.artifact_dir)
        print("hidden-structure research documentation verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
