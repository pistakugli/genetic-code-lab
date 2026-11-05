from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = (
    ROOT
    / "benchmarks/release/real-qwen2.5-0.5b-layer0-swiglu/amplitude-polarity-response"
)
CAPTURE_DIR = (
    ROOT
    / "benchmarks/release/real-qwen2.5-0.5b-layer0-swiglu/expanded-final-v2-capture"
)


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_probe() -> tuple[dict, dict[str, np.ndarray]]:
    report_path = EXPERIMENT_DIR / "probe-report.json"
    raw_path = EXPERIMENT_DIR / "amplitude-responses.npz"
    report = _read_json(report_path)
    if (
        report.get("kind") != "gcl-qwen-swiglu-amplitude-polarity-probe"
        or report.get("status") != "captured"
        or report.get("solverAccess") != "oracle-only"
        or report.get("teacherParametersUsedBySolver") is not False
        or report.get("hiddenUnitCountProvidedToSolver") is not False
    ):
        raise ValueError("probe report does not satisfy the oracle-only contract")
    if report.get("accessMode", {}).get("oracleTruthCapability") is not False:
        raise ValueError("probe report does not record an opaque truth-disabled oracle")
    design = report.get("directionDesign", {})
    if (
        design.get("count") != 96
        or design.get("naturalObserved") != 32
        or design.get("naturalCombinations") != 32
        or design.get("independentGaussian") != 32
    ):
        raise ValueError("probe direction counts do not match the recorded design")
    accounting = report.get("queryAccounting", {})
    if (
        accounting.get("oraclePoints") != 2304
        or accounting.get("requestedPoints") != 2304
        or accounting.get("oracleQueryBudget") != 2304
        or accounting.get("evalRequests") != 9
        or accounting.get("rawArtifact") != raw_path.name
        or accounting.get("rawArtifactSha256") != _sha256(raw_path)
    ):
        raise ValueError("probe query accounting or raw artifact digest is inconsistent")

    with np.load(raw_path, allow_pickle=False) as archive:
        raw = {key: archive[key] for key in archive.files}
    expected_keys = {
        "directions",
        "direction_ids",
        "direction_kinds",
        "source_natural_indices",
        "source_coefficients",
        "source_prompt_ids",
        "source_prompt_sha256",
        "source_token_positions",
        "amplitudes",
        "polarities",
        "exact_inputs",
        "oracle_outputs",
        "normalized_responses",
        "consecutive_relative_changes",
        "consecutive_cosines",
        "consecutive_cosine_defined",
        "log2_amplitude_slopes",
        "even_responses",
        "odd_responses",
        "stable_at_final_two_steps",
    }
    if set(raw) != expected_keys:
        raise ValueError("raw response artifact has an unexpected array schema")
    if raw["directions"].shape != (96, 896) or raw["exact_inputs"].shape != (
        96,
        12,
        2,
        896,
    ):
        raise ValueError("raw direction or input dimensions are inconsistent")
    if raw["oracle_outputs"].shape != (96, 12, 2, 896):
        raise ValueError("raw oracle output dimensions are inconsistent")
    if raw["amplitudes"].tolist() != report.get("amplitudes"):
        raise ValueError("raw amplitudes disagree with the probe report")
    if raw["polarities"].tolist() != [1.0, -1.0]:
        raise ValueError("raw polarity order is not positive then negative")
    reconstructed_inputs = (
        raw["directions"][:, None, None, :]
        * raw["amplitudes"][None, :, None, None]
        * raw["polarities"][None, None, :, None]
    )
    if not np.array_equal(reconstructed_inputs, raw["exact_inputs"]):
        raise ValueError("stored query inputs do not match directions/amplitudes/polarities")
    expected_normalized = (
        raw["oracle_outputs"].astype(np.float64)
        / raw["amplitudes"][None, :, None, None].astype(np.float64) ** 2
    ).astype(np.float32)
    if not np.array_equal(expected_normalized, raw["normalized_responses"]):
        raise ValueError("stored normalized responses disagree with raw oracle outputs")
    return report, raw


def verify_probe() -> None:
    _verify_probe()
    print("opaque amplitude artifact verified")


def verify_evaluation() -> None:
    probe, _ = _verify_probe()
    report_path = EXPERIMENT_DIR / "probe-report.json"
    raw_path = EXPERIMENT_DIR / "amplitude-responses.npz"
    evaluation = _read_json(EXPERIMENT_DIR / "evaluation-report.json")
    if (
        evaluation.get("kind") != "gcl-qwen-swiglu-amplitude-polarity-evaluation"
        or evaluation.get("status") != "evaluated_posthoc"
        or evaluation.get("groundTruthWeightsReadByEvaluator") is not True
        or evaluation.get("groundTruthWeightsUsedBySolver") is not False
    ):
        raise ValueError("post-hoc evaluation boundary is not recorded correctly")
    if evaluation.get("solverReportSha256") != _sha256(report_path):
        raise ValueError("post-hoc report is not bound to the current probe report")
    if evaluation.get("rawArtifactSha256") != _sha256(raw_path):
        raise ValueError("post-hoc report is not bound to the raw measurement artifact")
    if evaluation.get("captureReportSha256") != probe.get("captureReportSha256"):
        raise ValueError("post-hoc evaluation is not bound to the activation capture")
    replay = evaluation.get("independentOracleReplay", {})
    if replay.get("maxPerSampleRelativeL2Delta", float("inf")) > replay.get(
        "relativeL2Tolerance", 0.0
    ):
        raise ValueError("independent oracle replay exceeds its relative tolerance")
    if evaluation.get("asymptoticResponse", {}).get(
        "exactPolarityReversalIdentityMaxAbsoluteError", float("inf")
    ) > 1e-10:
        raise ValueError("post-hoc polarity-reversal identity check failed")
    if evaluation.get("responseStability", {}).get(
        "overallStableDirectionFraction"
    ) != 0.0:
        raise ValueError("the measured stability verdict changed")
    signatures = evaluation.get("posthocComponentSignatureAnalysis", {})
    if (
        signatures.get("teacherComponentCountReadOnlyAfterProbe") != 4864
        or signatures.get("isSolverRecoveryEvidence") is not False
    ):
        raise ValueError("post-hoc component signatures are misclassified")
    if evaluation.get("conclusion", {}).get(
        "structuralRecoveryAlgorithmImplemented"
    ) is not False or evaluation["conclusion"].get(
        "recoveryJustifiedByMeasurements"
    ) is not False:
        raise ValueError("the experiment report overstates structural recovery")
    print("posthoc amplitude evaluation verified")


def verify_documentation() -> None:
    probe, _ = _verify_probe()
    evaluation = _read_json(EXPERIMENT_DIR / "evaluation-report.json")
    research_path = ROOT / "docs/research/2026-11-01-qwen2.5-0.5b-swiglu-oracle-recovery.md"
    roadmap_path = ROOT / "docs/nn-decompiler-roadmap.md"
    research = " ".join(research_path.read_text(encoding="utf-8").lower().split())
    roadmap = " ".join(roadmap_path.read_text(encoding="utf-8").lower().split())
    if probe["queryAccounting"]["oraclePoints"] != 2304:
        raise ValueError("documented query count has no matching raw witness")
    if evaluation["responseStability"]["overallStableDirectionFraction"] != 0.0:
        raise ValueError("documented instability verdict has no matching evaluator witness")
    expected_text = (
        "did not justify an amplitude-response recovery",
        "adaptive activation-boundary localization",
    )
    if not all(token in research for token in expected_text):
        raise ValueError("research report omits the negative result or next direction")
    if not all(token in roadmap for token in ("amplitude-polarity", *expected_text[1:])):
        raise ValueError("roadmap omits the measured outcome or next direction")
    if not research_path.is_file() or not roadmap_path.is_file():
        raise FileNotFoundError("research documentation is missing")
    print("amplitude response evidence documentation verified")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("probe", "evaluation", "documentation"),
        required=True,
    )
    stage = parser.parse_args().stage
    {
        "probe": verify_probe,
        "evaluation": verify_evaluation,
        "documentation": verify_documentation,
    }[stage]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
