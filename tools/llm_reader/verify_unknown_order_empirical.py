#!/usr/bin/env python3
"""Independent verifier for unknown-order empirical SwiGLU ladder artifacts.

Recomputes, from the raw recorded files only, everything the ladder runner
decided:

- case-report structural gates (schema, splits, protocol, ladder shape);
- the selection rule: the inferred order must be the first sufficient
  candidate under the recorded threshold, and the upper-region
  non-identifiability flag must match the recorded r values;
- per-fit consistency: every fit's solver report must declare the same
  candidate atom count, must record an unknown-order run (unitCount null,
  opaque access audit, no truth requests), its oracle witness query count
  must match, and its frozen-recovery.npz must hash, load, and shape-check
  against the candidate;
- grading recompute: when the teacher-side response table is available,
  rSelection / rValidation / rOod / validationR2 / validationMae are
  recomputed from the frozen fit and the recorded responses with the same
  relative-MAE definition the runner used;
- pool provenance: when the pool file is available its sha256 must match the
  case report and its splits must be disjoint index sets of the pool;
- teacher-secret consistency (run-info + teacher manifest): the true m must
  equal the manifest unit count, and the artifact must record honestly
  whether the inferred order equals it (order correctness is a recorded
  scientific finding, not a verifier error).

Verifier dependencies: numpy, and `predict_swiglu`/`regression_metrics`
from the repo's own solver module (dependency recorded here per the
preregistered contract; the grading formulas are the same ones the runner
used, applied independently to the recorded raw tables).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.llm_reader.blind_swiglu_oracle_decompiler import (
    predict_swiglu,
    regression_metrics,
)


SPLIT_NAMES = ("train", "selection", "validation")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _close(actual: Any, expected: float, *, rel: float = 1e-6, abs_tol: float = 1e-9) -> bool:
    return _is_number(actual) and math.isclose(float(actual), expected, rel_tol=rel, abs_tol=abs_tol)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    std = float(np.std(actual))
    if std <= 0.0:
        raise RuntimeError("teacher response std is zero; r is undefined")
    return float(np.mean(np.abs(actual - predicted)) / std)


def _recompute_selection(fits: List[Dict[str, Any]], r_threshold: float, extra_steps: int) -> Dict[str, Any]:
    sufficient = [fit for fit in fits if not fit.get("fitFailed") and fit.get("sufficient") is True]
    inferred_order = None
    first_r = None
    if sufficient:
        inferred_order = int(sufficient[0]["candidate"])
        first_r = float(sufficient[0]["rSelection"])
    verdict = "inferred-order" if inferred_order is not None else "no-sufficient-order"
    later = [fit for fit in sufficient if int(fit["candidate"]) > inferred_order] if inferred_order is not None else []
    upper_region = bool(later and all(float(fit["rSelection"]) <= 2.0 * first_r for fit in later))
    # the traversal rule: after the first sufficient candidate the runner walks
    # at most `extra_steps` further ladder entries, and never exceeds max fits
    after_first = 0
    if inferred_order is not None:
        order = [int(fit["candidate"]) for fit in fits]
        first_index = order.index(inferred_order)
        after_first = len(order) - first_index - 1
    return {
        "inferredOrder": inferred_order,
        "verdict": verdict,
        "upperRegionNonIdentifiable": upper_region,
        "afterFirstSufficient": after_first,
        "firstSufficientRSelection": first_r,
        "sufficientCandidateCount": len(sufficient),
    }


def verify_case_report(report: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    prefix = "case-report: "
    if report.get("kind") != "unknown-order-swiglu-ladder-report":
        errors.append(prefix + "kind is not the unknown-order ladder report")
    if report.get("schemaVersion") != 1:
        errors.append(prefix + "schemaVersion must be 1")
    if report.get("synthetic") is not False:
        errors.append(prefix + "real-model lane must record synthetic=false")
    teacher = report.get("teacher", {})
    if teacher.get("kind") != "real-model":
        errors.append(prefix + "teacher kind must be real-model")
    for key in ("model", "layer"):
        if not teacher.get(key) and teacher.get(key) != 0:
            errors.append(prefix + f"teacher {key} is missing")
    pool = report.get("pool", {})
    splits = pool.get("splits", {})
    if not set(splits) == set(SPLIT_NAMES):
        errors.append(prefix + "pool splits must be exactly train/selection/validation")
    if _is_number(pool.get("pointCount")) and sum(int(v) for v in splits.values()) != int(pool["pointCount"]):
        errors.append(prefix + "pool split counts do not sum to the recorded point count")
    for name in SPLIT_NAMES:
        if not _is_number(splits.get(name)) or int(splits[name]) <= 0:
            errors.append(prefix + f"pool split {name} is not a positive count")
    if not isinstance(pool.get("sha256"), str) or len(pool.get("sha256", "")) != 64:
        errors.append(prefix + "pool sha256 is missing or malformed")
    protocol = report.get("protocol", {})
    ladder = protocol.get("ladder")
    if not isinstance(ladder, list) or not ladder or any(not isinstance(v, int) for v in ladder):
        errors.append(prefix + "ladder must be a non-empty integer list")
    else:
        if sorted(ladder) != ladder or len(set(ladder)) != len(ladder):
            errors.append(prefix + "ladder must be strictly ascending without duplicates")
        if any(v <= 0 for v in ladder):
            errors.append(prefix + "ladder candidates must be positive")
    if not _is_number(protocol.get("rThreshold")) or float(protocol["rThreshold"]) <= 0.0:
        errors.append(prefix + "rThreshold must be positive")
    if not isinstance(protocol.get("extraLadderSteps"), int) or protocol["extraLadderSteps"] < 0:
        errors.append(prefix + "extraLadderSteps must be a non-negative integer")
    if not isinstance(protocol.get("oracleBudgetPerFit"), int) or protocol["oracleBudgetPerFit"] <= 0:
        errors.append(prefix + "oracleBudgetPerFit must be positive")
    if not isinstance(protocol.get("maxFits"), int) or protocol["maxFits"] <= 0:
        errors.append(prefix + "maxFits must be positive")
    if not _is_number(protocol.get("quantizationStep")) or float(protocol["quantizationStep"]) <= 0.0:
        errors.append(prefix + "quantizationStep must be positive")
    fits = report.get("fits")
    if not isinstance(fits, list) or not fits:
        errors.append(prefix + "fits must be a non-empty list")
        return errors
    candidates: List[int] = []
    r_threshold = float(protocol.get("rThreshold", 0.0))
    for index, fit in enumerate(fits):
        fit_prefix = prefix + f"fit {index}: "
        candidate = fit.get("candidate")
        if not isinstance(candidate, int):
            errors.append(fit_prefix + "candidate is not an integer")
            continue
        candidates.append(candidate)
        if isinstance(ladder, list) and candidate not in ladder:
            errors.append(fit_prefix + f"candidate {candidate} is not in the preregistered ladder")
        if fit.get("fitFailed"):
            if fit.get("sufficient") is not False:
                errors.append(fit_prefix + "a failed fit must never be sufficient")
            continue
        for key in ("rSelection", "rValidation", "rOod"):
            if not _is_number(fit.get(key)):
                errors.append(fit_prefix + f"{key} is missing or not finite")
        if _is_number(fit.get("rSelection")):
            expected_sufficient = float(fit["rSelection"]) <= r_threshold
            if bool(fit.get("sufficient")) is not expected_sufficient:
                errors.append(fit_prefix + "sufficient flag disagrees with rSelection vs the recorded threshold")
        if not _is_number(fit.get("oracleCalls")) or int(fit.get("oracleCalls", -1)) < 0:
            errors.append(fit_prefix + "oracleCalls is missing or negative")
        elif isinstance(protocol.get("oracleBudgetPerFit"), int) and int(fit["oracleCalls"]) > int(
            protocol["oracleBudgetPerFit"]
        ):
            errors.append(fit_prefix + "oracleCalls exceeds the recorded per-fit budget")
        if fit.get("straceModelAccessRows") not in (None, []):
            errors.append(fit_prefix + "strace recorded solver model-file access rows")
    if len(set(candidates)) != len(candidates):
        errors.append(prefix + "ladder traversal repeats a candidate")
    if isinstance(protocol.get("maxFits"), int) and len(fits) > int(protocol["maxFits"]):
        errors.append(prefix + "fits exceed the recorded maxFits")

    recomputed = _recompute_selection(fits, r_threshold, int(protocol.get("extraLadderSteps", 0)))
    if report.get("inferredOrder") != recomputed["inferredOrder"]:
        errors.append(prefix + "inferredOrder disagrees with the first-sufficient-candidate rule")
    if report.get("verdict") != recomputed["verdict"]:
        errors.append(prefix + "verdict disagrees with the recorded selection data")
    if report.get("upperRegionNonIdentifiable") is not recomputed["upperRegionNonIdentifiable"]:
        errors.append(prefix + "upper-region non-identifiability flag disagrees with the recorded r values")
    if recomputed["afterFirstSufficient"] > int(protocol.get("extraLadderSteps", 0)):
        errors.append(prefix + "traversal continued past the preregistered extra ladder steps")
    return errors


def _load_frozen_fit(path: Path) -> Dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        return {
            "basis": np.asarray(data["basis"], dtype=np.float64),
            "gates": np.asarray(data["gates"], dtype=np.float64),
            "ups": np.asarray(data["ups"], dtype=np.float64),
            "scales": np.asarray(data["scales"], dtype=np.float64),
            "bias": float(np.asarray(data["bias"]).item()),
            "metadata": json.loads(str(data["metadata"].item())),
        }


def verify_fit_dirs(case_dir: Path, report: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    dimension = int(report["teacher"]["dimension"])
    for fit in report.get("fits", []):
        candidate = fit.get("candidate")
        if not isinstance(candidate, int) or fit.get("fitFailed"):
            continue
        prefix = f"fit m{candidate}: "
        fit_dir = case_dir / "fits" / f"m{candidate}"
        solver_path = fit_dir / "solver-report.json"
        if not solver_path.is_file():
            errors.append(prefix + "solver-report.json is missing")
            continue
        try:
            solver_report = json.loads(solver_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            errors.append(prefix + f"cannot read solver report: {error}")
            continue
        summary = solver_report.get("summary", {})
        if summary.get("atomCount") != candidate:
            errors.append(prefix + "solver report atom count does not match the ladder candidate")
        if summary.get("unknownOrder") is not True:
            errors.append(prefix + "solver report does not record an unknown-order run")
        if summary.get("unitCount") is not None:
            errors.append(prefix + "unknown-order solver report leaked unitCount")
        if summary.get("activeRank") != 2 * candidate:
            errors.append(prefix + "activeRank is not the 2-per-atom protocol rank")
        if summary.get("dimension") != dimension:
            errors.append(prefix + "solver report dimension does not match the teacher")
        witness_path = fit_dir / "oracle-witness.json"
        if not witness_path.is_file():
            errors.append(prefix + "oracle-witness.json is missing")
        else:
            try:
                witness = json.loads(witness_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                errors.append(prefix + f"cannot read oracle witness: {error}")
                witness = {}
            if witness.get("status") != "complete":
                errors.append(prefix + "oracle witness did not complete")
            if int(witness.get("queryCount", -1)) != int(fit.get("oracleCalls", -2)):
                errors.append(prefix + "oracle witness query count does not match the case report")
        audit = solver_report.get("accessAudit", {})
        for key in (
            "weightsUsedBySolver",
            "hooksUsedBySolver",
            "sourceIdsUsedBySolver",
            "truthCapabilityAvailable",
            "fitSnapshotUsedBySolver",
        ):
            if audit.get(key) is not False:
                errors.append(prefix + f"access audit {key} is not false")
        if audit.get("solverAccess") != "opaque-eval-only-oracle":
            errors.append(prefix + "solver access is not the opaque eval-only oracle")
        if int(audit.get("truthRequestCount", -1)) != 0:
            errors.append(prefix + "solver requested truth during the blind phase")
        if audit.get("teacherManifestFieldsObserved") not in (None, []):
            errors.append(prefix + "solver observed teacher manifest fields")
        frozen_path = fit_dir / "frozen-recovery.npz"
        if not frozen_path.is_file():
            errors.append(prefix + "frozen-recovery.npz is missing")
            continue
        frozen_sha = solver_report.get("frozenRecovery", {}).get("sha256")
        if not isinstance(frozen_sha, str) or sha256_file(frozen_path) != frozen_sha:
            errors.append(prefix + "frozen recovery sha256 does not match the solver report")
        try:
            frozen = _load_frozen_fit(frozen_path)
        except (OSError, ValueError, KeyError) as error:
            errors.append(prefix + f"frozen recovery does not load: {error}")
            continue
        meta = frozen["metadata"]
        if meta.get("atomCount") != candidate or meta.get("activeRank") != 2 * candidate:
            errors.append(prefix + "frozen recovery metadata does not match the candidate/rank")
        if meta.get("solverAccess") != "oracle-only":
            errors.append(prefix + "frozen recovery was not produced oracle-only")
        if meta.get("weightsUsedBySolver") is not False:
            errors.append(prefix + "frozen recovery metadata records weight access")
        if frozen["basis"].shape != (2 * candidate, dimension):
            errors.append(prefix + "frozen basis shape does not match (2m, dimension)")
        if frozen["gates"].shape != (candidate, 2 * candidate) or frozen["ups"].shape != (candidate, 2 * candidate):
            errors.append(prefix + "frozen gates/ups shapes do not match (m, 2m)")
        if frozen["scales"].shape != (candidate,):
            errors.append(prefix + "frozen scales shape does not match (m,)")
    return errors


def verify_teacher_side(case_dir: Path, report: Dict[str, Any]) -> List[str]:
    """Recompute the recorded grades from the frozen fits and response tables."""
    errors: List[str] = []
    table_path = case_dir / "teacher-side" / "eval-responses.npz"
    if not table_path.is_file():
        return errors
    with np.load(table_path, allow_pickle=False) as data:
        tables = {
            name: (np.asarray(data[f"{name}_x"], dtype=np.float64), np.asarray(data[f"{name}_f"], dtype=np.float64))
            for name in ("selection", "validation", "ood")
        }
    for fit in report.get("fits", []):
        candidate = fit.get("candidate")
        if not isinstance(candidate, int) or fit.get("fitFailed"):
            continue
        prefix = f"fit m{candidate}: "
        frozen_path = case_dir / "fits" / f"m{candidate}" / "frozen-recovery.npz"
        if not frozen_path.is_file():
            continue
        try:
            frozen = _load_frozen_fit(frozen_path)
        except (OSError, ValueError, KeyError):
            continue
        fit_params = {
            "gates": frozen["gates"],
            "ups": frozen["ups"],
            "scales": frozen["scales"],
            "bias": frozen["bias"],
        }
        basis = frozen["basis"]
        for name, key in (("selection", "rSelection"), ("validation", "rValidation"), ("ood", "rOod")):
            x_table, f_table = tables[name]
            predicted = predict_swiglu(x_table @ basis.T, fit_params)
            recomputed = relative_mae(f_table, predicted)
            if not _close(fit.get(key), recomputed, rel=1e-4, abs_tol=1e-9):
                errors.append(prefix + f"{key} does not match the recomputed relative MAE")
        x_table, f_table = tables["validation"]
        predicted = predict_swiglu(x_table @ basis.T, fit_params)
        metrics = regression_metrics(f_table, predicted)
        if _is_number(fit.get("validationR2")) and not _close(fit["validationR2"], metrics["r2"], rel=1e-4):
            errors.append(prefix + "validationR2 does not match the recomputed metric")
        if _is_number(fit.get("validationMae")) and not _close(fit["validationMae"], metrics["mae"], rel=1e-4):
            errors.append(prefix + "validationMae does not match the recomputed metric")
    return errors


def verify_pool(pool_path: Path, report: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    prefix = "pool: "
    if not pool_path.is_file():
        errors.append(prefix + f"pool file is missing at {pool_path}")
        return errors
    recorded_sha = report.get("pool", {}).get("sha256", "")
    if sha256_file(pool_path) != recorded_sha:
        errors.append(prefix + "sha256 does not match the case report")
        return errors
    with np.load(pool_path, allow_pickle=False) as data:
        if "x" not in data:
            errors.append(prefix + "pool does not carry an x table")
            return errors
        x = np.asarray(data["x"])
        indices = {name: np.asarray(data[name], dtype=np.int64) for name in SPLIT_NAMES}
    point_count = int(report["pool"]["pointCount"])
    dimension = int(report["teacher"]["dimension"])
    if x.shape != (point_count, dimension):
        errors.append(prefix + f"pool shape {x.shape} does not match the recorded counts")
    seen: set = set()
    for name, idx in indices.items():
        if idx.ndim != 1 or idx.size == 0 or idx.min() < 0 or idx.max() >= point_count:
            errors.append(prefix + f"split {name} is not a valid index set")
            continue
        if int(idx.size) != int(report["pool"]["splits"][name]):
            errors.append(prefix + f"split {name} size does not match the case report")
        overlap = seen.intersection(idx.tolist())
        if overlap:
            errors.append(prefix + f"split {name} overlaps an earlier split")
        seen.update(idx.tolist())
    return errors


def verify_teacher_secret(
    run_info: Dict[str, Any], teacher_manifest: Dict[str, Any], report: Dict[str, Any]
) -> List[str]:
    errors: List[str] = []
    prefix = "teacher-secret: "
    secret = run_info.get("teacherSecret", {})
    true_m = secret.get("trueM")
    if not isinstance(true_m, int) or true_m <= 0:
        errors.append(prefix + "run-info teacherSecret.trueM is missing")
        return errors
    units = teacher_manifest.get("units")
    if not isinstance(units, list) or not all(isinstance(v, int) for v in units):
        errors.append(prefix + "teacher manifest units are missing")
    elif len(units) != true_m:
        errors.append(prefix + "teacher manifest unit count does not equal the recorded true m")
    if secret.get("unitCount") not in (None, len(units) if isinstance(units, list) else None):
        errors.append(prefix + "run-info unitCount disagrees with the teacher manifest")
    channel = teacher_manifest.get("outputChannel")
    if not isinstance(channel, int) or channel < 0:
        errors.append(prefix + "teacher manifest outputChannel is missing")
    if teacher_manifest.get("model") != report["teacher"].get("model"):
        errors.append(prefix + "teacher manifest model disagrees with the case report")
    if teacher_manifest.get("layer") != report["teacher"].get("layer"):
        errors.append(prefix + "teacher manifest layer disagrees with the case report")
    inferred = report.get("inferredOrder")
    if inferred is not None and inferred != true_m:
        # recorded honestly as a finding: the ladder over/under-estimates the
        # hidden order; the artifact must keep both numbers side by side.
        if "orderCorrect" in run_info and run_info["orderCorrect"] is not False:
            errors.append(prefix + "orderCorrect flag disagrees with inferred vs true m")
    if run_info.get("timedOut") is True and report.get("verdict"):
        errors.append(prefix + "run-info records a timeout but the case report completed")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify an unknown-order empirical SwiGLU ladder artifact.")
    parser.add_argument(
        "--case-dir",
        required=True,
        help="ladder output directory (case-report.json + fits/ [+ teacher-side/])",
    )
    parser.add_argument("--pool", default="", help="optional recorded empirical pool .npz for provenance checks")
    parser.add_argument(
        "--run-info",
        default="",
        help="optional run-info.json (default: ../run-info.json relative to the case dir)",
    )
    parser.add_argument(
        "--teacher-manifest",
        default="",
        help="optional teacher-manifest.json (default: ../teacher-manifest.json relative to the case dir)",
    )
    args = parser.parse_args()

    case_dir = Path(args.case_dir).resolve()
    report_path = case_dir / "case-report.json"
    if not report_path.is_file():
        raise SystemExit(f"missing case report: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))

    errors = verify_case_report(report)
    errors += verify_fit_dirs(case_dir, report)
    errors += verify_teacher_side(case_dir, report)

    if args.pool:
        errors += verify_pool(Path(args.pool).resolve(), report)

    run_info_path = Path(args.run_info).resolve() if args.run_info else case_dir.parent / "run-info.json"
    teacher_manifest_path = (
        Path(args.teacher_manifest).resolve() if args.teacher_manifest else case_dir.parent / "teacher-manifest.json"
    )
    if run_info_path.is_file():
        run_info = json.loads(run_info_path.read_text(encoding="utf-8"))
        if teacher_manifest_path.is_file():
            teacher_manifest = json.loads(teacher_manifest_path.read_text(encoding="utf-8"))
            errors += verify_teacher_secret(run_info, teacher_manifest, report)
        else:
            errors.append("teacher-secret: teacher manifest is missing at " + str(teacher_manifest_path))
        if run_info.get("caseVerdict") != report.get("verdict"):
            errors.append("run-info: caseVerdict disagrees with the case report")
        if run_info.get("inferredOrder") != report.get("inferredOrder"):
            errors.append("run-info: inferredOrder disagrees with the case report")
    elif run_info_path != case_dir.parent / "run-info.json":
        errors.append("run-info: explicitly requested run-info file is missing")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    print("UNKNOWN_ORDER_EMPIRICAL_ARTIFACT_VERIFIED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
