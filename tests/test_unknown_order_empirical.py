from __future__ import annotations

import hashlib
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.llm_reader.verify_unknown_order_empirical import (  # noqa: E402
    _recompute_selection,
    relative_mae,
    sha256_file,
    verify_case_report,
    verify_fit_dirs,
    verify_pool,
    verify_teacher_secret,
    verify_teacher_side,
)


def _base_report(**overrides):
    report = {
        "schemaVersion": 1,
        "kind": "unknown-order-swiglu-ladder-report",
        "synthetic": False,
        "teacher": {"kind": "real-model", "model": "m", "layer": 1, "dimension": 8},
        "pool": {
            "sha256": "0" * 64,
            "pointCount": 100,
            "splits": {"train": 60, "selection": 20, "validation": 20},
        },
        "protocol": {
            "ladder": [4, 6, 8, 10],
            "rThreshold": 0.05,
            "extraLadderSteps": 1,
            "oracleBudgetPerFit": 1000,
            "maxFits": 9,
            "quantizationStep": 1e-06,
            "oodCount": 16,
            "oodSeed": 7,
        },
        "fits": [
            {"candidate": 4, "fitFailed": False, "rSelection": 0.08, "rValidation": 0.09, "rOod": 0.1,
             "oracleCalls": 100, "sufficient": False},
            {"candidate": 6, "fitFailed": False, "rSelection": 0.01, "rValidation": 0.01, "rOod": 0.02,
             "oracleCalls": 100, "sufficient": True},
            {"candidate": 8, "fitFailed": False, "rSelection": 0.01, "rValidation": 0.01, "rOod": 0.02,
             "oracleCalls": 100, "sufficient": True},
        ],
        "inferredOrder": 6,
        "verdict": "inferred-order",
        "upperRegionNonIdentifiable": True,
        "nullControl": None,
    }
    report.update(overrides)
    return report


class SelectionRuleTests(unittest.TestCase):
    def test_first_sufficient_rule(self):
        fits = _base_report()["fits"]
        result = _recompute_selection(fits, 0.05, 1)
        self.assertEqual(result["inferredOrder"], 6)
        self.assertEqual(result["verdict"], "inferred-order")
        self.assertTrue(result["upperRegionNonIdentifiable"])
        self.assertEqual(result["afterFirstSufficient"], 1)

    def test_no_sufficient_order(self):
        fits = [
            {"candidate": 4, "fitFailed": False, "rSelection": 0.5, "sufficient": False},
            {"candidate": 6, "fitFailed": True, "sufficient": False},
        ]
        result = _recompute_selection(fits, 0.05, 1)
        self.assertIsNone(result["inferredOrder"])
        self.assertEqual(result["verdict"], "no-sufficient-order")
        self.assertFalse(result["upperRegionNonIdentifiable"])

    def test_upper_region_identifiable_when_later_r_is_far(self):
        fits = [
            {"candidate": 4, "fitFailed": False, "rSelection": 0.08, "sufficient": False},
            {"candidate": 6, "fitFailed": False, "rSelection": 0.01, "sufficient": True},
            {"candidate": 8, "fitFailed": False, "rSelection": 0.04, "sufficient": True},
        ]
        result = _recompute_selection(fits, 0.05, 1)
        self.assertFalse(result["upperRegionNonIdentifiable"])


class CaseReportTests(unittest.TestCase):
    def test_valid_report_passes(self):
        self.assertEqual(verify_case_report(_base_report()), [])

    def test_synthetic_report_rejected_for_real_lane(self):
        report = _base_report(synthetic=True)
        self.assertTrue(any("synthetic" in e for e in verify_case_report(report)))

    def test_inferred_order_mismatch_detected(self):
        report = _base_report(inferredOrder=4)
        self.assertTrue(any("inferredOrder" in e for e in verify_case_report(report)))

    def test_sufficient_flag_mismatch_detected(self):
        report = _base_report()
        report["fits"][1]["sufficient"] = False
        errors = verify_case_report(report)
        self.assertTrue(any("sufficient flag" in e for e in errors))
        self.assertTrue(any("inferredOrder" in e for e in errors))

    def test_upper_region_mismatch_detected(self):
        report = _base_report(upperRegionNonIdentifiable=False)
        self.assertTrue(any("upper-region" in e for e in verify_case_report(report)))

    def test_extra_steps_violation_detected(self):
        report = _base_report()
        report["fits"].append(
            {"candidate": 10, "fitFailed": False, "rSelection": 0.01, "rValidation": 0.01, "rOod": 0.01,
             "oracleCalls": 10, "sufficient": True}
        )
        report["upperRegionNonIdentifiable"] = True
        self.assertTrue(any("extra ladder steps" in e for e in verify_case_report(report)))

    def test_split_sum_mismatch_detected(self):
        report = _base_report()
        report["pool"]["splits"]["train"] = 61
        self.assertTrue(any("sum to the recorded point count" in e for e in verify_case_report(report)))

    def test_unsorted_ladder_detected(self):
        report = _base_report()
        report["protocol"]["ladder"] = [6, 4]
        self.assertTrue(any("ascending" in e for e in verify_case_report(report)))

    def test_budget_overrun_detected(self):
        report = _base_report()
        report["fits"][0]["oracleCalls"] = 2000
        self.assertTrue(any("exceeds the recorded per-fit budget" in e for e in verify_case_report(report)))

    def test_strace_model_access_detected(self):
        report = _base_report()
        report["fits"][0]["straceModelAccessRows"] = ["/models--mistral"]
        self.assertTrue(any("strace" in e for e in verify_case_report(report)))


class FitDirTests(unittest.TestCase):
    def _write_fit(self, root: Path, candidate: int, *, atom_count=None, unit_count=None, sha=None, rank=None):
        fit_dir = root / "fits" / f"m{candidate}"
        fit_dir.mkdir(parents=True, exist_ok=True)
        rank = rank if rank is not None else 2 * candidate
        atom_count = atom_count if atom_count is not None else candidate
        basis = np.eye(rank, 8)
        gates = np.zeros((atom_count, rank))
        ups = np.zeros((atom_count, rank))
        scales = np.ones(atom_count)
        frozen_path = fit_dir / "frozen-recovery.npz"
        np.savez_compressed(
            frozen_path,
            basis=basis,
            gates=gates,
            ups=ups,
            scales=scales,
            bias=np.zeros(1),
            metadata=json.dumps(
                {
                    "kind": "blind-swiglu-fit-snapshot",
                    "dimension": 8,
                    "activeRank": rank,
                    "atomCount": atom_count,
                    "solverAccess": "oracle-only",
                    "weightsUsedBySolver": False,
                }
            ),
        )
        solver_report = {
            "schemaVersion": 1,
            "kind": "blind-swiglu-frozen-recovery",
            "summary": {
                "pass": True,
                "dimension": 8,
                "unitCount": unit_count,
                "unknownOrder": True,
                "activeRank": rank,
                "atomCount": atom_count,
                "rawOracleCalls": 10,
                "metrics": {"mae": 0.0, "mse": 0.0, "r2": 1.0, "worstError": 0.0},
            },
            "frozenRecovery": {
                "path": str(frozen_path),
                "sha256": sha if sha is not None else sha256_file(frozen_path),
                "frozenAfterValidation": True,
            },
            "accessAudit": {
                "solverAccess": "opaque-eval-only-oracle",
                "weightsUsedBySolver": False,
                "hooksUsedBySolver": False,
                "sourceIdsUsedBySolver": False,
                "teacherManifestFieldsObserved": [],
                "truthCapabilityAvailable": False,
                "truthRequestCount": 0,
                "fitSnapshotUsedBySolver": False,
            },
        }
        (fit_dir / "solver-report.json").write_text(json.dumps(solver_report), encoding="utf-8")
        (fit_dir / "oracle-witness.json").write_text(
            json.dumps({"status": "complete", "queryCount": 10}), encoding="utf-8"
        )

    def test_consistent_fit_dir_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fit(root, 4)
            report = _base_report()
            report["fits"] = [
                {"candidate": 4, "fitFailed": False, "rSelection": 0.08, "rValidation": 0.09, "rOod": 0.1,
                 "oracleCalls": 10, "sufficient": False}
            ]
            report["inferredOrder"] = None
            report["verdict"] = "no-sufficient-order"
            report["upperRegionNonIdentifiable"] = False
            self.assertEqual(verify_fit_dirs(root, report), [])

    def test_atom_count_mismatch_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fit(root, 4, atom_count=5)
            report = _base_report()
            report["fits"] = [
                {"candidate": 4, "fitFailed": False, "rSelection": 0.08, "rValidation": 0.09, "rOod": 0.1,
                 "oracleCalls": 10, "sufficient": False}
            ]
            report["inferredOrder"] = None
            report["verdict"] = "no-sufficient-order"
            report["upperRegionNonIdentifiable"] = False
            self.assertTrue(any("atom count" in e for e in verify_fit_dirs(root, report)))

    def test_unit_count_leak_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fit(root, 4, unit_count=6)
            report = _base_report()
            report["fits"] = [
                {"candidate": 4, "fitFailed": False, "rSelection": 0.08, "rValidation": 0.09, "rOod": 0.1,
                 "oracleCalls": 10, "sufficient": False}
            ]
            report["inferredOrder"] = None
            report["verdict"] = "no-sufficient-order"
            report["upperRegionNonIdentifiable"] = False
            self.assertTrue(any("leaked unitCount" in e for e in verify_fit_dirs(root, report)))

    def test_frozen_sha_mismatch_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fit(root, 4, sha="0" * 64)
            report = _base_report()
            report["fits"] = [
                {"candidate": 4, "fitFailed": False, "rSelection": 0.08, "rValidation": 0.09, "rOod": 0.1,
                 "oracleCalls": 10, "sufficient": False}
            ]
            report["inferredOrder"] = None
            report["verdict"] = "no-sufficient-order"
            report["upperRegionNonIdentifiable"] = False
            self.assertTrue(any("sha256" in e for e in verify_fit_dirs(root, report)))


class TeacherSideTests(unittest.TestCase):
    def test_grade_recompute_matches(self):
        rng = np.random.default_rng(11)
        dimension, rank, atoms = 8, 6, 3
        basis = np.eye(rank, dimension)
        gates = rng.normal(size=(atoms, rank))
        ups = rng.normal(size=(atoms, rank))
        scales = rng.normal(size=(atoms,))
        fit = {"gates": gates, "ups": ups, "scales": scales, "bias": 0.0}
        gates_full = gates @ basis
        ups_full = ups @ basis

        def f(x):
            gate_pre = x @ gates_full.T
            up_pre = x @ ups_full.T
            return (gate_pre / (1.0 + np.exp(-gate_pre)) * up_pre) @ scales

        def f_hat(x):
            coords = x @ basis.T
            activations = coords @ gates.T
            activated = activations / (1.0 + np.exp(-activations))
            return (activated * (coords @ ups.T)) @ scales

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "teacher-side").mkdir()
            selection_x = rng.normal(size=(50, dimension))
            validation_x = rng.normal(size=(40, dimension))
            ood_x = rng.normal(size=(30, dimension))
            np.savez_compressed(
                root / "teacher-side" / "eval-responses.npz",
                selection_x=selection_x,
                selection_f=f(selection_x),
                validation_x=validation_x,
                validation_f=f(validation_x),
                ood_x=ood_x,
                ood_f=f(ood_x),
            )
            (root / "fits" / "m3").mkdir(parents=True)
            np.savez_compressed(
                root / "fits" / "m3" / "frozen-recovery.npz",
                basis=basis,
                gates=gates,
                ups=ups,
                scales=scales,
                bias=np.zeros(1),
                metadata=json.dumps({"atomCount": atoms, "activeRank": rank}),
            )
            report = _base_report()
            report["fits"] = [
                {
                    "candidate": 3,
                    "fitFailed": False,
                    "rSelection": relative_mae(f(selection_x), f_hat(selection_x)),
                    "rValidation": relative_mae(f(validation_x), f_hat(validation_x)),
                    "rOod": relative_mae(f(ood_x), f_hat(ood_x)),
                    "oracleCalls": 10,
                    "sufficient": True,
                }
            ]
            report["inferredOrder"] = 3
            self.assertEqual(verify_teacher_side(root, report), [])

    def test_grade_recompute_mismatch_detected(self):
        rng = np.random.default_rng(13)
        dimension = 8
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "teacher-side").mkdir()
            x = rng.normal(size=(20, dimension))
            np.savez_compressed(
                root / "teacher-side" / "eval-responses.npz",
                selection_x=x,
                selection_f=rng.normal(size=20),
                validation_x=x,
                validation_f=rng.normal(size=20),
                ood_x=x,
                ood_f=rng.normal(size=20),
            )
            (root / "fits" / "m3").mkdir(parents=True)
            np.savez_compressed(
                root / "fits" / "m3" / "frozen-recovery.npz",
                basis=np.eye(6, dimension),
                gates=np.zeros((3, 6)),
                ups=np.zeros((3, 6)),
                scales=np.ones(3),
                bias=np.zeros(1),
                metadata=json.dumps({"atomCount": 3, "activeRank": 6}),
            )
            report = _base_report()
            report["fits"] = [
                {"candidate": 3, "fitFailed": False, "rSelection": 0.01, "rValidation": 0.01,
                 "rOod": 0.01, "oracleCalls": 10, "sufficient": True}
            ]
            errors = verify_teacher_side(root, report)
            self.assertTrue(any("rSelection" in e or "rValidation" in e or "rOod" in e for e in errors))


class PoolTests(unittest.TestCase):
    def test_pool_provenance(self):
        rng = np.random.default_rng(17)
        x = rng.normal(size=(100, 8)).astype(np.float32)
        train = np.arange(60)
        selection = np.arange(60, 80)
        validation = np.arange(80, 100)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pool.npz"
            np.savez_compressed(path, x=x, train=train, selection=selection, validation=validation)
            report = _base_report()
            report["pool"]["sha256"] = sha256_file(path)
            self.assertEqual(verify_pool(path, report), [])

    def test_pool_sha_mismatch_detected(self):
        rng = np.random.default_rng(19)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pool.npz"
            np.savez_compressed(path, x=rng.normal(size=(100, 8)))
            report = _base_report()
            self.assertTrue(any("sha256" in e for e in verify_pool(path, report)))

    def test_overlapping_splits_detected(self):
        rng = np.random.default_rng(23)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pool.npz"
            np.savez_compressed(
                path,
                x=rng.normal(size=(100, 8)),
                train=np.arange(60),
                selection=np.arange(59, 79),
                validation=np.arange(80, 100),
            )
            report = _base_report()
            report["pool"]["sha256"] = sha256_file(path)
            self.assertTrue(any("overlap" in e for e in verify_pool(path, report)))


class TeacherSecretTests(unittest.TestCase):
    def test_consistent_secret_passes_even_when_order_wrong(self):
        report = _base_report(inferredOrder=10)
        run_info = {
            "teacherSecret": {"trueM": 6, "unitCount": 6, "outputChannel": 3},
            "timedOut": False,
            "caseVerdict": "inferred-order",
            "inferredOrder": 10,
        }
        manifest = {"model": "m", "layer": 1, "units": [1, 2, 3, 4, 5, 6], "outputChannel": 3}
        self.assertEqual(verify_teacher_secret(run_info, manifest, report), [])

    def test_unit_count_mismatch_detected(self):
        report = _base_report(inferredOrder=6)
        run_info = {"teacherSecret": {"trueM": 6}, "timedOut": False}
        manifest = {"model": "m", "layer": 1, "units": [1, 2, 3], "outputChannel": 3}
        self.assertTrue(any("unit count" in e for e in verify_teacher_secret(run_info, manifest, report)))

    def test_timeout_with_completed_report_detected(self):
        report = _base_report()
        run_info = {"teacherSecret": {"trueM": 6}, "timedOut": True}
        manifest = {"model": "m", "layer": 1, "units": [1] * 6, "outputChannel": 3}
        self.assertTrue(any("timeout" in e for e in verify_teacher_secret(run_info, manifest, report)))


class RelativeMaeTests(unittest.TestCase):
    def test_definition(self):
        actual = np.array([1.0, 2.0, 3.0, 4.0])
        predicted = np.array([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(relative_mae(actual, predicted), 0.0)
        shifted = actual + 1.0
        expected = 1.0 / float(np.std(actual))
        self.assertAlmostEqual(relative_mae(actual, shifted), expected)

    def test_zero_std_is_error(self):
        with self.assertRaises(RuntimeError):
            relative_mae(np.zeros(4), np.zeros(4))


if __name__ == "__main__":
    unittest.main()
