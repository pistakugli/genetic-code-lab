from __future__ import annotations

import inspect
import sys
import tempfile
import unittest
from dataclasses import fields
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.llm_reader.swiglu_observability_experiments import (  # noqa: E402
    DEFAULT_QUERY_BUDGET,
    DEFAULT_SEEDS,
    PROTOCOLS,
    TEACHER_CASES,
    FullModuleSyntheticOracle,
    OracleObservations,
    _dimensions,
    _equivalent_teachers,
    make_synthetic_teacher,
    measure_protocol,
    run_synthetic_observability_study,
)


class SwiGLUObservabilityExperiments(unittest.TestCase):
    def test_measurement_interface_is_teacher_blind_and_budget_exact(self) -> None:
        signature = inspect.signature(measure_protocol)
        parameter_names = set(signature.parameters)
        self.assertEqual(
            parameter_names,
            {
                "oracle_query",
                "protocol",
                "oracle_input_dim",
                "oracle_output_dim",
                "query_budget",
                "seed",
            },
        )
        self.assertFalse(any("teacher" in name or "width" in name for name in parameter_names))
        observation_fields = {item.name for item in fields(OracleObservations)}
        self.assertEqual(
            observation_fields,
            {
                "protocol",
                "training_inputs",
                "training_outputs",
                "heldout_inputs",
                "heldout_outputs",
                "training_queries",
                "heldout_queries",
                "total_queries",
            },
        )
        self.assertFalse(any(
            any(term in name.lower() for term in ("teacher", "weight", "hidden", "unit"))
            for name in observation_fields
        ))
        observed_batches: list[np.ndarray] = []

        def oracle_query(inputs: np.ndarray) -> np.ndarray:
            values = np.asarray(inputs, dtype=np.float64)
            observed_batches.append(values.copy())
            return np.column_stack((values.sum(axis=1), (values**2).sum(axis=1)))

        for protocol in PROTOCOLS:
            observed_batches.clear()
            observations = measure_protocol(
                oracle_query,
                protocol=protocol,
                oracle_input_dim=8,
                oracle_output_dim=2,
                query_budget=12,
                seed=29,
            )
            self.assertEqual(observations.total_queries, 12)
            self.assertEqual(observations.training_queries, 8)
            self.assertEqual(observations.heldout_queries, 4)
            self.assertEqual(sum(len(batch) for batch in observed_batches), 12)
            self.assertEqual(observations.training_inputs.shape, (8, 8))
            self.assertEqual(observations.training_outputs.shape, (8, 2))
            self.assertEqual(observations.heldout_inputs.shape, (4, 8))
            self.assertEqual(observations.heldout_outputs.shape, (4, 2))
            np.testing.assert_array_equal(
                observations.training_outputs[:, 0],
                observations.training_inputs.sum(axis=1),
            )
            np.testing.assert_array_equal(
                observations.heldout_outputs[:, 1],
                (observations.heldout_inputs**2).sum(axis=1),
            )
            self.assertFalse(
                np.array_equal(
                    observations.training_inputs[:4],
                    observations.heldout_inputs,
                )
            )

    def test_synthetic_oracle_counts_full_module_queries(self) -> None:
        teacher = make_synthetic_teacher(
            "generic",
            input_dim=8,
            output_dim=6,
            width=5,
            seed=100,
        )
        oracle = FullModuleSyntheticOracle(teacher)
        observations = measure_protocol(
            oracle.query,
            protocol="iid_dense",
            oracle_input_dim=8,
            oracle_output_dim=6,
            query_budget=12,
            seed=7,
        )
        self.assertEqual(oracle.query_count, 12)
        np.testing.assert_allclose(
            observations.training_outputs,
            teacher.forward(observations.training_inputs),
            rtol=0.0,
            atol=0.0,
        )

    def test_controlled_teacher_cases_and_equivalences(self) -> None:
        expected_seeds = (11, 22, 33)
        for seed_index, seed in enumerate(expected_seeds):
            input_dim, output_dim, width = _dimensions(seed_index)
            for case_index, case in enumerate(TEACHER_CASES):
                teacher = make_synthetic_teacher(
                    case,
                    input_dim=input_dim,
                    output_dim=output_dim,
                    width=width,
                    seed=seed * 101 + case_index,
                )
                if case == "rank1_output":
                    self.assertEqual(np.linalg.matrix_rank(teacher.down), 1)
                elif case == "rank3_output":
                    self.assertEqual(
                        np.linalg.matrix_rank(teacher.down),
                        min(3, output_dim, width),
                    )
                elif case == "duplicate_gate":
                    np.testing.assert_array_equal(teacher.gate[0], teacher.gate[1])
                elif case == "null_component":
                    self.assertEqual(np.linalg.norm(teacher.down[:, -1]), 0.0)
                elif case == "weak_components":
                    self.assertLess(
                        np.linalg.norm(teacher.down[:, -1]),
                        1e-5,
                    )

                variants = _equivalent_teachers(teacher, seed=seed + 1)
                self.assertIn("permutation_reciprocal_scaling", variants)
                if case == "duplicate_gate":
                    self.assertIn("duplicate_gate_gl2_change_of_basis", variants)
                if case == "null_component":
                    self.assertIn("remove_zero_output_component", variants)
                test_inputs = np.random.default_rng(seed + case_index).normal(
                    size=(37, input_dim)
                )
                expected = teacher.forward(test_inputs)
                for equivalent in variants.values():
                    np.testing.assert_allclose(
                        equivalent.forward(test_inputs),
                        expected,
                        rtol=1e-13,
                        atol=1e-13,
                    )

    def test_study_emits_matched_raw_artifacts_and_nonself_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            report = run_synthetic_observability_study(
                Path(temporary) / "study",
                seeds=DEFAULT_SEEDS,
                query_budget=12,
            )
            expected_points = (
                3 * len(TEACHER_CASES) * len(PROTOCOLS) * 12
            )
            self.assertEqual(
                report["queryAccounting"]["oraclePoints"], expected_points
            )
            self.assertEqual(
                report["queryAccounting"]["expectedOraclePoints"],
                expected_points,
            )
            self.assertEqual(
                report["experimentDesign"]["newQwenOraclePointsSpent"], 0
            )
            self.assertEqual(
                report["qwenPilotDecision"]["syntheticInformationCriterionMet"],
                any(
                    item["syntheticInformationCriterionMet"]
                    for item in report["posthocComparisons"].values()
                ),
            )
            self.assertFalse(report["qwenPilotDecision"]["runQwenPilot"])
            self.assertFalse(
                report["qwenPilotDecision"]["oracleOnlyRecoveryDemonstrated"]
            )
            self.assertEqual(
                report["experimentDesign"]["newQwenOraclePointsSpent"], 0
            )
            self.assertEqual(len(report["cases"]), 3 * len(TEACHER_CASES))
            self.assertNotIn("reportSha256", report)
            for row in report["cases"]:
                self.assertEqual(len(row["protocols"]), len(PROTOCOLS))
                for protocol_row in row["protocols"]:
                    self.assertEqual(
                        protocol_row["queryAccounting"]["oraclePoints"], 12
                    )
                    self.assertEqual(
                        protocol_row["outputReadoutPostprocessingControl"][
                            "additionalOraclePoints"
                        ],
                        0,
                    )
                environment_row = next(
                    item
                    for item in row["protocols"]
                    if item["protocol"] == "activation_environments"
                )
                for cohort in environment_row[
                    "activationEnvironmentSubcohortsPosthoc"
                ].values():
                    self.assertEqual(
                        cohort["heldoutPoints"],
                        cohort["iidMatchedHeldoutPoints"],
                    )


if __name__ == "__main__":
    result = unittest.TextTestRunner().run(
        unittest.defaultTestLoader.loadTestsFromTestCase(
            SwiGLUObservabilityExperiments
        )
    )
    if not result.wasSuccessful():
        raise SystemExit(1)
    print("synthetic observability tests passed")
