#!/usr/bin/env python3
from __future__ import annotations

import unittest
import sys
import tempfile
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import tools.llm_reader.blind_swiglu_oracle_decompiler as swiglu_module
from tools.llm_reader.blind_swiglu_oracle_decompiler import (
    audit_swiglu_source_cancellation,
    apply_asymptotic_support_fallback,
    asymptotic_stitch_query_upper_bound,
    build_oracle_ready_payload,
    choose_gate_stitch_probe_direction,
    cluster_repeated_gate_normals,
    detect_piecewise_quadratic_boundaries,
    detect_swiglu_cancellation_pairs,
    fit_origin_hyperplane_normal,
    gate_norm_calibrate_swiglu_candidate,
    make_asymptotic_cold_start_scaffold,
    trust_region_structure_refine_swiglu_candidate,
    validate_opaque_oracle_metadata,
    maximum_weight_assignment,
    OracleBudgetExceeded,
    OracleClient,
    predict_swiglu_cubic_directional_derivatives,
    richardson_extrapolate_cubic_scores,
    richardson_extrapolate_symmetric_gradients,
    residual_block_refit_cubic_weights,
    recover_up_direction_from_hessian_gate_action,
    load_fit_snapshot,
    save_fit_snapshot,
    sample_active_region_inputs,
    select_residual_block_atoms,
    solve_bidirectional_gate_orientation_start,
    sequential_cancellation_pair_refit_swiglu_candidate,
    estimate_active_symmetric_richardson_gradients_at_coords,
    estimate_symmetric_multiscale_coordinate_gradients,
    estimate_active_forward_gradients_at_coords,
    symmetric_regression_slopes,
    validation_full_tail_polish_cubic_trigger,
    validation_gradient_polish_weights,
)
from tools.llm_reader.blind_swiglu_solver_worker import build_worker_parser
from tools.llm_reader.isolated_swiglu_cold_start import (
    classify_solver_file_access,
    find_solver_source_locator_matches,
)
from tools.llm_reader.swiglu_oracle_server import validate_eval_request


def diagnostic(scores: list[float]) -> dict[str, object]:
    return {
        "topAtoms": [
            {"atom": index, "tailResponsibilityScore": score}
            for index, score in enumerate(scores)
        ]
    }


class ResidualBlockSelectionTest(unittest.TestCase):
    def test_solver_source_locator_audit_rejects_output_path_leak(self) -> None:
        observed = find_solver_source_locator_matches(
            [
                "python",
                "blind_swiglu_solver_worker.py",
                "--output-json",
                "/repo/benchmarks/mistral-layer15/report.json",
            ],
            model="unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
            layer=15,
        )

        self.assertIn("mistral", observed)
        self.assertIn("layer15", observed)

    def test_solver_source_locator_audit_accepts_opaque_work_path(self) -> None:
        observed = find_solver_source_locator_matches(
            [
                "python",
                "blind_swiglu_solver_worker.py",
                "--output-json",
                "/tmp/gcl-solver-deadbeef/solver-report.json",
            ],
            model="unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
            layer=15,
        )

        self.assertEqual(observed, [])

    def test_solver_file_access_classifier_separates_package_metadata_from_weights(self) -> None:
        control = Path("/tmp/private-orchestrator.py")
        observed = classify_solver_file_access(
            "\n".join([
                '1 openat(AT_FDCWD, "/venv/site-packages/huggingface_hub-1.0.dist-info/entry_points.txt", O_RDONLY) = 3',
                '1 openat(AT_FDCWD, "/cache/hub/models--org--model/snapshots/abc/model.safetensors", O_RDONLY) = 4',
                f'1 openat(AT_FDCWD, "{control}", O_RDONLY) = 5',
            ]),
            sensitive_control_paths=[control],
        )

        self.assertEqual(observed["lineCount"], 3)
        self.assertEqual(len(observed["packageMetadataRows"]), 1)
        self.assertEqual(len(observed["modelAccessRows"]), 1)
        self.assertEqual(len(observed["controlAccessRows"]), 1)

    def test_solver_file_access_classifier_accepts_empty_model_access(self) -> None:
        observed = classify_solver_file_access(
            '1 openat(AT_FDCWD, "/repo/tools/llm_reader/blind_swiglu_oracle_decompiler.py", O_RDONLY) = 3',
            sensitive_control_paths=[Path("/tmp/private-orchestrator.py")],
        )

        self.assertEqual(observed["modelAccessRows"], [])
        self.assertEqual(observed["controlAccessRows"], [])

    def test_isolated_worker_parser_rejects_teacher_manifest_flags(self) -> None:
        base = [
            "--endpoint", "/tmp/oracle.sock",
            "--dimension", "4096",
            "--atom-count", "16",
            "--budget", "1500000",
            "--frozen-output", "/tmp/frozen.npz",
            "--output-json", "/tmp/report.json",
            "--output-md", "/tmp/report.md",
        ]
        parser = build_worker_parser()
        parser.parse_args(base)
        for forbidden in ("--model", "--layer", "--units", "--truth"):
            with patch("sys.stderr"), self.assertRaises(SystemExit):
                parser.parse_args(base + [forbidden, "canary"])

    def test_binary_eval_protocol_rejects_path_injection_fields(self) -> None:
        payload = np.zeros((2, 3), dtype="<f4").tobytes()
        header = {
            "type": "eval",
            "requestId": 0,
            "count": 2,
            "dimension": 3,
            "dtype": "float32-le",
            "payloadBytes": len(payload),
        }
        count, observed = validate_eval_request(
            header,
            payload,
            expected_request_id=0,
            dimension=3,
        )
        self.assertEqual(count, 2)
        self.assertEqual(observed.shape, (2, 3))
        for poisoned_key in ("inputPath", "outputPath", "truthPath"):
            poisoned = dict(header)
            poisoned[poisoned_key] = "/home/kugli/.cache/huggingface/canary"
            with self.assertRaisesRegex(RuntimeError, "unexpected eval header fields"):
                validate_eval_request(
                    poisoned,
                    payload,
                    expected_request_id=0,
                    dimension=3,
                )

    def test_opaque_eval_only_oracle_ready_hides_teacher_manifest(self) -> None:
        args = SimpleNamespace(
            oracle_quantization_step=1e-6,
            oracle_eval_only=True,
            oracle_ready_mode="opaque",
            model="private/model",
            layer=15,
            unit=7885,
            device="cuda",
        )
        observed = build_oracle_ready_payload(
            args=args,
            model_type="mistral",
            units=[7885, 2037],
            hidden_size=4096,
            intermediate_size=14336,
            oracle_mode="down-proj-output-channel",
            output_channel=2048,
        )

        self.assertEqual(observed["hiddenSize"], 4096)
        self.assertEqual(observed["unitCount"], 2)
        self.assertEqual(observed["protocolCapabilities"], {"eval": True, "truth": False})
        for forbidden in (
            "model", "modelType", "layer", "unit", "units",
            "intermediateSize", "device", "oracleMode", "outputChannel",
        ):
            self.assertNotIn(forbidden, observed)
        validate_opaque_oracle_metadata(observed)

        poisoned = dict(observed)
        poisoned["units"] = [7885, 2037]
        with self.assertRaisesRegex(RuntimeError, "exposed teacher manifest"):
            validate_opaque_oracle_metadata(poisoned)

    def test_oracle_truth_is_locked_until_recovery_freeze(self) -> None:
        oracle = OracleClient.__new__(OracleClient)
        oracle.query_count = 123
        oracle._truth_unlocked = False
        oracle.truth_unlock_stage = None
        oracle.truth_unlock_query_count = None
        oracle.truth_request_count = 0

        with self.assertRaisesRegex(RuntimeError, "truth is locked"):
            oracle.truth()

        oracle.unlock_posthoc_truth("final-validation-recovery-frozen")
        self.assertTrue(oracle._truth_unlocked)
        self.assertEqual(oracle.truth_unlock_stage, "final-validation-recovery-frozen")
        self.assertEqual(oracle.truth_unlock_query_count, 123)
        with self.assertRaisesRegex(RuntimeError, "already unlocked"):
            oracle.unlock_posthoc_truth("second-unlock")

    def test_asymptotic_query_bound_is_exact_for_16_atom_32_rank_profile(self) -> None:
        self.assertEqual(
            asymptotic_stitch_query_upper_bound(24, 1537, 16, 32, 5),
            103_192,
        )

    def test_asymptotic_cold_start_scaffold_is_constant_and_optimizer_free(self) -> None:
        targets = np.array([-1.0, 0.5, 1.5, 2.0], dtype=np.float64)
        observed = make_asymptotic_cold_start_scaffold(3, 4, targets)

        self.assertEqual(observed["startKind"], "asymptotic-cold-start-scaffold")
        self.assertEqual(observed["restartCount"], 0)
        self.assertEqual(observed["randomRestartCount"], 0)
        np.testing.assert_allclose(observed["scales"], np.zeros(3))
        np.testing.assert_allclose(
            swiglu_module.predict_swiglu(np.zeros((targets.size, 4)), observed),
            np.full(targets.size, np.mean(targets)),
        )

    def test_bidirectional_gate_orientation_recovers_signed_swiglu_structure(self) -> None:
        rng = np.random.default_rng(260730)
        coords = rng.normal(0.0, 0.8, size=(2048, 3))
        gate_norm = 0.35
        canonical_directions = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        true_gates = canonical_directions * np.array([-gate_norm, gate_norm])[:, None]
        true_ups = np.array([
            [0.0, 0.8, 0.6],
            [0.6, 0.0, -0.8],
        ])
        true_scales = np.array([0.7, -0.45])
        targets = (
            swiglu_module.silu_np(coords @ true_gates.T)
            * (coords @ true_ups.T)
            * true_scales[None, :]
        ).sum(axis=1) + 0.025

        solved = solve_bidirectional_gate_orientation_start(
            coords,
            targets,
            canonical_directions,
            gate_norm,
            sample_limit=1024,
        )

        self.assertIsNotNone(solved)
        assert solved is not None
        gates, ups, scales, bias, _mae = solved
        prediction = swiglu_module.predict_swiglu(coords, {
            "gates": gates,
            "ups": ups,
            "scales": scales,
            "bias": bias,
        })
        metrics = swiglu_module.regression_metrics(targets, prediction)
        self.assertGreater(float(np.dot(gates[0], true_gates[0])), 0.0)
        self.assertGreater(float(np.dot(gates[1], true_gates[1])), 0.0)
        self.assertGreater(metrics["r2"], 0.999999)
        self.assertLess(metrics["mae"], 1e-8)

    def test_asymptotic_support_court_falls_back_only_for_weak_atoms(self) -> None:
        recovered_gates = np.eye(3)
        recovered_ups: list[np.ndarray | None] = [
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            None,
        ]
        initial_gates = np.roll(np.eye(3), 1, axis=1)
        initial_ups = -np.eye(3)
        selected_gates, selected_ups, fallback_atoms = apply_asymptotic_support_fallback(
            recovered_gates,
            recovered_ups,
            np.array([8, 3, 7]),
            np.array([5, 5, 1]),
            initial_gates,
            initial_ups,
            4,
            3,
        )

        self.assertEqual(fallback_atoms, [1, 2])
        np.testing.assert_allclose(selected_gates[0], recovered_gates[0])
        np.testing.assert_allclose(selected_ups[0], recovered_ups[0])
        np.testing.assert_allclose(selected_gates[1:], initial_gates[1:])
        np.testing.assert_allclose(np.vstack(selected_ups[1:]), initial_ups[1:])

    def test_trust_region_structure_refine_corrects_nearby_swiglu_directions(self) -> None:
        rng = np.random.default_rng(260727)
        train = rng.normal(size=(1024, 3))
        selection = rng.normal(size=(512, 3))
        true_gates = np.array([
            [1.8, 0.0, 0.0],
            [0.0, 0.75, 0.0],
        ])
        true_ups = np.array([
            [0.0, 0.6, 0.8],
            [0.8, 0.0, -0.6],
        ])
        true_scales = np.array([0.35, -0.22])
        bias = -0.02

        def teacher(values: np.ndarray) -> np.ndarray:
            return np.sum(
                swiglu_module.silu_np(values @ true_gates.T)
                * (values @ true_ups.T)
                * true_scales[None, :],
                axis=1,
            ) + bias

        candidate_gate_dirs = true_gates / np.linalg.norm(true_gates, axis=1, keepdims=True)
        candidate_gate_dirs = candidate_gate_dirs + np.array([
            [0.0, 0.08, -0.05],
            [0.06, 0.0, 0.05],
        ])
        candidate_gate_dirs /= np.linalg.norm(candidate_gate_dirs, axis=1, keepdims=True)
        candidate_ups = true_ups + np.array([
            [0.07, -0.04, 0.0],
            [-0.03, 0.06, 0.0],
        ])
        candidate_ups /= np.linalg.norm(candidate_ups, axis=1, keepdims=True)
        candidate = {
            "gates": candidate_gate_dirs * np.array([1.4, 1.1])[:, None],
            "ups": candidate_ups,
            "scales": true_scales.copy(),
            "bias": bias,
        }
        before = swiglu_module.regression_metrics(
            teacher(selection),
            swiglu_module.predict_swiglu(selection, candidate),
        )
        with patch.dict(os.environ, {
            "GCL_BLIND_SWIGLU_STRUCTURE_REFINE_STEPS": "120",
            "GCL_BLIND_SWIGLU_STRUCTURE_REFINE_RESTARTS": "2",
            "GCL_BLIND_SWIGLU_STRUCTURE_REFINE_MAX_TANGENT": "0.25",
        }, clear=False):
            observed = trust_region_structure_refine_swiglu_candidate(
                train,
                teacher(train),
                candidate,
                selection,
                teacher(selection),
            )
        after = swiglu_module.regression_metrics(
            teacher(selection),
            swiglu_module.predict_swiglu(selection, observed),
        )

        self.assertTrue(observed["structureTrustRegionAccepted"])
        self.assertLess(after["mae"], before["mae"] * 1e-3)
        self.assertGreater(after["r2"], 0.999999)

    def test_gate_norm_variable_projection_recovers_fixed_swiglu_structure(self) -> None:
        rng = np.random.default_rng(260726)
        train = rng.normal(size=(768, 3))
        selection = rng.normal(size=(384, 3))
        gate_dirs = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        true_gate_norms = np.array([2.2, 0.65])
        ups = np.array([
            [0.0, 0.6, 0.8],
            [0.8, 0.0, -0.6],
        ])
        scales = np.array([0.4, -0.25])
        bias = 0.03

        def teacher(values: np.ndarray) -> np.ndarray:
            gates = gate_dirs * true_gate_norms[:, None]
            return np.sum(
                swiglu_module.silu_np(values @ gates.T)
                * (values @ ups.T)
                * scales[None, :],
                axis=1,
            ) + bias

        candidate = {
            "gates": gate_dirs * np.array([0.55, 2.4])[:, None],
            "ups": ups.copy(),
            "scales": scales.copy(),
            "bias": bias,
        }
        before = swiglu_module.regression_metrics(
            teacher(train),
            swiglu_module.predict_swiglu(train, candidate),
        )
        with patch.dict(os.environ, {
            "GCL_BLIND_SWIGLU_GATE_NORM_CALIBRATION_STEPS": "100",
            "GCL_BLIND_SWIGLU_GATE_NORM_CALIBRATION_RESTARTS": "3",
            "GCL_BLIND_SWIGLU_GATE_NORM_CALIBRATION_LR": "0.8",
            "GCL_BLIND_SWIGLU_GATE_NORM_CALIBRATION_MAX_NORM_RATIO": "10",
        }, clear=False):
            observed = gate_norm_calibrate_swiglu_candidate(
                train,
                teacher(train),
                candidate,
                selection,
                teacher(selection),
            )
        after = swiglu_module.regression_metrics(
            teacher(selection),
            swiglu_module.predict_swiglu(selection, observed),
        )

        self.assertTrue(observed["gateNormCalibrated"])
        self.assertEqual(observed["gateNormCalibrationMethod"], "bounded-variable-projection-lbfgs")
        self.assertLess(after["mae"], before["mae"] * 1e-3)
        self.assertGreater(after["r2"], 0.999999)

    def test_active_symmetric_richardson_gradients_remove_coordinate_step_bias(self) -> None:
        class Oracle:
            def __init__(self) -> None:
                self.query_count = 0

            def eval(self, inputs: np.ndarray, *, stage: str) -> np.ndarray:
                self.query_count += inputs.shape[0]
                values = np.asarray(inputs, dtype=np.float64)
                return (
                    values[:, 0] * values[:, 1] ** 3
                    + 0.5 * values[:, 1]
                    + 0.25 * values[:, 2] ** 3
                )

        oracle = Oracle()
        coords = np.array([
            [0.7, -0.2, 0.4],
            [-0.3, 0.6, -0.5],
            [0.2, 0.1, 0.8],
        ], dtype=np.float64)
        observed = estimate_active_symmetric_richardson_gradients_at_coords(
            oracle,
            np.eye(3, dtype=np.float64),
            coords,
            [0.1, 0.25],
            stage="unit-richardson",
            chunk_size=2,
        )
        expected = np.column_stack([
            coords[:, 1] ** 3,
            3.0 * coords[:, 0] * coords[:, 1] ** 2 + 0.5,
            0.75 * coords[:, 2] ** 2,
        ])

        np.testing.assert_allclose(observed, expected, rtol=2e-6, atol=2e-7)
        self.assertEqual(oracle.query_count, 2 * 2 * coords.shape[0] * coords.shape[1])

    def test_hessian_gate_action_recovers_up_direction(self) -> None:
        gate = np.array([0.4, -0.2, 0.7, -0.55])
        gate /= np.linalg.norm(gate)
        up = np.array([-0.3, 0.8, 0.25, 0.45])
        up /= np.linalg.norm(up)
        hessian_jump = np.outer(gate, up) + np.outer(up, gate)

        observed = recover_up_direction_from_hessian_gate_action(
            gate,
            hessian_jump @ gate,
        )

        self.assertIsNotNone(observed)
        assert observed is not None
        self.assertGreater(abs(float(observed @ up)), 0.999999999)

    def test_hessian_gate_action_recovers_up_direction_on_a_quiet_block(self) -> None:
        """The test above uses unit vectors, so the action has norm of order one and sits
        far above canonical_direction's absolute 1e-9 rejection floor. That is the regime
        where the bug never existed. Real blocks differ in output amplitude by a factor of
        237, and per-atom contributions by a further 105, so a weak atom on a quiet block
        produces an action many orders smaller. Scaled down by 1e-8 the old code returned
        None and the atom was discarded as unrecoverable."""
        gate = np.array([0.4, -0.2, 0.7, -0.55])
        gate /= np.linalg.norm(gate)
        up = np.array([-0.3, 0.8, 0.25, 0.45])
        up /= np.linalg.norm(up)
        hessian_jump = (np.outer(gate, up) + np.outer(up, gate)) * 1e-8

        observed = recover_up_direction_from_hessian_gate_action(gate, hessian_jump @ gate)

        self.assertIsNotNone(observed)
        assert observed is not None
        self.assertGreater(abs(float(observed @ up)), 0.999999999)

    def test_hessian_gate_action_rejects_a_degenerate_action(self) -> None:
        """The amplitude fix must not turn the degeneracy test off.

        Writing this test corrected a wrong assumption in the fix's own comment. For
        `effective_up = action - 0.5 (gate.action) gate` with a unit gate, an action of
        `c * gate` yields `0.5 c gate`, which is not a collapse. The map only degenerates
        when the action itself vanishes, so that is what the guard has to catch, and it
        has to catch it independently of amplitude."""
        gate = np.array([0.4, -0.2, 0.7, -0.55])
        gate /= np.linalg.norm(gate)

        self.assertIsNone(recover_up_direction_from_hessian_gate_action(gate, np.zeros(4)))
        self.assertIsNone(
            recover_up_direction_from_hessian_gate_action(gate, np.array([np.nan, 0.0, 0.0, 0.0]))
        )

    def test_hessian_gate_action_direction_is_scale_invariant(self) -> None:
        """The recovered direction must not depend on how loud the block is. This is the
        property the amplitude fix is really claiming, so it is asserted directly rather
        than inferred from one amplitude passing."""
        gate = np.array([0.4, -0.2, 0.7, -0.55])
        gate /= np.linalg.norm(gate)
        up = np.array([-0.3, 0.8, 0.25, 0.45])
        up /= np.linalg.norm(up)
        action = (np.outer(gate, up) + np.outer(up, gate)) @ gate

        loud = recover_up_direction_from_hessian_gate_action(gate, action)
        quiet = recover_up_direction_from_hessian_gate_action(gate, action * 1e-9)

        self.assertIsNotNone(loud)
        self.assertIsNotNone(quiet)
        assert loud is not None and quiet is not None
        np.testing.assert_allclose(loud, quiet, rtol=1e-9, atol=1e-12)

    def test_gate_normal_clustering_survives_amplitude_scaled_weights(self) -> None:
        """cluster_repeated_gate_normals sums weights that carry the block's amplitude and
        hands the sum to canonical_direction, which rejects below an absolute 1e-9. On a
        quiet block a valid cluster was discarded as centre-undefined. Weights of 1e-10
        reproduce that regime; the directions are identical to the loud case, so the only
        thing under test is whether amplitude alone can destroy the clustering."""
        rng = np.random.default_rng(1701)
        directions = []
        weights = []
        for atom in range(3):
            axis = np.zeros(4)
            axis[atom] = 1.0
            for repeat in range(7):
                jitter = rng.normal(0.0, 0.01, size=4)
                direction = axis + jitter
                directions.append(direction / np.linalg.norm(direction))
                weights.append((1.0 + 0.1 * repeat) * 1e-10)

        observed = cluster_repeated_gate_normals(
            np.vstack(directions),
            np.asarray(weights, dtype=np.float64),
            3,
        )

        self.assertEqual(len(observed), 3)
        for cluster in observed:
            self.assertEqual(cluster["support"], 7)

    def test_repeated_gate_normal_clustering_recovers_signed_direction_families(self) -> None:
        rng = np.random.default_rng(260729)
        truth = np.linalg.qr(rng.normal(size=(6, 6)))[0].T[:4]
        observations = []
        strengths = []
        for direction in truth:
            for repeat in range(7):
                signed = direction if repeat % 2 == 0 else -direction
                noisy = signed + rng.normal(0.0, 0.01, size=direction.size)
                observations.append(noisy / np.linalg.norm(noisy))
                strengths.append(1.0 + 0.1 * repeat)

        clusters = cluster_repeated_gate_normals(
            np.vstack(observations),
            np.asarray(strengths),
            4,
        )

        self.assertEqual(len(clusters), 4)
        recovered = np.vstack([item["direction"] for item in clusters])
        assignment = maximum_weight_assignment(np.abs(truth @ recovered.T))
        paired = np.abs(np.sum(truth * recovered[assignment], axis=1))
        self.assertTrue(np.all(paired > 0.999))
        self.assertTrue(all(item["support"] == 7 for item in clusters))

    def test_piecewise_quadratic_boundary_detector_recovers_known_gate_crossings(self) -> None:
        coordinates = np.linspace(-8.0, 8.0, 513)
        values = (
            1.5 * np.maximum(0.0, coordinates + 3.0) ** 2
            - 0.75 * np.maximum(0.0, coordinates - 0.5) ** 2
            + 0.4 * np.maximum(0.0, coordinates - 4.0) ** 2
        )

        observed = detect_piecewise_quadratic_boundaries(coordinates, values, 3)

        np.testing.assert_allclose(
            [item["coordinate"] for item in observed],
            [-3.0, 0.5, 4.0],
            atol=coordinates[1] - coordinates[0],
        )
        self.assertGreater(observed[0]["jump"], 0.0)
        self.assertLess(observed[1]["jump"], 0.0)

    def test_origin_hyperplane_fit_recovers_normal_with_outliers(self) -> None:
        rng = np.random.default_rng(260727)
        truth = np.array([0.4, -0.2, 0.7, -0.55])
        truth /= np.linalg.norm(truth)
        raw = rng.normal(size=(256, truth.size))
        points = raw - np.outer(raw @ truth, truth)
        points[:20] += rng.normal(0.0, 0.2, size=(20, truth.size))

        fitted = fit_origin_hyperplane_normal(points)

        self.assertIsNotNone(fitted)
        assert fitted is not None
        direction, residual = fitted
        self.assertGreater(abs(float(direction @ truth)), 0.999)
        self.assertLess(residual, 1e-10)

    def test_gate_stitch_probe_direction_avoids_near_orthogonal_gates(self) -> None:
        gates = np.eye(4)
        direction = choose_gate_stitch_probe_direction(
            np.random.default_rng(260728),
            gates,
            candidate_count=4096,
        )

        self.assertAlmostEqual(float(np.linalg.norm(direction)), 1.0, places=12)
        self.assertGreater(float(np.min(np.abs(gates @ direction))), 0.35)

    def test_active_forward_gradients_at_radial_coords_recover_linear_field(self) -> None:
        class LinearOracle:
            def eval(self, inputs: np.ndarray, *, stage: str = "oracle-eval") -> np.ndarray:
                self.stage = stage
                return np.asarray(inputs, dtype=np.float64) @ np.array([2.0, -3.0, 0.5])

        oracle = LinearOracle()
        basis = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        coords = np.array([
            [4.0, -2.0],
            [-3.0, 7.0],
        ])
        observed = estimate_active_forward_gradients_at_coords(
            oracle,
            basis,
            coords,
            [0.25, 0.5],
            stage="radial-test",
        )

        np.testing.assert_allclose(observed, np.array([[2.0, -3.0], [2.0, -3.0]]), atol=1e-12)
        self.assertEqual(oracle.stage, "radial-test-step-1")

    def test_active_radial_shell_queries_stay_inside_basis_and_cover_multiple_scales(self) -> None:
        basis = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        observed = sample_active_region_inputs(
            np.random.default_rng(260726),
            128,
            basis,
            0.5,
            [0.5, 2.0],
            "active-radial-shell",
        )

        self.assertEqual(observed.shape, (128, 3))
        np.testing.assert_allclose(observed[:, 2], 0.0, atol=0.0)
        norms = np.linalg.norm(observed, axis=1)
        self.assertGreater(float(np.max(norms)), 4.0)
        self.assertLess(float(np.min(norms)), 0.3)

    def test_oracle_client_rejects_over_budget_batch_before_io(self) -> None:
        client = OracleClient.__new__(OracleClient)
        client.max_oracle_calls = 10
        client.query_count = 8
        client._counter = 0
        client.work_dir = Path("/unused")
        with patch.object(np, "save") as save_mock, patch.object(client, "request") as request_mock:
            with self.assertRaises(OracleBudgetExceeded) as raised:
                client.eval(np.zeros((3, 4)), stage="validation")

        self.assertEqual(raised.exception.stage, "validation")
        self.assertEqual(raised.exception.requested, 3)
        save_mock.assert_not_called()
        request_mock.assert_not_called()

    def test_oracle_client_rejects_response_count_mismatch(self) -> None:
        client = OracleClient.__new__(OracleClient)
        client.max_oracle_calls = 10
        client.query_count = 0
        client._counter = 0
        with tempfile.TemporaryDirectory() as temp_dir:
            client.work_dir = Path(temp_dir)
            with patch.object(client, "request", return_value={
                "type": "ok",
                "count": 1,
                "totalCount": 1,
            }):
                with self.assertRaisesRegex(RuntimeError, "response count mismatch"):
                    client.eval(np.zeros((2, 4)), stage="train")
        self.assertEqual(client.query_count, 0)

    def test_residual_block_refit_enables_available_cubic_witness_by_default(self) -> None:
        keys = {
            "GCL_BLIND_SWIGLU_FIT_CUBIC_WEIGHT",
            "GCL_BLIND_SWIGLU_SELECTION_CUBIC_WEIGHT",
            "GCL_BLIND_SWIGLU_RESIDUAL_BLOCK_REFIT_CUBIC_FIT_WEIGHT",
            "GCL_BLIND_SWIGLU_RESIDUAL_BLOCK_REFIT_CUBIC_SELECTION_WEIGHT",
        }
        with patch.dict(os.environ, {}, clear=False):
            for key in keys:
                os.environ.pop(key, None)
            self.assertEqual(residual_block_refit_cubic_weights(True), (1.0, 0.0))
            self.assertEqual(residual_block_refit_cubic_weights(False), (0.0, 0.0))

    def test_asymptotic_structure_method_runs_when_cubic_witness_is_available(self) -> None:
        key = "GCL_BLIND_SWIGLU_VALIDATION_FULL_TAIL_POLISH_ON_CUBIC"
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(key, None)
            self.assertTrue(validation_full_tail_polish_cubic_trigger("asymptotic-gate-stitch", 2048))
            self.assertFalse(validation_full_tail_polish_cubic_trigger("asymptotic-gate-stitch", 0))

    def test_residual_block_refit_cubic_weights_override_global_weights(self) -> None:
        with patch.dict(os.environ, {
            "GCL_BLIND_SWIGLU_FIT_CUBIC_WEIGHT": "0.25",
            "GCL_BLIND_SWIGLU_SELECTION_CUBIC_WEIGHT": "0.01",
            "GCL_BLIND_SWIGLU_RESIDUAL_BLOCK_REFIT_CUBIC_FIT_WEIGHT": "1.5",
            "GCL_BLIND_SWIGLU_RESIDUAL_BLOCK_REFIT_CUBIC_SELECTION_WEIGHT": "0.03",
        }, clear=False):
            self.assertEqual(residual_block_refit_cubic_weights(True), (1.5, 0.03))

    def test_symmetric_richardson_removes_quadratic_step_bias(self) -> None:
        steps = [0.5, 1.25]
        truth = np.array([0.75, -1.5, 2.25])
        bias = np.array([0.4, -0.2, 0.1])
        estimates = np.vstack([truth + step ** 2 * bias for step in steps])

        observed = richardson_extrapolate_symmetric_gradients(estimates, steps)

        np.testing.assert_allclose(observed, truth, rtol=1e-12, atol=1e-12)

    def test_gradient_polish_runs_on_cubic_witness_by_default(self) -> None:
        key = "GCL_BLIND_SWIGLU_VALIDATION_FULL_TAIL_POLISH_ON_CUBIC"
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(key, None)
            self.assertTrue(validation_full_tail_polish_cubic_trigger("gradient-polish", 2048))
            self.assertFalse(validation_full_tail_polish_cubic_trigger("gradient-polish", 0))
            self.assertFalse(validation_full_tail_polish_cubic_trigger("residual-block-refit", 2048))

    def test_cubic_polish_trigger_can_be_explicitly_disabled(self) -> None:
        with patch.dict(os.environ, {
            "GCL_BLIND_SWIGLU_VALIDATION_FULL_TAIL_POLISH_ON_CUBIC": "0",
        }, clear=False):
            self.assertFalse(validation_full_tail_polish_cubic_trigger("gradient-polish", 2048))

    def test_validation_gradient_polish_uses_environment_cubic_weight(self) -> None:
        args = swiglu_module.build_parser().parse_args([])
        with patch.dict(os.environ, {
            "GCL_BLIND_SWIGLU_FIT_CUBIC_WEIGHT": "0.75",
            "GCL_BLIND_SWIGLU_VALIDATION_GRADIENT_POLISH_WEIGHT": "1.25",
        }, clear=False):
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("GCL_BLIND_SWIGLU_VALIDATION_GRADIENT_POLISH_CUBIC_WEIGHT", None)
                gradient_weight, cubic_weight = validation_gradient_polish_weights(args)

        self.assertEqual(gradient_weight, 1.25)
        self.assertEqual(cubic_weight, 0.75)

    def test_sequential_pair_refit_continues_after_first_pair_is_rejected(self) -> None:
        rng = np.random.default_rng(260725)
        coords = rng.normal(size=(512, 4))
        candidate = {
            "gates": np.array([
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ]),
            "ups": np.array([
                [0.0, 1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, -1.0],
            ]),
            "scales": np.ones(4),
            "bias": 0.0,
            "selectionScore": 1.0,
        }
        calls: list[list[int]] = []

        def fake_refit(*args: object, **_kwargs: object) -> dict[str, object]:
            current = dict(args[2])
            atoms = list(args[3])
            calls.append(atoms)
            if atoms == [0, 1]:
                current["residualGradientBlockRefit"] = False
                return current
            gates = np.asarray(current["gates"]).copy()
            ups = np.asarray(current["ups"]).copy()
            gates[3] = [0.0, 0.0, 0.0, 1.0]
            ups[3] = [1.0, 0.0, 0.0, 0.0]
            current.update({
                "gates": gates,
                "ups": ups,
                "residualGradientBlockRefit": True,
                "selectionScore": 0.5,
            })
            return current

        with patch.object(
            swiglu_module,
            "residual_gradient_block_refit_swiglu_candidate",
            side_effect=fake_refit,
        ):
            observed = sequential_cancellation_pair_refit_swiglu_candidate(
                coords,
                np.zeros(coords.shape[0]),
                candidate,
                np.zeros((2, 4)),
                np.zeros((2, 4)),
                260725,
            )

        self.assertEqual(calls, [[0, 1], [2, 3]])
        self.assertTrue(observed["residualGradientBlockRefitCancellationRescued"])
        self.assertEqual(
            observed["residualGradientBlockRefitCancellationRescueAttempts"][1]["atoms"],
            [2, 3],
        )

    def test_fit_snapshot_round_trip_preserves_cubic_witnesses(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            path = str(Path(raw_tmp) / "fit.npz")
            save_fit_snapshot(
                path,
                basis=np.eye(2),
                train_coords=np.array([[0.1, 0.2], [0.3, 0.4]]),
                train_targets=np.array([0.5, 0.6]),
                fit={
                    "gates": np.array([[1.0, 0.0]]),
                    "ups": np.array([[0.0, 1.0]]),
                    "scales": np.array([0.75]),
                    "bias": 0.01,
                    "mae": 0.02,
                    "mse": 0.0004,
                    "r2": 0.99,
                    "worstError": 0.03,
                    "restartCount": 4,
                },
                dimension=2,
                active_rank=2,
                atom_count=1,
                gradient_witness_count=3,
                eigenvalues=np.array([2.0, 1.0]),
                energy_ratio=0.95,
                fit_gradient_coords=np.array([[0.2, -0.1]]),
                fit_gradient_targets=np.array([[0.3, 0.4]]),
                cubic_directions=np.array([[1.0, 0.0], [0.0, 1.0]]),
                cubic_targets=np.array([0.7, -0.2]),
                cubic_holdout_directions=np.array([[0.6, 0.8]]),
                cubic_holdout_targets=np.array([0.11]),
                access_audit=self._clean_snapshot_audit(),
            )

            observed = load_fit_snapshot(path, dimension=2, active_rank=2, atom_count=1)

            np.testing.assert_allclose(observed["cubicDirections"], [[1.0, 0.0], [0.0, 1.0]])
            np.testing.assert_allclose(observed["cubicTargets"], [0.7, -0.2])
            np.testing.assert_allclose(observed["cubicHoldoutDirections"], [[0.6, 0.8]])
            np.testing.assert_allclose(observed["cubicHoldoutTargets"], [0.11])

    @staticmethod
    def _clean_snapshot_audit() -> dict:
        return {
            "solverAccess": "oracle-only",
            "weightsUsedBySolver": False,
            "hooksUsedBySolver": False,
            "truthRequestCount": 0,
            "externalOracleTransport": True,
        }

    def _write_snapshot_with_audit(self, path: str, audit: dict) -> None:
        save_fit_snapshot(
            path,
            basis=np.eye(2),
            train_coords=np.array([[0.1, 0.2], [0.3, 0.4]]),
            train_targets=np.array([0.5, 0.6]),
            fit={
                "gates": np.array([[1.0, 0.0]]),
                "ups": np.array([[0.0, 1.0]]),
                "scales": np.array([0.75]),
                "bias": 0.01,
                "mae": 0.02,
                "mse": 0.0004,
                "r2": 0.99,
                "worstError": 0.03,
                "restartCount": 4,
            },
            dimension=2,
            active_rank=2,
            atom_count=1,
            gradient_witness_count=3,
            eigenvalues=np.array([2.0, 1.0]),
            energy_ratio=0.95,
            fit_gradient_coords=np.array([[0.2, -0.1]]),
            fit_gradient_targets=np.array([[0.3, 0.4]]),
            cubic_directions=np.array([[1.0, 0.0], [0.0, 1.0]]),
            cubic_targets=np.array([0.7, -0.2]),
            cubic_holdout_directions=np.array([[0.6, 0.8]]),
            cubic_holdout_targets=np.array([0.11]),
            access_audit=audit,
        )

    def test_fit_snapshot_loader_rejects_a_run_that_requested_truth(self) -> None:
        """The loader's access check used to read back constants the writer had just
        written, so it could never refuse anything. It now reads measured state, and this
        proves it actually refuses a snapshot whose producing run touched the truth
        channel."""
        audit = self._clean_snapshot_audit()
        audit["truthRequestCount"] = 3
        with tempfile.TemporaryDirectory() as raw_tmp:
            path = str(Path(raw_tmp) / "fit.npz")
            self._write_snapshot_with_audit(path, audit)
            with self.assertRaises(RuntimeError) as caught:
                load_fit_snapshot(path, dimension=2, active_rank=2, atom_count=1)
            self.assertIn("requested truth", str(caught.exception))

    def test_fit_snapshot_loader_rejects_a_run_that_used_hooks(self) -> None:
        audit = self._clean_snapshot_audit()
        audit["hooksUsedBySolver"] = True
        with tempfile.TemporaryDirectory() as raw_tmp:
            path = str(Path(raw_tmp) / "fit.npz")
            self._write_snapshot_with_audit(path, audit)
            with self.assertRaises(RuntimeError) as caught:
                load_fit_snapshot(path, dimension=2, active_rank=2, atom_count=1)
            self.assertIn("used hooks", str(caught.exception))

    def test_fit_snapshot_loader_accepts_a_clean_run(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            path = str(Path(raw_tmp) / "fit.npz")
            self._write_snapshot_with_audit(path, self._clean_snapshot_audit())
            observed = load_fit_snapshot(path, dimension=2, active_rank=2, atom_count=1)
            np.testing.assert_allclose(observed["cubicTargets"], [0.7, -0.2])

    def test_symmetric_multiscale_retains_each_secant_witness(self) -> None:
        class Oracle:
            def eval(self, inputs: np.ndarray) -> np.ndarray:
                return inputs[:, 0] * inputs[:, 1] ** 3 + 0.5 * inputs[:, 1]

        base = np.array([0.7, -0.2], dtype=np.float32)
        steps = [0.1, 0.4]
        observed = estimate_symmetric_multiscale_coordinate_gradients(
            Oracle(),
            base,
            steps,
        )

        self.assertEqual(observed.shape, (2, 2))
        np.testing.assert_allclose(observed[:, 0], base[1] ** 3, rtol=1e-5, atol=1e-6)
        expected_second = np.array([
            base[0] * (3.0 * base[1] ** 2 + steps[0] ** 2) + 0.5,
            base[0] * (3.0 * base[1] ** 2 + steps[1] ** 2) + 0.5,
        ])
        np.testing.assert_allclose(observed[:, 1], expected_second, rtol=1e-5, atol=1e-6)

    def test_symmetric_regression_slope_uses_all_finite_steps(self) -> None:
        steps = [0.25, 0.5]
        expected = np.array([0.4, -1.25, 2.0])
        offsets = np.array([10.0, -3.0, 0.5])
        plus = np.vstack([offsets + step * expected for step in steps])
        minus = np.vstack([offsets - step * expected for step in steps])

        observed = symmetric_regression_slopes(plus, minus, steps)

        np.testing.assert_allclose(observed, expected, rtol=1e-12, atol=1e-12)

    def test_source_cancellation_audit_measures_weighted_output_cancellation(self) -> None:
        rng = np.random.default_rng(260724)
        coords = rng.normal(size=(4096, 3))
        gates = np.array([
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        effective_ups = np.array([
            [0.0, 1.0, 0.0],
            [0.0, -0.98, 0.0],
            [0.0, 0.0, 0.25],
        ])

        audit = audit_swiglu_source_cancellation(
            coords,
            gates,
            effective_ups,
            [10, 11, 12],
        )

        self.assertGreater(audit["cancellationFraction"], 0.7)
        self.assertGreaterEqual(audit["pairCountAtOrBelowMinus050"], 1)
        self.assertLess(audit["minimumPairCorrelation"], -0.99)
        self.assertEqual(audit["mostNegativePairs"][0]["sourceUnits"], [10, 11])

    def test_richardson_extrapolation_removes_cubic_stencil_h2_bias(self) -> None:
        true_scores = np.array([0.4, -1.2, 2.5])
        bias = np.array([0.7, -0.3, 1.1])
        steps = [0.35, 0.25]
        estimates = np.vstack([
            true_scores + bias * step ** 2
            for step in steps
        ])

        observed = richardson_extrapolate_cubic_scores(estimates, steps)

        np.testing.assert_allclose(observed, true_scores, rtol=1e-12, atol=1e-12)

    def test_cubic_directional_derivative_matches_swiglu_finite_difference(self) -> None:
        rng = np.random.default_rng(260723)
        directions = rng.normal(size=(12, 4))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        gates = rng.normal(size=(3, 4))
        ups = rng.normal(size=(3, 4))
        scales = np.array([0.7, -1.1, 0.35])
        expected = predict_swiglu_cubic_directional_derivatives(
            directions,
            gates,
            ups,
            scales,
        )

        step = 1e-3
        multipliers = np.array([2.0, 1.0, -1.0, -2.0])
        values = []
        for multiplier in multipliers:
            coords = multiplier * step * directions
            gate_projection = coords @ gates.T
            up_projection = coords @ ups.T
            silu = gate_projection / (1.0 + np.exp(-gate_projection))
            values.append(np.sum(silu * up_projection * scales[None, :], axis=1))
        observed = (
            values[0] - 2.0 * values[1] + 2.0 * values[2] - values[3]
        ) / (2.0 * step ** 3)

        np.testing.assert_allclose(observed, expected, rtol=2e-5, atol=2e-5)

    def test_detects_source_blind_cancellation_pair(self) -> None:
        rng = np.random.default_rng(260914)
        coords = rng.normal(size=(2048, 3))
        candidate = {
            "gates": np.array([
                [1.0, 0.0, 0.0],
                [0.99, 0.08, 0.0],
                [0.0, 0.0, 1.0],
            ]),
            "ups": np.array([
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
            ]),
            "scales": np.ones(3),
        }

        pairs = detect_swiglu_cancellation_pairs(coords, candidate)

        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["atoms"], [0, 1])
        self.assertGreater(pairs[0]["gateCosine"], 0.99)
        self.assertGreater(pairs[0]["upCosine"], 0.99)
        self.assertLess(pairs[0]["contributionCorrelation"], -0.99)

    def test_does_not_flag_independent_swiglu_atoms(self) -> None:
        rng = np.random.default_rng(260915)
        coords = rng.normal(size=(2048, 3))
        candidate = {
            "gates": np.eye(3),
            "ups": np.roll(np.eye(3), shift=1, axis=1),
            "scales": np.ones(3),
        }

        self.assertEqual(detect_swiglu_cancellation_pairs(coords, candidate), [])

    def test_maximum_weight_assignment_avoids_greedy_collision(self) -> None:
        assignment = maximum_weight_assignment(np.array([
            [10.0, 9.0, 0.0],
            [9.0, 0.0, 0.0],
            [0.0, 8.0, 8.0],
        ]))

        self.assertEqual(assignment, [1, 0, 2])

    def test_selects_four_atom_elbow_for_layer9_profile(self) -> None:
        gate_norms = np.array([
            0.8015, 0.7818, 0.7371, 0.7090,
            1.4019, 1.4661, 0.9854, 1.2493,
            0.8726, 0.9577, 0.9654, 1.0947,
            1.1311, 1.1587, 1.1801, 1.1990,
        ])
        atoms, telemetry = select_residual_block_atoms(
            diagnostic([
                0.007986804745446708,
                0.006593001315381546,
                0.005089776389715272,
                0.004277377561020204,
                0.0031239879326643736,
                0.0029614117320949287,
                0.0023178812924602435,
                0.0022521091873103413,
            ]),
            16,
            adaptive=True,
            fixed_top_k=4,
            min_atoms=4,
            max_atoms=8,
            gate_norms=gate_norms,
        )

        self.assertEqual(atoms, [0, 1, 2, 3])
        self.assertEqual(telemetry["method"], "gate-norm-tail-consensus-strong-cut")
        self.assertEqual(telemetry["selectedCount"], 4)

    def test_selects_five_atom_elbow_for_layer1_profile(self) -> None:
        gate_norms = np.array([
            0.7996, 0.8166, 0.8175, 0.8367,
            0.8653, 1.0026, 1.0066, 1.0139,
            1.0305, 1.0339, 1.0359, 1.0372,
            1.0519, 1.0549, 1.1038, 1.2050,
        ])
        atoms, telemetry = select_residual_block_atoms(
            diagnostic([
                0.0060454970020228345,
                0.0056014182367720294,
                0.0047109801927608845,
                0.0041296070497757725,
                0.0038404367088938256,
                0.0029525908340678627,
                0.002527195018048271,
                0.0024455954478864498,
            ]),
            16,
            adaptive=True,
            fixed_top_k=4,
            min_atoms=4,
            max_atoms=8,
            gate_norms=gate_norms,
        )

        self.assertEqual(atoms, [0, 1, 2, 3, 4])
        self.assertEqual(telemetry["method"], "gate-norm-weak-block")
        self.assertEqual(telemetry["selectedCount"], 5)

    def test_selects_eight_atom_weak_block_for_layer15_profile(self) -> None:
        atoms, telemetry = select_residual_block_atoms(
            {
                "topAtoms": [
                    {"atom": 5, "tailResponsibilityScore": 0.003935445789807495},
                    {"atom": 15, "tailResponsibilityScore": 0.0036477064686118448},
                    {"atom": 2, "tailResponsibilityScore": 0.0036222140433447994},
                    {"atom": 14, "tailResponsibilityScore": 0.0033467926530600643},
                    {"atom": 7, "tailResponsibilityScore": 0.003313670692545136},
                    {"atom": 10, "tailResponsibilityScore": 0.0033036917893401233},
                    {"atom": 9, "tailResponsibilityScore": 0.003178097972462981},
                    {"atom": 8, "tailResponsibilityScore": 0.0031146366536028186},
                ],
            },
            16,
            adaptive=True,
            fixed_top_k=4,
            min_atoms=4,
            max_atoms=8,
            gate_norms=np.array([
                0.9755, 1.3610, 0.6762, 1.0905,
                1.2928, 0.7918, 1.2027, 0.6748,
                0.7317, 0.6808, 1.0909, 1.1888,
                0.8714, 1.1895, 0.8147, 0.6777,
            ]),
        )

        self.assertEqual(atoms, [5, 15, 2, 14, 7, 9, 8, 12])
        self.assertEqual(telemetry["method"], "gate-norm-weak-block")
        self.assertEqual(telemetry["selectedCount"], 8)

    def test_completes_fixed_width_weak_block_from_tail_responsibility(self) -> None:
        tail_order = [3, 7, 6, 11, 12, 5, 1, 13, 15, 4, 8, 9, 10, 2, 14, 0]
        tail_scores = [
            9.18832519781308e-09,
            7.47368678870589e-09,
            6.950272942737514e-09,
            6.600722769661271e-09,
            6.269163736902217e-09,
            6.0865298271511556e-09,
            5.699158713507886e-09,
            5.612440612820833e-09,
            5.597111753455207e-09,
            5.022811199271221e-09,
            4.702578284626496e-09,
            4.689414726334617e-09,
            4.650321516201202e-09,
            4.60665057855175e-09,
            4.075666728096066e-09,
            4.019978437058104e-09,
        ]
        diagnostic_rows = {
            "topAtoms": [
                {"atom": atom, "tailResponsibilityScore": score}
                for atom, score in zip(tail_order, tail_scores)
            ]
        }
        atoms, telemetry = select_residual_block_atoms(
            diagnostic_rows,
            16,
            adaptive=True,
            fixed_top_k=8,
            min_atoms=8,
            max_atoms=8,
            gate_norms=np.array([
                0.20635378, 0.20608645, 0.18422737, 0.17675815,
                0.21825727, 0.17328820, 0.25880982, 0.19142435,
                0.16609833, 0.22253233, 0.21496464, 0.15597766,
                0.16088522, 0.17289745, 0.19432117, 0.16720257,
            ]),
        )

        self.assertEqual(set(atoms), {3, 5, 7, 8, 11, 12, 13, 15})
        self.assertEqual(telemetry["method"], "gate-norm-weak-block-tail-completion")
        self.assertEqual(telemetry["tailCompletionAtoms"], [7])
        self.assertEqual(telemetry["selectedCount"], 8)

    def test_fixed_mode_remains_available_for_diagnostics(self) -> None:
        atoms, telemetry = select_residual_block_atoms(
            diagnostic([0.8, 0.4, 0.1, 0.01]),
            4,
            adaptive=False,
            fixed_top_k=2,
            min_atoms=1,
            max_atoms=3,
        )

        self.assertEqual(atoms, [0, 1])
        self.assertEqual(telemetry["method"], "fixed-top-k")


if __name__ == "__main__":
    unittest.main()
