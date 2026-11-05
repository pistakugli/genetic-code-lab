#!/usr/bin/env python3
"""Verify measured cubic gate recovery has an oracle-only solver boundary."""
from __future__ import annotations

import inspect
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "llm_reader"))

import cubic_identification_probe as probe  # noqa: E402
from tools.llm_reader.adapters.qwen2_swiglu import Qwen2SwiGLUOracleAdapter  # noqa: E402
from tools.llm_reader.adapters.registry import create_adapter  # noqa: E402
from tools.llm_reader.cubic_identification_probe import (  # noqa: E402
    OracleQueryBudgetExceeded,
    OracleWorkspaceLimitExceeded,
    recover_gates_from_measured_oracle,
    validate_qwen_real_activation_artifacts,
)


class LockedOracle:
    def __init__(self, teacher: probe.Teacher) -> None:
        self.__teacher = teacher
        self.points = 0
        self.truth_unlock_count = 0
        self.truth_reads: list[str] = []
        self.__truth_unlocked = False

    def __call__(self, points: torch.Tensor) -> torch.Tensor:
        self.points += int(points.shape[0])
        return self.__teacher(points)

    def unlock_truth(self) -> None:
        if self.__truth_unlocked:
            raise AssertionError("post-hoc truth was unlocked more than once")
        self.truth_unlock_count += 1
        self.__truth_unlocked = True

    def _assert_truth_unlocked(self, name: str) -> None:
        self.truth_reads.append(name)
        if not self.__truth_unlocked:
            raise AssertionError(f"solver read {name} before the post-hoc unlock")

    @property
    def gate(self) -> torch.Tensor:
        self._assert_truth_unlocked("gate")
        return self.__teacher.gate

    @property
    def unit_count(self) -> int:
        self._assert_truth_unlocked("unit_count")
        return int(self.__teacher.gate.shape[0])

    def __getattr__(self, name: str) -> object:
        if name in {"units", "metadata", "meta", "weights", "counter"}:
            raise AssertionError(f"solver requested forbidden oracle field {name}")
        raise AttributeError(name)


class OracleOnlyCubicRecovery(unittest.TestCase):
    def test_overcomplete_count_and_gates_come_from_measured_oracle_span(self) -> None:
        teacher = probe.make_teacher(
            units=10,
            input_dim=8,
            output_dim=8,
            seed=0,
            device=torch.device("cpu"),
        )
        oracle = LockedOracle(teacher)
        parameters = inspect.signature(probe.recover_gates_from_measured_oracle).parameters
        self.assertFalse({"units", "unit_count", "teacher"} & parameters.keys())

        result = probe.recover_gates_from_measured_oracle(
            oracle=oracle,
            input_dim=8,
            output_dim=8,
            base_point_count=3,
            reader_count=8,
            base_scale=1.0,
            seed=0,
            tolerance=1e-6,
            span_tolerance=1e-5,
            purity=0.9,
            collinearity=0.99,
            steps=[0.02, 0.04],
            direction_multiple=3.0,
            device=torch.device("cpu"),
        )

        self.assertEqual(result["measuredSpanRank"], 20)
        self.assertEqual(result["spanImpliedUnitCount"], 10)
        self.assertTrue(result["spanRankIsEven"])
        self.assertTrue(result["gatesMatchSpanRank"])
        self.assertEqual(result["gatesReturned"], 10)
        self.assertEqual(result["oracleCallsPoints"], 8640)
        self.assertEqual(oracle.points, result["oracleCallsPoints"])
        self.assertEqual(oracle.truth_reads, [])
        self.assertEqual(oracle.truth_unlock_count, 0)

        oracle.unlock_truth()
        truth = oracle.gate
        truth = truth / truth.norm(dim=1, keepdim=True)
        best_cosines = (result["gates"] @ truth.T).abs().max(dim=0).values
        self.assertTrue(torch.all(best_cosines >= 0.99).item())
        self.assertEqual(oracle.truth_reads, ["gate"])
        self.assertEqual(oracle.truth_unlock_count, 1)

    def test_oracle_response_shape_mismatch_is_rejected(self) -> None:
        def malformed_oracle(points: torch.Tensor) -> torch.Tensor:
            return torch.zeros((points.shape[0], 1), dtype=points.dtype, device=points.device)

        with self.assertRaisesRegex(ValueError, "oracle output shape"):
            probe.recover_gates_from_measured_oracle(
                oracle=malformed_oracle,
                input_dim=2,
                output_dim=2,
                base_point_count=1,
                reader_count=1,
                base_scale=1.0,
                seed=0,
                tolerance=1e-6,
                span_tolerance=1e-5,
                purity=0.9,
                collinearity=0.99,
                steps=[0.02],
                direction_multiple=1.0,
                device=torch.device("cpu"),
            )


class QwenSwiGLUOracleContracts(unittest.TestCase):
    class FakeQwenMlp(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate_proj = torch.nn.Linear(4, 3, bias=False)
            self.up_proj = torch.nn.Linear(4, 3, bias=False)
            self.down_proj = torch.nn.Linear(3, 4, bias=False)
            self.act_fn = torch.nn.SiLU()

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.down_proj(self.act_fn(self.gate_proj(inputs)) * self.up_proj(inputs))

    def test_registry_selects_qwen2_adapter_and_exposes_swiglu(self) -> None:
        adapter = create_adapter("qwen2-swiglu")
        self.assertIsInstance(adapter, Qwen2SwiGLUOracleAdapter)
        self.assertIn("qwen2", adapter.supported_model_types)
        self.assertEqual(adapter.activation_family, "swiglu")

    def test_qwen2_mlp_view_exposes_the_real_swiglu_activation(self) -> None:
        mlp = self.FakeQwenMlp()
        model = SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace(mlp=mlp)]))
        view = create_adapter("qwen2").mlp_modules(model, [0])[0]
        captured: list[torch.Tensor] = []
        handle = view.act.register_forward_hook(
            lambda _module, _inputs, output: captured.append(output.detach().clone())
        )
        inputs = torch.randn(2, 5, 4)
        try:
            output = mlp(inputs)
        finally:
            handle.remove()

        expected_activation = mlp.act_fn(mlp.gate_proj(inputs)) * mlp.up_proj(inputs)
        expected_output = mlp.down_proj(expected_activation)
        self.assertTrue(torch.equal(output, expected_output))
        self.assertEqual(len(captured), 1)
        self.assertTrue(torch.equal(captured[0], expected_activation))
        self.assertEqual(tuple(view.c_proj.weight.shape), (3, 4))

    def test_qwen_full_width_workspace_guard_rejects_before_oracle_access(self) -> None:
        calls: list[int] = []

        def oracle(points: torch.Tensor) -> torch.Tensor:
            calls.append(int(points.shape[0]))
            return torch.zeros((points.shape[0], 896), dtype=points.dtype, device=points.device)

        with self.assertRaises(OracleWorkspaceLimitExceeded):
            recover_gates_from_measured_oracle(
                oracle=oracle,
                input_dim=896,
                output_dim=896,
                base_point_count=1,
                reader_count=1,
                base_scale=1.0,
                seed=0,
                tolerance=1e-6,
                span_tolerance=1e-5,
                purity=0.9,
                collinearity=0.99,
                steps=[0.02],
                direction_multiple=1.0,
                device=torch.device("cpu"),
                max_workspace_bytes=4 * 1024**3,
            )
        self.assertEqual(calls, [])

    def test_oracle_query_budget_stops_before_over_budget_call(self) -> None:
        calls: list[int] = []

        def oracle(points: torch.Tensor) -> torch.Tensor:
            calls.append(int(points.shape[0]))
            return torch.zeros((points.shape[0], 2), dtype=points.dtype, device=points.device)

        with self.assertRaises(OracleQueryBudgetExceeded) as error:
            recover_gates_from_measured_oracle(
                oracle=oracle,
                input_dim=2,
                output_dim=2,
                base_point_count=1,
                reader_count=1,
                base_scale=1.0,
                seed=0,
                tolerance=1e-6,
                span_tolerance=1e-5,
                purity=0.9,
                collinearity=0.99,
                steps=[0.02],
                direction_multiple=1.0,
                device=torch.device("cpu"),
                max_oracle_points=0,
            )
        self.assertEqual(error.exception.requested_points, 16)
        self.assertEqual(error.exception.spent_points, 0)
        self.assertEqual(calls, [])


class AdaptiveOracleSwiGLUDistillation(unittest.TestCase):
    def test_candidate_widths_are_measurement_hypotheses_not_teacher_metadata(self) -> None:
        self.assertEqual(probe._swiglu_candidate_widths(1), [1])
        self.assertEqual(probe._swiglu_candidate_widths(7), [1, 2, 4, 7])
        parameters = inspect.signature(
            probe.recover_swiglu_by_adaptive_oracle_distillation
        ).parameters
        self.assertFalse({"teacher", "units", "unit_count", "hidden_unit_count"} & parameters.keys())

    @unittest.skipUnless(torch.cuda.is_available(), "adaptive distillation requires CUDA")
    def test_query_budget_and_oracle_output_shape_are_checked_before_fitting(self) -> None:
        device = torch.device("cuda")
        calls: list[int] = []

        def oracle(points: np.ndarray) -> np.ndarray:
            calls.append(int(points.shape[0]))
            return np.zeros((points.shape[0], 1), dtype=np.float32)

        inputs = np.ones((4, 2), dtype=np.float32)
        outputs = np.ones((4, 2), dtype=np.float32)
        with self.assertRaises(probe.OracleQueryBudgetExceeded) as error:
            probe.recover_swiglu_by_adaptive_oracle_distillation(
                oracle, 2, 2, inputs, outputs,
                max_components=1, oracle_query_budget=7,
                initial_train_queries=4, validation_queries=4,
                adaptive_batch_queries=0, steps_per_stage=1,
                batch_size=2, learning_rate=0.01, seed=2, device=device,
            )
        self.assertEqual(error.exception.requested_points, 8)
        self.assertEqual(error.exception.spent_points, 0)
        self.assertEqual(calls, [])

        with self.assertRaisesRegex(ValueError, "oracle output shape"):
            probe.recover_swiglu_by_adaptive_oracle_distillation(
                oracle, 2, 2, inputs, outputs,
                max_components=1, oracle_query_budget=16,
                initial_train_queries=4, validation_queries=4,
                adaptive_batch_queries=0, steps_per_stage=1,
                batch_size=2, learning_rate=0.01, seed=2, device=device,
            )
        self.assertEqual(calls, [4])

    @unittest.skipUnless(torch.cuda.is_available(), "adaptive distillation requires CUDA")
    def test_student_distillation_and_cubic_gate_solver_share_only_the_synthetic_oracle(self) -> None:
        device = torch.device("cuda")
        teacher = probe.make_teacher(
            units=2, input_dim=4, output_dim=4, seed=31, device=device
        )
        rng = np.random.default_rng(99)
        recovery_inputs = (rng.normal(size=(64, 4)) * 0.6).astype(np.float32)
        oracle_points: list[int] = []

        def oracle_numpy(points: np.ndarray) -> np.ndarray:
            oracle_points.append(int(points.shape[0]))
            with torch.inference_mode():
                tensor = torch.as_tensor(points, dtype=torch.float64, device=device)
                return teacher(tensor).float().cpu().numpy()

        recovery_outputs = oracle_numpy(recovery_inputs)
        distillation = probe.recover_swiglu_by_adaptive_oracle_distillation(
            oracle=oracle_numpy,
            input_dim=4,
            output_dim=4,
            recovery_inputs=recovery_inputs,
            recovery_outputs=recovery_outputs,
            max_components=2,
            oracle_query_budget=1024,
            initial_train_queries=512,
            validation_queries=256,
            adaptive_batch_queries=0,
            steps_per_stage=500,
            batch_size=128,
            learning_rate=0.03,
            seed=7,
            device=device,
            recovery_metadata={
                "token_positions": np.arange(len(recovery_inputs), dtype=np.int32),
            },
        )
        cubic = recover_gates_from_measured_oracle(
            oracle=teacher,
            input_dim=4,
            output_dim=4,
            base_point_count=3,
            reader_count=4,
            base_scale=1.0,
            seed=9,
            tolerance=1e-6,
            span_tolerance=1e-5,
            purity=0.9,
            collinearity=0.99,
            steps=[0.02, 0.04],
            direction_multiple=3.0,
            device=device,
        )

        truth = teacher.gate / teacher.gate.norm(dim=1, keepdim=True)
        gate_cosines = (cubic["gates"] @ truth.T).abs().max(dim=0).values
        self.assertEqual(cubic["gatesReturned"], 2)
        self.assertTrue(torch.all(gate_cosines >= 0.99).item())
        self.assertEqual(distillation["selectedCandidateComponents"], 2)
        self.assertTrue(distillation["studentWidthSelectedByMeasurements"])
        self.assertFalse(distillation["teacherComponentCountInferred"])
        self.assertLess(distillation["validation"]["normalizedMse"], 0.2)
        self.assertEqual(
            distillation["queryAccounting"]["oraclePoints"],
            sum(oracle_points) - len(recovery_inputs),
        )
        self.assertEqual(
            distillation["trainingData"]["natural_token_positions"].shape[0],
            len(recovery_inputs) + 512,
        )
        self.assertTrue(
            np.all(
                distillation["trainingData"]["natural_token_positions"][
                    len(recovery_inputs):
                ] == -1
            )
        )


class QwenActivationArtifactVerification(unittest.TestCase):
    natural_splits = (
        "recovery_observations",
        "functional_evaluation",
        "structural_evaluation",
        "distributional_analysis",
    )

    def test_process_memory_helpers_read_linux_proc_metrics(self) -> None:
        if not Path("/proc/self/status").is_file():
            self.skipTest("Linux /proc process metrics are unavailable")
        self.assertGreater(probe._proc_vm_hwm_bytes() or 0, 0)
        self.assertGreater(probe._proc_status_memory_bytes("self", "VmRSS") or 0, 0)
        self.assertGreater(probe._proc_smaps_pss_bytes("self") or 0, 0)
        self.assertIsNone(probe._proc_status_memory_bytes(2**30, "VmRSS"))
        self.assertIsNone(probe._proc_smaps_pss_bytes(2**30))

    def test_extra_recovery_prompts_are_unique_and_recovery_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "prompts.json"
            source = [
                {"promptId": "recovery-extra-01", "text": "Explain a lunar eclipse."},
                {"promptId": "recovery-extra-02", "text": "Describe a sorting algorithm."},
            ]
            path.write_text(json.dumps(source), encoding="utf-8")
            prompts, raw = probe._load_extra_prompts(
                path, {"recovery-01", "functional-01"}, "recovery-extra-"
            )
            self.assertEqual(prompts, [
                ("recovery-extra-01", "Explain a lunar eclipse."),
                ("recovery-extra-02", "Describe a sorting algorithm."),
            ])
            self.assertEqual(raw, path.read_bytes())

            path.write_text(json.dumps([source[0], source[0]]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                probe._load_extra_prompts(path, set(), "recovery-extra-")

    def write_fixture(self, output_dir: Path) -> None:
        datasets: dict[str, dict[str, object]] = {}
        for index, split_name in enumerate(self.natural_splits):
            prompt_id = f"{split_name}-01"
            prompt_hash = hashlib.sha256(prompt_id.encode()).hexdigest()
            path = output_dir / f"{split_name}.npz"
            inputs = np.asarray([[index, index + 1], [index + 2, index + 3]], dtype=np.float32)
            outputs = inputs.copy()
            np.savez_compressed(
                path,
                inputs=inputs,
                outputs=outputs,
                token_ids=np.asarray([10, 11], dtype=np.int64),
                token_positions=np.asarray([0, 1], dtype=np.int32),
                prompt_ids=np.asarray([prompt_id, prompt_id]),
                prompt_sha256=np.asarray([prompt_hash, prompt_hash]),
            )
            datasets[split_name] = {
                "file": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "samples": 2,
                "inputShape": [2, 2],
                "outputShape": [2, 2],
                "activationProvenance": "natural_observed_model_inference",
                "accessMode": "instrumented_capture",
                "prompts": [{
                    "promptId": prompt_id,
                    "promptSha256": prompt_hash,
                    "tokenCount": 2,
                }],
            }

        generated_path = output_dir / "generated_oracle_probes.npz"
        generated_inputs = np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        np.savez_compressed(
            generated_path,
            inputs=generated_inputs,
            outputs=generated_inputs.copy(),
            provenance=np.asarray(["generated_oracle_probe"] * 2),
        )
        datasets["generated_oracle_probes"] = {
            "file": generated_path.name,
            "sha256": hashlib.sha256(generated_path.read_bytes()).hexdigest(),
            "samples": 2,
            "inputShape": [2, 2],
            "outputShape": [2, 2],
            "activationProvenance": "deliberately_generated_oracle_probes",
            "accessMode": "opaque_eval_only_oracle",
            "distribution": "iid_standard_normal_scaled_to_unit_expected_l2",
        }
        report = {
            "schemaVersion": 1,
            "kind": "gcl-real-qwen-activation-capture",
            "model": "Qwen/Qwen2.5-0.5B-Instruct",
            "module": "model.layers.0.mlp",
            "architecture": {
                "modelType": "qwen2",
                "inputDim": 2,
                "outputDim": 2,
                "intermediateDim": 3,
                "activation": "silu",
            },
            "accessMode": {
                "activationCapture": "instrumented",
                "fullModuleOracle": "opaque_eval_only",
                "solverReceivedWeights": False,
                "solverReceivedHooks": False,
                "solverReceivedHiddenUnitCount": False,
                "postHocGroundTruthUsed": False,
            },
            "datasets": datasets,
            "oracleSmoke": {
                "queryCount": 2,
                "queryBudget": 4,
                "unitCountExposedToOracleClient": False,
                "maxAbsDeltaAgainstIndependentDirectMlp": 0.0,
            },
            "recovery": {
                "status": "blocked_before_oracle_queries_by_workspace_guard",
                "reconstructionProduced": False,
                "oracleCallsSpentByRecoverySolver": 0,
            },
        }
        (output_dir / "capture-report.json").write_text(json.dumps(report), encoding="utf-8")

    def test_artifact_verifier_checks_hashes_shapes_and_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            self.write_fixture(output_dir)
            report, datasets = validate_qwen_real_activation_artifacts(output_dir)
            self.assertEqual(report["architecture"]["inputDim"], 2)
            self.assertEqual(datasets["recovery_observations"]["inputs"].shape, (2, 2))
            (output_dir / "generated_oracle_probes.npz").write_bytes(b"corrupted")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                validate_qwen_real_activation_artifacts(output_dir)

    def test_artifact_verifier_rejects_prompt_overlap_between_splits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            self.write_fixture(output_dir)
            report_path = output_dir / "capture-report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            duplicate_hash = report["datasets"]["recovery_observations"]["prompts"][0]["promptSha256"]
            functional_record = report["datasets"]["functional_evaluation"]
            functional_archive_path = output_dir / functional_record["file"]
            with np.load(functional_archive_path, allow_pickle=False) as archive:
                functional_arrays = {key: archive[key] for key in archive.files}
            functional_arrays["prompt_sha256"][:] = duplicate_hash
            np.savez_compressed(functional_archive_path, **functional_arrays)
            functional_record["sha256"] = hashlib.sha256(
                functional_archive_path.read_bytes()
            ).hexdigest()
            functional_record["prompts"][0]["promptSha256"] = duplicate_hash
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not disjoint"):
                validate_qwen_real_activation_artifacts(output_dir)

    def test_artifact_verifier_hash_binds_additional_prompt_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            self.write_fixture(output_dir)
            report_path = output_dir / "capture-report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            prompt_id = "recovery-extra-01"
            prompt_text = "Explain a lunar eclipse."
            prompt_hash = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()

            recovery_record = report["datasets"]["recovery_observations"]
            archive_path = output_dir / recovery_record["file"]
            with np.load(archive_path, allow_pickle=False) as archive:
                arrays = {key: archive[key] for key in archive.files}
            arrays["prompt_ids"][:] = prompt_id
            arrays["prompt_sha256"][:] = prompt_hash
            np.savez_compressed(archive_path, **arrays)
            recovery_record["sha256"] = hashlib.sha256(archive_path.read_bytes()).hexdigest()
            recovery_record["prompts"] = [{
                "promptId": prompt_id,
                "promptSha256": prompt_hash,
                "tokenCount": 2,
            }]

            source_path = output_dir / "additional-recovery-prompts.json"
            source_bytes = json.dumps([{
                "promptId": prompt_id,
                "text": prompt_text,
            }]).encode("utf-8")
            source_path.write_bytes(source_bytes)
            report["additionalPromptSources"] = {
                "recovery_observations": {
                    "file": source_path.name,
                    "sha256": hashlib.sha256(source_bytes).hexdigest(),
                    "promptCount": 1,
                }
            }
            report_path.write_text(json.dumps(report), encoding="utf-8")
            validate_qwen_real_activation_artifacts(output_dir)

            source_path.write_text(json.dumps([{
                "promptId": prompt_id,
                "text": "A changed prompt.",
            }]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "additional prompt source SHA-256 mismatch"):
                validate_qwen_real_activation_artifacts(output_dir)


def main() -> None:
    suite = unittest.TestSuite([
        unittest.defaultTestLoader.loadTestsFromTestCase(OracleOnlyCubicRecovery),
        unittest.defaultTestLoader.loadTestsFromTestCase(QwenSwiGLUOracleContracts),
        unittest.defaultTestLoader.loadTestsFromTestCase(AdaptiveOracleSwiGLUDistillation),
        unittest.defaultTestLoader.loadTestsFromTestCase(QwenActivationArtifactVerification),
    ])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    print("cubic oracle-only recovery passed")
    print("Qwen SwiGLU recovery contract tests passed")
    print("Qwen activation artifact verification tests passed")


if __name__ == "__main__":
    main()
