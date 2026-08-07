#!/usr/bin/env python3
"""Tests that must fail before the repair, and pass after.

Written from the audit register in `~/.claude/plans/wondrous-coalescing-cookie.md`. Each one
pins a defect that survived a full day of work because nothing could see it, and each is
written to fail against the code as it stood when the audit was made. A test added after a fix
proves the fix; a test that fails first proves the defect was real.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.llm_reader.jacobian_boundary_recovery import count_fully_recovered  # noqa: E402


class GateSignBlindSpotTest(unittest.TestCase):
    """F20 — the measure that certifies a recovery cannot see a reversed gate.

    `silu` is flat on one side of the knee and linear on the other, so a unit whose gate points
    the wrong way fires exactly when the real one is silent. The recovery was taught to get the
    sign right earlier the same day - 39 of 64 to 64 of 64 - but the measure that signs off on
    it takes an absolute cosine and cannot tell the two apart.
    """

    @staticmethod
    def block(units: int = 4, reverse_gate_of: int | None = None, flip_up_of: int | None = None):
        """A perfect recovery, optionally with one unit's gate or up direction reversed."""
        rng = np.random.default_rng(3)
        truth_gate = rng.normal(size=(units, 16))
        truth_up = rng.normal(size=(units, 16))
        truth_out = rng.normal(size=(units, 16))
        gate, up, out = truth_gate.copy(), truth_up.copy(), truth_out.copy()
        if reverse_gate_of is not None:
            gate[reverse_gate_of] *= -1.0
        if flip_up_of is not None:
            # Both together, which is the gauge freedom, not an error.
            up[flip_up_of] *= -1.0
            out[flip_up_of] *= -1.0
        return dict(
            recovered_gates=gate, truth_gates=truth_gate,
            recovered_ups=up, truth_ups=truth_up,
            recovered_outputs=out, truth_outputs=truth_out,
            gate_norms=np.linalg.norm(truth_gate, axis=1),
        )

    def test_a_clean_block_counts_every_unit(self) -> None:
        self.assertEqual(count_fully_recovered(**self.block()), 4)

    def test_a_reversed_gate_is_not_counted_as_recovered(self) -> None:
        # `silu` is flat on one side of the knee, so this unit is silent exactly when the real
        # one fires. Counting it is the failure the audit found.
        self.assertEqual(count_fully_recovered(**self.block(reverse_gate_of=0)), 3)

    def test_flipping_up_and_output_together_is_not_an_error(self) -> None:
        # `d_i silu(g.x)(u.x)` is unchanged when `u` and `d` flip together, so this is a gauge
        # choice and nothing can observe it. Refusing it would be a false negative.
        self.assertEqual(count_fully_recovered(**self.block(flip_up_of=1)), 4)


if __name__ == "__main__":
    unittest.main()


class SwallowedFailureTest(unittest.TestCase):
    """F01, F07, F08 — a broken card must not look like a working one.

    `solve_least_squares` used to wrap the whole GPU path in `except Exception: pass`. An out
    of memory, a driver fault, a shape error - all of them turned into a run that was merely
    slower and said nothing at all. `operator-integrity-lock` forbids exactly this: a failure
    that is silenced rather than repaired.

    One condition is a legitimate answer rather than a breakdown: CUDA's `gels` driver needs
    full rank, and a rank-deficient design is a real case here because near-collinear units are
    one of the negative cases this route must handle. That one is caught by name, and it still
    leaves a trace.
    """

    @staticmethod
    def source() -> str:
        return (REPO_ROOT / "tools/llm_reader/jacobian_boundary_recovery.py").read_text()

    def test_no_blanket_exception_swallows_a_failure(self) -> None:
        body = self.source()

        # Comments explaining the removed anti-pattern are allowed; code doing it is not.
        offending = [
            line for line in body.splitlines()
            if "except Exception" in line and not line.lstrip().startswith("#")
        ]

        self.assertEqual(offending, [])

    def test_a_fallback_leaves_a_trace(self) -> None:
        from tools.llm_reader.jacobian_boundary_recovery import _record_fallback

        notes: list[str] = []
        _record_fallback(notes, "card returned a non-finite solution")

        # Without this the run reports a clean result produced by a path nobody chose.
        self.assertEqual(len(notes), 1)
        self.assertIn("non-finite", notes[0])

    def test_absence_of_a_card_is_not_treated_as_a_fault(self) -> None:
        import ast

        tree = ast.parse(self.source())
        probe = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "_torch_cuda"
        )
        caught = [
            ast.unparse(handler.type) if handler.type else "bare"
            for block in ast.walk(probe) if isinstance(block, ast.Try)
            for handler in block.handlers
        ]

        # "No torch installed" is a state, not a failure, and it is the only thing this may
        # absorb. A driver that raises anything else has to surface, because a broken card and
        # an absent one are different situations that used to look identical.
        self.assertEqual(caught, ["ImportError"])


@unittest.skipUnless(
    Path("/home/kugli/Downloads/Genetic-Code-Lab/genetic-code-lab/.gcl-python-gpu").exists(),
    "needs the GPU environment that carries the model",
)
class OracleProtocolIntegrationTest(unittest.TestCase):
    """F25 — every other test in this suite talks to a synthetic teacher.

    Twenty-nine tests cover the mathematics and not one of them crosses the process boundary:
    no oracle protocol, no file handoff, no truth lock. Three of the defects this audit found -
    a lane whose artifacts had gone stale, a verification branch that had never run, and a
    diagnostic pass that rewrote 169 files in other lanes - all lived in exactly that gap.

    This one goes through `OracleClient` against the real model, on the smallest block that
    still exercises the whole path. It is slow, and that is the price of covering the part
    where the failures actually were.
    """

    MODEL = "unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit"

    def test_the_truth_lock_refuses_before_the_recovery_is_frozen(self) -> None:
        import tempfile

        from tools.llm_reader.blind_swiglu_oracle_decompiler import OracleClient
        from tools.llm_reader.jacobian_boundary_recovery import _oracle_namespace

        with tempfile.TemporaryDirectory() as work:
            oracle = OracleClient(
                _oracle_namespace(
                    model=self.MODEL, layer=15, units="8,23,43,98",
                    device="cuda", max_oracle_calls=20000, batch_size=256,
                ),
                Path(work),
            )
            try:
                # The invariant the whole lane rests on, exercised rather than declared: the
                # oracle will not hand over the weights until the solver is done with it.
                with self.assertRaises(RuntimeError):
                    oracle.truth()

                probe = np.zeros((4, int(oracle.meta["hiddenSize"])), dtype=np.float32)
                answered = oracle.eval(probe, stage="integration-test")
                self.assertEqual(answered.shape[0], 4)
                served = int(oracle.query_count)

                oracle.unlock_posthoc_truth("integration-test-frozen")
                gate, up = oracle.truth()

                # The two counters the lane compares, checked here against the protocol rather
                # than against a report the tool wrote about itself.
                self.assertEqual(oracle.truth_unlock_query_count, served)
                self.assertEqual(gate.shape[0], 4)
                self.assertEqual(up.shape[0], 4)
            finally:
                oracle.close()

    def test_a_second_unlock_is_refused(self) -> None:
        import tempfile

        from tools.llm_reader.blind_swiglu_oracle_decompiler import OracleClient
        from tools.llm_reader.jacobian_boundary_recovery import _oracle_namespace

        with tempfile.TemporaryDirectory() as work:
            oracle = OracleClient(
                _oracle_namespace(
                    model=self.MODEL, layer=15, units="8,23",
                    device="cuda", max_oracle_calls=8000, batch_size=128,
                ),
                Path(work),
            )
            try:
                oracle.unlock_posthoc_truth("first")
                # Unlocking twice would make the recorded query count meaningless, which is the
                # number the access evidence rests on.
                with self.assertRaises(RuntimeError):
                    oracle.unlock_posthoc_truth("second")
            finally:
                oracle.close()


class CollinearOutputDirectionsTest(unittest.TestCase):
    """F28 — the identity this route clusters by was the one axis nobody checked.

    The verdict inspected gate collinearity and anti-parallel outputs, but not parallel ones.
    Two units whose output directions nearly coincide either merge into one cluster - the
    output direction is how crossings are grouped - or return as near-parallel columns whose
    pseudo-inverse amplifies noise by the condition number. Measured: a pair at cosine 0.999
    took cond(D) to 6968 and the up recovery to one unit in six, while the verdict said
    `recovered` with nothing to report.
    """

    @staticmethod
    def verdict_for(output_cosine: float):
        from tools.llm_reader.jacobian_boundary_recovery import identifiability_verdict

        outputs = np.eye(4)
        outputs[1] = output_cosine * outputs[0] + np.sqrt(1 - output_cosine**2) * np.eye(4)[1]
        return identifiability_verdict(
            basis_spectrum=np.ones(4), active_rank=4,
            gates=np.eye(4), output_directions=outputs, clusters_found=4,
            duplicate_gate_cosine=0.98, collinear_threshold=0.9,
            weak_share_threshold=0.002, rank_energy_threshold=0.98, unit_budget=4,
        )

    def test_parallel_outputs_are_reported_as_ambiguous(self) -> None:
        result = self.verdict_for(0.999)

        families = {row["family"] for row in result["findings"]}
        self.assertIn("near-collinear-output-directions", families)
        self.assertEqual(result["verdict"], "qualified")

    def test_distinct_outputs_raise_nothing(self) -> None:
        result = self.verdict_for(0.2)

        self.assertEqual(result["findings"], [])


class ChunkedSolveTest(unittest.TestCase):
    """A design that does not fit the card must still be solved, and solved correctly.

    The 1024-unit up design is 100000 by 4096 - 1.6 GB plus the QR workspace - on a card
    where the oracle holds 2.5 of 7.65 GB. The F01 repair made that die honestly instead of
    finishing silently on the host; this is the half that gets the work done: the Gram matrix
    is `cols x cols` however many rows there are, so it is accumulated in chunks.

    Correctness is against numpy's fp64 lstsq on the same system, because normal equations
    square the condition number and this file has been burned by that twice. The chunked path
    is exercised directly rather than by faking an OOM, since what needs proving is the
    arithmetic, not the routing.
    """

    @unittest.skipUnless(
        Path("/home/kugli/Downloads/Genetic-Code-Lab/genetic-code-lab/.gcl-python-gpu").exists(),
        "needs the GPU environment",
    )
    def test_the_chunked_gram_agrees_with_direct_least_squares(self) -> None:
        import torch

        from tools.llm_reader.jacobian_boundary_recovery import _chunked_normal_equations

        if not torch.cuda.is_available():
            self.skipTest("no card")
        rng = np.random.default_rng(7)
        design = rng.normal(size=(20_000, 96))
        # Wildly different column scales on purpose - the silu-gated half of the real design
        # is orders of magnitude off the linear half, and unnormalised normal equations would
        # pass a test with friendly columns and fail on the real ones.
        design *= np.geomspace(1e-3, 1e3, 96)[None, :]
        truth = rng.normal(size=96)
        target = design @ truth + 0.01 * rng.normal(size=20_000)

        chunked = _chunked_normal_equations(torch, design, target, chunk_rows=3_000)
        direct, *_ = np.linalg.lstsq(design, target, rcond=None)

        worst = float(np.max(np.abs(chunked - direct) / np.maximum(np.abs(direct), 1e-12)))
        self.assertLess(worst, 1e-4)

    def test_the_memory_estimate_routes_before_allocating(self) -> None:
        # The routing constant is what stands between a working run and the OOM that killed
        # 1024 units: it must include the workspace, not only the design.
        source = (REPO_ROOT / "tools/llm_reader/jacobian_boundary_recovery.py").read_text()
        self.assertIn("torch.cuda.mem_get_info()", source)
        self.assertIn("2.5 * design.size * 4", source)


@unittest.skipUnless(
    Path("/home/kugli/Downloads/Genetic-Code-Lab/genetic-code-lab/.gcl-python-gpu").exists(),
    "needs the GPU environment that carries the model",
)
class OracleBudgetTest(unittest.TestCase):
    """The last empty cell of the coverage matrix: what happens at the call budget.

    A budget that truncates quietly is worse than no budget, because the recovery then runs on
    a response block that is short and says nothing about it - and every measure downstream,
    including the blind reconstruction, would be computed over data that was cut off. Measured:
    it raises, names the stage, and leaves the counter where it was, so nothing partial is
    handed back.
    """

    def test_exceeding_the_budget_raises_rather_than_truncating(self) -> None:
        import tempfile

        from tools.llm_reader.blind_swiglu_oracle_decompiler import OracleClient
        from tools.llm_reader.jacobian_boundary_recovery import _oracle_namespace

        with tempfile.TemporaryDirectory() as work:
            oracle = OracleClient(
                _oracle_namespace(
                    model="unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit", layer=15,
                    units="8,23,43,98", device="cuda", max_oracle_calls=2000, batch_size=256,
                ),
                Path(work),
            )
            try:
                width = int(oracle.meta["hiddenSize"])
                oracle.eval(np.zeros((1500, width), dtype=np.float32), stage="budget-probe")
                self.assertEqual(oracle.query_count, 1500)

                with self.assertRaises(Exception) as caught:
                    oracle.eval(np.zeros((900, width), dtype=np.float32), stage="budget-probe")

                message = str(caught.exception)
                self.assertIn("budget-probe", message)
                self.assertIn("2000", message)
                # Nothing was served, so nothing partial reached the solver.
                self.assertEqual(oracle.query_count, 1500)
            finally:
                oracle.close()


class ChunkedPathAgreesEndToEndTest(unittest.TestCase):
    """The chunked solve must return the same block, not merely the same arithmetic.

    Its unit test compares against numpy on a synthetic system, which proves the algebra and
    nothing about the route. The gap was structural: the chunked path only engages where the
    direct solve does not fit, so the only block that reached it was one that had never been
    recovered at all - there was nothing to compare against. `GCL_FORCE_CHUNKED_SOLVE` makes
    it reachable on a block with a known answer.

    Verified on the live 128-unit block as well: 128/128 fully recovered either way, held-out
    R2 0.9994209 to seven places both times, identical call count. This is the fast version of
    that check.
    """

    @staticmethod
    def teacher(units: int = 5, dim: int = 32, seed: int = 4):
        rng = np.random.default_rng(seed)
        gate = rng.normal(size=(units, dim)) / np.sqrt(dim)
        up = rng.normal(size=(units, dim)) / np.sqrt(dim)
        down = rng.normal(size=(dim, units))

        def evaluate(points: np.ndarray) -> np.ndarray:
            z = points @ gate.T
            return ((z / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))) * (points @ up.T)) @ down.T

        return evaluate, up

    def recover(self):
        from tools.llm_reader.jacobian_boundary_recovery import recover_block

        evaluate, up = self.teacher()
        result, _telemetry = recover_block(
            evaluate, input_dim=32, active_rank=10, line_count=50, samples=401,
            scale=2e4, step=1e-4, peak_fraction=0.05, separation=6, min_sharpness=3.0,
            cluster_min_cosine=0.95, basis_points_count=8, basis_scale=1.0,
            gate_mode="amplitude", amplitude_span=0.01, amplitude_min_alignment=0.5,
            duplicate_gate_cosine=0.98, gate_residual_tolerance=1.0, amplitude_crossings=4,
            min_cluster_size=1, up_sample_count=4000, up_sample_scale=2.0, seed=5,
        )
        return result, up

    @unittest.skipUnless(
        Path("/home/kugli/Downloads/Genetic-Code-Lab/genetic-code-lab/.gcl-python-gpu").exists(),
        "needs the GPU environment",
    )
    def test_both_solve_paths_recover_the_same_block(self) -> None:
        import os

        import torch

        if not torch.cuda.is_available():
            self.skipTest("no card")

        direct, _up = self.recover()
        os.environ["GCL_FORCE_CHUNKED_SOLVE"] = "1"
        try:
            chunked, _up = self.recover()
        finally:
            os.environ.pop("GCL_FORCE_CHUNKED_SOLVE", None)

        self.assertEqual(direct["upDirections"].shape, chunked["upDirections"].shape)
        unit = lambda rows: rows / np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-30)
        agreement = np.abs(np.sum(unit(direct["upDirections"]) * unit(chunked["upDirections"]), axis=1))
        self.assertGreater(float(np.min(agreement)), 0.9999)
        self.assertAlmostEqual(
            direct["reconstruction"]["rSquared"], chunked["reconstruction"]["rSquared"], places=6
        )


class WeakChannelFloorScalesTest(unittest.TestCase):
    """F31 — an absolute energy floor means something different at every block size.

    A unit's share of the block's gate energy is about `1/k`, so a fixed floor of 0.002 is
    generous at sixteen units and impossible past five hundred. The family's first live firing
    said so: 1023 of 1023 solutions flagged on a 1024-unit block that had recovered 1022 of
    them. The check now asks whether a solution carries far less than its peers.
    """

    @staticmethod
    def verdict_for(units: int, weak_units: int = 0):
        from tools.llm_reader.jacobian_boundary_recovery import identifiability_verdict

        gates = np.eye(units)
        for index in range(weak_units):
            gates[index] *= 0.01
        return identifiability_verdict(
            basis_spectrum=np.ones(units), active_rank=units,
            gates=gates, output_directions=np.eye(units), clusters_found=units,
            duplicate_gate_cosine=0.999, collinear_threshold=0.9,
            weak_share_threshold=0.05, rank_energy_threshold=0.98, unit_budget=units,
        )

    def families(self, result) -> set:
        return {row["family"] for row in result["findings"]}

    def test_a_large_uniform_block_is_not_flagged(self) -> None:
        # This is the case that broke: every unit at 1/1024 of the energy, all of them
        # correctly recovered, and the old floor called every one of them weak.
        self.assertNotIn("weak-channel-below-floor", self.families(self.verdict_for(1024)))

    def test_a_genuinely_weak_unit_is_still_caught(self) -> None:
        result = self.verdict_for(1024, weak_units=3)

        self.assertIn("weak-channel-below-floor", self.families(result))
        detail = [r["detail"] for r in result["findings"] if r["family"] == "weak-channel-below-floor"]
        self.assertIn("3 solution(s)", detail[0])

    def test_the_same_floor_behaves_the_same_at_every_size(self) -> None:
        # Scale freedom is the point: one weak unit in sixteen and one in a thousand are the
        # same statement, and the check has to make it the same way.
        for units in (16, 128, 1024):
            with self.subTest(units=units):
                self.assertIn(
                    "weak-channel-below-floor",
                    self.families(self.verdict_for(units, weak_units=1)),
                )
                self.assertNotIn("weak-channel-below-floor", self.families(self.verdict_for(units)))


class MixingInvertibilityTest(unittest.TestCase):
    """Unmixing the output channels is only defined while there are channels to spare.

    `R = A D` with `D` shaped units by output_dim. Below the boundary `D` has full row rank and
    the pseudo-inverse recovers the activations exactly; above it the system is wide, each row
    of `A` has more unknowns than there are equations, and `pinv` returns the minimum-norm
    member of an infinite family - a fabrication, not a measurement.

    That boundary is where a whole layer lives: 8192 units into 3072 channels. Measured on a
    32-dimensional teacher, held-out R2 with unmixing against projection: 0.9999 vs 0.9893 at
    12 units, 0.9076 vs 0.9253 at 43, 0.6154 vs 0.8456 at 86. It wins where it is defined and
    loses badly where it is not.
    """

    def test_a_narrow_full_rank_mixing_is_inverted(self) -> None:
        from tools.llm_reader.jacobian_boundary_recovery import recover_block

        _result, telemetry = self._recover(units=12, dim=32)
        self.assertTrue(any("mixing inverted" in note for note in telemetry.notes))

    def test_more_units_than_channels_falls_back_to_projection(self) -> None:
        _result, telemetry = self._recover(units=86, dim=32)
        self.assertTrue(any("more units than channels" in note for note in telemetry.notes))

    def test_the_rank_comes_from_the_matrix_not_the_count(self) -> None:
        # Near-collinear outputs make a narrow matrix behave like a wide one, and counting rows
        # would miss it. This is the same condition finding F28 reports as ambiguity.
        source = (REPO_ROOT / "tools/llm_reader/jacobian_boundary_recovery.py").read_text()
        self.assertIn("np.linalg.matrix_rank(output_directions)", source)

    @staticmethod
    def _recover(units: int, dim: int):
        from tools.llm_reader.jacobian_boundary_recovery import recover_block

        rng = np.random.default_rng(3)
        gate = rng.normal(size=(units, dim)) / np.sqrt(dim)
        up = rng.normal(size=(units, dim)) / np.sqrt(dim)
        down = rng.normal(size=(dim, units))

        def evaluate(points: np.ndarray) -> np.ndarray:
            z = points @ gate.T
            return ((z / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))) * (points @ up.T)) @ down.T

        return recover_block(
            evaluate, input_dim=dim, active_rank=dim, line_count=120, samples=401,
            scale=2e4, step=1e-4, peak_fraction=0.05, separation=6, min_sharpness=3.0,
            cluster_min_cosine=0.95, basis_points_count=16, basis_scale=1.0,
            gate_mode="amplitude", amplitude_span=0.01, amplitude_min_alignment=0.5,
            duplicate_gate_cosine=0.98, gate_residual_tolerance=1.0, amplitude_crossings=3,
            min_cluster_size=4, up_sample_count=4000, up_sample_scale=2.0, seed=5,
        )


class JointUpSolveTest(unittest.TestCase):
    """Above the channel boundary the units can only be separated together.

    Reading a unit through its own output direction carries every other unit scaled by
    `d_j . d_i`. Inverting that mixing removes it exactly, but only while there are more output
    channels than units - and a whole layer is 8192 units into 3072 channels. Above the
    boundary the per-unit route has nothing but the raw projection, where the leak grows as
    `sqrt(k/dim)` and passes one.

    Solving for every `u_j` together removes it, and the system has `k * rank` unknowns - 25
    million for a layer - which is why it was written off. Conjugate gradient never needs the
    matrix, only its action, and the action is two matrix products.
    """

    @staticmethod
    def block(units: int, dim: int, rows: int = 4000, seed: int = 1):
        rng = np.random.default_rng(seed)
        gate = rng.normal(size=(units, dim)) / np.sqrt(dim)
        up = rng.normal(size=(units, dim)) / np.sqrt(dim)
        down = rng.normal(size=(units, dim))
        down /= np.linalg.norm(down, axis=1, keepdims=True)
        points = rng.normal(0.0, 2.0, size=(rows, dim))
        z = points @ gate.T
        activation = z / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))
        return activation, points, (activation * (points @ up.T)) @ down, down, up

    def test_it_recovers_every_unit_where_projection_recovers_none(self) -> None:
        from tools.llm_reader.jacobian_boundary_recovery import solve_up_directions_jointly

        activation, points, responses, down, up = self.block(units=86, dim=32)

        solved = solve_up_directions_jointly(activation, points, responses, down)

        unit = lambda rows: rows / np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-30)
        agreement = np.abs(np.sum(unit(solved) * unit(up), axis=1))
        # The per-unit projection gets none of these above 0.99, median 0.9489.
        self.assertGreater(float(np.min(agreement)), 0.99)

    def test_it_is_exact_below_the_boundary_too(self) -> None:
        from tools.llm_reader.jacobian_boundary_recovery import solve_up_directions_jointly

        activation, points, responses, down, up = self.block(units=12, dim=32)

        solved = solve_up_directions_jointly(activation, points, responses, down)

        unit = lambda rows: rows / np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-30)
        self.assertGreater(float(np.min(np.abs(np.sum(unit(solved) * unit(up), axis=1)))), 0.99)

    def test_the_matrix_is_never_formed(self) -> None:
        # 25 million unknowns for a layer. If this ever starts building the operator the memory
        # will say so long before the arithmetic does.
        source = (REPO_ROOT / "tools/llm_reader/jacobian_boundary_recovery.py").read_text()
        body = source[source.index("def solve_up_directions_jointly") : source.index("def recover_block")]
        self.assertNotIn("np.kron", body)
        self.assertIn("chunk_rows", body)
