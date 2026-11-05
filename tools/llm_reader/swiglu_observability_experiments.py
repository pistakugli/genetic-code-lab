from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np


PROTOCOLS = (
    "iid_dense",
    "coordinate_erasure",
    "complementary_blocks",
    "activation_environments",
)
TEACHER_CASES = (
    "generic",
    "overlapping",
    "weak_components",
    "rank1_output",
    "rank3_output",
    "duplicate_gate",
    "null_component",
)
DEFAULT_SEEDS = (17, 29, 43)
DEFAULT_QUERY_BUDGET = 384
TRAINING_FRACTION = 2.0 / 3.0
POSTPROJECTION_IMPROVEMENT_THRESHOLD = 0.10
REQUIRED_SEED_FRACTION = 2.0 / 3.0


HYPOTHESES: dict[str, dict[str, str]] = {
    "coordinate_erasure": {
        "mathematical_reason": (
            "For a coordinate mask M, each component sees (g_i^T Mx, u_i^T Mx) "
            "rather than the same dense projections. A family of masks changes "
            "which coordinate groups jointly drive a gate and its value factor."
        ),
        "expected_observable_signature": (
            "Across mask-coded oracle inputs, post-hoc per-component response "
            "fingerprints may have lower nearest-neighbor cosine than IID dense "
            "queries, and the change should persist on an independent mask seed."
        ),
        "failure_conditions": (
            "Dense or near-collinear projections respond similarly to most masks; "
            "the down projection merges their response fingerprints; or masking "
            "reduces signal-to-noise without adding selectivity."
        ),
        "artifact_criterion": (
            "At equal query count, both the median and P95 nearest absolute cosine "
            "of held-out post-projection component fingerprints fall by at least "
            "0.10 and 0.05 respectively versus IID on at least two of three seeds."
        ),
    },
    "complementary_blocks": {
        "mathematical_reason": (
            "Restricting each query to one of disjoint coordinate blocks exposes "
            "the restrictions of g_i and u_i to different input subspaces; "
            "components with different block energy may become selectively "
            "observable without a derivative or large-amplitude limit."
        ),
        "expected_observable_signature": (
            "A component's response persists or disappears consistently across "
            "block cohorts, producing a reproducible block-response fingerprint "
            "not present in an IID query cohort."
        ),
        "failure_conditions": (
            "All teacher projections are dense and similarly distributed across "
            "blocks, or block restriction attenuates too many components at once."
        ),
        "artifact_criterion": (
            "The same held-out median/P95 post-projection nearest-cosine improvement "
            "threshold as the erasure hypothesis, with nonzero oracle-output energy "
            "in every held-out block cohort."
        ),
    },
    "activation_environments": {
        "mathematical_reason": (
            "The component response is a nonlinear ridge function of two "
            "projections. Sparse, dense, and correlated input laws change the "
            "joint distribution of those projections and can expose components "
            "that are suppressed or coactivated under one environment."
        ),
        "expected_observable_signature": (
            "Per-component response fingerprints remain distinguishable across "
            "several independently sampled input environments rather than being "
            "an artifact of one isotropic distribution."
        ),
        "failure_conditions": (
            "Distribution shifts only rescale common responses, hide weak units, "
            "or change the fingerprint ordering in a way that cannot be matched "
            "across environments from the summed output."
        ),
        "artifact_criterion": (
            "The held-out median/P95 post-projection nearest-cosine improvement "
            "versus IID is at least 0.10/0.05 on at least two of three seeds, and "
            "the median improvement is positive in both sparse and correlated "
            "held-out subcohorts."
        ),
    },
}


@dataclass(frozen=True)
class SyntheticSwiGLUTeacher:
    gate: np.ndarray
    up: np.ndarray
    down: np.ndarray

    @property
    def input_dim(self) -> int:
        return int(self.gate.shape[1])

    @property
    def output_dim(self) -> int:
        return int(self.down.shape[0])

    @property
    def hidden_width(self) -> int:
        return int(self.gate.shape[0])

    def validate(self) -> None:
        if self.gate.ndim != 2 or self.up.shape != self.gate.shape:
            raise ValueError("gate/up weights must be matching matrices")
        if self.down.ndim != 2 or self.down.shape[1] != self.hidden_width:
            raise ValueError("down weights must have one column per component")
        if not all(
            np.isfinite(value).all() for value in (self.gate, self.up, self.down)
        ):
            raise ValueError("teacher weights must be finite")

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        values = np.asarray(inputs, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.input_dim:
            raise ValueError("input matrix has incompatible dimensions")
        if not np.isfinite(values).all():
            raise ValueError("inputs must be finite")
        gate_values = values @ self.gate.T
        up_values = values @ self.up.T
        silu = gate_values / (1.0 + np.exp(-np.clip(gate_values, -700.0, 700.0)))
        return (silu * up_values) @ self.down.T

    def component_activations(self, inputs: np.ndarray) -> np.ndarray:
        values = np.asarray(inputs, dtype=np.float64)
        gate_values = values @ self.gate.T
        up_values = values @ self.up.T
        silu = gate_values / (1.0 + np.exp(-np.clip(gate_values, -700.0, 700.0)))
        return silu * up_values


class FullModuleSyntheticOracle:
    """Callable full-function boundary with dimensions but no unit-count metadata."""

    def __init__(self, teacher: SyntheticSwiGLUTeacher) -> None:
        teacher.validate()
        self.input_dim = teacher.input_dim
        self.output_dim = teacher.output_dim
        self.__forward: Callable[[np.ndarray], np.ndarray] = teacher.forward
        self.query_count = 0

    def query(self, inputs: np.ndarray) -> np.ndarray:
        values = np.asarray(inputs, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.input_dim:
            raise ValueError("oracle input dimensions are invalid")
        outputs = self.__forward(values)
        if outputs.shape != (len(values), self.output_dim):
            raise RuntimeError("full-module oracle returned an invalid shape")
        self.query_count += len(values)
        return outputs


@dataclass(frozen=True)
class OracleObservations:
    protocol: str
    training_inputs: np.ndarray
    training_outputs: np.ndarray
    heldout_inputs: np.ndarray
    heldout_outputs: np.ndarray
    training_queries: int
    heldout_queries: int
    total_queries: int


def _dimensions(seed_index: int) -> tuple[int, int, int]:
    return ((8, 5, 6), (10, 6, 8), (14, 8, 12))[seed_index % 3]


def make_synthetic_teacher(
    case: str,
    *,
    input_dim: int,
    output_dim: int,
    width: int,
    seed: int,
) -> SyntheticSwiGLUTeacher:
    if case not in TEACHER_CASES:
        raise ValueError(f"unknown teacher case {case!r}")
    if min(input_dim, output_dim, width) <= 0:
        raise ValueError("teacher dimensions must be positive")
    if case in {"duplicate_gate", "overlapping", "weak_components", "null_component"} and width < 2:
        raise ValueError(f"teacher case {case!r} requires width >= 2")
    rng = np.random.default_rng(seed)
    gate = rng.normal(0.0, 0.7, size=(width, input_dim))
    up = rng.normal(0.0, 0.7, size=(width, input_dim))
    down = rng.normal(0.0, 0.5, size=(output_dim, width))

    if case == "overlapping":
        gate[1] = gate[0] + rng.normal(0.0, 1e-3, size=input_dim)
        up[1] = up[0] + rng.normal(0.0, 1e-3, size=input_dim)
    elif case == "weak_components":
        weak_count = max(1, width // 4)
        down[:, -weak_count:] *= 1e-6
    elif case == "rank1_output":
        down = rng.normal(size=(output_dim, 1)) @ rng.normal(size=(1, width))
    elif case == "rank3_output":
        rank = min(3, output_dim, width)
        down = rng.normal(size=(output_dim, rank)) @ rng.normal(size=(rank, width))
    elif case == "duplicate_gate":
        gate[1] = gate[0]
    elif case == "null_component":
        down[:, -1] = 0.0

    teacher = SyntheticSwiGLUTeacher(
        gate=gate.astype(np.float64),
        up=up.astype(np.float64),
        down=down.astype(np.float64),
    )
    teacher.validate()
    return teacher


def _input_family(
    protocol: str,
    *,
    input_dim: int,
    count: int,
    seed: int,
) -> np.ndarray:
    if protocol not in PROTOCOLS:
        raise ValueError(f"unknown measurement protocol {protocol!r}")
    if input_dim <= 0 or count <= 0:
        raise ValueError("input dimension and sample count must be positive")
    rng = np.random.default_rng(seed)
    if protocol == "iid_dense":
        return rng.normal(size=(count, input_dim))

    if protocol == "coordinate_erasure":
        base = rng.normal(size=(count, input_dim))
        keep_rates = np.asarray([0.25, 0.5, 0.75], dtype=np.float64)
        rates = keep_rates[np.arange(count) % len(keep_rates)]
        masks = rng.random(size=(count, input_dim)) < rates[:, None]
        return base * masks / np.sqrt(rates[:, None])

    if protocol == "complementary_blocks":
        blocks = np.array_split(np.arange(input_dim), min(4, input_dim))
        inputs = np.zeros((count, input_dim), dtype=np.float64)
        block_ids = np.arange(count) % len(blocks)
        for block_index, coordinates in enumerate(blocks):
            rows = np.flatnonzero(block_ids == block_index)
            if len(rows):
                inputs[np.ix_(rows, coordinates)] = rng.normal(
                    size=(len(rows), len(coordinates))
                ) * math.sqrt(len(blocks))
        return inputs

    inputs = np.empty((count, input_dim), dtype=np.float64)
    environment_ids = np.arange(count) % 3
    sparse_rows = np.flatnonzero(environment_ids == 0)
    dense_rows = np.flatnonzero(environment_ids == 1)
    correlated_rows = np.flatnonzero(environment_ids == 2)
    if len(sparse_rows):
        values = rng.normal(size=(len(sparse_rows), input_dim))
        mask = rng.random(size=values.shape) < 0.2
        inputs[sparse_rows] = values * mask / math.sqrt(0.2)
    if len(dense_rows):
        inputs[dense_rows] = rng.normal(size=(len(dense_rows), input_dim))
    if len(correlated_rows):
        basis, _ = np.linalg.qr(rng.normal(size=(input_dim, input_dim)))
        spectrum = np.geomspace(1.0, 0.05, num=input_dim)
        transform = basis * np.sqrt(spectrum)[None, :]
        inputs[correlated_rows] = rng.normal(
            size=(len(correlated_rows), input_dim)
        ) @ transform.T
    return inputs


def measure_protocol(
    oracle_query: Callable[[np.ndarray], np.ndarray],
    *,
    protocol: str,
    oracle_input_dim: int,
    oracle_output_dim: int,
    query_budget: int,
    seed: int,
) -> OracleObservations:
    """Issue a budget-matched protocol without teacher or hidden-width arguments."""
    if oracle_input_dim <= 0 or oracle_output_dim <= 0:
        raise ValueError("oracle dimensions must be positive")
    if query_budget < 3 or query_budget % 3:
        raise ValueError("query_budget must be a positive multiple of three >= 3")
    training_queries = 2 * query_budget // 3
    heldout_queries = query_budget - training_queries
    training_inputs = _input_family(
        protocol,
        input_dim=oracle_input_dim,
        count=training_queries,
        seed=seed,
    )
    heldout_inputs = _input_family(
        protocol,
        input_dim=oracle_input_dim,
        count=heldout_queries,
        seed=seed + 1_000_003,
    )
    training_outputs = np.asarray(oracle_query(training_inputs), dtype=np.float64)
    heldout_outputs = np.asarray(oracle_query(heldout_inputs), dtype=np.float64)
    if training_outputs.shape != (training_queries, oracle_output_dim):
        raise ValueError("oracle returned an invalid training-output shape")
    if heldout_outputs.shape != (heldout_queries, oracle_output_dim):
        raise ValueError("oracle returned an invalid held-out-output shape")
    return OracleObservations(
        protocol=protocol,
        training_inputs=training_inputs,
        training_outputs=training_outputs,
        heldout_inputs=heldout_inputs,
        heldout_outputs=heldout_outputs,
        training_queries=training_queries,
        heldout_queries=heldout_queries,
        total_queries=training_queries + heldout_queries,
    )


def _nearest_absolute_cosine_summary(signatures: np.ndarray) -> dict[str, Any]:
    values = np.asarray(signatures, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("signatures must be a matrix with one component per column")
    norms = np.linalg.norm(values, axis=0)
    active = norms > 1e-20
    active_indices = np.flatnonzero(active)
    if len(active_indices) < 2:
        return {
            "componentCount": int(values.shape[1]),
            "activeComponents": int(len(active_indices)),
            "medianNearestAbsoluteCosine": None,
            "p95NearestAbsoluteCosine": None,
            "fractionWithNearestCosineAtLeast0.99": None,
        }
    normalized = values[:, active] / norms[active][None, :]
    cosines = np.abs(normalized.T @ normalized)
    np.fill_diagonal(cosines, 0.0)
    nearest = np.max(cosines, axis=1)
    return {
        "componentCount": int(values.shape[1]),
        "activeComponents": int(len(active_indices)),
        "medianNearestAbsoluteCosine": float(np.median(nearest)),
        "p95NearestAbsoluteCosine": float(np.quantile(nearest, 0.95)),
        "fractionWithNearestCosineAtLeast0.99": float(np.mean(nearest >= 0.99)),
    }


def _component_signature_metrics(
    teacher: SyntheticSwiGLUTeacher,
    inputs: np.ndarray,
) -> dict[str, Any]:
    activations = teacher.component_activations(inputs)
    before_projection = _nearest_absolute_cosine_summary(activations.T)
    projected_signatures = (
        activations.T[:, :, None] * teacher.down.T[:, None, :]
    ).reshape(teacher.hidden_width, -1).T
    after_projection = _nearest_absolute_cosine_summary(projected_signatures)
    norms = np.linalg.norm(projected_signatures, axis=0)
    maximum = float(np.max(norms)) if len(norms) else 0.0
    visibility = norms / max(maximum, 1e-30)
    return {
        "beforeOutputProjection": before_projection,
        "afterOutputProjection": after_projection,
        "relativeComponentResponseNorms": visibility.tolist(),
    }


def _rank(matrix: np.ndarray) -> int:
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    if not len(singular_values):
        return 0
    tolerance = max(matrix.shape) * np.finfo(np.float64).eps * singular_values[0]
    return int(np.sum(singular_values > tolerance))


def _functional_metrics(
    expected: np.ndarray,
    actual: np.ndarray,
) -> dict[str, float]:
    residual = np.asarray(actual, dtype=np.float64) - np.asarray(
        expected, dtype=np.float64
    )
    expected_values = np.asarray(expected, dtype=np.float64)
    denominator = max(float(np.linalg.norm(expected_values)), 1e-30)
    mean = float(np.mean(expected_values))
    variance = float(np.mean((expected_values - mean) ** 2))
    return {
        "maximumAbsoluteError": float(np.max(np.abs(residual))),
        "relativeError": float(np.linalg.norm(residual) / denominator),
        "normalizedMse": float(np.mean(residual**2) / max(variance, 1e-30)),
    }


def _equivalent_teachers(
    teacher: SyntheticSwiGLUTeacher,
    *,
    seed: int,
) -> dict[str, SyntheticSwiGLUTeacher]:
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(teacher.hidden_width)
    scales = np.exp(rng.normal(0.0, 0.5, size=teacher.hidden_width))
    permuted = SyntheticSwiGLUTeacher(
        gate=teacher.gate[permutation].copy(),
        up=(teacher.up[permutation] * scales[permutation, None]).copy(),
        down=(teacher.down[:, permutation] / scales[permutation][None, :]).copy(),
    )
    variants = {"permutation_reciprocal_scaling": permuted}

    matching_pair: tuple[int, int] | None = None
    for first in range(teacher.hidden_width):
        for second in range(first + 1, teacher.hidden_width):
            if np.array_equal(teacher.gate[first], teacher.gate[second]):
                matching_pair = (first, second)
                break
        if matching_pair:
            break
    if matching_pair is not None:
        first, second = matching_pair
        transform = np.asarray([[1.7, 0.4], [-0.3, 0.8]], dtype=np.float64)
        inverse = np.linalg.inv(transform)
        gate = teacher.gate.copy()
        up = teacher.up.copy()
        down = teacher.down.copy()
        pair = [first, second]
        down[:, pair] = down[:, pair] @ transform
        up[pair] = inverse @ up[pair]
        variants["duplicate_gate_gl2_change_of_basis"] = SyntheticSwiGLUTeacher(
            gate=gate,
            up=up,
            down=down,
        )
    if np.any(np.linalg.norm(teacher.down, axis=0) == 0.0):
        retained = np.arange(teacher.hidden_width - 1)
        variants["remove_zero_output_component"] = SyntheticSwiGLUTeacher(
            gate=teacher.gate[retained].copy(),
            up=teacher.up[retained].copy(),
            down=teacher.down[:, retained].copy(),
        )
    return variants


def _equivalence_metrics(
    teacher: SyntheticSwiGLUTeacher,
    *,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    unseen_inputs = rng.normal(size=(257, teacher.input_dim))
    target = teacher.forward(unseen_inputs)
    variants = _equivalent_teachers(teacher, seed=seed + 1)
    results: dict[str, Any] = {}
    for name, equivalent in variants.items():
        results[name] = _functional_metrics(target, equivalent.forward(unseen_inputs))
    return results


def _teacher_identifiability_label(case: str) -> str:
    return {
        "generic": "not_established_by_this_observability_study",
        "overlapping": "near_degenerate_and_potentially_ill_conditioned",
        "weak_components": "finite_precision_component_visibility_limit",
        "rank1_output": "output_rank_restricted_identifiability_not_decided",
        "rank3_output": "output_rank_restricted_identifiability_not_decided",
        "duplicate_gate": "nonunique_component_factorization_under_duplicate_gate_GL2",
        "null_component": "exactly_invisible_component_count_not_functionally_identifiable",
    }[case]


def _raw_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_synthetic_observability_study(
    output_dir: Path,
    *,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    query_budget: int = DEFAULT_QUERY_BUDGET,
) -> dict[str, Any]:
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("exactly three unique teacher seeds are required")
    if query_budget < 3 or query_budget % 3:
        raise ValueError("query_budget must be a positive multiple of three >= 3")
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite observability artifacts: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    report_cases: list[dict[str, Any]] = []
    total_oracle_points = 0
    for seed_index, seed in enumerate(seeds):
        input_dim, output_dim, width = _dimensions(seed_index)
        for case_index, case in enumerate(TEACHER_CASES):
            teacher_seed = int(seed * 1009 + case_index * 97)
            teacher = make_synthetic_teacher(
                case,
                input_dim=input_dim,
                output_dim=output_dim,
                width=width,
                seed=teacher_seed,
            )
            oracle = FullModuleSyntheticOracle(teacher)
            case_directory = output_dir / f"{case}-seed-{seed}"
            case_directory.mkdir()
            measured_protocols: list[tuple[int, str, OracleObservations, Path]] = []
            raw_files: dict[str, dict[str, str]] = {}
            for protocol_index, protocol in enumerate(PROTOCOLS):
                queries_before = oracle.query_count
                observation = measure_protocol(
                    oracle.query,
                    protocol=protocol,
                    oracle_input_dim=input_dim,
                    oracle_output_dim=output_dim,
                    query_budget=query_budget,
                    seed=int(seed * 65537 + protocol_index * 8191 + case_index),
                )
                actual_queries = oracle.query_count - queries_before
                if actual_queries != query_budget:
                    raise RuntimeError(
                        f"{protocol} used {actual_queries} oracle points, "
                        f"expected {query_budget}"
                    )
                raw_path = case_directory / f"{protocol}.npz"
                np.savez_compressed(
                    raw_path,
                    training_inputs=observation.training_inputs,
                    training_outputs=observation.training_outputs,
                    heldout_inputs=observation.heldout_inputs,
                    heldout_outputs=observation.heldout_outputs,
                )
                raw_files[protocol] = {
                    "file": str(raw_path.relative_to(output_dir)),
                    "sha256": "",
                }
                measured_protocols.append(
                    (protocol_index, protocol, observation, raw_path)
                )
                total_oracle_points += actual_queries

            # All oracle measurements are fixed before teacher-dependent analysis.
            protocol_records: list[dict[str, Any]] = []
            for protocol_index, protocol, observation, raw_path in measured_protocols:
                direct_holdout = teacher.forward(observation.heldout_inputs)
                functional = _functional_metrics(
                    direct_holdout,
                    observation.heldout_outputs,
                )
                signatures = _component_signature_metrics(
                    teacher, observation.heldout_inputs
                )
                environment_subcohorts = None
                if protocol == "activation_environments":
                    iid_observation = measured_protocols[0][2]
                    iid_rng = np.random.default_rng(
                        int(seed * 524287 + case_index * 8191)
                    )
                    available_iid_rows = np.arange(
                        iid_observation.heldout_queries
                    )
                    environment_subcohorts = {}
                    for environment_id, name in ((0, "sparse"), (2, "correlated")):
                        rows = np.flatnonzero(
                            np.arange(observation.heldout_queries) % 3
                            == environment_id
                        )
                        cohort = _component_signature_metrics(
                            teacher, observation.heldout_inputs[rows]
                        )["afterOutputProjection"]
                        median = cohort["medianNearestAbsoluteCosine"]
                        iid_rows = iid_rng.choice(
                            available_iid_rows,
                            size=len(rows),
                            replace=False,
                        )
                        available_iid_rows = available_iid_rows[
                            ~np.isin(available_iid_rows, iid_rows)
                        ]
                        iid_cohort = _component_signature_metrics(
                            teacher, iid_observation.heldout_inputs[iid_rows]
                        )["afterOutputProjection"]
                        iid_median = iid_cohort["medianNearestAbsoluteCosine"]
                        environment_subcohorts[name] = {
                            "heldoutPoints": int(len(rows)),
                            "medianNearestAbsoluteCosine": median,
                            "iidMatchedHeldoutPoints": int(len(iid_rows)),
                            "iidMedianNearestAbsoluteCosine": iid_median,
                            "improvementVsIid": (
                                float(iid_median - median)
                                if iid_median is not None and median is not None
                                else None
                            ),
                        }
                block_cohorts = None
                if protocol == "complementary_blocks":
                    blocks = np.array_split(
                        np.arange(input_dim), min(4, input_dim)
                    )
                    block_ids = (
                        np.arange(observation.heldout_queries) % len(blocks)
                    )
                    block_cohorts = []
                    for block_index, _ in enumerate(blocks):
                        rows = np.flatnonzero(block_ids == block_index)
                        block_cohorts.append({
                            "block": block_index,
                            "heldoutPoints": int(len(rows)),
                            "meanOutputNorm": float(np.mean(
                                np.linalg.norm(
                                    observation.heldout_outputs[rows], axis=1
                                )
                            )),
                        })
                readout_rng = np.random.default_rng(
                    int(seed * 131071 + protocol_index * 104729 + case_index)
                )
                readout = readout_rng.normal(
                    size=(min(3, output_dim), output_dim)
                )
                projected_outputs = observation.heldout_outputs @ readout.T
                training_readout_outputs = observation.training_outputs @ readout.T
                np.savez_compressed(
                    raw_path,
                    training_inputs=observation.training_inputs,
                    training_outputs=observation.training_outputs,
                    heldout_inputs=observation.heldout_inputs,
                    heldout_outputs=observation.heldout_outputs,
                    output_readout_matrix=readout,
                    training_readout_outputs=training_readout_outputs,
                    heldout_readout_outputs=projected_outputs,
                )
                with np.load(raw_path) as saved:
                    reconstructed_readout = (
                        saved["heldout_outputs"] @ saved["output_readout_matrix"].T
                    )
                    readout_error = float(np.max(np.abs(
                        saved["heldout_readout_outputs"] - reconstructed_readout
                    )))
                raw_files[protocol] = {
                    "file": str(raw_path.relative_to(output_dir)),
                    "sha256": _raw_file_sha256(raw_path),
                }
                protocol_records.append({
                    "protocol": protocol,
                    "queryAccounting": {
                        "requestedBudget": query_budget,
                        "trainingPoints": observation.training_queries,
                        "heldoutPoints": observation.heldout_queries,
                        "oraclePoints": observation.total_queries,
                    },
                    "heldoutFunctionalReplay": functional,
                    "posthocOnlyComponentFingerprints": signatures,
                    "activationEnvironmentSubcohortsPosthoc": environment_subcohorts,
                    "complementaryBlockCohorts": block_cohorts,
                    "outputReadoutPostprocessingControl": {
                        "derivedOnlyFromFullOutput": True,
                        "additionalOraclePoints": 0,
                        "reconstructionMaximumAbsoluteError": readout_error,
                    },
                })

            equivalences = _equivalence_metrics(
                teacher,
                seed=int(seed * 32771 + case_index),
            )
            report_cases.append({
                "case": case,
                "seed": int(seed),
                "teacherSeedForEvaluator": teacher_seed,
                "dimensions": {
                    "inputDim": input_dim,
                    "outputDim": output_dim,
                    "teacherHiddenWidthPosthocOnly": width,
                    "downProjectionRankPosthocOnly": _rank(teacher.down),
                },
                "solverInterface": {
                    "parameters": [
                        "callable_full_module_oracle",
                        "input_dimension",
                        "output_dimension",
                        "query_budget",
                    ],
                    "teacherWeightsPassedToMeasurement": False,
                    "teacherWidthPassedToMeasurement": False,
                    "componentIdentitiesPassedToMeasurement": False,
                },
                "identifiabilityClassPosthoc": _teacher_identifiability_label(case),
                "equivalentParameterizationUnseenInputMetricsPosthoc": equivalences,
                "protocols": protocol_records,
                "rawArtifacts": raw_files,
            })

    comparison: dict[str, Any] = {}
    generic_rows = [item for item in report_cases if item["case"] == "generic"]
    for protocol in PROTOCOLS[1:]:
        seed_results = []
        for row in generic_rows:
            by_protocol = {
                item["protocol"]: item for item in row["protocols"]
            }
            baseline = by_protocol["iid_dense"]["posthocOnlyComponentFingerprints"]
            candidate = by_protocol[protocol]["posthocOnlyComponentFingerprints"]
            baseline_after = baseline["afterOutputProjection"]
            candidate_after = candidate["afterOutputProjection"]
            baseline_before = baseline["beforeOutputProjection"]
            candidate_before = candidate["beforeOutputProjection"]
            median_improvement = (
                float(baseline_after["medianNearestAbsoluteCosine"])
                - float(candidate_after["medianNearestAbsoluteCosine"])
                if baseline_after["medianNearestAbsoluteCosine"] is not None
                and candidate_after["medianNearestAbsoluteCosine"] is not None
                else None
            )
            p95_improvement = (
                float(baseline_after["p95NearestAbsoluteCosine"])
                - float(candidate_after["p95NearestAbsoluteCosine"])
                if baseline_after["p95NearestAbsoluteCosine"] is not None
                and candidate_after["p95NearestAbsoluteCosine"] is not None
                else None
            )
            activation_improvement = (
                float(baseline_before["medianNearestAbsoluteCosine"])
                - float(candidate_before["medianNearestAbsoluteCosine"])
                if baseline_before["medianNearestAbsoluteCosine"] is not None
                and candidate_before["medianNearestAbsoluteCosine"] is not None
                else None
            )
            extra_criterion = True
            if protocol == "activation_environments":
                subcohorts = next(
                    item for item in row["protocols"]
                    if item["protocol"] == protocol
                )["activationEnvironmentSubcohortsPosthoc"]
                extra_criterion = bool(
                    subcohorts
                    and all(
                        metrics["improvementVsIid"] is not None
                        and metrics["improvementVsIid"] > 0.0
                        for metrics in subcohorts.values()
                    )
                )
            elif protocol == "complementary_blocks":
                cohorts = next(
                    item for item in row["protocols"]
                    if item["protocol"] == protocol
                )["complementaryBlockCohorts"]
                extra_criterion = bool(
                    cohorts
                    and all(cohort["meanOutputNorm"] > 0.0 for cohort in cohorts)
                )
            seed_results.append({
                "seed": row["seed"],
                "heldoutMedianPostprojectionCosineImprovement": median_improvement,
                "heldoutP95PostprojectionCosineImprovement": p95_improvement,
                "heldoutMedianPreprojectionCosineImprovement": activation_improvement,
                "protocolSpecificCriterionMet": extra_criterion,
                "criterionMet": bool(
                    median_improvement is not None
                    and p95_improvement is not None
                    and median_improvement >= POSTPROJECTION_IMPROVEMENT_THRESHOLD
                    and p95_improvement >= 0.05
                    and extra_criterion
                ),
            })
        passing = sum(item["criterionMet"] for item in seed_results)
        comparison[protocol] = {
            "queryBudgetMatched": True,
            "minimumMedianPostprojectionCosineImprovement": (
                POSTPROJECTION_IMPROVEMENT_THRESHOLD
            ),
            "minimumP95PostprojectionCosineImprovement": 0.05,
            "requiredPassingTeacherSeeds": math.ceil(
                len(seeds) * REQUIRED_SEED_FRACTION
            ),
            "passingTeacherSeeds": passing,
            "seedResults": seed_results,
            "syntheticInformationCriterionMet": (
                passing
                >= math.ceil(len(seeds) * REQUIRED_SEED_FRACTION)
            ),
            "interpretationBoundary": (
                "component fingerprints are post-hoc evaluator diagnostics, not "
                "features available to an oracle-only solver"
            ),
        }

    hypothesis_summary = {
        name: {
            **details,
            "syntheticInformationCriterionMet": comparison[name][
                "syntheticInformationCriterionMet"
            ],
        }
        for name, details in HYPOTHESES.items()
    }
    any_posthoc_screen_candidate = any(
        record["syntheticInformationCriterionMet"]
        for record in comparison.values()
    )
    result = {
        "schemaVersion": 1,
        "kind": "gcl-swiglu-hidden-structure-observability-study",
        "status": "synthetic_complete",
        "scope": "synthetic_only",
        "queryMechanisms": hypothesis_summary,
        "negativeControls": {
            "outputReadoutPostprocessing": {
                "hypothesis": (
                    "Adding scalar output readouts after observing the complete "
                    "output might separate components."
                ),
                "mathematicalResult": (
                    "No new information can be added: every readout is a "
                    "deterministic function of the full output already observed. "
                    "A rank-reducing readout can discard information."
                ),
                "syntheticCheck": (
                    "The verifier reconstructs random readouts from stored full "
                    "outputs and checks equality; no extra oracle calls occur."
                ),
            },
            "exactParameterEquivalences": {
                "permutation": (
                    "Permuting gate/up rows and matching down columns leaves "
                    "the full function unchanged."
                ),
                "reciprocalUpDownScaling": (
                    "Multiplying one up row by nonzero c and dividing its down "
                    "column by c leaves the function unchanged."
                ),
                "duplicateGateGL2": (
                    "For equal gate rows, only the product D_group U_group "
                    "appears; D_group R and R^-1 U_group are equivalent for "
                    "every invertible 2x2 R."
                ),
                "zeroDownColumn": (
                    "A component with a zero down column is absent from the "
                    "observable function and can be added or removed without "
                    "any query distinguishing the widths."
                ),
                "unseenInputsUsed": 257,
            },
        },
        "experimentDesign": {
            "teacherSeeds": [int(seed) for seed in seeds],
            "teacherCases": list(TEACHER_CASES),
            "protocols": list(PROTOCOLS),
            "inputOutputDimensionsVaryAcrossSeeds": True,
            "teacherWidthVariesAcrossSeeds": True,
            "queryBudgetPerTeacherProtocol": query_budget,
            "trainingFraction": TRAINING_FRACTION,
            "responseFingerprintsUseIndependentUnseenInputs": True,
            "solverReceivesTeacherWeights": False,
            "solverReceivesTeacherWidth": False,
            "solverReceivesComponentIdentities": False,
            "teacherUsedOnlyByEvaluatorAfterOracleMeasurements": True,
            "newQwenOraclePointsSpent": 0,
        },
        "queryAccounting": {
            "oraclePoints": total_oracle_points,
            "teacherCount": len(report_cases),
            "protocolCountPerTeacher": len(PROTOCOLS),
            "pointsPerProtocol": query_budget,
            "expectedOraclePoints": (
                len(report_cases) * len(PROTOCOLS) * query_budget
            ),
        },
        "posthocComparisons": comparison,
        "identifiabilityResultsPosthoc": {
            "caseLabels": {
                row["case"]: row["identifiabilityClassPosthoc"]
                for row in report_cases
            },
            "teacherWidthInferenceImplemented": False,
            "componentRecoveryImplemented": False,
            "oracleOnlyRecoveryImplemented": False,
            "oracleOnlyStructuralInformationDemonstrated": False,
            "genericStructuralIdentifiabilityProven": False,
            "exactEquivalentTeacherPairsConfirmed": all(
                metrics["relativeError"] <= 1e-13
                for row in report_cases
                for metrics in row[
                    "equivalentParameterizationUnseenInputMetricsPosthoc"
                ].values()
            ),
        },
        "qwenPilotDecision": {
            "syntheticInformationCriterionMet": any_posthoc_screen_candidate,
            "oracleOnlyRecoveryDemonstrated": False,
            "runQwenPilot": False,
            "criterion": (
                "at least two of three teacher seeds show both >=0.10 reduction "
                "in median and >=0.05 reduction in P95 nearest absolute cosine "
                "for held-out post-projection component fingerprints versus "
                "query-matched IID; this is only a post-hoc screening criterion"
            ),
            "reason": (
                "no candidate protocol crossed the preregistered post-hoc "
                "screening threshold"
                if not any_posthoc_screen_candidate
                else "post-hoc component fingerprints do not establish "
                "oracle-only recovery, and no synthetic recovery solver "
                "demonstrates the required structural signal"
            ),
        },
        "cases": report_cases,
    }
    report_path = output_dir / "study-report.json"
    report_path.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "benchmarks/research/swiglu-hidden-structure-observability"
        ),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--query-budget-per-protocol",
        type=int,
        default=DEFAULT_QUERY_BUDGET,
    )
    args = parser.parse_args()
    report = run_synthetic_observability_study(
        args.output_dir,
        seeds=tuple(args.seeds),
        query_budget=args.query_budget_per_protocol,
    )
    print(json.dumps({
        "status": report["status"],
        "outputDir": str(args.output_dir),
        "oraclePoints": report["queryAccounting"]["oraclePoints"],
        "qwenPilotDecision": report["qwenPilotDecision"],
        "hypotheses": {
            name: details["syntheticInformationCriterionMet"]
            for name, details in report["queryMechanisms"].items()
        },
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
