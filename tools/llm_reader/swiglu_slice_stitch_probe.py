"""Recover a high-rank SwiGLU sum by fitting low-rank slices of it and stitching them.

`swiglu_fit_scaling_probe.py` measured that the thing which governs recovery is the dimension the
fit sees, not the number of atoms: at 8 atoms, rank 16 recovers to MAE/scale 0.00003 and R2
1.000000 while rank 64 stalls at 0.04306 and 0.994897, and holding rank at 64 while moving atoms
from 8 to 32 barely moves the number at all. At rank 16 even 48 atoms reach R2 0.999946. So the
32-unit lane is not short of budget - three knobs were raised on it and bought nothing - it is
being asked to fit in the wrong dimension.

The reduction is exact, not an approximation. For

    f(x) = sum_i scale_i * silu(gate_i . x) * (up_i . x)

and an orthonormal `P` spanning a subspace,

    f(P s) = sum_i scale_i * silu((P^T gate_i) . s) * ((P^T up_i) . s)

which is the same model in the slice's coordinates with gates `P^T gate_i`. Restricting the probe
points to a slice is therefore a legitimate oracle query, not a different problem, and what comes
back is the projection of every atom onto that slice.

Stitching needs the projections to be attributable to the same atom across slices, and silu is not
homogeneous, so `gate` carries no scale freedom - its components are absolute. Every slice here is
built as `[core | block_j]` with one shared core, so two slices agree exactly on the core
components of the same atom, and that agreement is what the matching uses. The blocks tile the
remaining coordinates, so the matched pieces assemble into the full vector with nothing left over.

What that measured, at 32 atoms in rank 64: the slices fit beautifully and the stitching failed.
Every rank-16 slice reached R2 0.99988 to 0.99991 - exactly what the scaling probe predicted - but
32 atoms in 16 dimensions is overcomplete, so each slice returned a different equally good
decomposition, the core components agreed to only 0.59 to 0.73, and the assembled gates matched the
truth at a median cosine of 0.308. Widening the slices to rank 32 traded one for the other: cosine
0.747, slice R2 down to 0.99787. There is no setting of the two that satisfies both at 32 atoms,
because what governs the stitch is atoms per slice dimension and what governs the fit is the
dimension itself.

`SLICE_MODE=silenced` is the way out, and it uses the model rather than fighting it: `silu(0) = 0`,
so on a subspace orthogonal to `gate_i` atom i contributes exactly nothing. Silencing removes atoms
instead of compressing them, which moves both quantities the same way. Silencing 13 gates and
fitting the remaining 19 in rank 32 gives a median gate cosine of 0.99924, 0.98217 and 0.99211
across three seeds with exact directions, and 0.97661, 0.98336 and 0.99372 with the directions
perturbed to cosine 0.990 - the accuracy the live 32-unit run already has for the 13 atoms it
recovers above 0.99. The live run's own mean joint cosine is 0.635.

Boundary: this measures a mechanism, not a blind recovery. `SLICE_MODE=silenced` takes the
directions to silence from `truth()` and then perturbs them, which answers "does silencing work at
this direction accuracy" and does not answer "can the accuracy be reached oracle-only". In a live
solver those directions would come from the solver's own first-pass fit, and nothing here has yet
run that loop end to end.

`SLICE_MODE=peel` avoids slicing altogether, because a gate found on a slice is only its
projection and lifting it back by padding gives cosine sqrt(r/d) - measured over 400 draws at
d=64: 0.4914 at r=16, 0.7083 at r=32, 0.7903 at r=40, 0.9276 at r=55, against the 0.99 silencing
needs. So a slice-found atom cannot become the next round's silencing direction. Freezing can:
a confident atom is known in full coordinates, its contribution subtracts outright, and the
remainder is a smaller sum in the same rank.

Run: `npm run probe:swiglu-slice-stitch`, `:swiglu-slice-silenced`, `:swiglu-slice-consensus` or
`:swiglu-slice-peel`. Knobs:
SLICE_MODE, SLICE_ATOMS, SLICE_RANK, SLICE_CORE, SLICE_BLOCK, SLICE_DROP, SLICE_DROP_JITTER,
SLICE_RANK_USED, SLICE_RESTARTS, SLICE_STEPS, SLICE_SAMPLES_PER_PARAM, SLICE_SEED, SLICE_RUNS,
SLICE_ROUNDS, SLICE_CONSENSUS_THRESHOLD.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("GCL_BLIND_SWIGLU_STRUCTURED_STARTS", "0")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

import tools.llm_reader.blind_swiglu_oracle_decompiler as bs

ATOMS = int(os.environ.get("SLICE_ATOMS", "32"))
RANK = int(os.environ.get("SLICE_RANK", "64"))
CORE = int(os.environ.get("SLICE_CORE", "8"))
BLOCK = int(os.environ.get("SLICE_BLOCK", "8"))
RESTARTS = int(os.environ.get("SLICE_RESTARTS", "16"))
STEPS = int(os.environ.get("SLICE_STEPS", "3600"))
SAMPLES_PER_PARAM = float(os.environ.get("SLICE_SAMPLES_PER_PARAM", "31.0"))
SEED = int(os.environ.get("SLICE_SEED", "101"))
INPUT_SCALE = float(os.environ.get("SLICE_INPUT_SCALE", "0.35"))


class SliceOracle:
    """The block under test, reachable only by evaluating it at points.

    The weights are private and the call counter is the honest cost of the method: a slice run is
    only worth reporting next to how many oracle points it spent.
    """

    def __init__(self, atoms: int, rank: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        self._gate = rng.normal(size=(atoms, rank)) / np.sqrt(rank)
        self._up = rng.normal(size=(atoms, rank)) / np.sqrt(rank)
        self._scale = np.ones(atoms)
        self.calls = 0

    def evaluate(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        self.calls += int(points.shape[0])
        z = points @ self._gate.T
        return (((z / (1.0 + np.exp(-z))) * (points @ self._up.T)) * self._scale).sum(axis=1)

    def truth(self) -> tuple[np.ndarray, np.ndarray]:
        """Post-hoc only. Nothing above this line reads it, and the fits never see it."""
        return self._gate, self._up


def slice_bases(rank: int, core: int, block: int) -> tuple[np.ndarray, list[np.ndarray]]:
    """A shared core and disjoint blocks that together tile every coordinate.

    Axis aligned because the caller's coordinates are already an arbitrary recovered basis; a
    rotation here would only re-randomise something already random.
    """
    if core + block > rank:
        raise RuntimeError("core + block must fit inside rank")
    remaining = rank - core
    if remaining % block:
        raise RuntimeError(f"rank - core = {remaining} is not a multiple of block {block}")
    identity = np.eye(rank)
    core_basis = identity[:, :core]
    blocks = [identity[:, core + index * block: core + (index + 1) * block]
              for index in range(remaining // block)]
    return core_basis, blocks


def fit_on_slice(oracle: SliceOracle, basis: np.ndarray, atoms: int, seed: int) -> dict:
    """One low-rank fit, driven entirely through oracle calls at points inside the slice."""
    slice_rank = int(basis.shape[1])
    params = atoms * (2 * slice_rank + 1) + 1
    rows = int(params * SAMPLES_PER_PARAM)
    rng = np.random.default_rng(seed)
    coords = rng.normal(size=(rows, slice_rank)) * INPUT_SCALE
    targets = oracle.evaluate(coords @ basis.T)
    fit = bs.fit_swiglu_sum(
        coords, targets, atoms, seed=seed, steps=STEPS, learning_rate=0.05,
        restart_override=RESTARTS, fit_mode="batched",
    )
    return {
        "gates": np.asarray(fit["gates"], dtype=np.float64),
        "ups": np.asarray(fit["ups"], dtype=np.float64),
        "scales": np.asarray(fit["scales"], dtype=np.float64),
        "mae": float(fit["mae"]),
        "scale": float(np.std(targets)) or 1.0,
        "r2": float(fit.get("r2", float("nan"))),
    }


def match_by_core(reference_core: np.ndarray, candidate_core: np.ndarray) -> np.ndarray:
    """Which atom of the candidate slice is which atom of the reference slice.

    Both sides hold `core^T gate` for the same set of atoms, so the assignment that maximises total
    agreement is the correspondence. Hungarian when scipy is present, greedy otherwise - and the
    caller checks the result rather than trusting it, because a wrong permutation assembles a gate
    out of pieces of different atoms and every piece still looks well formed.
    """
    reference = reference_core / np.linalg.norm(reference_core, axis=1, keepdims=True).clip(min=1e-12)
    candidate = candidate_core / np.linalg.norm(candidate_core, axis=1, keepdims=True).clip(min=1e-12)
    similarity = reference @ candidate.T
    try:
        from scipy.optimize import linear_sum_assignment

        _rows, columns = linear_sum_assignment(-similarity)
        return np.asarray(columns, dtype=int)
    except ImportError:
        assignment = np.full(reference.shape[0], -1, dtype=int)
        taken: set[int] = set()
        order = np.argsort(-similarity.max(axis=1))
        for row in order:
            for column in np.argsort(-similarity[row]):
                if int(column) not in taken:
                    assignment[row] = int(column)
                    taken.add(int(column))
                    break
        return assignment


def stitch(oracle: SliceOracle, atoms: int, rank: int, core: int, block: int, seed: int) -> dict:
    core_basis, blocks = slice_bases(rank, core, block)
    assembled_gates = np.zeros((atoms, rank))
    assembled_ups = np.zeros((atoms, rank))
    core_gate_parts: list[np.ndarray] = []
    core_up_parts: list[np.ndarray] = []
    slice_reports: list[dict] = []
    reference_core: np.ndarray | None = None
    started = time.time()

    for index, block_basis in enumerate(blocks):
        basis = np.column_stack([core_basis, block_basis])
        fitted = fit_on_slice(oracle, basis, atoms, seed + 17 * index)
        gate_core = fitted["gates"][:, :core]
        gate_block = fitted["gates"][:, core:]
        up_core = fitted["ups"][:, :core]
        up_block = fitted["ups"][:, core:]
        if reference_core is None:
            reference_core = gate_core
            order = np.arange(atoms)
        else:
            order = match_by_core(reference_core, gate_core)
        # A permutation that repeats an index would silently build one atom twice and drop
        # another, so it is checked here rather than discovered in the cosines.
        if sorted(order.tolist()) != list(range(atoms)):
            raise RuntimeError(f"slice {index} produced a non-permutation matching")
        core_gate_parts.append(gate_core[order])
        core_up_parts.append(up_core[order])
        columns = slice(core + index * block, core + (index + 1) * block)
        assembled_gates[:, columns] = gate_block[order]
        assembled_ups[:, columns] = up_block[order]
        slice_reports.append({
            "slice": index,
            "maeOverScale": fitted["mae"] / fitted["scale"],
            "r2": fitted["r2"],
            "coreAgreement": None if reference_core is gate_core else float(np.median(
                np.sum(
                    reference_core / np.linalg.norm(reference_core, axis=1, keepdims=True).clip(min=1e-12)
                    * (gate_core[order] / np.linalg.norm(gate_core[order], axis=1, keepdims=True).clip(min=1e-12)),
                    axis=1,
                )
            )),
        })

    # Every slice measured the core, so the core components are an average of equals rather than
    # one arbitrary slice's copy.
    assembled_gates[:, :core] = np.mean(np.stack(core_gate_parts), axis=0)
    assembled_ups[:, :core] = np.mean(np.stack(core_up_parts), axis=0)

    true_gate, true_up = oracle.truth()
    gate_cosines = paired_cosines(assembled_gates, true_gate)
    up_cosines = paired_cosines(assembled_ups, true_up)
    return {
        "atoms": atoms, "rank": rank, "core": core, "block": block,
        "slices": len(blocks),
        "oracleCalls": oracle.calls,
        "seconds": round(time.time() - started, 1),
        "gateCosineMedian": float(np.median(gate_cosines)),
        "gateCosineMin": float(np.min(gate_cosines)),
        "upCosineMedian": float(np.median(up_cosines)),
        "sliceReports": slice_reports,
    }


def paired_cosines(recovered: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Best-matching absolute cosine per recovered vector, under one global assignment."""
    left = recovered / np.linalg.norm(recovered, axis=1, keepdims=True).clip(min=1e-12)
    right = truth / np.linalg.norm(truth, axis=1, keepdims=True).clip(min=1e-12)
    similarity = np.abs(left @ right.T)
    order = match_by_core(left, right)
    return similarity[np.arange(left.shape[0]), order]


def silenced_slice(oracle: SliceOracle, drop_directions: np.ndarray, slice_rank: int,
                   kept_atoms: int, rank: int, seed: int) -> dict:
    """A slice that switches atoms off instead of squeezing them into fewer dimensions.

    Plain slicing traded one problem for another: a rank-16 slice fits to R2 0.99988 but 32 atoms
    in 16 dimensions is overcomplete, every slice returns a different equally good decomposition,
    and the core components that were supposed to identify an atom across slices agreed to only
    0.59-0.73. Widening the slice to rank 32 moved the cosine from 0.308 to 0.747 and the fit the
    other way, to 0.99787 - two forces pulling against each other with no setting that satisfies
    both at 32 atoms.

    SwiGLU offers a third option that neither of those uses. An atom contributes
    `silu(gate . x) * (up . x)`, and `silu(0) = 0`, so on a subspace orthogonal to `gate_i` atom i
    contributes exactly nothing. Choosing the slice inside the orthogonal complement of a chosen
    set of gates therefore removes those atoms from the problem rather than compressing them, and
    what is left is a smaller atom count in a comfortable rank - the configuration the 16-unit lane
    already passes in.

    `drop_directions` are the gates to silence. Passing the true gates measures the ceiling of the
    idea; passing perturbed estimates measures whether it survives the accuracy a live run actually
    has for them.
    """
    complement = orthogonal_complement(drop_directions, rank)
    if complement.shape[1] < slice_rank:
        raise RuntimeError(
            f"complement has rank {complement.shape[1]}, too small for a rank {slice_rank} slice"
        )
    basis = complement[:, :slice_rank]
    fitted = fit_on_slice(oracle, basis, kept_atoms, seed)
    true_gate, true_up = oracle.truth()
    projected_gate = true_gate @ basis
    projected_up = true_up @ basis
    # Only the atoms the slice can still see are claimed. An atom that was silenced leaves no
    # trace here by construction, and scoring against it would be scoring against zero.
    energy = np.linalg.norm(projected_gate, axis=1)
    visible = np.argsort(-energy)[:kept_atoms]
    gate_cosines = paired_cosines(fitted["gates"], projected_gate[visible])
    up_cosines = paired_cosines(fitted["ups"], projected_up[visible])
    return {
        "sliceRank": int(slice_rank),
        "keptAtoms": int(kept_atoms),
        "atomsPerDim": round(kept_atoms / slice_rank, 3),
        "oracleCalls": oracle.calls,
        "sliceR2": fitted["r2"],
        "sliceMaeOverScale": fitted["mae"] / fitted["scale"],
        "gateCosineMedian": float(np.median(gate_cosines)),
        "gateCosineMin": float(np.min(gate_cosines)),
        "upCosineMedian": float(np.median(up_cosines)),
    }


def consensus_ranking(oracle: SliceOracle, atoms: int, rank: int, runs: int, seed: int) -> dict:
    """Can the solver tell which of its own atoms are trustworthy without the truth?

    Silencing needs directions, and the live run has them - 13 of its 32 atoms sit above cosine
    0.99 against the source - but it only knows which 13 because the report compares against
    weights afterwards. A solver that has to pick them oracle-only needs a criterion of its own.

    The candidate here is agreement between independent fits. A well-determined atom should come
    back from separate runs pointing the same way; an atom that is only a convenient piece of some
    decomposition should not. That is checkable against the truth once, here, and if it holds it is
    a criterion the solver can apply blind.

    Reports the rank correlation between consensus and truth, and how many of the atoms the
    criterion would pick are actually good - which is the number that decides whether the bootstrap
    is worth building.
    """
    solutions: list[np.ndarray] = []
    fit_reports: list[dict] = []
    for index in range(runs):
        fitted = fit_on_slice(oracle, np.eye(rank), atoms, seed + 4099 * index)
        solutions.append(fitted["gates"])
        fit_reports.append({"run": index, "r2": fitted["r2"],
                            "maeOverScale": fitted["mae"] / fitted["scale"]})

    reference = solutions[0]
    agreements = np.zeros((len(solutions) - 1, atoms))
    for position, other in enumerate(solutions[1:]):
        order = match_by_core(reference, other)
        left = reference / np.linalg.norm(reference, axis=1, keepdims=True).clip(min=1e-12)
        right = other[order] / np.linalg.norm(other[order], axis=1, keepdims=True).clip(min=1e-12)
        agreements[position] = np.abs(np.sum(left * right, axis=1))
    consensus = np.median(agreements, axis=0)

    true_gate, _true_up = oracle.truth()
    truth_cosines = paired_cosines(reference, true_gate)
    order_by_consensus = np.argsort(-consensus)
    picked = order_by_consensus[: int(np.sum(consensus >= 0.9))]
    return {
        "atoms": atoms, "rank": rank, "runs": runs,
        "oracleCalls": oracle.calls,
        "fits": fit_reports,
        "spearman": float(spearman(consensus, truth_cosines)),
        "truthCosineMedian": float(np.median(truth_cosines)),
        "truthCosineAbove099": int(np.sum(truth_cosines > 0.99)),
        "consensusAbove09": int(picked.size),
        "pickedTruthCosineMin": None if picked.size == 0 else float(np.min(truth_cosines[picked])),
        "pickedTruthCosineMedian": None if picked.size == 0 else float(np.median(truth_cosines[picked])),
        "pickedAbove099": None if picked.size == 0 else int(np.sum(truth_cosines[picked] > 0.99)),
        "top13TruthCosineMin": float(np.min(truth_cosines[order_by_consensus[:13]])),
        "top13Above099": int(np.sum(truth_cosines[order_by_consensus[:13]] > 0.99)),
    }


def peel(oracle: SliceOracle, atoms: int, rank: int, rounds: int, threshold: float,
         runs: int, seed: int) -> dict:
    """Freeze the atoms consensus vouches for, subtract them, and refit what is left.

    Slicing is deliberately not used here, and the reason is arithmetic rather than taste. A gate
    recovered on a rank-r slice of rank-d is only its projection, and lifting it back by padding
    gives cosine sqrt(r/d) against the true gate - measured over 400 draws at d=64: 0.4914 at r=16,
    0.7083 at r=32, 0.7903 at r=40, 0.9276 at r=55. Silencing needs 0.99. So an atom found on a
    slice cannot become the next round's silencing direction, and any peeling built on slices
    stalls after one step.

    Freezing does not have that problem. A confident atom is known in full coordinates, so its
    contribution can be subtracted outright and the remainder is a smaller SwiGLU sum in the same
    rank. Whether a smaller sum is easier to identify is the open question this measures: the
    scaling probe says the *fit error* at fixed rank barely moves with atom count, which is not the
    same claim as identification, and the live run already recovers 15 of 32 atoms above 0.99.
    """
    frozen_gates: list[np.ndarray] = []
    frozen_ups: list[np.ndarray] = []
    frozen_scales: list[float] = []
    round_reports: list[dict] = []
    true_gate, _true_up = oracle.truth()
    started = time.time()

    params = atoms * (2 * rank + 1) + 1
    rows = int(params * SAMPLES_PER_PARAM)
    rng = np.random.default_rng(seed)
    coords = rng.normal(size=(rows, rank)) * INPUT_SCALE
    targets = oracle.evaluate(coords)

    for round_index in range(rounds):
        remaining = atoms - len(frozen_gates)
        if remaining <= 0:
            break
        residual = targets.copy()
        for gate, up, scale in zip(frozen_gates, frozen_ups, frozen_scales):
            z = coords @ gate
            residual -= scale * (z / (1.0 + np.exp(-z))) * (coords @ up)
        solutions = []
        for run_index in range(runs):
            fit = bs.fit_swiglu_sum(
                coords, residual, remaining, seed=seed + 733 * round_index + 4099 * run_index,
                steps=STEPS, learning_rate=0.05, restart_override=RESTARTS, fit_mode="batched",
            )
            solutions.append(fit)
        best = solutions[0]
        reference = np.asarray(best["gates"], dtype=np.float64)
        agreements = []
        for other in solutions[1:]:
            other_gates = np.asarray(other["gates"], dtype=np.float64)
            order = match_by_core(reference, other_gates)
            left = reference / np.linalg.norm(reference, axis=1, keepdims=True).clip(min=1e-12)
            right = other_gates[order] / np.linalg.norm(other_gates[order], axis=1, keepdims=True).clip(min=1e-12)
            agreements.append(np.abs(np.sum(left * right, axis=1)))
        consensus = (np.median(np.stack(agreements), axis=0) if agreements
                     else np.zeros(reference.shape[0]))
        confident = np.flatnonzero(consensus >= threshold)
        truth_cosines = paired_cosines(reference, true_gate)
        round_reports.append({
            "round": round_index,
            "fittedAtoms": int(remaining),
            "r2": float(best.get("r2", float("nan"))),
            "confident": int(confident.size),
            "confidentTruthMin": None if confident.size == 0 else float(np.min(truth_cosines[confident])),
            "allTruthAbove099": int(np.sum(truth_cosines > 0.99)),
        })
        if confident.size == 0:
            break
        for index in confident:
            frozen_gates.append(reference[index])
            frozen_ups.append(np.asarray(best["ups"], dtype=np.float64)[index])
            frozen_scales.append(float(np.asarray(best["scales"], dtype=np.float64)[index]))

    frozen = np.asarray(frozen_gates) if frozen_gates else np.zeros((0, rank))
    frozen_truth = paired_cosines(frozen, true_gate) if frozen.shape[0] else np.zeros(0)
    return {
        "atoms": atoms, "rank": rank, "rounds": len(round_reports), "threshold": threshold,
        "oracleCalls": oracle.calls,
        "seconds": round(time.time() - started, 1),
        "frozenAtoms": int(frozen.shape[0]),
        "frozenTruthMin": None if frozen_truth.size == 0 else float(np.min(frozen_truth)),
        "frozenTruthMedian": None if frozen_truth.size == 0 else float(np.median(frozen_truth)),
        "frozenAbove099": int(np.sum(frozen_truth > 0.99)) if frozen_truth.size else 0,
        "roundReports": round_reports,
    }


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    """Rank correlation without pulling in scipy, which this environment does not have."""
    def ranks(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values)
        out = np.empty_like(order, dtype=np.float64)
        out[order] = np.arange(values.size, dtype=np.float64)
        return out

    a, b = ranks(np.asarray(left)), ranks(np.asarray(right))
    a = a - a.mean()
    b = b - b.mean()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 0.0 if denominator == 0.0 else float(np.dot(a, b) / denominator)


def orthogonal_complement(directions: np.ndarray, rank: int) -> np.ndarray:
    """An orthonormal basis of everything orthogonal to `directions`."""
    left, singular, _ = np.linalg.svd(np.asarray(directions, dtype=np.float64).T, full_matrices=True)
    kept = int(np.sum(singular > float(singular[0]) * 1e-10)) if singular.size else 0
    return left[:, kept:rank]


if __name__ == "__main__":
    mode = os.environ.get("SLICE_MODE", "stitch").strip()
    oracle = SliceOracle(ATOMS, RANK, seed=SEED)
    if mode == "peel":
        print(json.dumps(peel(
            oracle, ATOMS, RANK,
            rounds=int(os.environ.get("SLICE_ROUNDS", "4")),
            threshold=float(os.environ.get("SLICE_CONSENSUS_THRESHOLD", "0.9")),
            runs=int(os.environ.get("SLICE_RUNS", "4")), seed=SEED,
        )), flush=True)
    elif mode == "consensus":
        print(json.dumps(consensus_ranking(
            oracle, ATOMS, RANK, runs=int(os.environ.get("SLICE_RUNS", "4")), seed=SEED,
        )), flush=True)
    elif mode == "silenced":
        drop = int(os.environ.get("SLICE_DROP", str(ATOMS // 2)))
        jitter = float(os.environ.get("SLICE_DROP_JITTER", "0.0"))
        gates, _ups = oracle.truth()
        directions = gates[:drop].copy()
        if jitter > 0.0:
            rng = np.random.default_rng(SEED + 991)
            noise = rng.normal(size=directions.shape)
            noise /= np.linalg.norm(noise, axis=1, keepdims=True)
            directions = directions + jitter * np.linalg.norm(
                directions, axis=1, keepdims=True
            ) * noise
        report = silenced_slice(
            oracle, directions,
            slice_rank=int(os.environ.get("SLICE_RANK_USED", "32")),
            kept_atoms=ATOMS - drop, rank=RANK, seed=SEED,
        )
        report["droppedAtoms"] = drop
        report["dropJitter"] = jitter
        print(json.dumps(report), flush=True)
    else:
        print(json.dumps(stitch(oracle, ATOMS, RANK, CORE, BLOCK, seed=SEED)), flush=True)
