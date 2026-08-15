"""Which quantity actually governs how well a SwiGLU sum can be fitted.

The live 32-unit lane costs 45 to 75 minutes to answer one question, which is how three budget
knobs each got raised once on a hunch and each bought nothing: 4x the restarts returned the same
winning restart index and the same score to the last digit, and 2.3x the fit steps moved R2 by
0.00005. This drives the same `fit_swiglu_sum` against a synthetic SwiGLU sum with known weights,
so a knob can be swept in minutes and with no oracle at all.

What it found, at 8 atoms with samples per parameter held at 31 and 3600 steps:

    rank  16 -> MAE/scale 0.00003, R2 1.000000
    rank  32 -> MAE/scale 0.03034, R2 0.997330
    rank  64 -> MAE/scale 0.04306, R2 0.994897
    rank 128 -> MAE/scale 0.05632, R2 0.991980

Recovery is exact at rank 16 and degrades monotonically with dimension while the number of atoms,
the samples per parameter and the step count all stay fixed. Holding rank at 64 and moving atoms
from 8 to 32 instead leaves MAE/scale between 0.039 and 0.045 - flat. The subspace dimension is the
governing quantity, not the atom count, which is the opposite of how the lanes have been scaled.

Boundary: this is the bare fit path. Structured starts are off by default here and none of the
repair, polish or rescoring stages the live solver runs are present, so its absolute numbers are
well below what a live lane achieves at the same rank - the live 16-unit lane reaches MAE 0.00072
at rank 32. Compare trends within one configuration, not levels against a lane.

Knobs, all environment variables: CLIFF_ATOMS (comma separated), CLIFF_RANK (0 means 2 * atoms,
matching how the live lanes scale), CLIFF_RESTARTS, CLIFF_STEPS, CLIFF_SAMPLES_PER_PARAM.
"""
import os, sys, time, json
from pathlib import Path
os.environ.setdefault("GCL_BLIND_SWIGLU_STRUCTURED_STARTS", "0")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import tools.llm_reader.blind_swiglu_oracle_decompiler as bs

SAMPLES_PER_PARAM = float(os.environ.get("CLIFF_SAMPLES_PER_PARAM", "7.75"))
RESTARTS = int(os.environ.get("CLIFF_RESTARTS", "8"))
STEPS = int(os.environ.get("CLIFF_STEPS", "1200"))

def run(atoms: int, seed: int) -> dict:
    rank = int(os.environ.get("CLIFF_RANK", "0")) or atoms * 2
    params = atoms * (2 * rank + 1) + 1
    rows = int(params * SAMPLES_PER_PARAM)
    rng = np.random.default_rng(seed)
    gate = rng.normal(size=(atoms, rank)) / np.sqrt(rank)
    up = rng.normal(size=(atoms, rank)) / np.sqrt(rank)
    xs = rng.normal(size=(rows, rank)) * 0.35
    z = xs @ gate.T
    ys = ((z / (1.0 + np.exp(-z))) * (xs @ up.T)).sum(axis=1)
    started = time.time()
    fit = bs.fit_swiglu_sum(
        xs, ys, atoms, seed=seed, steps=STEPS, learning_rate=0.05,
        restart_override=RESTARTS, fit_mode="batched",
    )
    scale = float(np.std(ys)) or 1.0
    return {
        "atoms": atoms, "rank": rank, "rows": rows, "params": params,
        "seconds": round(time.time() - started, 1),
        "maeOverScale": float(fit["mae"]) / scale,
        "r2": float(fit.get("r2", float("nan"))),
        "spread": fit.get("rankScoreQuantilesByStartKind", {}).get("random"),
    }

if __name__ == "__main__":
    for atoms in [int(a) for a in os.environ.get("CLIFF_ATOMS", "8,16,32").split(",")]:
        print(json.dumps(run(atoms, seed=101)), flush=True)
