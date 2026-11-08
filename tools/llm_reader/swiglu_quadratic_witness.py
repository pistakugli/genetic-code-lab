#!/usr/bin/env python3
"""Structural quadratic-identity witness for blind SwiGLU oracle blocks.

Queries an isolated MLP block through the oracle-only server protocol
(the same selected-unit-ffn-vector path the recorded lanes use, with the
full unit list, which returns the complete MLP contribution to the residual
stream: down_proj(silu(gate_proj(x)) * up_proj(x))) and certifies, without
touching a single weight, that the block's even part

    E(x) = (f(x) + f(-x)) / 2

is exactly a vector-valued quadratic map — the defining structural property of
a bias-free SwiGLU FFN (silu(s) - silu(-s) = s, an identity of the SiLU
function itself). The checks are:

  1. homogeneity:      E(t*x) == t^2 * E(x)              (t = 2 and t = 3)
  2. polarization:     E(x+y+z) - E(x+y) - E(x+z) - E(y+z)
                       + E(x) + E(y) + E(z) == 0
  3. parallelogram:    E(x+y) + E(x-y) - 2E(x) - 2E(y) == 0

A block with biases, non-SwiGLU gating, or any affine term fails these at
O(1) relative error; a bias-free SwiGLU block passes to float-rounding
tolerance. The witness therefore certifies the target structure of the blind
recovery lanes from f(x)-only access.

Output: JSON report (and optional markdown) with per-check relative residual
statistics and a pass/fail gate.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

TOOL_PATH = Path(__file__).resolve().parent / "blind_swiglu_oracle_decompiler.py"


def make_random(seed: int):
    """Park-Miller LCG, identical sequence to the TS sweep wrapper (seed 260621)."""
    state = int(seed) % 2147483647
    if state <= 0:
        state += 2147483646

    def rnd():
        nonlocal state
        state = (state * 16807) % 2147483647
        return (state - 1) / 2147483646

    return rnd


class VectorOracle:
    """Minimal oracle-only client: all-units selected-unit-ffn-vector server."""

    def __init__(self, args: argparse.Namespace, work_dir: Path) -> None:
        self.work_dir = work_dir
        self.max_oracle_calls = int(args.max_oracle_calls)
        self.query_count = 0
        self._counter = 0
        units = list(range(int(args.intermediate_size)))
        command = [
            sys.executable, str(TOOL_PATH),
            "--oracle-server",
            "--model", args.model,
            "--layer", str(args.layer),
            "--unit", "0",
            "--units", ",".join(str(u) for u in units),
            "--device", args.device,
            "--oracle-output-mode", "selected-unit-ffn-vector",
            "--batch-size", str(args.batch_size),
            "--max-oracle-calls", str(args.max_oracle_calls),
        ]
        if args.local_files_only:
            command.append("--local-files-only")
        self.stderr_path = work_dir / "witness-oracle-stderr.log"
        self.stderr_file = self.stderr_path.open("w", encoding="utf-8", buffering=1)
        self.process = subprocess.Popen(
            command, cwd=str(TOOL_PATH.resolve().parents[2]),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.stderr_file,
            text=True, bufsize=1,
        )
        ready_line = self.process.stdout.readline()
        if not ready_line:
            raise RuntimeError(f"witness oracle did not become ready; stderr tail: "
                               f"{self.stderr_tail()}")
        self.meta = json.loads(ready_line)
        if self.meta.get("type") != "ready":
            raise RuntimeError(f"witness oracle not ready: {self.meta}")
        if self.meta.get("oracleOutputMode") != "selected-unit-ffn-vector":
            raise RuntimeError(f"expected selected-unit-ffn-vector oracle, got "
                               f"{self.meta.get('oracleOutputMode')!r}")
        if self.meta.get("solverAccess") != "oracle-only" or not self.meta.get("oracleOwnsWeights"):
            raise RuntimeError("witness oracle did not declare oracle-only weight ownership")
        self.hidden = int(self.meta["hiddenSize"])

    def stderr_tail(self, max_chars: int = 4000) -> str:
        try:
            self.stderr_file.flush()
            return self.stderr_path.read_text(encoding="utf-8", errors="replace")[-max_chars:]
        except OSError:
            return ""

    def eval(self, X: np.ndarray) -> np.ndarray:
        values = np.ascontiguousarray(X, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.hidden:
            raise RuntimeError(f"witness input shape mismatch: {values.shape}")
        requested = int(values.shape[0])
        if requested > self.max_oracle_calls - self.query_count:
            raise RuntimeError(
                f"witness oracle budget exceeded: requested={requested} "
                f"consumed={self.query_count} limit={self.max_oracle_calls}")
        self._counter += 1
        input_path = self.work_dir / f"witness-input-{self._counter}.npy"
        output_path = self.work_dir / f"witness-output-{self._counter}.npy"
        np.save(input_path, values)
        self.process.stdin.write(json.dumps({
            "cmd": "eval_npy",
            "inputPath": str(input_path),
            "outputPath": str(output_path),
        }) + "\n")
        self.process.stdin.flush()
        response_line = self.process.stdout.readline()
        if not response_line:
            raise RuntimeError(f"witness oracle closed stdout; stderr tail: "
                               f"{self.stderr_tail()}")
        response = json.loads(response_line)
        if response.get("type") != "ok":
            raise RuntimeError(f"witness oracle eval failed: {response}; "
                               f"stderr tail: {self.stderr_tail()}")
        observed = int(response.get("count", -1))
        if observed != requested:
            raise RuntimeError(f"witness oracle count mismatch: {observed} != {requested}")
        self.query_count = int(response.get("totalCount", self.query_count + requested))
        values_out = np.load(output_path).astype(np.float64)
        for path in (input_path, output_path):
            try:
                path.unlink()
            except OSError:
                pass
        return values_out

    def even(self, X: np.ndarray) -> np.ndarray:
        """E(x) = (f(x) + f(-x)) / 2 for a stack of probes."""
        return 0.5 * (self.eval(X) + self.eval(-X))

    def close(self) -> None:
        try:
            if self.process.poll() is None:
                self.process.stdin.write(json.dumps({"cmd": "stop"}) + "\n")
                self.process.stdin.flush()
                self.process.wait(timeout=15)
        finally:
            if self.process.poll() is None:
                self.process.kill()
            self.stderr_file.close()


def rel_resid(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-row relative residual of a vs b."""
    denom = np.maximum(np.linalg.norm(b, axis=1), 1e-30)
    return np.linalg.norm(a - b, axis=1) / denom


def run_witness(args: argparse.Namespace) -> int:
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    oracle = VectorOracle(args, work_dir)
    print("[witness] oracle ready:", json.dumps({k: oracle.meta.get(k) for k in (
        "hiddenSize", "oracleOutputMode", "oracleResponseWidth", "solverAccess",
        "oracleOwnsWeights")}), flush=True)

    random = make_random(args.seed)
    n = int(args.checks)

    def draw(count: int) -> np.ndarray:
        return np.asarray([[random() * 2.0 - 1.0 for _ in range(oracle.hidden)]
                           for _ in range(count)], dtype=np.float64) * float(args.scale)

    results = {}
    # --- check 1: homogeneity E(t x) = t^2 E(x) ------------------------------
    for t in (2.0, 3.0):
        X = draw(n)
        E_x = oracle.even(X)
        E_tx = oracle.even(X * t)
        r = rel_resid(E_tx, (t * t) * E_x)
        results[f"homogeneity-t{int(t)}"] = {
            "median": float(np.median(r)), "p90": float(np.quantile(r, 0.9)),
            "max": float(np.max(r)),
        }
        print(f"[witness] homogeneity t={t}: median {np.median(r):.3e}", flush=True)

    # --- check 2: polarization (three-point quadratic identity) --------------
    X, Y, Zc = draw(n), draw(n), draw(n)
    E_x, E_y, E_z = oracle.even(X), oracle.even(Y), oracle.even(Zc)
    E_xy, E_xz, E_yz = oracle.even(X + Y), oracle.even(X + Zc), oracle.even(Y + Zc)
    E_xyz = oracle.even(X + Y + Zc)
    lhs = E_xyz - E_xy - E_xz - E_yz + E_x + E_y + E_z
    # Violation magnitude relative to the map's own scale: lhs is identically
    # zero for a quadratic map, so ||lhs - E_xyz|| would be 1.0 exactly when
    # the identity holds — the violation norm, not the difference from the
    # reference, is the error.
    r = np.linalg.norm(lhs, axis=1) / np.maximum(np.linalg.norm(E_xyz, axis=1), 1e-30)
    results["polarization"] = {
        "median": float(np.median(r)), "p90": float(np.quantile(r, 0.9)),
        "max": float(np.max(r)),
    }
    print(f"[witness] polarization: median {np.median(r):.3e}", flush=True)

    # --- check 3: parallelogram ----------------------------------------------
    E_xmy = oracle.even(X - Y)
    lhs = E_xy + E_xmy - 2.0 * E_x - 2.0 * E_y
    ref = E_xy + E_xmy
    r = np.linalg.norm(lhs, axis=1) / np.maximum(np.linalg.norm(ref, axis=1), 1e-30)
    results["parallelogram"] = {
        "median": float(np.median(r)), "p90": float(np.quantile(r, 0.9)),
        "max": float(np.max(r)),
    }
    print(f"[witness] parallelogram: median {np.median(r):.3e}", flush=True)

    # --- check 4: control — a biased/affine block would fail homogeneity ------
    # f(0) itself: for a bias-free block f(0) = 0; record its norm relative to
    # typical |E(x)| as an affine-content bound.
    zero = np.zeros((1, oracle.hidden), dtype=np.float64)
    f_zero = oracle.eval(zero)[0]
    scale_ref = float(np.median(np.linalg.norm(E_x, axis=1)))
    results["originNormRelative"] = float(np.linalg.norm(f_zero) / max(scale_ref, 1e-30))
    print(f"[witness] f(0) relative norm: {results['originNormRelative']:.3e}", flush=True)

    oracle.close()

    tolerance = float(args.tolerance)
    gates = {}
    for key, stats in results.items():
        if key == "originNormRelative":
            gates[key] = bool(stats < tolerance)
        else:
            gates[key] = bool(stats["median"] < tolerance)
    passed = all(gates.values())

    report = {
        "kind": "swiglu-quadratic-identity-witness",
        "pass": passed,
        "model": args.model,
        "layer": int(args.layer),
        "seed": int(args.seed),
        "checks": int(args.checks),
        "scale": float(args.scale),
        "tolerance": tolerance,
        "oracleCalls": oracle.query_count,
        "hiddenSize": oracle.hidden,
        "oracleMeta": {k: oracle.meta.get(k) for k in (
            "hiddenSize", "oracleOutputMode", "oracleResponseWidth", "solverAccess",
            "oracleOwnsWeights", "oracleQuantizationStep")},
        "identities": {
            "evenPart": "E(x) = (f(x) + f(-x)) / 2",
            "basis": "silu(s) - silu(-s) = s holds for the SiLU function itself, so a "
                     "bias-free SwiGLU block has an exactly quadratic even part "
                     "regardless of the weight values (including quantized weights).",
        },
        "results": results,
        "gates": gates,
        "accessAudit": {
            "solverAccess": "oracle-only",
            "weightsUsedBySolver": False,
            "hooksUsedBySolver": False,
            "sourceIdsUsedBySolver": False,
            "postHocWitnessOnly": True,
        },
        "claimBoundary": [
            "This witness certifies the oracle block's even part is a vector-valued "
            "quadratic map to the recorded tolerance (float-rounding of the provider), "
            "which is the structural signature of a bias-free SiLU-gated GLU block.",
            "It does not certify weight values, unit identities, or the recovery quality "
            "of any solver; it certifies target structure only.",
        ],
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as handle:
        handle.write(json.dumps(report, indent=2) + "\n")
    if args.output_md:
        lines = [
            "# SwiGLU quadratic-identity witness (oracle-only)",
            "",
            f"- pass: `{passed}`",
            f"- model: `{args.model}` layer `{args.layer}`",
            f"- oracle calls: `{oracle.query_count}` (f(x)-only; weights never read)",
            "",
            "| check | median rel. residual | p90 | max | gate |",
            "| --- | --- | --- | --- | --- |",
        ]
        for key, stats in results.items():
            if key == "originNormRelative":
                lines.append(f"| {key} | {stats:.3e} | - | - | {gates[key]} |")
            else:
                lines.append(f"| {key} | {stats['median']:.3e} | {stats['p90']:.3e} | "
                             f"{stats['max']:.3e} | {gates[key]} |")
        lines += ["",
                  "Basis: `silu(s) - silu(-s) = s`, so `E(x) = (f(x)+f(-x))/2` is exactly",
                  "quadratic for any bias-free SwiGLU block — including quantized",
                  "providers. A biased or non-SwiGLU-gated block fails these identities",
                  "at O(1) relative error.", ""]
        with open(args.output_md, "w") as handle:
            handle.write("\n".join(lines))
    print(f"[witness] pass={passed} report={args.output_json}", flush=True)
    return 0 if passed else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-oracle-calls", type=int, default=200000)
    parser.add_argument("--intermediate-size", type=int, required=True,
                        help="unit count of the block (witness queries all units)")
    parser.add_argument("--checks", type=int, default=256)
    parser.add_argument("--scale", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=260621)
    parser.add_argument("--tolerance", type=float, default=0.02)
    parser.add_argument("--work-dir", default="/tmp/gcl-witness-work")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    started = time.time()
    status = run_witness(args)
    print(f"[witness] elapsed {time.time() - started:.1f}s", flush=True)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
