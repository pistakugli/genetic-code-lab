#!/usr/bin/env python3
"""Synthetic SwiGLU oracle server for sanity and methodology tests.

Speaks the exact eval-only protocol of `swiglu_oracle_server.py`
(`gcl-swiglu-eval-v1`, or `gcl-swiglu-eval-v1-unknown-order` with
`--hide-unit-count`), but the teacher is a recorded synthetic block
(`gate`, `up`, `scales`, optional `bias`) loaded from an .npz instead of a
real transformer. This is a development harness for the unknown-order
ladder: it lets the protocol, the ladder rules, and the null-permutation
control be exercised end-to-end on CPU before any GPU lane runs. It is NOT
a protocol oracle: every report produced against it must be labeled as
synthetic.

Manifest fields (JSON over --manifest-fd):
    syntheticTeacher: path to .npz with gate (m x d), up (m x d), scales (m,)
    quantizationStep: float (oracle output quantization, >= 0)
    maxOracleCalls: int
    batchSize: int (optional)

With --permute-responses-seed the response vector of every batch is
permuted with a frozen permutation before quantization; this implements the
null control of the preregistered unknown-order protocol (structure-free
responses on the same points).
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.llm_reader.swiglu_oracle_server import build_ready_frame
from tools.llm_reader.swiglu_oracle_protocol import (
    INITIAL_TRANSCRIPT,
    advance_transcript,
    recv_frame,
    send_frame,
)


def load_synthetic_teacher(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as data:
        gate = np.asarray(data["gate"], dtype=np.float64)
        up = np.asarray(data["up"], dtype=np.float64)
        scales = np.asarray(data["scales"], dtype=np.float64)
        bias = float(np.asarray(data["bias"]).item()) if "bias" in data else 0.0
    if gate.ndim != 2 or up.shape != gate.shape or scales.ndim != 1 or scales.shape[0] != gate.shape[0]:
        raise RuntimeError(f"synthetic teacher tables inconsistent: gate {gate.shape}, up {up.shape}, scales {scales.shape}")
    if not np.all(np.isfinite(gate)) or not np.all(np.isfinite(up)) or not np.all(np.isfinite(scales)):
        raise RuntimeError("synthetic teacher tables contain non-finite values")
    metadata = {
        "hiddenSize": int(gate.shape[1]),
        "unitCount": int(gate.shape[0]),
    }
    return gate, up, scales, bias, metadata


def serve(args: argparse.Namespace) -> int:
    with os.fdopen(int(args.manifest_fd), "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    gate, up, scales, bias, teacher_meta = load_synthetic_teacher(str(manifest["syntheticTeacher"]))
    quantization_step = float(manifest.get("quantizationStep", 0.0))
    max_oracle_calls = int(manifest["maxOracleCalls"])
    batch_size = max(1, int(manifest.get("batchSize", 2048)))
    permute_seed = int(getattr(args, "permute_responses_seed", 0) or 0)
    dimension = int(teacher_meta["hiddenSize"])
    socket_path = Path(args.socket).resolve()
    witness_path = Path(args.witness).resolve()
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    witness_path.parent.mkdir(parents=True, exist_ok=True)
    socket_path.unlink(missing_ok=True)
    witness_path.unlink(missing_ok=True)
    query_count = 0
    request_id = 0
    transcript = INITIAL_TRANSCRIPT
    status = "incomplete"
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        server.bind(str(socket_path))
        os.chmod(socket_path, 0o600)
        server.listen(1)
        connection, _ = server.accept()
        with connection:
            ready = build_ready_frame(
                {
                    "hiddenSize": dimension,
                    "unitCount": int(teacher_meta["unitCount"]),
                    "quantizationStep": quantization_step,
                    "maxOracleCalls": max_oracle_calls,
                },
                hide_unit_count=bool(getattr(args, "hide_unit_count", False)),
            )
            send_frame(connection, ready)
            transcript = advance_transcript(transcript, ready, b"")
            while True:
                header, payload = recv_frame(connection)
                command = header.get("type")
                if command == "stop":
                    send_frame(connection, {"type": "bye"})
                    status = "complete"
                    break
                if command != "eval":
                    send_frame(connection, {"type": "error", "error": "eval-only-protocol"})
                    continue
                transcript = advance_transcript(transcript, header, payload)
                count = int(header.get("count", -1))
                observed_dimension = int(header.get("dimension", -1))
                if (
                    int(header.get("requestId", -1)) != request_id
                    or count <= 0
                    or observed_dimension != dimension
                    or header.get("dtype") != "float32-le"
                    or len(payload) != count * dimension * 4
                ):
                    send_frame(connection, {"type": "error", "error": "invalid-eval-request"})
                    continue
                inputs = np.frombuffer(payload, dtype="<f4").reshape(count, dimension)
                if not np.all(np.isfinite(inputs)):
                    send_frame(connection, {"type": "error", "error": "non-finite-inputs"})
                    continue
                if count > max_oracle_calls - query_count:
                    response = {
                        "type": "error",
                        "error": "oracle-budget-exceeded",
                        "requested": count,
                        "consumed": query_count,
                        "limit": max_oracle_calls,
                    }
                    send_frame(connection, response)
                    transcript = advance_transcript(transcript, response, b"")
                    continue
                outputs: list[np.ndarray] = []
                for start in range(0, count, batch_size):
                    batch = np.array(inputs[start:start + batch_size], dtype=np.float64)
                    gate_pre = batch @ gate.T
                    up_pre = batch @ up.T
                    activated = gate_pre / (1.0 + np.exp(-gate_pre))
                    values = (activated * up_pre) @ scales + bias
                    if permute_seed:
                        rng = np.random.default_rng(permute_seed + query_count + start)
                        values = values[rng.permutation(values.shape[0])]
                    outputs.append(values)
                result = np.concatenate(outputs)
                if quantization_step > 0.0:
                    result = np.round(result / quantization_step) * quantization_step
                response_payload = np.asarray(result, dtype="<f4").tobytes(order="C")
                query_count += count
                response = {
                    "type": "ok",
                    "requestId": request_id,
                    "count": count,
                    "totalCount": query_count,
                    "dtype": "float32-le",
                }
                send_frame(connection, response, response_payload)
                transcript = advance_transcript(transcript, response, response_payload)
                request_id += 1
    finally:
        server.close()
        socket_path.unlink(missing_ok=True)
        witness_path.write_text(json.dumps({
            "schemaVersion": 1,
            "kind": "gcl-swiglu-eval-transcript",
            "synthetic": True,
            "permuted": bool(permute_seed),
            "status": status,
            "queryCount": query_count,
            "requestCount": request_id,
            "transcriptSha256": transcript,
        }, indent=2) + "\n", encoding="utf-8")
    return 0 if status == "complete" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Synthetic eval-only SwiGLU oracle (sanity harness)")
    parser.add_argument("--manifest-fd", type=int, required=True)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--witness", required=True)
    parser.add_argument("--hide-unit-count", action="store_true")
    parser.add_argument(
        "--permute-responses-seed",
        type=int,
        default=0,
        help="null control: permute each batch's responses with this frozen seed before quantization",
    )
    return serve(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
