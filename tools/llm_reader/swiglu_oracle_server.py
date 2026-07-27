#!/usr/bin/env python3
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

from tools.llm_reader.blind_swiglu_oracle_decompiler import dequantized_weight
from tools.llm_reader.swiglu_oracle_protocol import (
    INITIAL_TRANSCRIPT,
    advance_transcript,
    recv_frame,
    send_frame,
)


def validate_eval_request(
    header: dict[str, Any],
    payload: bytes,
    *,
    expected_request_id: int,
    dimension: int,
) -> tuple[int, np.ndarray]:
    allowed = {"type", "requestId", "count", "dimension", "dtype", "payloadBytes"}
    unexpected = sorted(set(header).difference(allowed))
    if unexpected:
        raise RuntimeError(f"unexpected eval header fields: {','.join(unexpected)}")
    observed_id = int(header.get("requestId", -1))
    count = int(header.get("count", -1))
    observed_dimension = int(header.get("dimension", -1))
    if observed_id != expected_request_id:
        raise RuntimeError(f"request id mismatch: {observed_id} != {expected_request_id}")
    if count <= 0 or observed_dimension != dimension:
        raise RuntimeError(f"invalid eval shape: count={count} dimension={observed_dimension}")
    if header.get("dtype") != "float32-le":
        raise RuntimeError(f"invalid eval dtype: {header.get('dtype')!r}")
    if len(payload) != count * dimension * 4:
        raise RuntimeError("eval payload size does not match count/dimension")
    inputs = np.frombuffer(payload, dtype="<f4").reshape(count, dimension)
    if not np.all(np.isfinite(inputs)):
        raise RuntimeError("oracle input contains non-finite values")
    return count, inputs


def read_manifest(file_descriptor: int) -> dict[str, Any]:
    with os.fdopen(file_descriptor, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError("oracle manifest must be a JSON object")
    return value


def load_teacher(manifest: dict[str, Any]) -> tuple[Any, Any, Any, dict[str, Any]]:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.utils import logging as transformers_logging

    transformers_logging.disable_progress_bar()
    model_name = str(manifest["model"])
    device = str(manifest.get("device", "cuda"))
    local_files_only = bool(manifest.get("localFilesOnly", True))
    config = AutoConfig.from_pretrained(model_name, local_files_only=local_files_only)
    model_type = str(getattr(config, "model_type", ""))
    if model_type not in {"mistral", "llama"}:
        raise RuntimeError(f"isolated SwiGLU oracle expected mistral/llama, got {model_type!r}")
    if getattr(config, "quantization_config", None):
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            local_files_only=local_files_only,
            device_map=device,
        )
    else:
        dtype = torch.float16 if device == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            local_files_only=local_files_only,
            dtype=dtype,
        )
        model.to(device)
    model.eval()
    layer = int(manifest["layer"])
    layers = model.model.layers
    if layer < 0 or layer >= len(layers):
        raise RuntimeError(f"layer {layer} outside range 0..{len(layers) - 1}")
    mlp = layers[layer].mlp
    gate_weight = dequantized_weight(mlp.gate_proj).detach().float().cpu()
    up_weight = dequantized_weight(mlp.up_proj).detach().float().cpu()
    units = [int(value) for value in manifest["units"]]
    if not units:
        raise RuntimeError("oracle manifest contains no units")
    if len(set(units)) != len(units):
        raise RuntimeError("oracle manifest contains duplicate units")
    for unit in units:
        if unit < 0 or unit >= gate_weight.shape[0]:
            raise RuntimeError(f"unit {unit} outside range 0..{gate_weight.shape[0] - 1}")
    gate = gate_weight[units].contiguous()
    up = up_weight[units].contiguous()
    output_channel = int(manifest.get("outputChannel", -1))
    if output_channel < 0:
        scales = torch.ones(len(units), dtype=torch.float32)
    else:
        down_weight = dequantized_weight(mlp.down_proj).detach().float().cpu()
        if output_channel >= down_weight.shape[0]:
            raise RuntimeError(f"output channel {output_channel} outside range 0..{down_weight.shape[0] - 1}")
        scales = down_weight[output_channel, units].contiguous()
    metadata = {
        "hiddenSize": int(gate.shape[1]),
        "unitCount": len(units),
        "quantizationStep": float(manifest.get("quantizationStep", 0.0)),
        "maxOracleCalls": int(manifest["maxOracleCalls"]),
        "batchSize": int(manifest.get("batchSize", 2048)),
    }
    return gate, up, scales, metadata


def serve(args: argparse.Namespace) -> int:
    import torch

    manifest = read_manifest(args.manifest_fd)
    gate, up, scales, metadata = load_teacher(manifest)
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
            ready = {
                "type": "ready",
                "protocol": "gcl-swiglu-eval-v1",
                "hiddenSize": metadata["hiddenSize"],
                "unitCount": metadata["unitCount"],
                "oracleQuantizationStep": metadata["quantizationStep"],
                "maxOracleCalls": metadata["maxOracleCalls"],
                "solverAccess": "opaque-eval-only-oracle",
                "protocolCapabilities": {"eval": True, "truth": False},
            }
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
                    send_frame(connection, {
                        "type": "error",
                        "error": "eval-only-protocol",
                    })
                    continue
                transcript = advance_transcript(transcript, header, payload)
                count, inputs = validate_eval_request(
                    header,
                    payload,
                    expected_request_id=request_id,
                    dimension=int(metadata["hiddenSize"]),
                )
                if count > metadata["maxOracleCalls"] - query_count:
                    response = {
                        "type": "error",
                        "error": "oracle-budget-exceeded",
                        "requested": count,
                        "consumed": query_count,
                        "limit": metadata["maxOracleCalls"],
                    }
                    send_frame(connection, response)
                    transcript = advance_transcript(transcript, response, b"")
                    continue
                outputs: list[np.ndarray] = []
                batch_size = max(1, int(metadata["batchSize"]))
                with torch.no_grad():
                    for start in range(0, count, batch_size):
                        batch = torch.from_numpy(np.array(inputs[start:start + batch_size], copy=True))
                        values = torch.nn.functional.silu(batch @ gate.T) * (batch @ up.T)
                        outputs.append((values @ scales).cpu().numpy())
                result = np.concatenate(outputs).astype(np.float64)
                quantization_step = float(metadata["quantizationStep"])
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
            "status": status,
            "queryCount": query_count,
            "requestCount": request_id,
            "transcriptSha256": transcript,
        }, indent=2) + "\n", encoding="utf-8")
    return 0 if status == "complete" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Eval-only isolated real-model SwiGLU oracle")
    parser.add_argument("--manifest-fd", type=int, required=True)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--witness", required=True)
    return serve(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
