#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

try:
    from tools.llm_reader.swiglu_oracle_protocol import (
        INITIAL_TRANSCRIPT,
        advance_transcript,
        recv_frame,
        send_frame,
    )
except ModuleNotFoundError:
    from swiglu_oracle_protocol import (
        INITIAL_TRANSCRIPT,
        advance_transcript,
        recv_frame,
        send_frame,
    )



def json_safe(value: Any) -> Any:
    """Convert a report payload into something JSON parsers outside Python can actually read.

    Python's json.dump writes bare NaN and Infinity by default. That is valid Python but not
    valid JSON, and every TypeScript consumer in this repo rejects it, so a lane whose metric
    went non-finite produced an artifact nobody could parse. The run was not recorded as
    failed; its evidence simply vanished, which is how a CEGIS lane finished with
    `"cubicRelativeRmse": NaN` and left no report.json at all.

    Non-finite floats become null, and every path that held one is listed under
    `nonFiniteFields` on the report, so the condition is stated rather than encoded in a
    token the reader cannot decode. Dropping the value silently would be the same defect in
    a quieter form.
    """
    non_finite: list[str] = []

    def convert(node: Any, path: str) -> Any:
        if isinstance(node, float):
            if math.isnan(node) or math.isinf(node):
                non_finite.append(f"{path} = {node}")
                return None
            return node
        if isinstance(node, dict):
            return {key: convert(item, f"{path}.{key}" if path else str(key)) for key, item in node.items()}
        if isinstance(node, (list, tuple)):
            return [convert(item, f"{path}[{index}]") for index, item in enumerate(node)]
        if isinstance(node, np.floating):
            return convert(float(node), path)
        if isinstance(node, np.integer):
            return int(node)
        if isinstance(node, np.ndarray):
            return convert(node.tolist(), path)
        return node

    converted = convert(value, "")
    if isinstance(converted, dict) and non_finite:
        converted["nonFiniteFields"] = sorted(non_finite)
    return converted


def dump_report_json(payload: Any) -> str:
    """Serialise a report so it is parseable everywhere, and fail loudly if it is not."""
    return json.dumps(json_safe(payload), indent=2, allow_nan=False)


def silu_np(values: np.ndarray) -> np.ndarray:
    return values / (1.0 + np.exp(-values))


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        raise RuntimeError("cannot normalize near-zero vector")
    return vector / norm


def normalize_recovered(vector: np.ndarray) -> np.ndarray:
    """Normalize a recovered atom, yielding the zero vector when it has collapsed.

    A recovery that misses the quality gate can leave one atom at effectively zero
    length. Raising on it kills the whole post-hoc comparison and destroys the
    alignment evidence for every other atom, which is the evidence needed to tell a
    wrong structure apart from a structure that is right but under-fitted. A zero
    vector scores cosine 0 against any source row, so the collapsed atom is reported
    as unaligned rather than hiding the fifteen that did align.

    Source rows keep the strict `normalize`: a near-zero row there would mean the
    witness itself is malformed, which is a real error and not a recovery outcome.
    """
    values = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(values))
    if norm < 1e-12 or not np.all(np.isfinite(values)):
        return np.zeros_like(values)
    return values / norm


def parse_units(unit: int, units: str | None) -> list[int]:
    if units is None or not units.strip():
        return [unit]
    parsed = [int(item.strip()) for item in units.split(",") if item.strip()]
    if not parsed:
        raise RuntimeError("--units did not contain any unit ids")
    return parsed


_REPORTED_FALLBACKS: set[str] = set()


def _record_fallback(reason: str) -> None:
    """Leave a trace when a path is abandoned for a reason the caller can survive.

    Four sites in this file caught everything and returned a default, so a missing
    dependency, a malformed candidate and a healthy decline all produced the same
    output and none of them produced a word. The message goes to stderr because
    stdout carries the oracle protocol, and each distinct reason is printed once so
    that a site inside a scoring loop cannot flood the log.
    """
    if reason in _REPORTED_FALLBACKS:
        return
    _REPORTED_FALLBACKS.add(reason)
    print(reason, file=sys.stderr)


def dequantized_weight(module: Any) -> Any:
    weight = module.weight
    quant_state = getattr(weight, "quant_state", None)
    if quant_state is None:
        return weight.detach()
    import bitsandbytes.functional as bnb_functional

    return bnb_functional.dequantize_4bit(weight.data, quant_state).detach()


def build_oracle_ready_payload(
    *,
    args: argparse.Namespace,
    model_type: str,
    units: list[int],
    hidden_size: int,
    intermediate_size: int,
    oracle_mode: str,
    output_channel: int | None,
    block_device: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": "ready",
        "hiddenSize": int(hidden_size),
        "unitCount": len(units),
        "oracleQuantizationStep": float(args.oracle_quantization_step),
        "oracleOwnsWeights": True,
        "solverAccess": "oracle-only",
        # How wide one call's answer is. A vector oracle returns `hiddenSize` numbers per
        # point where a scalar one returns 1, so a query count means something different in
        # the two modes and a budget comparison between them is not like for like. Recording
        # the width here keeps that visible in every artifact rather than in a lane's name.
        "oracleResponseWidth": int(hidden_size) if oracle_mode == "selected-unit-ffn-vector" else 1,
        "oracleOutputMode": oracle_mode,
        "protocolCapabilities": {
            "eval": True,
            "truth": not bool(args.oracle_eval_only),
        },
    }
    if args.oracle_ready_mode == "full":
        payload.update({
            "model": args.model,
            "modelType": model_type,
            "layer": int(args.layer),
            "unit": int(args.unit),
            "units": [int(value) for value in units],
            "intermediateSize": int(intermediate_size),
            "device": args.device,
            # `device` is the request; this is where the tensors that answer a call actually
            # live. They were not the same thing: the oracle accepted `--device cuda` and then
            # forced every weight to the host, so the only record of where the work happened
            # agreed with the flag and disagreed with reality. A field a caller can compare
            # against the flag turns that from an assumption into a check.
            "blockDevice": block_device,
            "oracleMode": oracle_mode,
            "outputChannel": output_channel,
        })
    return payload


def validate_opaque_oracle_metadata(metadata: dict[str, Any]) -> None:
    forbidden = {
        "model", "modelType", "layer", "unit", "units", "intermediateSize",
        "device", "blockDevice", "oracleMode", "outputChannel",
    }
    exposed = sorted(forbidden.intersection(metadata))
    if exposed:
        raise RuntimeError(f"opaque oracle exposed teacher manifest fields: {','.join(exposed)}")
    capabilities = metadata.get("protocolCapabilities")
    if not isinstance(capabilities, dict) or capabilities.get("eval") is not True:
        raise RuntimeError("opaque oracle does not advertise eval capability")
    if capabilities.get("truth") is not False:
        raise RuntimeError("opaque solver oracle must not advertise truth capability")
    if int(metadata.get("hiddenSize", 0)) <= 0 or int(metadata.get("unitCount", 0)) <= 0:
        raise RuntimeError("opaque oracle metadata requires positive hiddenSize and unitCount")


def oracle_server(args: argparse.Namespace) -> int:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.utils import logging as transformers_logging

    transformers_logging.disable_progress_bar()
    config = AutoConfig.from_pretrained(args.model, local_files_only=args.local_files_only)
    model_type = str(getattr(config, "model_type", ""))
    if model_type not in {"mistral", "llama"}:
        raise RuntimeError(f"blind SwiGLU oracle expected mistral/llama model_type, got {model_type!r}")

    if getattr(config, "quantization_config", None):
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            local_files_only=args.local_files_only,
            device_map=args.device,
        )
    else:
        dtype = torch.float16 if args.device == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            local_files_only=args.local_files_only,
            dtype=dtype,
        )
        model.to(args.device)
    model.eval()

    layers = model.model.layers
    if args.layer < 0 or args.layer >= len(layers):
        raise RuntimeError(f"layer {args.layer} outside model range 0..{len(layers) - 1}")
    mlp = layers[args.layer].mlp
    # The block the oracle answers from stays on the compute device. Every weight here used to
    # be pulled to the host with `.cpu()`, which made `--device cuda` decide only where the
    # model was dequantized: the answering matmuls then ran on the CPU thread pool while the
    # card held a model nothing read again. Measured on the 4096-unit rung: seven torch threads
    # at 21 minutes of CPU each over 42 minutes of wall clock, GPU utilisation 13%.
    compute_device = str(args.device)
    gate_weight = dequantized_weight(mlp.gate_proj).detach().float().to(compute_device)
    up_weight = dequantized_weight(mlp.up_proj).detach().float().to(compute_device)
    down_weight = None
    if args.output_channel >= 0:
        down_weight = dequantized_weight(mlp.down_proj).detach().float().to(compute_device)
        if args.output_channel >= down_weight.shape[0]:
            raise RuntimeError(
                f"output channel {args.output_channel} outside down_proj output range "
                f"0..{down_weight.shape[0] - 1}"
            )
    units = parse_units(args.unit, args.units)
    for unit in units:
        if unit < 0 or unit >= gate_weight.shape[0]:
            raise RuntimeError(f"unit {unit} outside SwiGLU intermediate range 0..{gate_weight.shape[0] - 1}")
    gate = gate_weight[units].contiguous()
    up = up_weight[units].contiguous()
    # A vector-valued oracle. The scalar modes collapse the block to one number, which is
    # what makes a unit's activation boundary visible only as a kink competing with every
    # other unit's kink for the same scalar. Returning the block's contribution to the whole
    # residual stream instead makes the Jacobian a matrix, and a boundary crossing a rank-1
    # jump in it, which is separable by linear algebra rather than by resolving overlaps in
    # time. Still black-box: the solver sees outputs, never a weight.
    source_matrix: torch.Tensor | None = None
    if args.oracle_output_mode == "selected-unit-ffn-vector":
        if down_weight is None:
            down_weight = dequantized_weight(mlp.down_proj).detach().float().to(compute_device)
        source_matrix = down_weight[:, units].contiguous()
        source_scales = torch.ones(len(units), dtype=torch.float32, device=compute_device)
        oracle_mode = "selected-unit-ffn-vector"
        output_channel = None
    elif down_weight is None:
        source_scales = torch.ones(len(units), dtype=torch.float32, device=compute_device)
        oracle_mode = "selected-unit-sum"
        output_channel = None
    else:
        source_scales = down_weight[args.output_channel, units].contiguous()
        oracle_mode = "down-proj-output-channel"
        output_channel = int(args.output_channel)
    hidden_size = int(gate.shape[1])

    print(json.dumps(build_oracle_ready_payload(
        args=args,
        model_type=model_type,
        units=units,
        hidden_size=hidden_size,
        intermediate_size=int(gate_weight.shape[0]),
        oracle_mode=oracle_mode,
        output_channel=output_channel,
        block_device=str(gate.device),
    )), flush=True)

    # Everything the oracle answers from is now a copy in `gate`, `up`, `source_matrix` and
    # `source_scales`; the model is dead weight from here on and holds 2.5 GB of a 7.65 GB card.
    # The 2048-unit rung died on a 680 MiB allocation with 463 MiB free, so this is the
    # difference between the upper rungs running and not running.
    del gate_weight, up_weight, down_weight, mlp, layers, model
    gc.collect()
    if compute_device.startswith("cuda"):
        torch.cuda.empty_cache()

    oracle_query_count = 0
    for line in sys.stdin:
        if not line.strip():
            continue
        request = json.loads(line)
        command = request.get("cmd")
        if command == "eval_npy":
            input_path = Path(request["inputPath"])
            output_path = Path(request["outputPath"])
            values = np.load(input_path).astype(np.float32, copy=False)
            if values.ndim != 2 or values.shape[1] != hidden_size:
                raise RuntimeError(f"oracle input shape {values.shape} does not match hidden size {hidden_size}")
            requested = int(values.shape[0])
            if requested > int(args.max_oracle_calls) - oracle_query_count:
                print(json.dumps({
                    "type": "error",
                    "error": "oracle-budget-exceeded",
                    "requested": requested,
                    "consumed": oracle_query_count,
                    "limit": int(args.max_oracle_calls),
                }), flush=True)
                continue
            outputs: list[np.ndarray] = []
            batch_size = max(1, int(request.get("batchSize", args.batch_size)))
            with torch.no_grad():
                for start in range(0, values.shape[0], batch_size):
                    batch = torch.from_numpy(values[start:start + batch_size]).to(compute_device)
                    gate_projection = batch @ gate.T
                    up_projection = batch @ up.T
                    atom_values = torch.nn.functional.silu(gate_projection) * up_projection
                    if source_matrix is None:
                        outputs.append((atom_values @ source_scales).cpu().numpy())
                    else:
                        # (batch, units) @ (units, hidden) -> the block's contribution to the
                        # residual stream, one row per probe point.
                        outputs.append((atom_values @ source_matrix.T).cpu().numpy())
            result = np.concatenate(outputs).astype(np.float64)
            if args.oracle_quantization_step > 0.0:
                result = np.round(result / args.oracle_quantization_step) * args.oracle_quantization_step
            np.save(output_path, result.astype(np.float32))
            oracle_query_count += requested
            print(json.dumps({
                "type": "ok",
                "count": requested,
                "totalCount": oracle_query_count,
            }), flush=True)
        elif command == "truth_npy":
            if args.oracle_eval_only:
                print(json.dumps({
                    "type": "error",
                    "error": "truth-disabled-for-eval-only-oracle",
                }), flush=True)
                continue
            gate_path = Path(request["gatePath"])
            up_path = Path(request["upPath"])
            np.save(gate_path, gate.cpu().numpy().astype(np.float32))
            np.save(
                up_path,
                (up * source_scales[:, None]).cpu().numpy().astype(np.float32),
            )
            # The output direction is part of what a vector oracle's caller recovers, so it
            # needs a witness like the other two. Without it a claim about `d_i` can only be
            # checked by reading weights outside the protocol, which is how one went
            # unverified while being reported as a result.
            down_path = request.get("downPath")
            if down_path is not None and source_matrix is not None:
                np.save(Path(down_path), source_matrix.cpu().numpy().astype(np.float32))
            print(json.dumps({
                "type": "ok",
                "truth": True,
                "downIncluded": bool(down_path is not None and source_matrix is not None),
            }), flush=True)
        elif command == "stop":
            print(json.dumps({"type": "bye"}), flush=True)
            return 0
        else:
            raise RuntimeError(f"unknown oracle command {command!r}")
    return 0


class OracleBudgetExceeded(RuntimeError):
    def __init__(self, *, stage: str, requested: int, consumed: int, limit: int) -> None:
        self.stage = stage
        self.requested = requested
        self.consumed = consumed
        self.limit = limit
        super().__init__(
            f"oracle budget exceeded at {stage}: requested={requested} "
            f"consumed={consumed} limit={limit}"
        )


class OracleClient:
    def __init__(self, args: argparse.Namespace, work_dir: Path) -> None:
        self.max_oracle_calls = int(args.max_oracle_calls)
        if self.max_oracle_calls <= 0:
            raise RuntimeError("--max-oracle-calls must be positive")
        self.stderr_path = work_dir / "oracle-stderr.log"
        self.stderr_file: Any | None = None
        self.process: subprocess.Popen[str] | None = None
        self.request_stream: Any
        self.response_stream: Any
        read_fd = int(args.oracle_read_fd)
        write_fd = int(args.oracle_write_fd)
        if (read_fd >= 0) != (write_fd >= 0):
            raise RuntimeError("external oracle transport requires both --oracle-read-fd and --oracle-write-fd")
        self.external_transport = read_fd >= 0
        if self.external_transport:
            self.response_stream = os.fdopen(read_fd, "r", encoding="utf-8", buffering=1)
            self.request_stream = os.fdopen(write_fd, "w", encoding="utf-8", buffering=1)
        else:
            self.stderr_file = self.stderr_path.open("w", encoding="utf-8", buffering=1)
            command = [
                sys.executable,
                __file__,
                "--oracle-server",
                "--model",
                args.model,
                "--layer",
                str(args.layer),
                "--unit",
                str(args.unit),
                "--device",
                args.device,
                "--batch-size",
                str(args.oracle_batch_size),
                "--max-oracle-calls",
                str(args.max_oracle_calls),
            ]
            if args.units:
                command.extend(["--units", args.units])
            if args.output_channel >= 0:
                command.extend(["--output-channel", str(args.output_channel)])
            if args.oracle_output_mode != "scalar":
                command.extend(["--oracle-output-mode", args.oracle_output_mode])
            if args.oracle_quantization_step > 0.0:
                command.extend(["--oracle-quantization-step", str(args.oracle_quantization_step)])
            if args.local_files_only:
                command.append("--local-files-only")
            self.process = subprocess.Popen(
                command,
                cwd=Path(__file__).resolve().parents[2],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self.stderr_file,
                text=True,
                bufsize=1,
            )
            assert self.process.stdin is not None
            assert self.process.stdout is not None
            self.request_stream = self.process.stdin
            self.response_stream = self.process.stdout
        ready_line = self.response_stream.readline()
        if not ready_line:
            exit_code = None if self.process is None else self.process.poll()
            raise RuntimeError(f"oracle did not become ready; exit={exit_code} stderr={self.stderr_tail()}")
        ready = json.loads(ready_line)
        if ready.get("type") != "ready":
            raise RuntimeError(f"oracle did not become ready: {ready}")
        if args.require_opaque_oracle:
            validate_opaque_oracle_metadata(ready)
        # A cuda oracle that answers from host tensors is the failure this catches: it produces
        # correct numbers at roughly a tenth of the throughput while holding the card's memory,
        # so nothing downstream reports a problem and the run merely looks slow. The oracle
        # reports where its block actually sits; a request for a card that lands on the host is
        # a broken run, not a slow one.
        block_device = ready.get("blockDevice")
        if isinstance(block_device, str) and str(args.device).startswith("cuda"):
            if not block_device.startswith("cuda"):
                raise RuntimeError(
                    f"oracle was asked for device {args.device!r} but answers from {block_device!r}"
                )
        self.meta = ready
        self.work_dir = work_dir
        self.query_count = 0
        self._counter = 0
        self._truth_unlocked = False
        self.truth_unlock_stage: str | None = None
        self.truth_unlock_query_count: int | None = None
        self.truth_request_count = 0

    def stderr_tail(self, max_chars: int = 4000) -> str:
        if self.stderr_file is None:
            return ""
        try:
            self.stderr_file.flush()
            text = self.stderr_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        return text[-max_chars:]

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.process is not None and self.process.poll() is not None:
            raise RuntimeError(f"oracle process exited before request {payload.get('cmd')!r}; exit={self.process.returncode} stderr={self.stderr_tail()}")
        try:
            self.request_stream.write(json.dumps(payload) + "\n")
            self.request_stream.flush()
        except (BrokenPipeError, OSError) as error:
            exit_code = None if self.process is None else self.process.poll()
            raise RuntimeError(f"oracle pipe failed during request {payload.get('cmd')!r}; exit={exit_code} stderr={self.stderr_tail()}") from error
        response_line = self.response_stream.readline()
        if not response_line:
            exit_code = None if self.process is None else self.process.poll()
            raise RuntimeError(f"oracle closed stdout during request {payload.get('cmd')!r}; exit={exit_code} stderr={self.stderr_tail()}")
        response = json.loads(response_line)
        if response.get("type") not in {"ok", "bye"}:
            raise RuntimeError(f"oracle request failed: {response}")
        return response

    def eval(self, inputs: np.ndarray, *, stage: str = "oracle-eval") -> np.ndarray:
        requested = int(inputs.shape[0])
        if requested > self.max_oracle_calls - self.query_count:
            raise OracleBudgetExceeded(
                stage=stage,
                requested=requested,
                consumed=self.query_count,
                limit=self.max_oracle_calls,
            )
        self._counter += 1
        input_path = self.work_dir / f"oracle-input-{self._counter}.npy"
        output_path = self.work_dir / f"oracle-output-{self._counter}.npy"
        np.save(input_path, inputs.astype(np.float32, copy=False))
        response = self.request({
            "cmd": "eval_npy",
            "inputPath": str(input_path),
            "outputPath": str(output_path),
        })
        observed = int(response.get("count", -1))
        if observed != requested:
            raise RuntimeError(
                f"oracle response count mismatch at {stage}: requested={requested} observed={observed}"
            )
        expected_total = self.query_count + requested
        observed_total = int(response.get("totalCount", expected_total))
        if observed_total != expected_total:
            raise RuntimeError(
                f"oracle response totalCount mismatch at {stage}: "
                f"expected={expected_total} observed={observed_total}"
            )
        self.query_count = expected_total
        values = np.load(output_path).astype(np.float64)
        # Both files have served their purpose the moment the array is in memory. Leaving
        # them behind cost 83 GB across two interrupted runs once the responses became
        # 3072 wide, and filled the disk mid-run - which surfaces as a write error inside the
        # oracle rather than as anything resembling its cause.
        for path in (input_path, output_path):
            try:
                path.unlink()
            except OSError:
                pass
        return values

    def unlock_posthoc_truth(self, stage: str) -> None:
        if not stage.strip():
            raise RuntimeError("post-hoc truth unlock requires a non-empty stage")
        if self._truth_unlocked:
            raise RuntimeError(f"post-hoc truth already unlocked at {self.truth_unlock_stage}")
        self._truth_unlocked = True
        self.truth_unlock_stage = stage
        self.truth_unlock_query_count = int(self.query_count)

    def truth_down(self) -> np.ndarray | None:
        """Post-hoc output directions, or None when this oracle has none to give.

        Same lock as `truth`: available only after the recovery is frozen.
        """
        if not self._truth_unlocked:
            raise RuntimeError("oracle truth is locked until recovery is frozen after validation")
        down_path = self.work_dir / "truth-down.npy"
        gate_path = self.work_dir / "truth-gate-ignored.npy"
        up_path = self.work_dir / "truth-up-ignored.npy"
        response = self.request({
            "cmd": "truth_npy",
            "gatePath": str(gate_path),
            "upPath": str(up_path),
            "downPath": str(down_path),
        })
        self.truth_request_count += 1
        if not response.get("downIncluded"):
            return None
        return np.load(down_path).astype(np.float64)

    def truth(self) -> tuple[np.ndarray, np.ndarray]:
        if not self._truth_unlocked:
            raise RuntimeError("oracle truth is locked until recovery is frozen after validation")
        gate_path = self.work_dir / "truth-gate.npy"
        up_path = self.work_dir / "truth-up.npy"
        self.request({
            "cmd": "truth_npy",
            "gatePath": str(gate_path),
            "upPath": str(up_path),
        })
        self.truth_request_count += 1
        return np.load(gate_path).astype(np.float64), np.load(up_path).astype(np.float64)

    def close(self) -> None:
        try:
            if self.process is None or self.process.poll() is None:
                self.request({"cmd": "stop"})
        finally:
            if self.process is not None:
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
            else:
                self.request_stream.close()
                self.response_stream.close()
            if self.stderr_file is not None:
                self.stderr_file.close()


class SocketOracleClient:
    def __init__(self, args: argparse.Namespace, work_dir: Path) -> None:
        del work_dir
        self.max_oracle_calls = int(args.max_oracle_calls)
        if self.max_oracle_calls <= 0:
            raise RuntimeError("--max-oracle-calls must be positive")
        self.connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        deadline = time.monotonic() + 60.0
        while True:
            try:
                self.connection.connect(args.oracle_socket)
                break
            except (FileNotFoundError, ConnectionRefusedError):
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"timed out connecting to opaque oracle socket: {args.oracle_socket}")
                time.sleep(0.05)
        ready, payload = recv_frame(self.connection)
        if payload or ready.get("type") != "ready":
            raise RuntimeError(f"invalid opaque oracle ready frame: {ready}")
        validate_opaque_oracle_metadata(ready)
        if ready.get("protocol") != "gcl-swiglu-eval-v1":
            raise RuntimeError(f"unsupported opaque oracle protocol: {ready.get('protocol')!r}")
        server_budget = int(ready.get("maxOracleCalls", -1))
        if server_budget != self.max_oracle_calls:
            raise RuntimeError(f"oracle budget mismatch: server={server_budget} solver={self.max_oracle_calls}")
        self.meta = ready
        self.query_count = 0
        self._counter = 0
        self.external_transport = True
        self.truth_request_count = 0
        self.truth_unlock_stage: str | None = None
        self.truth_unlock_query_count: int | None = None
        self.transcript_sha256 = advance_transcript(INITIAL_TRANSCRIPT, ready, b"")

    def eval(self, inputs: np.ndarray, *, stage: str = "oracle-eval") -> np.ndarray:
        values = np.asarray(inputs, dtype="<f4", order="C")
        if values.ndim != 2 or values.shape[1] != int(self.meta["hiddenSize"]):
            raise RuntimeError(f"opaque oracle input shape mismatch: {values.shape}")
        if not np.all(np.isfinite(values)):
            raise RuntimeError("opaque oracle input contains non-finite values")
        requested = int(values.shape[0])
        if requested > self.max_oracle_calls - self.query_count:
            raise OracleBudgetExceeded(
                stage=stage,
                requested=requested,
                consumed=self.query_count,
                limit=self.max_oracle_calls,
            )
        payload = values.tobytes(order="C")
        request = {
            "type": "eval",
            "requestId": self._counter,
            "count": requested,
            "dimension": int(values.shape[1]),
            "dtype": "float32-le",
        }
        send_frame(self.connection, request, payload)
        self.transcript_sha256 = advance_transcript(self.transcript_sha256, request, payload)
        response, response_payload = recv_frame(self.connection)
        self.transcript_sha256 = advance_transcript(self.transcript_sha256, response, response_payload)
        if response.get("type") != "ok":
            raise RuntimeError(f"opaque oracle request failed: {response}")
        if int(response.get("requestId", -1)) != self._counter:
            raise RuntimeError("opaque oracle response request id mismatch")
        if int(response.get("count", -1)) != requested:
            raise RuntimeError("opaque oracle response count mismatch")
        expected_total = self.query_count + requested
        if int(response.get("totalCount", -1)) != expected_total:
            raise RuntimeError("opaque oracle response total count mismatch")
        if response.get("dtype") != "float32-le" or len(response_payload) != requested * 4:
            raise RuntimeError("opaque oracle response payload mismatch")
        self.query_count = expected_total
        self._counter += 1
        return np.frombuffer(response_payload, dtype="<f4").astype(np.float64)

    def unlock_posthoc_truth(self, stage: str) -> None:
        del stage
        raise RuntimeError("truth is unavailable on the opaque eval-only oracle")

    def truth(self) -> tuple[np.ndarray, np.ndarray]:
        self.truth_request_count += 1
        raise RuntimeError("truth is unavailable on the opaque eval-only oracle")

    def close(self) -> None:
        try:
            send_frame(self.connection, {"type": "stop"})
            response, payload = recv_frame(self.connection)
            if payload or response.get("type") != "bye":
                raise RuntimeError(f"invalid opaque oracle close response: {response}")
        finally:
            self.connection.close()


def create_oracle_client(args: argparse.Namespace, work_dir: Path) -> OracleClient | SocketOracleClient:
    if args.oracle_socket:
        if args.oracle_read_fd >= 0 or args.oracle_write_fd >= 0:
            raise RuntimeError("choose either --oracle-socket or external oracle file descriptors")
        return SocketOracleClient(args, work_dir)
    return OracleClient(args, work_dir)


def estimate_gradient(
    oracle: OracleClient,
    rng: np.random.Generator,
    base: np.ndarray,
    direction_count: int,
    step: float,
) -> np.ndarray:
    dimension = base.shape[0]
    directions = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(direction_count, dimension))
    directions /= math.sqrt(dimension)
    plus = base[None, :] + step * directions
    minus = base[None, :] - step * directions
    values = oracle.eval(np.vstack([plus, minus]))
    directional = (values[:direction_count] - values[direction_count:]) / (2.0 * step)
    return dimension * np.mean(directional[:, None] * directions, axis=0)


def coordinate_perturbation_batch(base: np.ndarray, offsets: list[float]) -> np.ndarray:
    """`base` repeated with one coordinate shifted per row, one block per offset.

    This is the inner loop of every coordinate gradient, and it used to be written with a dense
    identity: `np.eye(dimension)` at `dim x dim`, then `base + step * eye`, then a `vstack` of
    the blocks. At the hidden size of a real model that is 37.7 MB of identity that is the same
    on every call, and three full passes over 9.4 million floats to express a batch whose rows
    differ from `base` in exactly one entry. The multiscale estimator paid it six times a probe
    point.

    One allocation, one diagonal write, and the row order is unchanged.
    """
    dimension = int(base.shape[0])
    batch = np.repeat(base[None, :], dimension * len(offsets), axis=0)
    index = np.arange(dimension)
    for position, offset in enumerate(offsets):
        batch[position * dimension + index, index] += offset
    return batch


def estimate_coordinate_gradient(
    oracle: OracleClient,
    base: np.ndarray,
    step: float,
) -> np.ndarray:
    dimension = base.shape[0]
    values = oracle.eval(coordinate_perturbation_batch(base, [step, -step]))
    return (values[:dimension] - values[dimension:]) / (2.0 * step)


def estimate_forward_coordinate_gradient(
    oracle: OracleClient,
    base: np.ndarray,
    step: float,
) -> np.ndarray:
    values = oracle.eval(
        np.vstack([base[None, :], coordinate_perturbation_batch(base, [step])])
    )
    return (values[1:] - values[0]) / step


def symmetric_regression_slopes(
    plus_values: np.ndarray,
    minus_values: np.ndarray,
    steps: list[float],
) -> np.ndarray:
    plus = np.asarray(plus_values, dtype=np.float64)
    minus = np.asarray(minus_values, dtype=np.float64)
    step_values = np.asarray(steps, dtype=np.float64)
    if (
        plus.shape != minus.shape
        or plus.ndim < 2
        or plus.shape[0] != step_values.shape[0]
        or step_values.shape[0] == 0
        or np.any(~np.isfinite(step_values))
        or np.any(step_values <= 0.0)
    ):
        raise RuntimeError("symmetric regression values/steps are incompatible")
    denominator = 2.0 * float(np.sum(step_values ** 2))
    weights_shape = (step_values.shape[0],) + (1,) * (plus.ndim - 1)
    weighted_differences = (
        step_values.reshape(weights_shape) * (plus - minus)
    )
    return np.sum(weighted_differences, axis=0) / max(denominator, 1e-15)


def estimate_symmetric_regression_coordinate_gradient(
    oracle: OracleClient,
    base: np.ndarray,
    steps: list[float],
) -> np.ndarray:
    gradients = estimate_symmetric_multiscale_coordinate_gradients(oracle, base, steps)
    plus_slopes = gradients * np.asarray(steps, dtype=np.float64)[:, None]
    denominator = float(np.sum(np.asarray(steps, dtype=np.float64) ** 2))
    return np.sum(plus_slopes * np.asarray(steps, dtype=np.float64)[:, None], axis=0) / max(denominator, 1e-15)


def richardson_extrapolate_symmetric_gradients(
    gradient_estimates: np.ndarray,
    steps: list[float],
) -> np.ndarray:
    estimates = np.asarray(gradient_estimates, dtype=np.float64)
    step_values = np.asarray(steps, dtype=np.float64)
    if (
        estimates.ndim < 2
        or estimates.shape[0] != step_values.shape[0]
        or step_values.shape[0] < 2
        or np.any(~np.isfinite(step_values))
        or np.any(step_values <= 0.0)
    ):
        raise RuntimeError("symmetric Richardson gradients require at least two positive finite steps")
    design = np.column_stack([
        np.ones(step_values.shape[0], dtype=np.float64),
        step_values ** 2,
    ])
    flattened = estimates.reshape(estimates.shape[0], -1)
    try:
        coefficients, *_ = np.linalg.lstsq(design, flattened, rcond=None)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError("failed to Richardson-extrapolate symmetric gradients") from exc
    return coefficients[0].reshape(estimates.shape[1:]).astype(np.float64)


def estimate_symmetric_richardson_coordinate_gradient(
    oracle: OracleClient,
    base: np.ndarray,
    steps: list[float],
) -> np.ndarray:
    gradients = estimate_symmetric_multiscale_coordinate_gradients(oracle, base, steps)
    return richardson_extrapolate_symmetric_gradients(gradients, steps)


def estimate_symmetric_multiscale_coordinate_gradients(
    oracle: OracleClient,
    base: np.ndarray,
    steps: list[float],
) -> np.ndarray:
    dimension = base.shape[0]
    offsets: list[float] = []
    for step in steps:
        offsets.extend([step, -step])
    values = oracle.eval(coordinate_perturbation_batch(base, offsets)).reshape(len(steps), 2, dimension)
    step_values = np.asarray(steps, dtype=np.float64)[:, None]
    return (values[:, 0, :] - values[:, 1, :]) / (2.0 * step_values)


def parse_step_list(raw: str | None, fallback: float) -> list[float]:
    if raw is None or not raw.strip():
        return [float(fallback)]
    steps = [float(value.strip()) for value in raw.split(",") if value.strip()]
    if not steps:
        return [float(fallback)]
    if any((not np.isfinite(step)) or step <= 0.0 for step in steps):
        raise RuntimeError(f"finite-difference steps must be positive finite values: {raw!r}")
    return steps


def parse_float_list(raw: str | None) -> list[float]:
    if raw is None or not raw.strip():
        return []
    values = [float(value.strip()) for value in raw.split(",") if value.strip()]
    if any((not np.isfinite(value)) or value <= 0.0 for value in values):
        raise RuntimeError(f"expected positive finite float list, got {raw!r}")
    return values


def parse_init_scale_schedule(raw: str | None, fallback: float) -> list[float]:
    values = parse_float_list(raw)
    return values if values else [fallback]


def sample_scale_mixture_inputs(
    rng: np.random.Generator,
    count: int,
    dimension: int,
    fallback_scale: float,
    scale_mixture: list[float],
) -> np.ndarray:
    scales = scale_mixture if scale_mixture else [fallback_scale]
    chunks: list[np.ndarray] = []
    remaining = count
    for index, scale in enumerate(scales):
        chunk_count = remaining if index == len(scales) - 1 else count // len(scales)
        remaining -= chunk_count
        if chunk_count > 0:
            chunks.append(rng.normal(0.0, scale, size=(chunk_count, dimension)).astype(np.float32))
    if not chunks:
        return np.zeros((0, dimension), dtype=np.float32)
    values = np.vstack(chunks)
    rng.shuffle(values, axis=0)
    return values.astype(np.float32, copy=False)


def sample_active_region_inputs(
    rng: np.random.Generator,
    count: int,
    basis: np.ndarray,
    fallback_scale: float,
    scale_mixture: list[float],
    mode: str,
) -> np.ndarray:
    normalized_mode = mode.strip().lower()
    if normalized_mode in {"", "gaussian", "full-gaussian"}:
        return sample_scale_mixture_inputs(rng, count, basis.shape[1], fallback_scale, scale_mixture)
    if normalized_mode in {"sobol-active-gaussian", "gaussian-sobol-active-mix"}:
        try:
            import torch
        except Exception as exc:
            raise RuntimeError("sobol-active-gaussian requires torch") from exc
        gaussian_count = 0
        sobol_count = count
        if normalized_mode == "gaussian-sobol-active-mix":
            gaussian_count = count // 2
            sobol_count = count - gaussian_count
        mixed_chunks: list[np.ndarray] = []
        if gaussian_count > 0:
            mixed_chunks.append(sample_scale_mixture_inputs(
                rng,
                gaussian_count,
                basis.shape[1],
                fallback_scale,
                scale_mixture,
            ))
        rank = basis.shape[0]
        scales = scale_mixture if scale_mixture else [fallback_scale]
        chunks: list[np.ndarray] = []
        remaining = sobol_count
        for scale_index, scale in enumerate(scales):
            chunk_count = remaining if scale_index == len(scales) - 1 else sobol_count // len(scales)
            remaining -= chunk_count
            if chunk_count <= 0:
                continue
            power = int(math.ceil(math.log2(max(1, chunk_count))))
            sampler = torch.quasirandom.SobolEngine(
                dimension=rank,
                scramble=True,
                seed=int(rng.integers(0, 2**31 - 1)),
            )
            uniforms_t = sampler.draw(2 ** power).to(dtype=torch.float64)[:chunk_count].clamp_(1e-7, 1.0 - 1e-7)
            active_coords = (math.sqrt(2.0) * torch.erfinv(2.0 * uniforms_t - 1.0)).cpu().numpy() * float(scale)
            chunks.append((active_coords @ basis).astype(np.float32))
        if not chunks:
            sobol_values = np.zeros((0, basis.shape[1]), dtype=np.float32)
        else:
            sobol_values = np.vstack(chunks)
        mixed_chunks.append(sobol_values)
        values = np.vstack([chunk for chunk in mixed_chunks if chunk.shape[0] > 0])
        if values.shape[0] == 0:
            return np.zeros((0, basis.shape[1]), dtype=np.float32)
        rng.shuffle(values, axis=0)
        return values.astype(np.float32, copy=False)
    if normalized_mode not in {"radial-shell", "active-radial-shell"}:
        raise RuntimeError(f"unknown active train sampling mode {mode!r}")
    rank = basis.shape[0]
    scales = scale_mixture if scale_mixture else [fallback_scale]
    chunks: list[np.ndarray] = []
    remaining = count
    for scale_index, scale in enumerate(scales):
        chunk_count = remaining if scale_index == len(scales) - 1 else count // len(scales)
        remaining -= chunk_count
        if chunk_count <= 0:
            continue
        directions = rng.normal(0.0, 1.0, size=(chunk_count, rank))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True).clip(min=1e-12)
        # Cover both the typical Gaussian shell and near-boundary radii. This
        # remains oracle-only: it only changes where the oracle is queried.
        low = 0.20 * float(scale) * np.sqrt(rank)
        high = 1.85 * float(scale) * np.sqrt(rank)
        radii = np.linspace(low, high, chunk_count, dtype=np.float64)
        rng.shuffle(radii)
        active_coords = directions * radii[:, None]
        chunks.append((active_coords @ basis).astype(np.float32))
    if not chunks:
        return np.zeros((0, basis.shape[1]), dtype=np.float32)
    values = np.vstack(chunks)
    rng.shuffle(values, axis=0)
    return values.astype(np.float32, copy=False)


def _eigh_preferring_device(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`numpy.linalg.eigh` on the card when there is one, on the host when there is not.

    Returns host arrays either way, so every caller is unchanged. A card that refuses the
    allocation is not a failure worth stopping for - the host answer is the same answer, only
    slower - but it is not swallowed either: the reason is recorded on the module so a run that
    quietly fell back can be asked about it rather than merely looking slow.
    """
    # torch is imported per call site in this module rather than at the top, and this follows
    # that: a machine without the package still runs the host path.
    try:
        import torch
    except ImportError as error:
        BASIS_EIGH_HOST_FALLBACKS.append(f"ImportError: {error}")
        return np.linalg.eigh(matrix)
    if not torch.cuda.is_available():
        return np.linalg.eigh(matrix)
    try:
        device_matrix = torch.as_tensor(matrix, device="cuda", dtype=torch.float64)
        values, vectors = torch.linalg.eigh(device_matrix)
        return values.cpu().numpy(), vectors.cpu().numpy()
    except (torch.OutOfMemoryError, RuntimeError) as error:
        BASIS_EIGH_HOST_FALLBACKS.append(f"{type(error).__name__}: {error}")
        return np.linalg.eigh(matrix)


BASIS_EIGH_HOST_FALLBACKS: list[str] = []


def recover_basis(gradients: np.ndarray, rank: int, moment: str = "auto") -> tuple[np.ndarray, np.ndarray, float]:
    # Centering n gradient witnesses caps the covariance rank at n - 1. For the
    # low-query path where n == rank, use raw second moments so a rank-2 SwiGLU
    # channel can still expose two independent active directions.
    if moment == "raw" or (moment == "auto" and gradients.shape[0] <= rank):
        basis_source = gradients
    elif moment in {"auto", "centered"}:
        basis_source = gradients - np.mean(gradients, axis=0, keepdims=True)
    else:
        raise RuntimeError(f"unknown basis moment mode {moment!r}")
    # The second moment is `dim x dim`, and `dim` is the model's hidden size - 3072 here, 4096
    # for Mistral - so this eigendecomposition is the largest dense solve on the blind path and
    # it was running on the host while the card sat idle holding the oracle. Measured on this
    # box at 3072 with 65 witnesses: 10.14 s in numpy against 0.78 s on the card, 13x, for a
    # step that runs once per case and again on every rescue cycle.
    #
    # Same arithmetic, same dtype, same ordering afterwards. The host path stays as the fallback
    # because it has to work on a machine without a card, and it is what every existing artifact
    # was produced with.
    matrix = basis_source.T @ basis_source / max(1, basis_source.shape[0])
    values, vectors = _eigh_preferring_device(matrix)
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]
    basis = vectors[:, :rank].T
    for index in range(rank):
        basis[index] = normalize(basis[index])
    total = float(np.sum(np.maximum(values, 0.0)))
    selected = float(np.sum(np.maximum(values[:rank], 0.0)))
    return basis, values[:rank], selected / max(total, 1e-12)


def estimate_active_forward_gradients(
    oracle: OracleClient,
    rng: np.random.Generator,
    basis: np.ndarray,
    count: int,
    input_scale: float,
    steps: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    rank = basis.shape[0]
    if count <= 0:
        return np.zeros((0, rank), dtype=np.float64), np.zeros((0, rank), dtype=np.float64)
    if not steps:
        raise RuntimeError("active-gradient estimation requires at least one positive step")
    coords = rng.normal(0.0, input_scale, size=(count, rank)).astype(np.float32)
    active_gradients = estimate_active_forward_gradients_at_coords(
        oracle,
        basis,
        coords,
        steps,
        stage="fit-active-gradient",
    )
    return coords.astype(np.float64), active_gradients


def estimate_active_forward_gradients_at_coords(
    oracle: OracleClient,
    basis: np.ndarray,
    coords: np.ndarray,
    steps: list[float],
    *,
    stage: str,
) -> np.ndarray:
    rank = basis.shape[0]
    active_coords = np.asarray(coords, dtype=np.float32)
    if active_coords.ndim != 2 or active_coords.shape[1] != rank:
        raise RuntimeError("active-gradient coordinates do not match basis rank")
    if not steps or any(not np.isfinite(step) or step <= 0.0 for step in steps):
        raise RuntimeError("active-gradient estimation requires positive finite steps")
    base_inputs = active_coords @ basis
    step_gradients = []
    for step_index, step in enumerate(steps):
        shifted_coords = np.repeat(active_coords[:, None, :], rank, axis=1)
        for index in range(rank):
            shifted_coords[:, index, index] += step
        count = active_coords.shape[0]
        shifted_inputs = shifted_coords.reshape(count * rank, rank) @ basis
        values = oracle.eval(
            np.vstack([base_inputs, shifted_inputs]).astype(np.float32),
            stage=f"{stage}-step-{step_index}",
        )
        base_values = values[:count]
        shifted_values = values[count:].reshape(count, rank)
        step_gradients.append((shifted_values - base_values[:, None]) / step)
    active_gradients = np.mean(np.stack(step_gradients, axis=0), axis=0)
    return active_gradients.astype(np.float64)


def estimate_active_symmetric_richardson_gradients_at_coords(
    oracle: OracleClient,
    basis: np.ndarray,
    coords: np.ndarray,
    steps: list[float],
    *,
    stage: str,
    chunk_size: int = 32,
) -> np.ndarray:
    rank = basis.shape[0]
    active_coords = np.asarray(coords, dtype=np.float32)
    if active_coords.ndim != 2 or active_coords.shape[1] != rank:
        raise RuntimeError("active-gradient coordinates do not match basis rank")
    if len(steps) < 2 or any(not np.isfinite(step) or step <= 0.0 for step in steps):
        raise RuntimeError("active Richardson gradients require at least two positive finite steps")
    if chunk_size <= 0:
        raise RuntimeError("active Richardson gradient chunk size must be positive")

    estimates = np.zeros(
        (len(steps), active_coords.shape[0], rank),
        dtype=np.float64,
    )
    for chunk_start in range(0, active_coords.shape[0], chunk_size):
        chunk = active_coords[chunk_start:chunk_start + chunk_size]
        count = chunk.shape[0]
        for step_index, step in enumerate(steps):
            plus_coords = np.repeat(chunk[:, None, :], rank, axis=1)
            minus_coords = plus_coords.copy()
            for index in range(rank):
                plus_coords[:, index, index] += step
                minus_coords[:, index, index] -= step
            plus_inputs = plus_coords.reshape(count * rank, rank) @ basis
            minus_inputs = minus_coords.reshape(count * rank, rank) @ basis
            values = oracle.eval(
                np.vstack([plus_inputs, minus_inputs]).astype(np.float32),
                stage=f"{stage}-chunk-{chunk_start // chunk_size}-step-{step_index}",
            ).reshape(2, count, rank)
            estimates[step_index, chunk_start:chunk_start + count] = (
                values[0] - values[1]
            ) / (2.0 * step)
    return richardson_extrapolate_symmetric_gradients(estimates, steps)


def estimate_active_symmetric_regression_gradients(
    oracle: OracleClient,
    rng: np.random.Generator,
    basis: np.ndarray,
    count: int,
    input_scale: float,
    steps: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    rank = basis.shape[0]
    if count <= 0:
        return np.zeros((0, rank), dtype=np.float64), np.zeros((0, rank), dtype=np.float64)
    if not steps:
        raise RuntimeError("active symmetric-gradient estimation requires positive steps")
    coords = rng.normal(0.0, input_scale, size=(count, rank)).astype(np.float32)
    query_blocks: list[np.ndarray] = []
    for step in steps:
        plus_coords = np.repeat(coords[:, None, :], rank, axis=1)
        minus_coords = plus_coords.copy()
        for index in range(rank):
            plus_coords[:, index, index] += step
            minus_coords[:, index, index] -= step
        query_blocks.extend([
            plus_coords.reshape(count * rank, rank) @ basis,
            minus_coords.reshape(count * rank, rank) @ basis,
        ])
    values = oracle.eval(np.vstack(query_blocks).astype(np.float32)).reshape(
        len(steps),
        2,
        count,
        rank,
    )
    active_gradients = symmetric_regression_slopes(
        values[:, 0, :, :],
        values[:, 1, :, :],
        steps,
    )
    return coords.astype(np.float64), active_gradients.astype(np.float64)


def estimate_active_cubic_directions(
    oracle: OracleClient,
    rng: np.random.Generator,
    basis: np.ndarray,
    count: int,
    steps: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    if count <= 0:
        return np.zeros((0, basis.shape[0]), dtype=np.float64), np.zeros(0, dtype=np.float64)
    if not steps or any(step <= 0.0 or not np.isfinite(step) for step in steps):
        raise RuntimeError("cubic direction probe steps must be positive and finite")
    rank = basis.shape[0]
    directions = rng.normal(0.0, 1.0, size=(count, rank)).astype(np.float64)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True).clip(min=1e-12)
    multipliers = np.array([2.0, 1.0, -1.0, -2.0], dtype=np.float64)
    score_estimates = []
    for step in steps:
        active_points = (multipliers[:, None, None] * step * directions[None, :, :]).reshape(count * 4, rank)
        values = oracle.eval((active_points @ basis).astype(np.float32)).reshape(4, count)
        # Central third directional derivative stencil:
        # f'''(0) ~= (f(2h) - 2f(h) + 2f(-h) - f(-2h)) / (2h^3)
        score_estimates.append(
            (values[0] - 2.0 * values[1] + 2.0 * values[2] - values[3])
            / (2.0 * step ** 3)
        )
    cubic_scores = richardson_extrapolate_cubic_scores(
        np.stack(score_estimates, axis=0),
        steps,
    )
    order = np.argsort(np.abs(cubic_scores))[::-1]
    return directions[order].astype(np.float64), cubic_scores[order].astype(np.float64)


def richardson_extrapolate_cubic_scores(
    score_estimates: np.ndarray,
    steps: list[float],
) -> np.ndarray:
    estimates = np.asarray(score_estimates, dtype=np.float64)
    if estimates.ndim != 2 or estimates.shape[0] != len(steps):
        raise RuntimeError("cubic score estimates must have one row per finite-difference step")
    if estimates.shape[0] == 1:
        return estimates[0].copy()
    step_squares = np.square(np.asarray(steps, dtype=np.float64))
    design = np.column_stack([
        np.ones(step_squares.shape[0], dtype=np.float64),
        step_squares,
    ])
    try:
        coefficients, *_ = np.linalg.lstsq(design, estimates, rcond=None)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError("failed to Richardson-extrapolate cubic witnesses") from exc
    # The central third-derivative stencil has D3 + c*h^2 + O(h^4) error.
    return coefficients[0].astype(np.float64)


def predict_swiglu_cubic_directional_derivatives(
    directions: np.ndarray,
    gates: np.ndarray,
    ups: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    directions = np.asarray(directions, dtype=np.float64)
    gates = np.asarray(gates, dtype=np.float64)
    ups = np.asarray(ups, dtype=np.float64)
    scales = np.asarray(scales, dtype=np.float64)
    if directions.ndim != 2 or gates.ndim != 2 or ups.shape != gates.shape:
        raise RuntimeError("cubic directional derivative inputs have incompatible shapes")
    if gates.shape[0] != scales.shape[0] or directions.shape[1] != gates.shape[1]:
        raise RuntimeError("cubic directional derivative atom dimensions do not match")
    gate_projection = directions @ gates.T
    up_projection = directions @ ups.T
    # silu(t a) * (t b) = 0.5 t^2 ab + 0.25 t^3 a^2 b + O(t^5).
    # The third derivative at zero is therefore 3! * 0.25 * a^2 b.
    return 1.5 * np.sum(
        gate_projection ** 2 * up_projection * scales[None, :],
        axis=1,
    )


def make_quadratic_atom_starts(
    coords: np.ndarray,
    targets: np.ndarray,
    seed: int,
    max_starts: int,
) -> list[tuple[np.ndarray, np.ndarray, float, float]]:
    rank = coords.shape[1]
    if rank < 3 or coords.shape[0] < rank * 8:
        return []
    rng = np.random.default_rng(seed)
    sample_count = min(coords.shape[0], max(2048, rank * 96))
    if sample_count < coords.shape[0]:
        sample_indices = np.sort(rng.choice(coords.shape[0], size=sample_count, replace=False))
        xs = coords[sample_indices]
        ys = targets[sample_indices]
    else:
        xs = coords
        ys = targets
    ys = ys.astype(np.float64) - float(np.mean(ys))
    # One column per unordered pair, so `rank(rank+1)/2` of them - 2080 at rank 64 - built over
    # `sample_count` rows, and then a normal-equation solve on that. Profiled inside
    # `fit_swiglu_atom`, this single call was 34.6% of the time, and the greedy pass calls it once
    # per atom, 32 times.
    #
    # It is the opposite shape from the LBFGS line search next door: one large dense product,
    # 6144 by 2080, and one 2080-square solve. Measured: 6.05 s on the host against 0.48 s on the
    # card, 12.7x, with coefficients agreeing to a median relative difference of 6.73e-15 and a
    # largest of 1.71e-11 - which is what a normal-equation solve on a matrix of this condition
    # gives in float64 whichever library runs it.
    rows_index, cols_index = np.triu_indices(rank)
    pairs = list(zip(rows_index.tolist(), cols_index.tolist()))
    off_diagonal = rows_index != cols_index
    coeffs = None
    try:
        import torch

        use_device = torch.cuda.is_available()
    except ImportError:
        use_device = False
    if use_device:
        try:
            device_xs = torch.as_tensor(xs, device="cuda", dtype=torch.float64)
            device_ys = torch.as_tensor(ys, device="cuda", dtype=torch.float64)
            device_rows = torch.as_tensor(rows_index, device="cuda")
            device_cols = torch.as_tensor(cols_index, device="cuda")
            feature_matrix = device_xs[:, device_rows] * device_xs[:, device_cols]
            feature_matrix = torch.where(
                torch.as_tensor(off_diagonal, device="cuda")[None, :],
                feature_matrix * 2.0,
                feature_matrix,
            )
            gram = feature_matrix.T @ feature_matrix
            gram += 1e-8 * torch.eye(gram.shape[0], device="cuda", dtype=torch.float64)
            coeffs = torch.linalg.solve(gram, feature_matrix.T @ device_ys).cpu().numpy()
            del device_xs, device_ys, feature_matrix, gram
            torch.cuda.empty_cache()
        except (torch.OutOfMemoryError, RuntimeError):
            coeffs = None
            torch.cuda.empty_cache()
    if coeffs is None:
        columns = [
            (2.0 if left != right else 1.0) * (xs[:, left] * xs[:, right])
            for left, right in pairs
        ]
        design = np.column_stack(columns)
        ridge = 1e-8 * np.eye(design.shape[1], dtype=np.float64)
        try:
            coeffs = np.linalg.solve(design.T @ design + ridge, design.T @ ys)
        except np.linalg.LinAlgError:
            try:
                coeffs, *_ = np.linalg.lstsq(design, ys, rcond=None)
            except np.linalg.LinAlgError:
                return []
    if not np.all(np.isfinite(coeffs)):
        return []
    quadratic = np.zeros((rank, rank), dtype=np.float64)
    for value, (left, right) in zip(coeffs, pairs):
        quadratic[left, right] = value
        quadratic[right, left] = value
    try:
        values, vectors = np.linalg.eigh(quadratic)
    except np.linalg.LinAlgError:
        return []
    order = np.argsort(np.abs(values))[::-1]
    basis_vectors = [normalize(vectors[:, index]) for index in order[:min(rank, 10)] if abs(values[index]) > 1e-12]
    if len(basis_vectors) < 2:
        return []
    gate_scales = np.array([0.2, 0.35, 0.55, 0.85, 1.25, 1.8, 2.6], dtype=np.float64)
    starts: list[tuple[float, np.ndarray, np.ndarray, float, float]] = []
    ones = np.ones(xs.shape[0], dtype=np.float64)
    raw_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for i, first in enumerate(basis_vectors[:8]):
        for second in basis_vectors[i + 1:8]:
            for sign in (1.0, -1.0):
                raw_pairs.append((first + sign * second, first - sign * second))
                raw_pairs.append((first - sign * second, first + sign * second))
    for vector in basis_vectors[:8]:
        for other in basis_vectors[:8]:
            if vector is other:
                continue
            raw_pairs.append((vector, other))
    seen: set[tuple[float, ...]] = set()
    for gate_raw, up_raw in raw_pairs:
        try:
            gate_dir = normalize(gate_raw)
            up_dir = normalize(up_raw)
        except RuntimeError:
            continue
        signature = tuple(np.round(np.concatenate([gate_dir, up_dir]), 3).tolist())
        if signature in seen:
            continue
        seen.add(signature)
        up_projection = xs @ up_dir
        for gate_scale in gate_scales:
            gate = gate_dir * gate_scale
            feature = silu_np(xs @ gate) * up_projection
            design_one = np.column_stack([feature, ones])
            try:
                solved, *_ = np.linalg.lstsq(design_one, targets[sample_indices] if sample_count < coords.shape[0] else targets, rcond=None)
            except np.linalg.LinAlgError:
                continue
            if not np.all(np.isfinite(solved)):
                continue
            predicted = design_one @ solved
            target_values = targets[sample_indices] if sample_count < coords.shape[0] else targets
            mae = float(np.mean(np.abs(predicted - target_values)))
            starts.append((mae, gate, up_dir, float(solved[0]), float(solved[1])))
    starts.sort(key=lambda item: item[0])
    return [(gate, up, scale, bias) for _mae, gate, up, scale, bias in starts[:max_starts]]


def make_bagged_quadratic_gate_candidates(
    coords: np.ndarray,
    targets: np.ndarray,
    atom_count: int,
    seed: int,
    max_candidates: int,
) -> list[np.ndarray]:
    rank = coords.shape[1]
    if atom_count < 8 or rank < 3 or coords.shape[0] < rank * 16:
        return []
    bag_count = int(os.environ.get(
        "GCL_BLIND_SWIGLU_BAGGED_QUADRATIC_BAGS",
        "32" if atom_count >= 16 else "16",
    ) or "0")
    sample_count = int(os.environ.get(
        "GCL_BLIND_SWIGLU_BAGGED_QUADRATIC_SAMPLE_COUNT",
        str(max(2048, atom_count * 192)),
    ) or "0")
    if bag_count <= 0 or sample_count <= rank * 8:
        return []
    rng = np.random.default_rng(seed)
    sample_count = min(coords.shape[0], sample_count)
    centered_targets = targets.astype(np.float64) - float(np.mean(targets))
    hard_order = np.argsort(np.abs(centered_targets))[::-1]
    hard_pool = hard_order[:min(hard_order.size, max(sample_count * 2, atom_count * 256))]
    candidates: list[np.ndarray] = []
    signatures: set[tuple[float, ...]] = set()

    def add_gate(raw_gate: np.ndarray, scale: float = 1.0) -> None:
        gate = np.asarray(raw_gate, dtype=np.float64) * scale
        norm = float(np.linalg.norm(gate))
        if norm < 1e-9 or not np.all(np.isfinite(gate)):
            return
        direction = gate / norm
        if direction[np.argmax(np.abs(direction))] < 0.0:
            direction = -direction
            gate = -gate
        signature = (
            round(float(np.log(max(norm, 1e-12))), 3),
            *np.round(direction, 4).tolist(),
        )
        if signature in signatures:
            return
        signatures.add(signature)
        candidates.append(gate)

    for bag_index in range(bag_count):
        if hard_pool.size >= sample_count and bag_index % 3 == 1:
            indices = np.sort(rng.choice(hard_pool, size=sample_count, replace=False))
        elif hard_pool.size > 0 and bag_index % 3 == 2:
            hard_count = min(sample_count // 2, hard_pool.size)
            uniform_count = sample_count - hard_count
            hard_indices = rng.choice(hard_pool, size=hard_count, replace=False)
            uniform_indices = rng.choice(coords.shape[0], size=uniform_count, replace=False)
            indices = np.sort(np.unique(np.concatenate([hard_indices, uniform_indices])))
            if indices.size < sample_count:
                fill = rng.choice(coords.shape[0], size=sample_count - indices.size, replace=False)
                indices = np.sort(np.unique(np.concatenate([indices, fill])))[:sample_count]
        else:
            indices = np.sort(rng.choice(coords.shape[0], size=sample_count, replace=False))
        atom_starts = make_quadratic_atom_starts(
            coords[indices],
            targets[indices],
            seed + 10007 + bag_index * 97,
            max_starts=max(atom_count * 4, 64),
        )
        for gate, _up, _scale, _bias in atom_starts:
            for scale in (0.55, 0.8, 1.0, 1.25, 1.6):
                add_gate(gate, scale)
                if len(candidates) >= max_candidates:
                    return candidates
    return candidates[:max_candidates]


def make_quadratic_multi_structured_starts(
    coords: np.ndarray,
    targets: np.ndarray,
    atom_count: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
    if atom_count < 4:
        return []
    atom_starts = make_quadratic_atom_starts(
        coords,
        targets,
        seed + 811,
        max_starts=max(atom_count * 16, 128),
    )
    if len(atom_starts) < atom_count:
        return []
    rng = np.random.default_rng(seed + 812)
    gate_pool = [gate for gate, _up, _scale, _bias in atom_starts]
    gate_pool.extend(make_bagged_quadratic_gate_candidates(
        coords,
        targets,
        atom_count,
        seed + 1901,
        max_candidates=max(atom_count * 24, 384),
    ))
    starts: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []

    def add_selected(selected_gates: list[np.ndarray]) -> None:
        if len(selected_gates) != atom_count:
            return
        solved = solve_fixed_gate_start(coords, targets, np.vstack(selected_gates))
        if solved is None:
            return
        gates, ups, scales, bias, _mae = solved
        starts.append((gates, ups, scales, bias))

    for max_abs_cosine in (0.999, 0.995, 0.985, 0.965):
        selected: list[np.ndarray] = []
        for gate in gate_pool:
            direction = canonical_direction(gate)
            if direction is None:
                continue
            if any(abs(float(np.dot(direction, existing / max(1e-9, np.linalg.norm(existing))))) >= max_abs_cosine for existing in selected):
                continue
            selected.append(gate)
            if len(selected) >= atom_count:
                break
        add_selected(selected)

    top_pool = gate_pool[:min(len(gate_pool), max(atom_count * 8, 96))]
    for stride in (2, 3, 5, 7):
        selected = []
        for index in range(stride - 1, len(top_pool), stride):
            gate = top_pool[index]
            direction = canonical_direction(gate)
            if direction is None:
                continue
            if any(abs(float(np.dot(direction, existing / max(1e-9, np.linalg.norm(existing))))) >= 0.985 for existing in selected):
                continue
            selected.append(gate)
            if len(selected) >= atom_count:
                break
        add_selected(selected)

    for _restart_index in range(8):
        selected = []
        for gate in rng.permutation(top_pool):
            direction = canonical_direction(gate)
            if direction is None:
                continue
            if any(abs(float(np.dot(direction, existing / max(1e-9, np.linalg.norm(existing))))) >= 0.99 for existing in selected):
                continue
            selected.append(gate)
            if len(selected) >= atom_count:
                break
        add_selected(selected)

    if not starts:
        return []
    scored: list[tuple[float, np.ndarray, np.ndarray, np.ndarray, float]] = []
    for gates, ups, scales, bias in starts:
        prediction = predict_swiglu(coords, {
            "gates": gates,
            "ups": ups,
            "scales": scales,
            "bias": bias,
        })
        mae = float(np.mean(np.abs(prediction - targets)))
        scored.append((mae, gates, ups, scales, bias))
    scored.sort(key=lambda item: item[0])
    unique: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []
    signatures: set[tuple[float, ...]] = set()
    for _mae, gates, ups, scales, bias in scored:
        signature_values = []
        for gate in gates:
            direction = canonical_direction(gate)
            if direction is not None:
                signature_values.extend(np.round(direction, 3).tolist())
        signature = tuple(signature_values)
        if signature in signatures:
            continue
        signatures.add(signature)
        unique.append((gates, ups, scales, bias))
        if len(unique) >= 8:
            break
    return unique


def fit_swiglu_atom(
    coords: np.ndarray,
    targets: np.ndarray,
    seed: int,
    steps: int,
    learning_rate: float,
) -> dict[str, Any]:
    import torch

    torch.manual_seed(seed)
    rank = coords.shape[1]
    # This one stays on the host, and the reason is measured twice with opposite answers.
    #
    # The Adam loop alone, at 32000 rows and 300 steps, is faster on the card from rank 4 up:
    # 1.11x, 1.25x at 16, 2.10x at 64, 10.34x at 128. On that evidence the card looks right.
    #
    # The whole function says otherwise. At rank 64 over 12000 rows it takes 7.75 s on the host
    # and 21.49 s on the card - 2.8x slower - because after the Adam loop comes LBFGS with
    # `strong_wolfe`, and a line search is a long sequence of tiny closure evaluations, each one
    # a synchronisation the card pays for and the host does not. The gate that comes back is the
    # same either way, cosine 1.000000000.
    #
    # The first measurement was the harness measuring a part instead of the whole, which is
    # exactly the failure it is easy to ship: a benchmark of the loop that dominates a profile is
    # not a benchmark of the function that contains it. The override remains for anyone who wants
    # to re-measure on other hardware.
    requested_device = os.environ.get("GCL_BLIND_SWIGLU_FIT_DEVICE", "").strip().lower()
    fit_device = torch.device(requested_device) if requested_device in {"cpu", "cuda"} else torch.device("cpu")
    x = torch.tensor(coords, dtype=torch.float32, device=fit_device)
    y = torch.tensor(targets, dtype=torch.float32, device=fit_device)
    target_scale = float(np.std(targets)) or 1.0
    target_variance = float(np.mean((targets - np.mean(targets)) ** 2))
    best: dict[str, Any] | None = None
    structured_starts: list[tuple[np.ndarray, np.ndarray, float, float]] = []
    if rank == 2:
        angles = np.linspace(0.0, math.pi, 24, endpoint=False)
        gate_scales = np.array([0.15, 0.25, 0.4, 0.65, 1.0, 1.6, 2.5, 4.0], dtype=np.float64)
        scored: list[tuple[float, np.ndarray, np.ndarray, float, float]] = []
        ones = np.ones(coords.shape[0], dtype=np.float64)
        for gate_angle in angles:
            gate_dir = np.array([math.cos(gate_angle), math.sin(gate_angle)], dtype=np.float64)
            for up_angle in angles:
                up_dir = np.array([math.cos(up_angle), math.sin(up_angle)], dtype=np.float64)
                up_projection = coords @ up_dir
                for gate_scale in gate_scales:
                    feature = silu_np(gate_scale * (coords @ gate_dir)) * up_projection
                    design = np.column_stack([feature, ones])
                    try:
                        coeffs, *_ = np.linalg.lstsq(design, targets, rcond=None)
                    except np.linalg.LinAlgError:
                        continue
                    predicted = design @ coeffs
                    mae = float(np.mean(np.abs(predicted - targets)))
                    scored.append((mae, gate_scale * gate_dir, up_dir, float(coeffs[0]), float(coeffs[1])))
        scored.sort(key=lambda item: item[0])
        structured_starts = [(gate, up, scale, bias) for _, gate, up, scale, bias in scored[:12]]
    elif rank > 2:
        structured_starts = make_quadratic_atom_starts(
            coords,
            targets,
            seed + 707,
            max_starts=16,
        )
    base_starts = [
        (np.array([1.0, 0.0]), np.array([0.0, 1.0])),
        (np.array([0.0, 1.0]), np.array([1.0, 0.0])),
        (np.array([1.0, 1.0]), np.array([1.0, -1.0])),
        (np.array([1.0, -1.0]), np.array([1.0, 1.0])),
    ]
    restart_count = len(structured_starts) + 16

    # The Adam phase is run for every restart at once, on the card, and that combination is the
    # whole point: measured at 12000 rows, 20 restarts, 300 steps, batching on the host is 5 to 8
    # times *slower* (2.73 s to 21.47 s at rank 2) and running the restarts one at a time on the
    # card is no better than the host. Together they are 16.8x at rank 2 and 17.2x at rank 64.
    # Each half alone is a pessimisation, which is why moving only the device, or only the
    # batching, kept measuring worse.
    #
    # Restarts are independent, so a summed loss gives each one exactly the gradient it would
    # have had alone, and Adam is elementwise. The starts are drawn on the host in the order they
    # were always drawn, then stacked, so these are the same restarts from the same seeds.
    #
    # LBFGS stays per restart afterwards. It is a quasi-Newton method with one history over its
    # parameters: batching it would let restarts pull on each other, which is a different
    # algorithm rather than a faster one.
    batch_device = fit_device
    if requested_device not in {"cpu", "cuda"} and torch.cuda.is_available():
        batch_device = torch.device("cuda")
    initial_gates: list[torch.Tensor] = []
    initial_ups: list[torch.Tensor] = []
    initial_scales: list[float] = []
    initial_biases: list[float] = []
    for restart in range(restart_count):
        if restart < len(structured_starts):
            gate_init, up_init, scale_init, bias_init = structured_starts[restart]
            initial_gates.append(torch.tensor(gate_init, dtype=torch.float32))
            initial_ups.append(torch.tensor(up_init, dtype=torch.float32))
            initial_scales.append(float(scale_init))
            initial_biases.append(float(bias_init))
        elif restart - len(structured_starts) < len(base_starts) and rank == 2:
            base_index = restart - len(structured_starts)
            gate_init, up_init = base_starts[base_index]
            initial_gates.append(torch.tensor(normalize(gate_init) * 0.75, dtype=torch.float32))
            initial_ups.append(torch.tensor(normalize(up_init) * 0.75, dtype=torch.float32))
            initial_scales.append(max(0.1, target_scale))
            initial_biases.append(float(np.mean(targets)))
        else:
            initial_gates.append(torch.randn(rank) * 0.8)
            initial_ups.append(torch.randn(rank) * 0.8)
            initial_scales.append(max(0.1, target_scale))
            initial_biases.append(float(np.mean(targets)))

    # Group size is the whole point of the measurement, not a tuning knob. On the card the
    # restarts go together; on the host they go one at a time, because batching there is 5 to 8
    # times slower - the grouped design is much larger and nothing on a CPU pays that back. A
    # group of one is exactly the sequential loop this replaced.
    batch_x = x.to(batch_device)
    batch_y = y.to(batch_device)
    group = restart_count if str(batch_device).startswith("cuda") else 1
    adam_gate_parts: list[torch.Tensor] = []
    adam_up_parts: list[torch.Tensor] = []
    adam_scale_parts: list[torch.Tensor] = []
    adam_bias_parts: list[torch.Tensor] = []
    for group_start in range(0, restart_count, group):
        group_end = min(group_start + group, restart_count)
        batch_gate = torch.nn.Parameter(
            torch.stack(initial_gates[group_start:group_end]).to(batch_device)
        )
        batch_up = torch.nn.Parameter(
            torch.stack(initial_ups[group_start:group_end]).to(batch_device)
        )
        batch_scale = torch.nn.Parameter(torch.tensor(
            initial_scales[group_start:group_end], dtype=torch.float32, device=batch_device
        ))
        batch_bias = torch.nn.Parameter(torch.tensor(
            initial_biases[group_start:group_end], dtype=torch.float32, device=batch_device
        ))
        batch_optimizer = torch.optim.Adam(
            [batch_gate, batch_up, batch_scale, batch_bias], lr=learning_rate
        )
        for _ in range(steps):
            batch_optimizer.zero_grad()
            batch_prediction = (
                batch_scale[None, :]
                * torch.nn.functional.silu(batch_x @ batch_gate.T)
                * (batch_x @ batch_up.T)
                + batch_bias[None, :]
            )
            torch.mean((batch_prediction - batch_y[:, None]) ** 2, dim=0).sum().backward()
            batch_optimizer.step()
        adam_gate_parts.append(batch_gate.detach().to(fit_device))
        adam_up_parts.append(batch_up.detach().to(fit_device))
        adam_scale_parts.append(batch_scale.detach().to(fit_device))
        adam_bias_parts.append(batch_bias.detach().to(fit_device))
        del batch_gate, batch_up, batch_scale, batch_bias, batch_optimizer
    adam_gates = torch.cat(adam_gate_parts)
    adam_ups = torch.cat(adam_up_parts)
    adam_scales = torch.cat(adam_scale_parts)
    adam_biases = torch.cat(adam_bias_parts)
    del batch_x, batch_y
    if str(batch_device).startswith("cuda"):
        torch.cuda.empty_cache()

    for restart in range(restart_count):
        gate = torch.nn.Parameter(adam_gates[restart].clone())
        up = torch.nn.Parameter(adam_ups[restart].clone())
        scale = torch.nn.Parameter(adam_scales[restart].clone())
        bias = torch.nn.Parameter(adam_biases[restart].clone())
        lbfgs = torch.optim.LBFGS([gate, up, scale, bias], lr=0.8, max_iter=160, line_search_fn="strong_wolfe")

        def closure() -> torch.Tensor:
            lbfgs.zero_grad()
            prediction = scale * torch.nn.functional.silu(x @ gate) * (x @ up) + bias
            loss = torch.mean((prediction - y) ** 2)
            loss.backward()
            return loss

        try:
            lbfgs.step(closure)
        except RuntimeError:
            pass
        with torch.no_grad():
            feature = (torch.nn.functional.silu(x @ gate) * (x @ up)).detach().cpu().numpy().astype(np.float64)
            design = np.column_stack([feature, np.ones(coords.shape[0], dtype=np.float64)])
            try:
                coeffs, *_ = np.linalg.lstsq(design, targets, rcond=None)
                if np.all(np.isfinite(coeffs)):
                    scale_value = float(coeffs[0])
                    bias_value = float(coeffs[1])
                    prediction_np = design @ coeffs
                    mae = float(np.mean(np.abs(prediction_np - targets)))
                else:
                    raise np.linalg.LinAlgError("non-finite least-squares coefficients")
            except np.linalg.LinAlgError:
                prediction = scale * torch.nn.functional.silu(x @ gate) * (x @ up) + bias
                mae = torch.mean(torch.abs(prediction - y)).item()
                scale_value = float(scale.detach().cpu().item())
                bias_value = float(bias.detach().cpu().item())
            candidate = {
                "restart": restart,
                "mae": mae,
                "gate": gate.detach().cpu().numpy().astype(np.float64),
                "up": up.detach().cpu().numpy().astype(np.float64),
                "scale": scale_value,
                "bias": bias_value,
            }
            if best is None or candidate["mae"] < best["mae"]:
                best = candidate
    assert best is not None
    best["restartCount"] = restart_count
    return best


def fit_swiglu_sum_batched(
    coords: np.ndarray,
    targets: np.ndarray,
    atom_count: int,
    seed: int,
    steps: int,
    learning_rate: float,
    restart_count: int,
    structured_starts: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]],
    gradient_coords: np.ndarray | None,
    gradient_targets: np.ndarray | None,
    gradient_weight: float,
    restart_chunk_size: int,
    selection_coords: np.ndarray | None = None,
    selection_targets: np.ndarray | None = None,
    extra_gate_candidates: list[np.ndarray] | None = None,
    cubic_directions: np.ndarray | None = None,
    cubic_targets: np.ndarray | None = None,
    cubic_selection_directions: np.ndarray | None = None,
    cubic_selection_targets: np.ndarray | None = None,
    cubic_weight_override: float | None = None,
    selection_cubic_weight_override: float | None = None,
) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("batched SwiGLU fit requires CUDA")

    torch.manual_seed(seed)
    emit_progress("fit-batched-entered", atomCount=int(atom_count), rows=int(coords.shape[0]))
    device = torch.device("cuda")
    fit_dtype = torch.float32
    x = torch.tensor(coords, dtype=fit_dtype, device=device)
    y = torch.tensor(targets, dtype=fit_dtype, device=device)
    rank = coords.shape[1]
    target_scale = float(np.std(targets)) or 1.0
    target_variance = float(np.mean((targets - np.mean(targets)) ** 2))
    has_external_selection = (
        selection_coords is not None
        and selection_targets is not None
        and selection_coords.shape[0] > 0
        and selection_coords.shape[0] == selection_targets.shape[0]
    )
    has_gradient_witness = (
        gradient_coords is not None
        and gradient_targets is not None
        and gradient_weight > 0.0
        and gradient_coords.shape[0] > 0
    )
    if has_gradient_witness:
        gx = torch.tensor(gradient_coords, dtype=fit_dtype, device=device)
        gy = torch.tensor(gradient_targets, dtype=fit_dtype, device=device)
        target_grad_scale = torch.mean(gy ** 2).detach().clamp_min(1e-12)
    else:
        gx = None
        gy = None
        target_grad_scale = None
    cubic_weight = float(
        cubic_weight_override
        if cubic_weight_override is not None
        else os.environ.get("GCL_BLIND_SWIGLU_FIT_CUBIC_WEIGHT", "0.0") or "0.0"
    )
    selection_cubic_weight = float(
        selection_cubic_weight_override
        if selection_cubic_weight_override is not None
        else os.environ.get("GCL_BLIND_SWIGLU_SELECTION_CUBIC_WEIGHT", "0.0") or "0.0"
    )
    if not np.isfinite(cubic_weight) or cubic_weight < 0.0:
        raise RuntimeError("GCL_BLIND_SWIGLU_FIT_CUBIC_WEIGHT must be non-negative and finite")
    if not np.isfinite(selection_cubic_weight) or selection_cubic_weight < 0.0:
        raise RuntimeError("GCL_BLIND_SWIGLU_SELECTION_CUBIC_WEIGHT must be non-negative and finite")
    has_cubic_witness = (
        cubic_directions is not None
        and cubic_targets is not None
        and cubic_weight > 0.0
        and cubic_directions.shape[0] > 0
        and cubic_directions.shape[0] == cubic_targets.shape[0]
    )
    if has_cubic_witness:
        cv = torch.tensor(cubic_directions, dtype=fit_dtype, device=device)
        cy = torch.tensor(cubic_targets, dtype=fit_dtype, device=device)
        target_cubic_scale = torch.mean(cy ** 2).detach().clamp_min(1e-12)
    else:
        cv = None
        cy = None
        target_cubic_scale = None

    best: dict[str, Any] | None = None
    top_candidates: list[dict[str, Any]] = []
    gate_candidate_pool: list[tuple[float, np.ndarray]] = []
    best_score_by_start_kind: dict[str, float] = {}
    best_metrics_by_start_kind: dict[str, dict[str, Any]] = {}
    selection_tail_weight = float(os.environ.get("GCL_BLIND_SWIGLU_SELECTION_TAIL_WEIGHT", "0.0") or "0.0")
    selection_tail_quantile = float(os.environ.get("GCL_BLIND_SWIGLU_SELECTION_TAIL_QUANTILE", "0.995") or "0.995")
    selection_worst_weight = float(os.environ.get("GCL_BLIND_SWIGLU_SELECTION_WORST_WEIGHT", "0.0") or "0.0")
    if not np.isfinite(selection_tail_weight) or selection_tail_weight < 0.0:
        raise RuntimeError("GCL_BLIND_SWIGLU_SELECTION_TAIL_WEIGHT must be non-negative and finite")
    if not np.isfinite(selection_tail_quantile) or selection_tail_quantile <= 0.0 or selection_tail_quantile >= 1.0:
        raise RuntimeError("GCL_BLIND_SWIGLU_SELECTION_TAIL_QUANTILE must be in (0, 1)")
    if not np.isfinite(selection_worst_weight) or selection_worst_weight < 0.0:
        raise RuntimeError("GCL_BLIND_SWIGLU_SELECTION_WORST_WEIGHT must be non-negative and finite")
    random_init_scales = parse_init_scale_schedule(
        os.environ.get("GCL_BLIND_SWIGLU_RANDOM_INIT_SCALES"),
        float(os.environ.get("GCL_BLIND_SWIGLU_RANDOM_INIT_SCALE", "0.6") or "0.6"),
    )
    structured_start_count = len(structured_starts)
    total_restart_count = restart_count + structured_start_count
    extra_score_rng = np.random.default_rng(seed + 73721)
    extra_score_count = min(
        coords.shape[0],
        int(os.environ.get("GCL_BLIND_SWIGLU_EXTRA_GATE_SCORE_SAMPLE_COUNT", str(max(2048, atom_count * 256))) or "0"),
    )
    if 0 < extra_score_count < coords.shape[0]:
        extra_score_indices = np.sort(extra_score_rng.choice(coords.shape[0], size=extra_score_count, replace=False))
        extra_score_coords = coords[extra_score_indices]
        extra_score_targets = targets[extra_score_indices]
    else:
        extra_score_coords = coords
        extra_score_targets = targets
    extra_score_ones = np.ones(extra_score_coords.shape[0], dtype=np.float64)

    def quick_gate_candidate_score(gate: np.ndarray) -> float:
        try:
            block = silu_np(extra_score_coords @ gate)[:, None] * extra_score_coords
            design = np.column_stack([block, extra_score_ones])
            coeffs, *_ = np.linalg.lstsq(design, extra_score_targets, rcond=None)
        except np.linalg.LinAlgError:
            return float("inf")
        if not np.all(np.isfinite(coeffs)):
            return float("inf")
        prediction = design @ coeffs
        return float(np.mean(np.abs(prediction - extra_score_targets)))

    extra_gate_candidate_count = 0
    if extra_gate_candidates:
        candidate_limit = int(os.environ.get(
            "GCL_BLIND_SWIGLU_EXTRA_GATE_CANDIDATE_LIMIT",
            "1536" if atom_count >= 16 else "512",
        ) or "0")
        usable = [
            np.asarray(gate, dtype=np.float64)
            for gate in extra_gate_candidates[:max(0, candidate_limit)]
            if np.all(np.isfinite(gate)) and np.linalg.norm(gate) > 1e-9
        ]
        # Up to 1536 of these at 16 atoms and above, each a block build and a least squares on
        # the host - the same shape of work as the dictionary scan, and on the same critical
        # path. Scored in one batch on the card when there is one, with the per-gate host scorer
        # kept as the fallback and as the definition of what the batch has to reproduce.
        quick_scores = batch_gate_candidate_mae(
            extra_score_coords, extra_score_targets, usable, quick_gate_candidate_score
        )
        for gate_value, gate_score in zip(usable, quick_scores):
            gate_candidate_pool.append((gate_score, gate_value))
            extra_gate_candidate_count += 1
        emit_progress("fit-extra-candidates-scored", scored=int(extra_gate_candidate_count))
    for gate_start, _up_start, _scale_start, _bias_start in structured_starts:
        for gate in gate_start:
            if np.all(np.isfinite(gate)) and np.linalg.norm(gate) > 1e-9:
                gate_candidate_pool.append((float("inf"), np.asarray(gate, dtype=np.float64)))
    ones = np.ones(coords.shape[0], dtype=np.float64)
    selection_fit_indices: np.ndarray | None = None
    selection_holdout_indices: np.ndarray | None = None
    if coords.shape[0] >= 1536:
        selection_rng = np.random.default_rng(seed + 90917)
        holdout_count = min(coords.shape[0] // 4, max(512, atom_count * 64))
        holdout_count = min(holdout_count, coords.shape[0] - max(512, atom_count * 64))
        if holdout_count > 0:
            permutation = selection_rng.permutation(coords.shape[0])
            selection_holdout_indices = np.sort(permutation[:holdout_count])
            selection_fit_indices = np.sort(permutation[holdout_count:])
    chunk_size = total_restart_count if restart_chunk_size <= 0 else max(1, min(total_restart_count, restart_chunk_size))
    chunk_specs: list[tuple[str, int, int]] = []
    for structured_start in range(0, structured_start_count, chunk_size):
        chunk_specs.append(("structured", structured_start, min(chunk_size, structured_start_count - structured_start)))
    for random_start in range(0, restart_count, chunk_size):
        chunk_specs.append(("random", random_start, min(chunk_size, restart_count - random_start)))

    def candidate_rank_score(candidate: dict[str, Any]) -> float:
        # A non-finite score must never compare as "better". Ranking used a bare `<`, and
        # every comparison against NaN is False, so a NaN score left the first candidate
        # in place and silently discarded the whole restart search. Mapping it to +inf
        # keeps a single diverged restart from winning; a run where every candidate is
        # non-finite is caught by the explicit check after the search.
        value = float(candidate.get("selectionScore", candidate.get("score", candidate["mae"])))
        return value if math.isfinite(value) else math.inf

    def cubic_relative_rmse(gates: np.ndarray, ups: np.ndarray, scales: np.ndarray) -> float | None:
        score_directions = (
            cubic_selection_directions
            if cubic_selection_directions is not None and cubic_selection_directions.shape[0] > 0
            else cubic_directions
        )
        score_targets = (
            cubic_selection_targets
            if cubic_selection_targets is not None and cubic_selection_targets.shape[0] > 0
            else cubic_targets
        )
        if score_directions is None or score_targets is None or score_directions.shape[0] != score_targets.shape[0]:
            return None
        # With cubic witnesses disabled the probe returns empty arrays rather than None,
        # which passes the guard above and then averages over nothing. "No witness" is
        # the honest answer here, not a NaN that poisons every selection score it is
        # added to.
        if score_directions.shape[0] == 0:
            return None
        predicted = predict_swiglu_cubic_directional_derivatives(
            score_directions,
            gates,
            ups,
            scales,
        )
        numerator = float(np.mean((predicted - score_targets) ** 2))
        denominator = max(float(np.mean(score_targets ** 2)), 1e-12)
        return math.sqrt(numerator / denominator)

    def residual_selection_score(prediction: np.ndarray, actual: np.ndarray) -> float:
        residual = np.abs(prediction - actual)
        # Scoring on an empty set is a configuration defect, not a score. The previous
        # guard read the mean first, so it returned the NaN it was meant to prevent.
        if residual.shape[0] == 0:
            raise RuntimeError("SwiGLU restart selection scored on an empty residual set")
        mae = float(np.mean(residual))
        score_value = mae
        if selection_tail_weight > 0.0:
            score_value += selection_tail_weight * float(np.quantile(residual, selection_tail_quantile))
        if selection_worst_weight > 0.0:
            score_value += selection_worst_weight * float(np.max(residual))
        return score_value

    def remember_top_candidate(candidate: dict[str, Any]) -> None:
        top_candidates.append(dict(candidate))
        top_candidates.sort(key=candidate_rank_score)
        keep_count = max(4, min(12, atom_count if atom_count >= 8 else 4))
        del top_candidates[keep_count:]

    def rescore_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
        gates_np = np.asarray(candidate["gates"], dtype=np.float64)
        ups_np = np.asarray(candidate["ups"], dtype=np.float64)
        scales_np = np.asarray(candidate["scales"], dtype=np.float64)
        bias_value = float(candidate["bias"])
        activation_np = silu_np(coords @ gates_np.T) * (coords @ ups_np.T)
        design = np.column_stack([activation_np, np.ones(coords.shape[0], dtype=np.float64)])
        try:
            coeffs, *_ = np.linalg.lstsq(design, targets, rcond=None)
            if np.all(np.isfinite(coeffs)):
                scales_np = coeffs[:-1].astype(np.float64)
                bias_value = float(coeffs[-1])
        except np.linalg.LinAlgError:
            pass
        prediction_np = activation_np @ scales_np + bias_value
        mae = float(np.mean(np.abs(prediction_np - targets)))
        mse = float(np.mean((prediction_np - targets) ** 2))
        selection_score = residual_selection_score(prediction_np, targets)
        if has_external_selection and selection_coords is not None and selection_targets is not None:
            selection_activation = silu_np(selection_coords @ gates_np.T) * (selection_coords @ ups_np.T)
            selection_prediction = selection_activation @ scales_np + bias_value
            selection_score = residual_selection_score(selection_prediction, selection_targets)
        elif selection_fit_indices is not None and selection_holdout_indices is not None:
            try:
                selection_coeffs, *_ = np.linalg.lstsq(
                    design[selection_fit_indices],
                    targets[selection_fit_indices],
                    rcond=None,
                )
                if np.all(np.isfinite(selection_coeffs)):
                    selection_prediction = design[selection_holdout_indices] @ selection_coeffs
                    selection_score = residual_selection_score(selection_prediction, targets[selection_holdout_indices])
            except np.linalg.LinAlgError:
                selection_score = residual_selection_score(prediction_np, targets)
        cubic_error = cubic_relative_rmse(gates_np, ups_np, scales_np)
        if cubic_error is not None:
            selection_score += selection_cubic_weight * cubic_error
        rescored = dict(candidate)
        rescored.update({
            "mae": mae,
            "mse": mse,
            "r2": 1.0 - mse / max(target_variance, 1e-12),
            "worstError": float(np.max(np.abs(prediction_np - targets))),
            "score": mae,
            "selectionScore": selection_score,
            "gates": gates_np,
            "ups": ups_np,
            "scales": scales_np,
            "bias": bias_value,
            "cubicRelativeRmse": cubic_error,
        })
        return rescored

    emit_progress("fit-chunks-start", chunks=len(chunk_specs), atomCount=int(atom_count), rows=int(coords.shape[0]))
    for chunk_kind, chunk_start, current_count in chunk_specs:
        if chunk_kind == "random":
            generator = torch.Generator(device=device)
            generator.manual_seed(seed + chunk_start)
            gates_init = torch.randn(current_count, atom_count, rank, dtype=fit_dtype, device=device, generator=generator)
            ups_init = torch.randn(current_count, atom_count, rank, dtype=fit_dtype, device=device, generator=generator)
            init_scale_values = torch.tensor(
                [random_init_scales[(chunk_start + index) % len(random_init_scales)] for index in range(current_count)],
                dtype=fit_dtype,
                device=device,
            ).view(current_count, 1, 1)
            gates_init = gates_init * init_scale_values
            ups_init = ups_init * init_scale_values
            scales_init = torch.ones(current_count, atom_count, dtype=fit_dtype, device=device) * max(0.1, target_scale / max(1, atom_count))
            bias_init = torch.ones(current_count, dtype=fit_dtype, device=device) * float(np.mean(targets))
        else:
            gates_init = torch.empty(current_count, atom_count, rank, dtype=fit_dtype, device=device)
            ups_init = torch.empty(current_count, atom_count, rank, dtype=fit_dtype, device=device)
            scales_init = torch.empty(current_count, atom_count, dtype=fit_dtype, device=device)
            bias_init = torch.empty(current_count, dtype=fit_dtype, device=device)
            for local_index in range(current_count):
                gate_start, up_start, scale_start, bias_start = structured_starts[chunk_start + local_index]
                gates_init[local_index] = torch.tensor(gate_start, dtype=fit_dtype, device=device)
                ups_init[local_index] = torch.tensor(up_start, dtype=fit_dtype, device=device)
                scales_init[local_index] = torch.tensor(scale_start, dtype=fit_dtype, device=device)
                bias_init[local_index] = float(bias_start)

        gates = torch.nn.Parameter(gates_init)
        ups = torch.nn.Parameter(ups_init)
        scales = torch.nn.Parameter(scales_init)
        bias = torch.nn.Parameter(bias_init)
        optimizer = torch.optim.Adam([gates, ups, scales, bias], lr=learning_rate)
        for _ in range(steps):
            optimizer.zero_grad()
            gate_projection = torch.einsum("nr,bkr->bnk", x, gates)
            up_projection = torch.einsum("nr,bkr->bnk", x, ups)
            activations = torch.nn.functional.silu(gate_projection) * up_projection
            prediction = torch.sum(activations * scales[:, None, :], dim=2) + bias[:, None]
            per_restart_loss = torch.mean((prediction - y[None, :]) ** 2, dim=1)
            if has_gradient_witness and gx is not None and gy is not None and target_grad_scale is not None:
                grad_gate_projection = torch.einsum("mr,bkr->bmk", gx, gates)
                grad_up_projection = torch.einsum("mr,bkr->bmk", gx, ups)
                sigmoid = torch.sigmoid(grad_gate_projection)
                silu_values = torch.nn.functional.silu(grad_gate_projection)
                silu_prime = sigmoid * (1.0 + grad_gate_projection * (1.0 - sigmoid))
                gradient_terms = scales[:, None, :, None] * (
                    silu_prime[:, :, :, None] * grad_up_projection[:, :, :, None] * gates[:, None, :, :]
                    + silu_values[:, :, :, None] * ups[:, None, :, :]
                )
                predicted_gradients = torch.sum(gradient_terms, dim=2)
                gradient_loss = torch.mean((predicted_gradients - gy[None, :, :]) ** 2, dim=(1, 2)) / target_grad_scale
                per_restart_loss = per_restart_loss + gradient_weight * gradient_loss
            if has_cubic_witness and cv is not None and cy is not None and target_cubic_scale is not None:
                cubic_gate_projection = torch.einsum("mr,bkr->bmk", cv, gates)
                cubic_up_projection = torch.einsum("mr,bkr->bmk", cv, ups)
                predicted_cubic = 1.5 * torch.sum(
                    cubic_gate_projection ** 2 * cubic_up_projection * scales[:, None, :],
                    dim=2,
                )
                cubic_loss = torch.mean((predicted_cubic - cy[None, :]) ** 2, dim=1) / target_cubic_scale
                per_restart_loss = per_restart_loss + cubic_weight * cubic_loss
            loss = torch.mean(per_restart_loss)
            loss.backward()
            optimizer.step()

        emit_progress("fit-chunk-optimised", chunkKind=chunk_kind, chunkStart=int(chunk_start), count=int(current_count))
        gates_np = gates.detach().cpu().numpy().astype(np.float64)
        ups_np = ups.detach().cpu().numpy().astype(np.float64)
        # The scoring loop below needs `silu(coords @ g.T) * (coords @ u.T)` for every restart in
        # the chunk, and it used to recompute all of it on the host - two `N x rank` by
        # `rank x atoms` products per restart, 512 restarts to a fit. Measured at N=32000,
        # atoms=32: about 30 s of host BLAS against 5 s for the least squares that follows, and
        # that is the load that showed up as 31 cores busy with the card idle.
        #
        # The card already holds `coords`, `gates` and `ups` - it just finished optimising them.
        # Computing the activations there in float64 gives the host the same quantity it was
        # about to compute, at the same precision. It is not bit-identical, because cuBLAS and
        # OpenBLAS sum in different orders, but it is the same arithmetic to ~1e-16 relative,
        # far below the 1e-3 scale at which the mae comparisons below choose between restarts.
        chunk_activations = None
        if str(x.device).startswith("cuda"):
            with torch.no_grad():
                coords_device = x.double()
                gates_device = gates.detach().double()
                ups_device = ups.detach().double()
                chunk_activations = (
                    torch.nn.functional.silu(coords_device @ gates_device.transpose(1, 2))
                    * (coords_device @ ups_device.transpose(1, 2))
                ).cpu().numpy()
                del coords_device, gates_device, ups_device
        for local_restart in range(current_count):
            if chunk_kind == "structured":
                restart = chunk_start + local_restart
                start_kind = "structured"
                random_restart_index = None
            else:
                random_restart_index = chunk_start + local_restart
                restart = structured_start_count + random_restart_index
                start_kind = "random"
            if chunk_activations is not None:
                activation_np = chunk_activations[local_restart]
            else:
                activation_np = silu_np(coords @ gates_np[local_restart].T) * (coords @ ups_np[local_restart].T)
            design = np.column_stack([activation_np, ones])
            try:
                coeffs, *_ = np.linalg.lstsq(design, targets, rcond=None)
                if not np.all(np.isfinite(coeffs)):
                    continue
            except np.linalg.LinAlgError:
                continue
            prediction_np = design @ coeffs
            mae = float(np.mean(np.abs(prediction_np - targets)))
            mse = float(np.mean((prediction_np - targets) ** 2))
            r2 = 1.0 - mse / max(target_variance, 1e-12)
            score = mae
            selection_score = residual_selection_score(prediction_np, targets)
            if has_external_selection and selection_coords is not None and selection_targets is not None:
                selection_activation = silu_np(selection_coords @ gates_np[local_restart].T) * (selection_coords @ ups_np[local_restart].T)
                selection_prediction = selection_activation @ coeffs[:-1].astype(np.float64) + float(coeffs[-1])
                selection_score = residual_selection_score(selection_prediction, selection_targets)
            elif selection_fit_indices is not None and selection_holdout_indices is not None:
                try:
                    selection_coeffs, *_ = np.linalg.lstsq(
                        design[selection_fit_indices],
                        targets[selection_fit_indices],
                        rcond=None,
                    )
                    if np.all(np.isfinite(selection_coeffs)):
                        selection_prediction = design[selection_holdout_indices] @ selection_coeffs
                        selection_score = residual_selection_score(selection_prediction, targets[selection_holdout_indices])
                except np.linalg.LinAlgError:
                    selection_score = residual_selection_score(prediction_np, targets)
            cubic_error = cubic_relative_rmse(
                gates_np[local_restart],
                ups_np[local_restart],
                coeffs[:-1].astype(np.float64),
            )
            if cubic_error is not None:
                selection_score += selection_cubic_weight * cubic_error
            candidate = {
                "restart": restart,
                "mae": mae,
                "mse": mse,
                "r2": r2,
                "score": score,
                "selectionScore": selection_score,
                "startKind": start_kind,
                "randomRestartIndex": random_restart_index,
                "gates": gates_np[local_restart],
                "ups": ups_np[local_restart],
                "scales": coeffs[:-1].astype(np.float64),
                "bias": float(coeffs[-1]),
                "cubicRelativeRmse": cubic_error,
            }
            for gate in gates_np[local_restart]:
                if np.all(np.isfinite(gate)) and np.linalg.norm(gate) > 1e-9:
                    gate_candidate_pool.append((score, gate.copy()))
            rank_score = candidate_rank_score(candidate)
            previous_kind_score = best_score_by_start_kind.get(start_kind)
            if previous_kind_score is None or rank_score < previous_kind_score:
                best_score_by_start_kind[start_kind] = rank_score
                best_metrics_by_start_kind[start_kind] = {
                    "restart": int(restart),
                    "randomRestartIndex": None if random_restart_index is None else int(random_restart_index),
                    "mae": mae,
                    "mse": mse,
                    "r2": r2,
                    "selectionScore": selection_score,
                }
            remember_top_candidate(candidate)
            if best is None or candidate_rank_score(candidate) < candidate_rank_score(best):
                best = candidate
        if len(gate_candidate_pool) > 4096:
            gate_candidate_pool.sort(key=lambda item: item[0])
            gate_candidate_pool = gate_candidate_pool[:2048]
        del gates, ups, scales, bias, optimizer, gates_np, ups_np
        torch.cuda.empty_cache()
    if best is None:
        raise RuntimeError("batched SwiGLU fit did not produce a finite candidate")
    if not math.isfinite(candidate_rank_score(best)):
        # Every restart scoring non-finite means the search ranked nothing: the winner
        # would be whichever candidate happened to be evaluated first. That is a broken
        # selection rule, and it stays loud rather than publishing an arbitrary fit.
        raise RuntimeError(
            "batched SwiGLU restart selection produced no finite selection score "
            f"across {total_restart_count} restarts"
        )
    forced_polish_candidates: list[dict[str, Any]] = []
    dictionary_rank_score: float | None = None
    if atom_count >= 8 and gate_candidate_pool:
        dictionary_fit = fit_swiglu_gate_dictionary(
            coords,
            targets,
            [gate for _score, gate in sorted(gate_candidate_pool, key=lambda item: item[0])],
            atom_count,
            seed + 424242,
        )
        if dictionary_fit is not None:
            dictionary_fit = rescore_candidate(dictionary_fit)
            dictionary_rank_score = candidate_rank_score(dictionary_fit)
            remember_top_candidate(dictionary_fit)
            forced_polish_candidates.append(dictionary_fit)
        if dictionary_fit is not None and candidate_rank_score(dictionary_fit) < candidate_rank_score(best):
            best = dictionary_fit
    polish_top_k = int(os.environ.get(
        "GCL_BLIND_SWIGLU_POLISH_TOP_K",
        "5" if atom_count >= 12 else "3",
    ))
    polish_candidates = top_candidates[:max(1, polish_top_k)]
    if not any(candidate.get("restart") == best.get("restart") and candidate.get("startKind") == best.get("startKind") for candidate in polish_candidates):
        polish_candidates.append(best)
    for forced_candidate in forced_polish_candidates:
        forced_signature = (forced_candidate.get("restart"), forced_candidate.get("startKind"))
        if not any((candidate.get("restart"), candidate.get("startKind")) == forced_signature for candidate in polish_candidates):
            polish_candidates.append(forced_candidate)
    polished_candidates: list[dict[str, Any]] = []
    for candidate in polish_candidates:
        polished = polish_swiglu_candidate(
            coords,
            targets,
            candidate,
            learning_rate,
            gradient_coords,
            gradient_targets,
            gradient_weight,
            cubic_directions,
            cubic_targets,
            cubic_weight,
        )
        polished_candidates.append(rescore_candidate(polished))
    best = min(polished_candidates, key=candidate_rank_score)
    best["polishedCandidateCount"] = len(polished_candidates)
    backfit_cycles = int(os.environ.get(
        "GCL_BLIND_SWIGLU_BACKFIT_CYCLES",
        "0",
    ))
    if backfit_cycles > 0:
        best = rescore_candidate(coordinate_backfit_swiglu_candidate(
            coords,
            targets,
            best,
            backfit_cycles,
        ))
    best = rescore_candidate(residual_region_atom_replace_swiglu_candidate(
        coords,
        targets,
        best,
    ))
    best = rescore_candidate(split_merge_tail_atom_refine_swiglu_candidate(
        coords,
        targets,
        best,
    ))
    # Runs after the residual and split/merge passes on purpose: both of those refit an atom
    # from where it already is, which cannot leave a collapsed pair's local minimum.
    best = rescore_candidate(respread_collapsed_atoms_swiglu_candidate(
        coords,
        targets,
        best,
    ))
    tail_polish_steps = int(os.environ.get(
        "GCL_BLIND_SWIGLU_TAIL_POLISH_STEPS",
        "420" if atom_count >= 12 else "0",
    ))
    if tail_polish_steps > 0:
        tail_polished = tail_error_polish_swiglu_candidate(
            coords,
            targets,
            best,
            tail_polish_steps,
        )
        if os.environ.get("GCL_BLIND_SWIGLU_TAIL_POLISH_RESCORE", "1").strip() == "0":
            best = tail_polished
        else:
            best = rescore_candidate(tail_polished)
    if os.environ.get("GCL_BLIND_SWIGLU_MINIMAX_BIAS_POLISH", "").strip() == "1":
        best = minimax_bias_polish_swiglu_candidate(
            coords,
            targets,
            best,
        )
    if int(os.environ.get("GCL_BLIND_SWIGLU_FINAL_LBFGS_STEPS", "0") or "0") > 0:
        best = final_lbfgs_polish_swiglu_candidate(
            coords,
            targets,
            best,
        )
    if os.environ.get("GCL_BLIND_SWIGLU_CHEBYSHEV_CALIBRATION", "").strip() == "1":
        best = chebyshev_calibrate_swiglu_candidate(
            coords,
            targets,
            best,
        )
    if os.environ.get("GCL_BLIND_SWIGLU_ROBUST_SCALAR_CALIBRATION", "").strip() == "1":
        best = robust_scalar_calibrate_swiglu_candidate(
            coords,
            targets,
            best,
        )
    best["restartCount"] = total_restart_count
    best["randomRestartCount"] = restart_count
    best["structuredStartCount"] = structured_start_count
    best["selectionTailWeight"] = selection_tail_weight
    best["selectionTailQuantile"] = selection_tail_quantile
    best["selectionWorstWeight"] = selection_worst_weight
    best["fitCubicWeight"] = cubic_weight
    best["selectionCubicWeight"] = selection_cubic_weight
    best["cubicRelativeRmse"] = cubic_relative_rmse(
        np.asarray(best["gates"], dtype=np.float64),
        np.asarray(best["ups"], dtype=np.float64),
        np.asarray(best["scales"], dtype=np.float64),
    )
    best["externalSelectionCount"] = int(selection_coords.shape[0]) if has_external_selection and selection_coords is not None else 0
    best["dictionaryCandidateRankScore"] = None if dictionary_rank_score is None else float(dictionary_rank_score)
    best["forcedPolishCandidateCount"] = len(forced_polish_candidates)
    best["extraGateCandidateCount"] = int(extra_gate_candidate_count)
    best["randomInitScales"] = [float(value) for value in random_init_scales]
    best["bestScoreByStartKind"] = {key: float(value) for key, value in best_score_by_start_kind.items()}
    best["bestMetricsByStartKind"] = best_metrics_by_start_kind
    best["restartChunkSize"] = chunk_size
    batched_mode = "batched-cuda" if chunk_size >= total_restart_count else "batched-cuda-chunked"
    suffixes = []
    if structured_start_count:
        suffixes.append("structured-starts")
    if best.get("dictionarySelectedCount"):
        suffixes.append("gate-dictionary-beam" if best.get("dictionaryBeamWidth") else "gate-dictionary")
    if best.get("adamRefined"):
        suffixes.append("adam-refine")
    if best.get("directionalGateValuePolished"):
        suffixes.append("directional-gate-value-polish")
    if best.get("fixedGateDirectionValuePolished"):
        suffixes.append("fixed-gate-value-polish")
    if best.get("minimaxBiasPolished"):
        suffixes.append("minimax-bias-polish")
    if best.get("finalLbfgsPolished"):
        suffixes.append("final-lbfgs-polish")
    if best.get("residualRegionAtomReplaced"):
        suffixes.append("residual-region-atom-replace")
    if best.get("splitMergeTailAtomRefined"):
        suffixes.append("split-merge-tail-atom")
    if best.get("chebyshevCalibrated"):
        suffixes.append("chebyshev-calibration")
    if best.get("robustScalarCalibrated"):
        suffixes.append("robust-scalar-calibration")
    if best.get("polished"):
        suffixes.append("lbfgs-polish")
    best["fitMode"] = "+".join([batched_mode] + suffixes)
    # Where the optimisation ran, read off a tensor rather than off the flag that asked for it.
    # This path has always been on the card, but nothing said so, and finding that out took a
    # profiler and `nvidia-smi pmon` on a live run. A field costs nothing and answers it.
    best["fitDevice"] = str(device)
    return best


def unique_gate_candidates(gates: list[np.ndarray], max_candidates: int) -> list[np.ndarray]:
    unique: list[np.ndarray] = []
    signatures: set[tuple[float, ...]] = set()
    scales = (0.5, 1.0, 2.0)
    for raw_gate in gates:
        gate = np.asarray(raw_gate, dtype=np.float64)
        norm = float(np.linalg.norm(gate))
        if norm < 1e-9 or not np.all(np.isfinite(gate)):
            continue
        direction = gate / norm
        if direction[np.argmax(np.abs(direction))] < 0.0:
            direction = -direction
        for scale in scales:
            candidate = direction * norm * scale
            candidate_norm = max(1e-9, np.linalg.norm(candidate))
            signature = (
                round(float(np.log(candidate_norm)), 3),
                *np.round(candidate / candidate_norm, 4).tolist(),
            )
            if signature in signatures:
                continue
            signatures.add(signature)
            unique.append(candidate)
            if len(unique) >= max_candidates:
                return unique
    return unique


def batch_gate_candidate_mae(
    coords: np.ndarray,
    targets: np.ndarray,
    gates: list[np.ndarray],
    host_scorer: Any,
) -> list[float]:
    """Mean absolute error of a one-atom fit for each candidate gate, in one batch.

    `host_scorer` is the per-gate function this replaces. It is passed in rather than
    reimplemented so the fallback cannot drift from the batched path, and so a machine without a
    card behaves exactly as before.

    Chunked over candidates: the design is `rows x (rank + 1)` per gate, so the whole set at once
    is not worth the memory. A gate whose solve is not finite scores `inf`, which is what the
    host scorer returns for the same case, and `inf` sorts to the back of the pool either way.
    """
    if not gates:
        return []
    try:
        import torch

        use_device = torch.cuda.is_available()
    except ImportError:
        use_device = False
    if not use_device:
        return [float(host_scorer(gate)) for gate in gates]

    rows, rank = int(coords.shape[0]), int(coords.shape[1])
    device_coords = torch.as_tensor(coords, device="cuda", dtype=torch.float64)
    device_targets = torch.as_tensor(targets, device="cuda", dtype=torch.float64)
    ones_column = torch.ones((rows, 1), device="cuda", dtype=torch.float64)
    chunk = max(1, min(len(gates), int(2 ** 27 // max(rows * (rank + 1), 1)) or 1))
    scores: list[float] = []
    for start in range(0, len(gates), chunk):
        block_gates = gates[start:start + chunk]
        count = len(block_gates)
        device_gates = torch.as_tensor(np.stack(block_gates), device="cuda", dtype=torch.float64)
        activation = torch.nn.functional.silu(device_coords @ device_gates.T).transpose(0, 1)
        blocks = activation.unsqueeze(2) * device_coords.unsqueeze(0)
        design = torch.cat([blocks, ones_column.expand(count, rows, 1)], dim=2)
        try:
            solution = torch.linalg.lstsq(design, device_targets.expand(count, rows).unsqueeze(2)).solution
        except RuntimeError:
            del blocks, design
            torch.cuda.empty_cache()
            scores.extend(float(host_scorer(gate)) for gate in block_gates)
            continue
        prediction = (design @ solution).squeeze(2)
        finite = torch.isfinite(solution).all(dim=1).squeeze(-1).cpu().numpy()
        mae = (prediction - device_targets.unsqueeze(0)).abs().mean(dim=1).cpu().numpy()
        del blocks, design, solution, prediction, activation
        torch.cuda.empty_cache()
        for position in range(count):
            scores.append(float(mae[position]) if bool(finite[position]) else float("inf"))
    return scores


def score_single_gate_blocks(
    xs: np.ndarray,
    residual: np.ndarray,
    candidates: np.ndarray,
    indices: list[int],
) -> list[tuple[int, float]]:
    """Residual left after projecting onto each candidate's block, one candidate at a time.

    This is the dictionary search's inner scan and it is where a 32-atom block spent its
    deadline. Per candidate it builds `silu(xs @ gate)[:, None] * xs` - 12288 by 64 at the sample
    size this uses - and solves a least squares against it. The beam runs it `atoms * width *
    pool` times: 32 by 4 by 384, which is 49152 solves.

    Measured at those shapes: 0.129 s a candidate on the host, so 6348 s - 106 minutes - for the
    scan alone, against a 120 minute deadline. On the card, 0.0180 s a candidate, 7x, and the
    scores agree to a median relative difference of 1.49e-16 and a largest of 4.48e-16, which is
    float64 rounding.

    Chunked, because the blocks are `chunk x rows x rank` and the whole pool at once is not worth
    the memory. Order is preserved, and a candidate whose solve returns non-finite coefficients is
    dropped exactly as the host loop dropped it.
    """
    if not indices:
        return []
    try:
        import torch

        use_device = torch.cuda.is_available()
    except ImportError:
        use_device = False
    if not use_device:
        scored: list[tuple[int, float]] = []
        for index in indices:
            block = silu_np(xs @ candidates[index]) [:, None] * xs
            try:
                coeffs, *_ = np.linalg.lstsq(block, residual, rcond=None)
            except np.linalg.LinAlgError:
                continue
            if not np.all(np.isfinite(coeffs)):
                continue
            projection = block @ coeffs
            scored.append((index, float(np.dot(residual - projection, residual - projection))))
        return scored

    rows = int(xs.shape[0])
    device_xs = torch.as_tensor(xs, device="cuda", dtype=torch.float64)
    device_residual = torch.as_tensor(residual, device="cuda", dtype=torch.float64)
    chunk = max(1, min(len(indices), int(2 ** 27 // max(rows * int(xs.shape[1]), 1)) or 1))
    scored: list[tuple[int, float]] = []
    for start in range(0, len(indices), chunk):
        block_indices = indices[start:start + chunk]
        gates = torch.as_tensor(
            np.stack([candidates[index] for index in block_indices]), device="cuda", dtype=torch.float64
        )
        blocks = torch.nn.functional.silu(device_xs @ gates.T).transpose(0, 1).unsqueeze(2) * device_xs.unsqueeze(0)
        targets = device_residual.expand(len(block_indices), rows).unsqueeze(2)
        try:
            solution = torch.linalg.lstsq(blocks, targets).solution
        except RuntimeError:
            del blocks
            torch.cuda.empty_cache()
            continue
        prediction = (blocks @ solution).squeeze(2)
        finite = torch.isfinite(solution).all(dim=1).squeeze(-1)
        scores = ((device_residual.unsqueeze(0) - prediction) ** 2).sum(dim=1)
        finite_host = finite.cpu().numpy()
        scores_host = scores.cpu().numpy()
        del blocks, targets, solution, prediction
        torch.cuda.empty_cache()
        for position, index in enumerate(block_indices):
            if not bool(finite_host[position]):
                continue
            scored.append((index, float(scores_host[position])))
    return scored


def fit_swiglu_gate_dictionary(
    coords: np.ndarray,
    targets: np.ndarray,
    gate_candidates: list[np.ndarray],
    atom_count: int,
    seed: int,
) -> dict[str, Any] | None:
    if not gate_candidates:
        return None
    rng = np.random.default_rng(seed)
    candidates = unique_gate_candidates(gate_candidates, max_candidates=512 if atom_count >= 12 else 256)
    if len(candidates) < atom_count:
        return None
    sample_count = min(coords.shape[0], max(2048, atom_count * 384))
    if sample_count < coords.shape[0]:
        sample_indices = np.sort(rng.choice(coords.shape[0], size=sample_count, replace=False))
        xs = coords[sample_indices]
        ys = targets[sample_indices]
    else:
        xs = coords
        ys = targets
    beam_width = int(os.environ.get(
        "GCL_BLIND_SWIGLU_DICTIONARY_BEAM_WIDTH",
        "4" if atom_count >= 16 else "1",
    ) or "1")
    beam_branch = int(os.environ.get(
        "GCL_BLIND_SWIGLU_DICTIONARY_BEAM_BRANCH",
        "24" if atom_count >= 16 else "12",
    ) or "12")
    beam_pool_limit = int(os.environ.get(
        "GCL_BLIND_SWIGLU_DICTIONARY_BEAM_POOL_LIMIT",
        "384" if atom_count >= 16 else "192",
    ) or "192")
    if beam_width < 1 or beam_branch < 1 or beam_pool_limit < atom_count:
        raise RuntimeError("GCL_BLIND_SWIGLU_DICTIONARY_BEAM_* values must be positive and pool_limit >= atom_count")

    def solve_sample_residual(selected_indices: list[int]) -> tuple[np.ndarray, float] | None:
        design_blocks = [silu_np(xs @ candidates[index])[:, None] * xs for index in selected_indices]
        design = np.column_stack(design_blocks + [np.ones(xs.shape[0], dtype=np.float64)])
        try:
            solved, *_ = np.linalg.lstsq(design, ys, rcond=None)
        except np.linalg.LinAlgError:
            return None
        if not np.all(np.isfinite(solved)):
            return None
        residual_value = ys - design @ solved
        score_value = float(np.dot(residual_value, residual_value))
        return residual_value, score_value

    if beam_width > 1:
        pool_count = min(len(candidates), beam_pool_limit)
        beams: list[tuple[float, list[int], np.ndarray]] = [
            (
                float(np.dot(ys.astype(np.float64) - float(np.mean(ys)), ys.astype(np.float64) - float(np.mean(ys)))),
                [],
                ys.astype(np.float64) - float(np.mean(ys)),
            )
        ]
        for _atom_index in range(atom_count):
            next_beams: list[tuple[float, list[int], np.ndarray]] = []
            for _score, selected_indices, residual in beams:
                selected_directions = [
                    candidates[index] / max(1e-9, np.linalg.norm(candidates[index]))
                    for index in selected_indices
                ]
                local_choices: list[tuple[float, int]] = []
                selected_set = set(selected_indices)
                eligible: list[int] = []
                for index in range(pool_count):
                    if index in selected_set:
                        continue
                    gate = candidates[index]
                    gate_direction = gate / max(1e-9, np.linalg.norm(gate))
                    if any(abs(float(np.dot(gate_direction, selected_direction))) > 0.995 for selected_direction in selected_directions):
                        continue
                    eligible.append(index)
                for index, score_value in score_single_gate_blocks(xs, residual, candidates, eligible):
                    local_choices.append((score_value, index))
                local_choices.sort(key=lambda item: item[0])
                for _candidate_score, index in local_choices[:beam_branch]:
                    new_selected = selected_indices + [index]
                    solved_residual = solve_sample_residual(new_selected)
                    if solved_residual is None:
                        continue
                    new_residual, new_score = solved_residual
                    next_beams.append((new_score, new_selected, new_residual))
            if not next_beams:
                break
            next_beams.sort(key=lambda item: item[0])
            deduped: list[tuple[float, list[int], np.ndarray]] = []
            signatures: set[tuple[int, ...]] = set()
            for item in next_beams:
                signature = tuple(sorted(item[1]))
                if signature in signatures:
                    continue
                signatures.add(signature)
                deduped.append(item)
                if len(deduped) >= beam_width:
                    break
            beams = deduped
        complete_beams = [beam for beam in beams if len(beam[1]) == atom_count]
        if complete_beams:
            complete_beams.sort(key=lambda item: item[0])
            selected = complete_beams[0][1]
            full_blocks = [silu_np(coords @ candidates[index])[:, None] * coords for index in selected]
            full_design = np.column_stack(full_blocks + [np.ones(coords.shape[0], dtype=np.float64)])
            try:
                solved, *_ = np.linalg.lstsq(full_design, targets, rcond=None)
            except np.linalg.LinAlgError:
                solved = None
            if solved is not None and np.all(np.isfinite(solved)):
                prediction = full_design @ solved
                mae = float(np.mean(np.abs(prediction - targets)))
                mse = float(np.mean((prediction - targets) ** 2))
                rank = coords.shape[1]
                up_vectors = []
                scales = []
                for atom_index in range(atom_count):
                    coeff_slice = solved[atom_index * rank:(atom_index + 1) * rank].astype(np.float64)
                    coeff_norm = float(np.linalg.norm(coeff_slice))
                    if coeff_norm < 1e-12:
                        up_vectors.append(np.zeros(rank, dtype=np.float64))
                        scales.append(0.0)
                    else:
                        up_vectors.append(coeff_slice / coeff_norm)
                        scales.append(coeff_norm)
                return {
                    "restart": -1,
                    "mae": mae,
                    "mse": mse,
                    "score": mae,
                    "selectionScore": mae,
                    "startKind": "dictionary-beam",
                    "randomRestartIndex": None,
                    "gates": np.vstack([candidates[index] for index in selected]).astype(np.float64),
                    "ups": np.vstack(up_vectors).astype(np.float64),
                    "scales": np.array(scales, dtype=np.float64),
                    "bias": float(solved[-1]),
                    "dictionarySelectedCount": len(selected),
                    "dictionaryBeamWidth": int(beam_width),
                    "dictionaryBeamBranch": int(beam_branch),
                    "dictionaryBeamPoolLimit": int(pool_count),
                }

    selected: list[int] = []
    selected_set: set[int] = set()
    selected_directions: list[np.ndarray] = []
    residual = ys.astype(np.float64) - float(np.mean(ys))
    for _ in range(atom_count):
        best_choice: tuple[float, int, np.ndarray] | None = None
        for index, gate in enumerate(candidates):
            if index in selected_set:
                continue
            gate_direction = gate / max(1e-9, np.linalg.norm(gate))
            if any(abs(float(np.dot(gate_direction, selected_direction))) > 0.995 for selected_direction in selected_directions):
                continue
            block = silu_np(xs @ gate)[:, None] * xs
            try:
                coeffs, *_ = np.linalg.lstsq(block, residual, rcond=None)
            except np.linalg.LinAlgError:
                continue
            if not np.all(np.isfinite(coeffs)):
                continue
            projection = block @ coeffs
            improvement = float(np.dot(residual, residual) - np.dot(residual - projection, residual - projection))
            if best_choice is None or improvement > best_choice[0]:
                best_choice = (improvement, index, coeffs.astype(np.float64))
        if best_choice is None or best_choice[0] <= 1e-12:
            break
        selected.append(best_choice[1])
        selected_set.add(best_choice[1])
        selected_gate = candidates[best_choice[1]]
        selected_directions.append(selected_gate / max(1e-9, np.linalg.norm(selected_gate)))
        design_blocks = [silu_np(xs @ candidates[index])[:, None] * xs for index in selected]
        design = np.column_stack(design_blocks + [np.ones(xs.shape[0], dtype=np.float64)])
        try:
            solved, *_ = np.linalg.lstsq(design, ys, rcond=None)
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(solved)):
            break
        residual = ys - design @ solved
    if len(selected) != atom_count:
        return None
    full_blocks = [silu_np(coords @ candidates[index])[:, None] * coords for index in selected]
    full_design = np.column_stack(full_blocks + [np.ones(coords.shape[0], dtype=np.float64)])
    try:
        solved, *_ = np.linalg.lstsq(full_design, targets, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(solved)):
        return None
    prediction = full_design @ solved
    mae = float(np.mean(np.abs(prediction - targets)))
    mse = float(np.mean((prediction - targets) ** 2))
    rank = coords.shape[1]
    up_vectors = []
    scales = []
    for atom_index in range(atom_count):
        coeff_slice = solved[atom_index * rank:(atom_index + 1) * rank].astype(np.float64)
        coeff_norm = float(np.linalg.norm(coeff_slice))
        if coeff_norm < 1e-12:
            up_vectors.append(np.zeros(rank, dtype=np.float64))
            scales.append(0.0)
        else:
            up_vectors.append(coeff_slice / coeff_norm)
            scales.append(coeff_norm)
    return {
        "restart": -1,
        "mae": mae,
        "mse": mse,
        "score": mae,
        "selectionScore": mae,
        "startKind": "dictionary",
        "randomRestartIndex": None,
        "gates": np.vstack([candidates[index] for index in selected]).astype(np.float64),
        "ups": np.vstack(up_vectors).astype(np.float64),
        "scales": np.array(scales, dtype=np.float64),
        "bias": float(solved[-1]),
        "dictionarySelectedCount": len(selected),
    }


def polish_swiglu_candidate(
    coords: np.ndarray,
    targets: np.ndarray,
    candidate: dict[str, Any],
    learning_rate: float,
    gradient_coords: np.ndarray | None,
    gradient_targets: np.ndarray | None,
    gradient_weight: float,
    cubic_directions: np.ndarray | None = None,
    cubic_targets: np.ndarray | None = None,
    cubic_weight: float = 0.0,
) -> dict[str, Any]:
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fit_dtype = torch.float32
    x = torch.tensor(coords, dtype=fit_dtype, device=device)
    y = torch.tensor(targets, dtype=fit_dtype, device=device)
    gates = torch.nn.Parameter(torch.tensor(candidate["gates"], dtype=fit_dtype, device=device))
    ups = torch.nn.Parameter(torch.tensor(candidate["ups"], dtype=fit_dtype, device=device))
    scales = torch.nn.Parameter(torch.tensor(candidate["scales"], dtype=fit_dtype, device=device))
    bias = torch.nn.Parameter(torch.tensor(float(candidate["bias"]), dtype=fit_dtype, device=device))
    has_gradient_witness = (
        gradient_coords is not None
        and gradient_targets is not None
        and gradient_weight > 0.0
        and gradient_coords.shape[0] > 0
    )
    if has_gradient_witness:
        gx = torch.tensor(gradient_coords, dtype=fit_dtype, device=device)
        gy = torch.tensor(gradient_targets, dtype=fit_dtype, device=device)
        target_grad_scale = torch.mean(gy ** 2).detach().clamp_min(1e-12)
    else:
        gx = None
        gy = None
        target_grad_scale = None
    has_cubic_witness = (
        cubic_directions is not None
        and cubic_targets is not None
        and cubic_weight > 0.0
        and cubic_directions.shape[0] > 0
        and cubic_directions.shape[0] == cubic_targets.shape[0]
    )
    if has_cubic_witness:
        cv = torch.tensor(cubic_directions, dtype=fit_dtype, device=device)
        cy = torch.tensor(cubic_targets, dtype=fit_dtype, device=device)
        target_cubic_scale = torch.mean(cy ** 2).detach().clamp_min(1e-12)
    else:
        cv = None
        cy = None
        target_cubic_scale = None

    lbfgs = torch.optim.LBFGS(
        [gates, ups, scales, bias],
        lr=max(0.2, min(0.8, learning_rate * 28.0)),
        max_iter=520 if gates.shape[0] >= 12 else 360,
        line_search_fn="strong_wolfe",
        # Left at the torch defaults of 1e-7 and 1e-9 this stopped early on quiet blocks.
        # The sibling refiners in this file already use these tighter values.
        tolerance_grad=1e-10,
        tolerance_change=1e-12,
    )

    # The data term was a mean square in raw oracle units, so it scales as amplitude
    # squared, while the gradient and cubic witness terms below are already divided by
    # their own scales. On a quiet block that leaves the data term around 1e-8 against
    # witnesses of order one, so the witnesses dominate the objective and the fit barely
    # follows the data at all. It also puts the loss under LBFGS's default
    # tolerance_change of 1e-9, which stops the optimiser after a step or two. Dividing by
    # the target scale makes the objective comparable across blocks, exactly as
    # fixed_gate_direction_value_polish and trust_region_structure_refine already do.
    polish_target_scale = max(
        float(np.std(np.asarray(targets, dtype=np.float64))),
        float(np.mean(np.abs(np.asarray(targets, dtype=np.float64)))),
        1e-30,
    )
    polish_scale_tensor = torch.tensor(polish_target_scale, dtype=fit_dtype, device=device)

    def closure() -> torch.Tensor:
        lbfgs.zero_grad()
        gate_projection = x @ gates.T
        up_projection = x @ ups.T
        activations = torch.nn.functional.silu(gate_projection) * up_projection
        prediction = activations @ scales + bias
        loss = torch.mean(((prediction - y) / polish_scale_tensor) ** 2)
        if has_gradient_witness and gx is not None and gy is not None and target_grad_scale is not None:
            grad_gate_projection = gx @ gates.T
            grad_up_projection = gx @ ups.T
            sigmoid = torch.sigmoid(grad_gate_projection)
            silu_values = torch.nn.functional.silu(grad_gate_projection)
            silu_prime = sigmoid * (1.0 + grad_gate_projection * (1.0 - sigmoid))
            gradient_terms = scales[None, :, None] * (
                silu_prime[:, :, None] * grad_up_projection[:, :, None] * gates[None, :, :]
                + silu_values[:, :, None] * ups[None, :, :]
            )
            predicted_gradients = torch.sum(gradient_terms, dim=1)
            loss = loss + gradient_weight * torch.mean((predicted_gradients - gy) ** 2) / target_grad_scale
        if has_cubic_witness and cv is not None and cy is not None and target_cubic_scale is not None:
            cubic_gate_projection = cv @ gates.T
            cubic_up_projection = cv @ ups.T
            predicted_cubic = 1.5 * torch.sum(
                cubic_gate_projection ** 2 * cubic_up_projection * scales[None, :],
                dim=1,
            )
            loss = loss + cubic_weight * torch.mean((predicted_cubic - cy) ** 2) / target_cubic_scale
        loss.backward()
        return loss

    try:
        lbfgs.step(closure)
    except RuntimeError:
        return candidate

    gates_np = gates.detach().cpu().numpy().astype(np.float64)
    ups_np = ups.detach().cpu().numpy().astype(np.float64)
    activation_np = silu_np(coords @ gates_np.T) * (coords @ ups_np.T)
    design = np.column_stack([activation_np, np.ones(coords.shape[0], dtype=np.float64)])
    try:
        coeffs, *_ = np.linalg.lstsq(design, targets, rcond=None)
        if not np.all(np.isfinite(coeffs)):
            return candidate
    except np.linalg.LinAlgError:
        return candidate

    prediction_np = design @ coeffs
    mae = float(np.mean(np.abs(prediction_np - targets)))
    mse = float(np.mean((prediction_np - targets) ** 2))
    enable_experimental_polish = os.environ.get("GCL_BLIND_SWIGLU_EXPERIMENTAL_POLISH", "").strip() == "1"
    if mae >= float(candidate["score"]):
        directional = candidate
        if enable_experimental_polish:
            refined = adam_refine_swiglu_candidate(
                coords,
                targets,
                candidate,
                gradient_coords,
                gradient_targets,
                gradient_weight,
            )
            directional = directional_gate_value_polish(
                coords,
                targets,
                refined,
                gradient_coords,
                gradient_targets,
                gradient_weight,
            )
        return fixed_gate_direction_value_polish(
            coords,
            targets,
            directional,
            gradient_coords,
            gradient_targets,
            gradient_weight,
        )
    polished = dict(candidate)
    polished.update({
        "mae": mae,
        "mse": mse,
        "score": mae,
        "gates": gates_np,
        "ups": ups_np,
        "scales": coeffs[:-1].astype(np.float64),
        "bias": float(coeffs[-1]),
        "polished": True,
        "polishMaeBefore": float(candidate["mae"]),
    })
    directional = polished
    if enable_experimental_polish:
        refined = adam_refine_swiglu_candidate(
            coords,
            targets,
            polished,
            gradient_coords,
            gradient_targets,
            gradient_weight,
        )
        directional = directional_gate_value_polish(
            coords,
            targets,
            refined,
            gradient_coords,
            gradient_targets,
            gradient_weight,
        )
    return fixed_gate_direction_value_polish(
        coords,
        targets,
        directional,
        gradient_coords,
        gradient_targets,
        gradient_weight,
    )


def adam_refine_swiglu_candidate(
    coords: np.ndarray,
    targets: np.ndarray,
    candidate: dict[str, Any],
    gradient_coords: np.ndarray | None,
    gradient_targets: np.ndarray | None,
    gradient_weight: float,
) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        return candidate
    gates_np = np.asarray(candidate["gates"], dtype=np.float64)
    if gates_np.ndim != 2 or gates_np.shape[0] < 2:
        return candidate
    device = torch.device("cuda")
    fit_dtype = torch.float32
    x = torch.tensor(coords, dtype=fit_dtype, device=device)
    y = torch.tensor(targets, dtype=fit_dtype, device=device)
    gates = torch.nn.Parameter(torch.tensor(gates_np, dtype=fit_dtype, device=device))
    ups = torch.nn.Parameter(torch.tensor(candidate["ups"], dtype=fit_dtype, device=device))
    scales = torch.nn.Parameter(torch.tensor(candidate["scales"], dtype=fit_dtype, device=device))
    bias = torch.nn.Parameter(torch.tensor(float(candidate["bias"]), dtype=fit_dtype, device=device))
    has_gradient_witness = (
        gradient_coords is not None
        and gradient_targets is not None
        and gradient_weight > 0.0
        and gradient_coords.shape[0] > 0
    )
    if has_gradient_witness:
        gx = torch.tensor(gradient_coords, dtype=fit_dtype, device=device)
        gy = torch.tensor(gradient_targets, dtype=fit_dtype, device=device)
        target_grad_scale = torch.mean(gy ** 2).detach().clamp_min(1e-12)
    else:
        gx = None
        gy = None
        target_grad_scale = None
    optimizer = torch.optim.AdamW([gates, ups, scales, bias], lr=0.006, weight_decay=2e-6)
    best_mae = float(candidate["mae"])
    best_state: tuple[np.ndarray, np.ndarray, np.ndarray, float] | None = None
    step_count = 2600 if gates_np.shape[0] >= 12 else 1600
    for step in range(step_count):
        optimizer.zero_grad()
        gate_projection = x @ gates.T
        up_projection = x @ ups.T
        activations = torch.nn.functional.silu(gate_projection) * up_projection
        prediction = activations @ scales + bias
        loss = torch.mean((prediction - y) ** 2)
        if has_gradient_witness and gx is not None and gy is not None and target_grad_scale is not None:
            grad_gate_projection = gx @ gates.T
            grad_up_projection = gx @ ups.T
            sigmoid = torch.sigmoid(grad_gate_projection)
            silu_values = torch.nn.functional.silu(grad_gate_projection)
            silu_prime = sigmoid * (1.0 + grad_gate_projection * (1.0 - sigmoid))
            gradient_terms = scales[None, :, None] * (
                silu_prime[:, :, None] * grad_up_projection[:, :, None] * gates[None, :, :]
                + silu_values[:, :, None] * ups[None, :, :]
            )
            predicted_gradients = torch.sum(gradient_terms, dim=1)
            loss = loss + gradient_weight * torch.mean((predicted_gradients - gy) ** 2) / target_grad_scale
        loss.backward()
        torch.nn.utils.clip_grad_norm_([gates, ups, scales, bias], 8.0)
        optimizer.step()
        if step == 0 or (step + 1) % 160 == 0 or step + 1 == step_count:
            with torch.no_grad():
                pred = (torch.nn.functional.silu(x @ gates.T) * (x @ ups.T)) @ scales + bias
                mae = float(torch.mean(torch.abs(pred - y)).detach().cpu().item())
                if mae < best_mae:
                    best_mae = mae
                    best_state = (
                        gates.detach().cpu().numpy().astype(np.float64),
                        ups.detach().cpu().numpy().astype(np.float64),
                        scales.detach().cpu().numpy().astype(np.float64),
                        float(bias.detach().cpu().item()),
                    )
    if best_state is None:
        return candidate
    gates_np, ups_np, scales_np, bias_value = best_state
    activation_np = silu_np(coords @ gates_np.T) * (coords @ ups_np.T)
    design = np.column_stack([activation_np, np.ones(coords.shape[0], dtype=np.float64)])
    try:
        coeffs, *_ = np.linalg.lstsq(design, targets, rcond=None)
    except np.linalg.LinAlgError:
        coeffs = None
    if coeffs is not None and np.all(np.isfinite(coeffs)):
        scales_np = coeffs[:-1].astype(np.float64)
        bias_value = float(coeffs[-1])
        prediction_np = design @ coeffs
    else:
        prediction_np = activation_np @ scales_np + bias_value
    mae = float(np.mean(np.abs(prediction_np - targets)))
    mse = float(np.mean((prediction_np - targets) ** 2))
    if mae >= float(candidate["score"]):
        return candidate
    refined = dict(candidate)
    refined.update({
        "mae": mae,
        "mse": mse,
        "score": mae,
        "gates": gates_np.astype(np.float64),
        "ups": ups_np.astype(np.float64),
        "scales": scales_np,
        "bias": bias_value,
        "adamRefined": True,
        "adamRefineMaeBefore": float(candidate["mae"]),
    })
    if candidate.get("polished"):
        refined["polished"] = True
    return refined


def directional_gate_value_polish(
    coords: np.ndarray,
    targets: np.ndarray,
    candidate: dict[str, Any],
    gradient_coords: np.ndarray | None,
    gradient_targets: np.ndarray | None,
    gradient_weight: float,
) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        return candidate
    gates = np.asarray(candidate["gates"], dtype=np.float64)
    if gates.ndim != 2 or gates.shape[0] < 2:
        return candidate
    gate_norms = np.linalg.norm(gates, axis=1).clip(min=1e-6)
    gate_dirs = gates / gate_norms[:, None]
    base_ups = np.asarray(candidate["ups"], dtype=np.float64) * np.asarray(candidate["scales"], dtype=np.float64)[:, None]

    device = torch.device("cuda")
    fit_dtype = torch.float32
    x = torch.tensor(coords, dtype=fit_dtype, device=device)
    y = torch.tensor(targets, dtype=fit_dtype, device=device)
    gate_raw = torch.nn.Parameter(torch.tensor(gate_dirs, dtype=fit_dtype, device=device))
    log_gate_norms = torch.nn.Parameter(torch.tensor(np.log(gate_norms), dtype=fit_dtype, device=device))
    up_vectors = torch.nn.Parameter(torch.tensor(base_ups, dtype=fit_dtype, device=device))
    bias = torch.nn.Parameter(torch.tensor(float(candidate["bias"]), dtype=fit_dtype, device=device))
    has_gradient_witness = (
        gradient_coords is not None
        and gradient_targets is not None
        and gradient_weight > 0.0
        and gradient_coords.shape[0] > 0
    )
    if has_gradient_witness:
        gx = torch.tensor(gradient_coords, dtype=fit_dtype, device=device)
        gy = torch.tensor(gradient_targets, dtype=fit_dtype, device=device)
        target_grad_scale = torch.mean(gy ** 2).detach().clamp_min(1e-12)
    else:
        gx = None
        gy = None
        target_grad_scale = None

    optimizer = torch.optim.AdamW([gate_raw, log_gate_norms, up_vectors, bias], lr=0.012, weight_decay=2e-6)
    best_mae = float(candidate["mae"])
    best_state: tuple[np.ndarray, np.ndarray, float] | None = None
    step_count = 3200 if gates.shape[0] >= 12 else 2200
    for step in range(step_count):
        optimizer.zero_grad()
        gate_dirs_t = torch.nn.functional.normalize(gate_raw, dim=1, eps=1e-8)
        gate_vectors = gate_dirs_t * torch.exp(log_gate_norms).clamp(1e-5, 20.0)[:, None]
        gate_projection = x @ gate_vectors.T
        up_projection = x @ up_vectors.T
        prediction = torch.sum(torch.nn.functional.silu(gate_projection) * up_projection, dim=1) + bias
        loss = torch.mean((prediction - y) ** 2)
        if has_gradient_witness and gx is not None and gy is not None and target_grad_scale is not None:
            grad_gate_projection = gx @ gate_vectors.T
            grad_up_projection = gx @ up_vectors.T
            sigmoid = torch.sigmoid(grad_gate_projection)
            silu_values = torch.nn.functional.silu(grad_gate_projection)
            silu_prime = sigmoid * (1.0 + grad_gate_projection * (1.0 - sigmoid))
            gradient_terms = (
                silu_prime[:, :, None] * grad_up_projection[:, :, None] * gate_vectors[None, :, :]
                + silu_values[:, :, None] * up_vectors[None, :, :]
            )
            predicted_gradients = torch.sum(gradient_terms, dim=1)
            loss = loss + gradient_weight * torch.mean((predicted_gradients - gy) ** 2) / target_grad_scale
        if gates.shape[0] >= 4:
            cosine = gate_dirs_t @ gate_dirs_t.T
            offdiag = cosine - torch.eye(gates.shape[0], dtype=fit_dtype, device=device)
            loss = loss + 1e-4 * torch.mean(torch.relu(torch.abs(offdiag) - 0.995) ** 2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([gate_raw, log_gate_norms, up_vectors, bias], 8.0)
        optimizer.step()
        if step == 0 or (step + 1) % 160 == 0 or step + 1 == step_count:
            with torch.no_grad():
                gate_dirs_t = torch.nn.functional.normalize(gate_raw, dim=1, eps=1e-8)
                gate_vectors = gate_dirs_t * torch.exp(log_gate_norms).clamp(1e-5, 20.0)[:, None]
                prediction = torch.sum(torch.nn.functional.silu(x @ gate_vectors.T) * (x @ up_vectors.T), dim=1) + bias
                mae = float(torch.mean(torch.abs(prediction - y)).detach().cpu().item())
                if mae < best_mae:
                    best_mae = mae
                    best_state = (
                        gate_vectors.detach().cpu().numpy().astype(np.float64),
                        up_vectors.detach().cpu().numpy().astype(np.float64),
                        float(bias.detach().cpu().item()),
                    )
    if best_state is None:
        return candidate
    gates_np, ups_np, bias_value = best_state
    activation_np = silu_np(coords @ gates_np.T) * (coords @ ups_np.T)
    design = np.column_stack([activation_np, np.ones(coords.shape[0], dtype=np.float64)])
    try:
        coeffs, *_ = np.linalg.lstsq(design, targets, rcond=None)
    except np.linalg.LinAlgError:
        coeffs = None
    if coeffs is not None and np.all(np.isfinite(coeffs)):
        ups_np = ups_np * coeffs[:-1, None]
        bias_value = float(coeffs[-1])
        prediction_np = design @ coeffs
    else:
        prediction_np = activation_np.sum(axis=1) + bias_value
    mae = float(np.mean(np.abs(prediction_np - targets)))
    mse = float(np.mean((prediction_np - targets) ** 2))
    if mae >= float(candidate["score"]):
        return candidate
    polished = dict(candidate)
    polished.update({
        "mae": mae,
        "mse": mse,
        "score": mae,
        "gates": gates_np.astype(np.float64),
        "ups": ups_np.astype(np.float64),
        "scales": np.ones(gates.shape[0], dtype=np.float64),
        "bias": bias_value,
        "directionalGateValuePolished": True,
        "directionalGateValueMaeBefore": float(candidate["mae"]),
    })
    if candidate.get("polished"):
        polished["polished"] = True
    return polished


def fixed_gate_direction_value_polish(
    coords: np.ndarray,
    targets: np.ndarray,
    candidate: dict[str, Any],
    gradient_coords: np.ndarray | None,
    gradient_targets: np.ndarray | None,
    gradient_weight: float,
) -> dict[str, Any]:
    import torch

    gates = np.asarray(candidate["gates"], dtype=np.float64)
    gate_norms = np.linalg.norm(gates, axis=1).clip(min=1e-6)
    gate_dirs = gates / gate_norms[:, None]
    base_ups = np.asarray(candidate["ups"], dtype=np.float64) * np.asarray(candidate["scales"], dtype=np.float64)[:, None]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fit_dtype = torch.float32
    x = torch.tensor(coords, dtype=fit_dtype, device=device)
    y = torch.tensor(targets, dtype=fit_dtype, device=device)
    target_scale = torch.std(y).detach().clamp_min(1e-8)
    gate_dirs_t = torch.tensor(gate_dirs, dtype=fit_dtype, device=device)
    initial_log_gate_norms = torch.tensor(np.log(gate_norms), dtype=fit_dtype, device=device)
    log_gate_norms = torch.nn.Parameter(initial_log_gate_norms.clone())
    up_vectors = torch.nn.Parameter(torch.tensor(base_ups, dtype=fit_dtype, device=device))
    bias = torch.nn.Parameter(torch.tensor(float(candidate["bias"]), dtype=fit_dtype, device=device))
    has_gradient_witness = (
        gradient_coords is not None
        and gradient_targets is not None
        and gradient_weight > 0.0
        and gradient_coords.shape[0] > 0
    )
    if has_gradient_witness:
        gx = torch.tensor(gradient_coords, dtype=fit_dtype, device=device)
        gy = torch.tensor(gradient_targets, dtype=fit_dtype, device=device)
        target_grad_scale = torch.mean(gy ** 2).detach().clamp_min(1e-12)
    else:
        gx = None
        gy = None
        target_grad_scale = None

    norm_lr = float(os.environ.get("GCL_BLIND_SWIGLU_FIXED_DIRECTION_NORM_LR", "0.012") or "0.012")
    value_lr = float(os.environ.get("GCL_BLIND_SWIGLU_FIXED_DIRECTION_VALUE_LR", "0.002") or "0.002")
    max_norm_ratio = float(os.environ.get("GCL_BLIND_SWIGLU_FIXED_DIRECTION_MAX_NORM_RATIO", "6.0") or "6.0")
    if (
        not np.isfinite(norm_lr)
        or norm_lr <= 0.0
        or not np.isfinite(value_lr)
        or value_lr <= 0.0
        or not np.isfinite(max_norm_ratio)
        or max_norm_ratio < 1.0
    ):
        raise RuntimeError("fixed-direction value polish learning rates/norm ratio are invalid")
    optimizer = torch.optim.AdamW([
        {"params": [log_gate_norms], "lr": norm_lr},
        {"params": [up_vectors, bias], "lr": value_lr},
    ], weight_decay=1e-8)
    log_norm_min = initial_log_gate_norms - math.log(max_norm_ratio)
    log_norm_max = initial_log_gate_norms + math.log(max_norm_ratio)
    best_state: tuple[float, np.ndarray, np.ndarray, float] | None = None
    default_steps = 2800 if gates.shape[0] >= 12 else 1800
    step_count = int(os.environ.get(
        "GCL_BLIND_SWIGLU_FIXED_DIRECTION_VALUE_STEPS",
        str(default_steps),
    ) or str(default_steps))
    if step_count <= 0:
        raise RuntimeError("GCL_BLIND_SWIGLU_FIXED_DIRECTION_VALUE_STEPS must be positive")
    for step in range(step_count):
        optimizer.zero_grad()
        gate_norms_t = torch.exp(log_gate_norms).clamp(1e-5, 20.0)
        gate_vectors = gate_dirs_t * gate_norms_t[:, None]
        gate_projection = x @ gate_vectors.T
        up_projection = x @ up_vectors.T
        prediction = torch.sum(torch.nn.functional.silu(gate_projection) * up_projection, dim=1) + bias
        loss = torch.mean(((prediction - y) / target_scale) ** 2)
        if has_gradient_witness and gx is not None and gy is not None and target_grad_scale is not None:
            grad_gate_projection = gx @ gate_vectors.T
            grad_up_projection = gx @ up_vectors.T
            sigmoid = torch.sigmoid(grad_gate_projection)
            silu_values = torch.nn.functional.silu(grad_gate_projection)
            silu_prime = sigmoid * (1.0 + grad_gate_projection * (1.0 - sigmoid))
            gradient_terms = (
                silu_prime[:, :, None] * grad_up_projection[:, :, None] * gate_vectors[None, :, :]
                + silu_values[:, :, None] * up_vectors[None, :, :]
            )
            predicted_gradients = torch.sum(gradient_terms, dim=1)
            loss = loss + gradient_weight * torch.mean((predicted_gradients - gy) ** 2) / target_grad_scale
        loss.backward()
        torch.nn.utils.clip_grad_norm_([log_gate_norms, up_vectors, bias], 10.0)
        optimizer.step()
        with torch.no_grad():
            log_gate_norms.clamp_(log_norm_min, log_norm_max)
        if step == 0 or (step + 1) % 100 == 0:
            with torch.no_grad():
                plain_prediction = torch.sum(
                    torch.nn.functional.silu((x @ (gate_dirs_t * torch.exp(log_gate_norms).clamp(1e-5, 20.0)[:, None]).T))
                    * (x @ up_vectors.T),
                    dim=1,
                ) + bias
                plain_loss = float(torch.mean(((plain_prediction - y) / target_scale) ** 2).detach().cpu().item())
                if best_state is None or plain_loss < best_state[0]:
                    best_state = (
                        plain_loss,
                        torch.exp(log_gate_norms).clamp(1e-5, 20.0).detach().cpu().numpy().astype(np.float64),
                        up_vectors.detach().cpu().numpy().astype(np.float64),
                        float(bias.detach().cpu().item()),
                    )
    if best_state is None:
        return candidate
    _loss, norm_values, up_values, bias_value = best_state
    gate_values = gate_dirs * norm_values[:, None]
    solved = solve_fixed_gate_start(coords, targets, gate_values)
    if solved is not None:
        gate_values, normalized_ups, solved_scales, bias_value, _solved_mae = solved
        up_values = normalized_ups
        scale_values = solved_scales
    else:
        up_norms = np.linalg.norm(up_values, axis=1).clip(min=1e-12)
        scale_values = up_norms
        up_values = up_values / up_norms[:, None]
    prediction_np = np.sum(
        silu_np(coords @ gate_values.T)
        * (coords @ up_values.T)
        * scale_values[None, :],
        axis=1,
    ) + bias_value
    metrics = regression_metrics(targets, prediction_np)
    base_metrics = regression_metrics(targets, predict_swiglu(coords, candidate))
    if (
        metrics["mae"] >= base_metrics["mae"]
        and metrics["worstError"] >= base_metrics["worstError"]
    ):
        return candidate
    polished = dict(candidate)
    polished.update({
        "mae": float(metrics["mae"]),
        "mse": float(metrics["mse"]),
        "r2": float(metrics["r2"]),
        "score": float(metrics["mae"]),
        "selectionScore": float(metrics["mae"]),
        "worstError": float(metrics["worstError"]),
        "gates": gate_values.astype(np.float64),
        "ups": up_values.astype(np.float64),
        "scales": scale_values.astype(np.float64),
        "bias": bias_value,
        "fixedGateDirectionValuePolished": True,
        "fixedGateDirectionMaeBefore": float(candidate["mae"]),
        "fixedGateDirectionValueSteps": int(step_count),
        "fixedGateDirectionNormLearningRate": float(norm_lr),
        "fixedGateDirectionValueLearningRate": float(value_lr),
        "fixedGateDirectionMaxNormRatio": float(max_norm_ratio),
    })
    if candidate.get("polished"):
        polished["polished"] = True
    return polished


def coordinate_backfit_swiglu_candidate(
    coords: np.ndarray,
    targets: np.ndarray,
    candidate: dict[str, Any],
    cycles: int,
) -> dict[str, Any]:
    if cycles <= 0:
        return candidate
    import torch

    gates_np = np.asarray(candidate["gates"], dtype=np.float64)
    ups_np = np.asarray(candidate["ups"], dtype=np.float64)
    scales_np = np.asarray(candidate["scales"], dtype=np.float64)
    if gates_np.ndim != 2 or ups_np.shape != gates_np.shape or scales_np.shape[0] != gates_np.shape[0]:
        return candidate
    atom_count, rank = gates_np.shape
    if atom_count < 2:
        return candidate

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fit_dtype = torch.float32
    x = torch.tensor(coords, dtype=fit_dtype, device=device)
    y = torch.tensor(targets, dtype=fit_dtype, device=device)
    gates = torch.tensor(gates_np, dtype=fit_dtype, device=device)
    up_vectors = torch.tensor(ups_np * scales_np[:, None], dtype=fit_dtype, device=device)
    bias_value = float(candidate["bias"])
    step_count = 420 if atom_count >= 12 else 300

    for _cycle in range(cycles):
        with torch.no_grad():
            contributions = torch.nn.functional.silu(x @ gates.T) * (x @ up_vectors.T)
        for atom_index in range(atom_count):
            with torch.no_grad():
                residual = y - float(bias_value) - torch.sum(contributions, dim=1) + contributions[:, atom_index]
            gate = torch.nn.Parameter(gates[atom_index].detach().clone())
            up = torch.nn.Parameter(up_vectors[atom_index].detach().clone())
            optimizer = torch.optim.AdamW([gate, up], lr=0.018, weight_decay=1e-7)
            for _step in range(step_count):
                optimizer.zero_grad()
                prediction = torch.nn.functional.silu(x @ gate) * (x @ up)
                loss = torch.mean((prediction - residual) ** 2)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([gate, up], 8.0)
                optimizer.step()
            lbfgs = torch.optim.LBFGS([gate, up], lr=0.45, max_iter=80, line_search_fn="strong_wolfe")

            def closure() -> torch.Tensor:
                lbfgs.zero_grad()
                prediction = torch.nn.functional.silu(x @ gate) * (x @ up)
                loss = torch.mean((prediction - residual) ** 2)
                loss.backward()
                return loss

            try:
                lbfgs.step(closure)
            except RuntimeError:
                pass
            with torch.no_grad():
                gates[atom_index] = gate.detach()
                up_vectors[atom_index] = up.detach()
                contributions[:, atom_index] = torch.nn.functional.silu(x @ gates[atom_index]) * (x @ up_vectors[atom_index])

        activation_np = contributions.detach().cpu().numpy().astype(np.float64)
        design = np.column_stack([activation_np, np.ones(coords.shape[0], dtype=np.float64)])
        try:
            coeffs, *_ = np.linalg.lstsq(design, targets, rcond=None)
        except np.linalg.LinAlgError:
            continue
        if not np.all(np.isfinite(coeffs)):
            continue
        with torch.no_grad():
            up_vectors *= torch.tensor(coeffs[:-1], dtype=fit_dtype, device=device)[:, None]
            bias_value = float(coeffs[-1])

    gates_final = gates.detach().cpu().numpy().astype(np.float64)
    ups_final = up_vectors.detach().cpu().numpy().astype(np.float64)
    activation_np = silu_np(coords @ gates_final.T) * (coords @ ups_final.T)
    design = np.column_stack([activation_np, np.ones(coords.shape[0], dtype=np.float64)])
    try:
        coeffs, *_ = np.linalg.lstsq(design, targets, rcond=None)
    except np.linalg.LinAlgError:
        coeffs = None
    if coeffs is not None and np.all(np.isfinite(coeffs)):
        ups_final = ups_final * coeffs[:-1, None]
        bias_value = float(coeffs[-1])
        prediction_np = design @ coeffs
    else:
        prediction_np = activation_np.sum(axis=1) + bias_value
    mae = float(np.mean(np.abs(prediction_np - targets)))
    mse = float(np.mean((prediction_np - targets) ** 2))
    if mae >= float(candidate["score"]):
        return candidate
    backfit = dict(candidate)
    backfit.update({
        "mae": mae,
        "mse": mse,
        "score": mae,
        "gates": gates_final,
        "ups": ups_final,
        "scales": np.ones(atom_count, dtype=np.float64),
        "bias": bias_value,
        "coordinateBackfit": True,
        "coordinateBackfitCycles": int(cycles),
        "coordinateBackfitMaeBefore": float(candidate["mae"]),
    })
    return backfit


def tail_error_polish_swiglu_candidate(
    coords: np.ndarray,
    targets: np.ndarray,
    candidate: dict[str, Any],
    steps: int,
    selection_coords: np.ndarray | None = None,
    selection_targets: np.ndarray | None = None,
) -> dict[str, Any]:
    if steps <= 0:
        return candidate
    import torch

    gates_np = np.asarray(candidate["gates"], dtype=np.float64)
    ups_np = np.asarray(candidate["ups"], dtype=np.float64)
    scales_np = np.asarray(candidate["scales"], dtype=np.float64)
    if gates_np.ndim != 2 or ups_np.shape != gates_np.shape or scales_np.shape[0] != gates_np.shape[0]:
        return candidate
    atom_count = gates_np.shape[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fit_dtype = torch.float32
    x = torch.tensor(coords, dtype=fit_dtype, device=device)
    y = torch.tensor(targets, dtype=fit_dtype, device=device)
    base_prediction_np = predict_swiglu(coords, candidate)
    base_prediction = torch.tensor(base_prediction_np, dtype=fit_dtype, device=device)
    base_abs_residual = torch.tensor(np.abs(base_prediction_np - targets), dtype=fit_dtype, device=device)
    if selection_coords is not None and selection_targets is not None and selection_coords.shape[0] > 0:
        selection_x = torch.tensor(selection_coords, dtype=fit_dtype, device=device)
        selection_y = torch.tensor(selection_targets, dtype=fit_dtype, device=device)
    else:
        selection_x = x
        selection_y = y
    gates = torch.nn.Parameter(torch.tensor(gates_np, dtype=fit_dtype, device=device))
    up_vectors = torch.nn.Parameter(torch.tensor(ups_np * scales_np[:, None], dtype=fit_dtype, device=device))
    bias = torch.nn.Parameter(torch.tensor(float(candidate["bias"]), dtype=fit_dtype, device=device))
    tail_weight = float(os.environ.get("GCL_BLIND_SWIGLU_TAIL_POLISH_WEIGHT", "0.35") or "0.35")
    tail_fraction = float(os.environ.get("GCL_BLIND_SWIGLU_TAIL_POLISH_FRACTION", "0.125") or "0.125")
    tail_lr = float(os.environ.get("GCL_BLIND_SWIGLU_TAIL_POLISH_LR", "0.006") or "0.006")
    tail_loss_kind = os.environ.get("GCL_BLIND_SWIGLU_TAIL_POLISH_LOSS", "topk").strip().lower() or "topk"
    tail_smoothness = float(os.environ.get("GCL_BLIND_SWIGLU_TAIL_POLISH_SMOOTHNESS", "40.0") or "40.0")
    tail_calibration = os.environ.get("GCL_BLIND_SWIGLU_TAIL_POLISH_CALIBRATION", "ls").strip().lower() or "ls"
    anchor_weight = float(os.environ.get("GCL_BLIND_SWIGLU_TAIL_POLISH_ANCHOR_WEIGHT", "0.0") or "0.0")
    drift_weight = float(os.environ.get("GCL_BLIND_SWIGLU_TAIL_POLISH_DRIFT_WEIGHT", "0.0") or "0.0")
    mae_weight = float(os.environ.get("GCL_BLIND_SWIGLU_TAIL_POLISH_MAE_WEIGHT", "0.0") or "0.0")
    best_mae_weight = float(os.environ.get("GCL_BLIND_SWIGLU_TAIL_POLISH_BEST_MAE_WEIGHT", "0.0") or "0.0")
    if not np.isfinite(tail_weight) or tail_weight < 0.0:
        raise RuntimeError("GCL_BLIND_SWIGLU_TAIL_POLISH_WEIGHT must be non-negative and finite")
    if not np.isfinite(tail_fraction) or tail_fraction <= 0.0 or tail_fraction > 1.0:
        raise RuntimeError("GCL_BLIND_SWIGLU_TAIL_POLISH_FRACTION must be in (0, 1]")
    if not np.isfinite(tail_lr) or tail_lr <= 0.0:
        raise RuntimeError("GCL_BLIND_SWIGLU_TAIL_POLISH_LR must be positive and finite")
    if tail_loss_kind not in {"topk", "smoothmax"}:
        raise RuntimeError("GCL_BLIND_SWIGLU_TAIL_POLISH_LOSS must be topk or smoothmax")
    if tail_calibration not in {"ls", "direct", "best-worst"}:
        raise RuntimeError("GCL_BLIND_SWIGLU_TAIL_POLISH_CALIBRATION must be ls, direct, or best-worst")
    if not np.isfinite(tail_smoothness) or tail_smoothness <= 0.0:
        raise RuntimeError("GCL_BLIND_SWIGLU_TAIL_POLISH_SMOOTHNESS must be positive and finite")
    for name, value in [
        ("GCL_BLIND_SWIGLU_TAIL_POLISH_ANCHOR_WEIGHT", anchor_weight),
        ("GCL_BLIND_SWIGLU_TAIL_POLISH_DRIFT_WEIGHT", drift_weight),
        ("GCL_BLIND_SWIGLU_TAIL_POLISH_MAE_WEIGHT", mae_weight),
        ("GCL_BLIND_SWIGLU_TAIL_POLISH_BEST_MAE_WEIGHT", best_mae_weight),
    ]:
        if not np.isfinite(value) or value < 0.0:
            raise RuntimeError(f"{name} must be non-negative and finite")
    optimizer = torch.optim.AdamW([gates, up_vectors, bias], lr=tail_lr, weight_decay=1e-7)
    tail_count = max(64, min(coords.shape[0], int(coords.shape[0] * tail_fraction), atom_count * 128))
    best_state: tuple[float, float, float, np.ndarray, np.ndarray, float] | None = None
    for _step in range(steps):
        optimizer.zero_grad()
        prediction = torch.sum(torch.nn.functional.silu(x @ gates.T) * (x @ up_vectors.T), dim=1) + bias
        residual = prediction - y
        mse_loss = torch.mean(residual ** 2)
        tail_values = torch.topk(torch.abs(residual), k=tail_count, largest=True).values
        if tail_loss_kind == "smoothmax":
            tail_loss = torch.logsumexp(tail_values * tail_smoothness, dim=0) / tail_smoothness
        else:
            tail_loss = torch.mean(tail_values ** 2)
        loss = mse_loss + tail_weight * tail_loss
        if anchor_weight > 0.0:
            loss = loss + anchor_weight * torch.mean((prediction - base_prediction) ** 2)
        if drift_weight > 0.0:
            loss = loss + drift_weight * torch.mean(
                torch.relu(torch.abs(residual) - base_abs_residual) ** 2
            )
        if mae_weight > 0.0:
            loss = loss + mae_weight * torch.mean(torch.abs(residual))
        loss.backward()
        torch.nn.utils.clip_grad_norm_([gates, up_vectors, bias], 8.0)
        optimizer.step()
        if _step == 0 or (_step + 1) % 80 == 0:
            with torch.no_grad():
                current_prediction = torch.sum(
                    torch.nn.functional.silu(selection_x @ gates.T) * (selection_x @ up_vectors.T),
                    dim=1,
                ) + bias
                current_abs_error = torch.abs(current_prediction - selection_y)
                worst = float(torch.max(current_abs_error).detach().cpu().item())
                mae_now = float(torch.mean(current_abs_error).detach().cpu().item())
                score = worst + best_mae_weight * mae_now
                if best_state is None or score < best_state[0]:
                    best_state = (
                        score,
                        worst,
                        mae_now,
                        gates.detach().cpu().numpy().astype(np.float64),
                        up_vectors.detach().cpu().numpy().astype(np.float64),
                        float(bias.detach().cpu().item()),
                    )
    if best_state is None:
        return candidate
    best_score, best_worst, best_mae, gate_values, up_values, bias_value = best_state
    activation_np = silu_np(coords @ gate_values.T) * (coords @ up_values.T)
    direct_prediction = activation_np.sum(axis=1) + bias_value
    direct_metrics = regression_metrics(targets, direct_prediction)
    selected_up_values = up_values
    selected_bias_value = bias_value
    prediction_np = direct_prediction
    calibration_mode = "direct"
    if tail_calibration in {"ls", "best-worst"}:
        design = np.column_stack([activation_np, np.ones(coords.shape[0], dtype=np.float64)])
        try:
            coeffs, *_ = np.linalg.lstsq(design, targets, rcond=None)
        except np.linalg.LinAlgError:
            coeffs = None
        if coeffs is not None and np.all(np.isfinite(coeffs)):
            ls_up_values = up_values * coeffs[:-1, None]
            ls_bias_value = float(coeffs[-1])
            ls_prediction = design @ coeffs
            ls_metrics = regression_metrics(targets, ls_prediction)
            use_ls = tail_calibration == "ls" or (
                tail_calibration == "best-worst"
                and (
                    ls_metrics["worstError"] < direct_metrics["worstError"]
                    or (
                        abs(ls_metrics["worstError"] - direct_metrics["worstError"]) <= 1e-12
                        and ls_metrics["mae"] <= direct_metrics["mae"]
                    )
                )
            )
            if use_ls:
                selected_up_values = ls_up_values
                selected_bias_value = ls_bias_value
                prediction_np = ls_prediction
                calibration_mode = "ls"
    mae = float(np.mean(np.abs(prediction_np - targets)))
    mse = float(np.mean((prediction_np - targets) ** 2))
    worst = float(np.max(np.abs(prediction_np - targets)))
    previous_worst = float(candidate.get("worstError", float("inf")))
    if mae >= float(candidate["score"]) and worst >= previous_worst:
        return candidate
    polished = dict(candidate)
    polished.update({
        "mae": mae,
        "mse": mse,
        "score": mae,
        "worstError": worst,
        "gates": gate_values,
        "ups": selected_up_values,
        "scales": np.ones(atom_count, dtype=np.float64),
        "bias": selected_bias_value,
        "tailErrorPolished": True,
        "tailErrorPolishSteps": int(steps),
        "tailErrorPolishWeight": float(tail_weight),
        "tailErrorPolishFraction": float(tail_fraction),
        "tailErrorPolishCount": int(tail_count),
        "tailErrorPolishLearningRate": float(tail_lr),
        "tailErrorPolishLoss": tail_loss_kind,
        "tailErrorPolishSmoothness": float(tail_smoothness),
        "tailErrorPolishCalibration": calibration_mode,
        "tailErrorPolishAnchorWeight": float(anchor_weight),
        "tailErrorPolishDriftWeight": float(drift_weight),
        "tailErrorPolishMaeWeight": float(mae_weight),
        "tailErrorPolishBestMaeWeight": float(best_mae_weight),
        "tailErrorPolishBestObjectiveScore": float(best_score),
        "tailErrorPolishBestObjectiveMae": float(best_mae),
        "tailErrorPolishBestObjectiveWorst": float(best_worst),
        "tailErrorPolishSelectionCount": int(0 if selection_coords is None else selection_coords.shape[0]),
        "tailErrorPolishMaeBefore": float(candidate["mae"]),
    })
    return polished


def final_lbfgs_polish_swiglu_candidate(
    coords: np.ndarray,
    targets: np.ndarray,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    import torch

    if (
        os.environ.get("GCL_BLIND_SWIGLU_FINAL_LBFGS_COURT", "").strip() == "1"
        and not candidate.get("_finalLbfgsCourtActive")
    ):
        schedule_raw = os.environ.get(
            "GCL_BLIND_SWIGLU_FINAL_LBFGS_COURT_SCHEDULE",
            "topk:1.25:0.25:45:best-worst;smoothmax:1.5:0.25:45:best-worst;smoothmax:2.0:0.25:55:best-worst;smoothmax:2.5:0.20:65:best-worst",
        )
        target_worst = float(os.environ.get("GCL_BLIND_SWIGLU_FINAL_LBFGS_COURT_TARGET_WORST", "0.12") or "0.12")
        target_mae = float(os.environ.get("GCL_BLIND_SWIGLU_FINAL_LBFGS_COURT_TARGET_MAE", "0.015") or "0.015")
        court_candidates = [candidate]
        original_env = {
            key: os.environ.get(key)
            for key in [
                "GCL_BLIND_SWIGLU_FINAL_LBFGS_LOSS",
                "GCL_BLIND_SWIGLU_FINAL_LBFGS_TAIL_WEIGHT",
                "GCL_BLIND_SWIGLU_FINAL_LBFGS_TAIL_FRACTION",
                "GCL_BLIND_SWIGLU_FINAL_LBFGS_SMOOTHNESS",
                "GCL_BLIND_SWIGLU_FINAL_LBFGS_CALIBRATION",
                "GCL_BLIND_SWIGLU_FINAL_LBFGS_COURT",
            ]
        }
        try:
            os.environ["GCL_BLIND_SWIGLU_FINAL_LBFGS_COURT"] = "0"
            for index, raw_spec in enumerate(item.strip() for item in schedule_raw.split(";") if item.strip()):
                parts = raw_spec.split(":")
                if len(parts) not in {4, 5}:
                    continue
                loss_kind, weight, fraction, smoothness = parts[:4]
                calibration = parts[4] if len(parts) == 5 else "best-worst"
                os.environ["GCL_BLIND_SWIGLU_FINAL_LBFGS_LOSS"] = loss_kind
                os.environ["GCL_BLIND_SWIGLU_FINAL_LBFGS_TAIL_WEIGHT"] = weight
                os.environ["GCL_BLIND_SWIGLU_FINAL_LBFGS_TAIL_FRACTION"] = fraction
                os.environ["GCL_BLIND_SWIGLU_FINAL_LBFGS_SMOOTHNESS"] = smoothness
                os.environ["GCL_BLIND_SWIGLU_FINAL_LBFGS_CALIBRATION"] = calibration
                court_seed = dict(candidate)
                court_seed["_finalLbfgsCourtActive"] = True
                polished = final_lbfgs_polish_swiglu_candidate(coords, targets, court_seed)
                polished.pop("_finalLbfgsCourtActive", None)
                polished["finalLbfgsCourtSpec"] = raw_spec
                polished["finalLbfgsCourtIndex"] = int(index)
                court_candidates.append(polished)
        finally:
            for key, value in original_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        rescored_candidates = [
            rescore_swiglu_candidate_on_data(coords, targets, item, recalibrate_scales=False)
            for item in court_candidates
        ]

        def court_score(item: dict[str, Any]) -> tuple[float, float, float]:
            mae = float(item.get("mae", float("inf")))
            worst = float(item.get("worstError", float("inf")))
            violation = max(0.0, worst - target_worst) * 10.0 + max(0.0, mae - target_mae) * 3.0
            return (violation, worst, mae)

        selected = min(rescored_candidates, key=court_score)
        selected = dict(selected)
        selected.update({
            "finalLbfgsCourt": True,
            "finalLbfgsCourtCandidateCount": int(len(rescored_candidates)),
            "finalLbfgsCourtTargetWorst": float(target_worst),
            "finalLbfgsCourtTargetMae": float(target_mae),
        })
        return selected

    gates_np = np.asarray(candidate["gates"], dtype=np.float64)
    ups_np = np.asarray(candidate["ups"], dtype=np.float64)
    scales_np = np.asarray(candidate["scales"], dtype=np.float64)
    if gates_np.ndim != 2 or ups_np.shape != gates_np.shape or scales_np.shape[0] != gates_np.shape[0]:
        return candidate
    max_iter = int(os.environ.get("GCL_BLIND_SWIGLU_FINAL_LBFGS_STEPS", "0") or "0")
    if max_iter <= 0:
        return candidate
    tail_weight = float(os.environ.get("GCL_BLIND_SWIGLU_FINAL_LBFGS_TAIL_WEIGHT", "0.0") or "0.0")
    tail_fraction = float(os.environ.get("GCL_BLIND_SWIGLU_FINAL_LBFGS_TAIL_FRACTION", "0.20") or "0.20")
    tail_loss_kind = os.environ.get("GCL_BLIND_SWIGLU_FINAL_LBFGS_LOSS", "topk").strip().lower() or "topk"
    tail_smoothness = float(os.environ.get("GCL_BLIND_SWIGLU_FINAL_LBFGS_SMOOTHNESS", "45.0") or "45.0")
    calibration_mode_request = os.environ.get("GCL_BLIND_SWIGLU_FINAL_LBFGS_CALIBRATION", "ls").strip().lower() or "ls"
    if not np.isfinite(tail_weight) or tail_weight < 0.0:
        raise RuntimeError("GCL_BLIND_SWIGLU_FINAL_LBFGS_TAIL_WEIGHT must be non-negative and finite")
    if not np.isfinite(tail_fraction) or tail_fraction <= 0.0 or tail_fraction > 1.0:
        raise RuntimeError("GCL_BLIND_SWIGLU_FINAL_LBFGS_TAIL_FRACTION must be in (0, 1]")
    if tail_loss_kind not in {"topk", "smoothmax"}:
        raise RuntimeError("GCL_BLIND_SWIGLU_FINAL_LBFGS_LOSS must be topk or smoothmax")
    if not np.isfinite(tail_smoothness) or tail_smoothness <= 0.0:
        raise RuntimeError("GCL_BLIND_SWIGLU_FINAL_LBFGS_SMOOTHNESS must be positive and finite")
    if calibration_mode_request not in {"ls", "direct", "best-worst"}:
        raise RuntimeError("GCL_BLIND_SWIGLU_FINAL_LBFGS_CALIBRATION must be ls, direct, or best-worst")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fit_dtype = torch.float32
    x = torch.tensor(coords, dtype=fit_dtype, device=device)
    y = torch.tensor(targets, dtype=fit_dtype, device=device)
    gates = torch.nn.Parameter(torch.tensor(gates_np, dtype=fit_dtype, device=device))
    up_vectors = torch.nn.Parameter(torch.tensor(ups_np * scales_np[:, None], dtype=fit_dtype, device=device))
    bias = torch.nn.Parameter(torch.tensor(float(candidate["bias"]), dtype=fit_dtype, device=device))
    tail_count = max(64, min(coords.shape[0], int(coords.shape[0] * tail_fraction), gates_np.shape[0] * 192))
    optimizer = torch.optim.LBFGS(
        [gates, up_vectors, bias],
        lr=float(os.environ.get("GCL_BLIND_SWIGLU_FINAL_LBFGS_LR", "0.35") or "0.35"),
        max_iter=max_iter,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-8,
        tolerance_change=1e-10,
        history_size=int(os.environ.get("GCL_BLIND_SWIGLU_FINAL_LBFGS_HISTORY", "80") or "80"),
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        prediction = torch.sum(torch.nn.functional.silu(x @ gates.T) * (x @ up_vectors.T), dim=1) + bias
        residual = prediction - y
        loss = torch.mean(residual ** 2)
        if tail_weight > 0.0:
            tail_values = torch.topk(torch.abs(residual), k=tail_count, largest=True).values
            if tail_loss_kind == "smoothmax":
                tail_loss = torch.logsumexp(tail_values * tail_smoothness, dim=0) / tail_smoothness
            else:
                tail_loss = torch.mean(tail_values ** 2)
            loss = loss + tail_weight * tail_loss
        loss.backward()
        return loss

    try:
        optimizer.step(closure)
    except RuntimeError:
        return candidate

    gate_values = gates.detach().cpu().numpy().astype(np.float64)
    up_values = up_vectors.detach().cpu().numpy().astype(np.float64)
    bias_value = float(bias.detach().cpu().item())
    activation_np = silu_np(coords @ gate_values.T) * (coords @ up_values.T)
    direct_prediction = activation_np.sum(axis=1) + bias_value
    direct_metrics = regression_metrics(targets, direct_prediction)
    selected_up_values = up_values
    selected_bias_value = bias_value
    prediction_np = direct_prediction
    calibration_mode = "direct"

    design = np.column_stack([activation_np, np.ones(coords.shape[0], dtype=np.float64)])
    try:
        coeffs, *_ = np.linalg.lstsq(design, targets, rcond=None)
    except np.linalg.LinAlgError:
        coeffs = None
    if coeffs is not None and np.all(np.isfinite(coeffs)) and calibration_mode_request != "direct":
        ls_up_values = up_values * coeffs[:-1, None]
        ls_bias_value = float(coeffs[-1])
        ls_prediction = design @ coeffs
        ls_metrics = regression_metrics(targets, ls_prediction)
        if (
            calibration_mode_request == "ls"
            and ls_metrics["mae"] < direct_metrics["mae"]
            or (
                calibration_mode_request == "best-worst"
                and (
                    ls_metrics["worstError"] < direct_metrics["worstError"]
                    or (
                        abs(ls_metrics["worstError"] - direct_metrics["worstError"]) <= 1e-12
                        and ls_metrics["mae"] <= direct_metrics["mae"]
                    )
                )
            )
        ):
            selected_up_values = ls_up_values
            selected_bias_value = ls_bias_value
            prediction_np = ls_prediction
            calibration_mode = "ls"

    metrics = regression_metrics(targets, prediction_np)
    previous_mae = float(candidate.get("mae", candidate.get("score", float("inf"))))
    previous_worst = float(candidate.get("worstError", float("inf")))
    if metrics["mae"] >= previous_mae and metrics["worstError"] >= previous_worst:
        return candidate
    polished = dict(candidate)
    polished.update({
        "mae": float(metrics["mae"]),
        "mse": float(metrics["mse"]),
        "score": float(metrics["mae"]),
        "worstError": float(metrics["worstError"]),
        "gates": gate_values,
        "ups": selected_up_values,
        "scales": np.ones(gates_np.shape[0], dtype=np.float64),
        "bias": selected_bias_value,
        "finalLbfgsPolished": True,
        "finalLbfgsSteps": int(max_iter),
        "finalLbfgsTailWeight": float(tail_weight),
        "finalLbfgsTailFraction": float(tail_fraction),
        "finalLbfgsTailCount": int(tail_count),
        "finalLbfgsLoss": tail_loss_kind,
        "finalLbfgsSmoothness": float(tail_smoothness),
        "finalLbfgsCalibration": calibration_mode,
        "finalLbfgsMaeBefore": previous_mae,
        "finalLbfgsWorstBefore": previous_worst,
    })
    return polished


def residual_region_atom_replace_swiglu_candidate(
    coords: np.ndarray,
    targets: np.ndarray,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    cycles = int(os.environ.get("GCL_BLIND_SWIGLU_RESIDUAL_REPLACE_CYCLES", "0") or "0")
    if cycles <= 0:
        return candidate
    import torch

    gates_np = np.asarray(candidate["gates"], dtype=np.float64)
    ups_np = np.asarray(candidate["ups"], dtype=np.float64)
    scales_np = np.asarray(candidate["scales"], dtype=np.float64)
    if gates_np.ndim != 2 or ups_np.shape != gates_np.shape or scales_np.shape[0] != gates_np.shape[0]:
        return candidate
    atom_count, rank = gates_np.shape
    if atom_count < 2 or coords.shape[0] < max(256, atom_count * 32):
        return candidate
    if not torch.cuda.is_available():
        return candidate

    steps = int(os.environ.get("GCL_BLIND_SWIGLU_RESIDUAL_REPLACE_STEPS", "600") or "600")
    tail_fraction = float(os.environ.get("GCL_BLIND_SWIGLU_RESIDUAL_REPLACE_TAIL_FRACTION", "0.08") or "0.08")
    tail_repeat = int(os.environ.get("GCL_BLIND_SWIGLU_RESIDUAL_REPLACE_TAIL_REPEAT", "3") or "3")
    candidate_limit = int(os.environ.get(
        "GCL_BLIND_SWIGLU_RESIDUAL_REPLACE_ATOM_LIMIT",
        str(min(atom_count, 8)),
    ) or "0")
    base_limit = int(os.environ.get("GCL_BLIND_SWIGLU_RESIDUAL_REPLACE_BASE_LIMIT", "8192") or "8192")
    lr = float(os.environ.get("GCL_BLIND_SWIGLU_RESIDUAL_REPLACE_LR", "0.012") or "0.012")
    if (
        steps <= 0
        or not np.isfinite(tail_fraction)
        or tail_fraction <= 0.0
        or tail_fraction > 1.0
        or tail_repeat < 1
        or candidate_limit <= 0
        or base_limit <= 0
        or not np.isfinite(lr)
        or lr <= 0.0
    ):
        raise RuntimeError("invalid GCL_BLIND_SWIGLU_RESIDUAL_REPLACE_* configuration")

    rng = np.random.default_rng(int(os.environ.get("GCL_BLIND_SWIGLU_RESIDUAL_REPLACE_SEED", "71131") or "71131"))
    current = rescore_swiglu_candidate_on_data(coords, targets, candidate, recalibrate_scales=True)
    best = current
    best_prediction = predict_swiglu(coords, best)
    best_residual = targets - best_prediction
    best_mae = float(best["mae"])
    best_worst = float(best.get("worstError", np.max(np.abs(best_residual))))
    replacement_count = 0
    initial_mae = best_mae
    initial_worst = best_worst

    base_indices = np.arange(coords.shape[0])
    if 0 < base_limit < coords.shape[0]:
        base_indices = np.sort(rng.choice(coords.shape[0], size=base_limit, replace=False))

    device = torch.device("cuda")
    fit_dtype = torch.float32

    for _cycle in range(cycles):
        gates = np.asarray(best["gates"], dtype=np.float64)
        ups_scaled = np.asarray(best["ups"], dtype=np.float64) * np.asarray(best["scales"], dtype=np.float64)[:, None]
        bias_value = float(best["bias"])
        activations = silu_np(coords @ gates.T) * (coords @ ups_scaled.T)
        prediction = activations.sum(axis=1) + bias_value
        residual = targets - prediction
        abs_residual = np.abs(residual)
        tail_count = max(64, min(coords.shape[0], int(math.ceil(coords.shape[0] * tail_fraction)), atom_count * 256))
        tail_indices = np.argpartition(abs_residual, -tail_count)[-tail_count:]
        atom_scores = []
        for atom_index in range(atom_count):
            contribution = activations[:, atom_index]
            score = float(np.mean(np.abs(residual[tail_indices] + contribution[tail_indices])))
            atom_scores.append((score, atom_index))
        atom_scores.sort(reverse=True)
        accepted_in_cycle = False

        for _score, atom_index in atom_scores[:min(candidate_limit, atom_count)]:
            target_without_others = targets - bias_value - (activations.sum(axis=1) - activations[:, atom_index])
            aug_indices = np.concatenate([base_indices] + [tail_indices for _ in range(tail_repeat)])
            x_aug = torch.tensor(coords[aug_indices], dtype=fit_dtype, device=device)
            y_aug = torch.tensor(target_without_others[aug_indices], dtype=fit_dtype, device=device)
            gate = torch.nn.Parameter(torch.tensor(gates[atom_index], dtype=fit_dtype, device=device))
            up = torch.nn.Parameter(torch.tensor(ups_scaled[atom_index], dtype=fit_dtype, device=device))
            optimizer = torch.optim.AdamW([gate, up], lr=lr, weight_decay=1e-7)
            for _step in range(steps):
                optimizer.zero_grad()
                pred = torch.nn.functional.silu(x_aug @ gate) * (x_aug @ up)
                err = pred - y_aug
                loss = torch.mean(err ** 2)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([gate, up], 8.0)
                optimizer.step()

            new_gates = gates.copy()
            new_ups_scaled = ups_scaled.copy()
            new_gates[atom_index] = gate.detach().cpu().numpy().astype(np.float64)
            new_ups_scaled[atom_index] = up.detach().cpu().numpy().astype(np.float64)
            activation_np = silu_np(coords @ new_gates.T) * (coords @ new_ups_scaled.T)
            design = np.column_stack([activation_np, np.ones(coords.shape[0], dtype=np.float64)])
            try:
                coeffs, *_ = np.linalg.lstsq(design, targets, rcond=None)
            except np.linalg.LinAlgError:
                continue
            if not np.all(np.isfinite(coeffs)):
                continue
            scaled_ups = new_ups_scaled * coeffs[:-1, None]
            trial = dict(best)
            trial.update({
                "gates": new_gates,
                "ups": scaled_ups,
                "scales": np.ones(atom_count, dtype=np.float64),
                "bias": float(coeffs[-1]),
            })
            trial_prediction = predict_swiglu(coords, trial)
            trial_metrics = regression_metrics(targets, trial_prediction)
            trial_mae = float(trial_metrics["mae"])
            trial_worst = float(trial_metrics["worstError"])
            improves_tail = trial_worst < best_worst and trial_mae <= best_mae * 1.035
            improves_mae = trial_mae < best_mae and trial_worst <= best_worst * 1.015
            if not (improves_tail or improves_mae):
                continue
            trial.update({
                "mae": trial_mae,
                "mse": float(trial_metrics["mse"]),
                "score": trial_mae,
                "selectionScore": trial_mae,
                "worstError": trial_worst,
            })
            best = trial
            best_prediction = trial_prediction
            best_residual = targets - best_prediction
            best_mae = trial_mae
            best_worst = trial_worst
            replacement_count += 1
            accepted_in_cycle = True
            break
        if not accepted_in_cycle:
            break

    if replacement_count == 0:
        return candidate
    refined = dict(best)
    refined.update({
        "residualRegionAtomReplaced": True,
        "residualRegionReplacementCycles": int(cycles),
        "residualRegionReplacementCount": int(replacement_count),
        "residualRegionReplacementSteps": int(steps),
        "residualRegionReplacementTailFraction": float(tail_fraction),
        "residualRegionReplacementTailRepeat": int(tail_repeat),
        "residualRegionReplacementMaeBefore": float(initial_mae),
        "residualRegionReplacementWorstBefore": float(initial_worst),
    })
    return refined


def detect_collapsed_swiglu_atoms(
    coords: np.ndarray,
    candidate: dict[str, Any],
    *,
    gate_cosine_threshold: float,
    up_cosine_threshold: float,
    contribution_correlation_threshold: float,
) -> list[dict[str, Any]]:
    """Find atoms that have converged onto one shared direction.

    `detect_swiglu_cancellation_pairs` looks for the opposite sign of the same geometry: two
    aligned atoms whose contributions *cancel*. A collapsed pair is aligned and correlated,
    so it slipped through that filter entirely. Measured on the randomized LLaMA layer-15
    block, atoms 12 and 14 sat at gate cosine 0.62 to 0.70 against both of two source units,
    and the principal cosines between their span and the sources' span were 0.9907 and
    0.02265: one direction covered twice, the second not represented at all.

    The surface fit hides this. R2 stayed at 0.9986 because whatever the missing direction
    contributed was absorbed by the other fourteen atoms, so nothing in the loss objects to
    spending two atoms on one direction.
    """
    gates = np.asarray(candidate["gates"], dtype=np.float64)
    ups = np.asarray(candidate["ups"], dtype=np.float64)
    scales = np.asarray(candidate["scales"], dtype=np.float64)
    points = np.asarray(coords, dtype=np.float64)
    if (
        gates.ndim != 2
        or ups.shape != gates.shape
        or scales.shape != (gates.shape[0],)
        or points.ndim != 2
        or points.shape[1] != gates.shape[1]
        or points.shape[0] < 2
    ):
        return []
    effective_ups = ups * scales[:, None]
    gate_norms = np.linalg.norm(gates, axis=1)
    up_norms = np.linalg.norm(effective_ups, axis=1)
    valid = (gate_norms > 1e-12) & (up_norms > 1e-12)
    gate_directions = gates / np.maximum(gate_norms[:, None], 1e-12)
    up_directions = effective_ups / np.maximum(up_norms[:, None], 1e-12)
    contributions = silu_np(points @ gates.T) * (points @ effective_ups.T)
    centered = contributions - np.mean(contributions, axis=0, keepdims=True)
    contribution_norms = np.linalg.norm(centered, axis=0)
    mean_abs = np.mean(np.abs(contributions), axis=0)

    pairs: list[dict[str, Any]] = []
    for left, right in combinations(range(gates.shape[0]), 2):
        if not valid[left] or not valid[right]:
            continue
        gate_cosine = abs(float(np.dot(gate_directions[left], gate_directions[right])))
        up_cosine = abs(float(np.dot(up_directions[left], up_directions[right])))
        denominator = float(contribution_norms[left] * contribution_norms[right])
        if denominator <= 1e-15:
            continue
        correlation = float(np.dot(centered[:, left], centered[:, right]) / denominator)
        # The magnitude, not the sign. Rescaling the atoms flips it: on a synthetic collapsed
        # pair, least-squares recalibration turned scales of 1.0 into -5.49 and 10.51 and the
        # contribution correlation from +0.22 into -0.999, while the geometry - gate cosine
        # 1.000, up cosine 1.000 - did not move. A sign test would have found the same pair
        # before recalibration and lost it after.
        if (
            gate_cosine < gate_cosine_threshold
            or up_cosine < up_cosine_threshold
            or abs(correlation) < contribution_correlation_threshold
        ):
            continue
        # The weaker of the two is the one to re-seed: it is contributing less, so moving it
        # risks less of the fit that already works.
        weaker = left if mean_abs[left] <= mean_abs[right] else right
        pairs.append({
            "atoms": [int(left), int(right)],
            "gateCosine": gate_cosine,
            "upCosine": up_cosine,
            "contributionCorrelation": correlation,
            "absContributionCorrelation": abs(correlation),
            "weakerAtom": int(weaker),
            "meanAbsContribution": [float(mean_abs[left]), float(mean_abs[right])],
        })
    pairs.sort(key=lambda item: -float(item["gateCosine"]) * float(item["upCosine"]))
    return pairs


def respread_collapsed_atoms_swiglu_candidate(
    coords: np.ndarray,
    targets: np.ndarray,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    """Re-seed one atom of a collapsed pair against the residual the pair cannot explain.

    Refitting a collapsed atom from where it already sits is what produced the collapse: the
    pair is in a local minimum that the surface loss is content with. So each attempt starts
    the atom somewhere else - directions drawn from the residual's own principal axes, plus
    random restarts - fits it against the target the other atoms leave behind, and is kept
    only if the full candidate improves on the same acceptance rule the residual-replace path
    uses. Nothing is weakened when no start helps: the original candidate is returned.
    """
    if os.environ.get("GCL_BLIND_SWIGLU_RESPREAD_COLLAPSED_ATOMS", "").strip() != "1":
        return candidate
    import torch

    gates_np = np.asarray(candidate["gates"], dtype=np.float64)
    ups_np = np.asarray(candidate["ups"], dtype=np.float64)
    scales_np = np.asarray(candidate["scales"], dtype=np.float64)
    if gates_np.ndim != 2 or ups_np.shape != gates_np.shape or scales_np.shape[0] != gates_np.shape[0]:
        return candidate
    atom_count, rank = gates_np.shape
    if atom_count < 2 or not torch.cuda.is_available():
        return candidate

    gate_threshold = float(os.environ.get("GCL_BLIND_SWIGLU_RESPREAD_GATE_COSINE", "0.55") or "0.55")
    up_threshold = float(os.environ.get("GCL_BLIND_SWIGLU_RESPREAD_UP_COSINE", "0.55") or "0.55")
    correlation_threshold = float(os.environ.get("GCL_BLIND_SWIGLU_RESPREAD_CORRELATION", "0.30") or "0.30")
    starts = int(os.environ.get("GCL_BLIND_SWIGLU_RESPREAD_STARTS", "24") or "24")
    steps = int(os.environ.get("GCL_BLIND_SWIGLU_RESPREAD_STEPS", "900") or "900")
    lr = float(os.environ.get("GCL_BLIND_SWIGLU_RESPREAD_LR", "0.02") or "0.02")
    max_pairs = int(os.environ.get("GCL_BLIND_SWIGLU_RESPREAD_MAX_PAIRS", "4") or "4")
    seed = int(os.environ.get("GCL_BLIND_SWIGLU_RESPREAD_SEED", "73109") or "73109")
    if starts <= 0 or steps <= 0 or not np.isfinite(lr) or lr <= 0.0 or max_pairs <= 0:
        raise RuntimeError("invalid GCL_BLIND_SWIGLU_RESPREAD_* configuration")

    best = rescore_swiglu_candidate_on_data(coords, targets, candidate, recalibrate_scales=True)
    best_mae = float(best["mae"])
    best_worst = float(best.get("worstError", float("inf")))
    initial_mae, initial_worst = best_mae, best_worst
    pairs_before = detect_collapsed_swiglu_atoms(
        coords,
        best,
        gate_cosine_threshold=gate_threshold,
        up_cosine_threshold=up_threshold,
        contribution_correlation_threshold=correlation_threshold,
    )
    if not pairs_before:
        # Returning the candidate untouched leaves no way to tell "no collapsed pair here"
        # from "this pass never ran", and two live runs came back bit-identical with nothing
        # in the artifact to say which. The closest pair is recorded whether or not it clears
        # the thresholds.
        closest = detect_collapsed_swiglu_atoms(
            coords,
            best,
            gate_cosine_threshold=0.0,
            up_cosine_threshold=0.0,
            contribution_correlation_threshold=0.0,
        )
        observed = dict(candidate)
        observed.update({
            "collapsedAtomsRespread": False,
            "collapsedPairsBefore": 0,
            "collapsedAtomRespreadThresholds": {
                "gateCosine": gate_threshold,
                "upCosine": up_threshold,
                "absContributionCorrelation": correlation_threshold,
            },
            "collapsedClosestPair": closest[0] if closest else None,
        })
        return observed

    rng = np.random.default_rng(seed)
    device = torch.device("cuda")
    fit_dtype = torch.float32
    x = torch.tensor(coords, dtype=fit_dtype, device=device)
    respread_atoms: list[int] = []

    for pair in pairs_before[:max_pairs]:
        atom_index = int(pair["weakerAtom"])
        gates = np.asarray(best["gates"], dtype=np.float64)
        ups_scaled = np.asarray(best["ups"], dtype=np.float64) * np.asarray(best["scales"], dtype=np.float64)[:, None]
        activations = silu_np(coords @ gates.T) * (coords @ ups_scaled.T)

        # Refit the remaining atoms with this one removed before taking the residual. The
        # scales the collapsed candidate carries were least-squares fitted around the collapse
        # - on the synthetic pair, 1.0 became -5.49 and 10.51 - so subtracting the other atoms
        # at those scales leaves a residual shaped by the very defect being repaired. Measured:
        # without this the best start lands back on the direction already covered.
        keep = [index for index in range(atom_count) if index != atom_index]
        design_without = np.column_stack([activations[:, keep], np.ones(coords.shape[0], dtype=np.float64)])
        try:
            coeffs_without, *_ = np.linalg.lstsq(design_without, targets, rcond=None)
        except np.linalg.LinAlgError:
            continue
        if not np.all(np.isfinite(coeffs_without)):
            continue
        for position, index in enumerate(keep):
            ups_scaled[index] = ups_scaled[index] * float(coeffs_without[position])
        bias_value = float(coeffs_without[-1])
        activations = silu_np(coords @ gates.T) * (coords @ ups_scaled.T)
        target_without_others = targets - bias_value - (activations[:, keep].sum(axis=1))

        # Directions the residual itself points at: principal axes of the inputs weighted by
        # how much of the target this atom is being asked to explain.
        weights = np.abs(target_without_others)
        weight_sum = float(np.sum(weights))
        seeds: list[np.ndarray] = []
        if weight_sum > 1e-12:
            weighted = coords * (weights / weight_sum)[:, None]
            covariance = weighted.T @ coords
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            for column in range(min(rank, 4)):
                axis = eigenvectors[:, -1 - column]
                seeds.append(axis / max(float(np.linalg.norm(axis)), 1e-12))
        while len(seeds) < starts:
            draw = rng.normal(0.0, 1.0, size=rank)
            seeds.append(draw / max(float(np.linalg.norm(draw)), 1e-12))

        y = torch.tensor(target_without_others, dtype=fit_dtype, device=device)
        gate_init = np.stack(seeds[:starts])
        # The up direction is drawn independently. Deriving it from the gate seeds - the
        # obvious way to reuse the residual axes - ties the two halves of every start
        # together and searches a slice of the space instead of the space: measured on a
        # synthetic collapsed pair, that reached MSE 0.66 where independent draws reach 1e-6.
        up_init = rng.normal(0.0, 1.0, size=(starts, rank))
        up_init /= np.maximum(np.linalg.norm(up_init, axis=1, keepdims=True), 1e-12)
        gate = torch.nn.Parameter(torch.tensor(gate_init, dtype=fit_dtype, device=device))
        up = torch.nn.Parameter(torch.tensor(up_init, dtype=fit_dtype, device=device))
        optimizer = torch.optim.AdamW([gate, up], lr=lr, weight_decay=1e-7)
        for _step in range(steps):
            optimizer.zero_grad()
            prediction = torch.nn.functional.silu(x @ gate.T) * (x @ up.T)
            loss = torch.mean((prediction - y[:, None]) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([gate, up], 8.0)
            optimizer.step()

        gate_out = gate.detach().cpu().numpy().astype(np.float64)
        up_out = up.detach().cpu().numpy().astype(np.float64)
        # Rank the starts before testing acceptance. Taking the first start that clears the
        # bar leaves the search unused: on the synthetic pair that reached gate cosine 0.78
        # where the best of the same 24 starts reaches 1.0.
        start_order = sorted(
            range(starts),
            key=lambda index: float(np.mean((
                silu_np(coords @ gate_out[index]) * (coords @ up_out[index]) - target_without_others
            ) ** 2))
        )
        accepted = False
        for start_index in start_order:
            new_gates = gates.copy()
            new_ups_scaled = ups_scaled.copy()
            new_gates[atom_index] = gate_out[start_index]
            new_ups_scaled[atom_index] = up_out[start_index]
            activation_np = silu_np(coords @ new_gates.T) * (coords @ new_ups_scaled.T)
            design = np.column_stack([activation_np, np.ones(coords.shape[0], dtype=np.float64)])
            try:
                coeffs, *_ = np.linalg.lstsq(design, targets, rcond=None)
            except np.linalg.LinAlgError:
                continue
            if not np.all(np.isfinite(coeffs)):
                continue
            trial = dict(best)
            trial.update({
                "gates": new_gates,
                "ups": new_ups_scaled * coeffs[:-1, None],
                "scales": np.ones(atom_count, dtype=np.float64),
                "bias": float(coeffs[-1]),
            })
            metrics = regression_metrics(targets, predict_swiglu(coords, trial))
            trial_mae = float(metrics["mae"])
            trial_worst = float(metrics["worstError"])
            improves_tail = trial_worst < best_worst and trial_mae <= best_mae * 1.035
            improves_mae = trial_mae < best_mae and trial_worst <= best_worst * 1.015
            if not (improves_tail or improves_mae):
                continue
            trial.update({
                "mae": trial_mae,
                "mse": float(metrics["mse"]),
                "score": trial_mae,
                "selectionScore": trial_mae,
                "worstError": trial_worst,
            })
            best, best_mae, best_worst = trial, trial_mae, trial_worst
            accepted = True
            break
        if accepted:
            respread_atoms.append(atom_index)

    if not respread_atoms:
        return candidate
    pairs_after = detect_collapsed_swiglu_atoms(
        coords,
        best,
        gate_cosine_threshold=gate_threshold,
        up_cosine_threshold=up_threshold,
        contribution_correlation_threshold=correlation_threshold,
    )
    refined = dict(best)
    refined.update({
        "collapsedAtomsRespread": True,
        "collapsedAtomRespreadAtoms": respread_atoms,
        "collapsedPairsBefore": len(pairs_before),
        "collapsedPairsAfter": len(pairs_after),
        "collapsedAtomRespreadStarts": int(starts),
        "collapsedAtomRespreadMaeBefore": float(initial_mae),
        "collapsedAtomRespreadWorstBefore": float(initial_worst),
    })
    return refined


def split_merge_tail_atom_refine_swiglu_candidate(
    coords: np.ndarray,
    targets: np.ndarray,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    if os.environ.get("GCL_BLIND_SWIGLU_SPLIT_MERGE_REFINEMENT", "").strip() != "1":
        return candidate
    gates_np = np.asarray(candidate["gates"], dtype=np.float64)
    ups_np = np.asarray(candidate["ups"], dtype=np.float64)
    scales_np = np.asarray(candidate["scales"], dtype=np.float64)
    if gates_np.ndim != 2 or ups_np.shape != gates_np.shape or scales_np.shape[0] != gates_np.shape[0]:
        return candidate
    atom_count, rank = gates_np.shape
    if atom_count < 4:
        return candidate
    rng = np.random.default_rng(int(os.environ.get("GCL_BLIND_SWIGLU_SPLIT_MERGE_SEED", "81283") or "81283"))
    trial_limit = int(os.environ.get("GCL_BLIND_SWIGLU_SPLIT_MERGE_TRIALS", "48") or "48")
    tail_fraction = float(os.environ.get("GCL_BLIND_SWIGLU_SPLIT_MERGE_TAIL_FRACTION", "0.08") or "0.08")
    if trial_limit <= 0 or not np.isfinite(tail_fraction) or tail_fraction <= 0.0 or tail_fraction > 1.0:
        raise RuntimeError("invalid GCL_BLIND_SWIGLU_SPLIT_MERGE_* configuration")

    base = rescore_swiglu_candidate_on_data(coords, targets, candidate, recalibrate_scales=True)
    base_prediction = predict_swiglu(coords, base)
    base_metrics = regression_metrics(targets, base_prediction)
    gates_np = np.asarray(base["gates"], dtype=np.float64)
    ups_scaled = np.asarray(base["ups"], dtype=np.float64) * np.asarray(base["scales"], dtype=np.float64)[:, None]
    bias_value = float(base["bias"])
    activations = silu_np(coords @ gates_np.T) * (coords @ ups_scaled.T)
    residual = targets - base_prediction
    tail_count = max(64, min(coords.shape[0], int(math.ceil(coords.shape[0] * tail_fraction)), atom_count * 256))
    tail_indices = np.argpartition(np.abs(residual), -tail_count)[-tail_count:]
    atom_scores = []
    for atom_index in range(atom_count):
        atom_scores.append((
            float(np.mean(np.abs(residual[tail_indices] + activations[tail_indices, atom_index]))),
            atom_index,
        ))
    atom_scores.sort(reverse=True)
    candidate_atoms = [atom_index for _score, atom_index in atom_scores[:min(atom_count, 6)]]
    eps_values = [0.035, 0.07, 0.12, 0.18]
    best = base
    best_metrics = base_metrics

    for trial_index in range(trial_limit):
        atom_index = candidate_atoms[trial_index % len(candidate_atoms)]
        eps = eps_values[(trial_index // len(candidate_atoms)) % len(eps_values)]
        direction = rng.normal(size=rank)
        gate = gates_np[atom_index]
        gate_norm = max(1e-9, float(np.linalg.norm(gate)))
        gate_dir = gate / gate_norm
        direction -= float(np.dot(direction, gate_dir)) * gate_dir
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm < 1e-9:
            continue
        direction /= direction_norm
        split_gates = np.vstack([
            gates_np[:atom_index],
            gate + eps * gate_norm * direction,
            gate - eps * gate_norm * direction,
            gates_np[atom_index + 1:],
        ])
        split_ups = np.vstack([
            ups_scaled[:atom_index],
            0.5 * ups_scaled[atom_index],
            0.5 * ups_scaled[atom_index],
            ups_scaled[atom_index + 1:],
        ])
        split_features = silu_np(coords @ split_gates.T) * (coords @ split_ups.T)
        split_design = np.column_stack([split_features, np.ones(coords.shape[0], dtype=np.float64)])
        try:
            split_coeffs, *_ = np.linalg.lstsq(split_design, targets, rcond=None)
        except np.linalg.LinAlgError:
            continue
        if not np.all(np.isfinite(split_coeffs)):
            continue
        atom_energy = np.std(split_features, axis=0) * np.abs(split_coeffs[:-1])
        keep = np.sort(np.argsort(atom_energy)[-atom_count:])
        kept_gates = split_gates[keep]
        kept_ups = split_ups[keep]
        kept_features = silu_np(coords @ kept_gates.T) * (coords @ kept_ups.T)
        kept_design = np.column_stack([kept_features, np.ones(coords.shape[0], dtype=np.float64)])
        try:
            kept_coeffs, *_ = np.linalg.lstsq(kept_design, targets, rcond=None)
        except np.linalg.LinAlgError:
            continue
        if not np.all(np.isfinite(kept_coeffs)):
            continue
        trial_fit = dict(base)
        trial_fit.update({
            "gates": kept_gates,
            "ups": kept_ups * kept_coeffs[:-1, None],
            "scales": np.ones(atom_count, dtype=np.float64),
            "bias": float(kept_coeffs[-1]),
        })
        trial_prediction = predict_swiglu(coords, trial_fit)
        trial_metrics = regression_metrics(targets, trial_prediction)
        if (
            trial_metrics["worstError"] < best_metrics["worstError"] and trial_metrics["mae"] <= best_metrics["mae"] * 1.04
        ) or (
            trial_metrics["mae"] < best_metrics["mae"] and trial_metrics["worstError"] <= best_metrics["worstError"] * 1.01
        ):
            best = trial_fit
            best_metrics = trial_metrics

    if best is base:
        return candidate
    refined = dict(best)
    refined.update({
        "mae": float(best_metrics["mae"]),
        "mse": float(best_metrics["mse"]),
        "score": float(best_metrics["mae"]),
        "selectionScore": float(best_metrics["mae"]),
        "worstError": float(best_metrics["worstError"]),
        "splitMergeTailAtomRefined": True,
        "splitMergeTailAtomTrials": int(trial_limit),
        "splitMergeTailAtomMaeBefore": float(base_metrics["mae"]),
        "splitMergeTailAtomWorstBefore": float(base_metrics["worstError"]),
    })
    return refined


def selective_atom_scale_repair_swiglu_candidate(
    coords: np.ndarray,
    targets: np.ndarray,
    candidate: dict[str, Any],
    atom_indices: list[int],
) -> dict[str, Any]:
    if not atom_indices:
        return candidate
    gates = np.asarray(candidate["gates"], dtype=np.float64)
    ups = np.asarray(candidate["ups"], dtype=np.float64)
    scales = np.asarray(candidate["scales"], dtype=np.float64)
    if gates.ndim != 2 or ups.shape != gates.shape or scales.shape[0] != gates.shape[0]:
        return candidate
    atom_count = gates.shape[0]
    selected = sorted({int(index) for index in atom_indices if 0 <= int(index) < atom_count})
    if not selected:
        return candidate
    max_delta = float(os.environ.get("GCL_BLIND_SWIGLU_ATOM_SCALE_REPAIR_MAX_DELTA", "0.35") or "0.35")
    tail_fraction = float(os.environ.get("GCL_BLIND_SWIGLU_ATOM_SCALE_REPAIR_TAIL_FRACTION", "0.08") or "0.08")
    tail_weight = float(os.environ.get("GCL_BLIND_SWIGLU_ATOM_SCALE_REPAIR_TAIL_WEIGHT", "2.0") or "2.0")
    worst_weight = float(os.environ.get("GCL_BLIND_SWIGLU_ATOM_SCALE_REPAIR_WORST_WEIGHT", "0.25") or "0.25")
    if not np.isfinite(max_delta) or max_delta <= 0.0:
        raise RuntimeError("GCL_BLIND_SWIGLU_ATOM_SCALE_REPAIR_MAX_DELTA must be positive and finite")
    if not np.isfinite(tail_fraction) or tail_fraction <= 0.0 or tail_fraction > 1.0:
        raise RuntimeError("GCL_BLIND_SWIGLU_ATOM_SCALE_REPAIR_TAIL_FRACTION must be in (0, 1]")
    for name, value in [
        ("GCL_BLIND_SWIGLU_ATOM_SCALE_REPAIR_TAIL_WEIGHT", tail_weight),
        ("GCL_BLIND_SWIGLU_ATOM_SCALE_REPAIR_WORST_WEIGHT", worst_weight),
    ]:
        if not np.isfinite(value) or value < 0.0:
            raise RuntimeError(f"{name} must be non-negative and finite")
    base_prediction = predict_swiglu(coords, candidate)
    base_metrics = regression_metrics(targets, base_prediction)
    contribution = silu_np(coords @ gates.T) * (coords @ (ups * scales[:, None]).T)
    base_abs_error = np.abs(base_prediction - targets)
    tail_count = max(1, min(coords.shape[0], int(math.ceil(coords.shape[0] * tail_fraction))))
    tail_indices = np.argpartition(base_abs_error, -tail_count)[-tail_count:]

    def repair_score(prediction: np.ndarray) -> float:
        abs_error = np.abs(prediction - targets)
        return (
            float(np.mean(abs_error))
            + tail_weight * float(np.mean(abs_error[tail_indices]))
            + worst_weight * float(np.max(abs_error))
        )

    base_score = repair_score(base_prediction)

    def make_repaired_candidate(deltas: np.ndarray, bias_delta: float) -> tuple[dict[str, Any], dict[str, float], float]:
        repaired_ups = ups.copy()
        for local_index, atom_index in enumerate(selected):
            repaired_ups[atom_index] = repaired_ups[atom_index] * (1.0 + float(deltas[local_index]))
        repaired = dict(candidate)
        repaired.update({
            "ups": repaired_ups,
            "bias": float(candidate["bias"]) + float(bias_delta),
        })
        repaired_prediction = predict_swiglu(coords, repaired)
        repaired_metrics = regression_metrics(targets, repaired_prediction)
        return repaired, repaired_metrics, repair_score(repaired_prediction)

    trial_candidates: list[tuple[float, dict[str, Any], dict[str, float], np.ndarray, float, str]] = []
    design = np.column_stack([
        contribution[:, selected],
        np.ones(coords.shape[0], dtype=np.float64),
    ])
    residual = targets - base_prediction
    try:
        solved, *_ = np.linalg.lstsq(design, residual, rcond=None)
    except np.linalg.LinAlgError:
        return candidate
    if not np.all(np.isfinite(solved)):
        return candidate
    deltas = np.clip(solved[:-1].astype(np.float64), -max_delta, max_delta)
    repaired, repaired_metrics, score = make_repaired_candidate(deltas, float(solved[-1]))
    trial_candidates.append((score, repaired, repaired_metrics, deltas, float(solved[-1]), "joint-ls"))
    tail_residual = residual[tail_indices]
    for local_index, atom_index in enumerate(selected):
        atom_tail = contribution[tail_indices, atom_index]
        denom = float(np.dot(atom_tail, atom_tail))
        if denom <= 1e-12:
            continue
        atom_delta = float(np.dot(tail_residual, atom_tail) / denom)
        atom_deltas = np.zeros(len(selected), dtype=np.float64)
        atom_deltas[local_index] = float(np.clip(atom_delta, -max_delta, max_delta))
        atom_repaired, atom_metrics, atom_score = make_repaired_candidate(atom_deltas, 0.0)
        trial_candidates.append((atom_score, atom_repaired, atom_metrics, atom_deltas, 0.0, f"tail-atom-{atom_index}"))
    if not trial_candidates:
        return candidate
    score, repaired, repaired_metrics, deltas, bias_delta, mode = min(trial_candidates, key=lambda item: item[0])
    if (
        score >= base_score
        and repaired_metrics["mae"] >= base_metrics["mae"]
        and repaired_metrics["worstError"] >= base_metrics["worstError"]
    ):
        return candidate
    repaired.update({
        "mae": float(repaired_metrics["mae"]),
        "mse": float(repaired_metrics["mse"]),
        "score": float(repaired_metrics["mae"]),
        "selectionScore": float(repaired_metrics["mae"]),
        "worstError": float(repaired_metrics["worstError"]),
        "selectiveAtomScaleRepaired": True,
        "selectiveAtomScaleRepairAtoms": selected,
        "selectiveAtomScaleRepairDeltas": [float(value) for value in deltas],
        "selectiveAtomScaleRepairBiasDelta": float(bias_delta),
        "selectiveAtomScaleRepairMode": mode,
        "selectiveAtomScaleRepairScore": float(score),
        "selectiveAtomScaleRepairScoreBefore": float(base_score),
        "selectiveAtomScaleRepairTailCount": int(tail_count),
        "selectiveAtomScaleRepairTailWeight": float(tail_weight),
        "selectiveAtomScaleRepairWorstWeight": float(worst_weight),
        "selectiveAtomScaleRepairMaeBefore": float(base_metrics["mae"]),
        "selectiveAtomScaleRepairWorstBefore": float(base_metrics["worstError"]),
    })
    return repaired


def directional_atom_repair_swiglu_candidate(
    coords: np.ndarray,
    targets: np.ndarray,
    candidate: dict[str, Any],
    atom_indices: list[int],
    selection_coords: np.ndarray | None = None,
    selection_targets: np.ndarray | None = None,
) -> dict[str, Any]:
    if not atom_indices:
        return candidate
    import torch

    gates_np = np.asarray(candidate["gates"], dtype=np.float64)
    ups_np = np.asarray(candidate["ups"], dtype=np.float64)
    scales_np = np.asarray(candidate["scales"], dtype=np.float64)
    if gates_np.ndim != 2 or ups_np.shape != gates_np.shape or scales_np.shape[0] != gates_np.shape[0]:
        return candidate
    atom_count = gates_np.shape[0]
    selected = sorted({int(index) for index in atom_indices if 0 <= int(index) < atom_count})
    if not selected or not torch.cuda.is_available():
        return candidate
    steps = int(os.environ.get("GCL_BLIND_SWIGLU_ATOM_DIRECTION_REPAIR_STEPS", "420") or "420")
    lr = float(os.environ.get("GCL_BLIND_SWIGLU_ATOM_DIRECTION_REPAIR_LR", "0.006") or "0.006")
    tail_weight = float(os.environ.get("GCL_BLIND_SWIGLU_ATOM_DIRECTION_REPAIR_TAIL_WEIGHT", "0.75") or "0.75")
    tail_fraction = float(os.environ.get("GCL_BLIND_SWIGLU_ATOM_DIRECTION_REPAIR_TAIL_FRACTION", "0.20") or "0.20")
    anchor_weight = float(os.environ.get("GCL_BLIND_SWIGLU_ATOM_DIRECTION_REPAIR_ANCHOR_WEIGHT", "0.65") or "0.65")
    drift_weight = float(os.environ.get("GCL_BLIND_SWIGLU_ATOM_DIRECTION_REPAIR_DRIFT_WEIGHT", "0.45") or "0.45")
    mae_weight = float(os.environ.get("GCL_BLIND_SWIGLU_ATOM_DIRECTION_REPAIR_MAE_WEIGHT", "0.20") or "0.20")
    best_mae_weight = float(os.environ.get("GCL_BLIND_SWIGLU_ATOM_DIRECTION_REPAIR_BEST_MAE_WEIGHT", "2.0") or "2.0")
    if steps <= 0 or not np.isfinite(lr) or lr <= 0.0:
        raise RuntimeError("invalid GCL_BLIND_SWIGLU_ATOM_DIRECTION_REPAIR_STEPS/LR")
    if not np.isfinite(tail_weight) or tail_weight < 0.0:
        raise RuntimeError("GCL_BLIND_SWIGLU_ATOM_DIRECTION_REPAIR_TAIL_WEIGHT must be non-negative")
    if not np.isfinite(tail_fraction) or tail_fraction <= 0.0 or tail_fraction > 1.0:
        raise RuntimeError("GCL_BLIND_SWIGLU_ATOM_DIRECTION_REPAIR_TAIL_FRACTION must be in (0, 1]")
    for name, value in [
        ("GCL_BLIND_SWIGLU_ATOM_DIRECTION_REPAIR_ANCHOR_WEIGHT", anchor_weight),
        ("GCL_BLIND_SWIGLU_ATOM_DIRECTION_REPAIR_DRIFT_WEIGHT", drift_weight),
        ("GCL_BLIND_SWIGLU_ATOM_DIRECTION_REPAIR_MAE_WEIGHT", mae_weight),
        ("GCL_BLIND_SWIGLU_ATOM_DIRECTION_REPAIR_BEST_MAE_WEIGHT", best_mae_weight),
    ]:
        if not np.isfinite(value) or value < 0.0:
            raise RuntimeError(f"{name} must be non-negative")

    base = rescore_swiglu_candidate_on_data(coords, targets, candidate, recalibrate_scales=True)
    gates_np = np.asarray(base["gates"], dtype=np.float64)
    ups_scaled = np.asarray(base["ups"], dtype=np.float64) * np.asarray(base["scales"], dtype=np.float64)[:, None]
    base_prediction = predict_swiglu(coords, base)
    base_metrics = regression_metrics(targets, base_prediction)
    base_residual = base_prediction - targets
    selected_array = np.array(selected, dtype=np.int64)
    fixed_contribution = silu_np(coords @ gates_np.T) * (coords @ ups_scaled.T)
    original_selected = np.sum(fixed_contribution[:, selected_array], axis=1)
    fixed_base = (
        float(base["bias"])
        + np.sum(fixed_contribution, axis=1)
        - original_selected
    )
    has_selection = (
        selection_coords is not None
        and selection_targets is not None
        and selection_coords.shape[0] > 0
        and selection_coords.shape[0] == selection_targets.shape[0]
        and selection_coords.shape[1] == coords.shape[1]
    )
    if has_selection and selection_coords is not None and selection_targets is not None:
        selection_contribution = silu_np(selection_coords @ gates_np.T) * (selection_coords @ ups_scaled.T)
        selection_original = np.sum(selection_contribution[:, selected_array], axis=1)
        selection_fixed_base = (
            float(base["bias"])
            + np.sum(selection_contribution, axis=1)
            - selection_original
        )
    else:
        selection_fixed_base = None

    device = torch.device("cuda")
    fit_dtype = torch.float32
    x = torch.tensor(coords, dtype=fit_dtype, device=device)
    y = torch.tensor(targets, dtype=fit_dtype, device=device)
    fixed = torch.tensor(fixed_base, dtype=fit_dtype, device=device)
    original_selected_tensor = torch.tensor(original_selected, dtype=fit_dtype, device=device)
    base_abs_residual = torch.tensor(np.abs(base_residual), dtype=fit_dtype, device=device)
    if has_selection and selection_coords is not None and selection_targets is not None and selection_fixed_base is not None:
        selection_x = torch.tensor(selection_coords, dtype=fit_dtype, device=device)
        selection_y = torch.tensor(selection_targets, dtype=fit_dtype, device=device)
        selection_fixed = torch.tensor(selection_fixed_base, dtype=fit_dtype, device=device)
    else:
        selection_x = None
        selection_y = None
        selection_fixed = None
    gates = torch.nn.Parameter(torch.tensor(gates_np[selected_array], dtype=fit_dtype, device=device))
    ups = torch.nn.Parameter(torch.tensor(ups_scaled[selected_array], dtype=fit_dtype, device=device))
    optimizer = torch.optim.AdamW([gates, ups], lr=lr, weight_decay=1e-7)
    tail_count = max(32, min(coords.shape[0], int(coords.shape[0] * tail_fraction), len(selected) * 256))
    best_state: tuple[float, float, float, np.ndarray, np.ndarray] | None = None
    for step in range(steps):
        optimizer.zero_grad()
        selected_prediction = torch.sum(torch.nn.functional.silu(x @ gates.T) * (x @ ups.T), dim=1)
        prediction = fixed + selected_prediction
        residual = prediction - y
        loss = torch.mean(residual ** 2)
        if anchor_weight > 0.0:
            loss = loss + anchor_weight * torch.mean((selected_prediction - original_selected_tensor) ** 2)
        if drift_weight > 0.0:
            loss = loss + drift_weight * torch.mean(
                torch.relu(torch.abs(residual) - base_abs_residual) ** 2
            )
        if mae_weight > 0.0:
            loss = loss + mae_weight * torch.mean(torch.abs(residual))
        if tail_weight > 0.0:
            tail_values = torch.topk(torch.abs(residual), k=tail_count, largest=True).values
            loss = loss + tail_weight * torch.mean(tail_values ** 2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([gates, ups], 8.0)
        optimizer.step()
        if step == 0 or (step + 1) % 80 == 0 or step + 1 == steps:
            with torch.no_grad():
                if selection_x is not None and selection_y is not None and selection_fixed is not None:
                    selected_prediction = torch.sum(torch.nn.functional.silu(selection_x @ gates.T) * (selection_x @ ups.T), dim=1)
                    prediction = selection_fixed + selected_prediction
                    target_values = selection_y
                else:
                    selected_prediction = torch.sum(torch.nn.functional.silu(x @ gates.T) * (x @ ups.T), dim=1)
                    prediction = fixed + selected_prediction
                    target_values = y
                abs_error = torch.abs(prediction - target_values)
                worst = float(torch.max(abs_error).detach().cpu().item())
                mae = float(torch.mean(abs_error).detach().cpu().item())
                score = worst + best_mae_weight * mae
                if best_state is None or score < best_state[0]:
                    best_state = (
                        float(score),
                        worst,
                        mae,
                        gates.detach().cpu().numpy().astype(np.float64),
                        ups.detach().cpu().numpy().astype(np.float64),
                    )
    if best_state is None:
        return candidate
    best_score, best_worst, best_mae, repaired_gates, repaired_ups = best_state
    all_gates = gates_np.copy()
    all_ups = ups_scaled.copy()
    all_gates[selected_array] = repaired_gates
    all_ups[selected_array] = repaired_ups
    if has_selection and selection_coords is not None and selection_targets is not None:
        calibration_coords = np.vstack([coords, selection_coords])
        calibration_targets = np.concatenate([targets, selection_targets])
    else:
        calibration_coords = coords
        calibration_targets = targets
    features = silu_np(calibration_coords @ all_gates.T) * (calibration_coords @ all_ups.T)
    design = np.column_stack([features, np.ones(calibration_coords.shape[0], dtype=np.float64)])
    try:
        coeffs, *_ = np.linalg.lstsq(design, calibration_targets, rcond=None)
    except np.linalg.LinAlgError:
        return candidate
    if not np.all(np.isfinite(coeffs)):
        return candidate
    repaired = dict(base)
    repaired.update({
        "gates": all_gates,
        "ups": all_ups * coeffs[:-1, None],
        "scales": np.ones(atom_count, dtype=np.float64),
        "bias": float(coeffs[-1]),
    })
    repaired_prediction = predict_swiglu(coords, repaired)
    repaired_metrics = regression_metrics(targets, repaired_prediction)
    if (
        repaired_metrics["mae"] >= base_metrics["mae"]
        and repaired_metrics["worstError"] >= base_metrics["worstError"]
    ):
        return candidate
    repaired.update({
        "mae": float(repaired_metrics["mae"]),
        "mse": float(repaired_metrics["mse"]),
        "score": float(repaired_metrics["mae"]),
        "selectionScore": float(repaired_metrics["mae"]),
        "worstError": float(repaired_metrics["worstError"]),
        "directionalAtomRepaired": True,
        "directionalAtomRepairAtoms": selected,
        "directionalAtomRepairSteps": int(steps),
        "directionalAtomRepairTailWeight": float(tail_weight),
        "directionalAtomRepairAnchorWeight": float(anchor_weight),
        "directionalAtomRepairDriftWeight": float(drift_weight),
        "directionalAtomRepairMaeWeight": float(mae_weight),
        "directionalAtomRepairBestMaeWeight": float(best_mae_weight),
        "directionalAtomRepairBestObjectiveScore": float(best_score),
        "directionalAtomRepairBestObjectiveMae": float(best_mae),
        "directionalAtomRepairBestObjectiveWorst": float(best_worst),
        "directionalAtomRepairSelectionCount": int(selection_coords.shape[0]) if has_selection and selection_coords is not None else 0,
        "directionalAtomRepairMaeBefore": float(base_metrics["mae"]),
        "directionalAtomRepairWorstBefore": float(base_metrics["worstError"]),
    })
    return repaired


def canonical_direction(vector: np.ndarray) -> np.ndarray | None:
    direction = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-9 or not np.all(np.isfinite(direction)):
        return None
    direction = direction / norm
    pivot = int(np.argmax(np.abs(direction)))
    if direction[pivot] < 0.0:
        direction = -direction
    return direction


def append_diverse_direction(
    directions: list[np.ndarray],
    candidate: np.ndarray,
    max_abs_cosine: float,
) -> None:
    direction = canonical_direction(candidate)
    if direction is None:
        return
    if any(abs(float(np.dot(direction, existing))) >= max_abs_cosine for existing in directions):
        return
    directions.append(direction)


def make_gradient_gate_directions(
    gradient_coords: np.ndarray | None,
    gradient_targets: np.ndarray | None,
    atom_count: int,
    seed: int,
    max_candidates: int,
) -> list[np.ndarray]:
    if gradient_targets is None or gradient_targets.shape[0] < max(4, atom_count):
        return []
    rng = np.random.default_rng(seed)
    gradients = np.asarray(gradient_targets, dtype=np.float64)
    norms = np.linalg.norm(gradients, axis=1)
    valid = np.where(np.isfinite(norms) & (norms > 1e-9))[0]
    if valid.size == 0:
        return []
    directions: list[np.ndarray] = []

    if (
        gradient_coords is not None
        and gradient_coords.shape[0] == gradients.shape[0]
        and gradient_coords.shape[1] == gradients.shape[1]
        and gradient_coords.shape[0] >= max(gradients.shape[1] + 2, atom_count)
    ):
        fit_coords = np.asarray(gradient_coords, dtype=np.float64)
        valid_coords = fit_coords[valid]
        valid_gradients = gradients[valid]
        centered_coords = valid_coords - np.mean(valid_coords, axis=0, keepdims=True)
        centered_gradients = valid_gradients - np.mean(valid_gradients, axis=0, keepdims=True)
        try:
            # Least-squares linearization of the oracle gradient field:
            # grad f(x) ~= H x + b. For SwiGLU/GELU-like channels, high-energy
            # eigendirections of the symmetric Hessian estimate expose gate
            # and up subspace atoms better than raw gradient vectors alone.
            coeffs, *_ = np.linalg.lstsq(
                np.column_stack([centered_coords, np.ones(centered_coords.shape[0], dtype=np.float64)]),
                centered_gradients,
                rcond=None,
            )
            linear_map = coeffs[:-1].T.astype(np.float64)
            sym_hessian = 0.5 * (linear_map + linear_map.T)
            values, vectors = np.linalg.eigh(sym_hessian)
            for index in np.argsort(np.abs(values))[::-1][:min(vectors.shape[1], atom_count * 3)]:
                append_diverse_direction(directions, vectors[:, index], 0.995)
                append_diverse_direction(directions, -vectors[:, index], 0.995)
            try:
                left, _singular_values, right_t = np.linalg.svd(linear_map, full_matrices=False)
                for row in left.T[:min(left.shape[1], atom_count * 2)]:
                    append_diverse_direction(directions, row, 0.995)
                for row in right_t[:min(right_t.shape[0], atom_count * 2)]:
                    append_diverse_direction(directions, row, 0.995)
            except np.linalg.LinAlgError:
                pass
            row_norms = np.linalg.norm(linear_map, axis=1)
            col_norms = np.linalg.norm(linear_map, axis=0)
            for index in np.argsort(row_norms)[::-1][:min(linear_map.shape[0], atom_count * 2)]:
                append_diverse_direction(directions, linear_map[index], 0.995)
            for index in np.argsort(col_norms)[::-1][:min(linear_map.shape[1], atom_count * 2)]:
                append_diverse_direction(directions, linear_map[:, index], 0.995)
        except np.linalg.LinAlgError:
            pass

        bag_count = int(os.environ.get(
            "GCL_BLIND_SWIGLU_GRADIENT_FIELD_BAGS",
            "24" if atom_count >= 16 else "8",
        ) or "0")
        if bag_count > 0:
            sample_count = min(valid_coords.shape[0], max(atom_count * 8, valid_coords.shape[1] + 4))
            for bag_index in range(bag_count):
                if valid_coords.shape[0] <= sample_count:
                    bag_indices = np.arange(valid_coords.shape[0])
                else:
                    bag_indices = rng.choice(valid_coords.shape[0], size=sample_count, replace=False)
                bag_coords = valid_coords[bag_indices] - np.mean(valid_coords[bag_indices], axis=0, keepdims=True)
                bag_gradients = valid_gradients[bag_indices] - np.mean(valid_gradients[bag_indices], axis=0, keepdims=True)
                try:
                    bag_coeffs, *_ = np.linalg.lstsq(
                        np.column_stack([bag_coords, np.ones(bag_coords.shape[0], dtype=np.float64)]),
                        bag_gradients,
                        rcond=None,
                    )
                    bag_map = bag_coeffs[:-1].T.astype(np.float64)
                    bag_sym = 0.5 * (bag_map + bag_map.T)
                    bag_values, bag_vectors = np.linalg.eigh(bag_sym)
                    for index in np.argsort(np.abs(bag_values))[::-1][:min(4, bag_vectors.shape[1])]:
                        append_diverse_direction(directions, bag_vectors[:, index], 0.995)
                except np.linalg.LinAlgError:
                    continue
                if len(directions) >= max_candidates:
                    return directions[:max_candidates]

    centered = gradients[valid] - np.mean(gradients[valid], axis=0, keepdims=True)
    try:
        _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
        for row in vh[:min(vh.shape[0], atom_count * 2)]:
            append_diverse_direction(directions, row, 0.995)
            append_diverse_direction(directions, -row, 0.995)
    except np.linalg.LinAlgError:
        pass

    for index in valid[np.argsort(norms[valid])[::-1]]:
        append_diverse_direction(directions, gradients[index], 0.995)
        if len(directions) >= max_candidates:
            return directions[:max_candidates]

    pair_count = min(max_candidates * 8, max(0, valid.size * 3))
    for _ in range(pair_count):
        left, right = rng.choice(valid, size=2, replace=False)
        append_diverse_direction(directions, gradients[left] - gradients[right], 0.995)
        append_diverse_direction(directions, gradients[left] + gradients[right], 0.995)
        if len(directions) >= max_candidates:
            break
    return directions[:max_candidates]


def solve_fixed_gate_start(
    coords: np.ndarray,
    targets: np.ndarray,
    gates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float] | None:
    if gates.shape[0] == 0:
        return None
    blocks = [silu_np(coords @ gate)[:, None] * coords for gate in gates]
    design = np.column_stack(blocks + [np.ones(coords.shape[0], dtype=np.float64)])
    try:
        coeffs, *_ = np.linalg.lstsq(design, targets, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(coeffs)):
        return None
    rank = coords.shape[1]
    raw_up_values = coeffs[:-1].reshape(gates.shape[0], rank).astype(np.float64)
    up_values = np.zeros_like(raw_up_values)
    scales = np.zeros(gates.shape[0], dtype=np.float64)
    for index, raw_up in enumerate(raw_up_values):
        up_norm = float(np.linalg.norm(raw_up))
        if up_norm > 1e-12:
            up_values[index] = raw_up / up_norm
            scales[index] = up_norm
    bias = float(coeffs[-1])
    prediction = design @ coeffs
    mae = float(np.mean(np.abs(prediction - targets)))
    return gates.astype(np.float64), up_values, scales, bias, mae


def make_gradient_structured_starts(
    coords: np.ndarray,
    targets: np.ndarray,
    gradient_coords: np.ndarray | None,
    gradient_targets: np.ndarray | None,
    gate_seed_directions: np.ndarray | None,
    atom_count: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
    if atom_count < 4:
        return []
    rank = coords.shape[1]
    if rank <= 0:
        return []
    rng = np.random.default_rng(seed + 331)
    directions: list[np.ndarray] = []

    # Geometry-cover starts are deliberately oracle-only: they use only active
    # coordinates, finite-difference witnesses, and oracle-scored residuals.
    basis = np.eye(rank, dtype=np.float64)
    for row in basis:
        append_diverse_direction(directions, row, 0.998)
    for i in range(rank):
        for j in range(i + 1, rank):
            append_diverse_direction(directions, basis[i] + basis[j], 0.998)
            append_diverse_direction(directions, basis[i] - basis[j], 0.998)

    for _ in range(max(atom_count * 6, rank * 8, 64)):
        append_diverse_direction(directions, rng.normal(size=rank), 0.998)
        append_diverse_direction(directions, rng.choice([-1.0, 1.0], size=rank), 0.998)
    for _ in range(max(2, atom_count // 2)):
        random_matrix = rng.normal(size=(rank, rank))
        try:
            q, _r = np.linalg.qr(random_matrix)
            for row in q.T:
                append_diverse_direction(directions, row, 0.998)
        except np.linalg.LinAlgError:
            pass

    centered_coords = coords - np.mean(coords, axis=0, keepdims=True)
    try:
        _u, _s, vh = np.linalg.svd(centered_coords, full_matrices=False)
        for row in vh[:rank]:
            append_diverse_direction(directions, row, 0.998)
    except np.linalg.LinAlgError:
        pass

    target_centered = targets.astype(np.float64) - float(np.mean(targets))
    try:
        linear_coeffs, *_ = np.linalg.lstsq(coords, target_centered, rcond=None)
        append_diverse_direction(directions, linear_coeffs, 0.998)
    except np.linalg.LinAlgError:
        pass

    directions = make_gradient_gate_directions(
        gradient_coords,
        gradient_targets,
        atom_count,
        seed,
        max_candidates=max(atom_count * 12, 96),
    ) + directions
    if gate_seed_directions is not None and gate_seed_directions.shape[0] > 0:
        for row in gate_seed_directions:
            append_diverse_direction(directions, row, 0.995)
    if len(directions) < atom_count:
        return []
    sample_count = min(coords.shape[0], max(4096, atom_count * 384))
    if sample_count < coords.shape[0]:
        sample_indices = np.sort(rng.choice(coords.shape[0], size=sample_count, replace=False))
        xs = coords[sample_indices]
        ys = targets[sample_indices]
    else:
        xs = coords
        ys = targets
    gate_norms = np.array([0.18, 0.28, 0.42, 0.65, 0.95, 1.35, 1.9, 2.7], dtype=np.float64)
    scored: list[tuple[float, np.ndarray]] = []
    ones = np.ones(xs.shape[0], dtype=np.float64)
    for direction in directions:
        for norm_value in gate_norms:
            gate = direction * norm_value
            block = silu_np(xs @ gate)[:, None] * xs
            design = np.column_stack([block, ones])
            try:
                coeffs, *_ = np.linalg.lstsq(design, ys, rcond=None)
            except np.linalg.LinAlgError:
                continue
            if not np.all(np.isfinite(coeffs)):
                continue
            predicted = design @ coeffs
            scored.append((float(np.mean(np.abs(predicted - ys))), gate))
    if len(scored) < atom_count:
        return []
    scored.sort(key=lambda item: item[0])

    def solve_selected(selected_gates: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, float] | None:
        if len(selected_gates) != atom_count:
            return None
        solved_start = solve_fixed_gate_start(coords, targets, np.vstack(selected_gates))
        if solved_start is None:
            return None
        gates, ups, scales, bias, _mae = solved_start
        return gates, ups, scales, bias

    def greedy_residual_selected(
        candidate_pool: list[np.ndarray],
        max_abs_cosine: float,
        pool_limit: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float] | None:
        pool = candidate_pool[:min(len(candidate_pool), pool_limit)]
        if len(pool) < atom_count:
            return None
        selected_gates: list[np.ndarray] = []
        selected_indices: set[int] = set()
        residual = ys.astype(np.float64) - float(np.mean(ys))
        solved: np.ndarray | None = None
        for _atom_index in range(atom_count):
            best_choice: tuple[float, int] | None = None
            for index, gate in enumerate(pool):
                if index in selected_indices:
                    continue
                direction = canonical_direction(gate)
                if direction is None:
                    continue
                if any(abs(float(np.dot(direction, existing / max(1e-9, np.linalg.norm(existing))))) >= max_abs_cosine for existing in selected_gates):
                    continue
                block = silu_np(xs @ gate)[:, None] * xs
                try:
                    coeffs, *_ = np.linalg.lstsq(block, residual, rcond=None)
                except np.linalg.LinAlgError:
                    continue
                if not np.all(np.isfinite(coeffs)):
                    continue
                projection = block @ coeffs
                improvement = float(np.dot(residual, residual) - np.dot(residual - projection, residual - projection))
                if best_choice is None or improvement > best_choice[0]:
                    best_choice = (improvement, index)
            if best_choice is None or best_choice[0] <= 1e-12:
                return None
            selected_indices.add(best_choice[1])
            selected_gates.append(pool[best_choice[1]])
            blocks = [silu_np(xs @ gate)[:, None] * xs for gate in selected_gates]
            design = np.column_stack(blocks + [np.ones(xs.shape[0], dtype=np.float64)])
            try:
                solved, *_ = np.linalg.lstsq(design, ys, rcond=None)
            except np.linalg.LinAlgError:
                return None
            if not np.all(np.isfinite(solved)):
                return None
            residual = ys - design @ solved
        return solve_selected(selected_gates)

    selected: list[np.ndarray] = []
    for _score, gate in scored:
        direction = canonical_direction(gate)
        if direction is None:
            continue
        if any(abs(float(np.dot(direction, existing / max(1e-9, np.linalg.norm(existing))))) >= 0.992 for existing in selected):
            continue
        selected.append(gate)
        if len(selected) >= atom_count:
            break
    if len(selected) < atom_count:
        return []

    starts: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []
    scored_pool = [gate for _score, gate in scored]
    for max_abs_cosine, pool_limit in ((0.999, 192), (0.995, 256), (0.985, 320), (0.965, 384)):
        greedy_start = greedy_residual_selected(scored_pool, max_abs_cosine, pool_limit)
        if greedy_start is not None:
            starts.append(greedy_start)

    for norm_value in (0.42, 0.65, 0.95, 1.35):
        for _cover_index in range(4):
            try:
                q, _r = np.linalg.qr(rng.normal(size=(rank, rank)))
            except np.linalg.LinAlgError:
                continue
            gates_cover = q.T[:atom_count] * norm_value
            solved_cover = solve_selected([gate for gate in gates_cover])
            if solved_cover is not None:
                starts.append(solved_cover)
            if rank >= atom_count * 2:
                mixed = []
                for atom_index in range(atom_count):
                    direction = q[:, atom_index] + q[:, atom_index + atom_count]
                    mixed.append(normalize(direction) * norm_value)
                solved_mixed = solve_selected(mixed)
                if solved_mixed is not None:
                    starts.append(solved_mixed)

    solved = solve_selected(selected)
    if solved is not None:
        starts.append(solved)

    for scale in (0.55, 0.75, 1.15, 1.45):
        scaled = np.vstack([gate * scale for gate in selected])
        solved_scaled = solve_selected([gate for gate in scaled])
        if solved_scaled is not None:
            starts.append(solved_scaled)

    axes_pool = [gate for _score, gate in scored[:min(len(scored), max(atom_count * 12, 96))]]
    for stride in (1, 2, 3, 5, 7):
        chosen: list[np.ndarray] = []
        for index in range(stride - 1, len(axes_pool), stride):
            gate = axes_pool[index]
            direction = canonical_direction(gate)
            if direction is None:
                continue
            if any(abs(float(np.dot(direction, existing / max(1e-9, np.linalg.norm(existing))))) >= 0.985 for existing in chosen):
                continue
            chosen.append(gate)
            if len(chosen) >= atom_count:
                break
        solved_stride = solve_selected(chosen)
        if solved_stride is not None:
            starts.append(solved_stride)

    if len(scored) >= atom_count * 2:
        pool = [gate for _score, gate in scored[:min(len(scored), atom_count * 8)]]
        for _restart_index in range(8):
            chosen: list[np.ndarray] = []
            for gate in rng.permutation(pool):
                direction = canonical_direction(gate)
                if direction is None:
                    continue
                if any(abs(float(np.dot(direction, existing / max(1e-9, np.linalg.norm(existing))))) >= 0.99 for existing in chosen):
                    continue
                chosen.append(gate)
                if len(chosen) >= atom_count:
                    break
            solved_random = solve_selected(chosen)
            if solved_random is not None:
                starts.append(solved_random)

    unique_starts: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []
    signatures: set[tuple[float, ...]] = set()
    for gates, ups, scales, bias in starts:
        gate_dirs = []
        for gate in gates:
            direction = canonical_direction(gate)
            if direction is not None:
                gate_dirs.extend(np.round(direction, 3).tolist())
        signature = tuple(gate_dirs[:min(len(gate_dirs), atom_count * rank)])
        if signature in signatures:
            continue
        signatures.add(signature)
        unique_starts.append((gates, ups, scales, bias))
    return unique_starts[:18]


def make_oracle_gate_candidate_pool(
    gradient_coords: np.ndarray | None,
    gradient_targets: np.ndarray | None,
    gate_seed_directions: np.ndarray | None,
    atom_count: int,
    seed: int,
) -> list[np.ndarray]:
    if atom_count < 4:
        return []
    raw_norms = os.environ.get("GCL_BLIND_SWIGLU_EXTRA_GATE_NORMS")
    gate_norms = parse_init_scale_schedule(raw_norms, 0.65) if raw_norms else [0.18, 0.28, 0.42, 0.65, 0.95, 1.35, 1.9, 2.7]
    gate_norms = [float(value) for value in gate_norms if np.isfinite(value) and value > 0.0]
    if not gate_norms:
        return []
    max_directions = int(os.environ.get(
        "GCL_BLIND_SWIGLU_EXTRA_GATE_DIRECTION_LIMIT",
        str(max(atom_count * 48, 768) if atom_count >= 16 else max(atom_count * 24, 256)),
    ) or "0")
    if max_directions <= 0:
        return []
    directions: list[np.ndarray] = []
    for direction in make_gradient_gate_directions(
        gradient_coords,
        gradient_targets,
        atom_count,
        seed + 6073,
        max_candidates=max_directions,
    ):
        append_diverse_direction(directions, direction, 0.9995)
        if len(directions) >= max_directions:
            break
    if gate_seed_directions is not None and gate_seed_directions.shape[0] > 0:
        for row in gate_seed_directions:
            append_diverse_direction(directions, row, 0.9995)
            append_diverse_direction(directions, -row, 0.9995)
            if len(directions) >= max_directions:
                break
    candidates: list[np.ndarray] = []
    for direction in directions[:max_directions]:
        canonical = canonical_direction(direction)
        if canonical is None:
            continue
        for norm_value in gate_norms:
            candidates.append(canonical * norm_value)
    return candidates


def expand_structured_start_variants(
    coords: np.ndarray,
    targets: np.ndarray,
    starts: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]],
    atom_count: int,
    seed: int,
    max_starts: int,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
    if not starts or max_starts <= 0:
        return starts
    rng = np.random.default_rng(seed)
    scored: list[tuple[float, np.ndarray, np.ndarray, np.ndarray, float]] = []
    signatures: set[tuple[float, ...]] = set()

    def add_gates(raw_gates: np.ndarray) -> None:
        gates = np.asarray(raw_gates, dtype=np.float64)
        if gates.shape != (atom_count, coords.shape[1]) or not np.all(np.isfinite(gates)):
            return
        normalized_rows = []
        for gate in gates:
            direction = canonical_direction(gate)
            if direction is None:
                return
            normalized_rows.extend(np.round(direction, 3).tolist())
        signature = tuple(normalized_rows)
        if signature in signatures:
            return
        signatures.add(signature)
        solved = solve_fixed_gate_start(coords, targets, gates)
        if solved is None:
            return
        solved_gates, solved_ups, solved_scales, solved_bias, mae = solved
        scored.append((mae, solved_gates, solved_ups, solved_scales, solved_bias))

    for gates, _ups, _scales, _bias in starts:
        gates = np.asarray(gates, dtype=np.float64)
        add_gates(gates)
        for scale in (0.72, 0.88, 1.12, 1.35):
            add_gates(gates * scale)
        gate_norms = np.linalg.norm(gates, axis=1, keepdims=True).clip(min=1e-9)
        gate_dirs = gates / gate_norms
        for jitter in (0.025, 0.055, 0.09, 0.14):
            noise = rng.normal(size=gates.shape)
            noise -= np.sum(noise * gate_dirs, axis=1, keepdims=True) * gate_dirs
            noise_norm = np.linalg.norm(noise, axis=1, keepdims=True).clip(min=1e-9)
            perturbed = gate_dirs + jitter * (noise / noise_norm)
            perturbed /= np.linalg.norm(perturbed, axis=1, keepdims=True).clip(min=1e-9)
            add_gates(perturbed * gate_norms)
        for _ in range(2):
            shuffled = gates.copy()
            rng.shuffle(shuffled, axis=0)
            add_gates(shuffled)

    if not scored:
        return starts
    scored.sort(key=lambda item: item[0])
    return [(gates, ups, scales, bias) for _mae, gates, ups, scales, bias in scored[:max_starts]]


def fit_swiglu_sum(
    coords: np.ndarray,
    targets: np.ndarray,
    atom_count: int,
    seed: int,
    steps: int,
    learning_rate: float,
    restart_override: int | None = None,
    gradient_coords: np.ndarray | None = None,
    gradient_targets: np.ndarray | None = None,
    gate_seed_directions: np.ndarray | None = None,
    gradient_weight: float = 0.0,
    fit_mode: str = "auto",
    restart_chunk_size: int = 0,
    selection_coords: np.ndarray | None = None,
    selection_targets: np.ndarray | None = None,
    warm_starts: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]] | None = None,
    gate_seed_scores: np.ndarray | None = None,
    cubic_selection_directions: np.ndarray | None = None,
    cubic_selection_scores: np.ndarray | None = None,
    cubic_directions: np.ndarray | None = None,
    cubic_targets: np.ndarray | None = None,
    cubic_weight_override: float | None = None,
    selection_cubic_weight_override: float | None = None,
) -> dict[str, Any]:
    if atom_count == 1:
        single = fit_swiglu_atom(coords, targets, seed, steps, learning_rate)
        return {
            "mae": single["mae"],
            "restart": single["restart"],
            "gates": np.expand_dims(single["gate"], axis=0),
            "ups": np.expand_dims(single["up"], axis=0),
            "scales": np.array([single["scale"]], dtype=np.float64),
            "bias": single["bias"],
            "restartCount": int(single.get("restartCount", 1)),
            "restartChunkSize": None,
            "fitMode": "single-atom",
        }

    import torch

    torch.manual_seed(seed)
    fit_dtype = torch.float64 if atom_count >= 4 else torch.float32
    # Every tensor here used to be created without a device, so the whole optimisation - steps
    # times restarts, 2600 x 512 in the high-atom profile - ran through libtorch_cpu. Measured on
    # a live 32-unit run: 31.13 cores busy in the solver, zero disk traffic, and no compute rows
    # for that pid in `nvidia-smi pmon` while the card held the oracle and idled.
    #
    # `GCL_BLIND_SWIGLU_FIT_DEVICE` forces the choice, because the card is not always the right
    # answer: a fit of four atoms over six hundred points is a stream of tiny kernels where the
    # launch overhead outweighs the arithmetic, and the card can also be busy with something
    # else entirely. The chosen device is reported in the fit result, so which processor did the
    # work is a recorded fact rather than an assumption about a flag.
    requested_device = os.environ.get("GCL_BLIND_SWIGLU_FIT_DEVICE", "").strip().lower()
    if requested_device in {"cpu", "cuda"}:
        fit_device = torch.device(requested_device)
    else:
        fit_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.tensor(coords, dtype=fit_dtype, device=fit_device)
    y = torch.tensor(targets, dtype=fit_dtype, device=fit_device)
    has_gradient_witness = (
        gradient_coords is not None
        and gradient_targets is not None
        and gradient_weight > 0.0
        and gradient_coords.shape[0] > 0
    )
    if has_gradient_witness:
        gx = torch.tensor(gradient_coords, dtype=fit_dtype, device=fit_device)
        gy = torch.tensor(gradient_targets, dtype=fit_dtype, device=fit_device)
    else:
        gx = None
        gy = None
    cubic_weight = float(
        cubic_weight_override
        if cubic_weight_override is not None
        else os.environ.get("GCL_BLIND_SWIGLU_FIT_CUBIC_WEIGHT", "0.0") or "0.0"
    )
    if not np.isfinite(cubic_weight) or cubic_weight < 0.0:
        raise RuntimeError("SwiGLU cubic fit weight must be non-negative and finite")
    effective_cubic_directions = (
        cubic_directions
        if cubic_directions is not None
        else gate_seed_directions
    )
    effective_cubic_targets = (
        cubic_targets
        if cubic_targets is not None
        else gate_seed_scores
    )
    has_cubic_witness = (
        effective_cubic_directions is not None
        and effective_cubic_targets is not None
        and cubic_weight > 0.0
        and effective_cubic_directions.shape[0] > 0
        and effective_cubic_directions.shape[0] == effective_cubic_targets.shape[0]
    )
    if has_cubic_witness:
        cv = torch.tensor(effective_cubic_directions, dtype=fit_dtype, device=fit_device)
        cy = torch.tensor(effective_cubic_targets, dtype=fit_dtype, device=fit_device)
        target_cubic_scale = torch.mean(cy ** 2).detach().clamp_min(1e-12)
    else:
        cv = None
        cy = None
        target_cubic_scale = None
    rank = coords.shape[1]
    target_scale = float(np.std(targets)) or 1.0
    target_variance = float(np.mean((targets - np.mean(targets)) ** 2))
    restart_count = max(64, atom_count * 32)
    if atom_count >= 4:
        restart_count = max(restart_count, atom_count * 48)
    if restart_override is not None:
        restart_count = max(1, restart_override)
    best: dict[str, Any] | None = None
    structured_starts: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []
    if warm_starts:
        for gates_start, ups_start, scales_start, bias_start in warm_starts:
            gates_start = np.asarray(gates_start, dtype=np.float64)
            ups_start = np.asarray(ups_start, dtype=np.float64)
            scales_start = np.asarray(scales_start, dtype=np.float64)
            if (
                gates_start.shape == (atom_count, rank)
                and ups_start.shape == (atom_count, rank)
                and scales_start.shape == (atom_count,)
                and np.all(np.isfinite(gates_start))
                and np.all(np.isfinite(ups_start))
                and np.all(np.isfinite(scales_start))
                and np.isfinite(float(bias_start))
            ):
                structured_starts.append((
                    gates_start.copy(),
                    ups_start.copy(),
                    scales_start.copy(),
                    float(bias_start),
                ))
    if atom_count > 1:
        residual = targets.astype(np.float64).copy()
        residual_bias = float(np.mean(residual))
        residual -= residual_bias
        residual_gates: list[np.ndarray] = []
        residual_ups: list[np.ndarray] = []
        residual_scales: list[float] = []
        residual_steps = max(240, min(700, steps // 4))
        try:
            for atom_index in range(atom_count):
                # One `fit_swiglu_atom` an atom, each one a full restart sweep, and at 32 atoms
                # this greedy pass is the whole gap between `train-sampled` and the batched fit
                # entering. It had no marker, so the gap looked like a single unexplained stall.
                emit_progress("fit-greedy-atom", atom=int(atom_index), of=int(atom_count))
                single = fit_swiglu_atom(
                    coords,
                    residual,
                    seed + 1009 + atom_index * 131,
                    residual_steps,
                    learning_rate,
                )
                component = predict_swiglu(coords, single)
                residual_bias += float(single["bias"])
                residual_gates.append(single["gate"])
                residual_ups.append(single["up"])
                residual_scales.append(float(single["scale"]))
                residual -= component
            structured_starts.append((
                np.vstack(residual_gates),
                np.vstack(residual_ups),
                np.array(residual_scales, dtype=np.float64),
                residual_bias,
            ))
        except (ValueError, KeyError, IndexError) as exc:
            # A malformed single-atom candidate or an empty residual stack. Anything else here
            # is a defect in the construction above and must not be turned into "no starts".
            _record_fallback(f"structured starts abandoned: {type(exc).__name__}: {exc}")
            structured_starts = []

    # Four more construction steps stand between the greedy pass and the batched fit, and none of
    # them said anything. Measured on a live 32-unit run, the greedy pass ends at 138.6 s and the
    # next marker had not arrived 340 s later - a second unnamed stall immediately after the
    # first one was named. Each step reports now, so the next run says which.
    emit_progress("starts-greedy-done", starts=len(structured_starts))
    gradient_starts = make_gradient_structured_starts(
        coords,
        targets,
        gradient_coords,
        gradient_targets,
        gate_seed_directions,
        atom_count,
        seed + 2029,
    )
    emit_progress("starts-gradient-done", added=len(gradient_starts))
    quadratic_multi_starts = make_quadratic_multi_structured_starts(
        coords,
        targets,
        atom_count,
        seed + 3037,
    )
    emit_progress("starts-quadratic-multi-done", added=len(quadratic_multi_starts))
    structured_starts.extend(quadratic_multi_starts)
    structured_starts.extend(gradient_starts)
    if structured_starts and atom_count >= 4:
        structured_starts = expand_structured_start_variants(
            coords,
            targets,
            structured_starts,
            atom_count,
            seed + 4051,
            max_starts=max(12, min(24, atom_count + 2)),
        )
        emit_progress("starts-expanded", starts=len(structured_starts))
    extra_gate_candidates = make_oracle_gate_candidate_pool(
        gradient_coords,
        gradient_targets,
        gate_seed_directions,
        atom_count,
        seed + 5087,
    )
    emit_progress("starts-oracle-pool-done", candidates=len(extra_gate_candidates))

    if fit_mode in {"auto", "batched"} and atom_count >= 4:
        try:
            return fit_swiglu_sum_batched(
                coords,
                targets,
                atom_count,
                seed,
                steps,
                learning_rate,
                restart_count,
                structured_starts,
                gradient_coords,
                gradient_targets,
                gradient_weight,
                restart_chunk_size,
                selection_coords,
                selection_targets,
                extra_gate_candidates,
                effective_cubic_directions,
                effective_cubic_targets,
                cubic_selection_directions,
                cubic_selection_scores,
                cubic_weight,
                selection_cubic_weight_override,
            )
        except Exception:
            if fit_mode == "batched":
                raise

    structured_start_count = len(structured_starts)
    total_restart_count = restart_count + structured_start_count
    best_score_by_start_kind: dict[str, float] = {}
    best_metrics_by_start_kind: dict[str, dict[str, Any]] = {}
    for restart in range(total_restart_count):
        torch.manual_seed(seed + restart)
        if restart < structured_start_count:
            gate_init, up_init, scale_init, bias_init = structured_starts[restart]
            gates = torch.nn.Parameter(torch.tensor(gate_init, dtype=fit_dtype, device=fit_device))
            ups = torch.nn.Parameter(torch.tensor(up_init, dtype=fit_dtype, device=fit_device))
            scales = torch.nn.Parameter(torch.tensor(scale_init, dtype=fit_dtype, device=fit_device))
            bias = torch.nn.Parameter(torch.tensor(bias_init, dtype=fit_dtype, device=fit_device))
        else:
            gates = torch.nn.Parameter(torch.randn(atom_count, rank, dtype=fit_dtype, device=fit_device) * 0.6)
            ups = torch.nn.Parameter(torch.randn(atom_count, rank, dtype=fit_dtype, device=fit_device) * 0.6)
            scales = torch.nn.Parameter(torch.ones(atom_count, dtype=fit_dtype, device=fit_device) * max(0.1, target_scale / max(1, atom_count)))
            bias = torch.nn.Parameter(torch.tensor(float(np.mean(targets)), dtype=fit_dtype, device=fit_device))
        optimizer = torch.optim.Adam([gates, ups, scales, bias], lr=learning_rate)
        for _ in range(steps):
            optimizer.zero_grad()
            activations = torch.nn.functional.silu(x @ gates.T) * (x @ ups.T)
            prediction = activations @ scales + bias
            loss = torch.mean((prediction - y) ** 2)
            if has_gradient_witness and gx is not None and gy is not None:
                gate_projection = gx @ gates.T
                up_projection = gx @ ups.T
                sigmoid = torch.sigmoid(gate_projection)
                silu_values = torch.nn.functional.silu(gate_projection)
                silu_prime = sigmoid * (1.0 + gate_projection * (1.0 - sigmoid))
                gradient_terms = scales[None, :, None] * (
                    silu_prime[:, :, None] * up_projection[:, :, None] * gates[None, :, :]
                    + silu_values[:, :, None] * ups[None, :, :]
                )
                predicted_gradients = torch.sum(gradient_terms, dim=1)
                target_grad_scale = torch.mean(gy ** 2).detach().clamp_min(torch.tensor(1e-12, dtype=fit_dtype, device=fit_device))
                loss = loss + gradient_weight * torch.mean((predicted_gradients - gy) ** 2) / target_grad_scale
            if has_cubic_witness and cv is not None and cy is not None and target_cubic_scale is not None:
                cubic_gate_projection = cv @ gates.T
                cubic_up_projection = cv @ ups.T
                predicted_cubic = 1.5 * torch.sum(
                    cubic_gate_projection ** 2 * cubic_up_projection * scales[None, :],
                    dim=1,
                )
                loss = loss + cubic_weight * torch.mean((predicted_cubic - cy) ** 2) / target_cubic_scale
            loss.backward()
            optimizer.step()

        lbfgs = torch.optim.LBFGS(
            [gates, ups, scales, bias],
            lr=0.7,
            max_iter=420 if atom_count >= 4 else 260,
            line_search_fn="strong_wolfe"
        )

        def closure() -> torch.Tensor:
            lbfgs.zero_grad()
            activations = torch.nn.functional.silu(x @ gates.T) * (x @ ups.T)
            prediction = activations @ scales + bias
            loss = torch.mean((prediction - y) ** 2)
            if has_gradient_witness and gx is not None and gy is not None:
                gate_projection = gx @ gates.T
                up_projection = gx @ ups.T
                sigmoid = torch.sigmoid(gate_projection)
                silu_values = torch.nn.functional.silu(gate_projection)
                silu_prime = sigmoid * (1.0 + gate_projection * (1.0 - sigmoid))
                gradient_terms = scales[None, :, None] * (
                    silu_prime[:, :, None] * up_projection[:, :, None] * gates[None, :, :]
                    + silu_values[:, :, None] * ups[None, :, :]
                )
                predicted_gradients = torch.sum(gradient_terms, dim=1)
                target_grad_scale = torch.mean(gy ** 2).detach().clamp_min(torch.tensor(1e-12, dtype=fit_dtype, device=fit_device))
                loss = loss + gradient_weight * torch.mean((predicted_gradients - gy) ** 2) / target_grad_scale
            if has_cubic_witness and cv is not None and cy is not None and target_cubic_scale is not None:
                cubic_gate_projection = cv @ gates.T
                cubic_up_projection = cv @ ups.T
                predicted_cubic = 1.5 * torch.sum(
                    cubic_gate_projection ** 2 * cubic_up_projection * scales[None, :],
                    dim=1,
                )
                loss = loss + cubic_weight * torch.mean((predicted_cubic - cy) ** 2) / target_cubic_scale
            loss.backward()
            return loss

        try:
            lbfgs.step(closure)
        except RuntimeError:
            pass

        with torch.no_grad():
            activations = torch.nn.functional.silu(x @ gates.T) * (x @ ups.T)
            activation_np = activations.detach().cpu().numpy().astype(np.float64)
            design = np.column_stack([activation_np, np.ones(coords.shape[0], dtype=np.float64)])
            try:
                coeffs, *_ = np.linalg.lstsq(design, targets, rcond=None)
                if np.all(np.isfinite(coeffs)):
                    scale_values = coeffs[:-1].astype(np.float64)
                    bias_value = float(coeffs[-1])
                    prediction_np = design @ coeffs
                    mae = float(np.mean(np.abs(prediction_np - targets)))
                    mse = float(np.mean((prediction_np - targets) ** 2))
                else:
                    raise np.linalg.LinAlgError("non-finite least-squares coefficients")
            except np.linalg.LinAlgError:
                prediction = activations @ scales + bias
                mae = torch.mean(torch.abs(prediction - y)).item()
                mse = torch.mean((prediction - y) ** 2).item()
                scale_values = scales.detach().cpu().numpy().astype(np.float64)
                bias_value = float(bias.detach().cpu().item())
            score = mae
            r2 = 1.0 - mse / max(target_variance, 1e-12)
            candidate = {
                "restart": restart,
                "mae": mae,
                "mse": mse,
                "r2": r2,
                "score": score,
                "startKind": "structured" if restart < structured_start_count else "random",
                "randomRestartIndex": restart - structured_start_count if restart >= structured_start_count else None,
                "gates": gates.detach().cpu().numpy().astype(np.float64),
                "ups": ups.detach().cpu().numpy().astype(np.float64),
                "scales": scale_values,
                "bias": bias_value,
                # Where the optimisation actually ran, taken from a tensor rather than from the
                # request. This is the field that would have shown, without a profiler, that the
                # whole fit was on the host while the card held the oracle and idled.
                "fitDevice": str(gates.device),
            }
            start_kind = str(candidate["startKind"])
            previous_kind_score = best_score_by_start_kind.get(start_kind)
            if previous_kind_score is None or score < previous_kind_score:
                best_score_by_start_kind[start_kind] = score
                best_metrics_by_start_kind[start_kind] = {
                    "restart": int(restart),
                    "randomRestartIndex": None if candidate["randomRestartIndex"] is None else int(candidate["randomRestartIndex"]),
                    "mae": mae,
                    "mse": mse,
                    "r2": r2,
                }
            if best is None or candidate["score"] < best["score"]:
                best = candidate
    assert best is not None
    best["restartCount"] = total_restart_count
    best["randomRestartCount"] = restart_count
    best["structuredStartCount"] = structured_start_count
    best["bestScoreByStartKind"] = {key: float(value) for key, value in best_score_by_start_kind.items()}
    best["bestMetricsByStartKind"] = best_metrics_by_start_kind
    best["fitMode"] = "sequential+structured-starts" if structured_start_count else "sequential"
    return best


def predict_swiglu(coords: np.ndarray, fit: dict[str, Any]) -> np.ndarray:
    if "gates" in fit:
        activations = silu_np(coords @ fit["gates"].T) * (coords @ fit["ups"].T)
        return activations @ fit["scales"] + fit["bias"]
    return fit["scale"] * silu_np(coords @ fit["gate"]) * (coords @ fit["up"]) + fit["bias"]


def predict_swiglu_active_gradients(coords: np.ndarray, fit: dict[str, Any]) -> np.ndarray:
    gates = np.asarray(fit["gates"], dtype=np.float64)
    ups = np.asarray(fit["ups"], dtype=np.float64)
    scales = np.asarray(fit["scales"], dtype=np.float64)
    gate_projection = coords @ gates.T
    up_projection = coords @ ups.T
    sigmoid = 1.0 / (1.0 + np.exp(-np.clip(gate_projection, -60.0, 60.0)))
    silu_values = gate_projection * sigmoid
    silu_prime = sigmoid * (1.0 + gate_projection * (1.0 - sigmoid))
    gradient_terms = scales[None, :, None] * (
        silu_prime[:, :, None] * up_projection[:, :, None] * gates[None, :, :]
        + silu_values[:, :, None] * ups[None, :, :]
    )
    return np.sum(gradient_terms, axis=1)


def rescore_swiglu_candidate_on_data(
    coords: np.ndarray,
    targets: np.ndarray,
    candidate: dict[str, Any],
    recalibrate_scales: bool = True,
) -> dict[str, Any]:
    gates_np = np.asarray(candidate["gates"], dtype=np.float64)
    ups_np = np.asarray(candidate["ups"], dtype=np.float64)
    scales_np = np.asarray(candidate["scales"], dtype=np.float64)
    bias_value = float(candidate["bias"])
    if recalibrate_scales and gates_np.ndim == 2 and ups_np.shape == gates_np.shape:
        activation_np = silu_np(coords @ gates_np.T) * (coords @ ups_np.T)
        design = np.column_stack([activation_np, np.ones(coords.shape[0], dtype=np.float64)])
        try:
            coeffs, *_ = np.linalg.lstsq(design, targets, rcond=None)
        except np.linalg.LinAlgError:
            coeffs = None
        if coeffs is not None and np.all(np.isfinite(coeffs)):
            scales_np = coeffs[:-1].astype(np.float64)
            bias_value = float(coeffs[-1])
    rescored = dict(candidate)
    rescored.update({
        "scales": scales_np,
        "bias": bias_value,
    })
    prediction = predict_swiglu(coords, rescored)
    metrics = regression_metrics(targets, prediction)
    rescored.update({
        "mae": float(metrics["mae"]),
        "mse": float(metrics["mse"]),
        "score": float(metrics["mae"]),
        "selectionScore": float(metrics["mae"]),
        "worstError": float(metrics["worstError"]),
    })
    return rescored


def fixed_gate_up_refit_swiglu_candidate(
    coords: np.ndarray,
    targets: np.ndarray,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    gates = np.asarray(candidate["gates"], dtype=np.float64)
    if gates.ndim != 2 or gates.shape[0] == 0:
        return candidate
    solved = solve_fixed_gate_start(coords, targets, gates)
    if solved is None:
        return candidate
    solved_gates, solved_ups, solved_scales, solved_bias, solved_mae = solved
    refit = dict(candidate)
    refit.update({
        "gates": solved_gates,
        "ups": solved_ups,
        "scales": solved_scales,
        "bias": solved_bias,
        "mae": float(solved_mae),
        "score": float(solved_mae),
        "selectionScore": float(solved_mae),
        "fixedGateUpRefit": True,
        "fixedGateUpRefitMaeBefore": float(candidate.get("mae", float("inf"))),
    })
    prediction = predict_swiglu(coords, refit)
    metrics = regression_metrics(targets, prediction)
    refit.update({
        "mse": float(metrics["mse"]),
        "worstError": float(metrics["worstError"]),
    })
    return refit


def gate_dictionary_refit_swiglu_candidate(
    coords: np.ndarray,
    targets: np.ndarray,
    candidate: dict[str, Any],
    gradient_coords: np.ndarray | None,
    gradient_targets: np.ndarray | None,
    atom_count: int,
    seed: int,
) -> dict[str, Any]:
    raw_candidates: list[np.ndarray] = []
    for key in ("gates", "ups"):
        values = np.asarray(candidate.get(key, np.zeros((0, coords.shape[1]), dtype=np.float64)), dtype=np.float64)
        if values.ndim == 2 and values.shape[1] == coords.shape[1]:
            for row in values:
                if np.all(np.isfinite(row)) and np.linalg.norm(row) > 1e-9:
                    raw_candidates.append(row)
                    raw_candidates.append(-row)
    raw_candidates.extend(make_oracle_gate_candidate_pool(
        gradient_coords,
        gradient_targets,
        np.vstack(raw_candidates) if raw_candidates else None,
        atom_count,
        seed + 31,
    ))
    dictionary_fit = fit_swiglu_gate_dictionary(
        coords,
        targets,
        raw_candidates,
        atom_count,
        seed + 53,
    )
    if dictionary_fit is None:
        return candidate
    dictionary_fit = rescore_swiglu_candidate_on_data(
        coords,
        targets,
        dictionary_fit,
        recalibrate_scales=True,
    )
    dictionary_fit["gateDictionaryRefit"] = True
    dictionary_fit["gateDictionaryRefitCandidateCount"] = int(len(raw_candidates))
    dictionary_fit["gateDictionaryRefitMaeBefore"] = float(candidate.get("mae", float("inf")))
    return dictionary_fit


def residual_gradient_atom_replace_swiglu_candidate(
    coords: np.ndarray,
    targets: np.ndarray,
    candidate: dict[str, Any],
    atom_indices: list[int],
    residual_gradient_coords: np.ndarray,
    residual_gradient_targets: np.ndarray,
    seed: int,
    selection_coords: np.ndarray | None = None,
    selection_targets: np.ndarray | None = None,
) -> dict[str, Any]:
    gates = np.asarray(candidate["gates"], dtype=np.float64)
    ups = np.asarray(candidate["ups"], dtype=np.float64)
    scales = np.asarray(candidate["scales"], dtype=np.float64)
    if gates.ndim != 2 or gates.shape[0] == 0:
        return candidate
    atom_count, rank = gates.shape
    selected = sorted({int(index) for index in atom_indices if 0 <= int(index) < atom_count})
    if not selected:
        return candidate
    if ups.shape != gates.shape or scales.shape != (atom_count,):
        return candidate

    def atom_gradients(points: np.ndarray, gate: np.ndarray, raw_up: np.ndarray) -> np.ndarray:
        gate_projection = points @ gate
        sigmoid = 1.0 / (1.0 + np.exp(-np.clip(gate_projection, -60.0, 60.0)))
        silu_values = gate_projection * sigmoid
        silu_prime = sigmoid * (1.0 + gate_projection * (1.0 - sigmoid))
        up_projection = points @ raw_up
        return (
            silu_prime[:, None] * up_projection[:, None] * gate[None, :]
            + silu_values[:, None] * raw_up[None, :]
        )

    def solve_atom_up(
        points: np.ndarray,
        gradient_target: np.ndarray,
        gate: np.ndarray,
    ) -> tuple[np.ndarray, float] | None:
        gate_projection = points @ gate
        sigmoid = 1.0 / (1.0 + np.exp(-np.clip(gate_projection, -60.0, 60.0)))
        silu_values = gate_projection * sigmoid
        silu_prime = sigmoid * (1.0 + gate_projection * (1.0 - sigmoid))
        cross_weight = silu_prime * silu_values
        weighted_coords = points * ((silu_prime ** 2) * float(np.dot(gate, gate)))[:, None]
        normal = points.T @ weighted_coords
        cross_vector = points.T @ cross_weight
        normal += np.outer(cross_vector, gate) + np.outer(gate, cross_vector)
        normal += float(np.sum(silu_values ** 2)) * np.eye(rank, dtype=np.float64)
        gate_target = gradient_target @ gate
        rhs = points.T @ (silu_prime * gate_target)
        rhs += np.sum(silu_values[:, None] * gradient_target, axis=0)
        ridge = max(1e-12, float(np.trace(normal)) / max(1, rank) * 1e-10)
        try:
            raw_up = np.linalg.solve(normal + ridge * np.eye(rank, dtype=np.float64), rhs)
        except np.linalg.LinAlgError:
            return None
        if not np.all(np.isfinite(raw_up)) or float(np.linalg.norm(raw_up)) < 1e-12:
            return None
        prediction = atom_gradients(points, gate, raw_up)
        residual = gradient_target - prediction
        row_error = np.linalg.norm(residual, axis=1)
        score = float(np.mean(row_error)) + 0.15 * float(np.max(row_error))
        return raw_up.astype(np.float64), score

    current_gradient = predict_swiglu_active_gradients(residual_gradient_coords, candidate)
    oracle_gradient = current_gradient + np.asarray(residual_gradient_targets, dtype=np.float64)
    base_gradient_residual = oracle_gradient - current_gradient
    base_gradient_error = np.linalg.norm(base_gradient_residual, axis=1)
    base_gradient_score = float(np.mean(base_gradient_error)) + 0.15 * float(np.max(base_gradient_error))
    proposed_atoms: dict[int, tuple[np.ndarray, np.ndarray, float]] = {}
    total_candidate_count = 0

    candidate_limit = max(8, int(os.environ.get(
        "GCL_BLIND_SWIGLU_RESIDUAL_GRADIENT_REPLACE_CANDIDATES",
        "128",
    ) or "128"))
    angle_values = [
        abs(float(value))
        for value in os.environ.get(
            "GCL_BLIND_SWIGLU_RESIDUAL_GRADIENT_REPLACE_ANGLES",
            "0.0125,0.025,0.05,0.1,0.2,0.35",
        ).split(",")
        if value.strip()
    ]
    norm_factors = [
        float(value)
        for value in os.environ.get(
            "GCL_BLIND_SWIGLU_RESIDUAL_GRADIENT_REPLACE_NORM_FACTORS",
            "0.9,1.0,1.1",
        ).split(",")
        if value.strip() and float(value) > 0.0
    ]
    if not angle_values or not norm_factors:
        return candidate

    replace_count = len(selected)
    residual_dirs = make_gradient_gate_directions(
        residual_gradient_coords,
        residual_gradient_targets,
        replace_count,
        seed + 11,
        max_candidates=max(candidate_limit, replace_count * 48),
    )
    if not residual_dirs:
        diagnosed = dict(candidate)
        diagnosed.update({
            "residualGradientAtomReplaced": False,
            "residualGradientAtomReplaceAtoms": selected,
            "residualGradientAtomReplaceAcceptedAtoms": [],
            "residualGradientAtomReplaceDirectionCount": 0,
            "residualGradientAtomReplaceCandidateCount": 0,
            "residualGradientAtomReplaceGradientScoreBefore": base_gradient_score,
            "residualGradientAtomReplaceGradientScoreAfter": base_gradient_score,
        })
        return diagnosed

    for atom_index in selected:
        old_raw_up = scales[atom_index] * ups[atom_index]
        old_atom_gradient = atom_gradients(
            residual_gradient_coords,
            gates[atom_index],
            old_raw_up,
        )
        other_gradient = current_gradient - old_atom_gradient
        atom_gradient_target = oracle_gradient - other_gradient
        base_direction = gates[atom_index]
        base_norm = float(np.linalg.norm(base_direction))
        if not np.isfinite(base_norm) or base_norm < 1e-9:
            continue
        base_unit = base_direction / base_norm
        gate_candidates: list[np.ndarray] = [base_direction.copy()]
        for residual_direction in residual_dirs:
            direction = canonical_direction(residual_direction)
            if direction is None:
                continue
            tangent = direction - float(np.dot(direction, base_unit)) * base_unit
            tangent_norm = float(np.linalg.norm(tangent))
            if tangent_norm < 1e-8:
                continue
            tangent /= tangent_norm
            for angle in angle_values:
                if not np.isfinite(angle) or angle <= 0.0:
                    continue
                for sign in (-1.0, 1.0):
                    rotated = np.cos(angle) * base_unit + sign * np.sin(angle) * tangent
                    for norm_factor in norm_factors:
                        gate_candidates.append(base_norm * norm_factor * rotated)
                        if len(gate_candidates) >= candidate_limit:
                            break
                    if len(gate_candidates) >= candidate_limit:
                        break
                if len(gate_candidates) >= candidate_limit:
                    break
            if len(gate_candidates) >= candidate_limit:
                break

        best_gate = base_direction.copy()
        best_raw_up = old_raw_up.copy()
        base_atom_residual = atom_gradient_target - old_atom_gradient
        base_atom_error = np.linalg.norm(base_atom_residual, axis=1)
        best_atom_score = float(np.mean(base_atom_error)) + 0.15 * float(np.max(base_atom_error))
        for gate in gate_candidates:
            solved_atom = solve_atom_up(residual_gradient_coords, atom_gradient_target, gate)
            total_candidate_count += 1
            if solved_atom is None:
                continue
            raw_up, atom_score = solved_atom
            if atom_score < best_atom_score:
                best_atom_score = atom_score
                best_gate = gate.copy()
                best_raw_up = raw_up.copy()
        if best_atom_score >= float(np.mean(base_atom_error)) + 0.15 * float(np.max(base_atom_error)) - 1e-12:
            continue
        raw_up_norm = float(np.linalg.norm(best_raw_up))
        if raw_up_norm < 1e-12:
            continue
        proposed_atoms[atom_index] = (best_gate, best_raw_up / raw_up_norm, raw_up_norm)

    base_full = rescore_swiglu_candidate_on_data(coords, targets, candidate, recalibrate_scales=True)
    base_metrics_full = regression_metrics(targets, predict_swiglu(coords, base_full))
    selection_count = 0
    base_decision_score = float(base_metrics_full["mae"]) + 0.25 * float(base_metrics_full["worstError"])
    if (
        selection_coords is not None
        and selection_targets is not None
        and selection_coords.shape[0] == selection_targets.shape[0]
        and selection_coords.shape[0] > 0
    ):
        selection_count = int(selection_coords.shape[0])
        base_selection_metrics = regression_metrics(selection_targets, predict_swiglu(selection_coords, base_full))
        base_decision_score = float(base_selection_metrics["mae"]) + 0.35 * float(base_selection_metrics["worstError"])

    proposed_indices = sorted(proposed_atoms)
    subset_limit = min(6, len(proposed_indices))
    subset_candidates: list[tuple[int, ...]] = []
    for subset_size in range(1, subset_limit + 1):
        subset_candidates.extend(combinations(proposed_indices, subset_size))
    if len(proposed_indices) > subset_limit:
        subset_candidates.append(tuple(proposed_indices))

    best_trial: dict[str, Any] | None = None
    best_trial_metrics: dict[str, float] | None = None
    best_subset: list[int] = []
    best_decision_score = base_decision_score
    best_gradient_score = base_gradient_score
    subset_evaluated = 0
    for subset in subset_candidates:
        trial_gates = gates.copy()
        trial_gradient = current_gradient.copy()
        for atom_index in subset:
            proposed_gate, proposed_up, proposed_scale = proposed_atoms[atom_index]
            old_raw_up = scales[atom_index] * ups[atom_index]
            trial_gradient -= atom_gradients(residual_gradient_coords, gates[atom_index], old_raw_up)
            trial_gradient += atom_gradients(
                residual_gradient_coords,
                proposed_gate,
                proposed_scale * proposed_up,
            )
            trial_gates[atom_index] = proposed_gate
        solved = solve_fixed_gate_start(coords, targets, trial_gates)
        subset_evaluated += 1
        if solved is None:
            continue
        solved_gates, solved_ups, solved_scales, solved_bias, _mae = solved
        trial = dict(candidate)
        trial.update({
            "gates": solved_gates,
            "ups": solved_ups,
            "scales": solved_scales,
            "bias": solved_bias,
        })
        trial_metrics = regression_metrics(targets, predict_swiglu(coords, trial))
        trial_decision_score = float(trial_metrics["mae"]) + 0.25 * float(trial_metrics["worstError"])
        if selection_count > 0 and selection_coords is not None and selection_targets is not None:
            trial_selection_metrics = regression_metrics(selection_targets, predict_swiglu(selection_coords, trial))
            trial_decision_score = float(trial_selection_metrics["mae"]) + 0.35 * float(trial_selection_metrics["worstError"])
        gradient_residual = oracle_gradient - trial_gradient
        gradient_error = np.linalg.norm(gradient_residual, axis=1)
        trial_gradient_score = float(np.mean(gradient_error)) + 0.15 * float(np.max(gradient_error))
        train_guard = (
            trial_metrics["mae"] <= base_metrics_full["mae"] * 1.05
            or trial_metrics["worstError"] < base_metrics_full["worstError"]
        )
        if train_guard and trial_decision_score < best_decision_score:
            best_trial = trial
            best_trial_metrics = trial_metrics
            best_subset = list(subset)
            best_decision_score = trial_decision_score
            best_gradient_score = trial_gradient_score

    if best_trial is None or best_trial_metrics is None:
        refined = dict(candidate)
        refined.update({
            "residualGradientAtomReplaced": False,
            "residualGradientAtomReplaceAtoms": selected,
            "residualGradientAtomReplaceProposedAtoms": proposed_indices,
            "residualGradientAtomReplaceAcceptedAtoms": [],
            "residualGradientAtomReplaceDirectionCount": int(len(residual_dirs)),
            "residualGradientAtomReplaceCandidateCount": int(total_candidate_count),
            "residualGradientAtomReplaceSubsetCount": int(subset_evaluated),
            "residualGradientAtomReplaceGradientScoreBefore": base_gradient_score,
            "residualGradientAtomReplaceGradientScoreAfter": base_gradient_score,
            "residualGradientAtomReplaceSelectionCount": selection_count,
            "residualGradientAtomReplaceDecisionScoreBefore": base_decision_score,
            "residualGradientAtomReplaceDecisionScoreAfter": base_decision_score,
            "residualGradientAtomReplaceMaeBefore": float(base_metrics_full["mae"]),
            "residualGradientAtomReplaceWorstBefore": float(base_metrics_full["worstError"]),
        })
        return refined
    refined = dict(best_trial)
    refined.update({
        "residualGradientAtomReplaced": True,
        "residualGradientAtomReplaceAtoms": selected,
        "residualGradientAtomReplaceProposedAtoms": proposed_indices,
        "residualGradientAtomReplaceAcceptedAtoms": best_subset,
        "residualGradientAtomReplaceDirectionCount": int(len(residual_dirs)),
        "residualGradientAtomReplaceCandidateCount": int(total_candidate_count),
        "residualGradientAtomReplaceSubsetCount": int(subset_evaluated),
        "residualGradientAtomReplaceGradientScoreBefore": base_gradient_score,
        "residualGradientAtomReplaceGradientScoreAfter": best_gradient_score,
        "residualGradientAtomReplaceSelectionCount": selection_count,
        "residualGradientAtomReplaceDecisionScoreBefore": base_decision_score,
        "residualGradientAtomReplaceDecisionScoreAfter": best_decision_score,
        "residualGradientAtomReplaceMaeBefore": float(base_metrics_full["mae"]),
        "residualGradientAtomReplaceWorstBefore": float(base_metrics_full["worstError"]),
        "mae": float(best_trial_metrics["mae"]),
        "mse": float(best_trial_metrics["mse"]),
        "score": float(best_trial_metrics["mae"]),
        "selectionScore": float(best_decision_score),
        "worstError": float(best_trial_metrics["worstError"]),
    })
    return refined


def detect_swiglu_cancellation_pairs(
    coords: np.ndarray,
    candidate: dict[str, Any],
    gate_cosine_threshold: float = 0.75,
    up_cosine_threshold: float = 0.75,
    contribution_correlation_threshold: float = -0.50,
) -> list[dict[str, Any]]:
    gates = np.asarray(candidate["gates"], dtype=np.float64)
    ups = np.asarray(candidate["ups"], dtype=np.float64)
    scales = np.asarray(candidate["scales"], dtype=np.float64)
    points = np.asarray(coords, dtype=np.float64)
    if (
        gates.ndim != 2
        or ups.shape != gates.shape
        or scales.shape != (gates.shape[0],)
        or points.ndim != 2
        or points.shape[1] != gates.shape[1]
        or points.shape[0] < 2
    ):
        return []
    effective_ups = ups * scales[:, None]
    gate_norms = np.linalg.norm(gates, axis=1)
    up_norms = np.linalg.norm(effective_ups, axis=1)
    valid = (gate_norms > 1e-12) & (up_norms > 1e-12)
    gate_directions = gates / np.maximum(gate_norms[:, None], 1e-12)
    up_directions = effective_ups / np.maximum(up_norms[:, None], 1e-12)
    contributions = silu_np(points @ gates.T) * (points @ effective_ups.T)
    centered = contributions - np.mean(contributions, axis=0, keepdims=True)
    contribution_norms = np.linalg.norm(centered, axis=0)

    pairs: list[dict[str, Any]] = []
    for left, right in combinations(range(gates.shape[0]), 2):
        if not valid[left] or not valid[right]:
            continue
        gate_cosine = abs(float(np.dot(gate_directions[left], gate_directions[right])))
        up_cosine = abs(float(np.dot(up_directions[left], up_directions[right])))
        correlation_denom = float(contribution_norms[left] * contribution_norms[right])
        if correlation_denom <= 1e-15:
            continue
        contribution_correlation = float(np.dot(centered[:, left], centered[:, right]) / correlation_denom)
        if (
            gate_cosine < gate_cosine_threshold
            or up_cosine < up_cosine_threshold
            or contribution_correlation > contribution_correlation_threshold
        ):
            continue
        score = math.sqrt(gate_cosine * up_cosine) * -contribution_correlation
        pairs.append({
            "atoms": [int(left), int(right)],
            "gateCosine": gate_cosine,
            "upCosine": up_cosine,
            "contributionCorrelation": contribution_correlation,
            "score": float(score),
        })
    pairs.sort(key=lambda item: (-float(item["score"]), item["atoms"]))
    return pairs


def diagnose_unpaired_atoms(
    joint_cosines: np.ndarray,
    gate_cosines: np.ndarray,
    up_cosines: np.ndarray,
    assignment: np.ndarray,
    source_units: list[int],
    recovered_gate_coords: np.ndarray,
    truth_gate_coords: np.ndarray,
    *,
    threshold: float = 0.99,
) -> dict[str, Any]:
    """Say what an atom that failed to pair with its source is actually aligned to.

    The Hungarian assignment reports one number per atom, which says an atom did not match
    its unit and nothing about why. Two failures look identical in that number and need
    opposite repairs: an atom spread thinly across several units has not resolved anything,
    while a pair of atoms that each match the other's unit has recovered the right subspace
    and only mixed the assignment inside it.

    For every atom below `threshold` this records its best alternative sources, whether the
    atom it is confused with is confused back, and the principal cosines between the span of
    the confused atoms' gates and the span of their assigned sources. A high subspace cosine
    with low individual cosines is the rotation case; both low is a genuine miss.

    Post-hoc: every input here is derived from truth the oracle releases only after the
    recovery is frozen.
    """
    atom_count = int(joint_cosines.shape[0])
    below = [index for index in range(atom_count) if float(joint_cosines[index, assignment[index]]) < threshold]
    rows: list[dict[str, Any]] = []
    for atom in below:
        order = np.argsort(joint_cosines[atom])[::-1]
        alternatives = [
            {
                "sourceIndex": int(source_index),
                "sourceUnit": int(source_units[source_index]),
                "jointMinCosine": float(joint_cosines[atom, source_index]),
                "gateCosine": float(gate_cosines[atom, source_index]),
                "upCosine": float(up_cosines[atom, source_index]),
                "isAssigned": bool(int(assignment[atom]) == int(source_index)),
            }
            for source_index in order[:3]
        ]
        # Which atom holds the source this one was assigned, if any other atom matches it better?
        assigned_source = int(assignment[atom])
        best_atom_for_source = int(np.argmax(joint_cosines[:, assigned_source]))
        rows.append({
            "recoveredAtom": int(atom),
            "assignedSourceUnit": int(source_units[assigned_source]),
            "assignedJointMinCosine": float(joint_cosines[atom, assigned_source]),
            "bestAlternatives": alternatives,
            "sourceBestMatchedByAtom": best_atom_for_source,
            "sourceBestMatchIsThisAtom": bool(best_atom_for_source == atom),
        })

    subspace: dict[str, Any] | None = None
    if len(below) >= 2:
        recovered_span = np.vstack([recovered_gate_coords[atom] for atom in below])
        truth_span = np.vstack([truth_gate_coords[int(assignment[atom])] for atom in below])
        cosines = principal_cosines(recovered_span, truth_span)
        subspace = {
            "atoms": [int(atom) for atom in below],
            "sourceUnits": [int(source_units[int(assignment[atom])]) for atom in below],
            "principalCosines": [float(value) for value in cosines],
            "minPrincipalCosine": float(np.min(cosines)),
            # A span recovered but not resolved: the atoms cover the right directions and only
            # the split between them is wrong, which is a different repair from a genuine miss.
            "spanRecoveredButUnresolved": bool(
                float(np.min(cosines)) >= 0.9
                and max(float(joint_cosines[atom, int(assignment[atom])]) for atom in below) < threshold
            ),
        }

    return {
        "threshold": float(threshold),
        "unpairedAtomCount": len(below),
        "atoms": rows,
        "confusedSubspace": subspace,
    }


def paired_atom_identity_passes(
    source_alignment: dict[str, Any],
    threshold: float,
    atom_count: int,
) -> bool:
    """Does every recovered atom match its assigned source unit at `threshold`?

    The sweep enforces this bar on top of the solver's metric thresholds. While the solver
    was not told about it, a case could publish `pass: true` in its own report next to a
    `status: fail` manifest written from the same run, because the two verdicts were reached
    from different criteria. A threshold of 0 leaves the bar off, as lanes that do not claim
    unit identity are entitled to.
    """
    if threshold <= 0.0:
        return True
    minimum = source_alignment.get("pairedAtomMinJointCosine")
    count = source_alignment.get(
        "pairedAtomCountAbove099" if threshold >= 0.99 else "pairedAtomCountAbove095"
    )
    # A bar that was asked for and cannot be measured fails; it does not quietly vanish.
    if minimum is None or count is None:
        return False
    return float(minimum) >= threshold and int(count) >= int(atom_count)


def audit_swiglu_source_cancellation(
    coords: np.ndarray,
    gates: np.ndarray,
    effective_ups: np.ndarray,
    source_units: list[int],
) -> dict[str, Any]:
    points = np.asarray(coords, dtype=np.float64)
    gate_values = np.asarray(gates, dtype=np.float64)
    up_values = np.asarray(effective_ups, dtype=np.float64)
    if (
        points.ndim != 2
        or gate_values.ndim != 2
        or up_values.shape != gate_values.shape
        or points.shape[1] != gate_values.shape[1]
        or len(source_units) != gate_values.shape[0]
    ):
        raise RuntimeError("source cancellation audit received incompatible shapes")
    contributions = silu_np(points @ gate_values.T) * (points @ up_values.T)
    gross = np.sum(np.abs(contributions), axis=1)
    net = np.abs(np.sum(contributions, axis=1))
    mean_gross = float(np.mean(gross))
    mean_net = float(np.mean(net))
    cancellation_fraction = (
        0.0
        if mean_gross <= 1e-15
        else max(0.0, min(1.0, 1.0 - mean_net / mean_gross))
    )
    centered = contributions - np.mean(contributions, axis=0, keepdims=True)
    norms = np.linalg.norm(centered, axis=0)
    pairs: list[dict[str, Any]] = []
    for left, right in combinations(range(gate_values.shape[0]), 2):
        denominator = float(norms[left] * norms[right])
        if denominator <= 1e-15:
            continue
        correlation = float(np.dot(centered[:, left], centered[:, right]) / denominator)
        pairs.append({
            "sourceIndices": [int(left), int(right)],
            "sourceUnits": [int(source_units[left]), int(source_units[right])],
            "contributionCorrelation": correlation,
            "meanAbsContributionLeft": float(np.mean(np.abs(contributions[:, left]))),
            "meanAbsContributionRight": float(np.mean(np.abs(contributions[:, right]))),
        })
    pairs.sort(key=lambda item: float(item["contributionCorrelation"]))
    # Per-source amplitude, which only the pair list carried before, and then only for the
    # eight most negatively correlated pairs. A unit that contributes little to the sum is
    # the one the fit has least reason to resolve, so this is the first thing to look at when
    # some atoms pair with their source and others do not - and it was not readable from the
    # artifact for the units that failed to pair.
    per_source_mean_abs = [float(np.mean(np.abs(contributions[:, index]))) for index in range(gate_values.shape[0])]
    per_source = [
        {
            "sourceIndex": int(index),
            "sourceUnit": int(source_units[index]),
            "meanAbsContribution": per_source_mean_abs[index],
            "shareOfGross": (
                0.0 if mean_gross <= 1e-15 else per_source_mean_abs[index] / mean_gross
            ),
        }
        for index in range(gate_values.shape[0])
    ]
    return {
        "meanGrossContribution": mean_gross,
        "meanNetContribution": mean_net,
        "cancellationFraction": cancellation_fraction,
        "perSourceContribution": per_source,
        "minSourceMeanAbsContribution": float(min(per_source_mean_abs)),
        "maxSourceMeanAbsContribution": float(max(per_source_mean_abs)),
        "sourceAmplitudeRatio": (
            None
            if min(per_source_mean_abs) <= 1e-15
            else float(max(per_source_mean_abs) / min(per_source_mean_abs))
        ),
        "negativePairCount": int(sum(
            1 for item in pairs
            if float(item["contributionCorrelation"]) < 0.0
        )),
        "pairCountAtOrBelowMinus050": int(sum(
            1 for item in pairs
            if float(item["contributionCorrelation"]) <= -0.50
        )),
        "minimumPairCorrelation": (
            None
            if not pairs
            else float(pairs[0]["contributionCorrelation"])
        ),
        "mostNegativePairs": pairs[:min(8, len(pairs))],
    }


def residual_block_refit_cubic_weights(has_cubic_witness: bool) -> tuple[float, float]:
    fit_default = os.environ.get(
        "GCL_BLIND_SWIGLU_FIT_CUBIC_WEIGHT",
        "1.0" if has_cubic_witness else "0.0",
    )
    fit_weight = float(os.environ.get(
        "GCL_BLIND_SWIGLU_RESIDUAL_BLOCK_REFIT_CUBIC_FIT_WEIGHT",
        fit_default,
    ) or "0.0")
    selection_weight = float(os.environ.get(
        "GCL_BLIND_SWIGLU_RESIDUAL_BLOCK_REFIT_CUBIC_SELECTION_WEIGHT",
        os.environ.get("GCL_BLIND_SWIGLU_SELECTION_CUBIC_WEIGHT", "0.0"),
    ) or "0.0")
    if not np.isfinite(fit_weight) or fit_weight < 0.0:
        raise RuntimeError(
            "GCL_BLIND_SWIGLU_RESIDUAL_BLOCK_REFIT_CUBIC_FIT_WEIGHT "
            "must be non-negative and finite"
        )
    if not np.isfinite(selection_weight) or selection_weight < 0.0:
        raise RuntimeError(
            "GCL_BLIND_SWIGLU_RESIDUAL_BLOCK_REFIT_CUBIC_SELECTION_WEIGHT "
            "must be non-negative and finite"
        )
    return fit_weight, selection_weight


def detect_piecewise_quadratic_boundaries(
    line_coordinates: np.ndarray,
    values: np.ndarray,
    expected_count: int,
) -> list[dict[str, float]]:
    coordinates = np.asarray(line_coordinates, dtype=np.float64)
    observed = np.asarray(values, dtype=np.float64)
    if (
        coordinates.ndim != 1
        or observed.shape != coordinates.shape
        or coordinates.size < max(17, expected_count * 4 + 1)
        or expected_count <= 0
    ):
        return []
    steps = np.diff(coordinates)
    step = float(np.median(steps))
    if not np.isfinite(step) or step <= 0.0 or np.max(np.abs(steps - step)) > step * 1e-5:
        return []
    second = np.diff(observed, n=2) / (step ** 2)
    if second.size < 7:
        return []
    smoothed = np.convolve(second, np.array([0.2, 0.6, 0.2]), mode="same")
    third_jump = np.diff(smoothed)
    scores = np.abs(third_jump)
    score_median = float(np.median(scores))
    score_mad = float(np.median(np.abs(scores - score_median)))
    score_floor = max(
        score_median + 8.0 * 1.4826 * score_mad,
        float(np.max(scores)) * 1e-4,
        1e-18,
    )
    margin = 4
    candidate_order = np.argsort(scores)[::-1]
    separation = max(2, coordinates.size // max(8, expected_count * 8))
    selected: list[int] = []
    for raw_index in candidate_order:
        index = int(raw_index)
        if float(scores[index]) < score_floor:
            break
        if index < margin or index >= scores.size - margin:
            continue
        if any(abs(index - existing) <= separation for existing in selected):
            continue
        selected.append(index)
        if len(selected) >= expected_count:
            break
    if not selected:
        return []

    boundaries: list[dict[str, float]] = []
    for index in selected:
        local_start = max(0, index - separation)
        local_stop = min(scores.size, index + separation + 1)
        local_scores = scores[local_start:local_stop]
        local_positions = 0.5 * (
            coordinates[local_start + 1:local_stop + 1]
            + coordinates[local_start + 2:local_stop + 2]
        )
        denominator = float(np.sum(local_scores))
        boundary = (
            float(np.sum(local_scores * local_positions) / denominator)
            if denominator > 1e-18
            else float(0.5 * (coordinates[index + 1] + coordinates[index + 2]))
        )
        window = max(2, separation)
        left = smoothed[max(0, index - 2 * window):max(1, index - window)]
        right = smoothed[min(smoothed.size - 1, index + window + 1):min(smoothed.size, index + 2 * window + 1)]
        if left.size == 0 or right.size == 0:
            jump = float(third_jump[index])
        else:
            jump = float(np.median(right) - np.median(left))
        boundaries.append({
            "coordinate": boundary,
            "jump": jump,
            "score": float(scores[index]),
        })
    boundaries.sort(key=lambda item: float(item["coordinate"]))
    return boundaries


def fit_origin_hyperplane_normal(points: np.ndarray) -> tuple[np.ndarray, float] | None:
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] <= values.shape[1] or values.shape[1] < 2:
        return None
    working = values
    normal: np.ndarray | None = None
    for _ in range(3):
        try:
            _u, _s, vh = np.linalg.svd(working, full_matrices=False)
        except np.linalg.LinAlgError:
            return None
        normal = canonical_direction(vh[-1])
        if normal is None:
            return None
        residual = np.abs(values @ normal)
        if working.shape[0] <= values.shape[1] + 8:
            break
        cutoff = float(np.quantile(residual, 0.85))
        keep = residual <= cutoff
        if int(np.sum(keep)) <= values.shape[1]:
            break
        working = values[keep]
    if normal is None:
        return None
    residual = np.abs(values @ normal)
    scale = np.linalg.norm(values, axis=1).clip(min=1e-9)
    return normal, float(np.median(residual / scale))


def choose_gate_stitch_probe_direction(
    rng: np.random.Generator,
    gate_directions: np.ndarray,
    candidate_count: int = 2048,
) -> np.ndarray:
    directions = np.asarray(gate_directions, dtype=np.float64)
    best: tuple[float, np.ndarray] | None = None
    for _ in range(max(64, candidate_count)):
        candidate = normalize(rng.normal(size=directions.shape[1]))
        projections = np.abs(directions @ candidate)
        score = float(np.min(projections)) + 0.25 * float(np.quantile(projections, 0.25))
        if best is None or score > best[0]:
            best = (score, candidate)
    assert best is not None
    return best[1]


_CLUSTER_LAST_REASON: dict[str, Any] = {"reason": None}


def cluster_repeated_gate_normals(
    normals: np.ndarray,
    strengths: np.ndarray,
    expected_count: int,
    min_abs_cosine: float = 0.965,
) -> list[dict[str, Any]]:
    """Group boundary normals into one direction per atom, searching the grouping
    threshold instead of trusting a single hardcoded value.

    The clustering is all-or-nothing: a run that ends up one cluster short returns an
    empty list, and the caller then refuses the whole structure. On a weak block the
    recovered normals scatter, so a threshold that is right for a strong block splits one
    atom's normals across two groups and the whole recovery is thrown away with 350 usable
    normals in hand.

    The acceptance criterion is unchanged: exactly `expected_count` clusters, each with at
    least two supporting normals. Only the threshold is searched, starting at the caller's
    value so a block that already groups cleanly is unaffected. A wrongly merged pair
    cannot sneak through, because the recovered structure still has to clear the
    post-hoc paired-atom cosine and the held-out fidelity gates.
    """
    ladder_raw = os.environ.get("GCL_BLIND_SWIGLU_GATE_NORMAL_CLUSTER_COSINE_LADDER", "").strip()
    if ladder_raw:
        ladder = [float(value) for value in ladder_raw.split(",") if value.strip()]
    else:
        ladder = [0.95, 0.93, 0.90, 0.86]
    thresholds = [float(min_abs_cosine)]
    for value in ladder:
        if 0.0 < value < 1.0 and value < thresholds[-1]:
            thresholds.append(value)
    primary: list[dict[str, Any]] = []
    for index, threshold in enumerate(thresholds):
        attempt = _cluster_gate_normals_at_cosine(normals, strengths, expected_count, threshold)
        if index == 0:
            primary = attempt
        if len(attempt) == expected_count:
            return attempt
    return primary


def _cluster_gate_normals_at_cosine(
    normals: np.ndarray,
    strengths: np.ndarray,
    expected_count: int,
    min_abs_cosine: float,
) -> list[dict[str, Any]]:
    directions = np.asarray(normals, dtype=np.float64)
    weights = np.asarray(strengths, dtype=np.float64)
    if (
        directions.ndim != 2
        or weights.shape != (directions.shape[0],)
        or expected_count <= 0
        or directions.shape[0] < expected_count * 2
        or not 0.0 < min_abs_cosine < 1.0
    ):
        _CLUSTER_LAST_REASON["reason"] = "shape-guard"
        return []
    normalized: list[np.ndarray] = []
    normalized_weights: list[float] = []
    for direction, weight in zip(directions, weights):
        canonical = canonical_direction(direction)
        if canonical is None or not np.isfinite(weight) or weight <= 0.0:
            continue
        normalized.append(canonical)
        normalized_weights.append(float(weight))
    if len(normalized) < expected_count * 2:
        _CLUSTER_LAST_REASON["reason"] = "normalized-count-guard"
        return []
    values = np.vstack(normalized)
    raw_weights = np.asarray(normalized_weights, dtype=np.float64)
    weight_cap = max(float(np.quantile(raw_weights, 0.90)), 1e-12)
    robust_weights = np.minimum(raw_weights, weight_cap)
    # Cluster centres are the weighted sum of unit member directions, and that sum is fed
    # to canonical_direction, which rejects anything with norm below 1e-9. The weights
    # carry the block's own signal scale, so on a low-amplitude block the sum lands under
    # that absolute floor and a perfectly good cluster is discarded as "centre undefined".
    # Rescaling the weights to a maximum of one cannot change the resulting direction, a
    # positive rescale of a vector leaves its normalisation identical, and it removes the
    # dependence on how loud the block happens to be. raw_weights is left untouched
    # because the reported cluster strength is meant to stay in the block's own units.
    robust_weight_peak = float(np.max(robust_weights)) if robust_weights.size else 0.0
    if robust_weight_peak > 0.0:
        robust_weights = robust_weights / robust_weight_peak

    remaining = np.ones(values.shape[0], dtype=bool)
    clusters: list[dict[str, Any]] = []
    for _ in range(expected_count):
        candidates = np.flatnonzero(remaining)
        best: tuple[int, float, int, np.ndarray] | None = None
        for index in candidates:
            support_mask = remaining & (np.abs(values @ values[index]) >= min_abs_cosine)
            support_count = int(np.sum(support_mask))
            support_weight = float(np.sum(robust_weights[support_mask]))
            score = (support_count, support_weight)
            if best is None or score > (best[0], best[1]):
                best = (support_count, support_weight, int(index), support_mask)
        if best is None or best[0] < 2:
            _CLUSTER_LAST_REASON["reason"] = "no-support-candidate"
            return []
        support_mask = best[3]
        members = values[support_mask].copy()
        anchor = values[best[2]]
        members[np.sum(members * anchor, axis=1) < 0.0] *= -1.0
        center = canonical_direction(np.sum(
            members * robust_weights[support_mask, None],
            axis=0,
        ))
        if center is None:
            _CLUSTER_LAST_REASON["reason"] = "center-undefined"
            return []
        clusters.append({
            "direction": center,
            "support": int(np.sum(support_mask)),
            "strength": float(np.sum(raw_weights[support_mask])),
        })
        remaining[support_mask] = False

    centers = np.vstack([item["direction"] for item in clusters])
    for _ in range(3):
        scores = np.abs(values @ centers.T)
        assignments = np.argmax(scores, axis=1)
        refreshed: list[dict[str, Any]] = []
        for cluster_index in range(expected_count):
            member_mask = (
                (assignments == cluster_index)
                & (scores[:, cluster_index] >= min_abs_cosine)
            )
            if int(np.sum(member_mask)) < 2:
                _CLUSTER_LAST_REASON["reason"] = "refine-member-shortfall"
                return []
            members = values[member_mask].copy()
            anchor = centers[cluster_index]
            members[np.sum(members * anchor, axis=1) < 0.0] *= -1.0
            center = canonical_direction(np.sum(
                members * robust_weights[member_mask, None],
                axis=0,
            ))
            if center is None:
                _CLUSTER_LAST_REASON["reason"] = "refine-center-undefined"
                return []
            refreshed.append({
                "direction": center,
                "support": int(np.sum(member_mask)),
                "strength": float(np.sum(raw_weights[member_mask])),
            })
        clusters = refreshed
        centers = np.vstack([item["direction"] for item in clusters])
    return clusters


def recover_up_direction_from_hessian_gate_action(
    gate_direction: np.ndarray,
    hessian_jump_times_gate: np.ndarray,
) -> np.ndarray | None:
    gate = canonical_direction(gate_direction)
    action = np.asarray(hessian_jump_times_gate, dtype=np.float64)
    if gate is None or action.shape != gate.shape or not np.all(np.isfinite(action)):
        return None
    # `action` is a second difference of oracle gradients, so it carries the block's output
    # amplitude and nothing here normalises it. Only its direction is wanted, but
    # canonical_direction rejects below an absolute norm of 1e-9, so on a quiet block a
    # perfectly good up-direction is returned as None and the atom falls back. Dividing by
    # the action norm first makes the test relative, so what disqualifies a sample is the
    # geometry rather than how loud the block is.
    #
    # An earlier version of this comment claimed the disqualifying case was an action
    # parallel to the gate. That is wrong, and the test that pins this behaviour says so:
    # for a unit gate, an action of c*gate gives 0.5*c*gate, not a collapse. The map
    # degenerates only when the action itself vanishes.
    action_norm = float(np.linalg.norm(action))
    if not np.isfinite(action_norm) or action_norm <= 0.0:
        return None
    effective_up = (action - 0.5 * float(np.dot(gate, action)) * gate) / action_norm
    if float(np.linalg.norm(effective_up)) < 1e-6:
        return None
    return canonical_direction(effective_up)


def apply_asymptotic_support_fallback(
    recovered_gate_directions: np.ndarray,
    recovered_up_directions: list[np.ndarray | None],
    gate_supports: np.ndarray,
    up_supports: np.ndarray,
    initial_gate_directions: np.ndarray,
    initial_up_directions: np.ndarray,
    minimum_gate_support: int,
    minimum_up_support: int,
) -> tuple[np.ndarray, list[np.ndarray], list[int]]:
    recovered_gates = np.asarray(recovered_gate_directions, dtype=np.float64)
    initial_gates = np.asarray(initial_gate_directions, dtype=np.float64)
    initial_ups = np.asarray(initial_up_directions, dtype=np.float64)
    gate_support_values = np.asarray(gate_supports, dtype=np.int64)
    up_support_values = np.asarray(up_supports, dtype=np.int64)
    atom_count = recovered_gates.shape[0]
    if (
        recovered_gates.ndim != 2
        or initial_gates.shape != recovered_gates.shape
        or initial_ups.shape != recovered_gates.shape
        or len(recovered_up_directions) != atom_count
        or gate_support_values.shape != (atom_count,)
        or up_support_values.shape != (atom_count,)
        or minimum_gate_support <= 0
        or minimum_up_support <= 0
    ):
        raise RuntimeError("asymptotic support fallback inputs are incompatible")
    selected_gates = recovered_gates.copy()
    selected_ups: list[np.ndarray] = []
    fallback_atoms: list[int] = []
    for atom_index in range(atom_count):
        recovered_up = recovered_up_directions[atom_index]
        use_fallback = (
            gate_support_values[atom_index] < minimum_gate_support
            or up_support_values[atom_index] < minimum_up_support
            or recovered_up is None
        )
        if use_fallback:
            selected_gates[atom_index] = initial_gates[atom_index]
            selected_ups.append(initial_ups[atom_index].copy())
            fallback_atoms.append(atom_index)
        else:
            selected_ups.append(np.asarray(recovered_up, dtype=np.float64).copy())
    return selected_gates, selected_ups, fallback_atoms


def make_asymptotic_cold_start_scaffold(
    atom_count: int,
    rank: int,
    targets: np.ndarray,
) -> dict[str, Any]:
    if atom_count <= 0 or rank <= 0:
        raise RuntimeError("asymptotic cold start requires positive atom_count and rank")
    target_values = np.asarray(targets, dtype=np.float64)
    if target_values.ndim != 1 or target_values.size == 0 or not np.all(np.isfinite(target_values)):
        raise RuntimeError("asymptotic cold start requires finite one-dimensional targets")
    gates = np.zeros((atom_count, rank), dtype=np.float64)
    ups = np.zeros_like(gates)
    for atom_index in range(atom_count):
        gates[atom_index, atom_index % rank] = 1.0
        ups[atom_index, (atom_index + 1) % rank] = 1.0
    bias = float(np.mean(target_values))
    baseline = np.full(target_values.shape, bias, dtype=np.float64)
    metrics = regression_metrics(target_values, baseline)
    return {
        "gates": gates,
        "ups": ups,
        "scales": np.zeros(atom_count, dtype=np.float64),
        "bias": bias,
        "mae": float(metrics["mae"]),
        "mse": float(metrics["mse"]),
        "r2": float(metrics["r2"]),
        "worstError": float(metrics["worstError"]),
        "score": float(metrics["mae"]),
        "selectionScore": float(metrics["mae"]),
        "restartCount": 0,
        "randomRestartCount": 0,
        "structuredStartCount": 0,
        "startKind": "asymptotic-cold-start-scaffold",
        "fitMode": "asymptotic-structure-cold-start",
        "bestScoreByStartKind": {},
        "bestMetricsByStartKind": {},
    }


def asymptotic_stitch_query_upper_bound(
    line_count: int,
    samples_per_line: int,
    atom_count: int,
    rank: int,
    hessian_representatives: int,
) -> int:
    if min(line_count, samples_per_line, atom_count, rank, hessian_representatives) <= 0:
        raise RuntimeError("asymptotic query-bound inputs must be positive")
    line_queries = line_count * samples_per_line
    normal_queries = line_count * atom_count * 2 * (rank + 1)
    hessian_queries = atom_count * hessian_representatives * 4 * rank * 2 * 2
    return int(line_queries + normal_queries + hessian_queries)


def solve_bidirectional_gate_orientation_start(
    coords: np.ndarray,
    targets: np.ndarray,
    gate_directions: np.ndarray,
    gate_norm: float,
    *,
    sample_limit: int = 4096,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float] | None:
    points = np.asarray(coords, dtype=np.float64)
    values = np.asarray(targets, dtype=np.float64)
    directions = np.asarray(gate_directions, dtype=np.float64)
    if (
        points.ndim != 2
        or values.shape != (points.shape[0],)
        or directions.ndim != 2
        or directions.shape[1] != points.shape[1]
        or directions.shape[0] == 0
        or not np.isfinite(gate_norm)
        or gate_norm <= 0.0
    ):
        return None

    count = min(points.shape[0], max(512, int(sample_limit)))
    if count < points.shape[0]:
        sample_indices = np.linspace(0, points.shape[0] - 1, count, dtype=np.int64)
        court_points = points[sample_indices]
        court_values = values[sample_indices]
    else:
        court_points = points
        court_values = values

    positive_gates = directions * gate_norm
    signed_gates = np.vstack([positive_gates, -positive_gates])
    blocks = [silu_np(court_points @ gate)[:, None] * court_points for gate in signed_gates]
    design = np.column_stack(blocks + [np.ones(court_points.shape[0], dtype=np.float64)])
    try:
        coefficients, *_ = np.linalg.lstsq(design, court_values, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(coefficients)):
        return None
    rank = points.shape[1]
    raw_ups = coefficients[:-1].reshape(signed_gates.shape[0], rank)
    positive_strength = np.linalg.norm(raw_ups[:directions.shape[0]], axis=1)
    negative_strength = np.linalg.norm(raw_ups[directions.shape[0]:], axis=1)
    signs = np.where(positive_strength >= negative_strength, 1.0, -1.0)
    oriented_gates = positive_gates * signs[:, None]
    solved = solve_fixed_gate_start(points, values, oriented_gates)
    if solved is None:
        return None
    solved_gates, solved_ups, solved_scales, solved_bias, solved_mae = solved
    return solved_gates, solved_ups, solved_scales, solved_bias, solved_mae


def asymptotic_gate_stitch_swiglu_candidate(
    oracle: OracleClient,
    basis: np.ndarray,
    coords: np.ndarray,
    targets: np.ndarray,
    candidate: dict[str, Any],
    seed: int,
    *,
    selection_coords: np.ndarray | None = None,
    selection_targets: np.ndarray | None = None,
    gradient_coords: np.ndarray | None = None,
    gradient_targets: np.ndarray | None = None,
    cubic_directions: np.ndarray | None = None,
    cubic_targets: np.ndarray | None = None,
    line_count: int,
    samples_per_line: int,
    line_range: float,
    offset_scale: float,
    asymptotic_scale: float,
    cold_start: bool = False,
) -> dict[str, Any]:
    gates = np.asarray(candidate["gates"], dtype=np.float64)
    if gates.ndim != 2 or gates.shape[0] < 2 or gates.shape[1] != basis.shape[0]:
        return candidate
    atom_count, rank = gates.shape
    if (
        line_count < 4
        or samples_per_line < atom_count * 4 + 1
        or samples_per_line % 2 == 0
        or line_range <= 0.0
        or offset_scale <= 0.0
        or asymptotic_scale <= 0.0
    ):
        raise RuntimeError("asymptotic gate stitch configuration is invalid")
    rng = np.random.default_rng(seed)
    if cold_start:
        initial_gate_norm = float(os.environ.get(
            "GCL_BLIND_SWIGLU_ASYMPTOTIC_COLD_START_GATE_NORM",
            "0.25",
        ) or "0.25")
        if not np.isfinite(initial_gate_norm) or initial_gate_norm <= 0.0:
            raise RuntimeError("asymptotic cold-start gate norm must be positive and finite")
        gate_norms = np.full(atom_count, initial_gate_norm, dtype=np.float64)
        initial_directions = np.zeros_like(gates)
        balanced_line_direction = normalize(rng.normal(size=rank))
    else:
        gate_norms = np.linalg.norm(gates, axis=1).clip(min=1e-9)
        initial_directions = gates / gate_norms[:, None]
        balanced_line_direction = choose_gate_stitch_probe_direction(rng, initial_directions)
    line_coordinates = np.linspace(-line_range, line_range, samples_per_line, dtype=np.float64)
    line_rows: list[dict[str, Any]] = []
    failed_lines = 0
    for line_index in range(line_count):
        if line_index == 0:
            line_direction = balanced_line_direction
        else:
            line_direction = normalize(rng.normal(size=rank))
            if not cold_start:
                for _ in range(64):
                    if float(np.min(np.abs(initial_directions @ line_direction))) >= 0.02:
                        break
                    line_direction = normalize(rng.normal(size=rank))
        offset = rng.normal(0.0, offset_scale, size=rank)
        offset -= float(np.dot(offset, line_direction)) * line_direction
        active_points = offset[None, :] + line_coordinates[:, None] * line_direction[None, :]
        inputs = (asymptotic_scale * active_points) @ basis
        outputs = oracle.eval(
            inputs.astype(np.float32),
            stage=f"asymptotic-gate-stitch-line-{line_index}",
        ) / (asymptotic_scale ** 2)
        boundaries = detect_piecewise_quadratic_boundaries(
            line_coordinates,
            outputs,
            atom_count,
        )
        if len(boundaries) < max(3, atom_count // 4):
            failed_lines += 1
            continue
        line_rows.append({
            "offset": offset,
            "direction": line_direction,
            "boundaries": boundaries,
        })
    if len(line_rows) < 4:
        diagnosed = dict(candidate)
        diagnosed.update({
            "asymptoticGateStitched": False,
            "asymptoticGateStitchLineCount": int(line_count),
            "asymptoticGateStitchUsableLineCount": int(len(line_rows)),
            "asymptoticGateStitchFailedLineCount": int(failed_lines),
        })
        return diagnosed

    coordinate_step = float(line_coordinates[1] - line_coordinates[0])
    boundary_rows: list[dict[str, Any]] = []
    for row in line_rows:
        offset = np.asarray(row["offset"], dtype=np.float64)
        row_direction = np.asarray(row["direction"], dtype=np.float64)
        boundaries = row["boundaries"]
        boundary_coordinates = np.asarray([
            float(item["coordinate"])
            for item in boundaries
        ], dtype=np.float64)
        for boundary_index, boundary in enumerate(boundaries):
            neighbor_gaps = []
            if boundary_index > 0:
                neighbor_gaps.append(boundary_coordinates[boundary_index] - boundary_coordinates[boundary_index - 1])
            if boundary_index + 1 < len(boundaries):
                neighbor_gaps.append(boundary_coordinates[boundary_index + 1] - boundary_coordinates[boundary_index])
            nearest_gap = min(neighbor_gaps) if neighbor_gaps else line_range
            if nearest_gap < coordinate_step * 6.0:
                continue
            side_delta = min(0.08, nearest_gap * 0.25)
            side_delta = max(side_delta, coordinate_step * 2.5)
            boundary_point = offset + float(boundary["coordinate"]) * row_direction
            boundary_rows.append({
                "boundaryPoint": boundary_point,
                "minus": boundary_point - side_delta * row_direction,
                "plus": boundary_point + side_delta * row_direction,
                "score": float(boundary["score"]),
                "sideDelta": float(side_delta),
                "lineDirection": row_direction,
            })

    normal_directions: list[np.ndarray] = []
    normal_strengths: list[float] = []
    normal_witness_rows: list[dict[str, Any]] = []
    gradient_step = max(1e-4, coordinate_step * 0.25)
    gradient_chunk_size = 32
    for chunk_start in range(0, len(boundary_rows), gradient_chunk_size):
        chunk = boundary_rows[chunk_start:chunk_start + gradient_chunk_size]
        side_coords = np.vstack([
            row[side]
            for row in chunk
            for side in ("minus", "plus")
        ])
        side_gradients = estimate_active_forward_gradients_at_coords(
            oracle,
            basis,
            asymptotic_scale * side_coords,
            [asymptotic_scale * gradient_step],
            stage=f"asymptotic-gate-normal-{chunk_start // gradient_chunk_size}",
        ) / asymptotic_scale
        for local_index, row in enumerate(chunk):
            minus_gradient = side_gradients[2 * local_index]
            plus_gradient = side_gradients[2 * local_index + 1]
            jump = plus_gradient - minus_gradient
            direction = canonical_direction(jump)
            strength = float(np.linalg.norm(jump))
            if direction is None or not np.isfinite(strength) or strength <= 1e-10:
                continue
            normal_directions.append(direction)
            normal_strengths.append(strength * max(float(row["score"]), 1e-12))
            normal_witness_rows.append(row)

    clusters = cluster_repeated_gate_normals(
        np.vstack(normal_directions) if normal_directions else np.zeros((0, rank)),
        np.asarray(normal_strengths, dtype=np.float64),
        atom_count,
    )
    if len(clusters) != atom_count:
        diagnosed = dict(candidate)
        diagnosed.update({
            "asymptoticGateStitched": False,
            "asymptoticGateStitchLineCount": int(line_count),
            "asymptoticGateStitchUsableLineCount": int(len(line_rows)),
            "asymptoticGateStitchFailedLineCount": int(failed_lines),
            "asymptoticGateStitchBoundaryNormalCount": int(len(normal_directions)),
            "asymptoticGateStitchClusterCount": int(len(clusters)),
        })
        return diagnosed

    recovered_centers = np.vstack([item["direction"] for item in clusters])
    normal_matrix = np.vstack(normal_directions)
    normal_cluster_scores = np.abs(normal_matrix @ recovered_centers.T)
    normal_cluster_assignments = np.argmax(normal_cluster_scores, axis=1)
    hessian_probe_coords: list[np.ndarray] = []
    hessian_probe_meta: list[tuple[int, float, float]] = []
    hessian_representatives = int(os.environ.get(
        "GCL_BLIND_SWIGLU_ASYMPTOTIC_STITCH_HESSIAN_REPRESENTATIVES",
        "5",
    ) or "5")
    if hessian_representatives <= 0:
        raise RuntimeError("asymptotic stitch Hessian representatives must be positive")
    for cluster_index, gate_direction in enumerate(recovered_centers):
        member_indices = np.flatnonzero(
            (normal_cluster_assignments == cluster_index)
            & (normal_cluster_scores[:, cluster_index] >= 0.965)
        )
        if member_indices.size == 0:
            continue
        ranked_members = sorted(
            [int(index) for index in member_indices],
            key=lambda index: normal_strengths[index],
            reverse=True,
        )[:hessian_representatives]
        for representative_index in ranked_members:
            representative = normal_witness_rows[representative_index]
            boundary_point = np.asarray(representative["boundaryPoint"], dtype=np.float64)
            normal_side_delta = float(os.environ.get(
                "GCL_BLIND_SWIGLU_ASYMPTOTIC_STITCH_HESSIAN_NORMAL_DELTA",
                "0.10",
            ) or "0.10")
            if not np.isfinite(normal_side_delta) or normal_side_delta <= 0.0:
                raise RuntimeError("asymptotic stitch Hessian normal delta must be positive and finite")
            minus_side = boundary_point - normal_side_delta * gate_direction
            plus_side = boundary_point + normal_side_delta * gate_direction
            hessian_step = min(
                0.01,
                max(gradient_step * 2.0, normal_side_delta * 0.08),
            )
            hessian_probe_coords.extend([
                minus_side - hessian_step * gate_direction,
                minus_side + hessian_step * gate_direction,
                plus_side - hessian_step * gate_direction,
                plus_side + hessian_step * gate_direction,
            ])
            hessian_probe_meta.append((
                cluster_index,
                hessian_step,
                float(normal_strengths[representative_index]),
            ))

    recovered_up_samples: list[list[tuple[np.ndarray, float]]] = [
        [] for _ in range(atom_count)
    ]
    if hessian_probe_meta:
        hessian_gradient_base_step = asymptotic_scale * gradient_step
        hessian_gradient_steps = [
            hessian_gradient_base_step,
            2.0 * hessian_gradient_base_step,
        ]
        hessian_gradient_chunk_size = int(os.environ.get(
            "GCL_BLIND_SWIGLU_ASYMPTOTIC_STITCH_HESSIAN_GRADIENT_CHUNK_SIZE",
            "32",
        ) or "32")
        hessian_probe_gradients = estimate_active_symmetric_richardson_gradients_at_coords(
            oracle,
            basis,
            asymptotic_scale * np.vstack(hessian_probe_coords),
            hessian_gradient_steps,
            stage="asymptotic-up-hessian-action",
            chunk_size=hessian_gradient_chunk_size,
        ) / asymptotic_scale
        for probe_index, (cluster_index, hessian_step, witness_strength) in enumerate(hessian_probe_meta):
            base = probe_index * 4
            hessian_minus_action = (
                hessian_probe_gradients[base + 1]
                - hessian_probe_gradients[base]
            ) / (2.0 * hessian_step)
            hessian_plus_action = (
                hessian_probe_gradients[base + 3]
                - hessian_probe_gradients[base + 2]
            ) / (2.0 * hessian_step)
            recovered_up = recover_up_direction_from_hessian_gate_action(
                recovered_centers[cluster_index],
                hessian_plus_action - hessian_minus_action,
            )
            if recovered_up is not None:
                recovered_up_samples[cluster_index].append((recovered_up, witness_strength))

    recovered_up_centers: list[np.ndarray | None] = []
    recovered_up_supports: list[int] = []
    for samples in recovered_up_samples:
        if not samples:
            recovered_up_centers.append(None)
            recovered_up_supports.append(0)
            continue
        sample_directions = np.vstack([direction for direction, _weight in samples])
        sample_weights = np.asarray([weight for _direction, weight in samples], dtype=np.float64)
        similarities = np.abs(sample_directions @ sample_directions.T)
        anchor_index = int(np.argmax(np.median(similarities, axis=1)))
        anchor = sample_directions[anchor_index]
        inlier_mask = similarities[anchor_index] >= 0.85
        inlier_directions = sample_directions[inlier_mask].copy()
        inlier_directions[np.sum(inlier_directions * anchor, axis=1) < 0.0] *= -1.0
        inlier_weights = sample_weights[inlier_mask]
        weight_cap = max(float(np.quantile(inlier_weights, 0.80)), 1e-12)
        # Same construction as the gate-normal cluster centre, and it needs the same
        # rescale. These weights are a product of two amplitude-carrying quantities, the
        # gradient-jump norm and the boundary score, so they scale as amplitude squared:
        # across a 237x block range that is a factor of about 5.6e4. The weighted sum of a
        # handful of unit vectors then lands under the 1e-9 floor inside
        # canonical_direction, which returns None, and the atom silently drops to the
        # linear-up fallback that the isolated gate refuses on sourceFallbackAtoms.
        # Rescaling to a peak of one cannot change the resulting direction.
        robust_up_weights = np.minimum(inlier_weights, weight_cap)
        up_weight_peak = float(np.max(robust_up_weights)) if robust_up_weights.size else 0.0
        if up_weight_peak > 0.0:
            robust_up_weights = robust_up_weights / up_weight_peak
        recovered_up_centers.append(canonical_direction(np.sum(
            inlier_directions * robust_up_weights[:, None],
            axis=0,
        )))
        recovered_up_supports.append(int(np.sum(inlier_mask)))

    if cold_start:
        assignment = np.asarray(sorted(
            range(atom_count),
            key=lambda index: tuple(np.round(recovered_centers[index], decimals=12).tolist()),
        ), dtype=np.int64)
        current_directions = recovered_centers[assignment].copy()
    else:
        assignment = maximum_weight_assignment(np.abs(initial_directions @ recovered_centers.T))
        current_directions = recovered_centers[assignment]
        current_directions[
            np.sum(current_directions * initial_directions, axis=1) < 0.0
        ] *= -1.0
    assigned_gate_supports = np.asarray([
        clusters[index]["support"]
        for index in assignment
    ], dtype=np.int64)

    assigned_up_directions = [recovered_up_centers[index] for index in assignment]
    assigned_up_supports = np.asarray([
        recovered_up_supports[index]
        for index in assignment
    ], dtype=np.int64)
    minimum_gate_support = int(os.environ.get(
        "GCL_BLIND_SWIGLU_ASYMPTOTIC_STITCH_MIN_GATE_SUPPORT",
        "4",
    ) or "4")
    minimum_up_support = int(os.environ.get(
        "GCL_BLIND_SWIGLU_ASYMPTOTIC_STITCH_MIN_UP_SUPPORT",
        "3",
    ) or "3")
    if minimum_gate_support <= 0 or minimum_up_support <= 0:
        raise RuntimeError("asymptotic stitch minimum gate/up support must be positive")
    if cold_start:
        fallback_atoms = [
            atom_index
            for atom_index in range(atom_count)
            if (
                assigned_gate_supports[atom_index] < minimum_gate_support
                or assigned_up_supports[atom_index] < minimum_up_support
                or assigned_up_directions[atom_index] is None
            )
        ]
    else:
        initial_up_norms = np.linalg.norm(np.asarray(candidate["ups"], dtype=np.float64), axis=1).clip(min=1e-12)
        initial_up_directions = np.asarray(candidate["ups"], dtype=np.float64) / initial_up_norms[:, None]
        current_directions, assigned_up_directions, fallback_atoms = apply_asymptotic_support_fallback(
            current_directions,
            assigned_up_directions,
            assigned_gate_supports,
            assigned_up_supports,
            initial_directions,
            initial_up_directions,
            minimum_gate_support,
            minimum_up_support,
        )

    stitched_gates = current_directions * gate_norms[:, None]
    independent_hessian_up_complete = all(
        direction is not None and assigned_up_supports[index] >= minimum_up_support
        for index, direction in enumerate(assigned_up_directions)
    )
    hessian_up_complete = (
        independent_hessian_up_complete
        if cold_start
        else all(direction is not None for direction in assigned_up_directions)
    )
    structure_snapshot_output = os.environ.get(
        "GCL_BLIND_SWIGLU_ASYMPTOTIC_STITCH_STRUCTURE_SNAPSHOT_OUTPUT",
        "",
    ).strip()
    if cold_start:
        oriented = solve_bidirectional_gate_orientation_start(
            coords,
            targets,
            current_directions,
            float(gate_norms[0]),
            sample_limit=int(os.environ.get(
                "GCL_BLIND_SWIGLU_ASYMPTOTIC_COLD_START_ORIENTATION_SAMPLES",
                "4096",
            ) or "4096"),
        )
        if oriented is None:
            diagnosed = dict(candidate)
            diagnosed.update({
                "asymptoticGateStitched": False,
                "asymptoticGateStitchColdStart": True,
                "asymptoticGateStitchFailure": "bidirectional-orientation-court-failed",
            })
            return diagnosed
        oriented_gates, linear_ups, _linear_scales, _linear_bias, _mae = oriented
        current_directions = oriented_gates / np.linalg.norm(oriented_gates, axis=1).clip(min=1e-12)[:, None]
        linear_up_norms = np.linalg.norm(linear_ups, axis=1).clip(min=1e-12)
        linear_up_directions = linear_ups / linear_up_norms[:, None]
        selected_up_directions: list[np.ndarray] = []
        for atom_index, recovered_up in enumerate(assigned_up_directions):
            if (
                recovered_up is None
                or assigned_gate_supports[atom_index] < minimum_gate_support
                or assigned_up_supports[atom_index] < minimum_up_support
            ):
                selected_up_directions.append(linear_up_directions[atom_index])
            else:
                selected_up_directions.append(np.asarray(recovered_up, dtype=np.float64))
        assigned_up_directions = selected_up_directions
        stitched_gates = current_directions * gate_norms[:, None]
        stitched_ups = np.vstack(assigned_up_directions)
        scalar_features = silu_np(coords @ stitched_gates.T) * (coords @ stitched_ups.T)
        scalar_design = np.column_stack([
            scalar_features,
            np.ones(coords.shape[0], dtype=np.float64),
        ])
        try:
            scalar_coefficients, *_ = np.linalg.lstsq(scalar_design, targets, rcond=None)
        except np.linalg.LinAlgError:
            scalar_coefficients = None
        if scalar_coefficients is None or not np.all(np.isfinite(scalar_coefficients)):
            return candidate
        solved_gates = stitched_gates
        solved_ups = stitched_ups
        solved_scales = scalar_coefficients[:-1]
        solved_bias = float(scalar_coefficients[-1])
    elif hessian_up_complete:
        stitched_ups = np.vstack(assigned_up_directions)
        scalar_features = silu_np(coords @ stitched_gates.T) * (coords @ stitched_ups.T)
        scalar_design = np.column_stack([
            scalar_features,
            np.ones(coords.shape[0], dtype=np.float64),
        ])
        try:
            scalar_coefficients, *_ = np.linalg.lstsq(scalar_design, targets, rcond=None)
        except np.linalg.LinAlgError:
            scalar_coefficients = None
        if scalar_coefficients is None or not np.all(np.isfinite(scalar_coefficients)):
            return candidate
        solved_gates = stitched_gates
        solved_ups = stitched_ups
        solved_scales = scalar_coefficients[:-1]
        solved_bias = float(scalar_coefficients[-1])
    else:
        solved = solve_fixed_gate_start(coords, targets, stitched_gates)
        if solved is None:
            return candidate
        solved_gates, solved_ups, solved_scales, solved_bias, _mae = solved
    if structure_snapshot_output and all(direction is not None for direction in assigned_up_directions):
        structure_snapshot_path = Path(structure_snapshot_output).expanduser().resolve()
        structure_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            structure_snapshot_path,
            gate_directions=current_directions,
            up_directions=np.vstack(assigned_up_directions),
            gate_supports=assigned_gate_supports,
            up_supports=assigned_up_supports,
            fallback_atoms=np.asarray([] if cold_start else fallback_atoms, dtype=np.int64),
            linear_identified_atoms=np.asarray(fallback_atoms if cold_start else [], dtype=np.int64),
            cold_start=np.asarray(cold_start, dtype=np.bool_),
        )
    trial = dict(candidate)
    trial.update({
        "gates": solved_gates,
        "ups": solved_ups,
        "scales": solved_scales,
        "bias": solved_bias,
    })
    base_metrics = regression_metrics(targets, predict_swiglu(coords, candidate))
    trial_metrics = regression_metrics(targets, predict_swiglu(coords, trial))
    trial.update({
        "mae": float(trial_metrics["mae"]),
        "mse": float(trial_metrics["mse"]),
        "r2": float(trial_metrics["r2"]),
        "score": float(trial_metrics["mae"]),
        "selectionScore": float(trial_metrics["mae"]),
        "worstError": float(trial_metrics["worstError"]),
    })
    structure_calibrated = False
    structure_calibration_enabled = os.environ.get(
        "GCL_BLIND_SWIGLU_ASYMPTOTIC_STITCH_STRUCTURE_CALIBRATION",
        "1" if hessian_up_complete else "0",
    ).strip() == "1"
    if structure_calibration_enabled:
        calibrated_trial = gate_norm_calibrate_swiglu_candidate(
            coords,
            targets,
            trial,
            selection_coords,
            selection_targets,
        )
        structure_calibrated = bool(calibrated_trial.get("gateNormCalibrated", False))
        trial = calibrated_trial
        trial_metrics = regression_metrics(targets, predict_swiglu(coords, trial))
    structure_refined = False
    structure_refine_enabled = os.environ.get(
        "GCL_BLIND_SWIGLU_ASYMPTOTIC_STITCH_STRUCTURE_REFINE",
        "1" if hessian_up_complete else "0",
    ).strip() == "1"
    if structure_refine_enabled:
        refined_trial = trust_region_structure_refine_swiglu_candidate(
            coords,
            targets,
            trial,
            selection_coords,
            selection_targets,
            gradient_coords,
            gradient_targets,
            cubic_directions,
            cubic_targets,
            max_tangent_override=(
                float(os.environ.get(
                    "GCL_BLIND_SWIGLU_ASYMPTOTIC_COLD_START_MAX_TANGENT",
                    "0.12",
                ) or "0.12")
                if cold_start
                else None
            ),
            gradient_weight_override=(0.0 if cold_start else None),
            cubic_weight_override=(0.0 if cold_start else None),
        )
        structure_refined = bool(refined_trial.get("structureTrustRegionAccepted", False))
        trial = refined_trial
        trial_metrics = regression_metrics(targets, predict_swiglu(coords, trial))
    value_polish_enabled = os.environ.get(
        "GCL_BLIND_SWIGLU_ASYMPTOTIC_STITCH_VALUE_POLISH",
        "0" if hessian_up_complete else "1",
    ).strip() == "1"
    value_polished = False
    if value_polish_enabled:
        polished_trial = fixed_gate_direction_value_polish(
            coords,
            targets,
            trial,
            None,
            None,
            0.0,
        )
        value_polished = bool(polished_trial.get("fixedGateDirectionValuePolished", False))
        trial = polished_trial
        trial_metrics = regression_metrics(targets, predict_swiglu(coords, trial))
    base_score = float(base_metrics["mae"]) + 0.25 * float(base_metrics["worstError"])
    trial_score = float(trial_metrics["mae"]) + 0.25 * float(trial_metrics["worstError"])
    accepted = trial_score < base_score
    refined = dict(trial if accepted else candidate)
    refined.update({
        "asymptoticGateStitched": bool(accepted),
        "asymptoticGateStitchLineCount": int(line_count),
        "asymptoticGateStitchUsableLineCount": int(len(line_rows)),
        "asymptoticGateStitchFailedLineCount": int(failed_lines),
        "asymptoticGateStitchSamplesPerLine": int(samples_per_line),
        "asymptoticGateStitchAsymptoticScale": float(asymptotic_scale),
        "asymptoticGateStitchBoundaryNormalCount": int(len(normal_directions)),
        "asymptoticGateStitchDetectedBoundaryCount": int(sum(
            len(row["boundaries"])
            for row in line_rows
        )),
        "asymptoticGateStitchClusterCount": int(len(clusters)),
        "asymptoticGateStitchClusterSupports": [int(item["support"]) for item in clusters],
        "asymptoticGateStitchGradientStep": float(gradient_step),
        "asymptoticGateStitchHessianUpComplete": bool(hessian_up_complete),
        "asymptoticGateStitchIndependentHessianUpComplete": bool(independent_hessian_up_complete),
        "asymptoticGateStitchHessianUpCount": int(sum(
            direction is not None
            for direction in recovered_up_centers
        )),
        "asymptoticGateStitchHessianUpSupports": recovered_up_supports,
        "asymptoticGateStitchAssignedGateSupports": assigned_gate_supports.tolist(),
        "asymptoticGateStitchAssignedUpSupports": assigned_up_supports.tolist(),
        "asymptoticGateStitchFallbackAtoms": ([] if cold_start else fallback_atoms),
        "asymptoticGateStitchLinearIdentifiedAtoms": (fallback_atoms if cold_start else []),
        "asymptoticGateStitchColdStart": bool(cold_start),
        "asymptoticGateStitchOrientationMethod": (
            "bidirectional-fixed-swiglu-variable-projection"
            if cold_start
            else "prior-candidate-alignment"
        ),
        "asymptoticGateStitchHessianGradientEstimator": "symmetric-richardson-coordinate",
        "asymptoticGateStitchHessianGradientSteps": (
            hessian_gradient_steps
            if hessian_probe_meta
            else []
        ),
        "asymptoticGateStitchValuePolished": bool(value_polished),
        "asymptoticGateStitchStructureCalibrated": bool(structure_calibrated),
        "asymptoticGateStitchStructureRefined": bool(structure_refined),
        "asymptoticGateStitchStructureRefineAttempted": bool(
            trial.get("structureTrustRegionAttempted", False)
        ),
        "asymptoticGateStitchStructureRefineSelectionScoreBefore": trial.get(
            "structureTrustRegionSelectionScoreBefore"
        ),
        "asymptoticGateStitchStructureRefineSelectionScoreAfter": trial.get(
            "structureTrustRegionSelectionScoreAfter"
        ),
        "asymptoticGateStitchStructureRefineRestarts": trial.get(
            "structureTrustRegionRestarts",
            [],
        ),
        "asymptoticGateStitchStructureCalibrationAttempted": bool(
            trial.get("gateNormCalibrationAttempted", False)
        ),
        "asymptoticGateStitchStructureCalibrationAttemptR2": trial.get(
            "gateNormCalibrationAttemptR2"
        ),
        "asymptoticGateStitchStructureCalibrationAttemptMae": trial.get(
            "gateNormCalibrationAttemptMae"
        ),
        "asymptoticGateStitchStructureCalibrationAttemptWorst": trial.get(
            "gateNormCalibrationAttemptWorst"
        ),
        "asymptoticGateStitchStructureCalibrationRestarts": trial.get(
            "gateNormCalibrationRestartSummaries",
            [],
        ),
        "asymptoticGateStitchStructureSnapshotOutput": (
            structure_snapshot_output
            if structure_snapshot_output and hessian_up_complete
            else None
        ),
        "asymptoticGateStitchRecoveredDirections": current_directions.tolist(),
        "asymptoticGateStitchRecoveredUpDirections": (
            np.vstack(assigned_up_directions).tolist()
            if hessian_up_complete
            else None
        ),
        "asymptoticGateStitchMeanDirectionDelta": (
            0.0
            if cold_start
            else float(np.mean(
                1.0 - np.abs(np.sum(current_directions * initial_directions, axis=1))
            ))
        ),
        "asymptoticGateStitchScoreBefore": float(base_score),
        "asymptoticGateStitchScoreAfter": float(trial_score),
        "asymptoticGateStitchTrialR2": float(trial_metrics["r2"]),
        "asymptoticGateStitchTrialMae": float(trial_metrics["mae"]),
        "asymptoticGateStitchTrialWorstError": float(trial_metrics["worstError"]),
    })
    if accepted:
        refined.update({
            "mae": float(trial_metrics["mae"]),
            "mse": float(trial_metrics["mse"]),
            "r2": float(trial_metrics["r2"]),
            "score": float(trial_metrics["mae"]),
            "selectionScore": float(trial_score),
            "worstError": float(trial_metrics["worstError"]),
        })
        if cold_start:
            refined.update({
                "restartCount": 0,
                "randomRestartCount": 0,
                "structuredStartCount": 1,
                "restartChunkSize": 0,
                "startKind": "asymptotic-cold-start",
                "fitMode": "asymptotic-structure-cold-start",
                "randomInitScales": [],
                "bestScoreByStartKind": {
                    "asymptotic-cold-start": float(trial_score),
                },
                "bestMetricsByStartKind": {
                    "asymptotic-cold-start": {
                        "r2": float(trial_metrics["r2"]),
                        "mae": float(trial_metrics["mae"]),
                        "worstError": float(trial_metrics["worstError"]),
                    },
                },
            })
    return refined


def residual_gradient_block_refit_swiglu_candidate(
    coords: np.ndarray,
    targets: np.ndarray,
    candidate: dict[str, Any],
    atom_indices: list[int],
    gradient_coords: np.ndarray,
    residual_gradient_targets: np.ndarray,
    seed: int,
    selection_coords: np.ndarray | None = None,
    selection_targets: np.ndarray | None = None,
    enable_cancellation_rescue: bool = True,
    fit_steps_override: int | None = None,
    fit_restarts_override: int | None = None,
    cubic_directions: np.ndarray | None = None,
    cubic_targets: np.ndarray | None = None,
    cubic_selection_directions: np.ndarray | None = None,
    cubic_selection_targets: np.ndarray | None = None,
) -> dict[str, Any]:
    gates = np.asarray(candidate["gates"], dtype=np.float64)
    ups = np.asarray(candidate["ups"], dtype=np.float64)
    scales = np.asarray(candidate["scales"], dtype=np.float64)
    if gates.ndim != 2 or ups.shape != gates.shape or scales.shape != (gates.shape[0],):
        return candidate
    atom_count, rank = gates.shape
    selected = sorted({int(index) for index in atom_indices if 0 <= int(index) < atom_count})
    keep = [index for index in range(atom_count) if index not in selected]
    if not selected or not keep:
        return candidate

    def atom_sum(points: np.ndarray, indices: list[int]) -> np.ndarray:
        if not indices:
            return np.zeros(points.shape[0], dtype=np.float64)
        activations = silu_np(points @ gates[indices].T) * (points @ ups[indices].T)
        return activations @ scales[indices]

    keep_fit = {
        "gates": gates[keep],
        "ups": ups[keep],
        "scales": scales[keep],
        "bias": 0.0,
    }
    residual_targets = targets - atom_sum(coords, keep)
    oracle_gradients = predict_swiglu_active_gradients(gradient_coords, candidate) + np.asarray(
        residual_gradient_targets,
        dtype=np.float64,
    )
    residual_gradients = oracle_gradients - predict_swiglu_active_gradients(gradient_coords, keep_fit)
    residual_selection_targets: np.ndarray | None = None
    if (
        selection_coords is not None
        and selection_targets is not None
        and selection_coords.shape[0] == selection_targets.shape[0]
        and selection_coords.shape[0] > 0
    ):
        residual_selection_targets = selection_targets - atom_sum(selection_coords, keep)
    has_cubic_witness = (
        cubic_directions is not None
        and cubic_targets is not None
        and cubic_directions.shape[0] > 0
        and cubic_directions.shape[0] == cubic_targets.shape[0]
    )
    residual_cubic_targets: np.ndarray | None = None
    if has_cubic_witness and cubic_directions is not None and cubic_targets is not None:
        residual_cubic_targets = cubic_targets - predict_swiglu_cubic_directional_derivatives(
            cubic_directions,
            gates[keep],
            ups[keep],
            scales[keep],
        )
    cubic_fit_weight, cubic_decision_weight = residual_block_refit_cubic_weights(
        has_cubic_witness
    )
    has_cubic_selection_witness = (
        cubic_selection_directions is not None
        and cubic_selection_targets is not None
        and cubic_selection_directions.shape[0] > 0
        and cubic_selection_directions.shape[0] == cubic_selection_targets.shape[0]
    )
    residual_cubic_selection_targets: np.ndarray | None = None
    if (
        has_cubic_selection_witness
        and cubic_selection_directions is not None
        and cubic_selection_targets is not None
    ):
        residual_cubic_selection_targets = (
            cubic_selection_targets
            - predict_swiglu_cubic_directional_derivatives(
                cubic_selection_directions,
                gates[keep],
                ups[keep],
                scales[keep],
            )
        )

    direction_limit = max(128, len(selected) * 96)
    residual_directions = make_gradient_gate_directions(
        gradient_coords,
        residual_gradients,
        len(selected),
        seed + 19,
        direction_limit,
    )
    gate_seed_directions = np.vstack(residual_directions) if residual_directions else None
    warm_start = [(
        gates[selected].copy(),
        ups[selected].copy(),
        scales[selected].copy(),
        float(candidate.get("bias", 0.0)),
    )]
    fit_steps = max(200, int(
        fit_steps_override
        if fit_steps_override is not None
        else os.environ.get("GCL_BLIND_SWIGLU_RESIDUAL_BLOCK_REFIT_STEPS", "1800") or "1800"
    ))
    fit_restarts = max(16, int(
        fit_restarts_override
        if fit_restarts_override is not None
        else os.environ.get(
            "GCL_BLIND_SWIGLU_RESIDUAL_BLOCK_REFIT_RESTARTS",
            str(max(192, len(selected) * 64)),
        ) or "0"
    ))
    fit_learning_rate = float(os.environ.get(
        "GCL_BLIND_SWIGLU_RESIDUAL_BLOCK_REFIT_LEARNING_RATE",
        "0.008",
    ) or "0.008")
    fit_gradient_weight = float(os.environ.get(
        "GCL_BLIND_SWIGLU_RESIDUAL_BLOCK_REFIT_GRADIENT_WEIGHT",
        "1.0",
    ) or "1.0")
    fit_chunk = max(1, int(os.environ.get(
        "GCL_BLIND_SWIGLU_RESIDUAL_BLOCK_REFIT_CHUNK_SIZE",
        "96",
    ) or "96"))
    residual_fit = fit_swiglu_sum(
        coords,
        residual_targets,
        len(selected),
        seed + 31,
        fit_steps,
        fit_learning_rate,
        fit_restarts,
        gradient_coords,
        residual_gradients,
        gate_seed_directions,
        fit_gradient_weight,
        "batched",
        fit_chunk,
        selection_coords,
        residual_selection_targets,
        warm_start,
        cubic_selection_directions=cubic_selection_directions,
        cubic_selection_scores=residual_cubic_selection_targets,
        cubic_directions=cubic_directions,
        cubic_targets=residual_cubic_targets,
        cubic_weight_override=cubic_fit_weight,
        selection_cubic_weight_override=cubic_decision_weight,
    )

    combined_gates = gates.copy()
    for local_index, atom_index in enumerate(selected):
        combined_gates[atom_index] = np.asarray(residual_fit["gates"], dtype=np.float64)[local_index]
    solved = solve_fixed_gate_start(coords, targets, combined_gates)
    if solved is None:
        return candidate
    solved_gates, solved_ups, solved_scales, solved_bias, _mae = solved
    trial = dict(candidate)
    trial.update({
        "gates": solved_gates,
        "ups": solved_ups,
        "scales": solved_scales,
        "bias": solved_bias,
    })
    # This refit is where atoms collapse. It takes the weak block - on the randomized LLaMA
    # layer-15 case, atoms [2, 5, 7, 8, 9, 12, 14, 15], which contains both members of the
    # measured collapsed pair - and fits them together against one residual, where nothing
    # stops two of them settling on the same direction. Running the respread pass only in the
    # initial fit left this untouched: the earlier fit had no collapsed pair to find, and the
    # numbers came back bit-identical.
    trial = respread_collapsed_atoms_swiglu_candidate(coords, targets, trial)
    base = rescore_swiglu_candidate_on_data(coords, targets, candidate, recalibrate_scales=True)
    base_metrics = regression_metrics(targets, predict_swiglu(coords, base))
    trial_metrics = regression_metrics(targets, predict_swiglu(coords, trial))
    base_score = float(base_metrics["mae"]) + 0.25 * float(base_metrics["worstError"])
    trial_score = float(trial_metrics["mae"]) + 0.25 * float(trial_metrics["worstError"])
    base_output_score = base_score
    trial_output_score = trial_score
    selection_count = 0
    if residual_selection_targets is not None and selection_coords is not None and selection_targets is not None:
        selection_count = int(selection_coords.shape[0])
        base_selection_metrics = regression_metrics(selection_targets, predict_swiglu(selection_coords, base))
        trial_selection_metrics = regression_metrics(selection_targets, predict_swiglu(selection_coords, trial))
        base_score = float(base_selection_metrics["mae"]) + 0.35 * float(base_selection_metrics["worstError"])
        trial_score = float(trial_selection_metrics["mae"]) + 0.35 * float(trial_selection_metrics["worstError"])
        base_output_score = base_score
        trial_output_score = trial_score

    def cubic_relative_rmse(model: dict[str, Any]) -> float | None:
        score_directions = (
            cubic_selection_directions
            if has_cubic_selection_witness
            else cubic_directions
        )
        score_targets = (
            cubic_selection_targets
            if has_cubic_selection_witness
            else cubic_targets
        )
        if score_directions is None or score_targets is None:
            return None
        # Same emptiness case as the fit-path scorer: no witness means None, not NaN.
        if score_directions.shape[0] == 0 or score_targets.shape[0] == 0:
            return None
        predicted = predict_swiglu_cubic_directional_derivatives(
            score_directions,
            np.asarray(model["gates"], dtype=np.float64),
            np.asarray(model["ups"], dtype=np.float64),
            np.asarray(model["scales"], dtype=np.float64),
        )
        denominator = max(float(np.mean(score_targets ** 2)), 1e-12)
        return math.sqrt(float(np.mean((predicted - score_targets) ** 2)) / denominator)

    base_cubic_rmse = cubic_relative_rmse(base)
    trial_cubic_rmse = cubic_relative_rmse(trial)
    if (
        base_cubic_rmse is not None
        and trial_cubic_rmse is not None
        and cubic_decision_weight > 0.0
    ):
        base_score += cubic_decision_weight * base_cubic_rmse
        trial_score += cubic_decision_weight * trial_cubic_rmse

    cancellation_pairs_before = detect_swiglu_cancellation_pairs(coords, trial)
    cancellation_pairs_after = cancellation_pairs_before
    cancellation_rescue_attempted = False
    cancellation_rescued = False
    cancellation_rescue_atoms: list[int] = []
    cancellation_rescue_score_before = trial_score
    cancellation_rescue_score_after = trial_score
    cancellation_rescue_attempts: list[dict[str, Any]] = []
    cancellation_rescue_enabled = (
        enable_cancellation_rescue
        and os.environ.get(
            "GCL_BLIND_SWIGLU_RESIDUAL_BLOCK_REFIT_CANCELLATION_RESCUE",
            "1",
        ).strip() == "1"
    )
    if cancellation_rescue_enabled and cancellation_pairs_before:
        cancellation_rescue_attempted = True
        rescue_steps = max(200, int(os.environ.get(
            "GCL_BLIND_SWIGLU_RESIDUAL_BLOCK_REFIT_CANCELLATION_RESCUE_STEPS",
            str(min(fit_steps, 1800)),
        ) or "0"))
        rescue_restarts = max(16, int(os.environ.get(
            "GCL_BLIND_SWIGLU_RESIDUAL_BLOCK_REFIT_CANCELLATION_RESCUE_RESTARTS",
            str(min(fit_restarts, 256)),
        ) or "0"))
        rescue_pair_limit = max(1, int(os.environ.get(
            "GCL_BLIND_SWIGLU_RESIDUAL_BLOCK_REFIT_CANCELLATION_RESCUE_PAIR_LIMIT",
            "4",
        ) or "4"))
        attempted_pairs: set[tuple[int, int]] = set()

        def cancellation_pair_score(
            pairs: list[dict[str, Any]],
            atoms: list[int],
        ) -> float:
            key = tuple(sorted(int(index) for index in atoms))
            for pair in pairs:
                if tuple(sorted(int(index) for index in pair["atoms"])) == key:
                    return float(pair["score"])
            return 0.0

        while len(cancellation_rescue_attempts) < rescue_pair_limit:
            current_pairs = detect_swiglu_cancellation_pairs(coords, trial)
            pair = next((
                item for item in current_pairs
                if tuple(sorted(int(index) for index in item["atoms"])) not in attempted_pairs
            ), None)
            if pair is None:
                break
            pair_atoms = [int(index) for index in pair["atoms"]]
            pair_key = tuple(sorted(pair_atoms))
            attempted_pairs.add(pair_key)
            cancellation_rescue_atoms.extend(
                index for index in pair_atoms
                if index not in cancellation_rescue_atoms
            )
            trial_residual_gradient_targets = oracle_gradients - predict_swiglu_active_gradients(
                gradient_coords,
                trial,
            )
            rescued_trial = residual_gradient_block_refit_swiglu_candidate(
                coords,
                targets,
                trial,
                pair_atoms,
                gradient_coords,
                trial_residual_gradient_targets,
                seed + 104729 + len(cancellation_rescue_attempts) * 1009,
                selection_coords,
                selection_targets,
                enable_cancellation_rescue=False,
                fit_steps_override=rescue_steps,
                fit_restarts_override=rescue_restarts,
                cubic_directions=cubic_directions,
                cubic_targets=cubic_targets,
                cubic_selection_directions=cubic_selection_directions,
                cubic_selection_targets=cubic_selection_targets,
            )
            rescued_metrics = regression_metrics(targets, predict_swiglu(coords, rescued_trial))
            rescued_output_score = (
                float(rescued_metrics["mae"])
                + 0.25 * float(rescued_metrics["worstError"])
            )
            rescued_score = rescued_output_score
            if (
                residual_selection_targets is not None
                and selection_coords is not None
                and selection_targets is not None
            ):
                rescued_selection_metrics = regression_metrics(
                    selection_targets,
                    predict_swiglu(selection_coords, rescued_trial),
                )
                rescued_output_score = (
                    float(rescued_selection_metrics["mae"])
                    + 0.35 * float(rescued_selection_metrics["worstError"])
                )
                rescued_score = rescued_output_score
            rescued_cubic_rmse = cubic_relative_rmse(rescued_trial)
            if rescued_cubic_rmse is not None and cubic_decision_weight > 0.0:
                rescued_score += cubic_decision_weight * rescued_cubic_rmse
            rescued_pairs = detect_swiglu_cancellation_pairs(coords, rescued_trial)
            pair_score_before = float(pair["score"])
            pair_score_after = cancellation_pair_score(rescued_pairs, pair_atoms)
            accepted_pair = (
                bool(rescued_trial.get("residualGradientBlockRefit"))
                and rescued_score < trial_score
                and pair_score_after < pair_score_before * 0.5
            )
            cancellation_rescue_attempts.append({
                "atoms": pair_atoms,
                "accepted": bool(accepted_pair),
                "decisionScoreBefore": float(trial_score),
                "decisionScoreAfter": float(rescued_score),
                "pairScoreBefore": pair_score_before,
                "pairScoreAfter": pair_score_after,
            })
            cancellation_rescue_score_after = rescued_score
            if accepted_pair:
                cancellation_rescued = True
                trial = rescued_trial
                trial_metrics = rescued_metrics
                trial_score = rescued_score
                trial_output_score = rescued_output_score
                trial_cubic_rmse = rescued_cubic_rmse
                cancellation_pairs_after = rescued_pairs
        cancellation_pairs_after = detect_swiglu_cancellation_pairs(coords, trial)
    accepted = (
        trial_score < base_score
        and (
            trial_output_score <= base_output_score * 1.05
            or (
                base_cubic_rmse is not None
                and trial_cubic_rmse is not None
                and trial_cubic_rmse < base_cubic_rmse
                and trial_metrics["mae"] <= base_metrics["mae"] * 1.05
                and trial_metrics["worstError"] <= base_metrics["worstError"] * 1.05
            )
        )
    )
    refined = dict(trial if accepted else candidate)
    refined.update({
        "residualGradientBlockRefit": bool(accepted),
        "residualGradientBlockRefitAtoms": selected,
        "residualGradientBlockRefitDirectionCount": int(len(residual_directions)),
        "residualGradientBlockRefitRestarts": int(fit_restarts),
        "residualGradientBlockRefitSteps": int(fit_steps),
        "residualGradientBlockRefitSelectionCount": int(selection_count),
        "residualGradientBlockRefitScoreBefore": float(base_score),
        "residualGradientBlockRefitScoreAfter": float(trial_score),
        "residualGradientBlockRefitOutputScoreBefore": float(base_output_score),
        "residualGradientBlockRefitOutputScoreAfter": float(trial_output_score),
        "residualGradientBlockRefitCubicRelativeRmseBefore": base_cubic_rmse,
        "residualGradientBlockRefitCubicRelativeRmseAfter": trial_cubic_rmse,
        "residualGradientBlockRefitCubicSelectionWeight": float(cubic_decision_weight),
        "residualGradientBlockRefitCubicFitWeight": float(cubic_fit_weight),
        "residualGradientBlockRefitCubicDirectionCount": int(
            0 if cubic_directions is None else cubic_directions.shape[0]
        ),
        "residualGradientBlockRefitCubicSelectionDirectionCount": int(
            0 if cubic_selection_directions is None else cubic_selection_directions.shape[0]
        ),
        "residualGradientBlockRefitMaeBefore": float(base_metrics["mae"]),
        "residualGradientBlockRefitWorstBefore": float(base_metrics["worstError"]),
        "residualGradientBlockRefitCancellationPairsBefore": cancellation_pairs_before,
        "residualGradientBlockRefitCancellationPairsAfter": cancellation_pairs_after,
        "residualGradientBlockRefitCancellationRescueAttempted": bool(cancellation_rescue_attempted),
        "residualGradientBlockRefitCancellationRescued": bool(cancellation_rescued),
        "residualGradientBlockRefitCancellationRescueAtoms": cancellation_rescue_atoms,
        "residualGradientBlockRefitCancellationRescueScoreBefore": float(cancellation_rescue_score_before),
        "residualGradientBlockRefitCancellationRescueScoreAfter": float(cancellation_rescue_score_after),
        "residualGradientBlockRefitCancellationRescueAttempts": cancellation_rescue_attempts,
    })
    if accepted:
        refined.update({
            "mae": float(trial_metrics["mae"]),
            "mse": float(trial_metrics["mse"]),
            "score": float(trial_metrics["mae"]),
            "selectionScore": float(trial_score),
            "worstError": float(trial_metrics["worstError"]),
            "cubicRelativeRmse": trial_cubic_rmse,
        })
    return refined


def sequential_cancellation_pair_refit_swiglu_candidate(
    coords: np.ndarray,
    targets: np.ndarray,
    candidate: dict[str, Any],
    gradient_coords: np.ndarray,
    oracle_gradient_targets: np.ndarray,
    seed: int,
    selection_coords: np.ndarray | None = None,
    selection_targets: np.ndarray | None = None,
    cubic_directions: np.ndarray | None = None,
    cubic_targets: np.ndarray | None = None,
    cubic_selection_directions: np.ndarray | None = None,
    cubic_selection_targets: np.ndarray | None = None,
) -> dict[str, Any]:
    current = dict(candidate)
    initial_pairs = detect_swiglu_cancellation_pairs(coords, current)
    pair_limit = max(1, int(os.environ.get(
        "GCL_BLIND_SWIGLU_RESIDUAL_BLOCK_REFIT_CANCELLATION_RESCUE_PAIR_LIMIT",
        "4",
    ) or "4"))
    fit_steps = max(200, int(os.environ.get(
        "GCL_BLIND_SWIGLU_RESIDUAL_BLOCK_REFIT_CANCELLATION_RESCUE_STEPS",
        "1800",
    ) or "1800"))
    fit_restarts = max(16, int(os.environ.get(
        "GCL_BLIND_SWIGLU_RESIDUAL_BLOCK_REFIT_CANCELLATION_RESCUE_RESTARTS",
        "256",
    ) or "256"))
    attempts: list[dict[str, Any]] = []
    attempted_pairs: set[tuple[int, int]] = set()
    rescued_any = False

    while len(attempts) < pair_limit:
        current_pairs = detect_swiglu_cancellation_pairs(coords, current)
        pair = next((
            item for item in current_pairs
            if tuple(sorted(int(index) for index in item["atoms"])) not in attempted_pairs
        ), None)
        if pair is None:
            break
        atoms = [int(index) for index in pair["atoms"]]
        pair_key = tuple(sorted(atoms))
        attempted_pairs.add(pair_key)
        residual_gradient_targets = (
            np.asarray(oracle_gradient_targets, dtype=np.float64)
            - predict_swiglu_active_gradients(gradient_coords, current)
        )
        trial = residual_gradient_block_refit_swiglu_candidate(
            coords,
            targets,
            current,
            atoms,
            gradient_coords,
            residual_gradient_targets,
            seed + len(attempts) * 1009,
            selection_coords,
            selection_targets,
            enable_cancellation_rescue=False,
            fit_steps_override=fit_steps,
            fit_restarts_override=fit_restarts,
            cubic_directions=cubic_directions,
            cubic_targets=cubic_targets,
            cubic_selection_directions=cubic_selection_directions,
            cubic_selection_targets=cubic_selection_targets,
        )
        trial_pairs = detect_swiglu_cancellation_pairs(coords, trial)
        pair_score_after = next((
            float(item["score"])
            for item in trial_pairs
            if tuple(sorted(int(index) for index in item["atoms"])) == pair_key
        ), 0.0)
        accepted = (
            bool(trial.get("residualGradientBlockRefit"))
            and pair_score_after < float(pair["score"]) * 0.5
        )
        attempts.append({
            "atoms": atoms,
            "accepted": bool(accepted),
            "pairScoreBefore": float(pair["score"]),
            "pairScoreAfter": pair_score_after,
            "decisionScoreBefore": current.get("selectionScore"),
            "decisionScoreAfter": trial.get("selectionScore"),
        })
        if accepted:
            rescued_any = True
            current = trial

    result = dict(current)
    result.update({
        "residualGradientBlockRefit": bool(rescued_any),
        "residualGradientBlockRefitAtoms": sorted({
            atom
            for attempt in attempts
            for atom in attempt["atoms"]
        }),
        "residualGradientBlockRefitCancellationPairsBefore": initial_pairs,
        "residualGradientBlockRefitCancellationPairsAfter": detect_swiglu_cancellation_pairs(
            coords,
            current,
        ),
        "residualGradientBlockRefitCancellationRescueAttempted": bool(attempts),
        "residualGradientBlockRefitCancellationRescued": bool(rescued_any),
        "residualGradientBlockRefitCancellationRescueAtoms": sorted({
            atom
            for attempt in attempts
            for atom in attempt["atoms"]
        }),
        "residualGradientBlockRefitCancellationRescueAttempts": attempts,
        "residualGradientBlockRefitRestarts": int(fit_restarts),
        "residualGradientBlockRefitSteps": int(fit_steps),
    })
    return result


def minimax_bias_polish_swiglu_candidate(
    coords: np.ndarray,
    targets: np.ndarray,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    prediction = predict_swiglu(coords, candidate)
    residual = targets - prediction
    if residual.shape[0] == 0 or not np.all(np.isfinite(residual)):
        return candidate
    shift = (float(np.max(residual)) + float(np.min(residual))) / 2.0
    if not np.isfinite(shift) or abs(shift) < 1e-15:
        return candidate
    polished = dict(candidate)
    polished["bias"] = float(candidate["bias"]) + shift
    polished_prediction = prediction + shift
    polished_residual = targets - polished_prediction
    mae = float(np.mean(np.abs(polished_residual)))
    mse = float(np.mean(polished_residual ** 2))
    worst = float(np.max(np.abs(polished_residual)))
    previous_worst = float(candidate.get("worstError", float(np.max(np.abs(residual)))))
    if worst > previous_worst:
        return candidate
    polished.update({
        "mae": mae,
        "mse": mse,
        "score": mae,
        "worstError": worst,
        "minimaxBiasPolished": True,
        "minimaxBiasShift": float(shift),
        "minimaxBiasWorstBefore": previous_worst,
    })
    return polished


def affine_chebyshev_calibrate_swiglu_candidate(
    coords: np.ndarray,
    targets: np.ndarray,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    prediction = predict_swiglu(coords, candidate)
    if prediction.shape[0] == 0 or not np.all(np.isfinite(prediction)):
        return candidate
    previous_metrics = regression_metrics(targets, prediction)
    span = float(os.environ.get("GCL_BLIND_SWIGLU_AFFINE_CHEBYSHEV_SCALE_SPAN", "0.08") or "0.08")
    steps = int(os.environ.get("GCL_BLIND_SWIGLU_AFFINE_CHEBYSHEV_STEPS", "321") or "321")
    mae_weight = float(os.environ.get("GCL_BLIND_SWIGLU_AFFINE_CHEBYSHEV_MAE_WEIGHT", "1.5") or "1.5")
    max_mae_ratio = float(os.environ.get("GCL_BLIND_SWIGLU_AFFINE_CHEBYSHEV_MAX_MAE_RATIO", "1.04") or "1.04")
    if not np.isfinite(span) or span < 0.0:
        raise RuntimeError("GCL_BLIND_SWIGLU_AFFINE_CHEBYSHEV_SCALE_SPAN must be non-negative and finite")
    if steps < 1:
        raise RuntimeError("GCL_BLIND_SWIGLU_AFFINE_CHEBYSHEV_STEPS must be positive")
    if not np.isfinite(mae_weight) or mae_weight < 0.0:
        raise RuntimeError("GCL_BLIND_SWIGLU_AFFINE_CHEBYSHEV_MAE_WEIGHT must be non-negative and finite")
    if not np.isfinite(max_mae_ratio) or max_mae_ratio < 1.0:
        raise RuntimeError("GCL_BLIND_SWIGLU_AFFINE_CHEBYSHEV_MAX_MAE_RATIO must be >= 1")
    scales = np.linspace(1.0 - span, 1.0 + span, steps, dtype=np.float64)
    best: tuple[float, float, float, dict[str, float]] | None = None
    for scale in scales:
        residual = targets - scale * prediction
        shift = (float(np.max(residual)) + float(np.min(residual))) / 2.0
        adjusted = scale * prediction + shift
        metrics = regression_metrics(targets, adjusted)
        if metrics["mae"] > previous_metrics["mae"] * max_mae_ratio:
            continue
        score = float(metrics["worstError"]) + mae_weight * float(metrics["mae"])
        if best is None or score < best[0]:
            best = (score, float(scale), float(shift), metrics)
    if best is None:
        return candidate
    score, scale, shift, metrics = best
    if (
        metrics["worstError"] >= previous_metrics["worstError"]
        and metrics["mae"] >= previous_metrics["mae"]
    ):
        return candidate
    calibrated = dict(candidate)
    calibrated.update({
        "scales": np.asarray(candidate["scales"], dtype=np.float64) * scale,
        "bias": float(candidate["bias"]) * scale + shift,
        "mae": float(metrics["mae"]),
        "mse": float(metrics["mse"]),
        "score": float(metrics["mae"]),
        "selectionScore": float(metrics["mae"]),
        "worstError": float(metrics["worstError"]),
        "affineChebyshevCalibrated": True,
        "affineChebyshevScale": float(scale),
        "affineChebyshevShift": float(shift),
        "affineChebyshevScore": float(score),
        "affineChebyshevMaeBefore": float(previous_metrics["mae"]),
        "affineChebyshevWorstBefore": float(previous_metrics["worstError"]),
    })
    return calibrated


def chebyshev_calibrate_swiglu_candidate(
    coords: np.ndarray,
    targets: np.ndarray,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    try:
        from scipy.optimize import linprog
    except ImportError:
        # scipy absent is the one survivable outcome; an error raised while importing an scipy
        # that is present is a broken environment and should say so rather than read as decline.
        _record_fallback("chebyshev calibration skipped: scipy.optimize.linprog unavailable")
        return candidate
    gates_np = np.asarray(candidate["gates"], dtype=np.float64)
    ups_np = np.asarray(candidate["ups"], dtype=np.float64)
    if gates_np.ndim != 2 or ups_np.shape != gates_np.shape:
        return candidate
    features = silu_np(coords @ gates_np.T) * (coords @ ups_np.T)
    if features.shape[0] == 0 or not np.all(np.isfinite(features)):
        return candidate
    sample_limit = int(os.environ.get("GCL_BLIND_SWIGLU_CHEBYSHEV_SAMPLE_LIMIT", "12000") or "12000")
    if sample_limit > 0 and features.shape[0] > sample_limit:
        rng = np.random.default_rng(int(os.environ.get("GCL_BLIND_SWIGLU_CHEBYSHEV_SEED", "91703") or "91703"))
        indices = np.sort(rng.choice(features.shape[0], size=sample_limit, replace=False))
        fit_features = features[indices]
        fit_targets = targets[indices]
    else:
        fit_features = features
        fit_targets = targets
    row_count, atom_count = fit_features.shape
    # Variables are atom scales, bias, and non-negative max error t.
    objective = np.zeros(atom_count + 2, dtype=np.float64)
    objective[-1] = 1.0
    upper = np.zeros((row_count * 2, atom_count + 2), dtype=np.float64)
    upper[:row_count, :atom_count] = fit_features
    upper[:row_count, atom_count] = 1.0
    upper[:row_count, -1] = -1.0
    upper[row_count:, :atom_count] = -fit_features
    upper[row_count:, atom_count] = -1.0
    upper[row_count:, -1] = -1.0
    bounds = [(None, None)] * (atom_count + 1) + [(0.0, None)]
    rhs = np.concatenate([fit_targets, -fit_targets]).astype(np.float64)
    try:
        result = linprog(
            objective,
            A_ub=upper,
            b_ub=rhs,
            bounds=bounds,
            method="highs",
            options={"presolve": True},
        )
    except ValueError as exc:
        # linprog rejects a program it cannot read - wrong shapes, non-finite bounds. That is a
        # calibration that could not run, which is worth knowing, and it is not the same event
        # as the solver running and declining to improve the candidate.
        _record_fallback(f"chebyshev calibration skipped: linprog rejected the program: {exc}")
        return candidate
    if not result.success or result.x is None or not np.all(np.isfinite(result.x)):
        return candidate
    scales = result.x[:atom_count].astype(np.float64)
    bias = float(result.x[atom_count])
    prediction = features @ scales + bias
    metrics = regression_metrics(targets, prediction)
    previous_worst = float(candidate.get("worstError", float("inf")))
    previous_mae = float(candidate.get("mae", float("inf")))
    # Chebyshev calibration is allowed to trade tiny MAE for lower max error,
    # but not to materially degrade average fit.
    max_mae_ratio = float(os.environ.get("GCL_BLIND_SWIGLU_CHEBYSHEV_MAX_MAE_RATIO", "1.08") or "1.08")
    if metrics["worstError"] > previous_worst or metrics["mae"] > previous_mae * max_mae_ratio:
        return candidate
    calibrated = dict(candidate)
    calibrated.update({
        "mae": float(metrics["mae"]),
        "mse": float(metrics["mse"]),
        "score": float(metrics["mae"]),
        "worstError": float(metrics["worstError"]),
        "scales": scales,
        "bias": bias,
        "chebyshevCalibrated": True,
        "chebyshevObjective": float(result.fun),
        "chebyshevSampleCount": int(row_count),
        "chebyshevWorstBefore": previous_worst,
    })
    return calibrated


def robust_scalar_calibrate_swiglu_candidate(
    coords: np.ndarray,
    targets: np.ndarray,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    import torch

    gates_np = np.asarray(candidate["gates"], dtype=np.float64)
    ups_np = np.asarray(candidate["ups"], dtype=np.float64)
    scales_np = np.asarray(candidate["scales"], dtype=np.float64)
    if gates_np.ndim != 2 or ups_np.shape != gates_np.shape or scales_np.shape[0] != gates_np.shape[0]:
        return candidate
    features_np = silu_np(coords @ gates_np.T) * (coords @ ups_np.T)
    if features_np.shape[0] == 0 or not np.all(np.isfinite(features_np)):
        return candidate
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    features = torch.tensor(features_np, dtype=dtype, device=device)
    y = torch.tensor(targets, dtype=dtype, device=device)
    scales = torch.nn.Parameter(torch.tensor(scales_np, dtype=dtype, device=device))
    bias = torch.nn.Parameter(torch.tensor(float(candidate["bias"]), dtype=dtype, device=device))
    steps = int(os.environ.get("GCL_BLIND_SWIGLU_SCALAR_CALIBRATION_STEPS", "900") or "900")
    lr = float(os.environ.get("GCL_BLIND_SWIGLU_SCALAR_CALIBRATION_LR", "0.01") or "0.01")
    tail_weight = float(os.environ.get("GCL_BLIND_SWIGLU_SCALAR_CALIBRATION_TAIL_WEIGHT", "1.0") or "1.0")
    tail_fraction = float(os.environ.get("GCL_BLIND_SWIGLU_SCALAR_CALIBRATION_TAIL_FRACTION", "0.20") or "0.20")
    if steps <= 0 or not np.isfinite(lr) or lr <= 0.0:
        return candidate
    tail_count = max(64, min(features_np.shape[0], int(features_np.shape[0] * tail_fraction)))
    optimizer = torch.optim.AdamW([scales, bias], lr=lr, weight_decay=1e-8)
    best_state: tuple[float, np.ndarray, float] | None = None
    for step_index in range(steps):
        optimizer.zero_grad()
        prediction = features @ scales + bias
        residual = prediction - y
        mse_loss = torch.mean(residual ** 2)
        tail_values = torch.topk(torch.abs(residual), k=tail_count, largest=True).values
        loss = mse_loss + tail_weight * torch.mean(tail_values ** 2)
        loss.backward()
        optimizer.step()
        if step_index == 0 or (step_index + 1) % 60 == 0:
            with torch.no_grad():
                current_prediction = features @ scales + bias
                score = float(torch.max(torch.abs(current_prediction - y)).detach().cpu().item())
                if best_state is None or score < best_state[0]:
                    best_state = (
                        score,
                        scales.detach().cpu().numpy().astype(np.float64),
                        float(bias.detach().cpu().item()),
                    )
    if best_state is None:
        return candidate
    _score, calibrated_scales, calibrated_bias = best_state
    prediction_np = features_np @ calibrated_scales + calibrated_bias
    metrics = regression_metrics(targets, prediction_np)
    previous_worst = float(candidate.get("worstError", float("inf")))
    previous_mae = float(candidate.get("mae", float("inf")))
    max_mae_ratio = float(os.environ.get("GCL_BLIND_SWIGLU_SCALAR_CALIBRATION_MAX_MAE_RATIO", "1.08") or "1.08")
    if metrics["worstError"] > previous_worst or metrics["mae"] > previous_mae * max_mae_ratio:
        return candidate
    calibrated = dict(candidate)
    calibrated.update({
        "mae": float(metrics["mae"]),
        "mse": float(metrics["mse"]),
        "score": float(metrics["mae"]),
        "worstError": float(metrics["worstError"]),
        "scales": calibrated_scales,
        "bias": calibrated_bias,
        "robustScalarCalibrated": True,
        "robustScalarCalibrationSteps": int(steps),
        "robustScalarCalibrationTailWeight": float(tail_weight),
        "robustScalarCalibrationTailFraction": float(tail_fraction),
        "robustScalarCalibrationWorstBefore": previous_worst,
    })
    return calibrated


def gate_norm_calibrate_swiglu_candidate(
    coords: np.ndarray,
    targets: np.ndarray,
    candidate: dict[str, Any],
    selection_coords: np.ndarray | None = None,
    selection_targets: np.ndarray | None = None,
) -> dict[str, Any]:
    import torch

    gates_np = np.asarray(candidate["gates"], dtype=np.float64)
    ups_np = np.asarray(candidate["ups"], dtype=np.float64)
    scales_np = np.asarray(candidate["scales"], dtype=np.float64)
    if gates_np.ndim != 2 or ups_np.shape != gates_np.shape or scales_np.shape[0] != gates_np.shape[0]:
        return candidate
    if coords.shape[0] < max(256, gates_np.shape[0] * 16):
        return candidate
    gate_norms = np.linalg.norm(gates_np, axis=1).clip(min=1e-9)
    gate_dirs = gates_np / gate_norms[:, None]
    up_norms = np.linalg.norm(ups_np, axis=1).clip(min=1e-12)
    up_dirs = ups_np / up_norms[:, None]
    effective_scales = scales_np * up_norms
    if not (
        np.all(np.isfinite(gate_dirs))
        and np.all(np.isfinite(up_dirs))
        and np.all(np.isfinite(effective_scales))
    ):
        return candidate

    steps = int(os.environ.get("GCL_BLIND_SWIGLU_GATE_NORM_CALIBRATION_STEPS", "160") or "160")
    lr = float(os.environ.get("GCL_BLIND_SWIGLU_GATE_NORM_CALIBRATION_LR", "0.8") or "0.8")
    restart_count = int(os.environ.get("GCL_BLIND_SWIGLU_GATE_NORM_CALIBRATION_RESTARTS", "4") or "4")
    tail_weight = float(os.environ.get("GCL_BLIND_SWIGLU_GATE_NORM_CALIBRATION_TAIL_WEIGHT", "0.0") or "0.0")
    tail_fraction = float(os.environ.get("GCL_BLIND_SWIGLU_GATE_NORM_CALIBRATION_TAIL_FRACTION", "0.12") or "0.12")
    mae_weight = float(os.environ.get("GCL_BLIND_SWIGLU_GATE_NORM_CALIBRATION_MAE_WEIGHT", "0.0") or "0.0")
    best_mae_weight = float(os.environ.get("GCL_BLIND_SWIGLU_GATE_NORM_CALIBRATION_BEST_MAE_WEIGHT", "2.0") or "2.0")
    max_norm_ratio = float(os.environ.get("GCL_BLIND_SWIGLU_GATE_NORM_CALIBRATION_MAX_NORM_RATIO", "6.0") or "6.0")
    if steps <= 0 or not np.isfinite(lr) or lr <= 0.0 or restart_count <= 0:
        raise RuntimeError("invalid GCL_BLIND_SWIGLU_GATE_NORM_CALIBRATION_STEPS/LR")
    for name, value in [
        ("GCL_BLIND_SWIGLU_GATE_NORM_CALIBRATION_TAIL_WEIGHT", tail_weight),
        ("GCL_BLIND_SWIGLU_GATE_NORM_CALIBRATION_MAE_WEIGHT", mae_weight),
        ("GCL_BLIND_SWIGLU_GATE_NORM_CALIBRATION_BEST_MAE_WEIGHT", best_mae_weight),
    ]:
        if not np.isfinite(value) or value < 0.0:
            raise RuntimeError(f"{name} must be non-negative and finite")
    if not np.isfinite(tail_fraction) or tail_fraction <= 0.0 or tail_fraction > 1.0:
        raise RuntimeError("GCL_BLIND_SWIGLU_GATE_NORM_CALIBRATION_TAIL_FRACTION must be in (0, 1]")
    if not np.isfinite(max_norm_ratio) or max_norm_ratio < 1.0:
        raise RuntimeError("GCL_BLIND_SWIGLU_GATE_NORM_CALIBRATION_MAX_NORM_RATIO must be >= 1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    target_scale = max(
        float(np.std(targets)),
        float(np.mean(np.abs(targets))),
        1e-8,
    )
    x = torch.tensor(coords, dtype=dtype, device=device)
    y = torch.tensor(targets / target_scale, dtype=dtype, device=device)
    gate_dirs_t = torch.tensor(gate_dirs, dtype=dtype, device=device)
    up_dirs_t = torch.tensor(up_dirs, dtype=dtype, device=device)
    up_projection = x @ up_dirs_t.T
    if selection_coords is not None and selection_targets is not None and selection_coords.shape[0] > 0:
        selection_x = torch.tensor(selection_coords, dtype=dtype, device=device)
        selection_y = torch.tensor(selection_targets / target_scale, dtype=dtype, device=device)
        selection_up_projection = selection_x @ up_dirs_t.T
    else:
        selection_x = x
        selection_y = y
        selection_up_projection = up_projection

    initial_log_norms = torch.tensor(np.log(gate_norms), dtype=dtype, device=device)
    log_norm_span = math.log(max_norm_ratio)
    tail_count = max(64, min(coords.shape[0], int(coords.shape[0] * tail_fraction)))
    base_metrics = regression_metrics(targets, predict_swiglu(coords, candidate))
    best_state: tuple[float, float, float, np.ndarray, np.ndarray, float] | None = None

    def make_design(
        input_x: torch.Tensor,
        input_up_projection: torch.Tensor,
        norms: torch.Tensor,
    ) -> torch.Tensor:
        gate_projection = (input_x @ gate_dirs_t.T) * norms
        features = torch.nn.functional.silu(gate_projection) * input_up_projection
        return torch.cat([
            features,
            torch.ones((features.shape[0], 1), dtype=dtype, device=device),
        ], dim=1)

    def solve_linear_coefficients(design: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        sample_count = max(1, design.shape[0])
        gram = (design.T @ design) / sample_count
        rhs = (design.T @ values) / sample_count
        ridge_scale = torch.trace(gram).detach().clamp_min(1e-12) * 1e-10 / gram.shape[0]
        regularizer = torch.eye(gram.shape[0], dtype=dtype, device=device) * ridge_scale
        regularizer[-1, -1] = 0.0
        return torch.linalg.solve(gram + regularizer, rhs)

    restart_summaries: list[dict[str, float]] = []
    index_values = torch.arange(gates_np.shape[0], dtype=dtype, device=device)
    for restart_index in range(restart_count):
        if restart_index == 0:
            initial_raw = torch.zeros_like(initial_log_norms)
        else:
            phase = float(restart_index) * 1.618033988749895
            initial_raw = 0.35 * torch.sin(index_values * (0.73 + 0.11 * restart_index) + phase)
        raw_norm_offsets = torch.nn.Parameter(initial_raw)
        optimizer = torch.optim.LBFGS(
            [raw_norm_offsets],
            lr=lr,
            max_iter=steps,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-10,
            tolerance_change=1e-12,
            history_size=min(80, max(20, steps // 2)),
        )

        def current_state() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            norms = torch.exp(initial_log_norms + torch.tanh(raw_norm_offsets) * log_norm_span)
            train_design = make_design(x, up_projection, norms)
            coefficients = solve_linear_coefficients(train_design, y)
            train_prediction = train_design @ coefficients
            selection_design = make_design(selection_x, selection_up_projection, norms)
            selection_prediction = selection_design @ coefficients
            return norms, coefficients, train_prediction, selection_prediction

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            _norms, _coefficients, train_prediction, selection_prediction = current_state()
            train_residual = train_prediction - y
            selection_residual = selection_prediction - selection_y
            loss = 0.25 * torch.mean(train_residual ** 2) + 0.75 * torch.mean(selection_residual ** 2)
            if tail_weight > 0.0:
                tail_values = torch.topk(torch.abs(train_residual), k=tail_count, largest=True).values
                loss = loss + tail_weight * torch.mean(tail_values ** 2)
            if mae_weight > 0.0:
                loss = loss + mae_weight * torch.mean(torch.abs(selection_residual))
            loss.backward()
            return loss

        try:
            optimizer.step(closure)
            with torch.no_grad():
                norms, coefficients, _train_prediction, selection_prediction = current_state()
                selected_abs = torch.abs(selection_prediction - selection_y)
                worst = float(torch.max(selected_abs).detach().cpu().item())
                mae_now = float(torch.mean(selected_abs).detach().cpu().item())
                score = worst + best_mae_weight * mae_now
                restart_summaries.append({
                    "restart": float(restart_index),
                    "score": float(score),
                    "worst": float(worst),
                    "mae": float(mae_now),
                })
                if best_state is None or score < best_state[0]:
                    coefficient_values = coefficients.detach().cpu().numpy().astype(np.float64)
                    best_state = (
                        score,
                        worst,
                        mae_now,
                        norms.detach().cpu().numpy().astype(np.float64),
                        coefficient_values[:-1],
                        float(coefficient_values[-1]),
                    )
        except (RuntimeError, ValueError):
            continue
    if best_state is None:
        return candidate
    best_score, best_worst, best_mae, learned_norms, learned_scales, learned_bias = best_state
    calibrated = dict(candidate)
    calibrated.update({
        "gates": gate_dirs * learned_norms[:, None],
        "ups": up_dirs,
        "scales": learned_scales * target_scale,
        "bias": float(learned_bias * target_scale),
    })
    prediction = predict_swiglu(coords, calibrated)
    metrics = regression_metrics(targets, prediction)
    calibration_diagnostics = {
        "gateNormCalibrationAttempted": True,
        "gateNormCalibrationMethod": "bounded-variable-projection-lbfgs",
        "gateNormCalibrationRestarts": int(restart_count),
        "gateNormCalibrationRestartSummaries": restart_summaries,
        "gateNormCalibrationAttemptMae": float(metrics["mae"]),
        "gateNormCalibrationAttemptWorst": float(metrics["worstError"]),
        "gateNormCalibrationAttemptR2": float(metrics["r2"]),
        "gateNormCalibrationTargetScale": float(target_scale),
    }
    if metrics["mae"] >= base_metrics["mae"] and metrics["worstError"] >= base_metrics["worstError"]:
        rejected = dict(candidate)
        rejected.update(calibration_diagnostics)
        return rejected
    calibrated.update({
        "mae": float(metrics["mae"]),
        "mse": float(metrics["mse"]),
        "score": float(metrics["mae"]),
        "selectionScore": float(metrics["mae"]),
        "worstError": float(metrics["worstError"]),
        "gateNormCalibrated": True,
        "gateNormCalibrationSteps": int(steps),
        "gateNormCalibrationLearningRate": float(lr),
        **calibration_diagnostics,
        "gateNormCalibrationTailWeight": float(tail_weight),
        "gateNormCalibrationTailFraction": float(tail_fraction),
        "gateNormCalibrationBestObjectiveScore": float(best_score),
        "gateNormCalibrationBestObjectiveMae": float(best_mae),
        "gateNormCalibrationBestObjectiveWorst": float(best_worst),
        "gateNormCalibrationSelectionCount": int(0 if selection_coords is None else selection_coords.shape[0]),
        "gateNormCalibrationMaeBefore": float(base_metrics["mae"]),
        "gateNormCalibrationWorstBefore": float(base_metrics["worstError"]),
        "gateNormCalibrationTargetScale": float(target_scale),
    })
    return calibrated


def trust_region_structure_refine_swiglu_candidate(
    coords: np.ndarray,
    targets: np.ndarray,
    candidate: dict[str, Any],
    selection_coords: np.ndarray | None = None,
    selection_targets: np.ndarray | None = None,
    gradient_coords: np.ndarray | None = None,
    gradient_targets: np.ndarray | None = None,
    cubic_directions: np.ndarray | None = None,
    cubic_targets: np.ndarray | None = None,
    *,
    max_tangent_override: float | None = None,
    gradient_weight_override: float | None = None,
    cubic_weight_override: float | None = None,
) -> dict[str, Any]:
    import torch

    gates_np = np.asarray(candidate["gates"], dtype=np.float64)
    ups_np = np.asarray(candidate["ups"], dtype=np.float64)
    scales_np = np.asarray(candidate["scales"], dtype=np.float64)
    if gates_np.ndim != 2 or ups_np.shape != gates_np.shape or scales_np.shape != (gates_np.shape[0],):
        return candidate
    if coords.shape[0] < max(256, gates_np.shape[0] * 16):
        return candidate

    gate_norms = np.linalg.norm(gates_np, axis=1).clip(min=1e-9)
    gate_anchors = gates_np / gate_norms[:, None]
    up_norms = np.linalg.norm(ups_np, axis=1).clip(min=1e-12)
    up_anchors = ups_np / up_norms[:, None]
    if not (np.all(np.isfinite(gate_anchors)) and np.all(np.isfinite(up_anchors))):
        return candidate

    max_tangent = (
        float(max_tangent_override)
        if max_tangent_override is not None
        else float(os.environ.get(
            "GCL_BLIND_SWIGLU_STRUCTURE_REFINE_MAX_TANGENT",
            "0.25",
        ) or "0.25")
    )
    max_norm_ratio = float(os.environ.get(
        "GCL_BLIND_SWIGLU_STRUCTURE_REFINE_MAX_NORM_RATIO",
        "6.0",
    ) or "6.0")
    steps = int(os.environ.get("GCL_BLIND_SWIGLU_STRUCTURE_REFINE_STEPS", "180") or "180")
    restart_count = int(os.environ.get("GCL_BLIND_SWIGLU_STRUCTURE_REFINE_RESTARTS", "2") or "2")
    learning_rate = float(os.environ.get("GCL_BLIND_SWIGLU_STRUCTURE_REFINE_LR", "0.6") or "0.6")
    gradient_weight = (
        float(gradient_weight_override)
        if gradient_weight_override is not None
        else float(os.environ.get(
            "GCL_BLIND_SWIGLU_STRUCTURE_REFINE_GRADIENT_WEIGHT",
            "0.25",
        ) or "0.25")
    )
    cubic_weight = (
        float(cubic_weight_override)
        if cubic_weight_override is not None
        else float(os.environ.get(
            "GCL_BLIND_SWIGLU_STRUCTURE_REFINE_CUBIC_WEIGHT",
            "0.5",
        ) or "0.5")
    )
    if (
        not np.isfinite(max_tangent)
        or max_tangent <= 0.0
        or max_tangent >= 1.0
        or not np.isfinite(max_norm_ratio)
        or max_norm_ratio < 1.0
        or steps <= 0
        or restart_count <= 0
        or not np.isfinite(learning_rate)
        or learning_rate <= 0.0
        or not np.isfinite(gradient_weight)
        or gradient_weight < 0.0
        or not np.isfinite(cubic_weight)
        or cubic_weight < 0.0
    ):
        raise RuntimeError("invalid trust-region structure refinement configuration")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    target_scale = max(float(np.std(targets)), float(np.mean(np.abs(targets))), 1e-8)
    x = torch.tensor(coords, dtype=dtype, device=device)
    y = torch.tensor(targets / target_scale, dtype=dtype, device=device)
    gate_anchor_t = torch.tensor(gate_anchors, dtype=dtype, device=device)
    up_anchor_t = torch.tensor(up_anchors, dtype=dtype, device=device)
    initial_log_norms = torch.tensor(np.log(gate_norms), dtype=dtype, device=device)
    log_norm_span = math.log(max_norm_ratio)
    if selection_coords is not None and selection_targets is not None and selection_coords.shape[0] > 0:
        selection_x = torch.tensor(selection_coords, dtype=dtype, device=device)
        selection_y = torch.tensor(selection_targets / target_scale, dtype=dtype, device=device)
    else:
        selection_x = x
        selection_y = y

    has_gradient_witness = (
        gradient_coords is not None
        and gradient_targets is not None
        and gradient_weight > 0.0
        and gradient_coords.shape[0] > 0
        and gradient_targets.shape == gradient_coords.shape
    )
    if has_gradient_witness:
        gradient_x = torch.tensor(gradient_coords, dtype=dtype, device=device)
        gradient_y = torch.tensor(gradient_targets / target_scale, dtype=dtype, device=device)
        gradient_scale = torch.mean(gradient_y ** 2).detach().clamp_min(1e-12)
    else:
        gradient_x = None
        gradient_y = None
        gradient_scale = None
    has_cubic_witness = (
        cubic_directions is not None
        and cubic_targets is not None
        and cubic_weight > 0.0
        and cubic_directions.shape[0] > 0
        and cubic_targets.shape == (cubic_directions.shape[0],)
    )
    if has_cubic_witness:
        cubic_x = torch.tensor(cubic_directions, dtype=dtype, device=device)
        cubic_y = torch.tensor(cubic_targets / target_scale, dtype=dtype, device=device)
        cubic_scale = torch.mean(cubic_y ** 2).detach().clamp_min(1e-12)
    else:
        cubic_x = None
        cubic_y = None
        cubic_scale = None

    def bounded_directions(anchors: torch.Tensor, raw_offsets: torch.Tensor) -> torch.Tensor:
        tangent = raw_offsets - torch.sum(raw_offsets * anchors, dim=1, keepdim=True) * anchors
        bounded = max_tangent * tangent / torch.sqrt(
            1.0 + torch.sum(tangent ** 2, dim=1, keepdim=True)
        )
        directions = anchors + bounded
        return directions / torch.linalg.norm(directions, dim=1, keepdim=True).clamp_min(1e-12)

    def make_design(
        input_x: torch.Tensor,
        gate_directions_t: torch.Tensor,
        up_directions_t: torch.Tensor,
        norms: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.nn.functional.silu(
            (input_x @ gate_directions_t.T) * norms
        ) * (input_x @ up_directions_t.T)
        return torch.cat([
            features,
            torch.ones((features.shape[0], 1), dtype=dtype, device=device),
        ], dim=1)

    def solve_coefficients(design: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        count = max(1, design.shape[0])
        gram = (design.T @ design) / count
        rhs = (design.T @ values) / count
        ridge = torch.trace(gram).detach().clamp_min(1e-12) * 1e-10 / gram.shape[0]
        regularizer = torch.eye(gram.shape[0], dtype=dtype, device=device) * ridge
        regularizer[-1, -1] = 0.0
        return torch.linalg.solve(gram + regularizer, rhs)

    base_selection_metrics = regression_metrics(
        np.asarray(selection_targets if selection_targets is not None else targets, dtype=np.float64),
        predict_swiglu(
            np.asarray(selection_coords if selection_coords is not None else coords, dtype=np.float64),
            candidate,
        ),
    )
    base_selection_score = float(base_selection_metrics["worstError"]) + 2.0 * float(base_selection_metrics["mae"])
    best_state: dict[str, Any] | None = None
    restart_summaries: list[dict[str, float]] = []
    atom_indices = torch.arange(gates_np.shape[0], dtype=dtype, device=device)[:, None]
    coordinate_indices = torch.arange(gates_np.shape[1], dtype=dtype, device=device)[None, :]

    for restart_index in range(restart_count):
        if restart_index == 0:
            gate_start = torch.zeros_like(gate_anchor_t)
            up_start = torch.zeros_like(up_anchor_t)
            norm_start = torch.zeros(gates_np.shape[0], dtype=dtype, device=device)
        else:
            phase = float(restart_index) * 1.324717957244746
            gate_start = 0.03 * torch.sin(atom_indices * 0.71 + coordinate_indices * 0.37 + phase)
            up_start = 0.03 * torch.cos(atom_indices * 0.43 + coordinate_indices * 0.61 + phase)
            norm_start = 0.08 * torch.sin(atom_indices[:, 0] * 0.79 + phase)
        raw_gate_offsets = torch.nn.Parameter(gate_start)
        raw_up_offsets = torch.nn.Parameter(up_start)
        raw_norm_offsets = torch.nn.Parameter(norm_start)
        optimizer = torch.optim.LBFGS(
            [raw_gate_offsets, raw_up_offsets, raw_norm_offsets],
            lr=learning_rate,
            max_iter=steps,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-10,
            tolerance_change=1e-12,
            history_size=min(80, max(20, steps // 2)),
        )

        def current_state() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            gate_directions_t = bounded_directions(gate_anchor_t, raw_gate_offsets)
            up_directions_t = bounded_directions(up_anchor_t, raw_up_offsets)
            norms = torch.exp(initial_log_norms + torch.tanh(raw_norm_offsets) * log_norm_span)
            train_design = make_design(x, gate_directions_t, up_directions_t, norms)
            coefficients = solve_coefficients(train_design, y)
            train_prediction = train_design @ coefficients
            selection_prediction = make_design(
                selection_x,
                gate_directions_t,
                up_directions_t,
                norms,
            ) @ coefficients
            return gate_directions_t, up_directions_t, norms, coefficients, train_prediction, selection_prediction

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            gate_directions_t, up_directions_t, norms, coefficients, train_prediction, selection_prediction = current_state()
            loss = 0.35 * torch.mean((train_prediction - y) ** 2)
            loss = loss + 0.65 * torch.mean((selection_prediction - selection_y) ** 2)
            if has_gradient_witness and gradient_x is not None and gradient_y is not None and gradient_scale is not None:
                gate_vectors_t = gate_directions_t * norms[:, None]
                gate_projection = gradient_x @ gate_vectors_t.T
                up_projection = gradient_x @ up_directions_t.T
                sigmoid = torch.sigmoid(gate_projection)
                silu_values = torch.nn.functional.silu(gate_projection)
                silu_prime = sigmoid * (1.0 + gate_projection * (1.0 - sigmoid))
                predicted_gradients = torch.sum(
                    coefficients[:-1][None, :, None] * (
                        silu_prime[:, :, None] * up_projection[:, :, None] * gate_vectors_t[None, :, :]
                        + silu_values[:, :, None] * up_directions_t[None, :, :]
                    ),
                    dim=1,
                )
                loss = loss + gradient_weight * torch.mean(
                    (predicted_gradients - gradient_y) ** 2
                ) / gradient_scale
            if has_cubic_witness and cubic_x is not None and cubic_y is not None and cubic_scale is not None:
                gate_vectors_t = gate_directions_t * norms[:, None]
                predicted_cubic = 1.5 * torch.sum(
                    (cubic_x @ gate_vectors_t.T) ** 2
                    * (cubic_x @ up_directions_t.T)
                    * coefficients[:-1][None, :],
                    dim=1,
                )
                loss = loss + cubic_weight * torch.mean(
                    (predicted_cubic - cubic_y) ** 2
                ) / cubic_scale
            loss = loss + 1e-7 * (
                torch.mean(raw_gate_offsets ** 2)
                + torch.mean(raw_up_offsets ** 2)
                + torch.mean(raw_norm_offsets ** 2)
            )
            loss.backward()
            return loss

        try:
            optimizer.step(closure)
            with torch.no_grad():
                gate_directions_t, up_directions_t, norms, coefficients, train_prediction, selection_prediction = current_state()
                train_abs = torch.abs(train_prediction - y) * target_scale
                selection_abs = torch.abs(selection_prediction - selection_y) * target_scale
                train_mae = float(torch.mean(train_abs).detach().cpu().item())
                selection_mae = float(torch.mean(selection_abs).detach().cpu().item())
                selection_worst = float(torch.max(selection_abs).detach().cpu().item())
                score = selection_worst + 2.0 * selection_mae
                restart_summaries.append({
                    "restart": float(restart_index),
                    "score": float(score),
                    "trainMae": float(train_mae),
                    "selectionMae": float(selection_mae),
                    "selectionWorst": float(selection_worst),
                })
                if best_state is None or score < best_state["score"]:
                    best_state = {
                        "score": score,
                        "gateDirections": gate_directions_t.detach().cpu().numpy().astype(np.float64),
                        "upDirections": up_directions_t.detach().cpu().numpy().astype(np.float64),
                        "norms": norms.detach().cpu().numpy().astype(np.float64),
                        "coefficients": coefficients.detach().cpu().numpy().astype(np.float64),
                    }
        except (RuntimeError, ValueError):
            continue

    if best_state is None:
        return candidate
    coefficients_np = np.asarray(best_state["coefficients"], dtype=np.float64)
    refined = dict(candidate)
    refined.update({
        "gates": np.asarray(best_state["gateDirections"]) * np.asarray(best_state["norms"])[:, None],
        "ups": np.asarray(best_state["upDirections"]),
        "scales": coefficients_np[:-1] * target_scale,
        "bias": float(coefficients_np[-1] * target_scale),
    })
    train_metrics = regression_metrics(targets, predict_swiglu(coords, refined))
    refined_selection_metrics = regression_metrics(
        np.asarray(selection_targets if selection_targets is not None else targets, dtype=np.float64),
        predict_swiglu(
            np.asarray(selection_coords if selection_coords is not None else coords, dtype=np.float64),
            refined,
        ),
    )
    refined_selection_score = float(refined_selection_metrics["worstError"]) + 2.0 * float(refined_selection_metrics["mae"])
    if refined_selection_score >= base_selection_score:
        rejected = dict(candidate)
        rejected.update({
            "structureTrustRegionAttempted": True,
            "structureTrustRegionAccepted": False,
            "structureTrustRegionRestarts": restart_summaries,
            "structureTrustRegionSelectionScoreBefore": float(base_selection_score),
            "structureTrustRegionSelectionScoreAfter": float(refined_selection_score),
        })
        return rejected
    refined.update({
        "mae": float(train_metrics["mae"]),
        "mse": float(train_metrics["mse"]),
        "r2": float(train_metrics["r2"]),
        "score": float(train_metrics["mae"]),
        "selectionScore": float(refined_selection_score),
        "worstError": float(train_metrics["worstError"]),
        "structureTrustRegionAttempted": True,
        "structureTrustRegionAccepted": True,
        "structureTrustRegionRestarts": restart_summaries,
        "structureTrustRegionSelectionScoreBefore": float(base_selection_score),
        "structureTrustRegionSelectionScoreAfter": float(refined_selection_score),
        "structureTrustRegionMaxTangent": float(max_tangent),
        "structureTrustRegionMinGateAnchorCosine": float(np.min(np.abs(np.sum(
            np.asarray(best_state["gateDirections"]) * gate_anchors,
            axis=1,
        )))),
        "structureTrustRegionMinUpAnchorCosine": float(np.min(np.abs(np.sum(
            np.asarray(best_state["upDirections"]) * up_anchors,
            axis=1,
        )))),
        "structureTrustRegionUsedGradientWitness": bool(has_gradient_witness),
        "structureTrustRegionUsedCubicWitness": bool(has_cubic_witness),
    })
    return refined


def pair_swap_refine_swiglu_candidate(
    coords: np.ndarray,
    targets: np.ndarray,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    gates_np = np.asarray(candidate["gates"], dtype=np.float64)
    ups_np = np.asarray(candidate["ups"], dtype=np.float64)
    scales_np = np.asarray(candidate["scales"], dtype=np.float64)
    if gates_np.ndim != 2 or ups_np.shape != gates_np.shape or scales_np.shape[0] != gates_np.shape[0]:
        return candidate
    atom_count = gates_np.shape[0]
    if atom_count < 3 or coords.shape[0] < max(256, atom_count * 16):
        return candidate
    cycles = int(os.environ.get("GCL_BLIND_SWIGLU_PAIR_SWAP_CYCLES", "2") or "2")
    mae_weight = float(os.environ.get("GCL_BLIND_SWIGLU_PAIR_SWAP_MAE_WEIGHT", "2.0") or "2.0")
    max_mae_ratio = float(os.environ.get("GCL_BLIND_SWIGLU_PAIR_SWAP_MAX_MAE_RATIO", "1.05") or "1.05")
    if cycles <= 0:
        return candidate
    if not np.isfinite(mae_weight) or mae_weight < 0.0:
        raise RuntimeError("GCL_BLIND_SWIGLU_PAIR_SWAP_MAE_WEIGHT must be non-negative and finite")
    if not np.isfinite(max_mae_ratio) or max_mae_ratio < 1.0:
        raise RuntimeError("GCL_BLIND_SWIGLU_PAIR_SWAP_MAX_MAE_RATIO must be >= 1")

    def fit_with(gates: np.ndarray, up_vectors: np.ndarray) -> tuple[dict[str, Any] | None, dict[str, float]]:
        features = silu_np(coords @ gates.T) * (coords @ up_vectors.T)
        design = np.column_stack([features, np.ones(coords.shape[0], dtype=np.float64)])
        try:
            coeffs, *_ = np.linalg.lstsq(design, targets, rcond=None)
        except np.linalg.LinAlgError:
            return None, {"mae": float("inf"), "mse": float("inf"), "r2": -float("inf"), "worstError": float("inf")}
        if not np.all(np.isfinite(coeffs)):
            return None, {"mae": float("inf"), "mse": float("inf"), "r2": -float("inf"), "worstError": float("inf")}
        trial = dict(candidate)
        trial.update({
            "gates": gates.copy(),
            "ups": up_vectors * coeffs[:-1, None],
            "scales": np.ones(atom_count, dtype=np.float64),
            "bias": float(coeffs[-1]),
        })
        prediction = predict_swiglu(coords, trial)
        metrics = regression_metrics(targets, prediction)
        return trial, metrics

    current, current_metrics = fit_with(gates_np, ups_np * scales_np[:, None])
    if current is None:
        return candidate
    best = current
    best_metrics = current_metrics
    initial_metrics = current_metrics
    up_vectors = np.asarray(best["ups"], dtype=np.float64)
    accepted_swaps: list[list[int]] = []

    def score(metrics: dict[str, float]) -> float:
        return float(metrics["worstError"]) + mae_weight * float(metrics["mae"])

    for _cycle in range(cycles):
        cycle_best: tuple[float, int, int, dict[str, Any], dict[str, float], np.ndarray] | None = None
        base_score = score(best_metrics)
        for left in range(atom_count):
            for right in range(left + 1, atom_count):
                swapped = up_vectors.copy()
                swapped[[left, right]] = swapped[[right, left]]
                trial, metrics = fit_with(np.asarray(best["gates"], dtype=np.float64), swapped)
                if trial is None:
                    continue
                if metrics["mae"] > best_metrics["mae"] * max_mae_ratio:
                    continue
                trial_score = score(metrics)
                if (
                    trial_score < base_score
                    and (cycle_best is None or trial_score < cycle_best[0])
                ):
                    cycle_best = (trial_score, left, right, trial, metrics, swapped)
        if cycle_best is None:
            break
        _trial_score, left, right, best, best_metrics, up_vectors = cycle_best
        accepted_swaps.append([int(left), int(right)])

    if not accepted_swaps:
        return candidate
    refined = dict(best)
    refined.update({
        "mae": float(best_metrics["mae"]),
        "mse": float(best_metrics["mse"]),
        "score": float(best_metrics["mae"]),
        "selectionScore": float(best_metrics["mae"]),
        "worstError": float(best_metrics["worstError"]),
        "pairSwapRefined": True,
        "pairSwapCycles": int(cycles),
        "pairSwapAccepted": accepted_swaps,
        "pairSwapMaeBefore": float(initial_metrics["mae"]),
        "pairSwapWorstBefore": float(initial_metrics["worstError"]),
    })
    return refined


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    residual = actual - predicted
    mae = float(np.mean(np.abs(residual)))
    mse = float(np.mean(residual ** 2))
    variance = float(np.mean((actual - np.mean(actual)) ** 2))
    r2 = 1.0 - mse / max(variance, 1e-12)
    worst = float(np.max(np.abs(residual)))
    return {"mae": mae, "mse": mse, "r2": r2, "worstError": worst}


def atom_tail_diagnostics(
    coords: np.ndarray,
    targets: np.ndarray,
    candidate: dict[str, Any],
    prediction: np.ndarray,
    tail_fraction: float,
    top_k: int,
) -> dict[str, Any]:
    gates = np.asarray(candidate["gates"], dtype=np.float64)
    ups = np.asarray(candidate["ups"], dtype=np.float64)
    scales = np.asarray(candidate["scales"], dtype=np.float64)
    if gates.ndim != 2 or ups.shape != gates.shape or scales.shape[0] != gates.shape[0]:
        return {"enabled": True, "error": "invalid-candidate-shape"}
    if coords.shape[0] == 0:
        return {"enabled": True, "error": "empty-coords"}
    tail_fraction = max(1e-4, min(1.0, float(tail_fraction)))
    atom_count = gates.shape[0]
    contribution = silu_np(coords @ gates.T) * (coords @ (ups * scales[:, None]).T)
    residual = targets - prediction
    abs_residual = np.abs(residual)
    tail_count = max(1, min(coords.shape[0], int(math.ceil(coords.shape[0] * tail_fraction))))
    tail_indices = np.argpartition(abs_residual, -tail_count)[-tail_count:]
    tail_residual = residual[tail_indices]
    tail_abs = np.abs(tail_residual)
    tail_mean_abs = float(np.mean(tail_abs))
    tail_worst = float(np.max(tail_abs))
    rows: list[dict[str, Any]] = []
    for atom_index in range(atom_count):
        atom_values = contribution[:, atom_index]
        atom_tail = atom_values[tail_indices]
        after_removal_residual = tail_residual + atom_tail
        atom_tail_mean_abs = float(np.mean(np.abs(atom_tail)))
        after_removal_mean_abs = float(np.mean(np.abs(after_removal_residual)))
        after_removal_worst = float(np.max(np.abs(after_removal_residual)))
        atom_std = float(np.std(atom_tail))
        residual_std = float(np.std(tail_residual))
        if atom_std > 1e-12 and residual_std > 1e-12:
            corr = float(np.corrcoef(tail_residual, atom_tail)[0, 1])
        else:
            corr = 0.0
        denom = float(np.dot(atom_tail, atom_tail))
        if denom > 1e-12:
            tail_scale_delta = float(np.dot(tail_residual, atom_tail) / denom)
            corrected_residual = tail_residual - tail_scale_delta * atom_tail
            corrected_mean_abs = float(np.mean(np.abs(corrected_residual)))
            corrected_worst = float(np.max(np.abs(corrected_residual)))
        else:
            tail_scale_delta = 0.0
            corrected_mean_abs = tail_mean_abs
            corrected_worst = tail_worst
        responsibility = float(np.mean(np.abs(tail_residual * atom_tail)))
        rows.append({
            "atom": int(atom_index),
            "tailContributionMeanAbs": atom_tail_mean_abs,
            "tailResidualCorrelation": corr,
            "tailResponsibilityScore": responsibility,
            "tailMeanAbsAfterRemoval": after_removal_mean_abs,
            "tailWorstAfterRemoval": after_removal_worst,
            "tailScaleDelta": tail_scale_delta,
            "tailMeanAbsAfterScaleDelta": corrected_mean_abs,
            "tailWorstAfterScaleDelta": corrected_worst,
        })
    rows.sort(
        key=lambda item: (
            -float(item["tailResponsibilityScore"]),
            float(item["tailMeanAbsAfterScaleDelta"]),
        )
    )
    keep = max(1, min(atom_count, int(top_k)))
    return {
        "enabled": True,
        "tailFraction": float(tail_fraction),
        "tailCount": int(tail_count),
        "tailMeanAbs": tail_mean_abs,
        "tailWorstError": tail_worst,
        "topAtoms": rows[:keep],
    }


def select_residual_block_atoms(
    diagnostic: dict[str, Any],
    atom_count: int,
    *,
    adaptive: bool,
    fixed_top_k: int,
    min_atoms: int,
    max_atoms: int,
    gate_norms: np.ndarray | None = None,
    gate_weak_ratio: float = 0.95,
    strong_cut_ratio: float = 0.75,
) -> tuple[list[int], dict[str, Any]]:
    rows = [
        item
        for item in diagnostic.get("topAtoms", [])
        if isinstance(item, dict) and isinstance(item.get("atom"), (int, np.integer))
    ]
    available = min(max(0, int(atom_count) - 1), len(rows))
    if available <= 0:
        return [], {
            "method": "none",
            "selectedCount": 0,
            "availableCount": 0,
            "cutRatios": [],
        }

    fixed_count = max(1, min(available, int(fixed_top_k)))
    if not adaptive:
        return [int(item["atom"]) for item in rows[:fixed_count]], {
            "method": "fixed-top-k",
            "selectedCount": fixed_count,
            "availableCount": available,
            "minAtoms": fixed_count,
            "maxAtoms": fixed_count,
            "cutRatios": [],
        }

    lower = max(1, min(available, int(min_atoms)))
    upper = max(lower, min(available, int(max_atoms)))
    scores = [max(0.0, float(item.get("tailResponsibilityScore", 0.0))) for item in rows[:available]]

    normalized_gate_norms = None
    if gate_norms is not None:
        candidate_gate_norms = np.asarray(gate_norms, dtype=np.float64)
        if (
            candidate_gate_norms.shape == (atom_count,)
            and np.all(np.isfinite(candidate_gate_norms))
            and np.all(candidate_gate_norms > 0.0)
            and np.isfinite(gate_weak_ratio)
            and 0.0 < gate_weak_ratio < 1.0
            and np.isfinite(strong_cut_ratio)
            and 0.0 < strong_cut_ratio < 1.0
        ):
            normalized_gate_norms = candidate_gate_norms

    if normalized_gate_norms is not None:
        gate_norm_median = float(np.median(normalized_gate_norms))
        gate_norm_threshold = gate_norm_median * float(gate_weak_ratio)
        weak_gate_atoms = [
            atom_index
            for atom_index in range(atom_count)
            if float(normalized_gate_norms[atom_index]) <= gate_norm_threshold
        ]
        consensus_rows = [
            item
            for item in rows[:available]
            if float(normalized_gate_norms[int(item["atom"])]) <= gate_norm_threshold
        ]
        fixed_width_tail_completion = (
            lower == upper
            and len(weak_gate_atoms) == lower - 1
            and len(consensus_rows) == len(weak_gate_atoms)
        )
        if (
            (len(consensus_rows) >= lower and len(weak_gate_atoms) >= lower)
            or fixed_width_tail_completion
        ):
            consensus_limit = min(len(consensus_rows), upper, len(weak_gate_atoms))
            consensus_cuts: list[dict[str, float | int]] = []
            for selected_count in range(lower, consensus_limit + 1):
                if selected_count >= len(consensus_rows):
                    break
                current = max(
                    0.0,
                    float(consensus_rows[selected_count - 1].get("tailResponsibilityScore", 0.0)),
                )
                following = max(
                    0.0,
                    float(consensus_rows[selected_count].get("tailResponsibilityScore", 0.0)),
                )
                consensus_cuts.append({
                    "selectedCount": int(selected_count),
                    "scoreBeforeCut": float(current),
                    "scoreAfterCut": float(following),
                    "ratio": float(following / max(current, 1e-15)),
                })
            strongest_cut = min(
                consensus_cuts,
                key=lambda item: (float(item["ratio"]), int(item["selectedCount"])),
                default=None,
            )
            if strongest_cut is not None and float(strongest_cut["ratio"]) <= strong_cut_ratio:
                selected_count = int(strongest_cut["selectedCount"])
                method = "gate-norm-tail-consensus-strong-cut"
                selected_atoms = [int(item["atom"]) for item in consensus_rows[:selected_count]]
            else:
                ordered_weak_atoms = [int(item["atom"]) for item in consensus_rows]
                ordered_weak_atoms.extend(sorted(
                    (atom for atom in weak_gate_atoms if atom not in ordered_weak_atoms),
                    key=lambda atom: float(normalized_gate_norms[atom]),
                ))
                tail_completion_atoms: list[int] = []
                if fixed_width_tail_completion:
                    for item in rows[:available]:
                        atom = int(item["atom"])
                        if atom in ordered_weak_atoms:
                            continue
                        ordered_weak_atoms.append(atom)
                        tail_completion_atoms.append(atom)
                        if len(ordered_weak_atoms) >= lower:
                            break
                selected_count = min(len(ordered_weak_atoms), upper)
                selected_atoms = ordered_weak_atoms[:selected_count]
                method = (
                    "gate-norm-weak-block-tail-completion"
                    if tail_completion_atoms
                    else "gate-norm-weak-block"
                )
            return selected_atoms, {
                "method": method,
                "selectedCount": int(selected_count),
                "availableCount": int(available),
                "consensusCount": int(len(consensus_rows)),
                "weakGateCount": int(len(weak_gate_atoms)),
                "minAtoms": int(lower),
                "maxAtoms": int(upper),
                "gateNormMedian": gate_norm_median,
                "gateNormWeakRatio": float(gate_weak_ratio),
                "gateNormThreshold": gate_norm_threshold,
                "strongCutRatio": float(strong_cut_ratio),
                "consensusAtoms": [int(item["atom"]) for item in consensus_rows],
                "weakGateAtoms": [int(atom) for atom in weak_gate_atoms],
                "tailCompletionAtoms": tail_completion_atoms if strongest_cut is None else [],
                "cutRatios": consensus_cuts,
            }

    cut_rows: list[dict[str, float | int]] = []
    for selected_count in range(lower, upper + 1):
        if selected_count >= available:
            break
        current = scores[selected_count - 1]
        following = scores[selected_count]
        ratio = following / max(current, 1e-15)
        cut_rows.append({
            "selectedCount": int(selected_count),
            "scoreBeforeCut": float(current),
            "scoreAfterCut": float(following),
            "ratio": float(ratio),
        })

    if cut_rows and scores[0] > 0.0:
        best_cut = min(cut_rows, key=lambda item: (float(item["ratio"]), int(item["selectedCount"])))
        selected_count = int(best_cut["selectedCount"])
        method = "responsibility-elbow"
    else:
        selected_count = fixed_count
        method = "adaptive-fallback-fixed-top-k"
    return [int(item["atom"]) for item in rows[:selected_count]], {
        "method": method,
        "selectedCount": int(selected_count),
        "availableCount": int(available),
        "minAtoms": int(lower),
        "maxAtoms": int(upper),
        "cutRatios": cut_rows,
    }


def passes_metric_thresholds(
    metrics: dict[str, float],
    min_r2: float,
    max_mae: float,
    max_worst_error: float,
) -> bool:
    return (
        metrics["r2"] >= min_r2
        and metrics["mae"] <= max_mae
        and metrics["worstError"] <= max_worst_error
    )


def independent_reporting_metrics(
    oracle: Any,
    args: argparse.Namespace,
    basis: np.ndarray,
    dimension: int,
    candidate: dict[str, Any],
    *,
    seed_offset: int,
    stage: str,
) -> dict[str, float] | None:
    """Score an accepted candidate on a draw that took no part in accepting it.

    Four validation-repair stages decided acceptance on a fresh oracle draw and then
    published that same draw's numbers. Each is a max-of-two selection: the repair is kept
    only when its draw looks good, so the reported figure is conditioned on having looked
    good. Evaluation is not on training data, so this is not leakage, but it biases every
    reported statistic upward, and hardest on worstError, which is a maximum over thousands
    of samples and therefore the most sensitive to being chosen for.

    Separating the two draws costs one more validation batch and removes the bias outright,
    rather than recording it and leaving the reader to correct for it. Returns None when the
    query budget cannot cover the extra batch; the caller must then decline the repair
    instead of publishing a number selected on itself.
    """
    if oracle.query_count + args.validation_count > args.max_oracle_calls:
        return None
    rng = np.random.default_rng(args.seed + seed_offset)
    inputs = rng.normal(0.0, args.validation_scale, size=(args.validation_count, dimension)).astype(np.float32)
    targets = oracle.eval(inputs, stage=stage)
    coords = inputs @ basis.T
    return regression_metrics(targets, predict_swiglu(coords, candidate))


def validation_gradient_polish_weights(args: argparse.Namespace) -> tuple[float, float]:
    fit_cubic_weight = float(os.environ.get(
        "GCL_BLIND_SWIGLU_FIT_CUBIC_WEIGHT",
        "0.0",
    ) or "0.0")
    gradient_weight = float(os.environ.get(
        "GCL_BLIND_SWIGLU_VALIDATION_GRADIENT_POLISH_WEIGHT",
        str(args.fit_gradient_weight),
    ) or args.fit_gradient_weight)
    cubic_weight = float(os.environ.get(
        "GCL_BLIND_SWIGLU_VALIDATION_GRADIENT_POLISH_CUBIC_WEIGHT",
        str(fit_cubic_weight),
    ) or fit_cubic_weight)
    if not np.isfinite(gradient_weight) or gradient_weight < 0.0:
        raise RuntimeError("validation gradient polish weight must be non-negative and finite")
    if not np.isfinite(cubic_weight) or cubic_weight < 0.0:
        raise RuntimeError("validation cubic polish weight must be non-negative and finite")
    return gradient_weight, cubic_weight


def validation_full_tail_polish_cubic_trigger(method: str, cubic_score_count: int) -> bool:
    raw = os.environ.get("GCL_BLIND_SWIGLU_VALIDATION_FULL_TAIL_POLISH_ON_CUBIC", "").strip()
    if raw:
        if raw not in {"0", "1"}:
            raise RuntimeError("validation full-tail cubic trigger must be 0 or 1")
        return raw == "1" and cubic_score_count > 0
    return method in {"gradient-polish", "asymptotic-gate-stitch"} and cubic_score_count > 0


_PROGRESS_CLOCK: dict[str, float] = {}
# Set once by `main` so code far from the argument namespace can still report. The fit is the
# longest phase in a run and had no markers inside it at all: `train-sampled` fires at 7.1 s and
# the next line arrives tens of minutes later, which says the fit is slow and nothing about which
# part. A module-level hook avoids threading a callback through a dozen signatures.
_PROGRESS_HOOK: dict[str, Any] = {}


def emit_progress(phase: str, **fields: Any) -> None:
    hook = _PROGRESS_HOOK.get("args")
    if hook is not None:
        log_progress(hook, phase, **fields)


def log_progress(args: argparse.Namespace, phase: str, **fields: Any) -> None:
    if not getattr(args, "progress", False):
        return
    # How long the run has been going, and how long since the last phase said anything. Without
    # these two numbers a stalled run and a working one look identical from outside, and finding
    # out where the minutes went took a profiler, `nvidia-smi pmon` and three wrong guesses. The
    # phase that is slow is now the phase that reports a large `sincePreviousSeconds`.
    now = time.monotonic()
    started = _PROGRESS_CLOCK.setdefault("start", now)
    previous = _PROGRESS_CLOCK.get("previous", started)
    _PROGRESS_CLOCK["previous"] = now
    # On stdout, which the sweep inherits. Its own comment says "stdout stays inherited so
    # progress streams live; stderr is piped so the last words of a process that dies are still
    # available" - and progress was going to the piped one, so it streamed nowhere. A run's phases
    # were only readable after it died, in the failure tail, which is the opposite of what a
    # progress line is for and is why locating a slow phase took a profiler.
    print(json.dumps({
        "type": "progress",
        "phase": phase,
        "elapsedSeconds": round(now - started, 1),
        "sincePreviousSeconds": round(now - previous, 1),
        **fields,
    }), flush=True)


def validation_counterexample_inputs(
    rng: np.random.Generator,
    validation_inputs: np.ndarray,
    validation_coords: np.ndarray,
    validation_targets: np.ndarray,
    validation_prediction: np.ndarray,
    basis: np.ndarray,
    counterexample_count: int,
    repeat_count: int,
    jitter_count: int,
    jitter_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    if counterexample_count <= 0:
        return (
            np.zeros((0, validation_inputs.shape[1]), dtype=np.float32),
            np.zeros(0, dtype=np.float64),
            np.zeros((0, validation_inputs.shape[1]), dtype=np.float32),
            0,
            0,
        )
    residual = validation_targets - validation_prediction
    order = np.argsort(np.abs(residual))[::-1]
    selected_count = min(counterexample_count, validation_inputs.shape[0])
    selected = order[:selected_count]
    repeats = max(1, repeat_count)
    hard_inputs = np.repeat(validation_inputs[selected], repeats, axis=0).astype(np.float32, copy=False)
    hard_targets = np.repeat(validation_targets[selected], repeats, axis=0).astype(np.float64, copy=False)

    if jitter_count <= 0 or jitter_scale <= 0.0:
        return hard_inputs, hard_targets, np.zeros((0, validation_inputs.shape[1]), dtype=np.float32), selected_count, 0
    base_indices = np.resize(selected, jitter_count)
    jitter_coords = validation_coords[base_indices] + rng.normal(
        0.0,
        jitter_scale,
        size=(jitter_count, validation_coords.shape[1]),
    )
    jitter_inputs = (jitter_coords @ basis).astype(np.float32)
    return (
        hard_inputs,
        hard_targets,
        jitter_inputs,
        selected_count,
        jitter_count,
    )


def make_tail_probe_polish_set(
    rng: np.random.Generator,
    oracle: OracleClient,
    basis: np.ndarray,
    fit: dict[str, Any],
    dimension: int,
    count: int,
    scale: float,
    repeat: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    if count <= 0:
        return (
            np.zeros((0, basis.shape[0]), dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            0,
            0,
        )
    repeats = max(1, repeat)
    probe_inputs = rng.normal(0.0, scale, size=(count, dimension)).astype(np.float32)
    probe_targets = oracle.eval(probe_inputs)
    probe_coords = probe_inputs @ basis.T
    probe_prediction = predict_swiglu(probe_coords, fit)
    residual = np.abs(probe_targets - probe_prediction)
    selected_count = min(max(32, int(np.sqrt(count)) * 4), count)
    selected = np.argsort(residual)[-selected_count:]
    hard_coords = np.repeat(probe_coords[selected], repeats, axis=0).astype(np.float64, copy=False)
    hard_targets = np.repeat(probe_targets[selected], repeats, axis=0).astype(np.float64, copy=False)
    return hard_coords, hard_targets, selected_count, count


def refit_weak_atoms_on_boundary_probes(
    rng: np.random.Generator,
    oracle: OracleClient,
    basis: np.ndarray,
    fit: dict[str, Any],
    train_coords: np.ndarray,
    train_targets: np.ndarray,
    *,
    probe_count_per_atom: int,
    atom_fraction: float,
    probe_scale: float,
    boundary_jitter: float,
    learning_rate: float,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Re-fit only the weakest atoms, against fresh probes on their own boundaries.

    In down-proj-output-channel mode the atom weights span two orders of magnitude. On
    llama layer 1 the strongest atom carries standard deviation 5.774e-03 and the weakest
    5.505e-05 against a total of 1.086e-02, a 105x spread, and the residual that keeps R2
    short is about the size of one or two of the weakest. Uniform samples barely constrain
    those atoms because almost nothing lands where their contribution turns on.

    A gate direction is scale-free, so once an atom is roughly placed its switching surface
    is known: `gate . x = 0`. Fresh points are slid onto that surface with a little jitter
    so the fit sees both sides of the kink.

    Concentrating points like that and then re-fitting everything makes the result worse,
    measured: selection MAE went from 2.5032e-04 to 4.7611e-04 because the dense surfaces
    pull the strong atoms off their already-correct positions. So the strong atoms are held
    fixed and their contribution is subtracted instead. The SwiGLU sum is additive, so the
    remainder is exactly what the weak atoms have to explain, and the concentrated points
    can only move the parameters that were underdetermined to begin with.

    Every probe is a fresh oracle draw. The graded validation set is never read.
    """
    gates = np.asarray(fit.get("gates"), dtype=np.float64)
    ups = np.asarray(fit.get("ups"), dtype=np.float64)
    scales = np.asarray(fit.get("scales"), dtype=np.float64)
    diagnostics: dict[str, Any] = {"attempted": False}
    if gates.ndim != 2 or ups.shape != gates.shape or scales.shape != (gates.shape[0],):
        return None, diagnostics
    if probe_count_per_atom <= 0 or not 0.0 < atom_fraction < 1.0:
        return None, diagnostics
    if probe_scale <= 0.0 or boundary_jitter <= 0.0:
        return None, diagnostics

    strengths = np.abs(scales) * np.linalg.norm(gates, axis=1) * np.linalg.norm(ups, axis=1)
    if not np.all(np.isfinite(strengths)) or float(np.max(strengths)) <= 0.0:
        return None, diagnostics
    weak_count = max(1, int(round(gates.shape[0] * atom_fraction)))
    if weak_count >= gates.shape[0]:
        return None, diagnostics
    order = np.argsort(strengths)
    weak_index = order[:weak_count]
    strong_index = order[weak_count:]

    probe_rows: list[np.ndarray] = []
    for atom in weak_index:
        gate_norm = float(np.linalg.norm(gates[int(atom)]))
        if gate_norm < 1e-12:
            continue
        unit_gate = gates[int(atom)] / gate_norm
        samples = rng.normal(0.0, probe_scale, size=(int(probe_count_per_atom), basis.shape[0]))
        samples = samples - np.outer(samples @ unit_gate, unit_gate)
        samples = samples + np.outer(rng.normal(0.0, boundary_jitter, size=int(probe_count_per_atom)), unit_gate)
        probe_rows.append(samples)
    if not probe_rows:
        return None, diagnostics

    probe_coords = np.vstack(probe_rows)
    probe_inputs = (probe_coords @ basis).astype(np.float32)
    probe_targets = np.asarray(oracle.eval(probe_inputs, stage="weak-atom-boundary-probe"), dtype=np.float64)

    coords = np.vstack([train_coords, probe_coords])
    targets = np.concatenate([train_targets, probe_targets])

    # Freeze the strong atoms by subtracting exactly what they contribute.
    strong_activation = silu_np(coords @ gates[strong_index].T) * (coords @ ups[strong_index].T)
    strong_contribution = strong_activation @ scales[strong_index] + float(fit.get("bias", 0.0))
    weak_targets = targets - strong_contribution

    weak_candidate = {
        "gates": gates[weak_index].copy(),
        "ups": ups[weak_index].copy(),
        "scales": scales[weak_index].copy(),
        "bias": 0.0,
    }
    # polish_swiglu_candidate reads score/mae off the candidate, so the sub-model is scored
    # on the remainder it has to explain before it is handed over.
    weak_candidate = rescore_swiglu_candidate_on_data(coords, weak_targets, weak_candidate, recalibrate_scales=True)
    polished = polish_swiglu_candidate(coords, weak_targets, weak_candidate, learning_rate, None, None, 0.0)
    polished = rescore_swiglu_candidate_on_data(coords, weak_targets, polished, recalibrate_scales=True)

    merged_gates = gates.copy()
    merged_ups = ups.copy()
    merged_scales = scales.copy()
    merged_gates[weak_index] = np.asarray(polished["gates"], dtype=np.float64)
    merged_ups[weak_index] = np.asarray(polished["ups"], dtype=np.float64)
    merged_scales[weak_index] = np.asarray(polished["scales"], dtype=np.float64)
    merged = dict(fit)
    merged.update({
        "gates": merged_gates,
        "ups": merged_ups,
        "scales": merged_scales,
        "bias": float(fit.get("bias", 0.0)) + float(polished.get("bias", 0.0)),
    })
    diagnostics = {
        "attempted": True,
        "weakAtomCount": int(weak_count),
        "probeRows": int(probe_coords.shape[0]),
        "weakStrengthShare": [float(strengths[int(i)] / float(np.max(strengths))) for i in weak_index],
    }
    return merged, diagnostics


def principal_cosines(basis: np.ndarray, truth_vectors: np.ndarray) -> np.ndarray:
    q1, _ = np.linalg.qr(basis.T)
    q2, _ = np.linalg.qr(truth_vectors.T)
    _, singular_values, _ = np.linalg.svd(q1.T @ q2, full_matrices=False)
    return singular_values


def write_basis_diagnostic(
    args: argparse.Namespace,
    oracle: OracleClient,
    basis: np.ndarray,
    eigenvalues: np.ndarray,
    energy_ratio: float,
    gradient_witness_count: int,
    basis_moment: str,
    cubic_directions: np.ndarray,
    cubic_scores: np.ndarray,
    output_json: Path,
    output_md: Path | None,
) -> bool:
    oracle.unlock_posthoc_truth("basis-diagnostic-recovery-frozen")
    truth_gate, truth_up = oracle.truth()
    if truth_gate.ndim == 1:
        truth_gate = np.expand_dims(truth_gate, axis=0)
    if truth_up.ndim == 1:
        truth_up = np.expand_dims(truth_up, axis=0)
    truth_basis = np.vstack([
        normalize(row) for row in truth_gate
    ] + [
        normalize(row) for row in truth_up
    ])
    cosines = principal_cosines(basis, truth_basis)
    minimum_cosine = float(np.min(cosines))
    cubic_relative_rmse: float | None = None
    cubic_cosine: float | None = None
    if cubic_directions.shape[0] > 0:
        truth_gate_coords = truth_gate @ basis.T
        truth_up_coords = truth_up @ basis.T
        true_cubic_scores = predict_swiglu_cubic_directional_derivatives(
            cubic_directions,
            truth_gate_coords,
            truth_up_coords,
            np.ones(truth_gate.shape[0], dtype=np.float64),
        )
        cubic_error = np.asarray(cubic_scores, dtype=np.float64) - true_cubic_scores
        cubic_relative_rmse = float(
            np.sqrt(np.mean(cubic_error ** 2))
            / max(np.sqrt(np.mean(true_cubic_scores ** 2)), 1e-15)
        )
        cubic_cosine = float(
            np.dot(cubic_scores, true_cubic_scores)
            / max(np.linalg.norm(cubic_scores) * np.linalg.norm(true_cubic_scores), 1e-15)
        )
    passed = (
        minimum_cosine >= args.min_subspace_cosine
        and (
            cubic_relative_rmse is None
            or cubic_relative_rmse <= args.max_cubic_witness_relative_rmse
        )
        and oracle.query_count <= args.max_oracle_calls
        and oracle.meta.get("solverAccess") == "oracle-only"
    )
    report = {
        "summary": {
            "pass": passed,
            "kind": "post-hoc-basis-diagnostic",
            "model": oracle.meta["model"],
            "modelType": oracle.meta["modelType"],
            "dimension": int(oracle.meta["hiddenSize"]),
            "layer": int(oracle.meta["layer"]),
            "units": oracle.meta.get("units", [oracle.meta["unit"]]),
            "unitCount": int(oracle.meta.get("unitCount", 1)),
            "oracleMode": oracle.meta.get("oracleMode", "selected-unit-sum"),
            "outputChannel": oracle.meta.get("outputChannel"),
            "activeRank": int(args.active_rank),
            "rawOracleCalls": int(oracle.query_count),
            "gradientMode": args.gradient_mode,
            "finiteDifferenceSteps": parse_step_list(
                args.finite_difference_steps,
                args.finite_difference_step,
            ),
            "gradientWitnessCount": int(gradient_witness_count),
            "basisMoment": basis_moment,
            "selectedGradientEnergyRatio": float(energy_ratio),
            "subspaceMinPrincipalCosine": minimum_cosine,
            "subspaceCosines": [float(value) for value in cosines],
            "eigenvalues": [float(value) for value in eigenvalues],
            "cubicDirectionCount": int(cubic_directions.shape[0]),
            "cubicWitnessRelativeRmse": cubic_relative_rmse,
            "cubicWitnessCosine": cubic_cosine,
        },
        # The unlock on the line above this function's recovery already put the evidence on the
        # client; it simply never reached the report, so this lane's 233 artifacts asserted the
        # invariant while the proof of it sat unread one attribute away.
        "accessAudit": oracle_access_audit(
            oracle,
            sourceIdsUsedBySolver=False,
            oracleOwnsWeights=True,
        ),
    }
    output_json.write_text(dump_report_json(report) + "\n", encoding="utf-8")
    if output_md:
        output_md.write_text(
            "\n".join([
                "# Blind SwiGLU Basis Diagnostic",
                "",
                f"- pass: `{str(passed).lower()}`",
                f"- gradient mode: `{args.gradient_mode}`",
                f"- gradient witnesses: `{gradient_witness_count}`",
                f"- minimum principal cosine: `{minimum_cosine}`",
                f"- cubic witness relative RMSE: `{cubic_relative_rmse}`",
                f"- cubic witness cosine: `{cubic_cosine}`",
                f"- raw oracle calls: `{oracle.query_count}`",
                "",
                "Source vectors are requested only after basis recovery and are used only for this diagnostic witness.",
                "",
            ]),
            encoding="utf-8",
        )
    print(json.dumps(report["summary"], indent=2))
    return passed


def maximum_weight_assignment(scores: np.ndarray) -> list[int]:
    matrix = np.asarray(scores, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] > matrix.shape[1]:
        raise RuntimeError("assignment scores must be a rows<=columns matrix")
    if matrix.shape[0] == 0:
        return []
    if not np.all(np.isfinite(matrix)):
        raise RuntimeError("assignment scores must be finite")

    # Hungarian algorithm over a shifted minimization cost. The returned item
    # at row i is the unique source column assigned to that recovered atom.
    cost = float(np.max(matrix)) - matrix
    row_count, column_count = cost.shape
    u = np.zeros(row_count + 1, dtype=np.float64)
    v = np.zeros(column_count + 1, dtype=np.float64)
    matched_row = np.zeros(column_count + 1, dtype=np.int64)
    predecessor = np.zeros(column_count + 1, dtype=np.int64)
    for row in range(1, row_count + 1):
        matched_row[0] = row
        minimum = np.full(column_count + 1, np.inf, dtype=np.float64)
        used = np.zeros(column_count + 1, dtype=bool)
        column = 0
        while True:
            used[column] = True
            current_row = int(matched_row[column])
            delta = np.inf
            next_column = 0
            for candidate_column in range(1, column_count + 1):
                if used[candidate_column]:
                    continue
                reduced = cost[current_row - 1, candidate_column - 1] - u[current_row] - v[candidate_column]
                if reduced < minimum[candidate_column]:
                    minimum[candidate_column] = reduced
                    predecessor[candidate_column] = column
                if minimum[candidate_column] < delta:
                    delta = minimum[candidate_column]
                    next_column = candidate_column
            for candidate_column in range(column_count + 1):
                if used[candidate_column]:
                    u[matched_row[candidate_column]] += delta
                    v[candidate_column] -= delta
                else:
                    minimum[candidate_column] -= delta
            column = next_column
            if matched_row[column] == 0:
                break
        while True:
            previous_column = int(predecessor[column])
            matched_row[column] = matched_row[previous_column]
            column = previous_column
            if column == 0:
                break
    assignment = [-1] * row_count
    for column in range(1, column_count + 1):
        row = int(matched_row[column])
        if row > 0:
            assignment[row - 1] = column - 1
    if any(column < 0 for column in assignment):
        raise RuntimeError("assignment solver did not match every recovered atom")
    return assignment


def oracle_access_audit(
    oracle: Any,
    *,
    solver_oracle_calls: int | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Build an access audit that cannot state its claim without carrying its evidence.

    A report saying `oracle-only` on its own say-so proves that the tool spelled the string
    correctly and nothing else - audit finding F17, and 442 released reports carried exactly
    that. What the claim is actually about is already tracked by the client: it refuses
    `truth()` until an explicit unlock and records the query count at the moment of unlocking,
    so a reader can see that no truth was available while the recovery was still running.

    Two counters agreeing is the strong form, and it is only evidence when they are maintained
    independently - filling both from `oracle.query_count` would compare a number with itself,
    which is the same circularity in a new place. So the caller passes its own count or passes
    nothing, and the audit says which of the two it got rather than implying the stronger one.

    A run whose truth was never unlocked has not demonstrated the invariant at all. That is not
    a weaker `oracle-only`, it is an unproven one, and it is labelled so.
    """
    unlock_stage = getattr(oracle, "truth_unlock_stage", None)
    unlock_count = getattr(oracle, "truth_unlock_query_count", None)
    query_count = int(getattr(oracle, "query_count", 0))
    proven = unlock_stage is not None and unlock_count is not None
    audit: dict[str, Any] = {
        "solverAccess": "oracle-only" if proven else "oracle-only-unproven",
        "weightsUsedBySolver": False,
        "hooksUsedBySolver": False,
        "postHocWitnessOnly": True,
        "truthUnlockStage": unlock_stage,
        "truthUnlockQueryCount": unlock_count,
        "queryCountAtReport": query_count,
        "truthRequestCount": int(getattr(oracle, "truth_request_count", 0)),
        "counterAgreementChecked": solver_oracle_calls is not None,
    }
    if solver_oracle_calls is not None:
        audit["solverOracleCalls"] = int(solver_oracle_calls)
    audit.update(extra)
    return audit


def snapshot_access_audit(args: argparse.Namespace, oracle: Any) -> dict[str, Any]:
    """Measure the access state of the run that is about to write a snapshot.

    Every field here is read off the live run rather than asserted: the transport comes
    from the flags that actually configured it, and the truth-request count is the
    solver's own counter. `weightsUsedBySolver` and `hooksUsedBySolver` stay False because
    this solver has no code path that opens either, which the isolated lane independently
    attests with an strace witness; recording them keeps the snapshot self-describing.
    """
    return {
        "solverAccess": "oracle-only",
        "weightsUsedBySolver": False,
        "hooksUsedBySolver": False,
        "truthRequestCount": int(getattr(oracle, "truth_request_count", 0)),
        "externalOracleTransport": bool(args.oracle_socket) or (
            int(getattr(args, "oracle_read_fd", -1)) >= 0 and int(getattr(args, "oracle_write_fd", -1)) >= 0
        ),
    }


def save_fit_snapshot(
    path: str,
    *,
    basis: np.ndarray,
    train_coords: np.ndarray,
    train_targets: np.ndarray,
    fit: dict[str, Any],
    dimension: int,
    active_rank: int,
    atom_count: int,
    gradient_witness_count: int,
    eigenvalues: np.ndarray,
    energy_ratio: float,
    fit_gradient_coords: np.ndarray,
    fit_gradient_targets: np.ndarray,
    cubic_directions: np.ndarray,
    cubic_targets: np.ndarray,
    cubic_holdout_directions: np.ndarray,
    cubic_holdout_targets: np.ndarray,
    access_audit: dict[str, Any],
) -> None:
    """Persist a fit snapshot together with the access state actually observed.

    The access fields used to be written as the constants "oracle-only" and False, and the
    loader then checked those same constants back. That check could only ever read what
    this writer had just written, so it proved nothing about how the snapshot was produced
    while reading, in code and in reports, as a preserved access invariant. The caller now
    has to supply measured state, which is what makes the loader's check mean something.
    """
    snapshot_path = Path(path)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schemaVersion": 2,
        "kind": "blind-swiglu-fit-snapshot",
        "dimension": int(dimension),
        "activeRank": int(active_rank),
        "atomCount": int(atom_count),
        "gradientWitnessCount": int(gradient_witness_count),
        "energyRatio": float(energy_ratio),
        "fitMae": float(fit.get("mae", 0.0)),
        "fitMse": float(fit.get("mse", 0.0)),
        "fitR2": float(fit.get("r2", 0.0)),
        "fitWorstError": float(fit.get("worstError", 0.0)),
        "fitSelectionScore": float(fit.get("selectionScore", fit.get("score", fit.get("mae", 0.0)))),
        "startKind": str(fit.get("startKind", "unknown")),
        "restartCount": int(fit.get("restartCount", 0)),
        "solverAccess": str(access_audit["solverAccess"]),
        "weightsUsedBySolver": bool(access_audit["weightsUsedBySolver"]),
        "hooksUsedBySolver": bool(access_audit["hooksUsedBySolver"]),
        "truthRequestCount": int(access_audit["truthRequestCount"]),
        "externalOracleTransport": bool(access_audit["externalOracleTransport"]),
    }
    np.savez_compressed(
        snapshot_path,
        metadata=np.array(json.dumps(metadata), dtype=np.str_),
        basis=np.asarray(basis, dtype=np.float64),
        train_coords=np.asarray(train_coords, dtype=np.float64),
        train_targets=np.asarray(train_targets, dtype=np.float64),
        gates=np.asarray(fit["gates"], dtype=np.float64),
        ups=np.asarray(fit["ups"], dtype=np.float64),
        scales=np.asarray(fit["scales"], dtype=np.float64),
        bias=np.array(float(fit["bias"]), dtype=np.float64),
        eigenvalues=np.asarray(eigenvalues, dtype=np.float64),
        fit_gradient_coords=np.asarray(fit_gradient_coords, dtype=np.float64),
        fit_gradient_targets=np.asarray(fit_gradient_targets, dtype=np.float64),
        cubic_directions=np.asarray(cubic_directions, dtype=np.float64),
        cubic_targets=np.asarray(cubic_targets, dtype=np.float64),
        cubic_holdout_directions=np.asarray(cubic_holdout_directions, dtype=np.float64),
        cubic_holdout_targets=np.asarray(cubic_holdout_targets, dtype=np.float64),
    )


def load_fit_snapshot(path: str, *, dimension: int, active_rank: int, atom_count: int) -> dict[str, Any]:
    snapshot_path = Path(path)
    if not snapshot_path.exists():
        raise RuntimeError(f"fit snapshot not found: {snapshot_path}")
    with np.load(snapshot_path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"].item()))
        if metadata.get("kind") != "blind-swiglu-fit-snapshot":
            raise RuntimeError(f"invalid fit snapshot kind: {metadata.get('kind')!r}")
        if int(metadata.get("dimension", -1)) != int(dimension):
            raise RuntimeError(f"snapshot dimension mismatch: {metadata.get('dimension')} != {dimension}")
        if int(metadata.get("activeRank", -1)) != int(active_rank):
            raise RuntimeError(f"snapshot activeRank mismatch: {metadata.get('activeRank')} != {active_rank}")
        if int(metadata.get("atomCount", -1)) != int(atom_count):
            raise RuntimeError(f"snapshot atomCount mismatch: {metadata.get('atomCount')} != {atom_count}")
        # Now a real check: the writer records measured access state, so a snapshot
        # produced by a run that touched weights, hooks, or the truth channel is refused
        # here instead of being waved through by a constant it wrote itself.
        if metadata.get("solverAccess") != "oracle-only" or metadata.get("weightsUsedBySolver") is not False:
            raise RuntimeError("fit snapshot does not preserve oracle-only access invariant")
        if metadata.get("hooksUsedBySolver") is True:
            raise RuntimeError("fit snapshot was produced by a run that used hooks")
        if int(metadata.get("truthRequestCount", 0)) != 0:
            raise RuntimeError(
                f"fit snapshot was produced by a run that requested truth {metadata.get('truthRequestCount')} times"
            )
        basis = np.asarray(data["basis"], dtype=np.float64)
        train_coords = np.asarray(data["train_coords"], dtype=np.float64)
        train_targets = np.asarray(data["train_targets"], dtype=np.float64)
        gates = np.asarray(data["gates"], dtype=np.float64)
        ups = np.asarray(data["ups"], dtype=np.float64)
        scales = np.asarray(data["scales"], dtype=np.float64)
        if basis.shape != (active_rank, dimension):
            raise RuntimeError(f"snapshot basis shape mismatch: {basis.shape}")
        if gates.shape != (atom_count, active_rank) or ups.shape != gates.shape or scales.shape != (atom_count,):
            raise RuntimeError("snapshot fit atom shapes do not match activeRank/atomCount")
        if train_coords.ndim != 2 or train_coords.shape[1] != active_rank or train_targets.shape[0] != train_coords.shape[0]:
            raise RuntimeError("snapshot train set shapes are inconsistent")
        for direction_key, target_key in (
            ("cubic_directions", "cubic_targets"),
            ("cubic_holdout_directions", "cubic_holdout_targets"),
        ):
            if direction_key not in data and target_key not in data:
                continue
            if direction_key not in data or target_key not in data:
                raise RuntimeError(f"snapshot has incomplete {direction_key} witness data")
            directions = np.asarray(data[direction_key], dtype=np.float64)
            targets = np.asarray(data[target_key], dtype=np.float64)
            if (
                directions.ndim != 2
                or directions.shape[1] != active_rank
                or targets.shape != (directions.shape[0],)
            ):
                raise RuntimeError(f"snapshot {direction_key} witness shapes are inconsistent")
        fit = {
            "gates": gates,
            "ups": ups,
            "scales": scales,
            "bias": float(np.asarray(data["bias"]).item()),
            "mae": float(metadata.get("fitMae", 0.0)),
            "mse": float(metadata.get("fitMse", 0.0)),
            "r2": float(metadata.get("fitR2", 0.0)),
            "worstError": float(metadata.get("fitWorstError", 0.0)),
            "score": float(metadata.get("fitMae", 0.0)),
            "selectionScore": float(metadata.get("fitSelectionScore", metadata.get("fitMae", 0.0))),
            "startKind": str(metadata.get("startKind", "snapshot")),
            "restartCount": int(metadata.get("restartCount", 0)),
            "randomRestartCount": int(metadata.get("restartCount", 0)),
            "structuredStartCount": 0,
            "restartChunkSize": 0,
            "randomInitScales": [],
            "bestScoreByStartKind": {},
            "bestMetricsByStartKind": {},
            "fitMode": "snapshot-replay",
        }
        return {
            "metadata": metadata,
            "basis": basis,
            "trainCoords": train_coords,
            "trainTargets": train_targets,
            "fit": fit,
            "eigenvalues": np.asarray(data["eigenvalues"], dtype=np.float64),
            "fitGradientCoords": np.asarray(data["fit_gradient_coords"], dtype=np.float64),
            "fitGradientTargets": np.asarray(data["fit_gradient_targets"], dtype=np.float64),
            "cubicDirections": (
                np.asarray(data["cubic_directions"], dtype=np.float64)
                if "cubic_directions" in data
                else np.zeros((0, active_rank), dtype=np.float64)
            ),
            "cubicTargets": (
                np.asarray(data["cubic_targets"], dtype=np.float64)
                if "cubic_targets" in data
                else np.zeros(0, dtype=np.float64)
            ),
            "cubicHoldoutDirections": (
                np.asarray(data["cubic_holdout_directions"], dtype=np.float64)
                if "cubic_holdout_directions" in data
                else np.zeros((0, active_rank), dtype=np.float64)
            ),
            "cubicHoldoutTargets": (
                np.asarray(data["cubic_holdout_targets"], dtype=np.float64)
                if "cubic_holdout_targets" in data
                else np.zeros(0, dtype=np.float64)
            ),
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def calibrate_input_scale(
    oracle: Any,
    dimension: int,
    *,
    candidates: list[float],
    probe_count: int,
    quantization_ratio_floor: float,
    seed: int,
) -> dict[str, Any]:
    """Choose the training input scale from oracle-visible output amplitude alone.

    A fixed absolute input scale is not selection-independent. Blocks differ in output
    amplitude by orders of magnitude, and a scale tuned for a strong block leaves a weak
    block's signal too close to the oracle's quantization floor to fit accurately, while
    a scale large enough for the weak block drives a strong block out of the regime where
    structure recovery succeeds.

    This walks the candidate scales from smallest to largest and stops at the first one
    whose measured output standard deviation clears `quantization_ratio_floor` multiples
    of the oracle's own quantization step. Tying the floor to the quantization step rather
    than to an absolute constant keeps the rule meaningful at any oracle precision. If no
    candidate clears the floor, the largest candidate is used; the candidate list is the
    structure-safety cap, because too large a scale breaks recovery outright.

    Only oracle evaluations are used, so this stays inside the blind-access invariant:
    output amplitude is something any opaque caller can observe. Nothing is fitted here.
    The probe cost is charged to the same query budget as every other call.
    """
    if not candidates:
        raise RuntimeError("input-scale calibration requires at least one candidate scale")
    ordered = sorted({float(scale) for scale in candidates if float(scale) > 0.0})
    if not ordered:
        raise RuntimeError("input-scale calibration candidates must be positive")
    quantization_step = float(oracle.meta.get("oracleQuantizationStep", 0.0) or 0.0)
    calibration_rng = np.random.default_rng(seed)
    measurements: list[dict[str, Any]] = []
    chosen = ordered[-1]
    reason = "no-candidate-cleared-amplitude-floor"
    for scale in ordered:
        probes = calibration_rng.normal(0.0, scale, size=(int(probe_count), dimension)).astype(np.float32)
        values = np.asarray(oracle.eval(probes, stage="input-scale-calibration"), dtype=np.float64)
        output_std = float(np.std(values))
        ratio = float(output_std / quantization_step) if quantization_step > 0.0 else float("inf")
        measurements.append({
            "scale": float(scale),
            "outputStd": output_std,
            "quantizationRatio": ratio,
        })
        if ratio >= float(quantization_ratio_floor):
            chosen = float(scale)
            reason = "smallest-candidate-above-amplitude-floor"
            break
    return {
        "selectedInputScale": float(chosen),
        "reason": reason,
        "candidates": ordered,
        "probeCount": int(probe_count),
        "quantizationStep": quantization_step,
        "quantizationRatioFloor": float(quantization_ratio_floor),
        "measurements": measurements,
    }


def run_solver(args: argparse.Namespace) -> int:
    if args.recovery_path == "asymptotic-cold-start":
        if args.fit_snapshot_input:
            raise RuntimeError("asymptotic cold start forbids --fit-snapshot-input")
        if not args.disable_adaptive_rescue:
            raise RuntimeError("asymptotic cold start requires --disable-adaptive-rescue to forbid nonlinear fallback")
    if args.defer_posthoc_truth:
        if args.recovery_path != "asymptotic-cold-start":
            raise RuntimeError("--defer-posthoc-truth requires --recovery-path asymptotic-cold-start")
        external_opaque_transport = bool(args.oracle_socket) or (
            args.oracle_read_fd >= 0 and args.oracle_write_fd >= 0
        )
        if not external_opaque_transport or not args.require_opaque_oracle:
            raise RuntimeError("--defer-posthoc-truth requires an external opaque oracle transport")
        if not args.frozen_fit_output:
            raise RuntimeError("--defer-posthoc-truth requires --frozen-fit-output")
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    if output_md:
        output_md.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="gcl-blind-swiglu-") as raw_tmp:
        work_dir = Path(raw_tmp)
        oracle = create_oracle_client(args, work_dir)
        try:
            dimension = int(oracle.meta["hiddenSize"])
            unit_count = int(oracle.meta.get("unitCount", 1))
            log_progress(args, "oracle-ready", dimension=dimension, unitCount=unit_count, layer=oracle.meta.get("layer"))
            if args.expected_dimension > 0 and dimension != args.expected_dimension:
                raise RuntimeError(f"oracle dimension={dimension} does not match expected={args.expected_dimension}")
            if args.atom_count is None:
                args.atom_count = unit_count
            if args.atom_count != unit_count:
                raise RuntimeError(f"atom_count={args.atom_count} does not match oracle unit_count={unit_count}")
            input_scale_calibration: dict[str, Any] | None = None
            if args.auto_input_scale:
                input_scale_calibration = calibrate_input_scale(
                    oracle,
                    dimension,
                    candidates=parse_float_list(args.auto_input_scale_candidates),
                    probe_count=args.auto_input_scale_probe_count,
                    quantization_ratio_floor=args.auto_input_scale_quantization_ratio,
                    seed=args.seed + 90210,
                )
                args.input_scale = float(input_scale_calibration["selectedInputScale"])
                log_progress(
                    args,
                    "input-scale-calibration",
                    selectedInputScale=args.input_scale,
                    reason=input_scale_calibration["reason"],
                    measurements=input_scale_calibration["measurements"],
                    rawOracleCalls=int(oracle.query_count),
                )
            rng = np.random.default_rng(args.seed)
            independent_rng = os.environ.get("GCL_BLIND_SWIGLU_INDEPENDENT_RNG", "").strip() == "1"
            basis_rng = np.random.default_rng(args.seed + 101) if independent_rng else rng
            cubic_rng = np.random.default_rng(args.seed + 202) if independent_rng else rng
            cubic_holdout_rng = np.random.default_rng(args.seed + 203) if independent_rng else rng
            train_rng = np.random.default_rng(args.seed + 303) if independent_rng else rng
            fit_gradient_rng = np.random.default_rng(args.seed + 404) if independent_rng else rng
            validation_rng = np.random.default_rng(args.seed + 505) if independent_rng else rng
            finite_difference_steps = parse_step_list(args.finite_difference_steps, args.finite_difference_step)
            cubic_steps = parse_step_list(args.cubic_steps, args.cubic_step)
            fit_gradient_steps = parse_step_list(
                args.fit_gradient_steps,
                args.fit_gradient_step if args.fit_gradient_step > 0 else args.finite_difference_step,
            )
            active_gradient_estimator = (
                estimate_active_symmetric_regression_gradients
                if args.fit_gradient_mode == "symmetric-regression"
                else estimate_active_forward_gradients
            )
            train_scale_mixture = parse_float_list(args.train_scale_mixture)
            if args.fit_snapshot_input:
                if not args.disable_adaptive_rescue:
                    raise RuntimeError("--fit-snapshot-input requires --disable-adaptive-rescue; replay mode does not carry full input-space train samples")
                snapshot = load_fit_snapshot(
                    args.fit_snapshot_input,
                    dimension=dimension,
                    active_rank=args.active_rank,
                    atom_count=args.atom_count,
                )
                basis = snapshot["basis"]
                train_coords = snapshot["trainCoords"]
                train_targets = snapshot["trainTargets"]
                train_inputs = np.zeros((0, dimension), dtype=np.float32)
                fit = snapshot["fit"]
                fit_gradient_coords = snapshot["fitGradientCoords"]
                fit_gradient_targets = snapshot["fitGradientTargets"]
                eigenvalues = snapshot["eigenvalues"]
                energy_ratio = float(snapshot["metadata"].get("energyRatio", 0.0))
                gradient_witness_count = int(snapshot["metadata"].get("gradientWitnessCount", 0))
                basis_moment = f"snapshot:{args.basis_moment}"
                cubic_gate_directions = snapshot["cubicDirections"]
                cubic_scores = snapshot["cubicTargets"]
                cubic_holdout_directions = snapshot["cubicHoldoutDirections"]
                cubic_holdout_scores = snapshot["cubicHoldoutTargets"]
                train_sampling_mode = f"snapshot:{args.train_sampling_mode.strip().lower()}"
                fit_selection_count = 0
                fit_selection_scale = 0.0
                fit_selection_coords = None
                fit_selection_targets = None
                log_progress(
                    args,
                    "fit-snapshot-loaded",
                    path=str(args.fit_snapshot_input),
                    trainCount=int(train_coords.shape[0]),
                    fitTrainMae=float(fit.get("mae", 0.0)),
                    rawOracleCalls=int(oracle.query_count),
                )
            else:
                gradients = []
                for _ in range(args.probe_points):
                    base = basis_rng.normal(0.0, args.input_scale, size=dimension).astype(np.float32)
                    if args.gradient_mode == "symmetric-multiscale-coordinate":
                        gradients.extend(estimate_symmetric_multiscale_coordinate_gradients(
                            oracle,
                            base,
                            finite_difference_steps,
                        ))
                        continue
                    if args.gradient_mode == "symmetric-regression-coordinate":
                        gradients.append(estimate_symmetric_regression_coordinate_gradient(
                            oracle,
                            base,
                            finite_difference_steps,
                        ))
                        continue
                    if args.gradient_mode == "symmetric-richardson-coordinate":
                        gradients.append(estimate_symmetric_richardson_coordinate_gradient(
                            oracle,
                            base,
                            finite_difference_steps,
                        ))
                        continue
                    for step in finite_difference_steps:
                        if args.gradient_mode == "coordinate":
                            gradients.append(estimate_coordinate_gradient(oracle, base, step))
                        elif args.gradient_mode == "forward-coordinate":
                            gradients.append(estimate_forward_coordinate_gradient(oracle, base, step))
                        else:
                            gradients.append(estimate_gradient(oracle, basis_rng, base, args.direction_count, step))
                gradient_matrix = np.vstack(gradients)
                basis_moment = args.basis_moment
                if basis_moment == "auto" and args.gradient_mode == "forward-coordinate":
                    basis_moment = "raw"
                if basis_moment == "centered" and gradient_matrix.shape[0] <= args.active_rank:
                    basis_moment = "raw"
                basis, eigenvalues, energy_ratio = recover_basis(gradient_matrix, args.active_rank, basis_moment)
                gradient_witness_count = int(gradient_matrix.shape[0])
                log_progress(
                    args,
                    "basis-recovered",
                    activeRank=int(args.active_rank),
                    gradientWitnessCount=gradient_witness_count,
                    energyRatio=float(energy_ratio),
                )
                cubic_gate_directions, cubic_scores = estimate_active_cubic_directions(
                    oracle,
                    cubic_rng,
                    basis,
                    args.cubic_direction_count,
                    cubic_steps,
                )
                cubic_holdout_directions, cubic_holdout_scores = estimate_active_cubic_directions(
                    oracle,
                    cubic_holdout_rng,
                    basis,
                    args.cubic_holdout_count,
                    cubic_steps,
                )
                if args.basis_diagnostic_only:
                    return 0 if write_basis_diagnostic(
                        args,
                        oracle,
                        basis,
                        eigenvalues,
                        energy_ratio,
                        gradient_witness_count,
                        basis_moment,
                        cubic_gate_directions,
                        cubic_scores,
                        output_json,
                        output_md,
                    ) else 1

                train_sampling_mode = args.train_sampling_mode.strip().lower()
                train_inputs = sample_active_region_inputs(
                    train_rng,
                    args.train_count,
                    basis,
                    args.input_scale,
                    train_scale_mixture,
                    train_sampling_mode,
                )
                train_targets = oracle.eval(train_inputs)
                train_coords = train_inputs @ basis.T
                log_progress(args, "train-sampled", trainCount=int(train_coords.shape[0]), rawOracleCalls=int(oracle.query_count))
                fit_gradient_coords, fit_gradient_targets = active_gradient_estimator(
                    oracle,
                    fit_gradient_rng,
                    basis,
                    args.fit_gradient_count,
                    args.input_scale,
                    fit_gradient_steps,
                )
                fit_selection_count = max(0, int(args.fit_selection_count))
                if fit_selection_count > 0:
                    fit_selection_scale = args.fit_selection_scale if args.fit_selection_scale > 0.0 else args.validation_scale
                    fit_selection_inputs = validation_rng.normal(
                        0.0,
                        fit_selection_scale,
                        size=(fit_selection_count, dimension),
                    ).astype(np.float32)
                    fit_selection_targets = oracle.eval(fit_selection_inputs)
                    fit_selection_coords = fit_selection_inputs @ basis.T
                    log_progress(
                        args,
                        "fit-selection-sampled",
                        selectionCount=int(fit_selection_count),
                        selectionScale=float(fit_selection_scale),
                        rawOracleCalls=int(oracle.query_count),
                    )
                else:
                    fit_selection_scale = 0.0
                    fit_selection_coords = None
                    fit_selection_targets = None
                if args.recovery_path == "asymptotic-cold-start":
                    asymptotic_line_count = int(os.environ.get(
                        "GCL_BLIND_SWIGLU_ASYMPTOTIC_STITCH_LINE_COUNT",
                        "24",
                    ) or "24")
                    asymptotic_samples_per_line = int(os.environ.get(
                        "GCL_BLIND_SWIGLU_ASYMPTOTIC_STITCH_SAMPLES_PER_LINE",
                        "1537",
                    ) or "1537")
                    asymptotic_line_range = float(os.environ.get(
                        "GCL_BLIND_SWIGLU_ASYMPTOTIC_STITCH_LINE_RANGE",
                        "6.0",
                    ) or "6.0")
                    asymptotic_offset_scale = float(os.environ.get(
                        "GCL_BLIND_SWIGLU_ASYMPTOTIC_STITCH_OFFSET_SCALE",
                        "0.75",
                    ) or "0.75")
                    asymptotic_scale = float(os.environ.get(
                        "GCL_BLIND_SWIGLU_ASYMPTOTIC_STITCH_SCALE",
                        "512.0",
                    ) or "512.0")
                    hessian_representatives = int(os.environ.get(
                        "GCL_BLIND_SWIGLU_ASYMPTOTIC_STITCH_HESSIAN_REPRESENTATIVES",
                        "5",
                    ) or "5")
                    asymptotic_query_bound = asymptotic_stitch_query_upper_bound(
                        asymptotic_line_count,
                        asymptotic_samples_per_line,
                        args.atom_count,
                        args.active_rank,
                        hessian_representatives,
                    )
                    mandatory_validation_reserve = max(1, int(args.validation_count))
                    requested_with_reserve = asymptotic_query_bound + mandatory_validation_reserve
                    if requested_with_reserve > args.max_oracle_calls - oracle.query_count:
                        raise OracleBudgetExceeded(
                            stage="asymptotic-cold-start-with-validation-reserve",
                            requested=requested_with_reserve,
                            consumed=oracle.query_count,
                            limit=args.max_oracle_calls,
                        )
                    fit = make_asymptotic_cold_start_scaffold(
                        args.atom_count,
                        args.active_rank,
                        train_targets,
                    )
                    log_progress(
                        args,
                        "asymptotic-cold-start",
                        lineCount=asymptotic_line_count,
                        samplesPerLine=asymptotic_samples_per_line,
                        requestedOracleCallUpperBound=asymptotic_query_bound,
                        mandatoryValidationReserve=mandatory_validation_reserve,
                        rawOracleCalls=int(oracle.query_count),
                    )
                    # The probe line range that best beats the quantization floor is
                    # block-dependent: widening it from 6 to 12 lifts mistral layer 28 from
                    # R2 0.978190 to 0.992712, while llama layer 1 is slightly better at 6.
                    # A single constant therefore leaves accuracy on the table for one block
                    # or the other, the same way a single input scale did.
                    #
                    # Each candidate range is stitched and scored on the solver's own
                    # selection sample, which is a separate oracle draw from the validation
                    # set the gate measures, so nothing here fits to the graded points. The
                    # scaffold is rebuilt per attempt so an attempt cannot inherit the
                    # previous one's state, and a candidate is only tried when its query
                    # bound still fits inside the remaining budget.
                    stitch_scaffold = fit
                    range_ladder: list[float] = [float(asymptotic_line_range)]
                    ladder_raw = os.environ.get("GCL_BLIND_SWIGLU_ASYMPTOTIC_STITCH_LINE_RANGE_LADDER", "").strip()
                    for value in (float(item) for item in ladder_raw.split(",") if item.strip()):
                        if value > 0.0 and value not in range_ladder:
                            range_ladder.append(value)

                    def score_stitched_candidate(candidate: dict[str, Any]) -> float:
                        if fit_selection_coords is None or fit_selection_targets is None:
                            score_coords, score_targets = train_coords, train_targets
                        else:
                            score_coords, score_targets = fit_selection_coords, fit_selection_targets
                        try:
                            prediction = predict_swiglu(score_coords, candidate)
                        except (ValueError, KeyError) as exc:
                            # An unusable candidate scores as infinitely bad, which is the
                            # intended outcome - but until now so did a defect in the predictor,
                            # and this runs once per candidate so the failure had no voice at all.
                            _record_fallback(f"candidate scored as unusable: {type(exc).__name__}: {exc}")
                            return float("inf")
                        if prediction.shape != score_targets.shape or not np.all(np.isfinite(prediction)):
                            return float("inf")
                        return float(np.mean(np.abs(prediction - score_targets)))

                    fit = None
                    best_score = float("inf")
                    attempted_ranges: list[dict[str, Any]] = []
                    for candidate_range in range_ladder:
                        if attempted_ranges:
                            remaining = args.max_oracle_calls - oracle.query_count - mandatory_validation_reserve
                            if asymptotic_query_bound > remaining:
                                break
                        candidate_fit = asymptotic_gate_stitch_swiglu_candidate(
                            oracle,
                            basis,
                            train_coords,
                            train_targets,
                            dict(stitch_scaffold),
                            args.seed + 71021,
                            selection_coords=fit_selection_coords,
                            selection_targets=fit_selection_targets,
                            gradient_coords=fit_gradient_coords,
                            gradient_targets=fit_gradient_targets,
                            cubic_directions=cubic_gate_directions,
                            cubic_targets=cubic_scores,
                            line_count=asymptotic_line_count,
                            samples_per_line=asymptotic_samples_per_line,
                            line_range=candidate_range,
                            offset_scale=asymptotic_offset_scale,
                            asymptotic_scale=asymptotic_scale,
                            cold_start=True,
                        )
                        stitched = bool(candidate_fit.get("asymptoticGateStitched", False))
                        candidate_score = score_stitched_candidate(candidate_fit) if stitched else float("inf")
                        attempted_ranges.append({
                            "lineRange": float(candidate_range),
                            "stitched": stitched,
                            "selectionMae": None if candidate_score == float("inf") else candidate_score,
                            "rawOracleCalls": int(oracle.query_count),
                        })
                        if fit is None or candidate_score < best_score:
                            fit = candidate_fit
                            best_score = candidate_score
                    if len(attempted_ranges) > 1:
                        log_progress(
                            args,
                            "asymptotic-stitch-range-search",
                            attempts=attempted_ranges,
                            selectedLineRange=next(
                                (row["lineRange"] for row in attempted_ranges
                                 if row["selectionMae"] is not None and row["selectionMae"] == best_score),
                                float(asymptotic_line_range),
                            ),
                            rawOracleCalls=int(oracle.query_count),
                        )
                    if not fit.get("asymptoticGateStitched", False):
                        failure = str(fit.get("asymptoticGateStitchFailure", "structure-court-rejected"))
                        # The stitch already computed why it refused: usable lines, boundary
                        # normals, and recovered cluster count. Raising without emitting them
                        # discards the only evidence that says which stage to repair, and
                        # leaves a caller with a bare "structure-court-rejected" string.
                        log_progress(
                            args,
                            "asymptotic-cold-start-rejected",
                            failure=failure,
                            atomCount=int(args.atom_count),
                            lineCount=fit.get("asymptoticGateStitchLineCount"),
                            usableLineCount=fit.get("asymptoticGateStitchUsableLineCount"),
                            failedLineCount=fit.get("asymptoticGateStitchFailedLineCount"),
                            boundaryNormalCount=fit.get("asymptoticGateStitchBoundaryNormalCount"),
                            clusterCount=fit.get("asymptoticGateStitchClusterCount"),
                            clusterRejectReason=_CLUSTER_LAST_REASON.get("reason"),
                            inputScale=float(args.input_scale),
                            rawOracleCalls=int(oracle.query_count),
                        )
                        raise RuntimeError(f"asymptotic cold start failed: {failure}")

                    weak_probe_count = int(os.environ.get("GCL_BLIND_SWIGLU_WEAK_ATOM_PROBE_COUNT", "0") or "0")
                    if weak_probe_count > 0:
                        # How many atoms count as weak is block-dependent. A quarter of them
                        # is accepted on llama layer 1 and a half is rejected there, so the
                        # boundary sits between, and it has no reason to sit in the same
                        # place on another block. Each fraction is tried and scored on the
                        # solver's own selection sample, which is a separate oracle draw
                        # from the graded validation set, and the best is kept only when it
                        # beats leaving the fit alone.
                        weak_fraction_ladder = [
                            float(item) for item in
                            (os.environ.get("GCL_BLIND_SWIGLU_WEAK_ATOM_PROBE_FRACTION", "0.25") or "0.25").split(",")
                            if item.strip()
                        ]
                        before_score = score_stitched_candidate(fit)
                        best_weak_candidate: dict[str, Any] | None = None
                        best_weak_score = before_score
                        weak_attempts: list[dict[str, Any]] = []
                        for fraction in weak_fraction_ladder:
                            weak_candidate, weak_diagnostics = refit_weak_atoms_on_boundary_probes(
                                np.random.default_rng(args.seed + 51407),
                                oracle,
                                basis,
                                fit,
                                train_coords,
                                train_targets,
                                probe_count_per_atom=weak_probe_count,
                                atom_fraction=fraction,
                                probe_scale=float(args.input_scale),
                                boundary_jitter=float(os.environ.get("GCL_BLIND_SWIGLU_WEAK_ATOM_PROBE_JITTER", "0.05") or "0.05"),
                                learning_rate=args.fit_learning_rate,
                            )
                            if weak_candidate is None:
                                continue
                            candidate_score = score_stitched_candidate(weak_candidate)
                            weak_attempts.append({
                                "atomFraction": float(fraction),
                                "weakAtomCount": weak_diagnostics.get("weakAtomCount"),
                                "selectionMae": None if candidate_score == float("inf") else candidate_score,
                            })
                            if candidate_score < best_weak_score:
                                best_weak_candidate = weak_candidate
                                best_weak_score = candidate_score
                        if weak_attempts:
                            accepted = best_weak_candidate is not None
                            if accepted:
                                fit = best_weak_candidate
                            log_progress(
                                args,
                                "weak-atom-boundary-refit",
                                attempts=weak_attempts,
                                selectionMaeBefore=None if before_score == float("inf") else before_score,
                                selectionMaeAfter=None if best_weak_score == float("inf") else best_weak_score,
                                accepted=bool(accepted),
                                rawOracleCalls=int(oracle.query_count),
                            )
                else:
                    fit = fit_swiglu_sum(
                        train_coords,
                        train_targets,
                        args.atom_count,
                        args.seed + 17,
                        args.fit_steps,
                        args.fit_learning_rate,
                        args.fit_restarts if args.fit_restarts > 0 else None,
                        fit_gradient_coords,
                        fit_gradient_targets,
                        cubic_gate_directions,
                        args.fit_gradient_weight,
                        args.fit_mode,
                        args.fit_restart_chunk_size,
                        fit_selection_coords,
                        fit_selection_targets,
                        cubic_selection_directions=cubic_holdout_directions,
                        cubic_selection_scores=cubic_holdout_scores,
                        cubic_directions=cubic_gate_directions,
                        cubic_targets=cubic_scores,
                    )
                log_progress(
                    args,
                    "initial-fit-complete",
                    fitTrainMae=float(fit["mae"]),
                    fitRestarts=int(fit["restartCount"]),
                    selectedStartKind=str(fit.get("startKind", "unknown")),
                )
                if args.fit_snapshot_output:
                    save_fit_snapshot(
                        args.fit_snapshot_output,
                        basis=basis,
                        train_coords=train_coords,
                        train_targets=train_targets,
                        fit=fit,
                        dimension=dimension,
                        active_rank=args.active_rank,
                        atom_count=args.atom_count,
                        gradient_witness_count=gradient_witness_count,
                        eigenvalues=eigenvalues,
                        energy_ratio=energy_ratio,
                        fit_gradient_coords=fit_gradient_coords,
                        fit_gradient_targets=fit_gradient_targets,
                        cubic_directions=cubic_gate_directions,
                        cubic_targets=cubic_scores,
                        cubic_holdout_directions=cubic_holdout_directions,
                        cubic_holdout_targets=cubic_holdout_scores,
                        access_audit=snapshot_access_audit(args, oracle),
                    )
                    log_progress(
                        args,
                        "fit-snapshot-saved",
                        path=str(args.fit_snapshot_output),
                        rawOracleCalls=int(oracle.query_count),
                    )
            tail_probe_selected = 0
            tail_probe_count = 0
            prevalidation_fit_attempts: list[dict[str, Any]] = []
            if args.tail_probe_count > 0:
                tail_probe_scale = args.tail_probe_scale if args.tail_probe_scale > 0.0 else args.validation_scale
                tail_probe_rng = np.random.default_rng(args.seed + 99017)
                tail_probe_coords, tail_probe_targets, tail_probe_selected, tail_probe_count = make_tail_probe_polish_set(
                    tail_probe_rng,
                    oracle,
                    basis,
                    fit,
                    dimension,
                    args.tail_probe_count,
                    tail_probe_scale,
                    args.tail_probe_repeat,
                )
                if tail_probe_coords.shape[0] > 0:
                    polish_coords = np.vstack([train_coords, tail_probe_coords])
                    polish_targets = np.concatenate([train_targets, tail_probe_targets])
                    tail_probe_polish_steps = int(os.environ.get("GCL_BLIND_SWIGLU_TAIL_PROBE_POLISH_STEPS", "420") or "0")
                    if tail_probe_polish_steps > 0:
                        fit = tail_error_polish_swiglu_candidate(
                            polish_coords,
                            polish_targets,
                            fit,
                            tail_probe_polish_steps,
                        )
                        log_progress(
                            args,
                            "tail-probe-polish-complete",
                            probeCount=int(tail_probe_count),
                            selected=int(tail_probe_selected),
                            trainCount=int(polish_coords.shape[0]),
                            rawOracleCalls=int(oracle.query_count),
                        )
                if (
                    os.environ.get("GCL_BLIND_SWIGLU_TAIL_PROBE_REFIT", "").strip() == "1"
                    and tail_probe_coords.shape[0] > 0
                ):
                    tail_refit_coords = np.vstack([train_coords, tail_probe_coords])
                    tail_refit_targets = np.concatenate([train_targets, tail_probe_targets])
                    if os.environ.get("GCL_BLIND_SWIGLU_TAIL_PROBE_WARM_REFINE", "1").strip() != "0":
                        warm_baseline = rescore_swiglu_candidate_on_data(
                            tail_refit_coords,
                            tail_refit_targets,
                            fit,
                            recalibrate_scales=True,
                        )
                        warm_refined = polish_swiglu_candidate(
                            tail_refit_coords,
                            tail_refit_targets,
                            warm_baseline,
                            args.fit_learning_rate,
                            fit_gradient_coords,
                            fit_gradient_targets,
                            args.fit_gradient_weight,
                        )
                        warm_refined = rescore_swiglu_candidate_on_data(
                            tail_refit_coords,
                            tail_refit_targets,
                            warm_refined,
                            recalibrate_scales=True,
                        )
                        warm_refined = robust_scalar_calibrate_swiglu_candidate(
                            tail_refit_coords,
                            tail_refit_targets,
                            warm_refined,
                        )
                        warm_prediction = predict_swiglu(tail_refit_coords, warm_refined)
                        warm_metrics = regression_metrics(tail_refit_targets, warm_prediction)
                        warm_holdout_count = int(os.environ.get("GCL_BLIND_SWIGLU_TAIL_PROBE_WARM_HOLDOUT_COUNT", "2048") or "0")
                        warm_holdout_metrics: dict[str, float] | None = None
                        warm_holdout_baseline_metrics: dict[str, float] | None = None
                        if warm_holdout_count > 0 and oracle.query_count + warm_holdout_count <= args.max_oracle_calls:
                            warm_holdout_rng = np.random.default_rng(args.seed + 99043)
                            warm_holdout_inputs = warm_holdout_rng.normal(
                                0.0,
                                tail_probe_scale,
                                size=(warm_holdout_count, dimension),
                            ).astype(np.float32)
                            warm_holdout_targets = oracle.eval(warm_holdout_inputs)
                            warm_holdout_coords = warm_holdout_inputs @ basis.T
                            warm_holdout_baseline_prediction = predict_swiglu(warm_holdout_coords, fit)
                            warm_holdout_prediction = predict_swiglu(warm_holdout_coords, warm_refined)
                            warm_holdout_baseline_metrics = regression_metrics(warm_holdout_targets, warm_holdout_baseline_prediction)
                            warm_holdout_metrics = regression_metrics(warm_holdout_targets, warm_holdout_prediction)
                        accept_warm = (
                            warm_holdout_metrics is not None
                            and warm_holdout_baseline_metrics is not None
                            and warm_holdout_metrics["mae"] <= warm_holdout_baseline_metrics["mae"]
                            and warm_holdout_metrics["worstError"] <= warm_holdout_baseline_metrics["worstError"]
                        )
                        prevalidation_fit_attempts.append({
                            "stage": "tail-probe-warm-refine",
                            "trainCount": int(tail_refit_coords.shape[0]),
                            "tailProbeCount": int(tail_probe_count),
                            "tailProbeSelected": int(tail_probe_selected),
                            "baselineTrainMae": float(warm_baseline["mae"]),
                            "baselineTrainWorstError": float(warm_baseline.get("worstError", float("inf"))),
                            "augmentedTrainR2": float(warm_metrics["r2"]),
                            "augmentedTrainMae": float(warm_metrics["mae"]),
                            "augmentedTrainWorstError": float(warm_metrics["worstError"]),
                            "holdoutCount": int(warm_holdout_count if warm_holdout_metrics is not None else 0),
                            "holdoutBaselineR2": None if warm_holdout_baseline_metrics is None else float(warm_holdout_baseline_metrics["r2"]),
                            "holdoutBaselineMae": None if warm_holdout_baseline_metrics is None else float(warm_holdout_baseline_metrics["mae"]),
                            "holdoutBaselineWorstError": None if warm_holdout_baseline_metrics is None else float(warm_holdout_baseline_metrics["worstError"]),
                            "holdoutR2": None if warm_holdout_metrics is None else float(warm_holdout_metrics["r2"]),
                            "holdoutMae": None if warm_holdout_metrics is None else float(warm_holdout_metrics["mae"]),
                            "holdoutWorstError": None if warm_holdout_metrics is None else float(warm_holdout_metrics["worstError"]),
                            "accepted": bool(accept_warm),
                            "rawOracleCalls": int(oracle.query_count),
                        })
                        log_progress(
                            args,
                            "tail-probe-warm-refine-complete",
                            accepted=bool(accept_warm),
                            augmentedTrainR2=float(warm_metrics["r2"]),
                            augmentedTrainMae=float(warm_metrics["mae"]),
                            augmentedTrainWorstError=float(warm_metrics["worstError"]),
                            holdoutMae=None if warm_holdout_metrics is None else float(warm_holdout_metrics["mae"]),
                            holdoutWorstError=None if warm_holdout_metrics is None else float(warm_holdout_metrics["worstError"]),
                            rawOracleCalls=int(oracle.query_count),
                        )
                        if accept_warm:
                            fit = warm_refined
                    if os.environ.get("GCL_BLIND_SWIGLU_TAIL_PROBE_REFIT_RANDOM", "1").strip() != "0":
                        tail_refit_steps = int(os.environ.get(
                            "GCL_BLIND_SWIGLU_TAIL_PROBE_REFIT_STEPS",
                            str(args.fit_steps),
                        ) or args.fit_steps)
                        tail_refit_restarts = int(os.environ.get(
                            "GCL_BLIND_SWIGLU_TAIL_PROBE_REFIT_RESTARTS",
                            str(args.fit_restarts if args.fit_restarts > 0 else max(1, args.atom_count * 48)),
                        ) or "0")
                        log_progress(
                            args,
                            "tail-probe-refit-start",
                            trainCount=int(tail_refit_coords.shape[0]),
                            probeCount=int(tail_probe_count),
                            selected=int(tail_probe_selected),
                            fitSteps=int(tail_refit_steps),
                            fitRestarts=int(tail_refit_restarts),
                            rawOracleCalls=int(oracle.query_count),
                        )
                        tail_refit = fit_swiglu_sum(
                            tail_refit_coords,
                            tail_refit_targets,
                            args.atom_count,
                            args.seed + 99031,
                            max(1, tail_refit_steps),
                            args.fit_learning_rate,
                            max(1, tail_refit_restarts),
                            fit_gradient_coords,
                            fit_gradient_targets,
                            cubic_gate_directions,
                            args.fit_gradient_weight,
                            args.fit_mode,
                            args.fit_restart_chunk_size,
                            fit_selection_coords,
                            fit_selection_targets,
                            cubic_selection_directions=cubic_holdout_directions,
                            cubic_selection_scores=cubic_holdout_scores,
                            cubic_directions=cubic_gate_directions,
                            cubic_targets=cubic_scores,
                        )
                        tail_refit_prediction = predict_swiglu(tail_refit_coords, tail_refit)
                        tail_refit_metrics = regression_metrics(tail_refit_targets, tail_refit_prediction)
                        accept_tail_refit = (
                            float(tail_refit.get("selectionScore", tail_refit.get("score", tail_refit["mae"])))
                            <= float(fit.get("selectionScore", fit.get("score", fit["mae"])))
                            or tail_refit_metrics["mae"] < float(fit["mae"])
                        )
                        prevalidation_fit_attempts.append({
                            "stage": "tail-probe-refit",
                            "trainCount": int(tail_refit_coords.shape[0]),
                            "tailProbeCount": int(tail_probe_count),
                            "tailProbeSelected": int(tail_probe_selected),
                            "fitSteps": int(tail_refit_steps),
                            "fitRestarts": int(tail_refit["restartCount"]),
                            "fitRandomRestarts": int(tail_refit.get("randomRestartCount", tail_refit["restartCount"])),
                            "fitStructuredStarts": int(tail_refit.get("structuredStartCount", 0)),
                            "selectedStartKind": str(tail_refit.get("startKind", "unknown")),
                            "fitTrainMae": float(tail_refit["mae"]),
                            "fitSelectionScore": float(tail_refit.get("selectionScore", tail_refit.get("score", tail_refit["mae"]))),
                            "augmentedTrainR2": float(tail_refit_metrics["r2"]),
                            "augmentedTrainMae": float(tail_refit_metrics["mae"]),
                            "augmentedTrainWorstError": float(tail_refit_metrics["worstError"]),
                            "accepted": bool(accept_tail_refit),
                            "rawOracleCalls": int(oracle.query_count),
                        })
                        log_progress(
                            args,
                            "tail-probe-refit-complete",
                            accepted=bool(accept_tail_refit),
                            augmentedTrainR2=float(tail_refit_metrics["r2"]),
                            augmentedTrainMae=float(tail_refit_metrics["mae"]),
                            augmentedTrainWorstError=float(tail_refit_metrics["worstError"]),
                            rawOracleCalls=int(oracle.query_count),
                        )
                        if accept_tail_refit:
                            fit = tail_refit

            validation_inputs = validation_rng.normal(0.0, args.validation_scale, size=(args.validation_count, dimension)).astype(np.float32)
            validation_targets = oracle.eval(validation_inputs)
            validation_coords = validation_inputs @ basis.T
            validation_prediction = predict_swiglu(validation_coords, fit)
            metrics = regression_metrics(validation_targets, validation_prediction)
            final_validation_source = "initial-validation"
            atom_tail_diagnostic: dict[str, Any] | None = None
            if os.environ.get("GCL_BLIND_SWIGLU_ATOM_TAIL_DIAGNOSTIC", "").strip() == "1":
                atom_tail_diagnostic = atom_tail_diagnostics(
                    validation_coords,
                    validation_targets,
                    fit,
                    validation_prediction,
                    float(os.environ.get("GCL_BLIND_SWIGLU_ATOM_TAIL_DIAGNOSTIC_FRACTION", "0.03") or "0.03"),
                    int(os.environ.get("GCL_BLIND_SWIGLU_ATOM_TAIL_DIAGNOSTIC_TOP_K", "8") or "8"),
                )
                log_progress(
                    args,
                    "atom-tail-diagnostic",
                    tailCount=int(atom_tail_diagnostic.get("tailCount", 0)),
                    tailWorstError=float(atom_tail_diagnostic.get("tailWorstError", 0.0)),
                    topAtoms=atom_tail_diagnostic.get("topAtoms", [])[:3],
                )
            log_progress(
                args,
                "validation-complete",
                r2=float(metrics["r2"]),
                mae=float(metrics["mae"]),
                worstError=float(metrics["worstError"]),
                rawOracleCalls=int(oracle.query_count),
            )
            fit_attempts = [{
                "stage": "initial",
                "trainCount": int(train_coords.shape[0]),
                "fitGradientCount": int(fit_gradient_coords.shape[0]),
                "fitSteps": int(args.fit_steps),
                "fitRestarts": int(fit["restartCount"]),
                "fitRandomRestarts": int(fit.get("randomRestartCount", fit["restartCount"])),
                "fitStructuredStarts": int(fit.get("structuredStartCount", 0)),
                "selectedStartKind": str(fit.get("startKind", "unknown")),
                "selectedRandomRestartIndex": fit.get("randomRestartIndex"),
                "bestScoreByStartKind": fit.get("bestScoreByStartKind", {}),
                "bestMetricsByStartKind": fit.get("bestMetricsByStartKind", {}),
                "fitMode": str(fit.get("fitMode", "unknown")),
                "fitTrainMae": float(fit["mae"]),
                "fitSelectionScore": float(fit.get("selectionScore", fit.get("score", fit["mae"]))),
                "fitSelectionCount": int(fit_selection_count),
                "fitSelectionScale": float(fit_selection_scale),
                "fitExternalSelectionCount": int(fit.get("externalSelectionCount", 0)),
                "extraGateCandidateCount": int(fit.get("extraGateCandidateCount", 0)),
                "tailProbeCount": int(tail_probe_count),
                "tailProbeSelected": int(tail_probe_selected),
                "tailProbeRepeat": int(args.tail_probe_repeat),
                "validationR2": float(metrics["r2"]),
                "validationMae": float(metrics["mae"]),
                "validationWorstError": float(metrics["worstError"]),
                "rawOracleCalls": int(oracle.query_count),
            }]
            fit_attempts.extend(prevalidation_fit_attempts)
            asymptotic_stitch_posthoc_directions: np.ndarray | None = None
            asymptotic_stitch_posthoc_up_directions: np.ndarray | None = None

            validation_atom_scale_repair_enabled = os.environ.get("GCL_BLIND_SWIGLU_VALIDATION_ATOM_SCALE_REPAIR", "").strip() == "1"
            if (
                not passes_metric_thresholds(metrics, args.min_r2, args.max_mae, args.max_worst_error)
                and validation_atom_scale_repair_enabled
                and atom_tail_diagnostic is not None
                and args.atom_count >= 4
                and oracle.query_count < args.max_oracle_calls
            ):
                top_atoms = [
                    int(item["atom"])
                    for item in atom_tail_diagnostic.get("topAtoms", [])[:int(os.environ.get("GCL_BLIND_SWIGLU_VALIDATION_ATOM_SCALE_REPAIR_TOP_K", "4") or "4")]
                ]
                scale_repair_rng = np.random.default_rng(args.seed + 70411)
                scale_repair_train_limit = int(os.environ.get("GCL_BLIND_SWIGLU_VALIDATION_ATOM_SCALE_REPAIR_TRAIN_LIMIT", "8000") or "8000")
                scale_repair_tail_count = int(os.environ.get("GCL_BLIND_SWIGLU_VALIDATION_ATOM_SCALE_REPAIR_TAIL_COUNT", "128") or "128")
                scale_repair_tail_repeat = int(os.environ.get("GCL_BLIND_SWIGLU_VALIDATION_ATOM_SCALE_REPAIR_TAIL_REPEAT", "8") or "8")
                scale_repair_holdout_pools = int(os.environ.get("GCL_BLIND_SWIGLU_VALIDATION_ATOM_SCALE_REPAIR_HOLDOUT_POOLS", "0") or "0")
                scale_repair_holdout_count = int(os.environ.get(
                    "GCL_BLIND_SWIGLU_VALIDATION_ATOM_SCALE_REPAIR_HOLDOUT_COUNT",
                    str(args.validation_count),
                ) or args.validation_count)
                scale_repair_holdout_tail_count = int(os.environ.get(
                    "GCL_BLIND_SWIGLU_VALIDATION_ATOM_SCALE_REPAIR_HOLDOUT_TAIL_COUNT",
                    str(scale_repair_tail_count),
                ) or scale_repair_tail_count)
                scale_repair_holdout_tail_repeat = int(os.environ.get(
                    "GCL_BLIND_SWIGLU_VALIDATION_ATOM_SCALE_REPAIR_HOLDOUT_TAIL_REPEAT",
                    str(max(1, scale_repair_tail_repeat // 2)),
                ) or max(1, scale_repair_tail_repeat // 2))
                if (
                    scale_repair_tail_count <= 0
                    or scale_repair_tail_repeat <= 0
                    or scale_repair_holdout_pools < 0
                    or scale_repair_holdout_count <= 0
                    or scale_repair_holdout_tail_count <= 0
                    or scale_repair_holdout_tail_repeat <= 0
                ):
                    raise RuntimeError("validation atom scale repair counts/repeats must be positive")
                residual_order = np.argsort(np.abs(validation_targets - validation_prediction))[::-1]
                selected_tail = residual_order[:max(1, min(scale_repair_tail_count, validation_coords.shape[0]))]
                if 0 < scale_repair_train_limit < train_coords.shape[0]:
                    scale_repair_base_indices = np.sort(scale_repair_rng.choice(train_coords.shape[0], size=scale_repair_train_limit, replace=False))
                    scale_repair_base_coords = train_coords[scale_repair_base_indices]
                    scale_repair_base_targets = train_targets[scale_repair_base_indices]
                else:
                    scale_repair_base_coords = train_coords
                    scale_repair_base_targets = train_targets
                scale_tail_coord_parts = [
                    np.repeat(validation_coords[selected_tail], max(1, scale_repair_tail_repeat), axis=0),
                ]
                scale_tail_target_parts = [
                    np.repeat(validation_targets[selected_tail], max(1, scale_repair_tail_repeat), axis=0),
                ]
                scale_holdout_selected = 0
                scale_holdout_evaluated = 0
                for _pool_index in range(scale_repair_holdout_pools):
                    if oracle.query_count + scale_repair_holdout_count > args.max_oracle_calls:
                        break
                    holdout_inputs = scale_repair_rng.normal(
                        0.0,
                        args.validation_scale,
                        size=(scale_repair_holdout_count, dimension),
                    ).astype(np.float32)
                    holdout_targets = oracle.eval(holdout_inputs)
                    holdout_coords = holdout_inputs @ basis.T
                    holdout_prediction = predict_swiglu(holdout_coords, fit)
                    holdout_residual_order = np.argsort(np.abs(holdout_targets - holdout_prediction))[::-1]
                    holdout_tail = holdout_residual_order[:max(1, min(scale_repair_holdout_tail_count, holdout_coords.shape[0]))]
                    scale_tail_coord_parts.append(
                        np.repeat(holdout_coords[holdout_tail], max(1, scale_repair_holdout_tail_repeat), axis=0)
                    )
                    scale_tail_target_parts.append(
                        np.repeat(holdout_targets[holdout_tail], max(1, scale_repair_holdout_tail_repeat), axis=0)
                    )
                    scale_holdout_selected += int(len(holdout_tail))
                    scale_holdout_evaluated += 1
                scale_repair_coords = np.vstack([scale_repair_base_coords] + scale_tail_coord_parts)
                scale_repair_targets = np.concatenate([scale_repair_base_targets] + scale_tail_target_parts)
                scale_repaired_fit = selective_atom_scale_repair_swiglu_candidate(
                    scale_repair_coords,
                    scale_repair_targets,
                    fit,
                    top_atoms,
                )
                scale_probe_prediction = predict_swiglu(validation_coords, scale_repaired_fit)
                scale_probe_metrics = regression_metrics(validation_targets, scale_probe_prediction)
                scale_revalidation_metrics: dict[str, float] | None = None
                scale_accepted = False
                if oracle.query_count + args.validation_count <= args.max_oracle_calls:
                    scale_validation_rng = np.random.default_rng(args.seed + 70427)
                    scale_validation_inputs = scale_validation_rng.normal(
                        0.0,
                        args.validation_scale,
                        size=(args.validation_count, dimension),
                    ).astype(np.float32)
                    scale_validation_targets = oracle.eval(scale_validation_inputs)
                    scale_validation_coords = scale_validation_inputs @ basis.T
                    scale_validation_prediction = predict_swiglu(scale_validation_coords, scale_repaired_fit)
                    scale_revalidation_metrics = regression_metrics(scale_validation_targets, scale_validation_prediction)
                    scale_accepted = (
                        passes_metric_thresholds(scale_revalidation_metrics, args.min_r2, args.max_mae, args.max_worst_error)
                        or (
                            scale_revalidation_metrics["mae"] <= metrics["mae"]
                            and scale_revalidation_metrics["worstError"] < metrics["worstError"]
                        )
                    )
                fit_attempts.append({
                    "stage": "validation-atom-scale-repair",
                    "atoms": top_atoms,
                    "trainCount": int(scale_repair_coords.shape[0]),
                    "baseTrainCount": int(scale_repair_base_coords.shape[0]),
                    "tailCount": int(len(selected_tail)),
                    "tailRepeat": int(max(1, scale_repair_tail_repeat)),
                    "holdoutPoolsRequested": int(max(0, scale_repair_holdout_pools)),
                    "holdoutPoolsEvaluated": int(scale_holdout_evaluated),
                    "holdoutTailSelected": int(scale_holdout_selected),
                    "holdoutTailRepeat": int(max(1, scale_repair_holdout_tail_repeat)),
                    "selectiveAtomScaleRepaired": bool(scale_repaired_fit.get("selectiveAtomScaleRepaired", False)),
                    "probeValidationR2": float(scale_probe_metrics["r2"]),
                    "probeValidationMae": float(scale_probe_metrics["mae"]),
                    "probeValidationWorstError": float(scale_probe_metrics["worstError"]),
                    "revalidationPerformed": scale_revalidation_metrics is not None,
                    "revalidationR2": None if scale_revalidation_metrics is None else float(scale_revalidation_metrics["r2"]),
                    "revalidationMae": None if scale_revalidation_metrics is None else float(scale_revalidation_metrics["mae"]),
                    "revalidationWorstError": None if scale_revalidation_metrics is None else float(scale_revalidation_metrics["worstError"]),
                    "accepted": bool(scale_accepted),
                    "rawOracleCalls": int(oracle.query_count),
                })
                log_progress(
                    args,
                    "validation-atom-scale-repair-complete",
                    accepted=bool(scale_accepted),
                    repaired=bool(scale_repaired_fit.get("selectiveAtomScaleRepaired", False)),
                    atoms=top_atoms,
                    holdoutPools=int(scale_holdout_evaluated),
                    holdoutTailSelected=int(scale_holdout_selected),
                    probeMae=float(scale_probe_metrics["mae"]),
                    probeWorstError=float(scale_probe_metrics["worstError"]),
                    revalidationMae=None if scale_revalidation_metrics is None else float(scale_revalidation_metrics["mae"]),
                    revalidationWorstError=None if scale_revalidation_metrics is None else float(scale_revalidation_metrics["worstError"]),
                    rawOracleCalls=int(oracle.query_count),
                )
                if scale_accepted and scale_revalidation_metrics is not None:
                    scale_reporting_metrics = independent_reporting_metrics(
                        oracle,
                        args,
                        basis,
                        dimension,
                        scale_repaired_fit,
                        seed_offset=70428,
                        stage="validation-atom-scale-repair-reporting",
                    )
                    if scale_reporting_metrics is None:
                        # No budget for a draw that did not choose this candidate, so the
                        # repair is declined rather than published on its own selection set.
                        scale_accepted = False
                        fit_attempts[-1]["accepted"] = False
                        fit_attempts[-1]["declinedForUnbiasedReportingBudget"] = True
                    else:
                        fit = scale_repaired_fit
                        metrics = scale_reporting_metrics
                        fit_attempts[-1].update({
                            "selectionR2": float(scale_revalidation_metrics["r2"]),
                            "selectionMae": float(scale_revalidation_metrics["mae"]),
                            "selectionWorstError": float(scale_revalidation_metrics["worstError"]),
                            "reportedR2": float(scale_reporting_metrics["r2"]),
                            "reportedMae": float(scale_reporting_metrics["mae"]),
                            "reportedWorstError": float(scale_reporting_metrics["worstError"]),
                            "selectionAndReportingSeparated": True,
                            "rawOracleCalls": int(oracle.query_count),
                        })
                    validation_inputs = scale_validation_inputs
                    validation_targets = scale_validation_targets
                    validation_coords = scale_validation_coords
                    validation_prediction = scale_validation_prediction
                    final_validation_source = "validation-atom-scale-repair-fresh-validation"

            validation_atom_direction_repair_enabled = os.environ.get("GCL_BLIND_SWIGLU_VALIDATION_ATOM_DIRECTION_REPAIR", "").strip() == "1"
            if (
                not passes_metric_thresholds(metrics, args.min_r2, args.max_mae, args.max_worst_error)
                and validation_atom_direction_repair_enabled
                and atom_tail_diagnostic is not None
                and args.atom_count >= 4
                and oracle.query_count < args.max_oracle_calls
            ):
                direction_top_atoms = [
                    int(item["atom"])
                    for item in atom_tail_diagnostic.get("topAtoms", [])[:int(os.environ.get("GCL_BLIND_SWIGLU_VALIDATION_ATOM_DIRECTION_REPAIR_TOP_K", "4") or "4")]
                ]
                direction_repair_rng = np.random.default_rng(args.seed + 70511)
                direction_train_limit = int(os.environ.get("GCL_BLIND_SWIGLU_VALIDATION_ATOM_DIRECTION_REPAIR_TRAIN_LIMIT", "8000") or "8000")
                direction_tail_count = int(os.environ.get("GCL_BLIND_SWIGLU_VALIDATION_ATOM_DIRECTION_REPAIR_TAIL_COUNT", "128") or "128")
                direction_tail_repeat = int(os.environ.get("GCL_BLIND_SWIGLU_VALIDATION_ATOM_DIRECTION_REPAIR_TAIL_REPEAT", "8") or "8")
                direction_holdout_pools = int(os.environ.get("GCL_BLIND_SWIGLU_VALIDATION_ATOM_DIRECTION_REPAIR_HOLDOUT_POOLS", "0") or "0")
                direction_selection_holdout_pools = int(os.environ.get(
                    "GCL_BLIND_SWIGLU_VALIDATION_ATOM_DIRECTION_REPAIR_SELECTION_HOLDOUT_POOLS",
                    "0",
                ) or "0")
                direction_holdout_count = int(os.environ.get(
                    "GCL_BLIND_SWIGLU_VALIDATION_ATOM_DIRECTION_REPAIR_HOLDOUT_COUNT",
                    str(args.validation_count),
                ) or args.validation_count)
                direction_holdout_tail_count = int(os.environ.get(
                    "GCL_BLIND_SWIGLU_VALIDATION_ATOM_DIRECTION_REPAIR_HOLDOUT_TAIL_COUNT",
                    str(direction_tail_count),
                ) or direction_tail_count)
                direction_holdout_tail_repeat = int(os.environ.get(
                    "GCL_BLIND_SWIGLU_VALIDATION_ATOM_DIRECTION_REPAIR_HOLDOUT_TAIL_REPEAT",
                    str(max(1, direction_tail_repeat // 2)),
                ) or max(1, direction_tail_repeat // 2))
                residual_order = np.argsort(np.abs(validation_targets - validation_prediction))[::-1]
                selected_tail = residual_order[:max(1, min(direction_tail_count, validation_coords.shape[0]))]
                if 0 < direction_train_limit < train_coords.shape[0]:
                    direction_base_indices = np.sort(direction_repair_rng.choice(train_coords.shape[0], size=direction_train_limit, replace=False))
                    direction_base_coords = train_coords[direction_base_indices]
                    direction_base_targets = train_targets[direction_base_indices]
                else:
                    direction_base_coords = train_coords
                    direction_base_targets = train_targets
                direction_base_train_count = int(direction_base_coords.shape[0])
                direction_extra_random_count = int(os.environ.get(
                    "GCL_BLIND_SWIGLU_VALIDATION_ATOM_DIRECTION_REPAIR_EXTRA_RANDOM_COUNT",
                    "0",
                ) or "0")
                direction_extra_random_scale = float(os.environ.get(
                    "GCL_BLIND_SWIGLU_VALIDATION_ATOM_DIRECTION_REPAIR_EXTRA_RANDOM_SCALE",
                    str(args.validation_scale),
                ) or args.validation_scale)
                if direction_extra_random_count < 0 or not np.isfinite(direction_extra_random_scale) or direction_extra_random_scale <= 0.0:
                    raise RuntimeError("validation atom direction repair extra random config is invalid")
                direction_extra_random_evaluated = 0
                if direction_extra_random_count > 0 and oracle.query_count + direction_extra_random_count <= args.max_oracle_calls:
                    direction_extra_inputs = direction_repair_rng.normal(
                        0.0,
                        direction_extra_random_scale,
                        size=(direction_extra_random_count, dimension),
                    ).astype(np.float32)
                    direction_extra_targets = oracle.eval(direction_extra_inputs)
                    direction_extra_coords = direction_extra_inputs @ basis.T
                    direction_base_coords = np.vstack([direction_base_coords, direction_extra_coords])
                    direction_base_targets = np.concatenate([direction_base_targets, direction_extra_targets])
                    direction_extra_random_evaluated = int(direction_extra_random_count)
                direction_tail_coord_parts = [
                    np.repeat(validation_coords[selected_tail], max(1, direction_tail_repeat), axis=0),
                ]
                direction_tail_target_parts = [
                    np.repeat(validation_targets[selected_tail], max(1, direction_tail_repeat), axis=0),
                ]
                direction_selection_coord_parts: list[np.ndarray] = []
                direction_selection_target_parts: list[np.ndarray] = []
                direction_holdout_selected = 0
                direction_holdout_evaluated = 0
                direction_selection_selected = 0
                direction_selection_evaluated = 0
                if direction_holdout_pools > 0:
                    if direction_holdout_count <= 0 or direction_holdout_tail_count <= 0 or direction_holdout_tail_repeat <= 0:
                        raise RuntimeError("validation atom direction repair holdout counts/repeats must be positive")
                    if direction_selection_holdout_pools < 0:
                        raise RuntimeError("validation atom direction repair selection holdout pools must be non-negative")
                    for pool_index in range(direction_holdout_pools):
                        if oracle.query_count + direction_holdout_count > args.max_oracle_calls:
                            break
                        holdout_inputs = direction_repair_rng.normal(
                            0.0,
                            args.validation_scale,
                            size=(direction_holdout_count, dimension),
                        ).astype(np.float32)
                        holdout_targets = oracle.eval(holdout_inputs)
                        holdout_coords = holdout_inputs @ basis.T
                        holdout_prediction = predict_swiglu(holdout_coords, fit)
                        holdout_residual_order = np.argsort(np.abs(holdout_targets - holdout_prediction))[::-1]
                        holdout_tail = holdout_residual_order[:max(1, min(direction_holdout_tail_count, holdout_coords.shape[0]))]
                        repeated_coords = np.repeat(holdout_coords[holdout_tail], max(1, direction_holdout_tail_repeat), axis=0)
                        repeated_targets = np.repeat(holdout_targets[holdout_tail], max(1, direction_holdout_tail_repeat), axis=0)
                        if direction_selection_evaluated < direction_selection_holdout_pools:
                            direction_selection_coord_parts.append(repeated_coords)
                            direction_selection_target_parts.append(repeated_targets)
                            direction_selection_selected += int(len(holdout_tail))
                            direction_selection_evaluated += 1
                        else:
                            direction_tail_coord_parts.append(repeated_coords)
                            direction_tail_target_parts.append(repeated_targets)
                            direction_holdout_selected += int(len(holdout_tail))
                            direction_holdout_evaluated += 1
                direction_repair_coords = np.vstack([direction_base_coords] + direction_tail_coord_parts)
                direction_repair_targets = np.concatenate([direction_base_targets] + direction_tail_target_parts)
                direction_selection_coords = (
                    np.vstack(direction_selection_coord_parts)
                    if direction_selection_coord_parts
                    else None
                )
                direction_selection_targets = (
                    np.concatenate(direction_selection_target_parts)
                    if direction_selection_target_parts
                    else None
                )
                direction_repaired_fit = directional_atom_repair_swiglu_candidate(
                    direction_repair_coords,
                    direction_repair_targets,
                    fit,
                    direction_top_atoms,
                    direction_selection_coords,
                    direction_selection_targets,
                )
                direction_probe_prediction = predict_swiglu(validation_coords, direction_repaired_fit)
                direction_probe_metrics = regression_metrics(validation_targets, direction_probe_prediction)
                direction_revalidation_metrics: dict[str, float] | None = None
                direction_revalidation_baseline_metrics: dict[str, float] | None = None
                direction_accepted = False
                if oracle.query_count + args.validation_count <= args.max_oracle_calls:
                    direction_validation_rng = np.random.default_rng(args.seed + 70527)
                    direction_validation_inputs = direction_validation_rng.normal(
                        0.0,
                        args.validation_scale,
                        size=(args.validation_count, dimension),
                    ).astype(np.float32)
                    direction_validation_targets = oracle.eval(direction_validation_inputs)
                    direction_validation_coords = direction_validation_inputs @ basis.T
                    direction_baseline_prediction = predict_swiglu(direction_validation_coords, fit)
                    direction_revalidation_baseline_metrics = regression_metrics(direction_validation_targets, direction_baseline_prediction)
                    direction_validation_prediction = predict_swiglu(direction_validation_coords, direction_repaired_fit)
                    direction_revalidation_metrics = regression_metrics(direction_validation_targets, direction_validation_prediction)
                    direction_accepted = (
                        passes_metric_thresholds(direction_revalidation_metrics, args.min_r2, args.max_mae, args.max_worst_error)
                        or (
                            direction_revalidation_baseline_metrics is not None
                            and direction_revalidation_metrics["mae"] <= direction_revalidation_baseline_metrics["mae"]
                            and direction_revalidation_metrics["worstError"] < direction_revalidation_baseline_metrics["worstError"]
                        )
                    )
                fit_attempts.append({
                    "stage": "validation-atom-direction-repair",
                    "atoms": direction_top_atoms,
                    "trainCount": int(direction_repair_coords.shape[0]),
                    "baseTrainCount": int(direction_base_train_count),
                    "extraRandomTrainCount": int(direction_extra_random_evaluated),
                    "tailCount": int(len(selected_tail)),
                    "tailRepeat": int(max(1, direction_tail_repeat)),
                    "holdoutPoolsRequested": int(max(0, direction_holdout_pools)),
                    "holdoutPoolsEvaluated": int(direction_holdout_evaluated),
                    "holdoutTailSelected": int(direction_holdout_selected),
                    "holdoutTailRepeat": int(max(1, direction_holdout_tail_repeat)),
                    "selectionHoldoutPoolsRequested": int(max(0, direction_selection_holdout_pools)),
                    "selectionHoldoutPoolsEvaluated": int(direction_selection_evaluated),
                    "selectionHoldoutTailSelected": int(direction_selection_selected),
                    "directionRepairSelectionCount": int(direction_repaired_fit.get("directionalAtomRepairSelectionCount", 0)),
                    "directionalAtomRepaired": bool(direction_repaired_fit.get("directionalAtomRepaired", False)),
                    "probeValidationR2": float(direction_probe_metrics["r2"]),
                    "probeValidationMae": float(direction_probe_metrics["mae"]),
                    "probeValidationWorstError": float(direction_probe_metrics["worstError"]),
                    "revalidationPerformed": direction_revalidation_metrics is not None,
                    "baselineRevalidationR2": None if direction_revalidation_baseline_metrics is None else float(direction_revalidation_baseline_metrics["r2"]),
                    "baselineRevalidationMae": None if direction_revalidation_baseline_metrics is None else float(direction_revalidation_baseline_metrics["mae"]),
                    "baselineRevalidationWorstError": None if direction_revalidation_baseline_metrics is None else float(direction_revalidation_baseline_metrics["worstError"]),
                    "revalidationR2": None if direction_revalidation_metrics is None else float(direction_revalidation_metrics["r2"]),
                    "revalidationMae": None if direction_revalidation_metrics is None else float(direction_revalidation_metrics["mae"]),
                    "revalidationWorstError": None if direction_revalidation_metrics is None else float(direction_revalidation_metrics["worstError"]),
                    "accepted": bool(direction_accepted),
                    "rawOracleCalls": int(oracle.query_count),
                })
                log_progress(
                    args,
                    "validation-atom-direction-repair-complete",
                    accepted=bool(direction_accepted),
                    repaired=bool(direction_repaired_fit.get("directionalAtomRepaired", False)),
                    atoms=direction_top_atoms,
                    holdoutPools=int(direction_holdout_evaluated),
                    holdoutTailSelected=int(direction_holdout_selected),
                    selectionHoldoutPools=int(direction_selection_evaluated),
                    selectionHoldoutTailSelected=int(direction_selection_selected),
                    extraRandomTrainCount=int(direction_extra_random_evaluated),
                    probeMae=float(direction_probe_metrics["mae"]),
                    probeWorstError=float(direction_probe_metrics["worstError"]),
                    baselineRevalidationMae=None if direction_revalidation_baseline_metrics is None else float(direction_revalidation_baseline_metrics["mae"]),
                    baselineRevalidationWorstError=None if direction_revalidation_baseline_metrics is None else float(direction_revalidation_baseline_metrics["worstError"]),
                    revalidationMae=None if direction_revalidation_metrics is None else float(direction_revalidation_metrics["mae"]),
                    revalidationWorstError=None if direction_revalidation_metrics is None else float(direction_revalidation_metrics["worstError"]),
                    rawOracleCalls=int(oracle.query_count),
                )
                if direction_accepted and direction_revalidation_metrics is not None:
                    direction_reporting_metrics = independent_reporting_metrics(
                        oracle,
                        args,
                        basis,
                        dimension,
                        direction_repaired_fit,
                        seed_offset=70528,
                        stage="validation-directional-repair-reporting",
                    )
                    if direction_reporting_metrics is None:
                        direction_accepted = False
                        fit_attempts[-1]["accepted"] = False
                        fit_attempts[-1]["declinedForUnbiasedReportingBudget"] = True
                    else:
                        fit = direction_repaired_fit
                        metrics = direction_reporting_metrics
                        fit_attempts[-1].update({
                            "selectionR2": float(direction_revalidation_metrics["r2"]),
                            "selectionMae": float(direction_revalidation_metrics["mae"]),
                            "selectionWorstError": float(direction_revalidation_metrics["worstError"]),
                            "reportedR2": float(direction_reporting_metrics["r2"]),
                            "reportedMae": float(direction_reporting_metrics["mae"]),
                            "reportedWorstError": float(direction_reporting_metrics["worstError"]),
                            "selectionAndReportingSeparated": True,
                            "rawOracleCalls": int(oracle.query_count),
                        })
                    validation_inputs = direction_validation_inputs
                    validation_targets = direction_validation_targets
                    validation_coords = direction_validation_coords
                    validation_prediction = direction_validation_prediction
                    final_validation_source = "validation-atom-direction-repair-fresh-validation"

            validation_full_tail_polish_enabled = os.environ.get("GCL_BLIND_SWIGLU_VALIDATION_FULL_TAIL_POLISH", "").strip() == "1"
            full_tail_method = os.environ.get(
                "GCL_BLIND_SWIGLU_VALIDATION_FULL_TAIL_POLISH_METHOD",
                "adam",
            ).strip().lower() or "adam"
            if full_tail_method not in {"adam", "lbfgs", "gate-norm", "pair-swap", "gradient-polish", "fixed-direction-value-polish", "param-refit", "fixed-gate-up-refit", "gate-dictionary-refit", "residual-gradient-replace", "residual-block-refit", "cancellation-pair-refit", "asymptotic-gate-stitch", "asymptotic-cold-rebuild"}:
                raise RuntimeError("GCL_BLIND_SWIGLU_VALIDATION_FULL_TAIL_POLISH_METHOD must be adam, lbfgs, gate-norm, pair-swap, gradient-polish, fixed-direction-value-polish, param-refit, fixed-gate-up-refit, gate-dictionary-refit, residual-gradient-replace, residual-block-refit, cancellation-pair-refit, or asymptotic-gate-stitch")
            validation_full_tail_polish_on_cubic = validation_full_tail_polish_cubic_trigger(
                full_tail_method,
                int(cubic_scores.size),
            )
            needs_full_tail_polish = (
                not passes_metric_thresholds(
                    metrics,
                    args.min_r2,
                    args.max_mae,
                    args.max_worst_error,
                )
                or (
                    validation_full_tail_polish_on_cubic
                    and cubic_scores.size > 0
                )
            )
            if (
                needs_full_tail_polish
                and validation_full_tail_polish_enabled
                and args.atom_count >= 4
                and oracle.query_count < args.max_oracle_calls
            ):
                full_tail_rng = np.random.default_rng(args.seed + 70611)
                full_tail_train_limit = int(os.environ.get("GCL_BLIND_SWIGLU_VALIDATION_FULL_TAIL_POLISH_TRAIN_LIMIT", "16000") or "16000")
                full_tail_count = int(os.environ.get("GCL_BLIND_SWIGLU_VALIDATION_FULL_TAIL_POLISH_TAIL_COUNT", "256") or "256")
                full_tail_repeat = int(os.environ.get("GCL_BLIND_SWIGLU_VALIDATION_FULL_TAIL_POLISH_TAIL_REPEAT", "8") or "8")
                full_tail_holdout_pools = int(os.environ.get("GCL_BLIND_SWIGLU_VALIDATION_FULL_TAIL_POLISH_HOLDOUT_POOLS", "4") or "4")
                full_tail_selection_holdout_pools = int(os.environ.get(
                    "GCL_BLIND_SWIGLU_VALIDATION_FULL_TAIL_POLISH_SELECTION_HOLDOUT_POOLS",
                    "0",
                ) or "0")
                full_tail_holdout_count = int(os.environ.get(
                    "GCL_BLIND_SWIGLU_VALIDATION_FULL_TAIL_POLISH_HOLDOUT_COUNT",
                    str(args.validation_count),
                ) or args.validation_count)
                full_tail_holdout_tail_count = int(os.environ.get(
                    "GCL_BLIND_SWIGLU_VALIDATION_FULL_TAIL_POLISH_HOLDOUT_TAIL_COUNT",
                    str(full_tail_count),
                ) or full_tail_count)
                full_tail_holdout_tail_repeat = int(os.environ.get(
                    "GCL_BLIND_SWIGLU_VALIDATION_FULL_TAIL_POLISH_HOLDOUT_TAIL_REPEAT",
                    str(max(1, full_tail_repeat // 2)),
                ) or max(1, full_tail_repeat // 2))
                full_tail_steps = int(os.environ.get("GCL_BLIND_SWIGLU_VALIDATION_FULL_TAIL_POLISH_STEPS", "520") or "520")
                if (
                    full_tail_count <= 0
                    or full_tail_repeat <= 0
                    or full_tail_holdout_pools < 0
                    or full_tail_selection_holdout_pools < 0
                    or full_tail_holdout_count <= 0
                    or full_tail_holdout_tail_count <= 0
                    or full_tail_holdout_tail_repeat <= 0
                    or (full_tail_method == "adam" and full_tail_steps <= 0)
                ):
                    raise RuntimeError("validation full-tail polish counts/repeats/steps must be positive")
                full_tail_residual_order = np.argsort(np.abs(validation_targets - validation_prediction))[::-1]
                full_tail_selected = full_tail_residual_order[:max(1, min(full_tail_count, validation_coords.shape[0]))]
                if 0 < full_tail_train_limit < train_coords.shape[0]:
                    full_tail_train_indices = np.sort(full_tail_rng.choice(train_coords.shape[0], size=full_tail_train_limit, replace=False))
                    full_tail_base_coords = train_coords[full_tail_train_indices]
                    full_tail_base_targets = train_targets[full_tail_train_indices]
                else:
                    full_tail_base_coords = train_coords
                    full_tail_base_targets = train_targets
                full_tail_extra_random_count = int(os.environ.get(
                    "GCL_BLIND_SWIGLU_VALIDATION_FULL_TAIL_POLISH_EXTRA_RANDOM_COUNT",
                    "0",
                ) or "0")
                full_tail_extra_random_scale = float(os.environ.get(
                    "GCL_BLIND_SWIGLU_VALIDATION_FULL_TAIL_POLISH_EXTRA_RANDOM_SCALE",
                    str(args.validation_scale),
                ) or args.validation_scale)
                if full_tail_extra_random_count < 0 or not np.isfinite(full_tail_extra_random_scale) or full_tail_extra_random_scale <= 0.0:
                    raise RuntimeError("validation full-tail polish extra random config is invalid")
                full_tail_extra_random_evaluated = 0
                if full_tail_extra_random_count > 0 and oracle.query_count + full_tail_extra_random_count <= args.max_oracle_calls:
                    extra_inputs = full_tail_rng.normal(
                        0.0,
                        full_tail_extra_random_scale,
                        size=(full_tail_extra_random_count, dimension),
                    ).astype(np.float32)
                    extra_targets = oracle.eval(extra_inputs)
                    extra_coords = extra_inputs @ basis.T
                    full_tail_base_coords = np.vstack([full_tail_base_coords, extra_coords])
                    full_tail_base_targets = np.concatenate([full_tail_base_targets, extra_targets])
                    full_tail_extra_random_evaluated = int(full_tail_extra_random_count)
                full_tail_radial_train_count = int(os.environ.get(
                    "GCL_BLIND_SWIGLU_VALIDATION_FULL_TAIL_POLISH_RADIAL_TRAIN_COUNT",
                    "0",
                ) or "0")
                full_tail_radial_selection_count = int(os.environ.get(
                    "GCL_BLIND_SWIGLU_VALIDATION_FULL_TAIL_POLISH_RADIAL_SELECTION_COUNT",
                    "0",
                ) or "0")
                full_tail_radial_scales = parse_float_list(os.environ.get(
                    "GCL_BLIND_SWIGLU_VALIDATION_FULL_TAIL_POLISH_RADIAL_SCALES",
                    "1.0,2.0,4.0,8.0",
                ))
                if (
                    full_tail_radial_train_count < 0
                    or full_tail_radial_selection_count < 0
                    or not full_tail_radial_scales
                    or any(not np.isfinite(scale) or scale <= 0.0 for scale in full_tail_radial_scales)
                ):
                    raise RuntimeError("validation full-tail radial witness config is invalid")
                radial_requested = full_tail_radial_train_count + full_tail_radial_selection_count
                if radial_requested > args.max_oracle_calls - oracle.query_count:
                    raise OracleBudgetExceeded(
                        stage="validation-full-tail-radial-witness",
                        requested=radial_requested,
                        consumed=oracle.query_count,
                        limit=args.max_oracle_calls,
                    )
                full_tail_radial_train_evaluated = 0
                full_tail_radial_selection_evaluated = 0
                full_tail_radial_gradient_evaluated = 0
                full_tail_fit_gradient_coords = fit_gradient_coords
                full_tail_fit_gradient_targets = fit_gradient_targets
                radial_selection_coords: np.ndarray | None = None
                radial_selection_targets: np.ndarray | None = None
                if full_tail_radial_train_count > 0:
                    radial_train_inputs = sample_active_region_inputs(
                        full_tail_rng,
                        full_tail_radial_train_count,
                        basis,
                        full_tail_radial_scales[0],
                        full_tail_radial_scales,
                        "active-radial-shell",
                    )
                    radial_train_targets = oracle.eval(
                        radial_train_inputs,
                        stage="validation-full-tail-radial-train",
                    )
                    radial_train_coords = radial_train_inputs @ basis.T
                    full_tail_base_coords = np.vstack([full_tail_base_coords, radial_train_coords])
                    full_tail_base_targets = np.concatenate([full_tail_base_targets, radial_train_targets])
                    full_tail_radial_train_evaluated = int(radial_train_coords.shape[0])
                if full_tail_radial_selection_count > 0:
                    radial_selection_inputs = sample_active_region_inputs(
                        full_tail_rng,
                        full_tail_radial_selection_count,
                        basis,
                        full_tail_radial_scales[0],
                        full_tail_radial_scales,
                        "active-radial-shell",
                    )
                    radial_selection_targets = oracle.eval(
                        radial_selection_inputs,
                        stage="validation-full-tail-radial-selection",
                    )
                    radial_selection_coords = radial_selection_inputs @ basis.T
                    full_tail_radial_selection_evaluated = int(radial_selection_coords.shape[0])
                full_tail_radial_gradient_count = int(os.environ.get(
                    "GCL_BLIND_SWIGLU_VALIDATION_FULL_TAIL_POLISH_RADIAL_GRADIENT_COUNT",
                    "0",
                ) or "0")
                full_tail_radial_gradient_scales = parse_float_list(os.environ.get(
                    "GCL_BLIND_SWIGLU_VALIDATION_FULL_TAIL_POLISH_RADIAL_GRADIENT_SCALES",
                    "2.0,4.0,8.0",
                ))
                full_tail_radial_gradient_steps = parse_float_list(os.environ.get(
                    "GCL_BLIND_SWIGLU_VALIDATION_FULL_TAIL_POLISH_RADIAL_GRADIENT_STEPS",
                    "0.25,0.5",
                ))
                if (
                    full_tail_radial_gradient_count < 0
                    or not full_tail_radial_gradient_scales
                    or not full_tail_radial_gradient_steps
                    or any(not np.isfinite(scale) or scale <= 0.0 for scale in full_tail_radial_gradient_scales)
                    or any(not np.isfinite(step) or step <= 0.0 for step in full_tail_radial_gradient_steps)
                ):
                    raise RuntimeError("validation full-tail radial gradient config is invalid")
                radial_gradient_cost = (
                    full_tail_radial_gradient_count
                    * (args.active_rank + 1)
                    * len(full_tail_radial_gradient_steps)
                )
                if radial_gradient_cost > args.max_oracle_calls - oracle.query_count:
                    raise OracleBudgetExceeded(
                        stage="validation-full-tail-radial-gradient",
                        requested=radial_gradient_cost,
                        consumed=oracle.query_count,
                        limit=args.max_oracle_calls,
                    )
                if full_tail_radial_gradient_count > 0:
                    radial_gradient_inputs = sample_active_region_inputs(
                        full_tail_rng,
                        full_tail_radial_gradient_count,
                        basis,
                        full_tail_radial_gradient_scales[0],
                        full_tail_radial_gradient_scales,
                        "active-radial-shell",
                    )
                    radial_gradient_coords = radial_gradient_inputs @ basis.T
                    radial_gradient_targets = estimate_active_forward_gradients_at_coords(
                        oracle,
                        basis,
                        radial_gradient_coords,
                        full_tail_radial_gradient_steps,
                        stage="validation-full-tail-radial-gradient",
                    )
                    full_tail_fit_gradient_coords = np.vstack([
                        full_tail_fit_gradient_coords,
                        radial_gradient_coords,
                    ])
                    full_tail_fit_gradient_targets = np.vstack([
                        full_tail_fit_gradient_targets,
                        radial_gradient_targets,
                    ])
                    full_tail_radial_gradient_evaluated = int(radial_gradient_coords.shape[0])
                full_tail_coord_parts = [
                    np.repeat(validation_coords[full_tail_selected], full_tail_repeat, axis=0),
                ]
                full_tail_target_parts = [
                    np.repeat(validation_targets[full_tail_selected], full_tail_repeat, axis=0),
                ]
                full_tail_selection_coord_parts: list[np.ndarray] = []
                full_tail_selection_target_parts: list[np.ndarray] = []
                if radial_selection_coords is not None and radial_selection_targets is not None:
                    full_tail_selection_coord_parts.append(radial_selection_coords)
                    full_tail_selection_target_parts.append(radial_selection_targets)
                full_tail_selection_include_full = os.environ.get(
                    "GCL_BLIND_SWIGLU_VALIDATION_FULL_TAIL_POLISH_SELECTION_INCLUDE_FULL",
                    "1" if full_tail_method in {"residual-block-refit", "cancellation-pair-refit", "asymptotic-gate-stitch", "asymptotic-cold-rebuild", "fixed-direction-value-polish"} else "0",
                ).strip() == "1"
                full_tail_holdout_selected = 0
                full_tail_holdout_evaluated = 0
                full_tail_selection_selected = 0
                full_tail_selection_full_count = 0
                full_tail_selection_evaluated = 0
                for _pool_index in range(full_tail_holdout_pools):
                    if oracle.query_count + full_tail_holdout_count > args.max_oracle_calls:
                        break
                    holdout_inputs = full_tail_rng.normal(
                        0.0,
                        args.validation_scale,
                        size=(full_tail_holdout_count, dimension),
                    ).astype(np.float32)
                    holdout_targets = oracle.eval(holdout_inputs)
                    holdout_coords = holdout_inputs @ basis.T
                    holdout_prediction = predict_swiglu(holdout_coords, fit)
                    holdout_order = np.argsort(np.abs(holdout_targets - holdout_prediction))[::-1]
                    holdout_tail = holdout_order[:max(1, min(full_tail_holdout_tail_count, holdout_coords.shape[0]))]
                    repeated_coords = np.repeat(holdout_coords[holdout_tail], full_tail_holdout_tail_repeat, axis=0)
                    repeated_targets = np.repeat(holdout_targets[holdout_tail], full_tail_holdout_tail_repeat, axis=0)
                    if full_tail_selection_evaluated < full_tail_selection_holdout_pools:
                        if full_tail_selection_include_full:
                            full_tail_selection_coord_parts.append(holdout_coords)
                            full_tail_selection_target_parts.append(holdout_targets)
                            full_tail_selection_full_count += int(holdout_coords.shape[0])
                        full_tail_selection_coord_parts.append(repeated_coords)
                        full_tail_selection_target_parts.append(repeated_targets)
                        full_tail_selection_selected += int(len(holdout_tail))
                        full_tail_selection_evaluated += 1
                    else:
                        full_tail_coord_parts.append(repeated_coords)
                        full_tail_target_parts.append(repeated_targets)
                        full_tail_holdout_selected += int(len(holdout_tail))
                        full_tail_holdout_evaluated += 1
                full_tail_polish_coords = np.vstack([full_tail_base_coords] + full_tail_coord_parts)
                full_tail_polish_targets = np.concatenate([full_tail_base_targets] + full_tail_target_parts)
                full_tail_selection_coords = (
                    np.vstack(full_tail_selection_coord_parts)
                    if full_tail_selection_coord_parts
                    else None
                )
                full_tail_selection_targets = (
                    np.concatenate(full_tail_selection_target_parts)
                    if full_tail_selection_target_parts
                    else None
                )
                param_refit_extra_gradient_evaluated = 0
                param_refit_fit_steps = 0
                param_refit_restarts = 0
                param_refit_gradient_weight = 0.0
                param_refit_train_count = 0
                param_refit_selection_count = 0
                residual_gradient_replace_atoms: list[int] = []
                residual_block_selection: dict[str, Any] = {
                    "method": "not-applicable",
                    "selectedCount": 0,
                    "availableCount": 0,
                    "cutRatios": [],
                }
                validation_gradient_weight = 0.0
                validation_cubic_weight = 0.0
                asymptotic_stitch_line_count = 0
                asymptotic_stitch_samples_per_line = 0
                asymptotic_stitch_line_range = 0.0
                asymptotic_stitch_offset_scale = 0.0
                asymptotic_stitch_scale = 0.0
                if full_tail_method == "lbfgs":
                    full_tail_polished_fit = final_lbfgs_polish_swiglu_candidate(
                        full_tail_polish_coords,
                        full_tail_polish_targets,
                        fit,
                    )
                elif full_tail_method == "gate-norm":
                    full_tail_polished_fit = gate_norm_calibrate_swiglu_candidate(
                        full_tail_polish_coords,
                        full_tail_polish_targets,
                        fit,
                        full_tail_selection_coords,
                        full_tail_selection_targets,
                    )
                elif full_tail_method == "pair-swap":
                    full_tail_polished_fit = pair_swap_refine_swiglu_candidate(
                        full_tail_polish_coords,
                        full_tail_polish_targets,
                        fit,
                    )
                elif full_tail_method == "fixed-gate-up-refit":
                    full_tail_polished_fit = fixed_gate_up_refit_swiglu_candidate(
                        full_tail_polish_coords,
                        full_tail_polish_targets,
                        fit,
                    )
                elif full_tail_method == "gate-dictionary-refit":
                    full_tail_polished_fit = gate_dictionary_refit_swiglu_candidate(
                        full_tail_polish_coords,
                        full_tail_polish_targets,
                        fit,
                        full_tail_fit_gradient_coords,
                        full_tail_fit_gradient_targets,
                        args.atom_count,
                        args.seed + 70731,
                    )
                elif full_tail_method == "cancellation-pair-refit":
                    full_tail_polished_fit = sequential_cancellation_pair_refit_swiglu_candidate(
                        full_tail_polish_coords,
                        full_tail_polish_targets,
                        fit,
                        fit_gradient_coords,
                        fit_gradient_targets,
                        args.seed + 70861,
                        full_tail_selection_coords,
                        full_tail_selection_targets,
                        cubic_directions=cubic_gate_directions,
                        cubic_targets=cubic_scores,
                        cubic_selection_directions=cubic_holdout_directions,
                        cubic_selection_targets=cubic_holdout_scores,
                    )
                elif full_tail_method in {"asymptotic-gate-stitch", "asymptotic-cold-rebuild"}:
                    asymptotic_stitch_line_count = int(os.environ.get(
                        "GCL_BLIND_SWIGLU_ASYMPTOTIC_STITCH_LINE_COUNT",
                        "16",
                    ) or "16")
                    asymptotic_stitch_samples_per_line = int(os.environ.get(
                        "GCL_BLIND_SWIGLU_ASYMPTOTIC_STITCH_SAMPLES_PER_LINE",
                        "1025",
                    ) or "1025")
                    asymptotic_stitch_line_range = float(os.environ.get(
                        "GCL_BLIND_SWIGLU_ASYMPTOTIC_STITCH_LINE_RANGE",
                        "4.0",
                    ) or "4.0")
                    asymptotic_stitch_offset_scale = float(os.environ.get(
                        "GCL_BLIND_SWIGLU_ASYMPTOTIC_STITCH_OFFSET_SCALE",
                        "0.75",
                    ) or "0.75")
                    asymptotic_stitch_scale = float(os.environ.get(
                        "GCL_BLIND_SWIGLU_ASYMPTOTIC_STITCH_SCALE",
                        "512.0",
                    ) or "512.0")
                    asymptotic_stitch_queries = asymptotic_stitch_line_count * asymptotic_stitch_samples_per_line
                    asymptotic_stitch_query_bound = asymptotic_stitch_query_upper_bound(
                        asymptotic_stitch_line_count,
                        asymptotic_stitch_samples_per_line,
                        args.atom_count,
                        args.active_rank,
                        int(os.environ.get(
                            "GCL_BLIND_SWIGLU_ASYMPTOTIC_STITCH_HESSIAN_REPRESENTATIVES",
                            "5",
                        ) or "5"),
                    )
                    if asymptotic_stitch_query_bound > args.max_oracle_calls - oracle.query_count:
                        raise OracleBudgetExceeded(
                            stage="asymptotic-gate-stitch",
                            requested=asymptotic_stitch_query_bound,
                            consumed=oracle.query_count,
                            limit=args.max_oracle_calls,
                        )
                    log_progress(
                        args,
                        "asymptotic-gate-stitch-start",
                        lineCount=int(asymptotic_stitch_line_count),
                        samplesPerLine=int(asymptotic_stitch_samples_per_line),
                        lineRange=float(asymptotic_stitch_line_range),
                        offsetScale=float(asymptotic_stitch_offset_scale),
                        asymptoticScale=float(asymptotic_stitch_scale),
                        requestedLineOracleCalls=int(asymptotic_stitch_queries),
                        requestedOracleCallUpperBound=int(asymptotic_stitch_query_bound),
                        rawOracleCalls=int(oracle.query_count),
                    )
                    stitch_cold_start = full_tail_method == "asymptotic-cold-rebuild"
                    stitch_candidate = (
                        make_asymptotic_cold_start_scaffold(
                            args.atom_count,
                            args.active_rank,
                            full_tail_polish_targets,
                        )
                        if stitch_cold_start
                        else fit
                    )
                    full_tail_polished_fit = asymptotic_gate_stitch_swiglu_candidate(
                        oracle,
                        basis,
                        full_tail_polish_coords,
                        full_tail_polish_targets,
                        stitch_candidate,
                        args.seed + 71021,
                        selection_coords=full_tail_selection_coords,
                        selection_targets=full_tail_selection_targets,
                        gradient_coords=full_tail_fit_gradient_coords,
                        gradient_targets=full_tail_fit_gradient_targets,
                        cubic_directions=cubic_gate_directions,
                        cubic_targets=cubic_scores,
                        line_count=asymptotic_stitch_line_count,
                        samples_per_line=asymptotic_stitch_samples_per_line,
                        line_range=asymptotic_stitch_line_range,
                        offset_scale=asymptotic_stitch_offset_scale,
                        asymptotic_scale=asymptotic_stitch_scale,
                        cold_start=stitch_cold_start,
                    )
                elif full_tail_method == "fixed-direction-value-polish":
                    fixed_direction_gradient_weight = float(os.environ.get(
                        "GCL_BLIND_SWIGLU_FIXED_DIRECTION_VALUE_GRADIENT_WEIGHT",
                        "0.0",
                    ) or "0.0")
                    full_tail_polished_fit = fixed_gate_direction_value_polish(
                        full_tail_polish_coords,
                        full_tail_polish_targets,
                        fit,
                        full_tail_fit_gradient_coords,
                        full_tail_fit_gradient_targets,
                        fixed_direction_gradient_weight,
                    )
                elif full_tail_method in {"residual-gradient-replace", "residual-block-refit"}:
                    if atom_tail_diagnostic is None:
                        local_atom_tail_diagnostic = atom_tail_diagnostics(
                            validation_coords,
                            validation_targets,
                            fit,
                            validation_prediction,
                            float(os.environ.get("GCL_BLIND_SWIGLU_ATOM_TAIL_DIAGNOSTIC_FRACTION", "0.03") or "0.03"),
                            int(os.environ.get("GCL_BLIND_SWIGLU_ATOM_TAIL_DIAGNOSTIC_TOP_K", "8") or "8"),
                        )
                    else:
                        local_atom_tail_diagnostic = atom_tail_diagnostic
                    fixed_residual_top_k = int(os.environ.get(
                        "GCL_BLIND_SWIGLU_RESIDUAL_GRADIENT_REPLACE_TOP_K",
                        "4",
                    ) or "4")
                    adaptive_residual_block = (
                        full_tail_method == "residual-block-refit"
                        and os.environ.get(
                            "GCL_BLIND_SWIGLU_RESIDUAL_BLOCK_REFIT_ADAPTIVE_TOP_K",
                            "1",
                        ).strip() == "1"
                    )
                    residual_block_min_atoms = int(os.environ.get(
                        "GCL_BLIND_SWIGLU_RESIDUAL_BLOCK_REFIT_MIN_ATOMS",
                        str(max(2, int(round(math.sqrt(args.atom_count))))),
                    ) or "0")
                    residual_block_max_atoms = int(os.environ.get(
                        "GCL_BLIND_SWIGLU_RESIDUAL_BLOCK_REFIT_MAX_ATOMS",
                        str(min(8, max(residual_block_min_atoms, args.atom_count // 2))),
                    ) or "0")
                    residual_block_gate_weak_ratio = float(os.environ.get(
                        "GCL_BLIND_SWIGLU_RESIDUAL_BLOCK_REFIT_GATE_WEAK_RATIO",
                        "0.95",
                    ) or "0.95")
                    residual_block_strong_cut_ratio = float(os.environ.get(
                        "GCL_BLIND_SWIGLU_RESIDUAL_BLOCK_REFIT_STRONG_CUT_RATIO",
                        "0.75",
                    ) or "0.75")
                    residual_gradient_replace_atoms, residual_block_selection = select_residual_block_atoms(
                        local_atom_tail_diagnostic,
                        args.atom_count,
                        adaptive=adaptive_residual_block,
                        fixed_top_k=fixed_residual_top_k,
                        min_atoms=residual_block_min_atoms,
                        max_atoms=residual_block_max_atoms,
                        gate_norms=np.linalg.norm(
                            np.asarray(fit["gates"], dtype=np.float64),
                            axis=1,
                        ),
                        gate_weak_ratio=residual_block_gate_weak_ratio,
                        strong_cut_ratio=residual_block_strong_cut_ratio,
                    )
                    residual_gradient_targets = full_tail_fit_gradient_targets - predict_swiglu_active_gradients(
                        full_tail_fit_gradient_coords,
                        fit,
                    )
                    if full_tail_method == "residual-block-refit":
                        full_tail_polished_fit = residual_gradient_block_refit_swiglu_candidate(
                            full_tail_polish_coords,
                            full_tail_polish_targets,
                            fit,
                            residual_gradient_replace_atoms,
                            full_tail_fit_gradient_coords,
                            residual_gradient_targets,
                            args.seed + 70901,
                            full_tail_selection_coords,
                            full_tail_selection_targets,
                            cubic_directions=cubic_gate_directions,
                            cubic_targets=cubic_scores,
                            cubic_selection_directions=cubic_holdout_directions,
                            cubic_selection_targets=cubic_holdout_scores,
                        )
                        full_tail_polished_fit["residualGradientBlockSelection"] = residual_block_selection
                    else:
                        full_tail_polished_fit = residual_gradient_atom_replace_swiglu_candidate(
                            full_tail_polish_coords,
                            full_tail_polish_targets,
                            fit,
                            residual_gradient_replace_atoms,
                            full_tail_fit_gradient_coords,
                            residual_gradient_targets,
                            args.seed + 70771,
                            full_tail_selection_coords,
                            full_tail_selection_targets,
                        )
                elif full_tail_method == "gradient-polish":
                    validation_gradient_weight, validation_cubic_weight = validation_gradient_polish_weights(args)
                    full_tail_polished_fit = polish_swiglu_candidate(
                        full_tail_polish_coords,
                        full_tail_polish_targets,
                        fit,
                        args.fit_learning_rate,
                        full_tail_fit_gradient_coords,
                        full_tail_fit_gradient_targets,
                        validation_gradient_weight,
                        cubic_gate_directions,
                        cubic_scores,
                        validation_cubic_weight,
                    )
                    full_tail_polished_fit = rescore_swiglu_candidate_on_data(
                        full_tail_polish_coords,
                        full_tail_polish_targets,
                        full_tail_polished_fit,
                        recalibrate_scales=True,
                    )
                elif full_tail_method == "param-refit":
                    param_refit_fit_steps = int(os.environ.get(
                        "GCL_BLIND_SWIGLU_PARAM_REFIT_STEPS",
                        str(args.fit_steps),
                    ) or args.fit_steps)
                    param_refit_restarts = int(os.environ.get(
                        "GCL_BLIND_SWIGLU_PARAM_REFIT_RESTARTS",
                        str(args.fit_restarts if args.fit_restarts > 0 else max(args.atom_count * 64, 512)),
                    ) or "0")
                    param_refit_gradient_weight = float(os.environ.get(
                        "GCL_BLIND_SWIGLU_PARAM_REFIT_GRADIENT_WEIGHT",
                        str(args.fit_gradient_weight),
                    ) or args.fit_gradient_weight)
                    param_refit_learning_rate = float(os.environ.get(
                        "GCL_BLIND_SWIGLU_PARAM_REFIT_LEARNING_RATE",
                        str(args.fit_learning_rate),
                    ) or args.fit_learning_rate)
                    param_refit_restart_chunk_size = int(os.environ.get(
                        "GCL_BLIND_SWIGLU_PARAM_REFIT_RESTART_CHUNK_SIZE",
                        str(args.fit_restart_chunk_size),
                    ) or args.fit_restart_chunk_size)
                    param_refit_extra_gradient_target = int(os.environ.get(
                        "GCL_BLIND_SWIGLU_PARAM_REFIT_EXTRA_GRADIENT_COUNT",
                        "0",
                    ) or "0")
                    param_refit_gradient_scale = float(os.environ.get(
                        "GCL_BLIND_SWIGLU_PARAM_REFIT_GRADIENT_SCALE",
                        str(args.validation_scale),
                    ) or args.validation_scale)
                    if (
                        param_refit_fit_steps <= 0
                        or param_refit_restarts <= 0
                        or param_refit_gradient_weight < 0.0
                        or not np.isfinite(param_refit_learning_rate)
                        or param_refit_learning_rate <= 0.0
                        or param_refit_extra_gradient_target < 0
                        or not np.isfinite(param_refit_gradient_scale)
                        or param_refit_gradient_scale <= 0.0
                    ):
                        raise RuntimeError("validation param-refit config is invalid")
                    param_fit_gradient_coords = fit_gradient_coords
                    param_fit_gradient_targets = fit_gradient_targets
                    if param_refit_extra_gradient_target > 0:
                        gradient_call_cost = max(1, len(fit_gradient_steps)) * param_refit_extra_gradient_target * (args.active_rank + 1)
                        if oracle.query_count + gradient_call_cost > args.max_oracle_calls:
                            available = max(0, args.max_oracle_calls - oracle.query_count)
                            per_gradient = max(1, len(fit_gradient_steps)) * (args.active_rank + 1)
                            param_refit_extra_gradient_target = available // per_gradient
                        if param_refit_extra_gradient_target > 0:
                            extra_gradient_coords, extra_gradient_targets = active_gradient_estimator(
                                oracle,
                                fit_gradient_rng,
                                basis,
                                param_refit_extra_gradient_target,
                                param_refit_gradient_scale,
                                fit_gradient_steps,
                            )
                            param_fit_gradient_coords = np.vstack([param_fit_gradient_coords, extra_gradient_coords])
                            param_fit_gradient_targets = np.vstack([param_fit_gradient_targets, extra_gradient_targets])
                            param_refit_extra_gradient_evaluated = int(extra_gradient_coords.shape[0])
                    seed_parts = []
                    for key in ("gates", "ups"):
                        values = np.asarray(fit.get(key, np.zeros((0, args.active_rank), dtype=np.float64)), dtype=np.float64)
                        if values.ndim == 2 and values.shape[1] == args.active_rank and values.shape[0] > 0:
                            seed_parts.append(values)
                    param_gate_seed_directions = np.vstack(seed_parts) if seed_parts else cubic_gate_directions
                    param_refit_train_count = int(full_tail_polish_coords.shape[0])
                    param_refit_selection_count = int(full_tail_selection_coords.shape[0]) if full_tail_selection_coords is not None else 0
                    log_progress(
                        args,
                        "validation-param-refit-start",
                        trainCount=param_refit_train_count,
                        fitGradientCount=int(param_fit_gradient_coords.shape[0]),
                        extraGradientCount=int(param_refit_extra_gradient_evaluated),
                        selectionCount=param_refit_selection_count,
                        fitSteps=int(param_refit_fit_steps),
                        fitRestarts=int(param_refit_restarts),
                        gradientWeight=float(param_refit_gradient_weight),
                        rawOracleCalls=int(oracle.query_count),
                    )
                    full_tail_polished_fit = fit_swiglu_sum(
                        full_tail_polish_coords,
                        full_tail_polish_targets,
                        args.atom_count,
                        args.seed + 70703,
                        param_refit_fit_steps,
                        param_refit_learning_rate,
                        param_refit_restarts,
                        param_fit_gradient_coords,
                        param_fit_gradient_targets,
                        param_gate_seed_directions,
                        param_refit_gradient_weight,
                    args.fit_mode,
                    param_refit_restart_chunk_size,
                    full_tail_selection_coords,
                    full_tail_selection_targets,
                    [(
                        np.asarray(fit["gates"], dtype=np.float64),
                        np.asarray(fit["ups"], dtype=np.float64),
                        np.asarray(fit["scales"], dtype=np.float64),
                        float(fit["bias"]),
                    )],
                )
                    full_tail_polished_fit = rescore_swiglu_candidate_on_data(
                        full_tail_polish_coords,
                        full_tail_polish_targets,
                        full_tail_polished_fit,
                        recalibrate_scales=True,
                    )
                    full_tail_polished_fit["paramRefit"] = True
                    full_tail_polished_fit["paramRefitExtraGradientCount"] = int(param_refit_extra_gradient_evaluated)
                    full_tail_polished_fit["paramRefitGradientCount"] = int(param_fit_gradient_coords.shape[0])
                    full_tail_polished_fit["paramRefitTrainCount"] = int(param_refit_train_count)
                    full_tail_polished_fit["paramRefitSelectionCount"] = int(param_refit_selection_count)
                else:
                    full_tail_polished_fit = tail_error_polish_swiglu_candidate(
                        full_tail_polish_coords,
                        full_tail_polish_targets,
                        fit,
                        full_tail_steps,
                        full_tail_selection_coords,
                        full_tail_selection_targets,
                    )
                raw_asymptotic_directions = full_tail_polished_fit.get(
                    "asymptoticGateStitchRecoveredDirections"
                )
                if raw_asymptotic_directions is not None:
                    observed_asymptotic_directions = np.asarray(
                        raw_asymptotic_directions,
                        dtype=np.float64,
                    )
                    if observed_asymptotic_directions.shape == (args.atom_count, args.active_rank):
                        asymptotic_stitch_posthoc_directions = observed_asymptotic_directions
                raw_asymptotic_up_directions = full_tail_polished_fit.get(
                    "asymptoticGateStitchRecoveredUpDirections"
                )
                if raw_asymptotic_up_directions is not None:
                    observed_asymptotic_up_directions = np.asarray(
                        raw_asymptotic_up_directions,
                        dtype=np.float64,
                    )
                    if observed_asymptotic_up_directions.shape == (args.atom_count, args.active_rank):
                        asymptotic_stitch_posthoc_up_directions = observed_asymptotic_up_directions
                full_tail_probe_prediction = predict_swiglu(validation_coords, full_tail_polished_fit)
                full_tail_probe_metrics = regression_metrics(validation_targets, full_tail_probe_prediction)
                full_tail_revalidation_metrics: dict[str, float] | None = None
                full_tail_revalidation_baseline_metrics: dict[str, float] | None = None
                full_tail_accepted = False
                residual_gradient_cegis_cycles = 0
                # Initialised at this scope, alongside the cycle counter, because the report
                # is assembled here whether or not the CEGIS branch below runs.
                cegis_cycle_metrics: list[dict[str, Any]] = []
                residual_gradient_cegis_selection_count = 0
                residual_gradient_cegis_counterexample_count = 0
                if oracle.query_count + args.validation_count <= args.max_oracle_calls:
                    full_tail_validation_rng = np.random.default_rng(args.seed + 70627)
                    full_tail_validation_inputs = full_tail_validation_rng.normal(
                        0.0,
                        args.validation_scale,
                        size=(args.validation_count, dimension),
                    ).astype(np.float32)
                    full_tail_validation_targets = oracle.eval(full_tail_validation_inputs)
                    full_tail_validation_coords = full_tail_validation_inputs @ basis.T
                    full_tail_baseline_prediction = predict_swiglu(full_tail_validation_coords, fit)
                    full_tail_revalidation_baseline_metrics = regression_metrics(full_tail_validation_targets, full_tail_baseline_prediction)
                    full_tail_validation_prediction = predict_swiglu(full_tail_validation_coords, full_tail_polished_fit)
                    full_tail_revalidation_metrics = regression_metrics(full_tail_validation_targets, full_tail_validation_prediction)
                    full_tail_accepted = (
                        passes_metric_thresholds(full_tail_revalidation_metrics, args.min_r2, args.max_mae, args.max_worst_error)
                        or (
                            full_tail_revalidation_baseline_metrics is not None
                            and full_tail_revalidation_metrics["mae"] <= full_tail_revalidation_baseline_metrics["mae"]
                            and full_tail_revalidation_metrics["worstError"] < full_tail_revalidation_baseline_metrics["worstError"]
                        )
                    )
                if (
                    full_tail_method in {"residual-gradient-replace", "residual-block-refit"}
                    and full_tail_revalidation_metrics is not None
                    and not passes_metric_thresholds(
                        full_tail_revalidation_metrics,
                        args.min_r2,
                        args.max_mae,
                        args.max_worst_error,
                    )
                ):
                    cegis_limit = max(0, int(os.environ.get(
                        "GCL_BLIND_SWIGLU_RESIDUAL_GRADIENT_REPLACE_CEGIS_CYCLES",
                        "2",
                    ) or "0"))
                    cegis_counterexample_target = max(1, int(os.environ.get(
                        "GCL_BLIND_SWIGLU_RESIDUAL_GRADIENT_REPLACE_CEGIS_COUNTEREXAMPLE_COUNT",
                        "128",
                    ) or "0"))
                    cegis_counterexample_repeat = max(1, int(os.environ.get(
                        "GCL_BLIND_SWIGLU_RESIDUAL_GRADIENT_REPLACE_CEGIS_COUNTEREXAMPLE_REPEAT",
                        "4",
                    ) or "0"))
                    cegis_selection_coords = full_tail_validation_coords.copy()
                    cegis_selection_targets = full_tail_validation_targets.copy()
                    cegis_train_coords = full_tail_polish_coords.copy()
                    cegis_train_targets = full_tail_polish_targets.copy()
                    cegis_candidate_pool = [full_tail_polished_fit]
                    # Every cycle draws a fresh oracle validation set, assigns its numbers to
                    # the reported metrics, and the loop breaks the moment a draw clears the
                    # thresholds. That is not train-on-test, since each cycle is evaluated on
                    # points the candidate has not seen, but it is a stopping rule: the
                    # published figure is the first of up to `cegis_limit` independent draws
                    # that happened to pass, which biases a max statistic like worstError.
                    # Recording every draw, not just the accepted one, is what lets a reader
                    # see that and judge the margin.
                    for cegis_cycle in range(cegis_limit):
                        if oracle.query_count + args.validation_count > args.max_oracle_calls:
                            break
                        residual_gradient_cegis_cycles += 1
                        residual_gradient_cegis_selection_count = int(cegis_selection_coords.shape[0])
                        cegis_current_prediction = predict_swiglu(cegis_selection_coords, full_tail_polished_fit)
                        cegis_current_order = np.argsort(
                            np.abs(cegis_selection_targets - cegis_current_prediction)
                        )[::-1]
                        cegis_counterexample_indices = cegis_current_order[:min(
                            cegis_counterexample_target,
                            cegis_selection_coords.shape[0],
                        )]
                        cegis_train_coords = np.vstack([
                            cegis_train_coords,
                            np.repeat(
                                cegis_selection_coords[cegis_counterexample_indices],
                                cegis_counterexample_repeat,
                                axis=0,
                            ),
                        ])
                        cegis_train_targets = np.concatenate([
                            cegis_train_targets,
                            np.repeat(
                                cegis_selection_targets[cegis_counterexample_indices],
                                cegis_counterexample_repeat,
                                axis=0,
                            ),
                        ])
                        residual_gradient_cegis_counterexample_count += int(len(cegis_counterexample_indices))
                        cegis_selection_mask = np.ones(cegis_selection_coords.shape[0], dtype=bool)
                        cegis_selection_mask[cegis_counterexample_indices] = False
                        cycle_selection_coords = cegis_selection_coords[cegis_selection_mask]
                        cycle_selection_targets = cegis_selection_targets[cegis_selection_mask]
                        if full_tail_selection_coords is not None and full_tail_selection_targets is not None:
                            cycle_selection_coords = np.vstack([
                                full_tail_selection_coords,
                                cycle_selection_coords,
                            ])
                            cycle_selection_targets = np.concatenate([
                                full_tail_selection_targets,
                                cycle_selection_targets,
                            ])
                        if cycle_selection_coords.shape[0] == 0:
                            cycle_selection_coords = None
                            cycle_selection_targets = None
                            residual_gradient_cegis_selection_count = 0
                        else:
                            residual_gradient_cegis_selection_count = int(cycle_selection_coords.shape[0])
                        cegis_residual_gradient_targets = fit_gradient_targets - predict_swiglu_active_gradients(
                            fit_gradient_coords,
                            full_tail_polished_fit,
                        )
                        if full_tail_method == "residual-block-refit":
                            cegis_candidate = residual_gradient_block_refit_swiglu_candidate(
                                cegis_train_coords,
                                cegis_train_targets,
                                full_tail_polished_fit,
                                residual_gradient_replace_atoms,
                                fit_gradient_coords,
                                cegis_residual_gradient_targets,
                                args.seed + 70841 + cegis_cycle * 97,
                                cycle_selection_coords,
                                cycle_selection_targets,
                                cubic_directions=cubic_gate_directions,
                                cubic_targets=cubic_scores,
                                cubic_selection_directions=cubic_holdout_directions,
                                cubic_selection_targets=cubic_holdout_scores,
                            )
                        else:
                            cegis_candidate = residual_gradient_atom_replace_swiglu_candidate(
                                cegis_train_coords,
                                cegis_train_targets,
                                full_tail_polished_fit,
                                residual_gradient_replace_atoms,
                                fit_gradient_coords,
                                cegis_residual_gradient_targets,
                                args.seed + 70841 + cegis_cycle * 97,
                                cycle_selection_coords,
                                cycle_selection_targets,
                            )
                        cegis_candidate_pool.append(cegis_candidate)
                        cegis_court = cegis_candidate_pool
                        court_coords = (
                            cycle_selection_coords
                            if cycle_selection_coords is not None
                            else cegis_selection_coords
                        )
                        court_targets = (
                            cycle_selection_targets
                            if cycle_selection_targets is not None
                            else cegis_selection_targets
                        )
                        cegis_candidate = min(
                            cegis_court,
                            key=lambda item: (
                                regression_metrics(
                                    court_targets,
                                    predict_swiglu(court_coords, item),
                                )["mae"]
                                + 0.35 * regression_metrics(
                                    court_targets,
                                    predict_swiglu(court_coords, item),
                                )["worstError"]
                            ),
                        )
                        cegis_validation_rng = np.random.default_rng(args.seed + 71641 + cegis_cycle * 1009)
                        cegis_validation_inputs = cegis_validation_rng.normal(
                            0.0,
                            args.validation_scale,
                            size=(args.validation_count, dimension),
                        ).astype(np.float32)
                        cegis_validation_targets = oracle.eval(cegis_validation_inputs)
                        cegis_validation_coords = cegis_validation_inputs @ basis.T
                        cegis_baseline_prediction = predict_swiglu(cegis_validation_coords, fit)
                        cegis_baseline_metrics = regression_metrics(cegis_validation_targets, cegis_baseline_prediction)
                        cegis_prediction = predict_swiglu(cegis_validation_coords, cegis_candidate)
                        cegis_metrics = regression_metrics(cegis_validation_targets, cegis_prediction)
                        full_tail_polished_fit = cegis_candidate
                        full_tail_validation_inputs = cegis_validation_inputs
                        full_tail_validation_targets = cegis_validation_targets
                        full_tail_validation_coords = cegis_validation_coords
                        full_tail_validation_prediction = cegis_prediction
                        full_tail_revalidation_baseline_metrics = cegis_baseline_metrics
                        full_tail_revalidation_metrics = cegis_metrics
                        full_tail_accepted = (
                            passes_metric_thresholds(cegis_metrics, args.min_r2, args.max_mae, args.max_worst_error)
                            or (
                                cegis_metrics["mae"] <= cegis_baseline_metrics["mae"]
                                and cegis_metrics["worstError"] < cegis_baseline_metrics["worstError"]
                            )
                        )
                        log_progress(
                            args,
                            "validation-residual-gradient-cegis-cycle",
                            cycle=int(cegis_cycle + 1),
                            selectionCount=int(residual_gradient_cegis_selection_count),
                            counterexampleCount=int(len(cegis_counterexample_indices)),
                            counterexampleRepeat=int(cegis_counterexample_repeat),
                            trainCount=int(cegis_train_coords.shape[0]),
                            replaced=bool(cegis_candidate.get("residualGradientAtomReplaced", False)),
                            acceptedAtoms=cegis_candidate.get("residualGradientAtomReplaceAcceptedAtoms", []),
                            r2=float(cegis_metrics["r2"]),
                            mae=float(cegis_metrics["mae"]),
                            worstError=float(cegis_metrics["worstError"]),
                            passThresholds=bool(passes_metric_thresholds(
                                cegis_metrics,
                                args.min_r2,
                                args.max_mae,
                                args.max_worst_error,
                            )),
                            rawOracleCalls=int(oracle.query_count),
                        )
                        cegis_cycle_passed = bool(passes_metric_thresholds(
                            cegis_metrics, args.min_r2, args.max_mae, args.max_worst_error
                        ))
                        cegis_cycle_metrics.append({
                            "cycle": int(cegis_cycle + 1),
                            "r2": float(cegis_metrics["r2"]),
                            "mae": float(cegis_metrics["mae"]),
                            "worstError": float(cegis_metrics["worstError"]),
                            "passedThresholds": cegis_cycle_passed,
                            "rawOracleCalls": int(oracle.query_count),
                        })
                        if cegis_cycle_passed:
                            break
                        cegis_selection_coords = cegis_validation_coords
                        cegis_selection_targets = cegis_validation_targets
                fit_attempts.append({
                    "stage": "validation-full-tail-polish",
                    "cegisCycleMetrics": cegis_cycle_metrics,
                    "cegisCyclesRun": len(cegis_cycle_metrics),
                    "cegisAcceptedCycle": (
                        cegis_cycle_metrics[-1]["cycle"]
                        if cegis_cycle_metrics and cegis_cycle_metrics[-1]["passedThresholds"]
                        else None
                    ),
                    "cegisStoppingRule": (
                        "first draw clearing the thresholds; reported metrics are that draw's, "
                        "so compare them against cegisCycleMetrics before treating the margin as typical"
                    ),
                    "trainCount": int(full_tail_polish_coords.shape[0]),
                    "baseTrainCount": int(full_tail_base_coords.shape[0]),
                    "extraRandomTrainCount": int(full_tail_extra_random_evaluated),
                    "radialTrainCount": int(full_tail_radial_train_evaluated),
                    "radialSelectionCount": int(full_tail_radial_selection_evaluated),
                    "radialGradientCount": int(full_tail_radial_gradient_evaluated),
                    "radialGradientScales": [float(scale) for scale in full_tail_radial_gradient_scales],
                    "radialGradientSteps": [float(step) for step in full_tail_radial_gradient_steps],
                    "radialScales": [float(scale) for scale in full_tail_radial_scales],
                    "tailCount": int(len(full_tail_selected)),
                    "tailRepeat": int(full_tail_repeat),
                    "holdoutPoolsRequested": int(full_tail_holdout_pools),
                    "holdoutPoolsEvaluated": int(full_tail_holdout_evaluated),
                    "holdoutTailSelected": int(full_tail_holdout_selected),
                    "holdoutTailRepeat": int(full_tail_holdout_tail_repeat),
                    "selectionHoldoutPoolsRequested": int(full_tail_selection_holdout_pools),
                    "selectionHoldoutPoolsEvaluated": int(full_tail_selection_evaluated),
                    "selectionHoldoutTailSelected": int(full_tail_selection_selected),
                    "selectionHoldoutFullCount": int(full_tail_selection_full_count),
                    "method": full_tail_method,
                    "gradientWeight": float(validation_gradient_weight),
                    "cubicWeight": float(validation_cubic_weight),
                    "tailErrorPolished": bool(full_tail_polished_fit.get("tailErrorPolished", False)),
                    "finalLbfgsPolished": bool(full_tail_polished_fit.get("finalLbfgsPolished", False)),
                    "finalLbfgsCourt": bool(full_tail_polished_fit.get("finalLbfgsCourt", False)),
                    "gateNormCalibrated": bool(full_tail_polished_fit.get("gateNormCalibrated", False)),
                    "pairSwapRefined": bool(full_tail_polished_fit.get("pairSwapRefined", False)),
                    "fixedGateUpRefit": bool(full_tail_polished_fit.get("fixedGateUpRefit", False)),
                    "gateDictionaryRefit": bool(full_tail_polished_fit.get("gateDictionaryRefit", False)),
                    "gateDictionaryRefitCandidateCount": int(full_tail_polished_fit.get("gateDictionaryRefitCandidateCount", 0)),
                    "asymptoticGateStitched": bool(full_tail_polished_fit.get("asymptoticGateStitched", False)),
                    "asymptoticGateStitchLineCount": int(full_tail_polished_fit.get("asymptoticGateStitchLineCount", asymptotic_stitch_line_count)),
                    "asymptoticGateStitchUsableLineCount": int(full_tail_polished_fit.get("asymptoticGateStitchUsableLineCount", 0)),
                    "asymptoticGateStitchFailedLineCount": int(full_tail_polished_fit.get("asymptoticGateStitchFailedLineCount", 0)),
                    "asymptoticGateStitchSamplesPerLine": int(full_tail_polished_fit.get("asymptoticGateStitchSamplesPerLine", asymptotic_stitch_samples_per_line)),
                    "asymptoticGateStitchLineRange": float(asymptotic_stitch_line_range),
                    "asymptoticGateStitchOffsetScale": float(asymptotic_stitch_offset_scale),
                    "asymptoticGateStitchScale": float(asymptotic_stitch_scale),
                    "asymptoticGateStitchMedianBoundaryResidual": full_tail_polished_fit.get("asymptoticGateStitchMedianBoundaryResidual"),
                    "asymptoticGateStitchMaxBoundaryResidual": full_tail_polished_fit.get("asymptoticGateStitchMaxBoundaryResidual"),
                    "asymptoticGateStitchMeanDirectionDelta": full_tail_polished_fit.get("asymptoticGateStitchMeanDirectionDelta"),
                    "asymptoticGateStitchBoundaryNormalCount": int(full_tail_polished_fit.get("asymptoticGateStitchBoundaryNormalCount", 0)),
                    "asymptoticGateStitchClusterCount": int(full_tail_polished_fit.get("asymptoticGateStitchClusterCount", 0)),
                    "asymptoticGateStitchClusterSupports": full_tail_polished_fit.get("asymptoticGateStitchClusterSupports", []),
                    "asymptoticGateStitchGradientStep": full_tail_polished_fit.get("asymptoticGateStitchGradientStep"),
                    "asymptoticGateStitchValuePolished": bool(full_tail_polished_fit.get("asymptoticGateStitchValuePolished", False)),
                    "asymptoticGateStitchHessianUpComplete": bool(full_tail_polished_fit.get("asymptoticGateStitchHessianUpComplete", False)),
                    "asymptoticGateStitchHessianUpCount": int(full_tail_polished_fit.get("asymptoticGateStitchHessianUpCount", 0)),
                    "asymptoticGateStitchHessianUpSupports": full_tail_polished_fit.get("asymptoticGateStitchHessianUpSupports", []),
                    "asymptoticGateStitchAssignedGateSupports": full_tail_polished_fit.get("asymptoticGateStitchAssignedGateSupports", []),
                    "asymptoticGateStitchAssignedUpSupports": full_tail_polished_fit.get("asymptoticGateStitchAssignedUpSupports", []),
                    "asymptoticGateStitchFallbackAtoms": full_tail_polished_fit.get("asymptoticGateStitchFallbackAtoms", []),
                    "asymptoticGateStitchStructureCalibrated": bool(full_tail_polished_fit.get("asymptoticGateStitchStructureCalibrated", False)),
                    "asymptoticGateStitchStructureRefined": bool(full_tail_polished_fit.get("asymptoticGateStitchStructureRefined", False)),
                    "asymptoticGateStitchStructureRefineAttempted": bool(full_tail_polished_fit.get("asymptoticGateStitchStructureRefineAttempted", False)),
                    "asymptoticGateStitchStructureRefineSelectionScoreBefore": full_tail_polished_fit.get("asymptoticGateStitchStructureRefineSelectionScoreBefore"),
                    "asymptoticGateStitchStructureRefineSelectionScoreAfter": full_tail_polished_fit.get("asymptoticGateStitchStructureRefineSelectionScoreAfter"),
                    "asymptoticGateStitchStructureRefineRestarts": full_tail_polished_fit.get("asymptoticGateStitchStructureRefineRestarts", []),
                    "asymptoticGateStitchTrialR2": full_tail_polished_fit.get("asymptoticGateStitchTrialR2"),
                    "asymptoticGateStitchTrialMae": full_tail_polished_fit.get("asymptoticGateStitchTrialMae"),
                    "asymptoticGateStitchTrialWorstError": full_tail_polished_fit.get("asymptoticGateStitchTrialWorstError"),
                    "asymptoticGateStitchScoreBefore": full_tail_polished_fit.get("asymptoticGateStitchScoreBefore"),
                    "asymptoticGateStitchScoreAfter": full_tail_polished_fit.get("asymptoticGateStitchScoreAfter"),
                    "residualGradientAtomReplaced": bool(full_tail_polished_fit.get("residualGradientAtomReplaced", False)),
                    "residualGradientAtomReplaceAtoms": residual_gradient_replace_atoms,
                    "residualGradientAtomReplaceProposedAtoms": full_tail_polished_fit.get("residualGradientAtomReplaceProposedAtoms", []),
                    "residualGradientAtomReplaceAcceptedAtoms": full_tail_polished_fit.get("residualGradientAtomReplaceAcceptedAtoms", []),
                    "residualGradientAtomReplaceDirectionCount": int(full_tail_polished_fit.get("residualGradientAtomReplaceDirectionCount", 0)),
                    "residualGradientAtomReplaceCandidateCount": int(full_tail_polished_fit.get("residualGradientAtomReplaceCandidateCount", 0)),
                    "residualGradientAtomReplaceSubsetCount": int(full_tail_polished_fit.get("residualGradientAtomReplaceSubsetCount", 0)),
                    "residualGradientAtomReplaceGradientScoreBefore": full_tail_polished_fit.get("residualGradientAtomReplaceGradientScoreBefore"),
                    "residualGradientAtomReplaceGradientScoreAfter": full_tail_polished_fit.get("residualGradientAtomReplaceGradientScoreAfter"),
                    "residualGradientAtomReplaceSelectionCount": int(full_tail_polished_fit.get("residualGradientAtomReplaceSelectionCount", 0)),
                    "residualGradientAtomReplaceDecisionScoreBefore": full_tail_polished_fit.get("residualGradientAtomReplaceDecisionScoreBefore"),
                    "residualGradientAtomReplaceDecisionScoreAfter": full_tail_polished_fit.get("residualGradientAtomReplaceDecisionScoreAfter"),
                    "residualGradientAtomReplaceCegisCycles": int(residual_gradient_cegis_cycles),
                    "residualGradientAtomReplaceCegisSelectionCount": int(residual_gradient_cegis_selection_count),
                    "residualGradientAtomReplaceCegisCounterexampleCount": int(
                        residual_gradient_cegis_counterexample_count
                    ),
                    "residualGradientBlockRefit": bool(full_tail_polished_fit.get("residualGradientBlockRefit", False)),
                    "residualGradientBlockRefitAtoms": full_tail_polished_fit.get("residualGradientBlockRefitAtoms", []),
                    "residualGradientBlockRefitDirectionCount": int(full_tail_polished_fit.get("residualGradientBlockRefitDirectionCount", 0)),
                    "residualGradientBlockRefitRestarts": int(full_tail_polished_fit.get("residualGradientBlockRefitRestarts", 0)),
                    "residualGradientBlockRefitSteps": int(full_tail_polished_fit.get("residualGradientBlockRefitSteps", 0)),
                    "residualGradientBlockSelection": full_tail_polished_fit.get(
                        "residualGradientBlockSelection",
                        residual_block_selection,
                    ),
                    "residualGradientBlockRefitSelectionCount": int(full_tail_polished_fit.get("residualGradientBlockRefitSelectionCount", 0)),
                    "residualGradientBlockRefitScoreBefore": full_tail_polished_fit.get("residualGradientBlockRefitScoreBefore"),
                    "residualGradientBlockRefitScoreAfter": full_tail_polished_fit.get("residualGradientBlockRefitScoreAfter"),
                    "residualGradientBlockRefitOutputScoreBefore": full_tail_polished_fit.get(
                        "residualGradientBlockRefitOutputScoreBefore"
                    ),
                    "residualGradientBlockRefitOutputScoreAfter": full_tail_polished_fit.get(
                        "residualGradientBlockRefitOutputScoreAfter"
                    ),
                    "residualGradientBlockRefitCubicRelativeRmseBefore": full_tail_polished_fit.get(
                        "residualGradientBlockRefitCubicRelativeRmseBefore"
                    ),
                    "residualGradientBlockRefitCubicRelativeRmseAfter": full_tail_polished_fit.get(
                        "residualGradientBlockRefitCubicRelativeRmseAfter"
                    ),
                    "residualGradientBlockRefitCubicSelectionWeight": float(full_tail_polished_fit.get(
                        "residualGradientBlockRefitCubicSelectionWeight",
                        0.0,
                    )),
                    "residualGradientBlockRefitCubicFitWeight": float(full_tail_polished_fit.get(
                        "residualGradientBlockRefitCubicFitWeight",
                        0.0,
                    )),
                    "residualGradientBlockRefitCubicDirectionCount": int(full_tail_polished_fit.get(
                        "residualGradientBlockRefitCubicDirectionCount",
                        0,
                    )),
                    "residualGradientBlockRefitCubicSelectionDirectionCount": int(full_tail_polished_fit.get(
                        "residualGradientBlockRefitCubicSelectionDirectionCount",
                        0,
                    )),
                    "residualGradientBlockRefitCancellationPairsBefore": full_tail_polished_fit.get(
                        "residualGradientBlockRefitCancellationPairsBefore",
                        [],
                    ),
                    "residualGradientBlockRefitCancellationPairsAfter": full_tail_polished_fit.get(
                        "residualGradientBlockRefitCancellationPairsAfter",
                        [],
                    ),
                    "residualGradientBlockRefitCancellationRescueAttempted": bool(full_tail_polished_fit.get(
                        "residualGradientBlockRefitCancellationRescueAttempted",
                        False,
                    )),
                    "residualGradientBlockRefitCancellationRescued": bool(full_tail_polished_fit.get(
                        "residualGradientBlockRefitCancellationRescued",
                        False,
                    )),
                    "residualGradientBlockRefitCancellationRescueAtoms": full_tail_polished_fit.get(
                        "residualGradientBlockRefitCancellationRescueAtoms",
                        [],
                    ),
                    "residualGradientBlockRefitCancellationRescueScoreBefore": full_tail_polished_fit.get(
                        "residualGradientBlockRefitCancellationRescueScoreBefore"
                    ),
                    "residualGradientBlockRefitCancellationRescueScoreAfter": full_tail_polished_fit.get(
                        "residualGradientBlockRefitCancellationRescueScoreAfter"
                    ),
                    "residualGradientBlockRefitCancellationRescueAttempts": full_tail_polished_fit.get(
                        "residualGradientBlockRefitCancellationRescueAttempts",
                        [],
                    ),
                    "gradientPolished": bool(full_tail_polished_fit.get("polished", False) or full_tail_polished_fit.get("fixedGateDirectionValuePolished", False)),
                    "paramRefit": bool(full_tail_polished_fit.get("paramRefit", False)),
                    "paramRefitFitSteps": int(param_refit_fit_steps),
                    "paramRefitRestarts": int(param_refit_restarts),
                    "paramRefitGradientWeight": float(param_refit_gradient_weight),
                    "paramRefitExtraGradientCount": int(param_refit_extra_gradient_evaluated),
                    "paramRefitGradientCount": int(full_tail_polished_fit.get("paramRefitGradientCount", 0)),
                    "paramRefitTrainCount": int(param_refit_train_count),
                    "paramRefitSelectionCount": int(param_refit_selection_count),
                    "tailErrorPolishSelectionCount": int(full_tail_polished_fit.get("tailErrorPolishSelectionCount", 0)),
                    "gateNormCalibrationSelectionCount": int(full_tail_polished_fit.get("gateNormCalibrationSelectionCount", 0)),
                    "asymptoticGateStitchStructureCalibrationAttempted": bool(full_tail_polished_fit.get(
                        "asymptoticGateStitchStructureCalibrationAttempted",
                        False,
                    )),
                    "asymptoticGateStitchStructureCalibrationAttemptR2": full_tail_polished_fit.get(
                        "asymptoticGateStitchStructureCalibrationAttemptR2"
                    ),
                    "asymptoticGateStitchStructureCalibrationAttemptMae": full_tail_polished_fit.get(
                        "asymptoticGateStitchStructureCalibrationAttemptMae"
                    ),
                    "asymptoticGateStitchStructureCalibrationAttemptWorst": full_tail_polished_fit.get(
                        "asymptoticGateStitchStructureCalibrationAttemptWorst"
                    ),
                    "asymptoticGateStitchStructureCalibrationRestarts": full_tail_polished_fit.get(
                        "asymptoticGateStitchStructureCalibrationRestarts",
                        [],
                    ),
                    "probeValidationR2": float(full_tail_probe_metrics["r2"]),
                    "probeValidationMae": float(full_tail_probe_metrics["mae"]),
                    "probeValidationWorstError": float(full_tail_probe_metrics["worstError"]),
                    "revalidationPerformed": full_tail_revalidation_metrics is not None,
                    "baselineRevalidationR2": None if full_tail_revalidation_baseline_metrics is None else float(full_tail_revalidation_baseline_metrics["r2"]),
                    "baselineRevalidationMae": None if full_tail_revalidation_baseline_metrics is None else float(full_tail_revalidation_baseline_metrics["mae"]),
                    "baselineRevalidationWorstError": None if full_tail_revalidation_baseline_metrics is None else float(full_tail_revalidation_baseline_metrics["worstError"]),
                    "revalidationR2": None if full_tail_revalidation_metrics is None else float(full_tail_revalidation_metrics["r2"]),
                    "revalidationMae": None if full_tail_revalidation_metrics is None else float(full_tail_revalidation_metrics["mae"]),
                    "revalidationWorstError": None if full_tail_revalidation_metrics is None else float(full_tail_revalidation_metrics["worstError"]),
                    "accepted": bool(full_tail_accepted),
                    "rawOracleCalls": int(oracle.query_count),
                })
                log_progress(
                    args,
                    "validation-full-tail-polish-complete",
                    accepted=bool(full_tail_accepted),
                    polished=bool(
                        full_tail_polished_fit.get("tailErrorPolished", False)
                        or full_tail_polished_fit.get("polished", False)
                        or full_tail_polished_fit.get("fixedGateDirectionValuePolished", False)
                        or full_tail_polished_fit.get("finalLbfgsPolished", False)
                        or full_tail_polished_fit.get("gateNormCalibrated", False)
                        or full_tail_polished_fit.get("structureTrustRegionAccepted", False)
                        or full_tail_polished_fit.get("pairSwapRefined", False)
                        or full_tail_polished_fit.get("fixedGateUpRefit", False)
                        or full_tail_polished_fit.get("gateDictionaryRefit", False)
                        or full_tail_polished_fit.get("residualGradientAtomReplaced", False)
                        or full_tail_polished_fit.get("residualGradientBlockRefit", False)
                        or full_tail_polished_fit.get("asymptoticGateStitched", False)
                    ),
                    tailErrorPolished=bool(full_tail_polished_fit.get("tailErrorPolished", False)),
                    method=full_tail_method,
                    lbfgsPolished=bool(full_tail_polished_fit.get("finalLbfgsPolished", False)),
                    lbfgsCourt=bool(full_tail_polished_fit.get("finalLbfgsCourt", False)),
                    gateNormCalibrated=bool(full_tail_polished_fit.get("gateNormCalibrated", False)),
                    pairSwapRefined=bool(full_tail_polished_fit.get("pairSwapRefined", False)),
                    fixedGateUpRefit=bool(full_tail_polished_fit.get("fixedGateUpRefit", False)),
                    gateDictionaryRefit=bool(full_tail_polished_fit.get("gateDictionaryRefit", False)),
                    asymptoticGateStitched=bool(full_tail_polished_fit.get("asymptoticGateStitched", False)),
                    asymptoticGateStitchUsableLineCount=int(full_tail_polished_fit.get("asymptoticGateStitchUsableLineCount", 0)),
                    asymptoticGateStitchFailedLineCount=int(full_tail_polished_fit.get("asymptoticGateStitchFailedLineCount", 0)),
                    asymptoticGateStitchMedianBoundaryResidual=full_tail_polished_fit.get("asymptoticGateStitchMedianBoundaryResidual"),
                    asymptoticGateStitchMaxBoundaryResidual=full_tail_polished_fit.get("asymptoticGateStitchMaxBoundaryResidual"),
                    asymptoticGateStitchMeanDirectionDelta=full_tail_polished_fit.get("asymptoticGateStitchMeanDirectionDelta"),
                    asymptoticGateStitchBoundaryNormalCount=int(full_tail_polished_fit.get("asymptoticGateStitchBoundaryNormalCount", 0)),
                    asymptoticGateStitchClusterCount=int(full_tail_polished_fit.get("asymptoticGateStitchClusterCount", 0)),
                    asymptoticGateStitchClusterSupports=full_tail_polished_fit.get("asymptoticGateStitchClusterSupports", []),
                    asymptoticGateStitchValuePolished=bool(full_tail_polished_fit.get("asymptoticGateStitchValuePolished", False)),
                    asymptoticGateStitchHessianUpComplete=bool(full_tail_polished_fit.get("asymptoticGateStitchHessianUpComplete", False)),
                    asymptoticGateStitchHessianUpCount=int(full_tail_polished_fit.get("asymptoticGateStitchHessianUpCount", 0)),
                    asymptoticGateStitchHessianUpSupports=full_tail_polished_fit.get("asymptoticGateStitchHessianUpSupports", []),
                    asymptoticGateStitchAssignedGateSupports=full_tail_polished_fit.get("asymptoticGateStitchAssignedGateSupports", []),
                    asymptoticGateStitchAssignedUpSupports=full_tail_polished_fit.get("asymptoticGateStitchAssignedUpSupports", []),
                    asymptoticGateStitchFallbackAtoms=full_tail_polished_fit.get("asymptoticGateStitchFallbackAtoms", []),
                    asymptoticGateStitchStructureRefined=bool(full_tail_polished_fit.get("asymptoticGateStitchStructureRefined", False)),
                    residualGradientAtomReplaced=bool(full_tail_polished_fit.get("residualGradientAtomReplaced", False)),
                    residualGradientAtomReplaceAtoms=residual_gradient_replace_atoms,
                    residualGradientAtomReplaceProposedAtoms=full_tail_polished_fit.get("residualGradientAtomReplaceProposedAtoms", []),
                    residualGradientAtomReplaceAcceptedAtoms=full_tail_polished_fit.get("residualGradientAtomReplaceAcceptedAtoms", []),
                    residualGradientAtomReplaceDirectionCount=int(full_tail_polished_fit.get("residualGradientAtomReplaceDirectionCount", 0)),
                    residualGradientAtomReplaceCandidateCount=int(full_tail_polished_fit.get("residualGradientAtomReplaceCandidateCount", 0)),
                    residualGradientAtomReplaceSubsetCount=int(full_tail_polished_fit.get("residualGradientAtomReplaceSubsetCount", 0)),
                    residualGradientAtomReplaceGradientScoreBefore=full_tail_polished_fit.get("residualGradientAtomReplaceGradientScoreBefore"),
                    residualGradientAtomReplaceGradientScoreAfter=full_tail_polished_fit.get("residualGradientAtomReplaceGradientScoreAfter"),
                    residualGradientAtomReplaceCegisCycles=int(residual_gradient_cegis_cycles),
                    residualGradientAtomReplaceCegisSelectionCount=int(residual_gradient_cegis_selection_count),
                    residualGradientAtomReplaceCegisCounterexampleCount=int(
                        residual_gradient_cegis_counterexample_count
                    ),
                    residualGradientBlockRefit=bool(full_tail_polished_fit.get("residualGradientBlockRefit", False)),
                    residualGradientBlockRefitAtoms=full_tail_polished_fit.get("residualGradientBlockRefitAtoms", []),
                    residualGradientBlockRefitDirectionCount=int(full_tail_polished_fit.get("residualGradientBlockRefitDirectionCount", 0)),
                    residualGradientBlockRefitRestarts=int(full_tail_polished_fit.get("residualGradientBlockRefitRestarts", 0)),
                    residualGradientBlockRefitSteps=int(full_tail_polished_fit.get("residualGradientBlockRefitSteps", 0)),
                    residualGradientBlockRefitCubicRelativeRmseBefore=full_tail_polished_fit.get(
                        "residualGradientBlockRefitCubicRelativeRmseBefore"
                    ),
                    residualGradientBlockRefitCubicRelativeRmseAfter=full_tail_polished_fit.get(
                        "residualGradientBlockRefitCubicRelativeRmseAfter"
                    ),
                    residualGradientBlockSelection=full_tail_polished_fit.get(
                        "residualGradientBlockSelection",
                        residual_block_selection,
                    ),
                    residualGradientBlockRefitCancellationRescueAttempted=bool(full_tail_polished_fit.get(
                        "residualGradientBlockRefitCancellationRescueAttempted",
                        False,
                    )),
                    residualGradientBlockRefitCancellationRescued=bool(full_tail_polished_fit.get(
                        "residualGradientBlockRefitCancellationRescued",
                        False,
                    )),
                    residualGradientBlockRefitCancellationRescueAtoms=full_tail_polished_fit.get(
                        "residualGradientBlockRefitCancellationRescueAtoms",
                        [],
                    ),
                    residualGradientBlockRefitCancellationPairsBefore=full_tail_polished_fit.get(
                        "residualGradientBlockRefitCancellationPairsBefore",
                        [],
                    ),
                    residualGradientBlockRefitCancellationPairsAfter=full_tail_polished_fit.get(
                        "residualGradientBlockRefitCancellationPairsAfter",
                        [],
                    ),
                    gradientPolished=bool(full_tail_polished_fit.get("polished", False) or full_tail_polished_fit.get("fixedGateDirectionValuePolished", False)),
                    paramRefit=bool(full_tail_polished_fit.get("paramRefit", False)),
                    paramRefitExtraGradientCount=int(param_refit_extra_gradient_evaluated),
                    holdoutPools=int(full_tail_holdout_evaluated),
                    holdoutTailSelected=int(full_tail_holdout_selected),
                    extraRandomTrainCount=int(full_tail_extra_random_evaluated),
                    radialTrainCount=int(full_tail_radial_train_evaluated),
                    radialSelectionCount=int(full_tail_radial_selection_evaluated),
                    radialGradientCount=int(full_tail_radial_gradient_evaluated),
                    selectionHoldoutPools=int(full_tail_selection_evaluated),
                    selectionHoldoutTailSelected=int(full_tail_selection_selected),
                    selectionHoldoutFullCount=int(full_tail_selection_full_count),
                    probeMae=float(full_tail_probe_metrics["mae"]),
                    probeWorstError=float(full_tail_probe_metrics["worstError"]),
                    baselineRevalidationMae=None if full_tail_revalidation_baseline_metrics is None else float(full_tail_revalidation_baseline_metrics["mae"]),
                    baselineRevalidationWorstError=None if full_tail_revalidation_baseline_metrics is None else float(full_tail_revalidation_baseline_metrics["worstError"]),
                    revalidationMae=None if full_tail_revalidation_metrics is None else float(full_tail_revalidation_metrics["mae"]),
                    revalidationWorstError=None if full_tail_revalidation_metrics is None else float(full_tail_revalidation_metrics["worstError"]),
                    rawOracleCalls=int(oracle.query_count),
                )
                # A fit can be excellent on output and wrong about structure, and no amount of
                # further optimisation finds that out, because the surface loss has nothing to
                # say about it. Measured on the randomized LLaMA layer-15 block: two source
                # units 0.0095 apart in gate direction were recovered as two copies of their
                # bisector - the projections of atoms 12 and 14 onto the sources' span have
                # cosine 0.999838, both at 41 degrees - while R2 held at 0.9986.
                #
                # A gate direction is not an opinion: it is where the function's second
                # derivative breaks. Probing for that break is an oracle-only measurement of
                # the thing the fit is guessing at, so the recovered gates are checked against
                # the boundaries they claim, and replaced by the boundaries actually found.
                # This runs after any full-tail method rather than instead of one, because the
                # block refit does improve the fourteen atoms it does resolve.
                if (
                    os.environ.get("GCL_BLIND_SWIGLU_POST_TAIL_ASYMPTOTIC_STITCH", "").strip() == "1"
                    and validation_full_tail_polish_enabled
                    and full_tail_accepted
                    and full_tail_method not in {"asymptotic-gate-stitch", "asymptotic-cold-rebuild"}
                ):
                    post_stitch_line_count = int(os.environ.get(
                        "GCL_BLIND_SWIGLU_POST_TAIL_STITCH_LINE_COUNT", "16") or "16")
                    post_stitch_samples = int(os.environ.get(
                        "GCL_BLIND_SWIGLU_POST_TAIL_STITCH_SAMPLES_PER_LINE", "1025") or "1025")
                    post_stitch_range = float(os.environ.get(
                        "GCL_BLIND_SWIGLU_POST_TAIL_STITCH_LINE_RANGE", "4.0") or "4.0")
                    post_stitch_offset = float(os.environ.get(
                        "GCL_BLIND_SWIGLU_POST_TAIL_STITCH_OFFSET_SCALE", "0.75") or "0.75")
                    post_stitch_scale = float(os.environ.get(
                        "GCL_BLIND_SWIGLU_POST_TAIL_STITCH_SCALE", "512.0") or "512.0")
                    post_stitch_bound = asymptotic_stitch_query_upper_bound(
                        post_stitch_line_count,
                        post_stitch_samples,
                        args.atom_count,
                        args.active_rank,
                        int(os.environ.get("GCL_BLIND_SWIGLU_ASYMPTOTIC_STITCH_HESSIAN_REPRESENTATIVES", "5") or "5"),
                    )
                    post_stitch_ran = False
                    post_stitch_accepted = False
                    post_stitch_selection_before = None
                    post_stitch_selection_after = None
                    if post_stitch_bound <= args.max_oracle_calls - oracle.query_count:
                        post_stitch_ran = True
                        stitched_fit = asymptotic_gate_stitch_swiglu_candidate(
                            oracle,
                            basis,
                            full_tail_polish_coords,
                            full_tail_polish_targets,
                            full_tail_polished_fit,
                            args.seed + 74231,
                            selection_coords=full_tail_selection_coords,
                            selection_targets=full_tail_selection_targets,
                            gradient_coords=fit_gradient_coords,
                            gradient_targets=fit_gradient_targets,
                            cubic_directions=cubic_gate_directions,
                            cubic_targets=cubic_scores,
                            line_count=post_stitch_line_count,
                            samples_per_line=post_stitch_samples,
                            line_range=post_stitch_range,
                            offset_scale=post_stitch_offset,
                            asymptotic_scale=post_stitch_scale,
                            cold_start=False,
                        )
                        # Decided on the selection pool, never on the draw that will be
                        # published, and kept only if it does not cost output quality.
                        if full_tail_selection_coords is not None and full_tail_selection_targets is not None:
                            before = regression_metrics(
                                full_tail_selection_targets,
                                predict_swiglu(full_tail_selection_coords, full_tail_polished_fit),
                            )
                            after = regression_metrics(
                                full_tail_selection_targets,
                                predict_swiglu(full_tail_selection_coords, stitched_fit),
                            )
                            post_stitch_selection_before = float(before["mae"]) + 0.35 * float(before["worstError"])
                            post_stitch_selection_after = float(after["mae"]) + 0.35 * float(after["worstError"])
                            post_stitch_accepted = post_stitch_selection_after <= post_stitch_selection_before
                        if post_stitch_accepted:
                            full_tail_polished_fit = stitched_fit
                            full_tail_probe_prediction = predict_swiglu(validation_coords, full_tail_polished_fit)
                            full_tail_probe_metrics = regression_metrics(validation_targets, full_tail_probe_prediction)
                    fit_attempts[-1].update({
                        "postTailAsymptoticStitchRan": bool(post_stitch_ran),
                        "postTailAsymptoticStitchAccepted": bool(post_stitch_accepted),
                        "postTailAsymptoticStitchQueryBound": int(post_stitch_bound),
                        "postTailAsymptoticStitchSelectionScoreBefore": post_stitch_selection_before,
                        "postTailAsymptoticStitchSelectionScoreAfter": post_stitch_selection_after,
                        "rawOracleCalls": int(oracle.query_count),
                    })
                if full_tail_accepted and full_tail_revalidation_metrics is not None:
                    # This is the stage the CEGIS stopping rule feeds, so it is the one where
                    # selecting on the reported draw bites hardest: up to cegis_limit draws,
                    # and the first that clears the thresholds becomes the published figure.
                    full_tail_reporting_metrics = independent_reporting_metrics(
                        oracle,
                        args,
                        basis,
                        dimension,
                        full_tail_polished_fit,
                        seed_offset=70628,
                        stage="validation-full-tail-polish-reporting",
                    )
                    if full_tail_reporting_metrics is None:
                        full_tail_accepted = False
                        fit_attempts[-1]["accepted"] = False
                        fit_attempts[-1]["declinedForUnbiasedReportingBudget"] = True
                    else:
                        fit = full_tail_polished_fit
                        metrics = full_tail_reporting_metrics
                        fit_attempts[-1].update({
                            "selectionR2": float(full_tail_revalidation_metrics["r2"]),
                            "selectionMae": float(full_tail_revalidation_metrics["mae"]),
                            "selectionWorstError": float(full_tail_revalidation_metrics["worstError"]),
                            "reportedR2": float(full_tail_reporting_metrics["r2"]),
                            "reportedMae": float(full_tail_reporting_metrics["mae"]),
                            "reportedWorstError": float(full_tail_reporting_metrics["worstError"]),
                            "selectionAndReportingSeparated": True,
                            "rawOracleCalls": int(oracle.query_count),
                        })
                    validation_inputs = full_tail_validation_inputs
                    validation_targets = full_tail_validation_targets
                    validation_coords = full_tail_validation_coords
                    validation_prediction = full_tail_validation_prediction
                    final_validation_source = "validation-full-tail-polish-fresh-validation"

            validation_affine_chebyshev_enabled = os.environ.get("GCL_BLIND_SWIGLU_VALIDATION_AFFINE_CHEBYSHEV", "").strip() == "1"
            if (
                not passes_metric_thresholds(metrics, args.min_r2, args.max_mae, args.max_worst_error)
                and validation_affine_chebyshev_enabled
                and oracle.query_count < args.max_oracle_calls
            ):
                affine_fit = affine_chebyshev_calibrate_swiglu_candidate(
                    validation_coords,
                    validation_targets,
                    fit,
                )
                affine_probe_prediction = predict_swiglu(validation_coords, affine_fit)
                affine_probe_metrics = regression_metrics(validation_targets, affine_probe_prediction)
                affine_revalidation_metrics: dict[str, float] | None = None
                affine_accepted = False
                if oracle.query_count + args.validation_count <= args.max_oracle_calls:
                    affine_validation_rng = np.random.default_rng(args.seed + 70647)
                    affine_validation_inputs = affine_validation_rng.normal(
                        0.0,
                        args.validation_scale,
                        size=(args.validation_count, dimension),
                    ).astype(np.float32)
                    affine_validation_targets = oracle.eval(affine_validation_inputs)
                    affine_validation_coords = affine_validation_inputs @ basis.T
                    affine_validation_prediction = predict_swiglu(affine_validation_coords, affine_fit)
                    affine_revalidation_metrics = regression_metrics(affine_validation_targets, affine_validation_prediction)
                    affine_changed = bool(affine_fit.get("affineChebyshevCalibrated", False))
                    affine_accepted = affine_changed and (
                        passes_metric_thresholds(affine_revalidation_metrics, args.min_r2, args.max_mae, args.max_worst_error)
                        or (
                            affine_revalidation_metrics["mae"] <= metrics["mae"]
                            and affine_revalidation_metrics["worstError"] < metrics["worstError"]
                        )
                    )
                fit_attempts.append({
                    "stage": "validation-affine-chebyshev",
                    "affineChebyshevCalibrated": bool(affine_fit.get("affineChebyshevCalibrated", False)),
                    "probeValidationR2": float(affine_probe_metrics["r2"]),
                    "probeValidationMae": float(affine_probe_metrics["mae"]),
                    "probeValidationWorstError": float(affine_probe_metrics["worstError"]),
                    "revalidationPerformed": affine_revalidation_metrics is not None,
                    "revalidationR2": None if affine_revalidation_metrics is None else float(affine_revalidation_metrics["r2"]),
                    "revalidationMae": None if affine_revalidation_metrics is None else float(affine_revalidation_metrics["mae"]),
                    "revalidationWorstError": None if affine_revalidation_metrics is None else float(affine_revalidation_metrics["worstError"]),
                    "accepted": bool(affine_accepted),
                    "rawOracleCalls": int(oracle.query_count),
                })
                log_progress(
                    args,
                    "validation-affine-chebyshev-complete",
                    accepted=bool(affine_accepted),
                    calibrated=bool(affine_fit.get("affineChebyshevCalibrated", False)),
                    probeMae=float(affine_probe_metrics["mae"]),
                    probeWorstError=float(affine_probe_metrics["worstError"]),
                    revalidationMae=None if affine_revalidation_metrics is None else float(affine_revalidation_metrics["mae"]),
                    revalidationWorstError=None if affine_revalidation_metrics is None else float(affine_revalidation_metrics["worstError"]),
                    rawOracleCalls=int(oracle.query_count),
                )
                if affine_accepted and affine_revalidation_metrics is not None:
                    affine_reporting_metrics = independent_reporting_metrics(
                        oracle,
                        args,
                        basis,
                        dimension,
                        affine_fit,
                        seed_offset=70728,
                        stage="validation-affine-readout-reporting",
                    )
                    if affine_reporting_metrics is None:
                        affine_accepted = False
                        fit_attempts[-1]["accepted"] = False
                        fit_attempts[-1]["declinedForUnbiasedReportingBudget"] = True
                    else:
                        fit = affine_fit
                        metrics = affine_reporting_metrics
                        fit_attempts[-1].update({
                            "selectionR2": float(affine_revalidation_metrics["r2"]),
                            "selectionMae": float(affine_revalidation_metrics["mae"]),
                            "selectionWorstError": float(affine_revalidation_metrics["worstError"]),
                            "reportedR2": float(affine_reporting_metrics["r2"]),
                            "reportedMae": float(affine_reporting_metrics["mae"]),
                            "reportedWorstError": float(affine_reporting_metrics["worstError"]),
                            "selectionAndReportingSeparated": True,
                            "rawOracleCalls": int(oracle.query_count),
                        })
                    validation_inputs = affine_validation_inputs
                    validation_targets = affine_validation_targets
                    validation_coords = affine_validation_coords
                    validation_prediction = affine_validation_prediction
                    final_validation_source = "validation-affine-chebyshev-fresh-validation"

            validation_chebyshev_enabled = os.environ.get("GCL_BLIND_SWIGLU_VALIDATION_CHEBYSHEV_CALIBRATION", "").strip() == "1"
            if (
                not passes_metric_thresholds(metrics, args.min_r2, args.max_mae, args.max_worst_error)
                and validation_chebyshev_enabled
                and oracle.query_count < args.max_oracle_calls
            ):
                chebyshev_fit = chebyshev_calibrate_swiglu_candidate(
                    validation_coords,
                    validation_targets,
                    fit,
                )
                chebyshev_probe_prediction = predict_swiglu(validation_coords, chebyshev_fit)
                chebyshev_probe_metrics = regression_metrics(validation_targets, chebyshev_probe_prediction)
                chebyshev_revalidation_metrics: dict[str, float] | None = None
                chebyshev_accepted = False
                if oracle.query_count + args.validation_count <= args.max_oracle_calls:
                    chebyshev_validation_rng = np.random.default_rng(args.seed + 70661)
                    chebyshev_validation_inputs = chebyshev_validation_rng.normal(
                        0.0,
                        args.validation_scale,
                        size=(args.validation_count, dimension),
                    ).astype(np.float32)
                    chebyshev_validation_targets = oracle.eval(chebyshev_validation_inputs)
                    chebyshev_validation_coords = chebyshev_validation_inputs @ basis.T
                    chebyshev_validation_prediction = predict_swiglu(chebyshev_validation_coords, chebyshev_fit)
                    chebyshev_revalidation_metrics = regression_metrics(chebyshev_validation_targets, chebyshev_validation_prediction)
                    chebyshev_changed = bool(chebyshev_fit.get("chebyshevCalibrated", False))
                    chebyshev_accepted = chebyshev_changed and (
                        passes_metric_thresholds(chebyshev_revalidation_metrics, args.min_r2, args.max_mae, args.max_worst_error)
                        or (
                            chebyshev_revalidation_metrics["mae"] <= metrics["mae"]
                            and chebyshev_revalidation_metrics["worstError"] < metrics["worstError"]
                        )
                    )
                fit_attempts.append({
                    "stage": "validation-chebyshev-calibration",
                    "chebyshevCalibrated": bool(chebyshev_fit.get("chebyshevCalibrated", False)),
                    "probeValidationR2": float(chebyshev_probe_metrics["r2"]),
                    "probeValidationMae": float(chebyshev_probe_metrics["mae"]),
                    "probeValidationWorstError": float(chebyshev_probe_metrics["worstError"]),
                    "revalidationPerformed": chebyshev_revalidation_metrics is not None,
                    "revalidationR2": None if chebyshev_revalidation_metrics is None else float(chebyshev_revalidation_metrics["r2"]),
                    "revalidationMae": None if chebyshev_revalidation_metrics is None else float(chebyshev_revalidation_metrics["mae"]),
                    "revalidationWorstError": None if chebyshev_revalidation_metrics is None else float(chebyshev_revalidation_metrics["worstError"]),
                    "accepted": bool(chebyshev_accepted),
                    "rawOracleCalls": int(oracle.query_count),
                })
                log_progress(
                    args,
                    "validation-chebyshev-calibration-complete",
                    accepted=bool(chebyshev_accepted),
                    calibrated=bool(chebyshev_fit.get("chebyshevCalibrated", False)),
                    probeMae=float(chebyshev_probe_metrics["mae"]),
                    probeWorstError=float(chebyshev_probe_metrics["worstError"]),
                    revalidationMae=None if chebyshev_revalidation_metrics is None else float(chebyshev_revalidation_metrics["mae"]),
                    revalidationWorstError=None if chebyshev_revalidation_metrics is None else float(chebyshev_revalidation_metrics["worstError"]),
                    rawOracleCalls=int(oracle.query_count),
                )
                if chebyshev_accepted and chebyshev_revalidation_metrics is not None:
                    fit = chebyshev_fit
                    metrics = chebyshev_revalidation_metrics
                    validation_inputs = chebyshev_validation_inputs
                    validation_targets = chebyshev_validation_targets
                    validation_coords = chebyshev_validation_coords
                    validation_prediction = chebyshev_validation_prediction
                    final_validation_source = "validation-chebyshev-calibration-fresh-validation"

            validation_residual_repair_enabled = os.environ.get("GCL_BLIND_SWIGLU_VALIDATION_RESIDUAL_REPAIR", "").strip() == "1"
            if (
                not passes_metric_thresholds(metrics, args.min_r2, args.max_mae, args.max_worst_error)
                and validation_residual_repair_enabled
                and args.atom_count >= 8
                and oracle.query_count < args.max_oracle_calls
            ):
                log_progress(args, "validation-residual-repair-start", rawOracleCalls=int(oracle.query_count))
                repair_rng = np.random.default_rng(args.seed + 70321)
                repair_count = int(os.environ.get("GCL_BLIND_SWIGLU_VALIDATION_RESIDUAL_REPAIR_COUNT", "192") or "192")
                repair_repeat = int(os.environ.get("GCL_BLIND_SWIGLU_VALIDATION_RESIDUAL_REPAIR_REPEAT", "8") or "8")
                repair_jitter_count = int(os.environ.get("GCL_BLIND_SWIGLU_VALIDATION_RESIDUAL_REPAIR_JITTER_COUNT", "0") or "0")
                repair_jitter_scale = float(os.environ.get("GCL_BLIND_SWIGLU_VALIDATION_RESIDUAL_REPAIR_JITTER_SCALE", "0.018") or "0.018")
                repair_train_limit = int(os.environ.get("GCL_BLIND_SWIGLU_VALIDATION_RESIDUAL_REPAIR_TRAIN_LIMIT", "12000") or "12000")
                hard_inputs, hard_targets, jitter_inputs, repair_selected, repair_jittered = validation_counterexample_inputs(
                    repair_rng,
                    validation_inputs,
                    validation_coords,
                    validation_targets,
                    validation_prediction,
                    basis,
                    repair_count,
                    repair_repeat,
                    repair_jitter_count,
                    repair_jitter_scale,
                )
                repair_inputs = hard_inputs
                repair_targets = hard_targets
                if jitter_inputs.shape[0] > 0:
                    jitter_targets = oracle.eval(jitter_inputs)
                    repair_inputs = np.vstack([repair_inputs, jitter_inputs])
                    repair_targets = np.concatenate([repair_targets, jitter_targets])
                if repair_inputs.shape[0] > 0:
                    if 0 < repair_train_limit < train_coords.shape[0]:
                        repair_train_indices = np.sort(repair_rng.choice(train_coords.shape[0], size=repair_train_limit, replace=False))
                        repair_base_coords = train_coords[repair_train_indices]
                        repair_base_targets = train_targets[repair_train_indices]
                    else:
                        repair_base_coords = train_coords
                        repair_base_targets = train_targets
                    repair_coords = np.vstack([repair_base_coords, repair_inputs @ basis.T])
                    repair_targets_all = np.concatenate([repair_base_targets, repair_targets])
                    previous_cycles = os.environ.get("GCL_BLIND_SWIGLU_RESIDUAL_REPLACE_CYCLES")
                    if previous_cycles is None:
                        os.environ["GCL_BLIND_SWIGLU_RESIDUAL_REPLACE_CYCLES"] = "1"
                    try:
                        repaired_fit = residual_region_atom_replace_swiglu_candidate(
                            repair_coords,
                            repair_targets_all,
                            fit,
                        )
                    finally:
                        if previous_cycles is None:
                            os.environ.pop("GCL_BLIND_SWIGLU_RESIDUAL_REPLACE_CYCLES", None)
                    repaired_fit = robust_scalar_calibrate_swiglu_candidate(
                        repair_coords,
                        repair_targets_all,
                        repaired_fit,
                    )
                    repair_probe_prediction = predict_swiglu(validation_coords, repaired_fit)
                    repair_probe_metrics = regression_metrics(validation_targets, repair_probe_prediction)
                    repair_revalidation_metrics: dict[str, float] | None = None
                    repair_accepted = False
                    if oracle.query_count + args.validation_count <= args.max_oracle_calls:
                        repair_validation_rng = np.random.default_rng(args.seed + 70339)
                        repair_validation_inputs = repair_validation_rng.normal(
                            0.0,
                            args.validation_scale,
                            size=(args.validation_count, dimension),
                        ).astype(np.float32)
                        repair_validation_targets = oracle.eval(repair_validation_inputs)
                        repair_validation_coords = repair_validation_inputs @ basis.T
                        repair_validation_prediction = predict_swiglu(repair_validation_coords, repaired_fit)
                        repair_revalidation_metrics = regression_metrics(repair_validation_targets, repair_validation_prediction)
                        repair_accepted = (
                            passes_metric_thresholds(repair_revalidation_metrics, args.min_r2, args.max_mae, args.max_worst_error)
                            or (
                                repair_revalidation_metrics["mae"] <= metrics["mae"]
                                and repair_revalidation_metrics["worstError"] < metrics["worstError"]
                            )
                        )
                    fit_attempts.append({
                        "stage": "validation-residual-repair",
                        "trainCount": int(repair_coords.shape[0]),
                        "baseTrainCount": int(repair_base_coords.shape[0]),
                        "counterexampleSelected": int(repair_selected),
                        "counterexampleRepeat": int(repair_repeat),
                        "counterexampleJittered": int(repair_jittered),
                        "residualRegionAtomReplaced": bool(repaired_fit.get("residualRegionAtomReplaced", False)),
                        "residualRegionReplacementCount": int(repaired_fit.get("residualRegionReplacementCount", 0)),
                        "probeValidationR2": float(repair_probe_metrics["r2"]),
                        "probeValidationMae": float(repair_probe_metrics["mae"]),
                        "probeValidationWorstError": float(repair_probe_metrics["worstError"]),
                        "revalidationPerformed": repair_revalidation_metrics is not None,
                        "revalidationR2": None if repair_revalidation_metrics is None else float(repair_revalidation_metrics["r2"]),
                        "revalidationMae": None if repair_revalidation_metrics is None else float(repair_revalidation_metrics["mae"]),
                        "revalidationWorstError": None if repair_revalidation_metrics is None else float(repair_revalidation_metrics["worstError"]),
                        "accepted": bool(repair_accepted),
                        "rawOracleCalls": int(oracle.query_count),
                    })
                    log_progress(
                        args,
                        "validation-residual-repair-complete",
                        accepted=bool(repair_accepted),
                        replaced=bool(repaired_fit.get("residualRegionAtomReplaced", False)),
                        probeMae=float(repair_probe_metrics["mae"]),
                        probeWorstError=float(repair_probe_metrics["worstError"]),
                        revalidationMae=None if repair_revalidation_metrics is None else float(repair_revalidation_metrics["mae"]),
                        revalidationWorstError=None if repair_revalidation_metrics is None else float(repair_revalidation_metrics["worstError"]),
                        rawOracleCalls=int(oracle.query_count),
                    )
                    if repair_accepted and repair_revalidation_metrics is not None:
                        fit = repaired_fit
                        metrics = repair_revalidation_metrics
                        validation_inputs = repair_validation_inputs
                        validation_targets = repair_validation_targets
                        validation_coords = repair_validation_coords
                        validation_prediction = repair_validation_prediction
                        final_validation_source = "validation-residual-repair-fresh-validation"

            tail_scalar_rescue_enabled = os.environ.get("GCL_BLIND_SWIGLU_TAIL_SCALAR_RESCUE", "1").strip() != "0"
            if (
                not passes_metric_thresholds(metrics, args.min_r2, args.max_mae, args.max_worst_error)
                and tail_scalar_rescue_enabled
                and args.counterexample_count > 0
                and args.atom_count >= 8
                and oracle.query_count < args.max_oracle_calls
            ):
                log_progress(args, "tail-scalar-rescue-start", rawOracleCalls=int(oracle.query_count))
                scalar_rescue_rng = np.random.default_rng(args.seed + 65003)
                hard_inputs, hard_targets, jitter_inputs, scalar_selected, scalar_jittered = validation_counterexample_inputs(
                    scalar_rescue_rng,
                    validation_inputs,
                    validation_coords,
                    validation_targets,
                    validation_prediction,
                    basis,
                    args.counterexample_count,
                    args.counterexample_repeat,
                    args.counterexample_jitter_count,
                    args.counterexample_jitter_scale,
                )
                scalar_cex_inputs = hard_inputs
                scalar_cex_targets = hard_targets
                if jitter_inputs.shape[0] > 0:
                    jitter_targets = oracle.eval(jitter_inputs)
                    scalar_cex_inputs = np.vstack([scalar_cex_inputs, jitter_inputs])
                    scalar_cex_targets = np.concatenate([scalar_cex_targets, jitter_targets])
                scalar_revalidation_metrics: dict[str, float] | None = None
                scalar_accepted = False
                scalar_fit = fit
                scalar_train_limit = int(os.environ.get("GCL_BLIND_SWIGLU_TAIL_SCALAR_TRAIN_LIMIT", "12000") or "12000")
                if scalar_cex_inputs.shape[0] > 0:
                    if scalar_train_limit > 0 and train_coords.shape[0] > scalar_train_limit:
                        scalar_train_indices = np.sort(scalar_rescue_rng.choice(train_coords.shape[0], size=scalar_train_limit, replace=False))
                        scalar_base_coords = train_coords[scalar_train_indices]
                        scalar_base_targets = train_targets[scalar_train_indices]
                    else:
                        scalar_base_coords = train_coords
                        scalar_base_targets = train_targets
                    scalar_cex_coords = scalar_cex_inputs @ basis.T
                    scalar_coords = np.vstack([scalar_base_coords, scalar_cex_coords])
                    scalar_targets = np.concatenate([scalar_base_targets, scalar_cex_targets])
                    tail_local_steps = int(os.environ.get("GCL_BLIND_SWIGLU_TAIL_LOCAL_RESCUE_STEPS", "900") or "0")
                    if tail_local_steps > 0:
                        scalar_fit = tail_error_polish_swiglu_candidate(
                            scalar_coords,
                            scalar_targets,
                            scalar_fit,
                            tail_local_steps,
                        )
                    scalar_fit = chebyshev_calibrate_swiglu_candidate(
                        scalar_coords,
                        scalar_targets,
                        scalar_fit,
                    )
                    scalar_fit = robust_scalar_calibrate_swiglu_candidate(
                        scalar_coords,
                        scalar_targets,
                        scalar_fit,
                    )
                    scalar_probe_prediction = predict_swiglu(validation_coords, scalar_fit)
                    scalar_probe_metrics = regression_metrics(validation_targets, scalar_probe_prediction)
                    if oracle.query_count + args.validation_count <= args.max_oracle_calls:
                        scalar_validation_rng = np.random.default_rng(args.seed + 65021)
                        scalar_validation_inputs = scalar_validation_rng.normal(
                            0.0,
                            args.validation_scale,
                            size=(args.validation_count, dimension),
                        ).astype(np.float32)
                        scalar_validation_targets = oracle.eval(scalar_validation_inputs)
                        scalar_validation_coords = scalar_validation_inputs @ basis.T
                        scalar_validation_prediction = predict_swiglu(scalar_validation_coords, scalar_fit)
                        scalar_revalidation_metrics = regression_metrics(scalar_validation_targets, scalar_validation_prediction)
                        scalar_accepted = (
                            passes_metric_thresholds(scalar_revalidation_metrics, args.min_r2, args.max_mae, args.max_worst_error)
                            or (
                                scalar_revalidation_metrics["mae"] <= metrics["mae"]
                                and scalar_revalidation_metrics["worstError"] < metrics["worstError"]
                            )
                        )
                        log_progress(
                            args,
                            "tail-scalar-rescue-revalidation-complete",
                            r2=float(scalar_revalidation_metrics["r2"]),
                            mae=float(scalar_revalidation_metrics["mae"]),
                            worstError=float(scalar_revalidation_metrics["worstError"]),
                            accepted=bool(scalar_accepted),
                            rawOracleCalls=int(oracle.query_count),
                        )
                    fit_attempts.append({
                        "stage": "tail-scalar-rescue",
                        "trainCount": int(scalar_coords.shape[0]),
                        "baseTrainCount": int(scalar_base_coords.shape[0]),
                        "counterexampleSelected": int(scalar_selected),
                        "counterexampleRepeat": int(args.counterexample_repeat),
                        "counterexampleJittered": int(scalar_jittered),
                        "probeValidationR2": float(scalar_probe_metrics["r2"]),
                        "probeValidationMae": float(scalar_probe_metrics["mae"]),
                        "probeValidationWorstError": float(scalar_probe_metrics["worstError"]),
                        "revalidationPerformed": scalar_revalidation_metrics is not None,
                        "revalidationR2": None if scalar_revalidation_metrics is None else float(scalar_revalidation_metrics["r2"]),
                        "revalidationMae": None if scalar_revalidation_metrics is None else float(scalar_revalidation_metrics["mae"]),
                        "revalidationWorstError": None if scalar_revalidation_metrics is None else float(scalar_revalidation_metrics["worstError"]),
                        "tailErrorPolished": bool(scalar_fit.get("tailErrorPolished", False)),
                        "tailErrorPolishSteps": int(scalar_fit.get("tailErrorPolishSteps", 0)),
                        "chebyshevCalibrated": bool(scalar_fit.get("chebyshevCalibrated", False)),
                        "robustScalarCalibrated": bool(scalar_fit.get("robustScalarCalibrated", False)),
                        "accepted": bool(scalar_accepted),
                        "rawOracleCalls": int(oracle.query_count),
                    })
                    if scalar_accepted and scalar_revalidation_metrics is not None:
                        fit = scalar_fit
                        metrics = scalar_revalidation_metrics
                        validation_inputs = scalar_validation_inputs
                        validation_targets = scalar_validation_targets
                        validation_coords = scalar_validation_coords
                        validation_prediction = scalar_validation_prediction
                        final_validation_source = "tail-scalar-rescue-fresh-validation"

            if (
                not passes_metric_thresholds(metrics, args.min_r2, args.max_mae, args.max_worst_error)
                and not args.disable_adaptive_rescue
                and args.atom_count >= 8
                and oracle.query_count < args.max_oracle_calls
            ):
                log_progress(args, "adaptive-rescue-start", rawOracleCalls=int(oracle.query_count))
                rescue_train_default = max(args.train_count, args.atom_count * 2000)
                rescue_gradient_default = max(args.fit_gradient_count, args.atom_count * 32)
                rescue_steps_default = max(args.fit_steps, 5200 if args.atom_count >= 16 else 4200)
                rescue_restarts_default = max(args.fit_restarts if args.fit_restarts > 0 else 0, 1024 if args.atom_count >= 16 else 768)
                rescue_train_target = int(os.environ.get(
                    "GCL_BLIND_SWIGLU_RESCUE_TRAIN_TARGET",
                    str(args.rescue_train_target if args.rescue_train_target > 0 else rescue_train_default),
                ) or rescue_train_default)
                rescue_gradient_target = int(os.environ.get(
                    "GCL_BLIND_SWIGLU_RESCUE_GRADIENT_TARGET",
                    str(args.rescue_gradient_target if args.rescue_gradient_target > 0 else rescue_gradient_default),
                ) or rescue_gradient_default)
                rescue_steps = int(os.environ.get(
                    "GCL_BLIND_SWIGLU_RESCUE_STEPS",
                    str(args.rescue_steps if args.rescue_steps > 0 else rescue_steps_default),
                ) or rescue_steps_default)
                rescue_restarts = int(os.environ.get(
                    "GCL_BLIND_SWIGLU_RESCUE_RESTARTS",
                    str(args.rescue_restarts if args.rescue_restarts > 0 else rescue_restarts_default),
                ) or rescue_restarts_default)
                rescue_train_target = max(args.train_count, rescue_train_target)
                rescue_gradient_target = max(args.fit_gradient_count, rescue_gradient_target)
                rescue_steps = max(1, rescue_steps)
                rescue_restarts = max(1, rescue_restarts)
                counterexample_selected = 0
                counterexample_jittered = 0
                counterexample_repeat = 0
                if args.counterexample_count > 0:
                    hard_inputs, hard_targets, jitter_inputs, counterexample_selected, counterexample_jittered = validation_counterexample_inputs(
                        train_rng,
                        validation_inputs,
                        validation_coords,
                        validation_targets,
                        validation_prediction,
                        basis,
                        args.counterexample_count,
                        args.counterexample_repeat,
                        args.counterexample_jitter_count,
                        args.counterexample_jitter_scale,
                    )
                    counterexample_repeat = max(1, args.counterexample_repeat)
                    cex_inputs = hard_inputs
                    cex_targets = hard_targets
                    if jitter_inputs.shape[0] > 0:
                        jitter_targets = oracle.eval(jitter_inputs)
                        cex_inputs = np.vstack([cex_inputs, jitter_inputs])
                        cex_targets = np.concatenate([cex_targets, jitter_targets])
                    if cex_inputs.shape[0] > 0:
                        train_inputs = np.vstack([train_inputs, cex_inputs])
                        train_targets = np.concatenate([train_targets, cex_targets])
                        train_coords = train_inputs @ basis.T
                        log_progress(
                            args,
                            "counterexamples-added",
                            selected=int(counterexample_selected),
                            repeat=int(counterexample_repeat),
                            jittered=int(counterexample_jittered),
                            trainCount=int(train_coords.shape[0]),
                            rawOracleCalls=int(oracle.query_count),
                        )
                extra_train = max(0, rescue_train_target - train_coords.shape[0])
                if extra_train > 0:
                    extra_inputs = sample_active_region_inputs(
                        train_rng,
                        extra_train,
                        basis,
                        args.input_scale,
                        train_scale_mixture,
                        train_sampling_mode,
                    )
                    extra_targets = oracle.eval(extra_inputs)
                    train_inputs = np.vstack([train_inputs, extra_inputs])
                    train_targets = np.concatenate([train_targets, extra_targets])
                    train_coords = train_inputs @ basis.T
                    log_progress(args, "rescue-train-expanded", trainCount=int(train_coords.shape[0]), rawOracleCalls=int(oracle.query_count))
                extra_gradient = max(0, rescue_gradient_target - fit_gradient_coords.shape[0])
                if extra_gradient > 0:
                    extra_coords, extra_targets = active_gradient_estimator(
                        oracle,
                        fit_gradient_rng,
                        basis,
                        extra_gradient,
                        args.input_scale,
                        fit_gradient_steps,
                    )
                    fit_gradient_coords = np.vstack([fit_gradient_coords, extra_coords])
                    fit_gradient_targets = np.vstack([fit_gradient_targets, extra_targets])
                    log_progress(args, "rescue-gradients-expanded", fitGradientCount=int(fit_gradient_coords.shape[0]), rawOracleCalls=int(oracle.query_count))
                log_progress(
                    args,
                    "adaptive-rescue-fit-start",
                    trainCount=int(train_coords.shape[0]),
                    fitGradientCount=int(fit_gradient_coords.shape[0]),
                    fitSteps=int(rescue_steps),
                    fitRestarts=int(rescue_restarts),
                    rawOracleCalls=int(oracle.query_count),
                )
                rescue_fit = fit_swiglu_sum(
                    train_coords,
                    train_targets,
                    args.atom_count,
                    args.seed + 7919,
                    rescue_steps,
                    args.fit_learning_rate,
                    rescue_restarts,
                    fit_gradient_coords,
                    fit_gradient_targets,
                    cubic_gate_directions,
                    args.fit_gradient_weight,
                    args.fit_mode,
                    args.fit_restart_chunk_size,
                    fit_selection_coords,
                    fit_selection_targets,
                    cubic_selection_directions=cubic_holdout_directions,
                    cubic_selection_scores=cubic_holdout_scores,
                    cubic_directions=cubic_gate_directions,
                    cubic_targets=cubic_scores,
                )
                rescue_prediction = predict_swiglu(validation_coords, rescue_fit)
                rescue_metrics = regression_metrics(validation_targets, rescue_prediction)
                log_progress(
                    args,
                    "adaptive-rescue-complete",
                    r2=float(rescue_metrics["r2"]),
                    mae=float(rescue_metrics["mae"]),
                    worstError=float(rescue_metrics["worstError"]),
                    rawOracleCalls=int(oracle.query_count),
                )
                fit_attempts.append({
                    "stage": "adaptive-rescue",
                    "trainCount": int(train_coords.shape[0]),
                    "fitGradientCount": int(fit_gradient_coords.shape[0]),
                    "fitGradientMode": args.fit_gradient_mode,
                    "fitSteps": int(rescue_steps),
                    "fitRestarts": int(rescue_fit["restartCount"]),
                    "fitRandomRestarts": int(rescue_fit.get("randomRestartCount", rescue_fit["restartCount"])),
                    "fitStructuredStarts": int(rescue_fit.get("structuredStartCount", 0)),
                    "selectedStartKind": str(rescue_fit.get("startKind", "unknown")),
                    "selectedRandomRestartIndex": rescue_fit.get("randomRestartIndex"),
                    "bestScoreByStartKind": rescue_fit.get("bestScoreByStartKind", {}),
                    "bestMetricsByStartKind": rescue_fit.get("bestMetricsByStartKind", {}),
                    "fitMode": str(rescue_fit.get("fitMode", "unknown")),
                    "fitTrainMae": float(rescue_fit["mae"]),
                    "fitSelectionScore": float(rescue_fit.get("selectionScore", rescue_fit.get("score", rescue_fit["mae"]))),
                    "extraGateCandidateCount": int(rescue_fit.get("extraGateCandidateCount", 0)),
                    "validationR2": float(rescue_metrics["r2"]),
                    "validationMae": float(rescue_metrics["mae"]),
                    "validationWorstError": float(rescue_metrics["worstError"]),
                    "counterexampleSelected": int(counterexample_selected),
                    "counterexampleRepeat": int(counterexample_repeat),
                    "counterexampleJittered": int(counterexample_jittered),
                    "rawOracleCalls": int(oracle.query_count),
                })
                if (
                    passes_metric_thresholds(rescue_metrics, args.min_r2, args.max_mae, args.max_worst_error)
                    or rescue_metrics["mae"] < metrics["mae"]
                    or rescue_metrics["worstError"] < metrics["worstError"]
                ):
                    revalidation_performed = False
                    revalidation_metrics: dict[str, float] | None = None
                    revalidation_raw_calls = oracle.query_count
                    if oracle.query_count + args.validation_count <= args.max_oracle_calls:
                        rescue_validation_rng = np.random.default_rng(args.seed + 66109)
                        rescue_validation_inputs = rescue_validation_rng.normal(
                            0.0,
                            args.validation_scale,
                            size=(args.validation_count, dimension),
                        ).astype(np.float32)
                        rescue_validation_targets = oracle.eval(rescue_validation_inputs)
                        rescue_validation_coords = rescue_validation_inputs @ basis.T
                        rescue_validation_prediction = predict_swiglu(rescue_validation_coords, rescue_fit)
                        revalidation_metrics = regression_metrics(rescue_validation_targets, rescue_validation_prediction)
                        revalidation_raw_calls = oracle.query_count
                        revalidation_performed = True
                        log_progress(
                            args,
                            "adaptive-rescue-revalidation-complete",
                            r2=float(revalidation_metrics["r2"]),
                            mae=float(revalidation_metrics["mae"]),
                            worstError=float(revalidation_metrics["worstError"]),
                            rawOracleCalls=int(oracle.query_count),
                        )
                    accept_rescue = False
                    if revalidation_metrics is not None:
                        accept_rescue = (
                            passes_metric_thresholds(revalidation_metrics, args.min_r2, args.max_mae, args.max_worst_error)
                            or (
                                revalidation_metrics["mae"] <= metrics["mae"]
                                and revalidation_metrics["worstError"] < metrics["worstError"]
                            )
                        )
                    else:
                        # Without a fresh validation witness, only accept rescue when it did
                        # not use validation counterexamples. Counterexample rescue must be
                        # revalidated before it can become the reported model.
                        accept_rescue = counterexample_selected == 0 and rescue_metrics["mae"] < metrics["mae"]
                    fit_attempts[-1].update({
                        "revalidationPerformed": bool(revalidation_performed),
                        "revalidationR2": None if revalidation_metrics is None else float(revalidation_metrics["r2"]),
                        "revalidationMae": None if revalidation_metrics is None else float(revalidation_metrics["mae"]),
                        "revalidationWorstError": None if revalidation_metrics is None else float(revalidation_metrics["worstError"]),
                        "revalidationRawOracleCalls": int(revalidation_raw_calls),
                        "accepted": bool(accept_rescue),
                    })
                    if accept_rescue:
                        fit = rescue_fit
                        if revalidation_metrics is not None:
                            metrics = revalidation_metrics
                            validation_inputs = rescue_validation_inputs
                            validation_targets = rescue_validation_targets
                            validation_coords = rescue_validation_coords
                            validation_prediction = rescue_validation_prediction
                            final_validation_source = "adaptive-rescue-fresh-validation"
                        else:
                            metrics = rescue_metrics
                            validation_prediction = rescue_prediction
                            final_validation_source = "adaptive-rescue-original-validation"

            final_fit_snapshot_output = os.environ.get("GCL_BLIND_SWIGLU_FINAL_FIT_SNAPSHOT_OUTPUT", "").strip()
            if final_fit_snapshot_output:
                save_fit_snapshot(
                    final_fit_snapshot_output,
                    basis=basis,
                    train_coords=train_coords,
                    train_targets=train_targets,
                    fit=fit,
                    dimension=dimension,
                    active_rank=args.active_rank,
                    atom_count=args.atom_count,
                    gradient_witness_count=gradient_witness_count,
                    eigenvalues=eigenvalues,
                    energy_ratio=energy_ratio,
                    fit_gradient_coords=fit_gradient_coords,
                    fit_gradient_targets=fit_gradient_targets,
                    cubic_directions=cubic_gate_directions,
                    cubic_targets=cubic_scores,
                    cubic_holdout_directions=cubic_holdout_directions,
                    cubic_holdout_targets=cubic_holdout_scores,
                    access_audit=snapshot_access_audit(args, oracle),
                )
                log_progress(
                    args,
                    "final-fit-snapshot-saved",
                    path=str(final_fit_snapshot_output),
                    finalValidationSource=final_validation_source,
                    r2=float(metrics["r2"]),
                    mae=float(metrics["mae"]),
                    worstError=float(metrics["worstError"]),
                    rawOracleCalls=int(oracle.query_count),
                )

            if args.defer_posthoc_truth:
                frozen_fit_path = Path(args.frozen_fit_output).resolve()
                save_fit_snapshot(
                    str(frozen_fit_path),
                    basis=basis,
                    train_coords=train_coords,
                    train_targets=train_targets,
                    fit=fit,
                    dimension=dimension,
                    active_rank=args.active_rank,
                    atom_count=args.atom_count,
                    gradient_witness_count=gradient_witness_count,
                    eigenvalues=eigenvalues,
                    energy_ratio=energy_ratio,
                    fit_gradient_coords=fit_gradient_coords,
                    fit_gradient_targets=fit_gradient_targets,
                    cubic_directions=cubic_gate_directions,
                    cubic_targets=cubic_scores,
                    cubic_holdout_directions=cubic_holdout_directions,
                    cubic_holdout_targets=cubic_holdout_scores,
                    access_audit=snapshot_access_audit(args, oracle),
                )
                forbidden_meta = sorted({
                    "model", "modelType", "layer", "unit", "units", "intermediateSize",
                    "device", "oracleMode", "outputChannel",
                }.intersection(oracle.meta))
                functional_pass = (
                    passes_metric_thresholds(metrics, args.min_r2, args.max_mae, args.max_worst_error)
                    and oracle.query_count <= args.max_oracle_calls
                    and args.atom_count == unit_count
                    and not forbidden_meta
                    and oracle.meta.get("protocolCapabilities") == {"eval": True, "truth": False}
                    and oracle.truth_request_count == 0
                )
                frozen_report = {
                    "schemaVersion": 1,
                    "kind": "blind-swiglu-frozen-recovery",
                    "summary": {
                        "pass": bool(functional_pass),
                        "dimension": int(dimension),
                        "unitCount": int(unit_count),
                        "activeRank": int(args.active_rank),
                        "atomCount": int(args.atom_count),
                        "rawOracleCalls": int(oracle.query_count),
                        "metrics": metrics,
                        "selectedGradientEnergyRatio": float(energy_ratio),
                    },
                    "frozenRecovery": {
                        "path": str(frozen_fit_path),
                        "sha256": sha256_file(frozen_fit_path),
                        "frozenAfterValidation": True,
                    },
                    "accessAudit": {
                        "solverAccess": "opaque-eval-only-oracle",
                        "externalOracleTransport": True,
                        "oracleTransport": "unix-binary-v1" if args.oracle_socket else "anonymous-jsonl-fd",
                        "weightsUsedBySolver": False,
                        "hooksUsedBySolver": False,
                        "sourceIdsUsedBySolver": False,
                        "teacherManifestFieldsObserved": forbidden_meta,
                        # Was a hardcoded False, so it asserted the invariant instead of
                        # reporting it. The oracle advertises its capabilities in the
                        # handshake, so read that: an oracle that ever did offer truth
                        # would now be visible here rather than papered over.
                        "truthCapabilityAvailable": bool(
                            dict(getattr(oracle, "meta", {}) or {}).get("protocolCapabilities", {}).get("truth", False)
                        ),
                        "truthRequestCount": int(oracle.truth_request_count),
                        "fitSnapshotUsedBySolver": False,
                        "nonlinearInitialFitUsed": False,
                        "queryTranscriptSha256": getattr(oracle, "transcript_sha256", None),
                    },
                    "solver": {
                        "kind": "blind-active-subspace-explicit-swiglu-sum",
                        "recoveryPath": str(args.recovery_path),
                        "fitMode": str(fit.get("fitMode", "unknown")),
                        "fitRestarts": int(fit.get("restartCount", 0)),
                        "fitRandomRestarts": int(fit.get("randomRestartCount", 0)),
                        "gradientMode": args.gradient_mode,
                        "gradientWitnessCount": int(gradient_witness_count),
                        "asymptoticStructure": {
                            "coldStart": bool(fit.get("asymptoticGateStitchColdStart", False)),
                            "stitched": bool(fit.get("asymptoticGateStitched", False)),
                            "clusterCount": int(fit.get("asymptoticGateStitchClusterCount", 0)),
                            "sourceFallbackAtoms": fit.get("asymptoticGateStitchFallbackAtoms", []),
                        },
                    },
                }
                output_json.write_text(dump_report_json(frozen_report) + "\n", encoding="utf-8")
                if output_md:
                    output_md.write_text(
                        "\n".join([
                            "# Blind SwiGLU Frozen Recovery",
                            "",
                            f"- pass: `{str(functional_pass).lower()}`",
                            f"- frozen SHA-256: `{frozen_report['frozenRecovery']['sha256']}`",
                            f"- raw oracle calls: `{oracle.query_count}`",
                            f"- validation R2: `{metrics['r2']}`",
                            f"- validation MAE: `{metrics['mae']}`",
                            # These three were string literals emitted on every run, so
                            # they asserted the isolation invariants rather than reporting
                            # them. They now interpolate the same audit values the JSON
                            # carries, two of which are measured from the oracle handshake
                            # and the solver's own truth-request counter.
                            f"- source ids observed by solver: `{str(bool(frozen_report['accessAudit']['sourceIdsUsedBySolver'])).lower()}`",
                            f"- truth capability available to solver: `{str(bool(frozen_report['accessAudit']['truthCapabilityAvailable'])).lower()}`",
                            f"- truth requests by solver: `{frozen_report['accessAudit']['truthRequestCount']}`",
                            "",
                        ]),
                        encoding="utf-8",
                    )
                print(json.dumps(frozen_report["summary"], indent=2))
                return 0 if functional_pass else 1

            oracle.unlock_posthoc_truth("final-validation-recovery-frozen")
            truth_gate, truth_up = oracle.truth()
            if truth_gate.ndim == 1:
                truth_gate = np.expand_dims(truth_gate, axis=0)
            if truth_up.ndim == 1:
                truth_up = np.expand_dims(truth_up, axis=0)
            truth_basis = np.vstack([normalize(row) for row in truth_gate] + [normalize(row) for row in truth_up])
            cosines = principal_cosines(basis, truth_basis)
            recovered_gates = np.vstack([normalize_recovered(row @ basis) for row in fit["gates"]])
            recovered_ups = np.vstack([normalize_recovered(row @ basis) for row in fit["ups"]])
            truth_gate_coords = truth_gate @ basis.T
            truth_up_coords = truth_up @ basis.T
            recovered_gate_coords = np.vstack([normalize_recovered(row) for row in np.asarray(fit["gates"], dtype=np.float64)])
            recovered_effective_ups = np.asarray(fit["ups"], dtype=np.float64) * np.asarray(
                fit["scales"],
                dtype=np.float64,
            )[:, None]
            recovered_up_coords = np.vstack([normalize_recovered(row) for row in recovered_effective_ups])
            normalized_truth_gate_coords = np.vstack([normalize(row) for row in truth_gate_coords])
            normalized_truth_up_coords = np.vstack([normalize(row) for row in truth_up_coords])
            paired_gate_cosines = recovered_gate_coords @ normalized_truth_gate_coords.T
            paired_up_cosines = recovered_up_coords @ normalized_truth_up_coords.T
            paired_joint_cosines = np.minimum(paired_gate_cosines, paired_up_cosines)
            atom_assignment = maximum_weight_assignment(paired_joint_cosines)
            source_units = oracle.meta.get("units", list(range(truth_gate.shape[0])))
            source_cancellation = audit_swiglu_source_cancellation(
                validation_coords,
                truth_gate_coords,
                truth_up_coords,
                [int(value) for value in source_units],
            )
            paired_rows = [
                {
                    "recoveredAtom": int(recovered_atom),
                    "sourceIndex": int(source_index),
                    "sourceUnit": int(source_units[source_index]),
                    "gateCosine": float(paired_gate_cosines[recovered_atom, source_index]),
                    "upCosine": float(paired_up_cosines[recovered_atom, source_index]),
                    "jointMinCosine": float(paired_joint_cosines[recovered_atom, source_index]),
                }
                for recovered_atom, source_index in enumerate(atom_assignment)
            ]
            paired_joint_values = np.asarray(
                [row["jointMinCosine"] for row in paired_rows],
                dtype=np.float64,
            )
            projected_source_prediction = np.sum(
                silu_np(validation_coords @ truth_gate_coords.T) * (validation_coords @ truth_up_coords.T),
                axis=1,
            )
            projected_source_metrics = regression_metrics(validation_targets, projected_source_prediction)
            source_alignment = {
                "gateMaxAbsCosine": float(max(abs(float(np.dot(recovered, truth))) for recovered in recovered_gates for truth in truth_basis)),
                "upMaxAbsCosine": float(max(abs(float(np.dot(recovered, truth))) for recovered in recovered_ups for truth in truth_basis)),
                "subspaceMinPrincipalCosine": float(np.min(cosines)),
                "subspaceCosines": [float(value) for value in cosines],
                "projectedSourceMae": float(projected_source_metrics["mae"]),
                "projectedSourceR2": float(projected_source_metrics["r2"]),
                "projectedSourceWorstError": float(projected_source_metrics["worstError"]),
                "pairedAtomAssignmentMethod": "hungarian-max-joint-signed-cosine",
                "pairedAtomMinJointCosine": float(np.min(paired_joint_values)),
                "pairedAtomMeanJointCosine": float(np.mean(paired_joint_values)),
                "pairedAtomCountAbove095": int(np.sum(paired_joint_values >= 0.95)),
                "pairedAtomCountAbove099": int(np.sum(paired_joint_values >= 0.99)),
                "pairedAtoms": paired_rows,
                "sourceCancellation": source_cancellation,
                "unpairedAtomDiagnostic": diagnose_unpaired_atoms(
                    paired_joint_cosines,
                    paired_gate_cosines,
                    paired_up_cosines,
                    atom_assignment,
                    [int(value) for value in source_units],
                    recovered_gate_coords,
                    normalized_truth_gate_coords,
                ),
            }
            if asymptotic_stitch_posthoc_directions is not None:
                normalized_asymptotic_directions = np.vstack([
                    normalize(row)
                    for row in asymptotic_stitch_posthoc_directions
                ])
                asymptotic_gate_cosines = np.abs(
                    normalized_asymptotic_directions
                    @ normalized_truth_gate_coords.T
                )
                asymptotic_assignment = maximum_weight_assignment(asymptotic_gate_cosines)
                asymptotic_paired_cosines = asymptotic_gate_cosines[
                    np.arange(args.atom_count),
                    asymptotic_assignment,
                ]
                source_alignment["asymptoticGateStitchPostHoc"] = {
                    "assignmentMethod": "hungarian-max-absolute-gate-cosine",
                    "minimumGateCosine": float(np.min(asymptotic_paired_cosines)),
                    "meanGateCosine": float(np.mean(asymptotic_paired_cosines)),
                    "countAbove095": int(np.sum(asymptotic_paired_cosines >= 0.95)),
                    "countAbove099": int(np.sum(asymptotic_paired_cosines >= 0.99)),
                    "pairedGateCosines": [float(value) for value in asymptotic_paired_cosines],
                }
                if asymptotic_stitch_posthoc_up_directions is not None:
                    normalized_asymptotic_up_directions = np.vstack([
                        normalize(row)
                        for row in asymptotic_stitch_posthoc_up_directions
                    ])
                    asymptotic_up_cosines = np.abs(
                        normalized_asymptotic_up_directions
                        @ normalized_truth_up_coords.T
                    )
                    paired_up_cosines = asymptotic_up_cosines[
                        np.arange(args.atom_count),
                        asymptotic_assignment,
                    ]
                    source_alignment["asymptoticGateStitchPostHoc"].update({
                        "minimumUpCosine": float(np.min(paired_up_cosines)),
                        "meanUpCosine": float(np.mean(paired_up_cosines)),
                        "upCountAbove095": int(np.sum(paired_up_cosines >= 0.95)),
                        "upCountAbove099": int(np.sum(paired_up_cosines >= 0.99)),
                        "pairedUpCosines": [float(value) for value in paired_up_cosines],
                    })

            # The sweep enforces the paired-atom identity bar on top of these thresholds.
            # While the solver was not told about it, a case could publish `pass: true` in
            # its own report and `status: fail` in the manifest written from the same run.
            # Applying it here makes the report's verdict the whole verdict.
            paired_atom_gate = paired_atom_identity_passes(
                source_alignment,
                args.min_paired_atom_cosine,
                args.atom_count,
            )
            pass_gate = (
                metrics["r2"] >= args.min_r2
                and metrics["mae"] <= args.max_mae
                and metrics["worstError"] <= args.max_worst_error
                and source_alignment["subspaceMinPrincipalCosine"] >= args.min_subspace_cosine
                and paired_atom_gate
                and oracle.query_count <= args.max_oracle_calls
                and args.atom_count == unit_count
                and oracle.meta.get("solverAccess") == "oracle-only"
            )
            report = {
                "summary": {
                    "pass": pass_gate,
                    "model": oracle.meta["model"],
                    "modelType": oracle.meta["modelType"],
                    "dimension": dimension,
                    "layer": oracle.meta["layer"],
                    "unit": oracle.meta["unit"],
                    "units": oracle.meta.get("units", [oracle.meta["unit"]]),
                    "unitCount": unit_count,
                    "oracleMode": oracle.meta.get("oracleMode", "selected-unit-sum"),
                    "outputChannel": oracle.meta.get("outputChannel"),
                    "activeRank": args.active_rank,
                    "atomCount": args.atom_count,
                    "rawOracleCalls": oracle.query_count,
                    "metrics": metrics,
                    "sourceAlignment": source_alignment,
                    "selectedGradientEnergyRatio": float(energy_ratio),
                    "pairedAtomThreshold": (
                        float(args.min_paired_atom_cosine)
                        if args.min_paired_atom_cosine > 0.0
                        else None
                    ),
                    "pairedAtomGate": bool(paired_atom_gate),
                },
                # This site already carried the unlock evidence and was the closer of the two;
                # it goes through the shared builder so that the claim, the unlock and the
                # honesty about which counters were compared stay attached to each other.
                "accessAudit": oracle_access_audit(
                    oracle,
                    sourceIdsUsedBySolver=False,
                    fitSnapshotUsedBySolver=bool(args.fit_snapshot_input),
                    nonlinearInitialFitUsed=bool(args.recovery_path != "asymptotic-cold-start"),
                    oracleOwnsWeights=True,
                    truthLockedUntilRecoveryFrozen=True,
                    processBoundary="solver parent communicates with oracle subprocess via npy batch files and JSONL commands",
                ),
                "solver": {
                    "kind": "blind-active-subspace-explicit-swiglu-sum",
                    "probePoints": args.probe_points,
                    "gradientMode": args.gradient_mode,
                    "basisMoment": basis_moment,
                    "directionCount": args.direction_count,
                    "gradientWitnessCount": gradient_witness_count,
                    "cubicDirectionCount": int(cubic_gate_directions.shape[0]),
                    "cubicStep": float(args.cubic_step),
                    "cubicSteps": [float(value) for value in cubic_steps],
                    "cubicHoldoutCount": int(cubic_holdout_directions.shape[0]),
                    "cubicMaxAbsScore": float(np.max(np.abs(cubic_scores))) if cubic_scores.size else 0.0,
                    "fitCubicWeight": float(fit.get("fitCubicWeight", 0.0)),
                    "selectionCubicWeight": float(fit.get("selectionCubicWeight", 0.0)),
                    "cubicRelativeRmse": (
                        None
                        if fit.get("cubicRelativeRmse") is None
                        else float(fit["cubicRelativeRmse"])
                    ),
                    "subspaceCount": args.subspace_count if args.subspace_count > 0 else None,
                    "finiteDifferenceStep": args.finite_difference_step,
                    "finiteDifferenceSteps": [float(step) for step in finite_difference_steps],
                    "independentRng": bool(independent_rng),
                    "trainScaleMixture": [float(value) for value in train_scale_mixture] if train_scale_mixture else None,
                    "trainSamplingMode": train_sampling_mode,
                    "trainCount": args.train_count,
                    "validationCount": args.validation_count,
                    "validationScale": args.validation_scale,
                    "finalValidationSource": final_validation_source,
                    "gradientEigenvalues": [float(value) for value in eigenvalues],
                    "fitTrainMae": float(fit["mae"]),
                    "fitRestarts": int(fit["restartCount"]),
                    "fitRandomRestarts": int(fit.get("randomRestartCount", fit["restartCount"])),
                    "fitStructuredStarts": int(fit.get("structuredStartCount", 0)),
                    "selectedStartKind": str(fit.get("startKind", "unknown")),
                    "selectedRandomRestartIndex": fit.get("randomRestartIndex"),
                    "bestScoreByStartKind": fit.get("bestScoreByStartKind", {}),
                    "bestMetricsByStartKind": fit.get("bestMetricsByStartKind", {}),
                    "fitRestartChunkSize": None if fit.get("restartChunkSize") is None else int(fit["restartChunkSize"]),
                    "fitMode": str(fit.get("fitMode", "unknown")),
                    "recoveryPath": str(args.recovery_path),
                    "inputScale": float(args.input_scale),
                    "autoInputScale": bool(args.auto_input_scale),
                    "inputScaleCalibration": input_scale_calibration,
                    "nonlinearInitialFitUsed": bool(args.recovery_path != "asymptotic-cold-start"),
                    "randomInitScales": fit.get("randomInitScales", [0.6]),
                    "fitGradientCount": int(fit_gradient_coords.shape[0]),
                    "fitGradientMode": args.fit_gradient_mode,
                    "fitGradientSteps": [float(step) for step in fit_gradient_steps],
                    "fitGradientWeight": float(args.fit_gradient_weight),
                    "selectionTailWeight": float(fit.get("selectionTailWeight", 0.0)),
                    "selectionTailQuantile": float(fit.get("selectionTailQuantile", 0.995)),
                    "selectionWorstWeight": float(fit.get("selectionWorstWeight", 0.0)),
                    "fitSelectionCount": int(fit_selection_count),
                    "fitSelectionScale": float(fit_selection_scale),
                    "fitExternalSelectionCount": int(fit.get("externalSelectionCount", 0)),
                    "dictionaryCandidateRankScore": fit.get("dictionaryCandidateRankScore"),
                    "forcedPolishCandidateCount": int(fit.get("forcedPolishCandidateCount", 0)),
                    "extraGateCandidateCount": int(fit.get("extraGateCandidateCount", 0)),
                    "residualGradientBlockRefitCubicRelativeRmseBefore": fit.get(
                        "residualGradientBlockRefitCubicRelativeRmseBefore"
                    ),
                    "residualGradientBlockRefitCubicRelativeRmseAfter": fit.get(
                        "residualGradientBlockRefitCubicRelativeRmseAfter"
                    ),
                    "residualGradientBlockRefitCubicSelectionWeight": float(fit.get(
                        "residualGradientBlockRefitCubicSelectionWeight",
                        0.0,
                    )),
                    "residualGradientBlockRefitCubicFitWeight": float(fit.get(
                        "residualGradientBlockRefitCubicFitWeight",
                        0.0,
                    )),
                    "residualGradientBlockRefitCubicDirectionCount": int(fit.get(
                        "residualGradientBlockRefitCubicDirectionCount",
                        0,
                    )),
                    "residualGradientBlockRefitCubicSelectionDirectionCount": int(fit.get(
                        "residualGradientBlockRefitCubicSelectionDirectionCount",
                        0,
                    )),
                    "residualGradientBlockRefitCancellationPairsBefore": fit.get(
                        "residualGradientBlockRefitCancellationPairsBefore",
                        [],
                    ),
                    "residualGradientBlockRefitCancellationPairsAfter": fit.get(
                        "residualGradientBlockRefitCancellationPairsAfter",
                        [],
                    ),
                    "residualGradientBlockRefitCancellationRescueAttempted": bool(fit.get(
                        "residualGradientBlockRefitCancellationRescueAttempted",
                        False,
                    )),
                    "residualGradientBlockRefitCancellationRescued": bool(fit.get(
                        "residualGradientBlockRefitCancellationRescued",
                        False,
                    )),
                    "residualGradientBlockRefitCancellationRescueAtoms": fit.get(
                        "residualGradientBlockRefitCancellationRescueAtoms",
                        [],
                    ),
                    "residualGradientBlockRefitCancellationRescueScoreBefore": fit.get(
                        "residualGradientBlockRefitCancellationRescueScoreBefore"
                    ),
                    "residualGradientBlockRefitCancellationRescueScoreAfter": fit.get(
                        "residualGradientBlockRefitCancellationRescueScoreAfter"
                    ),
                    "residualGradientBlockRefitCancellationRescueAttempts": fit.get(
                        "residualGradientBlockRefitCancellationRescueAttempts",
                        [],
                    ),
                    # Written whether or not the pass changed anything. Two live runs came
                    # back bit-identical after the pass was added, and the artifact could not
                    # say whether it had found nothing or never run.
                    "collapsedAtomsRespread": bool(fit.get("collapsedAtomsRespread", False)),
                    "collapsedAtomRespreadAtoms": [int(value) for value in fit.get("collapsedAtomRespreadAtoms", [])],
                    "collapsedPairsBefore": None if fit.get("collapsedPairsBefore") is None else int(fit["collapsedPairsBefore"]),
                    "collapsedPairsAfter": None if fit.get("collapsedPairsAfter") is None else int(fit["collapsedPairsAfter"]),
                    "collapsedClosestPair": fit.get("collapsedClosestPair"),
                    "collapsedAtomRespreadThresholds": fit.get("collapsedAtomRespreadThresholds"),
                    "residualRegionAtomReplaced": bool(fit.get("residualRegionAtomReplaced", False)),
                    "residualRegionReplacementCount": int(fit.get("residualRegionReplacementCount", 0)),
                    "residualRegionReplacementCycles": int(fit.get("residualRegionReplacementCycles", 0)),
                    "residualRegionReplacementMaeBefore": None if fit.get("residualRegionReplacementMaeBefore") is None else float(fit["residualRegionReplacementMaeBefore"]),
                    "residualRegionReplacementWorstBefore": None if fit.get("residualRegionReplacementWorstBefore") is None else float(fit["residualRegionReplacementWorstBefore"]),
                    "splitMergeTailAtomRefined": bool(fit.get("splitMergeTailAtomRefined", False)),
                    "splitMergeTailAtomTrials": int(fit.get("splitMergeTailAtomTrials", 0)),
                    "splitMergeTailAtomMaeBefore": None if fit.get("splitMergeTailAtomMaeBefore") is None else float(fit["splitMergeTailAtomMaeBefore"]),
                    "splitMergeTailAtomWorstBefore": None if fit.get("splitMergeTailAtomWorstBefore") is None else float(fit["splitMergeTailAtomWorstBefore"]),
                    "selectiveAtomScaleRepaired": bool(fit.get("selectiveAtomScaleRepaired", False)),
                    "selectiveAtomScaleRepairAtoms": fit.get("selectiveAtomScaleRepairAtoms", []),
                    "selectiveAtomScaleRepairDeltas": fit.get("selectiveAtomScaleRepairDeltas", []),
                    "selectiveAtomScaleRepairMaeBefore": None if fit.get("selectiveAtomScaleRepairMaeBefore") is None else float(fit["selectiveAtomScaleRepairMaeBefore"]),
                    "selectiveAtomScaleRepairWorstBefore": None if fit.get("selectiveAtomScaleRepairWorstBefore") is None else float(fit["selectiveAtomScaleRepairWorstBefore"]),
                    "directionalAtomRepaired": bool(fit.get("directionalAtomRepaired", False)),
                    "directionalAtomRepairAtoms": fit.get("directionalAtomRepairAtoms", []),
                    "directionalAtomRepairSteps": int(fit.get("directionalAtomRepairSteps", 0)),
                    "directionalAtomRepairMaeBefore": None if fit.get("directionalAtomRepairMaeBefore") is None else float(fit["directionalAtomRepairMaeBefore"]),
                    "directionalAtomRepairWorstBefore": None if fit.get("directionalAtomRepairWorstBefore") is None else float(fit["directionalAtomRepairWorstBefore"]),
                    "atomTailDiagnostic": atom_tail_diagnostic,
                    "counterexampleCount": int(args.counterexample_count),
                    "counterexampleRepeat": int(args.counterexample_repeat),
                    "counterexampleJitterCount": int(args.counterexample_jitter_count),
                    "counterexampleJitterScale": float(args.counterexample_jitter_scale),
                    "tailProbeCount": int(args.tail_probe_count),
                    "tailProbeRepeat": int(args.tail_probe_repeat),
                    "tailProbeScale": float(args.tail_probe_scale if args.tail_probe_scale > 0.0 else args.validation_scale),
                    "tailProbeSelected": int(tail_probe_selected),
                    "fitSelectionScore": float(fit.get("selectionScore", fit.get("score", fit["mae"]))),
                    "chebyshevCalibrated": bool(fit.get("chebyshevCalibrated", False)),
                    "chebyshevSampleCount": int(fit.get("chebyshevSampleCount", 0)),
                    "chebyshevObjective": None if fit.get("chebyshevObjective") is None else float(fit["chebyshevObjective"]),
                    "robustScalarCalibrated": bool(fit.get("robustScalarCalibrated", False)),
                    "robustScalarCalibrationSteps": int(fit.get("robustScalarCalibrationSteps", 0)),
                    "fitScales": [float(value) for value in fit["scales"]],
                    "fitBias": float(fit["bias"]),
                    "asymptoticStructure": {
                        "coldStart": bool(fit.get("asymptoticGateStitchColdStart", False)),
                        "stitched": bool(fit.get("asymptoticGateStitched", False)),
                        "orientationMethod": fit.get("asymptoticGateStitchOrientationMethod"),
                        "lineCount": int(fit.get("asymptoticGateStitchLineCount", 0)),
                        "usableLineCount": int(fit.get("asymptoticGateStitchUsableLineCount", 0)),
                        "failedLineCount": int(fit.get("asymptoticGateStitchFailedLineCount", 0)),
                        "clusterCount": int(fit.get("asymptoticGateStitchClusterCount", 0)),
                        "clusterSupports": fit.get("asymptoticGateStitchClusterSupports", []),
                        "assignedGateSupports": fit.get("asymptoticGateStitchAssignedGateSupports", []),
                        "assignedUpSupports": fit.get("asymptoticGateStitchAssignedUpSupports", []),
                        "independentHessianUpComplete": bool(fit.get(
                            "asymptoticGateStitchIndependentHessianUpComplete",
                            False,
                        )),
                        "linearIdentifiedAtoms": fit.get("asymptoticGateStitchLinearIdentifiedAtoms", []),
                        "sourceFallbackAtoms": fit.get("asymptoticGateStitchFallbackAtoms", []),
                        "structureCalibrated": bool(fit.get("asymptoticGateStitchStructureCalibrated", False)),
                        "structureRefined": bool(fit.get("asymptoticGateStitchStructureRefined", False)),
                        "structureRefineAttempted": bool(fit.get("asymptoticGateStitchStructureRefineAttempted", False)),
                        "structureRefineSelectionScoreBefore": fit.get(
                            "asymptoticGateStitchStructureRefineSelectionScoreBefore"
                        ),
                        "structureRefineSelectionScoreAfter": fit.get(
                            "asymptoticGateStitchStructureRefineSelectionScoreAfter"
                        ),
                        "structureRefineRestarts": fit.get("asymptoticGateStitchStructureRefineRestarts", []),
                        "calibrationAttemptR2": fit.get("asymptoticGateStitchStructureCalibrationAttemptR2"),
                        "calibrationAttemptMae": fit.get("asymptoticGateStitchStructureCalibrationAttemptMae"),
                        "calibrationAttemptWorst": fit.get("asymptoticGateStitchStructureCalibrationAttemptWorst"),
                        "calibrationRestarts": fit.get("asymptoticGateStitchStructureCalibrationRestarts", []),
                        "trialR2": fit.get("asymptoticGateStitchTrialR2"),
                        "trialMae": fit.get("asymptoticGateStitchTrialMae"),
                        "trialWorstError": fit.get("asymptoticGateStitchTrialWorstError"),
                    },
                    "fitAttempts": fit_attempts,
                },
                "oracleResponse": {
                    "quantizationStep": float(args.oracle_quantization_step),
                    "mode": oracle.meta.get("oracleMode", "selected-unit-sum"),
                    "outputChannel": oracle.meta.get("outputChannel"),
                },
                "claimBoundary": [
                    (
                        "This gate proves a selected real local down_proj output-channel slice can be queried "
                        "as a black-box oracle and recovered as an explicit sum of rank-2 SwiGLU atoms under "
                        "the recorded budget."
                        if oracle.meta.get("oracleMode") == "down-proj-output-channel"
                        else
                        "This gate proves selected real local SwiGLU unit outputs can be queried as a "
                        "black-box oracle and recovered as an explicit sum of rank-2 SwiGLU atoms under "
                        "the recorded budget."
                    ),
                    "It does not prove full FFN extraction, all-channel decompilation, exhaustive LLM circuit recovery, or semantic original-neuron identity from output-only access.",
                ],
            }
            output_json.write_text(dump_report_json(report) + "\n", encoding="utf-8")
            if output_md:
                output_md.write_text(
                    "\n".join([
                        "# Blind 4096D SwiGLU Oracle Decompiler",
                        "",
                        f"- pass: `{str(pass_gate).lower()}`",
                        f"- model: `{oracle.meta['model']}`",
                        f"- layer/unit: `{oracle.meta['layer']}/{oracle.meta.get('units', [oracle.meta['unit']])}`",
                        f"- oracle mode: `{oracle.meta.get('oracleMode', 'selected-unit-sum')}`",
                        f"- output channel: `{oracle.meta.get('outputChannel')}`",
                        f"- raw oracle calls: `{oracle.query_count}`",
                        f"- validation R2: `{metrics['r2']}`",
                        f"- validation MAE: `{metrics['mae']}`",
                        f"- subspace min principal cosine: `{source_alignment['subspaceMinPrincipalCosine']}`",
                        f"- post-hoc paired atom min joint cosine: `{source_alignment['pairedAtomMinJointCosine']}`",
                        f"- post-hoc paired atoms >= 0.99: `{source_alignment['pairedAtomCountAbove099']}/{args.atom_count}`",
                        "",
                        "Access audit:",
                        "",
                        "- solverAccess: `oracle-only`",
                        "- weightsUsedBySolver: `false`",
                        "- hooksUsedBySolver: `false`",
                        "- postHocWitnessOnly: `true`",
                        "",
                        "Boundary: this is a selected real SwiGLU-unit black-box gate, not full LLM decompilation.",
                        "",
                    ]),
                    encoding="utf-8",
                )
            print(json.dumps(report["summary"], indent=2))
            return 0 if pass_gate else 1
        finally:
            oracle.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Blind oracle-only recovery for one real 4096D SwiGLU unit.")
    parser.add_argument("--oracle-server", action="store_true")
    parser.add_argument("--oracle-ready-mode", choices=["full", "opaque"], default="full")
    parser.add_argument("--oracle-eval-only", action="store_true")
    parser.add_argument("--oracle-read-fd", type=int, default=-1)
    parser.add_argument("--oracle-write-fd", type=int, default=-1)
    parser.add_argument("--oracle-socket", default="")
    parser.add_argument("--require-opaque-oracle", action="store_true")
    parser.add_argument("--defer-posthoc-truth", action="store_true")
    parser.add_argument("--frozen-fit-output", default="")
    parser.add_argument("--model", default="unsloth/mistral-7b-instruct-v0.3-bnb-4bit")
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--unit", type=int, default=0)
    parser.add_argument("--units", default=None)
    parser.add_argument("--output-channel", type=int, default=-1)
    parser.add_argument(
        "--oracle-output-mode",
        choices=["scalar", "selected-unit-ffn-vector"],
        default="scalar",
        help=(
            "what one oracle call returns. `scalar` is the summed block or a single "
            "down_proj channel. `selected-unit-ffn-vector` returns the block's whole "
            "contribution to the residual stream, which makes the Jacobian a matrix and a "
            "unit's boundary a rank-1 jump in it"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--oracle-batch-size", type=int, default=2048)
    parser.add_argument("--oracle-quantization-step", type=float, default=0.0)
    parser.add_argument("--active-rank", type=int, default=2)
    parser.add_argument("--expected-dimension", type=int, default=0)
    parser.add_argument("--atom-count", type=int, default=None)
    parser.add_argument(
        "--gradient-mode",
        choices=[
            "coordinate",
            "forward-coordinate",
            "symmetric-regression-coordinate",
            "symmetric-richardson-coordinate",
            "symmetric-multiscale-coordinate",
            "random-direction",
        ],
        default="coordinate",
    )
    parser.add_argument("--basis-moment", choices=["auto", "centered", "raw"], default="auto")
    parser.add_argument("--probe-points", type=int, default=2)
    parser.add_argument("--direction-count", type=int, default=384)
    parser.add_argument("--subspace-count", type=int, default=0)
    parser.add_argument("--cubic-direction-count", type=int, default=0)
    parser.add_argument("--cubic-step", type=float, default=0.35)
    parser.add_argument("--cubic-steps", default="")
    parser.add_argument("--cubic-holdout-count", type=int, default=0)
    parser.add_argument("--finite-difference-step", type=float, default=0.08)
    parser.add_argument("--finite-difference-steps", default="")
    parser.add_argument("--input-scale", type=float, default=0.45)
    parser.add_argument(
        "--auto-input-scale",
        action="store_true",
        help="pick --input-scale per block from oracle-observed output amplitude instead of using a fixed value",
    )
    parser.add_argument(
        "--auto-input-scale-candidates",
        default="0.45,0.6,0.75,0.9",
        help="ascending candidate scales; the last one is the structure-safety cap",
    )
    parser.add_argument("--auto-input-scale-probe-count", type=int, default=512)
    parser.add_argument(
        "--auto-input-scale-quantization-ratio",
        type=float,
        default=1000.0,
        help="required output std as a multiple of the oracle quantization step",
    )
    parser.add_argument("--train-scale-mixture", default="")
    parser.add_argument("--train-sampling-mode", choices=["gaussian", "full-gaussian", "radial-shell", "active-radial-shell", "sobol-active-gaussian", "gaussian-sobol-active-mix"], default="gaussian")
    parser.add_argument("--validation-scale", type=float, default=0.65)
    parser.add_argument("--train-count", type=int, default=1600)
    parser.add_argument("--validation-count", type=int, default=512)
    parser.add_argument("--fit-steps", type=int, default=1200)
    parser.add_argument("--fit-learning-rate", type=float, default=0.025)
    parser.add_argument("--fit-restarts", type=int, default=0)
    parser.add_argument("--fit-restart-chunk-size", type=int, default=0)
    parser.add_argument("--fit-mode", choices=["auto", "sequential", "batched"], default="auto")
    parser.add_argument(
        "--recovery-path",
        choices=["nonlinear-fit", "asymptotic-cold-start"],
        default="nonlinear-fit",
    )
    parser.add_argument("--fit-gradient-count", type=int, default=0)
    parser.add_argument(
        "--fit-gradient-mode",
        choices=["forward", "symmetric-regression"],
        default="forward",
    )
    parser.add_argument("--fit-gradient-step", type=float, default=0.0)
    parser.add_argument("--fit-gradient-steps", default="")
    parser.add_argument("--fit-gradient-weight", type=float, default=0.2)
    parser.add_argument("--fit-selection-count", type=int, default=0)
    parser.add_argument("--fit-selection-scale", type=float, default=0.0)
    parser.add_argument("--selection-tail-weight", type=float, default=None)
    parser.add_argument("--selection-tail-quantile", type=float, default=None)
    parser.add_argument("--selection-worst-weight", type=float, default=None)
    parser.add_argument("--polish-top-k", type=int, default=None)
    parser.add_argument("--backfit-cycles", type=int, default=None)
    parser.add_argument("--tail-polish-steps", type=int, default=None)
    parser.add_argument("--tail-polish-weight", type=float, default=None)
    parser.add_argument("--tail-polish-fraction", type=float, default=None)
    parser.add_argument("--tail-polish-lr", type=float, default=None)
    parser.add_argument("--tail-polish-loss", choices=["topk", "smoothmax"], default=None)
    parser.add_argument("--tail-polish-smoothness", type=float, default=None)
    parser.add_argument("--tail-polish-calibration", choices=["ls", "direct", "best-worst"], default=None)
    parser.add_argument("--final-lbfgs-steps", type=int, default=None)
    parser.add_argument("--final-lbfgs-tail-weight", type=float, default=None)
    parser.add_argument("--final-lbfgs-tail-fraction", type=float, default=None)
    parser.add_argument("--robust-scalar-calibration", action="store_true")
    parser.add_argument("--scalar-calibration-steps", type=int, default=None)
    parser.add_argument("--scalar-calibration-lr", type=float, default=None)
    parser.add_argument("--scalar-calibration-tail-weight", type=float, default=None)
    parser.add_argument("--scalar-calibration-tail-fraction", type=float, default=None)
    parser.add_argument("--counterexample-count", type=int, default=0)
    parser.add_argument("--counterexample-repeat", type=int, default=4)
    parser.add_argument("--counterexample-jitter-count", type=int, default=0)
    parser.add_argument("--counterexample-jitter-scale", type=float, default=0.025)
    parser.add_argument("--tail-probe-count", type=int, default=0)
    parser.add_argument("--tail-probe-repeat", type=int, default=4)
    parser.add_argument("--tail-probe-scale", type=float, default=0.0)
    parser.add_argument("--rescue-train-target", type=int, default=0)
    parser.add_argument("--rescue-gradient-target", type=int, default=0)
    parser.add_argument("--rescue-steps", type=int, default=0)
    parser.add_argument("--rescue-restarts", type=int, default=0)
    parser.add_argument("--disable-adaptive-rescue", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--seed", type=int, default=260620)
    parser.add_argument("--min-r2", type=float, default=0.985)
    parser.add_argument("--max-mae", type=float, default=0.015)
    parser.add_argument("--max-worst-error", type=float, default=1e30)
    parser.add_argument("--min-subspace-cosine", type=float, default=0.93)
    parser.add_argument(
        "--min-paired-atom-cosine",
        type=float,
        default=0.0,
        help=(
            "post-hoc identity bar: every recovered atom must match its assigned source unit "
            "at this joint gate/up cosine. 0 disables it. The witness is read only after the "
            "recovery is frozen, so this changes the verdict, never the solver's input"
        ),
    )
    parser.add_argument("--max-cubic-witness-relative-rmse", type=float, default=float("inf"))
    parser.add_argument("--max-oracle-calls", type=int, default=20000)
    parser.add_argument("--basis-diagnostic-only", action="store_true")
    parser.add_argument("--fit-snapshot-input", default="")
    parser.add_argument("--fit-snapshot-output", default="")
    parser.add_argument("--output-json", required=False)
    parser.add_argument("--output-md", required=False)
    return parser


def apply_fit_env_overrides(args: argparse.Namespace) -> None:
    overrides: list[tuple[str, object | None]] = [
        ("GCL_BLIND_SWIGLU_SELECTION_TAIL_WEIGHT", args.selection_tail_weight),
        ("GCL_BLIND_SWIGLU_SELECTION_TAIL_QUANTILE", args.selection_tail_quantile),
        ("GCL_BLIND_SWIGLU_SELECTION_WORST_WEIGHT", args.selection_worst_weight),
        ("GCL_BLIND_SWIGLU_POLISH_TOP_K", args.polish_top_k),
        ("GCL_BLIND_SWIGLU_BACKFIT_CYCLES", args.backfit_cycles),
        ("GCL_BLIND_SWIGLU_TAIL_POLISH_STEPS", args.tail_polish_steps),
        ("GCL_BLIND_SWIGLU_TAIL_POLISH_WEIGHT", args.tail_polish_weight),
        ("GCL_BLIND_SWIGLU_TAIL_POLISH_FRACTION", args.tail_polish_fraction),
        ("GCL_BLIND_SWIGLU_TAIL_POLISH_LR", args.tail_polish_lr),
        ("GCL_BLIND_SWIGLU_TAIL_POLISH_LOSS", args.tail_polish_loss),
        ("GCL_BLIND_SWIGLU_TAIL_POLISH_SMOOTHNESS", args.tail_polish_smoothness),
        ("GCL_BLIND_SWIGLU_TAIL_POLISH_CALIBRATION", args.tail_polish_calibration),
        ("GCL_BLIND_SWIGLU_FINAL_LBFGS_STEPS", args.final_lbfgs_steps),
        ("GCL_BLIND_SWIGLU_FINAL_LBFGS_TAIL_WEIGHT", args.final_lbfgs_tail_weight),
        ("GCL_BLIND_SWIGLU_FINAL_LBFGS_TAIL_FRACTION", args.final_lbfgs_tail_fraction),
        ("GCL_BLIND_SWIGLU_SCALAR_CALIBRATION_STEPS", args.scalar_calibration_steps),
        ("GCL_BLIND_SWIGLU_SCALAR_CALIBRATION_LR", args.scalar_calibration_lr),
        ("GCL_BLIND_SWIGLU_SCALAR_CALIBRATION_TAIL_WEIGHT", args.scalar_calibration_tail_weight),
        ("GCL_BLIND_SWIGLU_SCALAR_CALIBRATION_TAIL_FRACTION", args.scalar_calibration_tail_fraction),
    ]
    for key, value in overrides:
        if value is not None:
            os.environ[key] = str(value)
    if args.robust_scalar_calibration:
        os.environ["GCL_BLIND_SWIGLU_ROBUST_SCALAR_CALIBRATION"] = "1"


def die_with_parent() -> str:
    """Ask the kernel to kill this process when its parent goes away.

    `spawnSync` with a timeout kills the process it started and nothing below it. The lane spawns
    the sweep, the sweep spawns this solver, so a lane that times out kills the middle one and
    leaves the solver running - reparented to init, holding the card and the cores, invisible to
    whoever reads the failed artifact. Measured after one such timeout: the solver was still
    going two minutes later at 25 cores, with `ppid` 2121.

    `PR_SET_PDEATHSIG` closes it at the kernel rather than by anyone remembering to clean up. It
    is Linux-only and best effort: a platform without it simply keeps the old behaviour, and the
    outcome is reported rather than assumed.
    """
    try:
        import ctypes

        PR_SET_PDEATHSIG = 1
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        if libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0) != 0:
            return f"prctl failed: errno {ctypes.get_errno()}"
    except (OSError, AttributeError, ImportError) as error:
        return f"unavailable: {type(error).__name__}: {error}"
    # The parent can already have died between the fork and this call, in which case the signal
    # will never arrive and this process would outlive it anyway.
    if os.getppid() == 1:
        return "parent already gone"
    return "armed"


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    parent_death = die_with_parent()
    _PROGRESS_HOOK["args"] = args
    if parent_death != "armed":
        # Not fatal, and not silent either: a run whose children can outlive it is a different
        # run, and this is the only place that knows.
        print(json.dumps({"type": "warning", "parentDeathSignal": parent_death}), file=sys.stderr, flush=True)
    if args.oracle_server:
        return oracle_server(args)
    if not args.output_json:
        parser.error("--output-json is required outside --oracle-server mode")
    apply_fit_env_overrides(args)
    return run_solver(args)


if __name__ == "__main__":
    raise SystemExit(main())
