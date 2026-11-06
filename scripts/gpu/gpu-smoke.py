#!/usr/bin/env python3
"""GPU environment smoke check for GCL live lanes.

Loads the exact model the ordered-path stability lane uses (through the
repository's own adapter registry), fingerprints the execution environment,
measures forward-pass throughput, and verifies same-device determinism plus
forward-hook capture. Works on cuda, mps, and cpu so the same check validates
any host before a live lane run.

Device resolution order (mirrors the ordered-path runner plus an environment
override that is useful in notebooks and containers):
    1. GCL_DEVICE environment variable (e.g. "cuda:0", "cpu")
    2. --device flag (default "auto": cuda -> mps -> cpu)

Output: JSON report (default under benchmarks/scratch/gpu-smoke/) plus a
console summary. Exit code 0 when the environment is usable, 1 otherwise.
A cpu-only resolution is usable but reported as not GPU-accelerated.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_ROOT = REPO_ROOT / "tools"
for path in (str(TOOLS_ROOT), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import torch  # noqa: E402

from tools.llm_reader.adapters.registry import create_adapter  # noqa: E402

DETERMINISM_THRESHOLD = 1e-5
SHORT_PROMPT = "The capital of France is"
LONG_PROMPT = (
    "The genetic code laboratory records how a language model computes an answer. "
    "Every circuit is replayed, stressed with paraphrases, and compared against "
    "neighbor layers before any claim is written down, so the reader can explain "
    "not just what the model said but why the path behind it held or broke."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a GCL live-lane execution environment.")
    parser.add_argument("--adapter", default="gpt2")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="auto",
        help="Device class to request; GCL_DEVICE env var overrides this flag.",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--short-runs", type=int, default=24)
    parser.add_argument("--long-runs", type=int, default=8)
    parser.add_argument(
        "--output-json",
        default="benchmarks/scratch/gpu-smoke/report.json",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return requested


def read_total_memory_gb() -> Optional[float]:
    try:
        text = Path("/proc/meminfo").read_text(encoding="utf-8")
        match = re.search(r"^MemTotal:\s+(\d+)\s+kB$", text, re.MULTILINE)
        if match:
            return round(int(match.group(1)) / (1024 * 1024), 2)
    except OSError:
        pass
    return None


def nvidia_driver_info() -> Optional[Dict[str, Any]]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    rows = []
    for line in result.stdout.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 3:
            rows.append({"name": parts[0], "driverVersion": parts[1], "memoryTotal": parts[2]})
    return {"gpus": rows} if rows else None


def probe_environment() -> Dict[str, Any]:
    gpus: List[Dict[str, Any]] = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            gpus.append(
                {
                    "index": index,
                    "name": props.name,
                    "totalMemoryGB": round(props.total_memory / (1024**3), 2),
                    "multiprocessorCount": getattr(props, "multi_processor_count", None),
                    "capability": f"{props.major}.{props.minor}",
                }
            )
    mps_available = bool(
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    )
    return {
        "os": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "numpy": __import__("numpy").__version__,
        "cudaBuild": torch.version.cuda,
        "cudaAvailable": torch.cuda.is_available(),
        "mpsAvailable": mps_available,
        "gpuCount": len(gpus),
        "gpus": gpus,
        "nvidiaSmi": nvidia_driver_info(),
        "cpuCount": torch.get_num_threads(),
        "totalMemoryGB": read_total_memory_gb(),
    }


def benchmark_forward(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    prompt: str,
    runs: int,
) -> Dict[str, Any]:
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    token_count = int(encoded["input_ids"].shape[1])
    with torch.no_grad():
        model(**encoded)  # warmup, excluded from timing
        started = time.perf_counter()
        for _ in range(max(1, runs)):
            model(**encoded)
        seconds = time.perf_counter() - started
    return {
        "promptHead": prompt[:48],
        "tokenCount": token_count,
        "runs": max(1, runs),
        "seconds": round(seconds, 4),
        "promptsPerSecond": round(max(1, runs) / seconds, 3),
        "tokensPerSecond": round(max(1, runs) * token_count / seconds, 3),
    }


def determinism_check(model: Any, tokenizer: Any, device: torch.device) -> Dict[str, Any]:
    encoded = tokenizer(SHORT_PROMPT, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        first = model(**encoded).logits.detach().cpu()
        second = model(**encoded).logits.detach().cpu()
    max_abs_diff = float((first - second).abs().max())
    return {
        "maxAbsLogitDiff": max_abs_diff,
        "threshold": DETERMINISM_THRESHOLD,
        "passed": max_abs_diff <= DETERMINISM_THRESHOLD,
    }


def hook_capture_check(model: Any, tokenizer: Any, device: torch.device) -> Dict[str, Any]:
    blocks = getattr(getattr(model, "transformer", None), "h", None)
    if blocks is None:
        return {"supported": False, "passed": True, "note": "model has no transformer.h blocks; skipped"}
    captured: Dict[str, Any] = {}

    def hook(_module: Any, _inputs: Any, output: Any) -> None:
        if "output" not in captured:
            captured["output"] = output.detach()

    handle = blocks[6].mlp.register_forward_hook(hook)
    try:
        encoded = tokenizer(SHORT_PROMPT, return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            model(**encoded)
    finally:
        handle.remove()
    tensor = captured.get("output")
    if tensor is None:
        return {"supported": True, "module": "transformer.h[6].mlp", "passed": False, "note": "hook never fired"}
    return {
        "supported": True,
        "module": "transformer.h[6].mlp",
        "outputShape": list(tensor.shape),
        "outputDevice": str(tensor.device),
        "finite": bool(torch.isfinite(tensor).all()),
        "passed": bool(torch.isfinite(tensor).all()),
    }


def main() -> None:
    args = parse_args()
    env_override = None
    requested = args.device
    override_value = os.environ.get("GCL_DEVICE")
    if override_value:
        requested = override_value
        env_override = {"variable": "GCL_DEVICE", "value": override_value}
    resolved = resolve_device(requested)
    if resolved == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")
    device = torch.device(resolved)

    environment = probe_environment()
    adapter = create_adapter(args.adapter)
    tokenizer, model = adapter.load_model_and_tokenizer(
        args.model,
        local_files_only=args.local_files_only,
        device=str(device),
    )
    param_count = sum(int(p.numel()) for p in model.parameters())
    model_dtype = str(next(model.parameters()).dtype)

    short_benchmark = benchmark_forward(model, tokenizer, device, SHORT_PROMPT, args.short_runs)
    long_benchmark = benchmark_forward(model, tokenizer, device, LONG_PROMPT, args.long_runs)
    determinism = determinism_check(model, tokenizer, device)
    hook_check = hook_capture_check(model, tokenizer, device)

    ok = bool(determinism["passed"] and hook_check["passed"] and short_benchmark["promptsPerSecond"] > 0)
    report = {
        "schemaVersion": "1",
        "kind": "gpu-smoke",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "ok": ok,
        "gpuAccelerated": resolved.startswith("cuda"),
        "requestedDevice": requested,
        "resolvedDevice": str(device),
        "deviceEnvironmentOverride": env_override,
        "environment": environment,
        "model": {
            "id": args.model,
            "adapter": args.adapter,
            "class": model.__class__.__name__,
            "paramCount": param_count,
            "dtype": model_dtype,
            "localFilesOnly": bool(args.local_files_only),
        },
        "benchmarks": {
            "shortPrompt": short_benchmark,
            "longPrompt": long_benchmark,
        },
        "determinism": determinism,
        "hookCheck": hook_check,
        "claimBoundary": (
            "This report validates one host for live lane execution: model load, forward-pass "
            "throughput at the recorded prompt lengths, same-device logit determinism, and "
            "forward-hook capture. It does not claim lane gate outcomes, cross-device numeric "
            "equality, or any benchmark comparison beyond the numbers recorded here."
        ),
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    gpu_tag = "GPU" if report["gpuAccelerated"] else "CPU-ONLY"
    print(f"[gpu-smoke] resolved device : {device} ({gpu_tag})")
    if env_override:
        print(f"[gpu-smoke] env override    : {env_override['variable']}={env_override['value']}")
    if report["gpuAccelerated"]:
        for gpu in environment["gpus"]:
            print(
                f"[gpu-smoke] gpu             : {gpu['name']} ({gpu['totalMemoryGB']} GB, "
                f"sm_{gpu['capability']}, {gpu['multiprocessorCount']} SMs)"
            )
    else:
        print("[gpu-smoke] WARNING: no GPU resolved; lanes run but without acceleration.")
    print(f"[gpu-smoke] model            : {args.model} ({param_count:,} params, {model_dtype})")
    print(
        f"[gpu-smoke] short prompt     : {short_benchmark['promptsPerSecond']} prompts/s "
        f"({short_benchmark['tokenCount']} tokens, {short_benchmark['seconds']}s / {short_benchmark['runs']} runs)"
    )
    print(
        f"[gpu-smoke] long prompt      : {long_benchmark['promptsPerSecond']} prompts/s "
        f"({long_benchmark['tokenCount']} tokens)"
    )
    print(
        f"[gpu-smoke] determinism      : max|dlogit| = {determinism['maxAbsLogitDiff']:.3e} "
        f"(threshold {DETERMINISM_THRESHOLD:.0e}) -> {'PASS' if determinism['passed'] else 'FAIL'}"
    )
    print(
        f"[gpu-smoke] hook capture     : {'PASS' if hook_check['passed'] else 'FAIL'} "
        f"({hook_check.get('module', 'skipped')})"
    )
    print(f"[gpu-smoke] verdict          : {'OK' if ok else 'FAILED'}")
    print(f"[gpu-smoke] report           : {output_path}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
