#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any


MODEL = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
LAYER = 15
OUTPUT_CHANNEL = 2048
UNITS = [7885, 2037, 13787, 2761, 5013, 3496, 5314, 9185, 2434, 2487, 223, 5617, 12804, 5572, 12373, 10618]
DIMENSION = 4096
ATOM_COUNT = 16
ACTIVE_RANK = 32
MAX_ORACLE_CALLS = 1_500_000


def classify_solver_file_access(
    strace_text: str,
    *,
    sensitive_control_paths: list[Path],
    teacher_model_markers: list[str] | None = None,
) -> dict[str, Any]:
    lines = [line for line in strace_text.splitlines() if line.strip()]
    package_metadata_rows = [
        line for line in lines
        if ".dist-info/" in line and any(
            package in line.lower()
            for package in ("huggingface_hub-", "safetensors-", "tokenizers-")
        )
    ]
    model_markers = [
        "/.cache/huggingface/", "/huggingface/hub/", "/hub/models--",
        "models--", "/snapshots/", "/blobs/", "mistral-7b",
        ".safetensors", "pytorch_model.bin", ".gguf",
    ]
    model_markers.extend(
        marker.lower() for marker in (teacher_model_markers or []) if marker.strip()
    )
    model_access_rows = [
        line for line in lines
        if any(marker in line.lower() for marker in model_markers)
        and line not in package_metadata_rows
    ]
    control_markers = [str(path.resolve()).lower() for path in sensitive_control_paths]
    control_access_rows = [
        line for line in lines
        if any(marker in line.lower() for marker in control_markers)
    ]
    return {
        "lineCount": len(lines),
        "packageMetadataRows": package_metadata_rows,
        "modelAccessRows": model_access_rows,
        "controlAccessRows": control_access_rows,
    }


def find_solver_source_locator_matches(
    solver_command: list[str],
    *,
    model: str,
    layer: int,
) -> list[str]:
    command_text = " ".join(solver_command).lower()
    return sorted({
        marker for marker in (
            "mistral", "llama", f"layer-{layer}", f"layer{layer}",
            model.lower(), model.rsplit("/", maxsplit=1)[-1].lower(),
        )
        if marker and marker in command_text
    })


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sanitized_env() -> dict[str, str]:
    allowed = {
        "PATH", "HOME", "USER", "LANG", "LC_ALL", "LD_LIBRARY_PATH",
        "CUDA_VISIBLE_DEVICES", "HF_HOME", "TRANSFORMERS_CACHE", "XDG_CACHE_HOME",
    }
    env = {key: value for key, value in os.environ.items() if key in allowed}
    env["PYTHONUNBUFFERED"] = "1"
    return env


def sanitized_solver_env() -> dict[str, str]:
    allowed = {"PATH", "LANG", "LC_ALL", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES"}
    env = {key: value for key, value in os.environ.items() if key in allowed}
    env["PYTHONUNBUFFERED"] = "1"
    return env


def load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"missing {label}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return value


def run_isolated(args: argparse.Namespace) -> int:
    model = str(args.model)
    layer = int(args.layer)
    units = [int(value) for value in str(args.units).split(",") if value.strip()]
    output_channel = int(args.output_channel)
    dimension = int(args.dimension)
    atom_count = int(args.atom_count)
    active_rank = int(args.active_rank or atom_count * 2)
    max_oracle_calls = int(args.max_oracle_calls)
    quantization_step = float(args.quantization_step)
    if not units or len(units) != atom_count or len(set(units)) != atom_count:
        raise RuntimeError("units must contain exactly atom-count unique source ids")
    if dimension <= 0 or atom_count <= 0 or active_rank < atom_count or max_oracle_calls <= 0:
        raise RuntimeError("dimension/atom-count/active-rank/max-oracle-calls are inconsistent")
    if quantization_step < 0.0:
        raise RuntimeError("quantization-step must be non-negative")
    output_json = Path(args.output_json).resolve()
    output_md = Path(args.output_md).resolve()
    output_dir = output_json.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    solver_work_dir = Path(tempfile.mkdtemp(prefix=f"gcl-solver-{run_id[:16]}-"))
    started_ns = time.time_ns()
    decompiler = Path(__file__).with_name("blind_swiglu_oracle_decompiler.py").resolve()
    oracle_server = Path(__file__).with_name("swiglu_oracle_server.py").resolve()
    solver_worker = Path(__file__).with_name("blind_swiglu_solver_worker.py").resolve()
    protocol_module = Path(__file__).with_name("swiglu_oracle_protocol.py").resolve()
    frozen_fit = output_dir / "frozen-recovery.npz"
    solver_report_path = output_dir / "solver-frozen-report.json"
    solver_md_path = output_dir / "solver-frozen-report.md"
    verifier_report_path = output_dir / "posthoc-verifier-report.json"
    verifier_md_path = output_dir / "posthoc-verifier-report.md"
    oracle_stderr_path = output_dir / "eval-oracle-stderr.log"
    oracle_socket_path = Path(tempfile.gettempdir()) / f"gcl-sw-{run_id[:16]}.sock"
    oracle_witness_path = output_dir / "eval-oracle-witness.json"
    solver_stdout_path = output_dir / "solver-stdout.log"
    solver_stderr_path = output_dir / "solver-stderr.log"
    solver_strace_path = output_dir / "solver-file-access.strace"
    private_frozen_fit = solver_work_dir / "frozen-recovery.npz"
    private_solver_report_path = solver_work_dir / "solver-report.json"
    private_solver_md_path = solver_work_dir / "solver-report.md"
    private_solver_strace_path = solver_work_dir / "file-access.strace"
    verifier_stdout_path = output_dir / "verifier-stdout.log"
    verifier_stderr_path = output_dir / "verifier-stderr.log"
    for generated in (
        output_json, output_md, frozen_fit, solver_report_path, solver_md_path,
        verifier_report_path, verifier_md_path, oracle_stderr_path,
        oracle_socket_path, oracle_witness_path,
        solver_stdout_path, solver_stderr_path, verifier_stdout_path,
        verifier_stderr_path, solver_strace_path,
    ):
        generated.unlink(missing_ok=True)

    env = sanitized_env()
    solver_env = sanitized_solver_env()
    manifest_read, manifest_write = os.pipe()
    manifest = {
        "model": model,
        "layer": layer,
        "units": units,
        "outputChannel": output_channel,
        "device": args.device,
        "localFilesOnly": True,
        "quantizationStep": quantization_step,
        "maxOracleCalls": max_oracle_calls,
        "batchSize": 2048,
    }
    oracle_command = [
        sys.executable, str(oracle_server),
        "--manifest-fd", str(manifest_read),
        "--socket", str(oracle_socket_path),
        "--witness", str(oracle_witness_path),
    ]
    solver_command = [
        sys.executable, str(solver_worker),
        "--endpoint", str(oracle_socket_path),
        "--dimension", str(dimension),
        "--atom-count", str(atom_count),
        "--budget", str(max_oracle_calls),
        "--frozen-output", str(private_frozen_fit),
        "--output-json", str(private_solver_report_path),
        "--output-md", str(private_solver_md_path),
    ]
    forbidden_solver_flags = {"--model", "--layer", "--unit", "--units", "--output-channel"}
    solver_forbidden_flags = sorted(forbidden_solver_flags.intersection(solver_command))
    solver_manifest_env = sorted(
        key for key in solver_env
        if key.startswith("GCL_") or any(token in key for token in ("UNIT", "LAYER", "CASE_MANIFEST"))
    )
    solver_source_locator_matches = find_solver_source_locator_matches(
        solver_command,
        model=model,
        layer=layer,
    )
    strace = shutil.which("strace")
    solver_launch_command = solver_command
    if strace:
        solver_launch_command = [
            strace, "-f", "-qq", "-e", "trace=open,openat,statx,readlink",
            "-o", str(private_solver_strace_path), "--", *solver_command,
        ]

    oracle_process: subprocess.Popen[bytes] | None = None
    solver_process: subprocess.Popen[bytes] | None = None
    solver_finished_ns: int | None = None
    verifier_started_ns: int | None = None
    oracle_exit: int | None = None
    solver_exit: int | None = None
    verifier_exit: int | None = None
    try:
        with oracle_stderr_path.open("wb") as oracle_stderr:
            oracle_process = subprocess.Popen(
                oracle_command,
                cwd=decompiler.parents[2],
                env=env,
                stderr=oracle_stderr,
                pass_fds=(manifest_read,),
                close_fds=True,
            )
        os.close(manifest_read)
        manifest_read = -1
        with os.fdopen(manifest_write, "w", encoding="utf-8") as manifest_stream:
            json.dump(manifest, manifest_stream)
        manifest_write = -1
        socket_deadline = time.monotonic() + min(args.timeout_seconds, 300)
        while not oracle_socket_path.exists():
            if oracle_process.poll() is not None:
                raise RuntimeError(
                    f"isolated oracle exited before socket ready: {oracle_process.returncode}; "
                    f"stderr={oracle_stderr_path.read_text(encoding='utf-8', errors='replace')[-4000:]}"
                )
            if time.monotonic() >= socket_deadline:
                raise RuntimeError("timed out waiting for isolated oracle socket")
            time.sleep(0.05)
        with solver_stdout_path.open("wb") as solver_stdout, solver_stderr_path.open("wb") as solver_stderr:
            solver_process = subprocess.Popen(
                solver_launch_command,
                cwd=decompiler.parents[2],
                env=solver_env,
                stdout=solver_stdout,
                stderr=solver_stderr,
                close_fds=True,
            )
            solver_exit = solver_process.wait(timeout=args.timeout_seconds)
        solver_finished_ns = time.time_ns()
        oracle_exit = oracle_process.wait(timeout=60)
    finally:
        for descriptor in (manifest_read, manifest_write):
            if descriptor >= 0:
                os.close(descriptor)
        if solver_process is not None and solver_process.poll() is None:
            solver_process.kill()
        if oracle_process is not None and oracle_process.poll() is None:
            oracle_process.kill()

    # A solver that misses the quality gate still writes its frozen fit and report before
    # exiting non-zero. Raising here used to throw that away, so a failed case left no
    # artifact at all: no recovered atoms, no post-hoc alignment, no trace. That makes the
    # difference between "the atoms are wrong" and "the atoms are right but the fit is
    # short" impossible to see, which is the difference between two completely different
    # repairs. When the artifacts exist the run continues and produces a report with
    # status fail; only a solver that produced nothing is still a hard error.
    solver_produced_artifacts = private_frozen_fit.is_file() and private_solver_report_path.is_file()
    recovery_failed = solver_exit != 0 or oracle_exit != 0
    if recovery_failed and not solver_produced_artifacts:
        raise RuntimeError(
            f"isolated recovery failed: solver={solver_exit} oracle={oracle_exit}; "
            f"solver stderr={solver_stderr_path.read_text(encoding='utf-8', errors='replace')[-4000:]} "
            f"oracle stderr={oracle_stderr_path.read_text(encoding='utf-8', errors='replace')[-4000:]}"
        )
    for private_path, public_path in (
        (private_frozen_fit, frozen_fit),
        (private_solver_report_path, solver_report_path),
        (private_solver_md_path, solver_md_path),
    ):
        if not private_path.is_file():
            raise RuntimeError(f"solver did not produce required artifact: {private_path.name}")
        shutil.copy2(private_path, public_path)
    if private_solver_strace_path.is_file():
        shutil.copy2(private_solver_strace_path, solver_strace_path)
    solver_report = load_json(solver_report_path, "solver frozen report")
    oracle_witness = load_json(oracle_witness_path, "eval oracle transcript witness")
    frozen_hash_before = sha256_file(frozen_fit)
    strace_text = solver_strace_path.read_text(encoding="utf-8", errors="replace") if solver_strace_path.exists() else ""
    solver_file_access = classify_solver_file_access(
        strace_text,
        sensitive_control_paths=[
            Path(__file__), oracle_server, oracle_witness_path,
            verifier_report_path, verifier_md_path,
        ],
        teacher_model_markers=[model, model.rsplit("/", maxsplit=1)[-1]],
    )
    solver_model_access_rows = list(solver_file_access["modelAccessRows"])
    solver_control_access_rows = list(solver_file_access["controlAccessRows"])
    solver_trace_complete = bool(
        strace
        and solver_strace_path.is_file()
        and int(solver_file_access["lineCount"]) > 0
    )
    solver_trace_sha256 = sha256_file(solver_strace_path) if solver_trace_complete else None

    verifier_command = [
        sys.executable, str(decompiler),
        "--model", model,
        "--layer", str(layer),
        "--unit", str(units[0]),
        "--units", ",".join(str(value) for value in units),
        "--output-channel", str(output_channel),
        "--device", args.device,
        "--local-files-only",
        "--oracle-quantization-step", str(quantization_step),
        "--active-rank", str(active_rank),
        "--atom-count", str(atom_count),
        "--fit-snapshot-input", str(frozen_fit),
        "--validation-count", "4096",
        "--validation-scale", "0.65",
        "--disable-adaptive-rescue",
        "--min-r2", "0.999",
        "--max-mae", "0.001",
        "--max-worst-error", "0.005",
        "--min-subspace-cosine", "0.999",
        "--max-oracle-calls", "8192",
        "--seed", str(args.seed),
        "--output-json", str(verifier_report_path),
        "--output-md", str(verifier_md_path),
    ]
    verifier_started_ns = time.time_ns()
    with verifier_stdout_path.open("wb") as verifier_stdout, verifier_stderr_path.open("wb") as verifier_stderr:
        verifier = subprocess.run(
            verifier_command,
            cwd=decompiler.parents[2],
            env=env,
            stdout=verifier_stdout,
            stderr=verifier_stderr,
            timeout=args.timeout_seconds,
            check=False,
        )
        verifier_exit = verifier.returncode
    # The verifier is the decompiler re-run with the same quality bars, so it exits
    # non-zero for exactly the runs that fail. Raising here undid the earlier fix one
    # layer down: the solver's artifacts were kept, and then this raise threw them away
    # again before output_json was ever written, so a failing case still produced no
    # report at all. As with the solver, a verifier that produced its report is a verdict,
    # not a crash; only a verifier that produced nothing is still a hard error.
    verifier_produced_report = verifier_report_path.is_file()
    if verifier_exit != 0 and not verifier_produced_report:
        raise RuntimeError(
            f"post-hoc verifier failed: exit={verifier_exit}; "
            f"stderr={verifier_stderr_path.read_text(encoding='utf-8', errors='replace')[-4000:]}"
        )
    verifier_report = load_json(verifier_report_path, "post-hoc verifier report")
    frozen_hash_after = sha256_file(frozen_fit)

    solver_summary = dict(solver_report.get("summary", {}))
    solver_access = dict(solver_report.get("accessAudit", {}))
    solver_structure = dict(dict(solver_report.get("solver", {})).get("asymptoticStructure", {}))
    verifier_summary = dict(verifier_report.get("summary", {}))
    verifier_access = dict(verifier_report.get("accessAudit", {}))
    source_alignment = dict(verifier_summary.get("sourceAlignment", {}))
    metrics = dict(solver_summary.get("metrics", {}))
    failures: list[str] = []
    if recovery_failed:
        failures.append(f"solver or oracle exited non-zero: solver={solver_exit} oracle={oracle_exit}")
    if verifier_exit != 0:
        failures.append(f"post-hoc verifier exited non-zero: {verifier_exit}")
    if solver_summary.get("pass") is not True:
        failures.append("isolated solver functional gate failed")
    if solver_forbidden_flags:
        failures.append(f"solver argv contains teacher flags: {solver_forbidden_flags}")
    if solver_manifest_env:
        failures.append(f"solver env contains teacher manifest keys: {solver_manifest_env}")
    if solver_source_locator_matches:
        failures.append(f"solver argv contains source-locator hints: {solver_source_locator_matches}")
    if not solver_trace_complete:
        failures.append("solver file-access trace is unavailable or empty")
    if solver_model_access_rows:
        failures.append("solver opened model/cache/weight paths")
    if solver_control_access_rows:
        failures.append("solver opened orchestrator/oracle/verifier control files")
    if solver_access.get("teacherManifestFieldsObserved") != []:
        failures.append("solver observed teacher manifest metadata")
    if solver_access.get("truthCapabilityAvailable") is not False or solver_access.get("truthRequestCount") != 0:
        failures.append("solver had or used truth capability")
    if solver_access.get("oracleTransport") != "unix-binary-v1":
        failures.append("solver did not use the Unix binary oracle transport")
    if solver_access.get("queryTranscriptSha256") != oracle_witness.get("transcriptSha256"):
        failures.append("solver/oracle query transcript digests differ")
    if solver_summary.get("rawOracleCalls") != oracle_witness.get("queryCount"):
        failures.append("solver/oracle query counts differ")
    if oracle_witness.get("status") != "complete":
        failures.append("eval oracle transcript is incomplete")
    if solver_access.get("fitSnapshotUsedBySolver") is not False:
        failures.append("solver used an input fit snapshot")
    if solver_access.get("nonlinearInitialFitUsed") is not False:
        failures.append("solver used a nonlinear initial fit")
    if solver_structure.get("coldStart") is not True or solver_structure.get("stitched") is not True:
        failures.append("cold-start structure was not stitched")
    if solver_structure.get("clusterCount") != atom_count or solver_structure.get("sourceFallbackAtoms") != []:
        failures.append("cold-start atom structure count/fallback invariant failed")
    if frozen_hash_before != frozen_hash_after:
        failures.append("frozen recovery changed during post-hoc verification")
    if solver_finished_ns is None or verifier_started_ns <= solver_finished_ns:
        failures.append("post-hoc verifier did not start after solver exit")
    if verifier_summary.get("pass") is not True:
        failures.append("post-hoc verifier gate failed")
    if float(source_alignment.get("subspaceMinPrincipalCosine", 0.0)) < 0.999:
        failures.append("post-hoc subspace cosine below 0.999")
    if float(source_alignment.get("pairedAtomMinJointCosine", 0.0)) < 0.99:
        failures.append("post-hoc paired atom minimum below 0.99")
    if int(source_alignment.get("pairedAtomCountAbove099", 0)) != atom_count:
        failures.append(f"post-hoc paired atom count is not {atom_count}/{atom_count}")
    if verifier_access.get("truthUnlockStage") != "final-validation-recovery-frozen":
        failures.append("verifier truth was not unlocked after frozen validation")
    if verifier_access.get("truthRequestCount") != 1:
        failures.append("verifier truth request count is not exactly one")

    final_report = {
        "schemaVersion": 1,
        "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generatedAtNs": time.time_ns(),
        "runId": run_id,
        "kind": "isolated-blind-swiglu-cold-start-court",
        "status": "pass" if not failures else "fail",
        "summary": {
            "pass": not failures,
            "dimension": dimension,
            "unitCount": atom_count,
            "quantizationStep": quantization_step,
            "r2": metrics.get("r2"),
            "mae": metrics.get("mae"),
            "worstError": metrics.get("worstError"),
            "subspaceMinPrincipalCosine": source_alignment.get("subspaceMinPrincipalCosine"),
            "pairedAtomMinJointCosine": source_alignment.get("pairedAtomMinJointCosine"),
            "pairedAtomCountAbove099": source_alignment.get("pairedAtomCountAbove099"),
            "rawOracleCalls": solver_summary.get("rawOracleCalls"),
            "maxOracleCalls": max_oracle_calls,
            "failures": failures,
        },
        "processIsolation": {
            "orchestratorOwnsTeacherManifest": True,
            "oracleReadyMode": "opaque",
            "oracleProtocol": "gcl-swiglu-eval-v1",
            "solverArgv": solver_command,
            "solverEnvironmentKeys": sorted(solver_env),
            "solverForbiddenTeacherFlags": solver_forbidden_flags,
            "solverManifestEnvKeys": solver_manifest_env,
            "solverSourceLocatorMatches": solver_source_locator_matches,
            "solverFinishedAtNs": solver_finished_ns,
            "postHocVerifierStartedAtNs": verifier_started_ns,
            "postHocStartedAfterSolverExit": bool(verifier_started_ns > solver_finished_ns),
            "solverTruthRequestCount": solver_access.get("truthRequestCount"),
            "verifierTruthRequestCount": verifier_access.get("truthRequestCount"),
            "solverQueryTranscriptSha256": solver_access.get("queryTranscriptSha256"),
            "oracleQueryTranscriptSha256": oracle_witness.get("transcriptSha256"),
            "queryTranscriptMatch": solver_access.get("queryTranscriptSha256") == oracle_witness.get("transcriptSha256"),
            "solverFileAccessAudit": "strace-open-stat",
            "solverFileAccessAuditComplete": solver_trace_complete,
            "solverFileAccessTraceSha256": solver_trace_sha256,
            "solverFileAccessTraceLineCount": solver_file_access["lineCount"],
            "solverPackageMetadataAccessCount": len(solver_file_access["packageMetadataRows"]),
            "solverModelAccessMatches": solver_model_access_rows,
            "solverControlFileAccessMatches": solver_control_access_rows,
        },
        "frozenRecovery": {
            "path": frozen_fit.name,
            "sha256BeforeVerification": frozen_hash_before,
            "sha256AfterVerification": frozen_hash_after,
            "immutableAcrossVerification": frozen_hash_before == frozen_hash_after,
        },
        "postHocWitness": {
            "model": model,
            "layer": layer,
            "units": units,
            "outputChannel": None if output_channel < 0 else output_channel,
            "oracleMode": "selected-unit-sum" if output_channel < 0 else "down-proj-output-channel",
        },
        "accessAudit": {
            "solverAccess": "opaque-eval-only-oracle",
            "weightsUsedBySolver": bool(solver_model_access_rows),
            "hooksUsedBySolver": False,
            "sourceIdsUsedBySolver": bool(
                solver_forbidden_flags
                or solver_manifest_env
                or solver_control_access_rows
                or solver_access.get("teacherManifestFieldsObserved")
            ),
            "fitSnapshotUsedBySolver": False,
            "nonlinearInitialFitUsed": False,
            "truthCapabilityAvailableToSolver": False,
            "postHocTruthAfterSolverExit": True,
        },
        "provenance": {
            "decompilerSha256": sha256_file(decompiler),
            "orchestratorSha256": sha256_file(Path(__file__).resolve()),
            "oracleServerSha256": sha256_file(oracle_server),
            "solverWorkerSha256": sha256_file(solver_worker),
            "protocolSha256": sha256_file(protocol_module),
            "startedAtNs": started_ns,
            "solverExit": solver_exit,
            "oracleExit": oracle_exit,
            "verifierExit": verifier_exit,
        },
        "artifacts": {
            "solverReport": solver_report_path.name,
            "postHocVerifierReport": verifier_report_path.name,
            "solverStdout": solver_stdout_path.name,
            "solverStderr": solver_stderr_path.name,
            "solverFileAccessTrace": solver_strace_path.name,
            "oracleStderr": oracle_stderr_path.name,
            "oracleWitness": oracle_witness_path.name,
        },
    }
    output_json.write_text(json.dumps(final_report, indent=2) + "\n", encoding="utf-8")
    output_md.write_text(
        "\n".join([
            "# Isolated Blind SwiGLU Cold Start",
            "",
            f"- status: `{'pass' if not failures else 'fail'}`",
            f"- run id: `{run_id}`",
            f"- R2: `{metrics.get('r2')}`",
            f"- paired atom minimum: `{source_alignment.get('pairedAtomMinJointCosine')}`",
            f"- paired atoms >= 0.99: `{source_alignment.get('pairedAtomCountAbove099')}/{atom_count}`",
            f"- raw oracle calls: `{solver_summary.get('rawOracleCalls')}/{max_oracle_calls}`",
            # These three were string literals, printed unchanged on every run. A reader
            # of this markdown saw three isolation invariants presented as findings when
            # they were constants: had the solver observed manifest fields or requested
            # truth, the bullets would still have asserted the clean values. They now
            # interpolate the same measured values the pass gate above checks.
            f"- solver teacher-manifest fields: `{solver_access.get('teacherManifestFieldsObserved')}`",
            f"- solver truth capability: `{str(bool(solver_access.get('truthCapabilityAvailableToSolver'))).lower()}`",
            f"- post-hoc verifier starts after solver exit: `{str(bool(verifier_started_ns > solver_finished_ns)).lower()}`",
            "",
        ]),
        encoding="utf-8",
    )
    shutil.rmtree(solver_work_dir, ignore_errors=True)
    print(json.dumps(final_report["summary"], indent=2))
    return 0 if not failures else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Strict process-isolated real-model SwiGLU cold-start court")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--layer", type=int, default=LAYER)
    parser.add_argument("--units", default=",".join(str(value) for value in UNITS))
    parser.add_argument("--output-channel", type=int, default=OUTPUT_CHANNEL)
    parser.add_argument("--dimension", type=int, default=DIMENSION)
    parser.add_argument("--atom-count", type=int, default=ATOM_COUNT)
    parser.add_argument("--active-rank", type=int, default=ACTIVE_RANK)
    parser.add_argument("--max-oracle-calls", type=int, default=MAX_ORACLE_CALLS)
    parser.add_argument("--quantization-step", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=260723)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    return run_isolated(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
