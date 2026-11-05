#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SUBJECTS = ("France", "Germany", "Italy", "Portugal", "Greece")
SOURCE_REPORT = (
    ROOT
    / "benchmarks/release/nn-live-gpt2-ordered-residual-path-circuit-finder"
    / "gpt2-50-facts-ordered-residual-path/report.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _close(actual: Any, expected: Any, tolerance: float = 2e-4) -> bool:
    return _finite(actual) and _finite(expected) and math.isclose(
        float(actual), float(expected), rel_tol=tolerance, abs_tol=tolerance
    )


def _case_errors(
    case: Dict[str, Any],
    source_case: Dict[str, Any],
    arrays: Any,
    layers: int,
    hidden: int,
    heads: int,
    inner: int,
    vocab: int,
    tokenizer: Any | None = None,
) -> List[str]:
    errors = []
    subject = case.get("subject", "<unknown>")
    prefix = f"{subject}: "
    if case.get("prompt") != source_case.get("prompt"):
        errors.append(prefix + "prompt differs from the immutable source artifact")
    if case.get("answer") != source_case.get("answer"):
        errors.append(prefix + "answer differs from the immutable source artifact")
    if case.get("sourcePath", {}).get("attention") != source_case.get("orderedResidualPath", {}).get("nodes", {}).get("attention"):
        errors.append(prefix + "selected attention path differs from source artifact")
    if case.get("sourcePath", {}).get("mlp") != source_case.get("orderedResidualPath", {}).get("nodes", {}).get("mlp"):
        errors.append(prefix + "selected MLP path differs from source artifact")
    if case.get("sourcePath", {}).get("sourceControls") != _source_control_rows(source_case):
        errors.append(prefix + "selected source matched controls differ from source artifact")

    tokenization = case.get("tokenization", {})
    ids = tokenization.get("tokenIds")
    if not isinstance(ids, list) or not ids or not all(isinstance(item, int) for item in ids):
        errors.append(prefix + "raw input token ids are missing")
        return errors
    if hashlib.sha256(str(case.get("prompt", "")).encode("utf-8")).hexdigest() != case.get("promptSha256"):
        errors.append(prefix + "prompt hash does not match the recorded exact prompt")
    if tokenizer is not None:
        encoded = tokenizer(
            case["prompt"],
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
        observed_ids = [int(token) for token in encoded["input_ids"]]
        observed_offsets = [[int(start), int(end)] for start, end in encoded["offset_mapping"]]
        observed_tokens = [tokenizer.convert_ids_to_tokens(token) for token in observed_ids]
        if observed_ids != ids:
            errors.append(prefix + "token IDs do not reproduce from the exact prompt and local tokenizer")
        if observed_offsets != tokenization.get("characterOffsets"):
            errors.append(prefix + "token character offsets do not reproduce from the exact prompt")
        if observed_tokens != tokenization.get("tokens"):
            errors.append(prefix + "token strings do not reproduce from the exact prompt")
    seq_len = len(ids)
    read_index = tokenization.get("readTokenIndex")
    if read_index != source_case.get("readTokenIndex") or read_index != seq_len - 1:
        errors.append(prefix + "read token index is inconsistent with source/input")
    target = case.get("target", {})
    source_target = source_case.get("target", {})
    target_id = target.get("tokenId")
    if target_id != source_target.get("tokenId") or target.get("token") != source_target.get("token"):
        errors.append(prefix + "target token identity differs from source artifact")
    if target.get("sourceTargetRank") != source_target.get("baselineRank"):
        errors.append(prefix + "source target rank was not preserved")

    archive_prefix = case.get("rawArchivePrefix")
    if not isinstance(archive_prefix, str):
        errors.append(prefix + "raw tensor archive prefix is missing")
        return errors
    required = {
        "token_ids": (seq_len,),
        "position_ids": (seq_len,),
        "input_offsets": (seq_len, 2),
        "token_embedding": (seq_len, hidden),
        "position_embedding": (seq_len, hidden),
        "embedding_output": (seq_len, hidden),
        "block_input": (layers, seq_len, hidden),
        "block_output": (layers, seq_len, hidden),
        "attention_residual_write": (layers, seq_len, hidden),
        "attention_context_by_head": (layers, seq_len, heads, hidden // heads),
        "attention_head_residual_write": (layers, seq_len, heads, hidden),
        "attention_projection_bias": (layers, hidden),
        "mlp_input": (layers, seq_len, hidden),
        "mlp_preactivation": (layers, seq_len, inner),
        "mlp_activation": (layers, seq_len, inner),
        "mlp_residual_write": (layers, seq_len, hidden),
        "mlp_neuron_first_order_target_logit_attribution": (layers, inner),
        "mlp_neuron_residual_write_norm": (layers, inner),
        "mlp_projection_bias": (layers, hidden),
        "attention_head_first_order_target_logit_attribution": (layers, heads),
        "residual_target_logit_gradient": (layers, hidden),
        "attention_output_target_logit_gradient": (layers, hidden),
        "mlp_output_target_logit_gradient": (layers, hidden),
        "final_norm_input": (seq_len, hidden),
        "final_norm_output": (seq_len, hidden),
        "lm_head_logits": (seq_len, vocab),
        "target_logits": (vocab,),
    }
    loaded: Dict[str, np.ndarray] = {}
    for suffix, shape in required.items():
        key = f"{archive_prefix}_{suffix}"
        if key not in arrays:
            errors.append(prefix + f"raw array {suffix} is missing")
            continue
        value = arrays[key]
        if tuple(value.shape) != shape:
            errors.append(prefix + f"raw array {suffix} has shape {value.shape}, expected {shape}")
        if not np.isfinite(value).all():
            errors.append(prefix + f"raw array {suffix} contains non-finite values")
        loaded[suffix] = value
    if len(loaded) != len(required):
        return errors
    if not np.array_equal(loaded["token_ids"], np.asarray(ids, dtype=np.int64)):
        errors.append(prefix + "archived token ids differ from the case tokenization record")
    if not np.array_equal(loaded["position_ids"], np.arange(seq_len, dtype=np.int64)):
        errors.append(prefix + "archived position ids are not the recorded left-to-right positions")
    offsets = tokenization.get("characterOffsets")
    if not isinstance(offsets, list) or not np.array_equal(
        loaded["input_offsets"], np.asarray(offsets, dtype=np.int64)
    ):
        errors.append(prefix + "archived token offsets differ from the case tokenization record")
    embedding_error = float(np.max(np.abs(
        loaded["embedding_output"]
        - loaded["token_embedding"]
        - loaded["position_embedding"]
    )))
    if embedding_error > 2e-4:
        errors.append(prefix + f"embedding output does not equal token plus position embeddings ({embedding_error})")

    layer_trace = case.get("layerTrace")
    if not isinstance(layer_trace, list) or len(layer_trace) != layers:
        errors.append(prefix + "layerwise residual trace is incomplete")
        return errors
    for layer, entry in enumerate(layer_trace):
        residual = (
            loaded["block_input"][layer]
            + loaded["attention_residual_write"][layer]
            + loaded["mlp_residual_write"][layer]
        )
        residual_error = float(np.max(np.abs(loaded["block_output"][layer] - residual)))
        if residual_error > 2e-4:
            errors.append(prefix + f"layer {layer} residual addition does not reconstruct (max error {residual_error})")
        if not _close(entry.get("residualAddMaxAbsReconstructionError"), residual_error):
            errors.append(prefix + f"layer {layer} reported residual reconstruction error is inconsistent")
        for field, measured in (
            ("residualInputL2", np.linalg.norm(loaded["block_input"][layer, read_index])),
            ("attentionResidualWriteL2", np.linalg.norm(loaded["attention_residual_write"][layer, read_index])),
            ("mlpResidualWriteL2", np.linalg.norm(loaded["mlp_residual_write"][layer, read_index])),
            ("residualOutputL2", np.linalg.norm(loaded["block_output"][layer, read_index])),
            (
                "attentionFirstOrderTargetLogitAttribution",
                np.dot(
                    loaded["attention_residual_write"][layer, read_index],
                    loaded["attention_output_target_logit_gradient"][layer],
                ),
            ),
            (
                "mlpFirstOrderTargetLogitAttribution",
                np.dot(
                    loaded["mlp_residual_write"][layer, read_index],
                    loaded["mlp_output_target_logit_gradient"][layer],
                ),
            ),
        ):
            if not _close(entry.get(field), float(measured)):
                errors.append(prefix + f"layer {layer} trace value {field} differs from raw tensors")
        for field, expected_attribution in (
            (
                "attentionBiasFirstOrderTargetLogitAttribution",
                np.dot(
                    loaded["attention_projection_bias"][layer],
                    loaded["attention_output_target_logit_gradient"][layer],
                ),
            ),
            (
                "mlpBiasFirstOrderTargetLogitAttribution",
                np.dot(
                    loaded["mlp_projection_bias"][layer],
                    loaded["mlp_output_target_logit_gradient"][layer],
                ),
            ),
        ):
            if not _close(entry.get(field), float(expected_attribution)):
                errors.append(prefix + f"layer {layer} trace value {field} differs from raw tensors")

        per_head = loaded["attention_head_residual_write"][layer].sum(axis=1)
        reconstructed_attention = per_head + loaded["attention_projection_bias"][layer]
        if not np.allclose(
            loaded["attention_residual_write"][layer],
            reconstructed_attention,
            rtol=2e-4,
            atol=2e-4,
        ):
            errors.append(prefix + f"layer {layer} per-head projected writes do not reconstruct attention output")

        attention_attr = float(np.sum(loaded["attention_head_first_order_target_logit_attribution"][layer]))
        head_attr_expected = np.einsum(
            "hd,d->h",
            loaded["attention_head_residual_write"][layer, read_index],
            loaded["attention_output_target_logit_gradient"][layer],
        )
        if not np.allclose(
            loaded["attention_head_first_order_target_logit_attribution"][layer],
            head_attr_expected,
            rtol=2e-4,
            atol=2e-4,
        ):
            errors.append(prefix + f"layer {layer} individual attention-head first-order scores are inconsistent")
        attention_attr += float(
            np.dot(
                loaded["attention_projection_bias"][layer],
                loaded["attention_output_target_logit_gradient"][layer],
            )
        )
        full_attention_attr = float(
            np.dot(
                loaded["attention_residual_write"][layer, read_index],
                loaded["attention_output_target_logit_gradient"][layer],
            )
        )
        if not _close(attention_attr, full_attention_attr):
            errors.append(prefix + f"layer {layer} head attributions do not reconstruct the attention first-order score")

        mlp_attr = float(np.sum(loaded["mlp_neuron_first_order_target_logit_attribution"][layer]))
        mlp_attr += float(
            np.dot(
                loaded["mlp_projection_bias"][layer],
                loaded["mlp_output_target_logit_gradient"][layer],
            )
        )
        full_mlp_attr = float(
            np.dot(
                loaded["mlp_residual_write"][layer, read_index],
                loaded["mlp_output_target_logit_gradient"][layer],
            )
        )
        if not _close(mlp_attr, full_mlp_attr):
            errors.append(prefix + f"layer {layer} neuron attributions do not reconstruct the MLP first-order score")
        gelu_input = loaded["mlp_preactivation"][layer]
        expected_activation = 0.5 * gelu_input * (
            1.0
            + np.tanh(
                math.sqrt(2.0 / math.pi)
                * (gelu_input + 0.044715 * np.power(gelu_input, 3))
            )
        )
        if not np.allclose(
            loaded["mlp_activation"][layer],
            expected_activation,
            rtol=2e-4,
            atol=2e-4,
        ):
            errors.append(prefix + f"layer {layer} MLP activation is inconsistent with GPT-2 GELU")
    if not np.allclose(
        loaded["block_input"][1:],
        loaded["block_output"][:-1],
        rtol=1e-5,
        atol=1e-5,
    ):
        errors.append(prefix + "residual stream does not continue between adjacent transformer layers")
    if not np.allclose(
        loaded["final_norm_input"],
        loaded["block_output"][-1],
        rtol=1e-5,
        atol=1e-5,
    ):
        errors.append(prefix + "final normalization input differs from final residual stream")
    logits = loaded["target_logits"]
    if target_id is None or not isinstance(target_id, int) or target_id < 0 or target_id >= logits.shape[0]:
        errors.append(prefix + "target token id is invalid for the captured vocabulary")
        return errors
    if not np.allclose(logits, loaded["lm_head_logits"][read_index], rtol=1e-5, atol=1e-5):
        errors.append(prefix + "full target-position logits do not match full-sequence lm_head capture")
    baseline = case.get("baseline", {})
    baseline_target = baseline.get("target", {})
    target_logit = float(logits[target_id])
    target_probability = float(_numpy_softmax(logits)[target_id])
    rank = int(np.sum(logits > target_logit)) + 1
    if not _close(baseline_target.get("logit"), target_logit):
        errors.append(prefix + "reported target logit does not match raw full-vocabulary logits")
    if not _close(baseline_target.get("probability"), target_probability, tolerance=1e-5):
        errors.append(prefix + "reported target probability does not match raw logits")
    if baseline_target.get("rank") != rank:
        errors.append(prefix + "reported target rank does not match raw logits")
    competitors = baseline.get("topTokens")
    if not isinstance(competitors, list) or len(competitors) != 8:
        errors.append(prefix + "top-eight competing predictions are missing")
    else:
        for item in competitors:
            token_index = item.get("tokenId")
            if not isinstance(token_index, int) or token_index < 0 or token_index >= vocab:
                errors.append(prefix + "top-token list contains an invalid token id")
                continue
            expected_probability = float(_numpy_softmax(logits)[token_index])
            if not _close(item.get("logit"), float(logits[token_index])):
                errors.append(prefix + "top-token logit differs from raw full-vocabulary logits")
            if not _close(item.get("probability"), expected_probability, tolerance=1e-5):
                errors.append(prefix + "top-token probability differs from raw full-vocabulary logits")
    if case.get("maxResidualAddReconstructionError", math.inf) > 2e-4:
        errors.append(prefix + "residual stream cannot be reconstructed at the reported tolerance")

    interventions = case.get("causalInterventions", {})
    expected_groups = ("selectedAttention", "selectedMlp", "selectedPath")
    for group in expected_groups:
        pair = interventions.get(group, {})
        for mode in ("remove", "restore"):
            record = pair.get(mode, {})
            if not _finite(record.get("targetLogit")) or not _finite(record.get("targetProbability")):
                errors.append(prefix + f"{group} {mode} did not record target score and probability")
            elif mode == "remove" and not _close(
                record.get("targetLogitDrop"), target_logit - float(record["targetLogit"])
            ):
                errors.append(prefix + f"{group} removal logit delta is inconsistent")
            elif mode == "remove" and not _close(
                record.get("targetProbabilityChange"),
                float(record["targetProbability"]) - target_probability,
                tolerance=1e-5,
            ):
                errors.append(prefix + f"{group} removal probability delta is inconsistent")
            elif mode == "restore" and not _close(record.get("targetLogit"), target_logit):
                errors.append(prefix + f"{group} restore did not recover baseline target logit")
            elif mode == "restore" and not _close(
                record.get("targetProbability"), target_probability, tolerance=1e-5
            ):
                errors.append(prefix + f"{group} restore did not recover baseline target probability")
    for group in ("neighboringLayers", "sourceControls"):
        pairs = interventions.get(group)
        if not isinstance(pairs, list) or not pairs:
            errors.append(prefix + f"{group} intervention controls are missing")
            continue
        for pair in pairs:
            removed = pair.get("remove", {})
            restored = pair.get("restore", {})
            if not _finite(removed.get("targetLogitDrop")) or not _finite(removed.get("targetProbability")):
                errors.append(prefix + f"{group} control lacks removal measurements")
            if not _close(restored.get("targetLogit"), target_logit):
                errors.append(prefix + f"{group} control restoration does not recover the baseline")

    explanations = case.get("explanationMarkdown", "")
    for section in ("**WHAT?**", "**HOW?**", "**WHY?**", "**LIMITATIONS?**"):
        if section not in explanations:
            errors.append(prefix + f"case explanation is missing section {section}")
    controls = interventions.get("sourceControls", [])
    if len(controls) >= 2:
        path_drop = interventions.get("selectedPath", {}).get("remove", {}).get("targetLogitDrop")
        random_head = controls[0]
        random_neuron = controls[1]
        if random_head.get("label") not in explanations:
            errors.append(prefix + "case explanation omits the selected-MLP/random-head control label")
        if random_neuron.get("label") not in explanations:
            errors.append(prefix + "case explanation omits the random-neuron/selected-attention control label")
        head_relation = "exceeded" if path_drop > random_head["remove"]["targetLogitDrop"] else "did not exceed"
        neuron_relation = (
            "exceeded"
            if random_neuron["remove"]["targetLogitDrop"] > path_drop
            else "did not exceed"
        )
        if f"selected path {head_relation} its source-recorded" not in explanations:
            errors.append(prefix + "case explanation misstates the selected-path/random-head comparison")
        if f"it {neuron_relation} the selected path effect" not in explanations:
            errors.append(prefix + "case explanation misstates the random-neuron comparison")
    return errors


def _source_control_rows(source_case: Dict[str, Any]) -> List[Dict[str, Any]]:
    source_controls = source_case.get("orderedResidualPath", {}).get("controlPaths", {}).get("items", [])
    wanted = (
        "selected-mlp-plus-random-attention-head",
        "random-active-same-amplitude-plus-selected-attention",
    )
    rows = []
    for kind in wanted:
        match = next((item for item in source_controls if item.get("kind") == kind), None)
        if match is not None:
            rows.append(match)
    return rows


def _numpy_softmax(values: np.ndarray) -> np.ndarray:
    shifted = values.astype(np.float64) - float(np.max(values))
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials)


def verify_report(report_path: Path, research_report_path: Path | None = None) -> List[str]:
    errors = []
    report_path = report_path.resolve()
    if not report_path.is_file():
        return [f"report file does not exist: {report_path}"]
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return [f"report JSON cannot be read: {error}"]

    if report.get("kind") != "gpt2-instrumented-internal-computation-trace":
        errors.append("report kind is not the GPT-2 internal computation trace")
    if report.get("synthetic") is not False or report.get("surrogate") is not False:
        errors.append("report must be a real model run, not synthetic or surrogate")
    if report.get("accessMode") != "instrumented-witness":
        errors.append("report does not explicitly identify instrumented witness access")
    model = report.get("model", {})
    if model.get("class") != "GPT2LMHeadModel" or model.get("id") != "gpt2":
        errors.append("report model identity is not the requested local GPT-2")
    if model.get("localFilesOnly") is not True:
        errors.append("model was not explicitly loaded local-files-only")
    if model.get("device") not in ("cuda", "cpu"):
        errors.append("model device is not recorded")
    model_config = model.get("config", {})
    if (
        model_config.get("n_layer") != 12
        or model_config.get("n_head") != 12
        or model_config.get("n_embd") != 768
        or model_config.get("vocab_size") != 50257
    ):
        errors.append("report does not contain the expected GPT-2 small configuration")
    if model.get("device") == "cuda" and not report.get("runtime", {}).get("gpu"):
        errors.append("CUDA model run lacks GPU identity and memory measurements")
    if model.get("device") == "cpu" and report.get("runtime", {}).get("cudaWasAvailable") is True:
        errors.append("CPU execution was selected even though CUDA was available")

    source = report.get("sourceArtifact", {})
    source_path = (ROOT / source.get("path", "")).resolve()
    if source_path != SOURCE_REPORT.resolve():
        errors.append("report does not name the required original ordered-path source report")
    else:
        if not source_path.is_file():
            errors.append("required original ordered-path source report is missing")
        elif source.get("sha256") != _sha256(source_path):
            errors.append("source artifact digest does not match the current original source file")

    source_report = {}
    if SOURCE_REPORT.is_file():
        try:
            source_report = json.loads(SOURCE_REPORT.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            errors.append("cannot parse original source artifact")
    source_cases = {
        case["subject"]: case
        for case in source_report.get("cases", [])
        if case.get("subject") in SUBJECTS and case.get("status") == "passed"
    }
    cases = report.get("cases")
    if not isinstance(cases, list) or [case.get("subject") for case in cases] != list(SUBJECTS):
        errors.append("report does not contain exactly the five required subjects in canonical order")
        cases = cases if isinstance(cases, list) else []
    tokenizer = None
    if len(cases) == len(SUBJECTS) and model.get("id") == "gpt2":
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
        except (ImportError, OSError, RuntimeError) as error:
            errors.append(f"local GPT-2 tokenizer cannot independently reproduce prompt encodings: {error}")
    archive_meta = report.get("rawTensorArchive", {})
    archive_path = (report_path.parent / archive_meta.get("file", "")).resolve()
    if not archive_path.is_relative_to(report_path.parent.resolve()):
        errors.append("raw tensor archive path escapes report directory")
    elif not archive_path.is_file():
        errors.append("raw tensor archive is missing")
    elif archive_meta.get("sha256") != _sha256(archive_path):
        errors.append("raw tensor archive SHA-256 does not match report")
    else:
        try:
            with np.load(archive_path, allow_pickle=False) as arrays:
                expected_arrays = archive_meta.get("arrays", {})
                if set(arrays.files) != set(expected_arrays):
                    errors.append("raw archive array names differ from its report inventory")
                for name, metadata in expected_arrays.items():
                    if name in arrays and (
                        list(arrays[name].shape) != metadata.get("shape")
                        or str(arrays[name].dtype) != metadata.get("dtype")
                    ):
                        errors.append(f"raw array metadata mismatch for {name}")
                layers = int(model_config.get("n_layer", 0))
                if layers != 12:
                    errors.append("GPT-2 config does not report the expected 12 transformer layers")
                hidden = int(model_config.get("n_embd", 0))
                heads = int(model_config.get("n_head", 0))
                inner = int(model_config.get("n_inner") or hidden * 4)
                vocab = int(model_config.get("vocab_size", 0))
                for case in cases:
                    source_case = source_cases.get(case.get("subject"))
                    if source_case is None:
                        errors.append(f"{case.get('subject')}: no passed source case exists")
                        continue
                    errors.extend(
                        _case_errors(
                            case,
                            source_case,
                            arrays,
                            layers,
                            hidden,
                            heads,
                            inner,
                            vocab,
                            tokenizer,
                        )
                    )
        except (OSError, ValueError, KeyError) as error:
            errors.append(f"raw tensor archive cannot be independently checked: {error}")

    capability = report.get("capabilityAssessment", {})
    required_capabilities = {
        "internalAccess": "measured",
        "computationalTrace": "layerwise-quantitative",
        "causalValidation": "measured-interventions-with-restoration-neighbor-and-source-controls",
        "explanatoryOutput": "five-evidence-grounded-case-explanations",
        "reproducibility": "source-hash-prompts-token-ids-config-raw-tensors-and-intervention-outputs-recorded",
    }
    for key, expected in required_capabilities.items():
        if capability.get(key) != expected:
            errors.append(f"capability assessment {key} is not complete")
    coverage = report.get("captureCoverage", {})
    for key in (
        "tokenAndPositionEmbeddings",
        "residualStreamInputAndOutputEveryLayer",
        "attentionBlockOutputEveryLayer",
        "attentionHeadProjectedResidualWritesEveryLayer",
        "mlpInputPreactivationAndActivationEveryLayer",
        "mlpNeuronFirstOrderContributionsAtReadPosition",
        "finalNormalizationAndFullUnembeddingLogits",
        "targetAndTopCompetingTokens",
    ):
        if coverage.get(key) is not True:
            errors.append(f"capture coverage {key} is not established")
    claim = report.get("claimBoundary", {})
    if not isinstance(claim.get("proves"), str) or not isinstance(claim.get("doesNotEstablish"), str):
        errors.append("report is missing an explicit positive or negative claim boundary")

    if research_report_path is not None:
        if not research_report_path.is_file():
            errors.append("research report is missing")
        else:
            research_text = research_report_path.read_text(encoding="utf-8")
            for phrase in (
                "VERIFIED",
                "NOT ESTABLISHED",
                "instrumented-witness",
                "causally",
                "not the greedy next token",
            ):
                if phrase.casefold() not in research_text.casefold():
                    errors.append(f"research report omits required claim boundary phrase: {phrase}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Independently verify GPT-2 internal trace evidence and claim boundaries.")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--research-report", type=Path)
    args = parser.parse_args()
    errors = verify_report(args.report, args.research_report)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    if args.research_report:
        print("GPT2_INTERNAL_TRACE_CLAIM_BOUNDARY_VERIFIED")
    else:
        print("GPT2_INTERNAL_TRACE_ARTIFACT_VERIFIED")


if __name__ == "__main__":
    main()
