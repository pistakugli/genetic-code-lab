from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from llm_reader.gpt2_counterfactual_transfer import (  # noqa: E402
    _bootstrap_family_statistics,
    _install_patch_hooks,
    _make_transfer_group,
    _pair_alignment,
    _sample_control_groups,
    _shared_token_mapping,
    _tokenize_prompt,
    build_pair_inventory,
)
from test_gpt2_internal_trace import TinyGPT2  # noqa: E402
from llm_reader.gpt2_internal_trace import GPT2TraceCapture  # noqa: E402
from llm_reader.verify_gpt2_counterfactual_transfer import validate_report_structure  # noqa: E402


class WordTokenizer:
    def __init__(self) -> None:
        self._ids: dict[str, int] = {}
        self._tokens: dict[int, str] = {}

    def __call__(
        self,
        text: str,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
    ) -> dict[str, list[int] | list[tuple[int, int]]]:
        del add_special_tokens
        matches = list(re.finditer(r"\S+", text))
        ids = []
        offsets = []
        for match in matches:
            token = match.group()
            token_id = self._ids.setdefault(token, len(self._ids) + 1)
            self._tokens[token_id] = token
            ids.append(token_id)
            offsets.append((match.start(), match.end()))
        if return_offsets_mapping:
            return {"input_ids": ids, "offset_mapping": offsets}
        return {"input_ids": ids}

    def decode(self, token_ids: list[int]) -> str:
        return " " + self._tokens[int(token_ids[0])]


def _case(subject: str, prompt: str, answer: str, tokenizer: WordTokenizer) -> dict:
    target_id = int(tokenizer(" " + answer)["input_ids"][0])
    read_index = len(tokenizer(prompt)["input_ids"]) - 1
    return {
        "status": "passed",
        "subject": subject,
        "prompt": prompt,
        "answer": answer,
        "target": {"tokenId": target_id, "token": tokenizer.decode([target_id])},
        "readTokenIndex": read_index,
        "orderedResidualPath": {
            "nodes": {
                "attention": [{"layer": 0, "head": 0, "ablateTokenPositions": [read_index]}],
                "mlp": [{"layer": 1, "neuron": 1, "ablateTokenPositions": [read_index]}],
            }
        },
    }


class GPT2CounterfactualTransferTests(unittest.TestCase):
    def test_inventory_uses_only_exactly_aligned_source_facts(self) -> None:
        tokenizer = WordTokenizer()
        cases = [
            _case("France", "The capital of France is", "Paris", tokenizer),
            _case("South Korea", "The capital of South Korea is", "Seoul", tokenizer),
        ]
        inventory = build_pair_inventory({"config": {"caseCount": 2}, "cases": cases}, tokenizer)
        self.assertEqual(inventory["inventory"]["eligibleUnorderedPairCount"], 1)
        self.assertEqual(inventory["inventory"]["excludedFactCount"], 0)
        pair = inventory["pairs"][0]
        self.assertEqual(pair["relationFamily"], "country-capital")
        self.assertEqual(pair["alignment"]["aToBPositionMap"]["attention:0:4"], 5)
        self.assertEqual(pair["alignment"]["bToAPositionMap"]["mlp:0:5"], 4)

    def test_alignment_rejects_a_path_position_inside_the_fact_slot(self) -> None:
        source = {
            "ids": [1, 2, 3, 4, 5],
            "offsets": [[0, 3], [4, 11], [12, 14], [15, 21], [22, 24]],
        }
        destination = {
            "ids": [1, 2, 3, 6, 7, 5],
            "offsets": [[0, 3], [4, 11], [12, 14], [15, 19], [20, 25], [26, 28]],
        }
        position, error = _shared_token_mapping(source, (15, 21), destination, (15, 25), 3)
        self.assertIsNone(position)
        self.assertIn("variable fact slot", str(error))

    def test_bidirectional_alignment_rejects_changed_static_token_ids(self) -> None:
        tokenizer = WordTokenizer()
        case_a = _case("France", "The capital of France is", "Paris", tokenizer)
        case_b = _case("Italy", "The capital of Italy was", "Rome", tokenizer)
        tokenized = {
            "France": _tokenize_prompt(case_a, tokenizer),
            "Italy": _tokenize_prompt(case_b, tokenizer),
        }
        slots = {
            "France": ("country-capital", "France", (15, 21)),
            "Italy": ("country-capital", "Italy", (15, 20)),
        }
        alignment, error = _pair_alignment(case_a, case_b, tokenizer, tokenized, slots)
        self.assertIsNone(alignment)
        self.assertIn("prompt skeleton mismatch", str(error))

    def test_path_transplant_and_recipient_restoration_use_exact_activations(self) -> None:
        torch.manual_seed(702)
        model = TinyGPT2().eval()
        donor_ids = torch.tensor([[2, 4, 6]])
        recipient_ids = torch.tensor([[3, 5, 7]])
        case = {
            "subject": "donor",
            "orderedResidualPath": {
                "nodes": {
                    "attention": [{"layer": 0, "head": 0, "ablateTokenPositions": [2]}],
                    "mlp": [{"layer": 1, "neuron": 1, "ablateTokenPositions": [2]}],
                }
            },
        }
        with torch.no_grad(), GPT2TraceCapture(model) as donor_capture:
            donor_logits = model(input_ids=donor_ids).logits
        donor_state = {
            "case": case,
            "captureValues": donor_capture.values,
            "headCount": 2,
        }
        recipient_case = {**case, "subject": "recipient"}
        with torch.no_grad(), GPT2TraceCapture(model) as recipient_capture:
            recipient_logits = model(input_ids=recipient_ids).logits
        recipient_state = {
            "case": recipient_case,
            "captureValues": recipient_capture.values,
            "headCount": 2,
        }
        entries = [
            {"kind": "attention", "layer": 0, "unit": 0, "sourcePositions": [2]},
            {"kind": "mlp", "layer": 1, "unit": 1, "sourcePositions": [2]},
        ]
        group = _make_transfer_group(
            donor_state,
            recipient_state,
            entries,
            {2: 2},
            model,
            "discovered-path-transplant",
        )
        self.assertTrue(torch.equal(
            group["sites"][0]["positions"][0]["donorValue"],
            donor_capture.values["attention_context"][0][0, 2, :4],
        ))
        baseline = recipient_logits[0, 2].detach()
        handles = _install_patch_hooks(model, [group], [(0, 0)])
        try:
            with torch.no_grad():
                transferred = model(input_ids=recipient_ids).logits[0, 2]
        finally:
            for handle in reversed(handles):
                handle.remove()
        self.assertGreater(float((transferred - baseline).abs().max()), 1e-8)

        restoration = _make_transfer_group(
            recipient_state,
            recipient_state,
            entries,
            {2: 2},
            model,
            "recipient-original-activation-restoration",
        )
        handles = _install_patch_hooks(model, [restoration], [(0, 0)])
        try:
            with torch.no_grad():
                restored = model(input_ids=recipient_ids).logits[0, 2]
        finally:
            for handle in reversed(handles):
                handle.remove()
        self.assertTrue(torch.allclose(restored, baseline, atol=1e-6, rtol=1e-6))

    def test_control_groups_preserve_structure_and_are_deterministic(self) -> None:
        torch.manual_seed(703)
        model = TinyGPT2().eval()
        ids = torch.tensor([[2, 4, 6]])
        case = {
            "subject": "fact",
            "orderedResidualPath": {
                "nodes": {
                    "attention": [{"layer": 0, "head": 0, "ablateTokenPositions": [2]}],
                    "mlp": [{"layer": 1, "neuron": 1, "ablateTokenPositions": [2]}],
                }
            },
        }
        with torch.no_grad(), GPT2TraceCapture(model) as capture:
            model(input_ids=ids)
        state = {"case": case, "captureValues": capture.values, "headCount": 2}
        controls, metadata = _sample_control_groups(
            state, model, {2: 2}, control_count=4, tolerance=0.2, seed=991
        )
        repeated, repeated_metadata = _sample_control_groups(
            state, model, {2: 2}, control_count=4, tolerance=0.2, seed=991
        )
        self.assertEqual(metadata, repeated_metadata)
        self.assertEqual(
            [[entry["unit"] for entry in item["entries"]] for item in controls],
            [[entry["unit"] for entry in item["entries"]] for item in repeated],
        )
        self.assertEqual(metadata["uniqueCount"], 4)
        for control in controls:
            self.assertEqual(
                [(entry["kind"], entry["layer"], entry["sourcePositions"]) for entry in control["entries"]],
                [("attention", 0, [2]), ("mlp", 1, [2])],
            )
            self.assertTrue(all(
                match["unit"] != (0 if match["componentType"] == "attention" else 1)
                for match in control["magnitudeMatches"]
            ))

    def test_invalid_activation_metadata_is_rejected(self) -> None:
        tokenizer = WordTokenizer()
        cases = [
            _case("France", "The capital of France is", "Paris", tokenizer),
            {
                **_case("Italy", "The capital of Italy is", "Rome", tokenizer),
                "orderedResidualPath": {"nodes": {"attention": [], "mlp": []}},
            },
        ]
        inventory = build_pair_inventory({"config": {"caseCount": 2}, "cases": cases}, tokenizer)
        self.assertEqual(inventory["inventory"]["eligibleUnorderedPairCount"], 0)
        self.assertEqual(inventory["inventory"]["excludedFactCount"], 2)
        reasons = [item["reason"] for item in inventory["inventory"]["excludedFacts"]]
        self.assertTrue(any("complete attention-plus-MLP" in reason for reason in reasons))

    def test_independent_verifier_rejects_wrong_report_kind(self) -> None:
        errors = validate_report_structure(
            {"kind": "unrelated-report", "schemaVersion": 1},
            SimpleNamespace(files=[]),
        )
        self.assertIn("unsupported report kind or schema version", errors)

    def test_invalid_magnitude_controls_block_specificity_inference_not_transfer_ci(self) -> None:
        records = [
            {
                "donorSubject": f"fact-{index}",
                "recipientSubject": f"fact-{(index + 1) % 10}",
                "pathTransfer": {
                    "transferContrastDeltaTowardDonor": 0.5 + index / 100,
                    "donorAnswerMeanLogLikelihoodChange": 0.2,
                },
                "meanControlTransferContrastDelta": 0.1,
                "specificityControlsValid": False,
            }
            for index in range(10)
        ]
        result = _bootstrap_family_statistics("synthetic", records, 200)
        self.assertIsNotNone(result["pathTransferContrastDelta95PercentClusterBootstrapCI"])
        self.assertIsNone(result["meanPathMinusMeanControlTransferContrastDelta"])
        self.assertIsNone(result["twoSidedCenteredClusterBootstrapPValue"])
        self.assertEqual(result["inferentialStatus"], "descriptive-only-invalid-controls")


if __name__ == "__main__":
    unittest.main()
