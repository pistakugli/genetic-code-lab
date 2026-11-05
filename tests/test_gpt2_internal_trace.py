from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from llm_reader.gpt2_internal_trace import (  # noqa: E402
    GPT2TraceCapture,
    full_distribution_change,
    interaction_from_logits,
    conv1d_head_writes,
    explain_case,
    matched_random_control_groups,
    mlp_neuron_first_order_attribution,
    ordered_source_cases,
    run_batched_control_interventions,
    run_intervention,
    summarize_logits,
)
from llm_reader.verify_gpt2_internal_trace import verify_report  # noqa: E402


class TinyConvProjection(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(width, width) * 0.05)
        self.bias = nn.Parameter(torch.zeros(width))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value @ self.weight + self.bias


class TinyAttention(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.num_heads = heads
        self.head_dim = width // heads
        self.c_proj = TinyConvProjection(width)

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, None]:
        return self.c_proj(hidden), None


class TinyMlp(nn.Module):
    def __init__(self, width: int, inner: int):
        super().__init__()
        self.c_fc = nn.Linear(width, inner)
        self.act = nn.Tanh()
        self.c_proj = nn.Linear(inner, width)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.c_proj(self.act(self.c_fc(hidden)))


class TinyBlock(nn.Module):
    def __init__(self, width: int, inner: int, heads: int):
        super().__init__()
        self.attn = TinyAttention(width, heads)
        self.mlp = TinyMlp(width, inner)

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor]:
        hidden = hidden + self.attn(hidden)[0]
        hidden = hidden + self.mlp(hidden)
        return (hidden,)


class TinyTransformer(nn.Module):
    def __init__(self, vocab: int, width: int, inner: int, heads: int, layers: int, max_positions: int):
        super().__init__()
        self.wte = nn.Embedding(vocab, width)
        self.wpe = nn.Embedding(max_positions, width)
        self.drop = nn.Dropout(0)
        self.h = nn.ModuleList([TinyBlock(width, inner, heads) for _ in range(layers)])
        self.ln_f = nn.LayerNorm(width)


class TinyGPT2(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(n_layer=2, n_head=2, n_inner=12, vocab_size=31)
        self.config.to_dict = lambda: {
            "n_layer": self.config.n_layer,
            "n_head": self.config.n_head,
            "n_inner": self.config.n_inner,
            "vocab_size": self.config.vocab_size,
        }
        self.transformer = TinyTransformer(31, 8, 12, 2, 2, 16)
        self.lm_head = nn.Linear(8, 31, bias=False)

    def forward(self, input_ids: torch.Tensor, use_cache: bool = False) -> SimpleNamespace:
        del use_cache
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        hidden = self.transformer.drop(self.transformer.wte(input_ids) + self.transformer.wpe(positions))
        for block in self.transformer.h:
            hidden = block(hidden)[0]
        logits = self.lm_head(self.transformer.ln_f(hidden))
        return SimpleNamespace(logits=logits)


class TinyTokenizer:
    def decode(self, ids: list[int]) -> str:
        return f"t{ids[0]}"


class GPT2InternalTraceTests(unittest.TestCase):
    def test_full_distribution_metrics_cover_all_logits_and_zero_for_identity(self) -> None:
        baseline = torch.tensor([2.0, 0.0, -1.0, 3.0])
        unchanged = full_distribution_change(baseline, baseline.clone())
        self.assertEqual(unchanged["totalVariation"], 0.0)
        self.assertEqual(unchanged["jensenShannonDivergence"], 0.0)

        shifted = full_distribution_change(baseline, torch.tensor([1.0, 1.0, -1.0, 3.0]))
        self.assertGreater(shifted["totalVariation"], 0.0)
        self.assertGreater(shifted["jensenShannonDivergence"], 0.0)

    def test_group_interaction_is_group_drop_minus_sum_of_single_drops(self) -> None:
        baseline = torch.tensor([10.0, 0.0])
        first = torch.tensor([8.0, 0.0])
        second = torch.tensor([7.0, 0.0])
        additive_group = torch.tensor([5.0, 0.0])
        additive = interaction_from_logits(baseline, additive_group, [first, second], 0)
        self.assertAlmostEqual(additive["logitDropNonAdditivity"], 0.0, places=6)
        self.assertAlmostEqual(additive["fullVocabularyLogitInteractionL2"], 0.0, places=6)

        synergistic = interaction_from_logits(
            baseline, torch.tensor([4.0, 0.0]), [first, second], 0
        )
        self.assertAlmostEqual(synergistic["logitDropNonAdditivity"], 1.0, places=6)

    def test_random_controls_are_deterministic_and_preserve_path_structure(self) -> None:
        torch.manual_seed(14)
        model = TinyGPT2().eval()
        ids = torch.tensor([[2, 4, 6]])
        with GPT2TraceCapture(model) as capture:
            with torch.no_grad():
                model(input_ids=ids)
            path_attention = [{"layer": 0, "head": 0, "ablateTokenPositions": [2]}]
            path_mlp = [{"layer": 1, "neuron": 3, "ablateTokenPositions": [2]}]
            first, metadata = matched_random_control_groups(
                attention_nodes=path_attention,
                mlp_nodes=path_mlp,
                model=model,
                capture=capture,
                count=12,
                tolerance=0.2,
                seed=701,
            )
            again, again_metadata = matched_random_control_groups(
                attention_nodes=path_attention,
                mlp_nodes=path_mlp,
                model=model,
                capture=capture,
                count=12,
                tolerance=0.2,
                seed=701,
            )
        self.assertEqual(first, again)
        self.assertEqual(metadata, again_metadata)
        self.assertEqual(len(first), 12)
        self.assertTrue(metadata["exactStructuralMatch"])
        observed_groups = {
            (
                tuple((node["layer"], node["head"], tuple(node["ablateTokenPositions"])) for node in control["attention"]),
                tuple((node["layer"], node["neuron"], tuple(node["ablateTokenPositions"])) for node in control["mlp"]),
            )
            for control in first
        }
        self.assertEqual(metadata["uniqueSampleCount"], len(observed_groups))
        self.assertEqual(metadata["samplingWithReplacement"], len(observed_groups) < len(first))
        for control in first:
            self.assertEqual(len(control["attention"]), len(path_attention))
            self.assertEqual(len(control["mlp"]), len(path_mlp))
            self.assertEqual(control["attention"][0]["layer"], path_attention[0]["layer"])
            self.assertEqual(control["mlp"][0]["layer"], path_mlp[0]["layer"])
            self.assertEqual(control["attention"][0]["ablateTokenPositions"], [2])
            self.assertEqual(control["mlp"][0]["ablateTokenPositions"], [2])
            self.assertNotEqual(control["attention"][0]["head"], 0)
            self.assertNotEqual(control["mlp"][0]["neuron"], 3)

    def test_batched_matched_control_uses_same_zeroing_operator_as_single_intervention(self) -> None:
        torch.manual_seed(31)
        model = TinyGPT2().eval()
        ids = torch.tensor([[2, 4, 6]])
        encoded = {"input_ids": ids}
        read_index = 2
        with torch.no_grad():
            baseline_logits = model(**encoded).logits[0, read_index]
        predicted_id = int(torch.argmax(baseline_logits))
        baseline_summary = summarize_logits(baseline_logits, predicted_id, TinyTokenizer())
        with GPT2TraceCapture(model) as capture:
            with torch.no_grad():
                model(**encoded)
            activations = {
                layer: capture.values[f"mlp_activation_{layer}"]
                for layer in range(capture.layers)
            }
            contexts = [
                capture.values["attention_context"][layer]
                for layer in range(capture.layers)
            ]
            controls, _ = matched_random_control_groups(
                attention_nodes=[{"layer": 0, "head": 0, "ablateTokenPositions": [2]}],
                mlp_nodes=[{"layer": 1, "neuron": 2, "ablateTokenPositions": [2]}],
                model=model,
                capture=capture,
                count=1,
                tolerance=1.0,
                seed=8,
            )
        batched = run_batched_control_interventions(
            model=model,
            encoded=encoded,
            read_index=read_index,
            baseline_logits=baseline_logits,
            baseline_summary=baseline_summary,
            tokenizer=TinyTokenizer(),
            controls=controls,
            batch_size=1,
        )[0]
        single = run_intervention(
            model=model,
            encoded=encoded,
            read_index=read_index,
            target_id=predicted_id,
            baseline_summary=baseline_summary,
            tokenizer=TinyTokenizer(),
            mlp_interventions=controls[0]["mlp"],
            attention_interventions=controls[0]["attention"],
            baseline_mlp_activations=activations,
            baseline_attention_context=contexts,
            mode="remove",
            baseline_logits=baseline_logits,
        )
        self.assertAlmostEqual(batched["targetLogitDrop"], single["targetLogitDrop"], places=5)
        self.assertAlmostEqual(
            batched["fullDistributionChange"]["totalVariation"],
            single["fullDistributionChange"]["totalVariation"],
            places=6,
        )
        self.assertEqual(batched["greedyPredictionAfterIntervention"]["tokenId"], single["greedyTokenAfterIntervention"]["tokenId"])

    def test_case_explanation_compares_paths_with_the_actual_control_components(self) -> None:
        case = {
            "subject": "Test",
            "prompt": "The capital is",
            "target": {"token": " X", "tokenId": 1, "rank": 1, "logit": 1.0, "probability": 0.5},
            "baseline": {
                "target": {"token": " X", "tokenId": 1, "rank": 1, "logit": 1.0, "probability": 0.5},
                "greedyNextToken": {"token": " X", "tokenId": 1, "probability": 0.5},
            },
            "sourcePath": {
                "attention": [{"head": 0, "layer": 2}],
                "mlp": [{"neuron": 3, "layer": 2}],
            },
            "selectedMlpActivation": 0.25,
            "selectedMlpFirstOrderAttribution": 0.1,
            "selectedHeadResidualWriteL2": 1.0,
            "selectedMlpNeuronResidualWriteContributionL2": 0.5,
            "maxResidualAddReconstructionError": 0.0,
            "tokenization": {"tokenIds": [5, 6]},
            "layerTrace": [{}, {}],
            "causalInterventions": {
                "selectedAttention": {"remove": {"targetLogitDrop": 0.7}},
                "selectedMlp": {"remove": {"targetLogitDrop": 0.2}},
                "selectedPath": {
                    "remove": {
                        "targetLogitDrop": 0.9,
                        "targetProbabilityChange": -0.1,
                    },
                    "restore": {"targetLogit": 1.0},
                },
                "sourceControls": [
                    {
                        "label": "selected-mlp-plus-random-attention-head",
                        "remove": {"targetLogitDrop": 0.6},
                    },
                    {
                        "label": "random-active-same-amplitude-plus-selected-attention",
                        "remove": {"targetLogitDrop": 1.2},
                    },
                ],
                "neighboringLayers": [],
            },
        }

        explanation = explain_case(case)

        self.assertIn("selected path exceeded its source-recorded", explanation)
        self.assertIn("selected-mlp-plus-random-attention-head control (0.600000)", explanation)
        self.assertIn("random-active-same-amplitude-plus-selected-attention control", explanation)
        self.assertIn("it exceeded the selected path effect", explanation)
        self.assertNotIn("selected head's removal effect", explanation)

    def test_attention_head_projection_decomposes_conv1d_inputs(self) -> None:
        torch.manual_seed(7)
        context = torch.randn(2, 3, 8)
        weight = torch.randn(8, 8)
        writes = conv1d_head_writes(context, weight, 2)

        self.assertEqual(tuple(writes.shape), (2, 3, 2, 8))
        self.assertTrue(torch.allclose(writes.sum(dim=-2), context @ weight))

    def test_neuron_first_order_attribution_uses_activation_and_projection_row(self) -> None:
        activations = torch.tensor([2.0, -1.0])
        weight = torch.tensor([[1.0, 3.0], [2.0, -1.0]])
        gradient = torch.tensor([0.5, 2.0])

        result = mlp_neuron_first_order_attribution(activations, weight, gradient)
        self.assertTrue(torch.allclose(result, torch.tensor([13.0, 1.0])))

    def test_capture_records_layer_internals_and_removes_hooks(self) -> None:
        torch.manual_seed(11)
        model = TinyGPT2().eval()
        ids = torch.tensor([[3, 5, 7]])
        with GPT2TraceCapture(model) as capture:
            output = model(input_ids=ids)
            self.assertEqual(tuple(output.logits.shape), (1, 3, 31))
            self.assertEqual(set(capture.values["block_input"]), {0, 1})
            self.assertEqual(set(capture.values["block_output"]), {0, 1})
            self.assertEqual(capture.values["mlp_activation_1"].shape, (1, 3, 12))
            self.assertEqual(capture.values["attention_context"][0].shape, (1, 3, 8))
        self.assertEqual(capture.handles, [])
        self.assertEqual(len(model.transformer.wte._forward_hooks), 0)
        self.assertEqual(len(model.transformer.h[0].attn.c_proj._forward_pre_hooks), 0)

    def test_capture_removes_hooks_after_forward_exception(self) -> None:
        model = TinyGPT2().eval()
        with self.assertRaisesRegex(RuntimeError, "synthetic stop"):
            with GPT2TraceCapture(model):
                raise RuntimeError("synthetic stop")
        self.assertEqual(len(model.transformer.wte._forward_hooks), 0)
        self.assertEqual(len(model.transformer.h[1].mlp.c_proj._forward_hooks), 0)

    def test_intervention_removal_and_restore_replay_on_model(self) -> None:
        torch.manual_seed(29)
        model = TinyGPT2().eval()
        ids = torch.tensor([[2, 4, 6]])
        encoded = {"input_ids": ids}
        target_id = 3
        read_index = 2
        with torch.no_grad():
            baseline_logits = model(**encoded).logits[0, read_index]
        baseline = summarize_logits(baseline_logits, target_id, TinyTokenizer())
        with GPT2TraceCapture(model) as capture:
            with torch.no_grad():
                model(**encoded)
            activations = {
                layer: capture.values[f"mlp_activation_{layer}"]
                for layer in range(capture.layers)
            }
            contexts = [
                capture.values["attention_context"][layer]
                for layer in range(capture.layers)
            ]
        sites = {
            "mlp": [{"layer": 0, "neuron": 1, "ablateTokenPositions": [read_index]}],
            "attention": [{"layer": 0, "head": 0, "ablateTokenPositions": [read_index]}],
        }
        removal = run_intervention(
            model=model,
            encoded=encoded,
            read_index=read_index,
            target_id=target_id,
            baseline_summary=baseline,
            tokenizer=TinyTokenizer(),
            mlp_interventions=sites["mlp"],
            attention_interventions=sites["attention"],
            baseline_mlp_activations=activations,
            baseline_attention_context=contexts,
            mode="remove",
        )
        restore = run_intervention(
            model=model,
            encoded=encoded,
            read_index=read_index,
            target_id=target_id,
            baseline_summary=baseline,
            tokenizer=TinyTokenizer(),
            mlp_interventions=sites["mlp"],
            attention_interventions=sites["attention"],
            baseline_mlp_activations=activations,
            baseline_attention_context=contexts,
            mode="restore",
        )
        self.assertAlmostEqual(restore["targetLogit"], baseline["target"]["logit"], places=6)
        self.assertAlmostEqual(restore["targetProbability"], baseline["target"]["probability"], places=6)
        self.assertAlmostEqual(
            removal["targetLogitDrop"],
            baseline["target"]["logit"] - removal["targetLogit"],
            places=6,
        )
        self.assertEqual(len(model.transformer.h[0].mlp.act._forward_hooks), 0)
        self.assertEqual(len(model.transformer.h[0].attn.c_proj._forward_pre_hooks), 0)

    def test_logit_summary_reports_target_probability_rank_and_competitors(self) -> None:
        logits = torch.tensor([1.0, 3.0, 2.0, -1.0])
        result = summarize_logits(logits, 2, TinyTokenizer(), count=3)
        self.assertEqual(result["target"]["rank"], 2)
        self.assertEqual(result["greedyNextToken"]["tokenId"], 1)
        self.assertAlmostEqual(
            result["target"]["probability"],
            torch.softmax(logits, dim=-1)[2].item(),
            places=7,
        )
        self.assertEqual(len(result["topTokens"]), 3)

    def test_source_case_selection_requires_each_exact_passed_subject(self) -> None:
        rows = [
            {"subject": subject, "status": "passed"}
            for subject in ("Greece", "France", "Italy", "Portugal", "Germany")
        ]
        ordered = ordered_source_cases({"cases": rows})
        self.assertEqual(
            [row["subject"] for row in ordered],
            ["France", "Germany", "Italy", "Portugal", "Greece"],
        )
        with self.assertRaisesRegex(ValueError, "missing required passed cases"):
            ordered_source_cases({"cases": rows[:-1]})

    def test_independent_verifier_rejects_missing_or_wrong_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "report.json"
            report_path.write_text(json.dumps({"kind": "wrong"}), encoding="utf-8")
            errors = verify_report(report_path)
        self.assertTrue(any("report kind" in error for error in errors))
        self.assertTrue(any("source report" in error for error in errors))


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(GPT2InternalTraceTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    print("GPT2_INTERNAL_TRACE_TESTS_PASSED")
