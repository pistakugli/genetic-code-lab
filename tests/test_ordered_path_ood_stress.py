from __future__ import annotations

import sys
import unittest
import importlib.util
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS_DIR))

from llm_reader.ood_paraphrases import (  # noqa: E402
    OOD_CAPITAL_PROMPT_TEMPLATES,
    ood_paraphrase_cases_for_fact,
    summarize_paraphrases_by_family,
)
from llm_reader.verify_ordered_path_ood_stress import verify_case_report  # noqa: E402


def load_stability_module():
    path = TOOLS_DIR / "llm-reader-ordered-path-stability.py"
    spec = importlib.util.spec_from_file_location("gcl_ordered_path_stability", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load stability reporter from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class OODParaphraseTests(unittest.TestCase):
    def test_prompts_are_deterministic_held_out_and_do_not_leak_answers(self) -> None:
        fact = {"subject": "France", "answer": "Paris"}
        first = ood_paraphrase_cases_for_fact(fact, len(OOD_CAPITAL_PROMPT_TEMPLATES))
        second = ood_paraphrase_cases_for_fact(fact, len(OOD_CAPITAL_PROMPT_TEMPLATES))
        standard_prompts = {
            "The capital city of France is",
            "In France, the capital is",
            "In France, the capital city is",
            "The capital of the country France is",
            "France's capital is",
            "For France, the capital is",
        }

        self.assertEqual(first, second)
        self.assertEqual(len(first), len(OOD_CAPITAL_PROMPT_TEMPLATES))
        self.assertEqual(len({case["prompt"] for case in first}), len(first))
        self.assertEqual(
            {case["family"] for case in first},
            {family for family, _ in OOD_CAPITAL_PROMPT_TEMPLATES},
        )
        for case in first:
            self.assertEqual(case["profile"], "ood")
            self.assertIn("France", case["prompt"])
            self.assertNotIn("Paris", case["prompt"])
            self.assertNotIn(case["prompt"], standard_prompts)

    def test_prompt_count_is_bounded_and_invalid_facts_are_rejected(self) -> None:
        self.assertEqual(ood_paraphrase_cases_for_fact({"subject": "Japan"}, 2)[0]["family"],
                         OOD_CAPITAL_PROMPT_TEMPLATES[0][0])
        self.assertEqual(len(ood_paraphrase_cases_for_fact({"subject": "Japan"}, 100)),
                         len(OOD_CAPITAL_PROMPT_TEMPLATES))
        self.assertEqual(ood_paraphrase_cases_for_fact({"subject": "Japan"}, 0), [])
        with self.assertRaises(ValueError):
            ood_paraphrase_cases_for_fact({"subject": "Japan"}, -1)
        with self.assertRaises(ValueError):
            ood_paraphrase_cases_for_fact({"subject": ""}, 1)

    def test_family_summary_uses_all_samples_and_keeps_families_separate(self) -> None:
        summary = summarize_paraphrases_by_family([
            {
                "family": "question",
                "pass": True,
                "ratioToOriginal": 0.8,
                "tokenCount": 8,
            },
            {
                "family": "question",
                "pass": False,
                "ratioToOriginal": 0.2,
                "tokenCount": 10,
            },
            {
                "family": "completion",
                "pass": True,
                "ratioToOriginal": 1.2,
                "tokenCount": 6,
            },
        ])

        self.assertEqual(summary["question"]["count"], 2)
        self.assertEqual(summary["question"]["passed"], 1)
        self.assertAlmostEqual(summary["question"]["passRate"], 0.5)
        self.assertAlmostEqual(summary["question"]["meanDropRatio"], 0.5)
        self.assertAlmostEqual(summary["question"]["meanTokenCount"], 9)
        self.assertEqual(summary["completion"]["count"], 1)
        with self.assertRaises(ValueError):
            summarize_paraphrases_by_family([{"pass": True, "ratioToOriginal": 1, "tokenCount": 1}])

    def test_artifact_verifier_checks_raw_prompt_family_measurements(self) -> None:
        family_names = [family for family, _ in OOD_CAPITAL_PROMPT_TEMPLATES]
        paraphrases = [
            {
                "profile": "ood",
                "family": family,
                "prompt": f"Which city is the capital of France? ({family})",
                "tokenCount": 7,
                "targetLogitDrop": 2.0,
                "ratioToOriginal": 1.0,
                "pass": True,
            }
            for family in family_names
        ]
        family_summary = {
            family: {
                "count": 1,
                "passed": 1,
                "passRate": 1.0,
                "meanDropRatio": 1.0,
                "meanTokenCount": 7.0,
            }
            for family in family_names
        }
        case_report = {
            "kind": "gpt2-live-ordered-path-ood-stress",
            "synthetic": False,
            "surrogate": False,
            "model": {"id": "gpt2", "adapter": "gpt2", "device": "cuda"},
            "config": {
                "caseCount": 1,
                "paraphraseCount": len(family_names),
                "paraphraseProfile": "ood",
                "minParaphrasePassRate": 0.66,
                "minLayerSpecificityRate": 0.8,
            },
            "summary": {
                "pass": True,
                "caseCount": 1,
                "originalReplayPassRate": 1.0,
                "paraphraseProfile": "ood",
                "paraphrasePassRate": 1.0,
                "paraphraseFamilySummary": family_summary,
                "layerSpecificityRate": 1.0,
                "maxReplayError": 0.0,
            },
            "cases": [{
                "subject": "France",
                "answer": "Paris",
                "status": "passed",
                "originalReplay": {
                    "targetLogitDrop": 2.0,
                    "absoluteReplayError": 0.0,
                    "pass": True,
                },
                "paraphrases": paraphrases,
                "layerControls": {"items": [{"shift": 1}], "pass": True},
            }],
            "claimBoundary": {
                "provesIfPasses": "Configured causal replay only.",
                "doesNotProve": "Arbitrary prompt stability.",
            },
        }

        self.assertEqual(verify_case_report(case_report, "gpt2", "gpt2"), [])
        case_report["summary"]["paraphrasePassRate"] = 0.5
        self.assertTrue(
            any(
                error.endswith("overall paraphrase pass rate does not match raw measurements")
                for error in verify_case_report(case_report, "gpt2", "gpt2")
            )
        )

    def test_artifact_verifier_accepts_recorded_cpu_device_and_rejects_unknown_devices(self) -> None:
        family_names = [family for family, _ in OOD_CAPITAL_PROMPT_TEMPLATES]
        paraphrases = [
            {
                "profile": "ood",
                "family": family,
                "prompt": f"Which city is the capital of France? ({family})",
                "tokenCount": 7,
                "targetLogitDrop": 2.0,
                "ratioToOriginal": 1.0,
                "pass": True,
            }
            for family in family_names
        ]
        family_summary = {
            family: {
                "count": 1,
                "passed": 1,
                "passRate": 1.0,
                "meanDropRatio": 1.0,
                "meanTokenCount": 7.0,
            }
            for family in family_names
        }

        def case_report_with_device(device: str):
            return {
                "kind": "gpt2-live-ordered-path-ood-stress",
                "synthetic": False,
                "surrogate": False,
                "model": {"id": "gpt2", "adapter": "gpt2", "device": device},
                "config": {
                    "caseCount": 1,
                    "paraphraseCount": len(family_names),
                    "paraphraseProfile": "ood",
                    "minParaphrasePassRate": 0.66,
                    "minLayerSpecificityRate": 0.8,
                },
                "summary": {
                    "pass": True,
                    "caseCount": 1,
                    "originalReplayPassRate": 1.0,
                    "paraphraseProfile": "ood",
                    "paraphrasePassRate": 1.0,
                    "paraphraseFamilySummary": family_summary,
                    "layerSpecificityRate": 1.0,
                    "maxReplayError": 0.0,
                },
                "cases": [{
                    "subject": "France",
                    "answer": "Paris",
                    "status": "passed",
                    "originalReplay": {
                        "targetLogitDrop": 2.0,
                        "absoluteReplayError": 0.0,
                        "pass": True,
                    },
                    "paraphrases": paraphrases,
                    "layerControls": {"items": [{"shift": 1}], "pass": True},
                }],
                "claimBoundary": {
                    "provesIfPasses": "Configured causal replay only.",
                    "doesNotProve": "Arbitrary prompt stability.",
                },
            }

        for device in ("cuda", "cpu"):
            self.assertEqual(
                verify_case_report(case_report_with_device(device), "gpt2", "gpt2"),
                [],
                f"a recorded {device} live run must verify",
            )
        errors = verify_case_report(case_report_with_device("gpu-cloud"), "gpt2", "gpt2")
        self.assertEqual(len(errors), 1)
        self.assertIn("supported live device", errors[0])

    def test_artifact_verifier_checks_replacement_path_result_and_aggregate(self) -> None:
        family_names = [family for family, _ in OOD_CAPITAL_PROMPT_TEMPLATES]
        paraphrases = [
            {
                "profile": "ood",
                "family": family,
                "prompt": f"Which city is the capital of France? ({family})",
                "tokenCount": 7,
                "targetLogitDrop": 2.0,
                "ratioToOriginal": 1.0,
                "pass": True,
            }
            for family in family_names
        ]
        paraphrases[0]["pass"] = False
        family_summary = {
            family: {
                "count": 1,
                "passed": int(index != 0),
                "passRate": float(index != 0),
                "meanDropRatio": 1.0,
                "meanTokenCount": 7.0,
            }
            for index, family in enumerate(family_names)
        }
        failed_prompt = paraphrases[0]["prompt"]
        report = {
            "kind": "gpt2-live-ordered-path-ood-stress",
            "synthetic": False,
            "surrogate": False,
            "model": {"id": "gpt2", "adapter": "gpt2", "device": "cuda"},
            "config": {
                "caseCount": 1,
                "paraphraseCount": len(family_names),
                "paraphraseProfile": "ood",
                "minParaphrasePassRate": 0.66,
                "minOriginalDrop": 0.1,
                "minParaphraseDropRatio": 0.1,
                "minLayerSpecificityRate": 0.8,
                "replacementPathSearchEnabled": True,
                "replacementPathTransferEnabled": True,
            },
            "summary": {
                "pass": True,
                "caseCount": 1,
                "originalReplayPassRate": 1.0,
                "paraphraseProfile": "ood",
                "paraphrasePassRate": 5 / 6,
                "paraphraseFamilySummary": family_summary,
                "layerSpecificityRate": 1.0,
                "maxReplayError": 0.0,
                "replacementPathSearch": {"attempted": 1, "passed": 1},
                "replacementPathTransfer": {"attempted": 5, "passed": 5},
            },
            "cases": [{
                "subject": "France",
                "answer": "Paris",
                "status": "passed",
                "originalReplay": {
                    "targetLogitDrop": 2.0,
                    "absoluteReplayError": 0.0,
                    "pass": True,
                },
                "paraphrases": paraphrases,
                "paraphraseRescue": {
                    "attempted": 1,
                    "found": 1,
                    "items": [{
                        "prompt": failed_prompt,
                        "family": family_names[0],
                        "replacementPathSearch": {
                            "prompt": failed_prompt,
                            "status": "passed",
                            "found": True,
                            "searchLayers": [1, 2],
                            "mlpInterventions": [{"layer": 2, "neuron": 8}],
                            "attentionInterventions": [{"layer": 1, "head": 0}],
                            "targetLogitDrop": 0.4,
                            "pathActivationDelta": 0.2,
                            "pathMarginOverControlP95": 0.1,
                            "searchScope": "local loaded layers",
                            "crossFamilyTransfer": {
                                "status": "passed",
                                "sourceFamily": family_names[0],
                                "sourcePrompt": failed_prompt,
                                "discoveryTargetLogitDrop": 0.4,
                                "minTargetLogitDrop": 0.1,
                                "minDropRatio": 0.1,
                                "expectedHeldoutFamilyCount": 5,
                                "items": [
                                    {
                                        "family": item["family"],
                                        "prompt": item["prompt"],
                                        "tokenCount": item["tokenCount"],
                                        "targetLogitDrop": 0.4,
                                        "ratioToDiscovery": 1.0,
                                        "pass": True,
                                    }
                                    for item in paraphrases[1:]
                                ],
                                "summary": {
                                    "attempted": 5,
                                    "passed": 5,
                                    "passRate": 1.0,
                                },
                            },
                        },
                    }],
                },
                "layerControls": {"items": [{"shift": 1}], "pass": True},
            }],
            "claimBoundary": {
                "provesIfPasses": "Configured causal replay only.",
                "doesNotProve": "Arbitrary prompt stability.",
            },
        }

        self.assertEqual(verify_case_report(report, "gpt2", "gpt2"), [])
        report["summary"]["replacementPathSearch"]["attempted"] = 2
        self.assertTrue(
            any("replacement-path attempt count does not match raw measurements" in error
                for error in verify_case_report(report, "gpt2", "gpt2"))
        )
        report["summary"]["replacementPathSearch"]["attempted"] = 1
        report["cases"][0]["paraphraseRescue"]["items"][0][
            "replacementPathSearch"
        ]["crossFamilyTransfer"]["items"][0]["pass"] = False
        self.assertTrue(
            any("transfer pass/fail disagrees with configured thresholds" in error
                for error in verify_case_report(report, "gpt2", "gpt2"))
        )

    def test_markdown_report_exposes_rescue_results_and_scope(self) -> None:
        make_markdown = load_stability_module().make_markdown
        report = {
            "summary": {
                "pass": False,
                "caseCount": 1,
                "originalReplayPassRate": 1.0,
                "paraphraseProfile": "ood",
                "paraphrasePassRate": 0.0,
                "layerSpecificityRate": 1.0,
                "maxReplayError": 0.0,
                "paraphraseFamilySummary": {},
            },
            "cases": [{
                "subject": "France",
                "originalReplay": {"targetLogitDrop": 2.0},
                "paraphraseSummary": {"passRate": 0.0},
                "layerControls": {"marginOverBestPositiveNeighbor": 1.0},
                "paraphraseRescue": {
                    "items": [{
                        "found": True,
                        "bestLayerShift": 1,
                        "brokenDrop": -0.2,
                        "bestDrop": 0.4,
                        "replacementPathSearch": {
                            "status": "passed",
                            "found": True,
                            "mlpInterventions": [{"layer": 2, "neuron": 8}],
                            "attentionInterventions": [{"layer": 1, "head": 0}],
                            "targetLogitDrop": 0.4,
                            "pathActivationDelta": 0.2,
                            "pathMarginOverControlP95": 0.1,
                        },
                    }],
                },
                "status": "failed",
            }],
        }

        markdown = make_markdown(report)

        self.assertIn("## Paraphrase rescue", markdown)
        self.assertIn("Same-component layer-shift hints", markdown)
        self.assertIn("| France | True | 1 | -0.200000 | 0.400000 |", markdown)
        self.assertIn("## Replacement path rediscovery", markdown)
        self.assertIn("| France | passed | 1 | 1 | 0.400000 | 0.200000 | 0.100000 |", markdown)

    def test_replacement_search_uses_failed_prompt_and_existing_ordered_solver(self) -> None:
        module = load_stability_module()
        expected = {
            "status": "passed",
            "orderedResidualPath": {
                "selection": {"mode": "replacement"},
                "nodes": {
                    "mlp": [{"layer": 2, "neuron": 9}],
                    "attention": [{"layer": 1, "head": 3}],
                },
                "pathPatch": {
                    "selectedPathActivationDelta": 0.5,
                    "marginOverControlPathP95": 0.25,
                },
            },
            "combined": {"targetLogitDrop": 1.2},
        }
        source = {"subject": "France", "answer": "Paris"}
        class FakeModel:
            def parameters(self):
                return iter([module.torch.zeros(1)])

        model = FakeModel()
        tokenizer = object()
        modules = {1: object(), 2: object()}
        args = Namespace(
            replacement_candidate_count=10,
            replacement_max_pair_attempts=32,
            replacement_max_set_attempts=24,
        )

        with (
            patch.object(module.ordered, "run_ordered_case", return_value=expected) as run_case,
            patch.object(
                module,
                "encode_prompt",
                return_value={"input_ids": module.torch.zeros((1, 4))},
            ),
        ):
            result = module.rediscover_replacement_path(
                prompt="Which city serves as the capital of France?",
                source=source,
                model=model,
                tokenizer=tokenizer,
                mlps=modules,
                attentions=modules,
                answer_pool=[],
                args=args,
            )

        self.assertTrue(result["found"])
        self.assertEqual(result["mlpInterventions"], expected["orderedResidualPath"]["nodes"]["mlp"])
        self.assertEqual(result["attentionInterventions"], expected["orderedResidualPath"]["nodes"]["attention"])
        call = run_case.call_args.kwargs
        self.assertEqual(call["prompt"], "Which city serves as the capital of France?")
        self.assertIs(call["model"], model)
        self.assertEqual(call["primary_mlps"], modules)
        self.assertEqual(call["args"].attention_max_target_rank, 0)
        self.assertEqual(call["args"].max_target_rank, 0)

    def test_replacement_path_is_replayed_on_all_other_ood_families(self) -> None:
        module = load_stability_module()

        class FakeModel:
            def parameters(self):
                return iter([module.torch.zeros(1)])

        prompt_cases = [
            {"family": family, "prompt": f"heldout {family}"}
            for family, _ in OOD_CAPITAL_PROMPT_TEMPLATES
        ]
        replacement = {
            "status": "passed",
            "prompt": prompt_cases[0]["prompt"],
            "readTokenIndex": 2,
            "targetLogitDrop": 0.4,
            "mlpInterventions": [{
                "layer": 2,
                "neuron": 9,
                "ablateTokenPositions": [2],
            }],
            "attentionInterventions": [{
                "layer": 1,
                "head": 3,
                "ablateTokenPositions": [2],
            }],
        }
        with (
            patch.object(
                module,
                "encode_prompt",
                side_effect=lambda tokenizer, prompt, device: {
                    "input_ids": module.torch.zeros((1, len(prompt.split())))
                },
            ),
            patch.object(
                module,
                "run_drop",
                side_effect=[
                    {"targetLogitDrop": 0.4},
                    {"targetLogitDrop": 0.2},
                    {"targetLogitDrop": 0.4},
                    {"targetLogitDrop": 0.05},
                    {"targetLogitDrop": 0.5},
                ],
            ) as run_drop,
        ):
            result = module.measure_replacement_path_transfer(
                replacement=replacement,
                source_family=prompt_cases[0]["family"],
                prompt_cases=prompt_cases,
                model=FakeModel(),
                tokenizer=object(),
                mlps={1: object(), 2: object()},
                attentions={1: object(), 2: object()},
                target_token_id=42,
                min_target_drop=0.1,
                min_drop_ratio=0.2,
            )

        self.assertEqual(result["summary"], {
            "attempted": 5,
            "passed": 4,
            "passRate": 0.8,
        })
        self.assertEqual(
            [item["family"] for item in result["items"]],
            [case["family"] for case in prompt_cases[1:]],
        )
        self.assertEqual(run_drop.call_count, 5)
        first_transfer = run_drop.call_args_list[0].kwargs
        self.assertEqual(
            first_transfer["mlp_interventions"][0]["ablateTokenPositions"],
            [1],
        )

    def test_missing_replacement_path_does_not_spend_transfer_queries(self) -> None:
        module = load_stability_module()
        with patch.object(module, "run_drop") as run_drop:
            result = module.measure_replacement_path_transfer(
                replacement={"status": "failed", "prompt": "source"},
                source_family="service-question",
                prompt_cases=[
                    {"family": "service-question", "prompt": "source"},
                    {"family": "imperative", "prompt": "heldout"},
                ],
                model=object(),
                tokenizer=object(),
                mlps={},
                attentions={},
                target_token_id=1,
                min_target_drop=0.1,
                min_drop_ratio=0.2,
            )

        self.assertEqual(result["status"], "not_applicable")
        self.assertEqual(result["summary"]["attempted"], 0)
        run_drop.assert_not_called()

    def test_replacement_transfer_abstains_without_other_prompt_families(self) -> None:
        module = load_stability_module()
        result = module.measure_replacement_path_transfer(
            replacement={
                "status": "passed",
                "prompt": "same",
                "readTokenIndex": 0,
                "targetLogitDrop": 0.4,
                "mlpInterventions": [],
                "attentionInterventions": [],
            },
            source_family="standard-template",
            prompt_cases=[{
                "family": "standard-template",
                "prompt": "same",
            }],
            model=object(),
            tokenizer=object(),
            mlps={},
            attentions={},
            target_token_id=1,
            min_target_drop=0.1,
            min_drop_ratio=0.2,
        )

        self.assertEqual(result["status"], "not_applicable")
        self.assertEqual(result["reason"], "no_heldout_prompt_families")
        self.assertEqual(result["summary"]["attempted"], 0)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(OODParaphraseTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    print("OOD_PARAPHRASE_TESTS_PASSED")
