from __future__ import annotations

import re
import sys
import tempfile
import unittest
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS_DIR))

from llm_reader.blackbox_decision import (  # noqa: E402
    analyze_next_token_decision,
    trace_greedy_generation,
)
from llm_reader.blackbox_decompiler import (  # noqa: E402
    OracleBudgetExceeded,
    PersistentOracleSession,
    _adaptive_context_probe_target,
    _adaptive_interaction_probe_target,
    _dependency_prediction_metrics,
    _infer_pair_residuals,
    _predicted_logit_margin,
    _prompt_formulation_effects,
    _prompt_wrapper_similarity,
    _sequence_edit_distance,
    _select_heldout_input_token_indices,
    _validation_metrics,
    allocate_query_budget,
    encode_prompt,
    encode_prompt_with_scope,
    generate_reconstruction,
    predict_next_token,
    run_decompilation,
    write_explanation_markdown,
)


class FakeTokenizer:
    ids = {
        "target": 0,
        "rival": 1,
        "alternative": 2,
        "A": 3,
        "B": 4,
        "C": 5,
    }

    def __init__(self) -> None:
        self.encoded_inputs: list[str] = []

    def __call__(self, text: str, *, add_special_tokens: bool) -> dict[str, list[int]]:
        del add_special_tokens
        self.encoded_inputs.append(text)
        token = text.strip()
        return {"input_ids": [self.ids[token]]} if token in self.ids else {"input_ids": []}

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        del skip_special_tokens, clean_up_tokenization_spaces
        reverse = {value: key for key, value in self.ids.items()}
        return reverse[token_ids[0]]


class FakeGenerationTokenizer:
    all_special_ids = [9, 12]
    eos_token_id = 9

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        del clean_up_tokenization_spaces
        names = {2: " answer", 3: " alternative", 9: "<eos>", 10: "word-a", 11: " word-b", 12: "<control>"}
        return "".join(
            names.get(token_id, f" token-{token_id}")
            for token_id in token_ids
            if not (skip_special_tokens and token_id in self.all_special_ids)
        )


class ChatFakeGenerationTokenizer(FakeGenerationTokenizer):
    all_special_ids = [8, 9, 12]

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        del clean_up_tokenization_spaces
        names = {
            2: " answer",
            3: " alternative",
            8: "<|im_end|>",
            9: "<eos>",
            10: " word-a",
            11: " word-b",
            12: "<|im_start|>",
            13: " word-c",
            14: "user",
            15: "\n",
        }
        return "".join(
            names.get(token_id, f" token-{token_id}")
            for token_id in token_ids
            if not (skip_special_tokens and token_id in self.all_special_ids)
        )


class FakeDecompilerTokenizer:
    all_special_ids = [9]
    eos_token_id = 9

    def __call__(self, text: str, *, add_special_tokens: bool) -> dict[str, list[int]]:
        del add_special_tokens
        known = {"alpha": 10, "beta": 11, "gamma": 12}
        token_ids = [
            known.get(word.strip(".,!?"), 13 + (sum(map(ord, word)) % 2))
            for word in text.split()
        ]
        return {"input_ids": token_ids or [10]}

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        del clean_up_tokenization_spaces
        return "".join(
            "<eos>" if token == 9 else f" t{token}"
            for token in token_ids
            if not (skip_special_tokens and token == 9)
        )


class FakeChatTemplateTokenizer(FakeDecompilerTokenizer):
    chat_template = "synthetic-template"
    all_special_ids = [9, 97, 98]

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> dict[str, list[int]] | str:
        self.assert_chat_options(tokenize, add_generation_prompt)
        rendered = (
            f"<|user|>{messages[0]['content']}<|assistant|>"
        )
        if not tokenize:
            return rendered
        return self(rendered, add_special_tokens=False)

    @staticmethod
    def assert_chat_options(tokenize: bool, add_generation_prompt: bool) -> None:
        del tokenize
        if not add_generation_prompt:
            raise AssertionError("Chat-template encoding options were not preserved.")

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: bool = False,
    ) -> dict[str, list[int] | list[tuple[int, int]]]:
        del add_special_tokens
        if text.startswith("<|user|>") and text.endswith("<|assistant|>"):
            start_marker = "<|user|>"
            end_marker = "<|assistant|>"
            content_start = len(start_marker)
            content_end = len(text) - len(end_marker)
            content = text[content_start:content_end]
            content_ids = super().__call__(
                content,
                add_special_tokens=False,
            )["input_ids"]
            ids = [97, *content_ids, 98]
            if not return_offsets_mapping:
                return {"input_ids": ids}
            offsets = [(0, len(start_marker))]
            offsets.extend(
                (content_start + start, content_start + end)
                for start, end in (
                    (match.start(), match.end())
                    for match in re.finditer(r"\S+", content)
                )
            )
            offsets.append((content_end, len(text)))
            return {"input_ids": ids, "offset_mapping": offsets}
        return super().__call__(text, add_special_tokens=False)


class FakeSystemChatTemplateTokenizer(FakeChatTemplateTokenizer):
    all_special_ids = [9, 96, 97, 98]

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> dict[str, list[int] | list[tuple[int, int]]] | str:
        self.assert_chat_options(tokenize, add_generation_prompt)
        rendered = (
            f"<|system|>You are Qwen.<|user|>{messages[0]['content']}"
            "<|assistant|>"
        )
        if not tokenize:
            return rendered
        return self(rendered, add_special_tokens=False)

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: bool = False,
    ) -> dict[str, list[int] | list[tuple[int, int]]]:
        del add_special_tokens
        if text.startswith("<|system|>") and "<|user|>" in text:
            ids: list[int] = []
            offsets: list[tuple[int, int]] = []
            token_ids = {
                "<|system|>": 96,
                "<|user|>": 97,
                "<|assistant|>": 98,
                "You": 30,
                "are": 31,
                "Qwen.": 32,
                "alpha": 10,
                "beta": 11,
                " ": 60,
            }
            for match in re.finditer(r"<\|[^|]+\|>|[^<\s]+|\s+", text):
                surface = match.group()
                ids.append(
                    token_ids.get(
                        surface,
                        40 + sum(map(ord, surface)) % 40,
                    )
                )
                offsets.append((match.start(), match.end()))
            if return_offsets_mapping:
                return {"input_ids": ids, "offset_mapping": offsets}
            return {"input_ids": ids}
        return super().__call__(
            text,
            add_special_tokens=False,
            return_offsets_mapping=return_offsets_mapping,
        )

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        del clean_up_tokenization_spaces
        surfaces = {
            9: "<eos>",
            10: "alpha",
            11: "beta",
            30: "You",
            31: " are",
            32: " Qwen.",
            60: " ",
            96: "<|system|>",
            97: "<|user|>",
            98: "<|assistant|>",
        }
        return "".join(
            surfaces.get(token, f" t{token}")
            for token in token_ids
            if not (skip_special_tokens and token in self.all_special_ids)
        )


class FakeSlowChatTemplateTokenizer(FakeChatTemplateTokenizer):
    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: bool = False,
    ) -> dict[str, list[int] | list[tuple[int, int]]]:
        if return_offsets_mapping:
            raise NotImplementedError("Offset mappings are unavailable.")
        return super().__call__(text, add_special_tokens=add_special_tokens)


class BlackBoxDecisionTests(unittest.TestCase):
    def test_prompt_encoding_uses_model_chat_template_and_supports_raw_override(self) -> None:
        tokenizer = FakeChatTemplateTokenizer()

        encoded, used_format = encode_prompt(tokenizer, "alpha beta")
        raw_encoded, raw_format = encode_prompt(tokenizer, "alpha beta", "raw")

        self.assertEqual(used_format, "chat-template")
        self.assertEqual(encoded[0], 97)
        self.assertEqual(encoded[-1], 98)
        self.assertEqual(raw_format, "raw")
        self.assertNotEqual(raw_encoded[0], 97)

        with self.assertRaisesRegex(ValueError, "does not provide a usable chat template"):
            encode_prompt(FakeDecompilerTokenizer(), "alpha", "chat-template")

    def test_chat_decompilation_limits_probes_and_heldout_pairs_to_user_content(self) -> None:
        tokenizer = FakeSystemChatTemplateTokenizer()
        token_ids, prompt_format, editable_indices = encode_prompt_with_scope(
            tokenizer,
            "alpha beta",
        )
        self.assertEqual(prompt_format, "chat-template")
        self.assertEqual(
            [token_ids[index] for index in editable_indices],
            [10, 60, 11],
        )
        self.assertNotIn(96, [token_ids[index] for index in editable_indices])
        self.assertNotIn(30, [token_ids[index] for index in editable_indices])

        class ContentOracle:
            query_count = 0

            def query_token_ids(self, input_ids: list[int]) -> list[float]:
                self.query_count += 1
                logits = [0.0] * 20
                has_alpha = 10 in input_ids
                has_beta = 11 in input_ids
                logits[2] = 5.0 if has_alpha and has_beta else 3.0
                logits[3] = 4.0 if has_alpha or has_beta else 2.0
                return logits

        with tempfile.TemporaryDirectory() as directory:
            report = run_decompilation(
                oracle=ContentOracle(),
                tokenizer=tokenizer,
                prompt="alpha beta",
                model_identity={"id": "synthetic-chat-scope"},
                session_dir=directory,
                query_budget=100,
                max_new_tokens=1,
                probes_per_token=4,
                explanation_probes=4,
                interaction_probes=1,
                branch_probes=0,
                branch_width=1,
                top_token_count=2,
                response_probes=1,
                resume=False,
                formulation_probes=0,
            )

        trace = report["trace"]
        allowed = set(editable_indices)
        self.assertEqual(trace["editableInputScope"], "user_message_content")
        self.assertEqual(trace["editableInputTokenIndices"], editable_indices)
        for step in trace["steps"]:
            for intervention in step["contextTokenInterventions"]:
                if intervention["source"] == "input":
                    self.assertTrue(
                        set(intervention["contextTokenIndices"]).issubset(allowed)
                    )
            for interaction in step["pairwiseInterventions"]:
                for deleted_unit in interaction["deletedUnits"]:
                    if deleted_unit["source"] == "input":
                        self.assertTrue(
                            set(deleted_unit["contextTokenIndices"]).issubset(allowed)
                        )
            for witness in step["minimalSufficientContext"]["singleTokenDeletionWitnesses"]:
                self.assertTrue(
                    set(witness["removedContextTokenIndices"]).issubset(allowed)
                )
        self.assertEqual(
            {
                variable["tokenIndex"]
                for variable in report["behaviorGraph"]["inputVariables"]
            },
            allowed,
        )
        self.assertEqual(
            report["behaviorGraph"]["editableInputScope"],
            "user_message_content",
        )
        self.assertTrue({
            index
            for index, token_id in enumerate(token_ids)
            if token_id in {30, 31, 32, 96, 97, 98}
        }.issubset(
            set(report["behaviorGraph"]["fixedPromptContextTokenIndices"])
        ))
        for case in report["validation"]["cases"]:
            self.assertEqual(
                case["editableInputScope"],
                "user_message_content",
            )
            self.assertTrue(
                set(case.get("removedTokenIndices") or []).issubset(allowed)
            )
        self.assertTrue(report["trace"]["responseCounterfactuals"])
        self.assertTrue(all(
            set(item["deletedInputTokenIndices"]).issubset(allowed)
            for item in report["trace"]["responseCounterfactuals"]
        ))

    def test_raw_prompt_scope_keeps_the_entire_prompt_editable(self) -> None:
        tokenizer = FakeChatTemplateTokenizer()
        token_ids, prompt_format, editable_indices = encode_prompt_with_scope(
            tokenizer,
            "alpha beta",
            "raw",
        )

        self.assertEqual(prompt_format, "raw")
        self.assertEqual(editable_indices, list(range(len(token_ids))))

    def test_chat_content_scope_maps_slow_tokenizers_by_unambiguous_token_span(self) -> None:
        token_ids, prompt_format, editable_indices = encode_prompt_with_scope(
            FakeSlowChatTemplateTokenizer(),
            "alpha beta",
        )

        self.assertEqual(prompt_format, "chat-template")
        self.assertEqual([token_ids[index] for index in editable_indices], [10, 11])

    def test_adaptive_interaction_probe_target_respects_remaining_query_budget(self) -> None:
        trace = {
            "steps": [
                {
                    "contextTokenInterventions": [{}, {}, {}],
                    "pairwiseInterventions": [{}],
                },
                {
                    "contextTokenInterventions": [{}, {}],
                    "pairwiseInterventions": [{}],
                },
            ],
        }

        self.assertEqual(_adaptive_interaction_probe_target(trace, 1, 1), 2)
        self.assertEqual(_adaptive_interaction_probe_target(trace, 1, 3), 2)
        self.assertEqual(_adaptive_interaction_probe_target(trace, 1, 0), 1)

    def test_adaptive_interactions_preserve_a_pair_holdout_at_every_decision(self) -> None:
        trace = {
            "steps": [
                {
                    "contextTokenInterventions": [{}, {}, {}],
                    "pairwiseInterventions": [{}],
                },
                {
                    "contextTokenInterventions": [{}, {}, {}, {}],
                    "pairwiseInterventions": [{}],
                },
            ],
        }

        self.assertEqual(_adaptive_interaction_probe_target(trace, 1, 100), 2)

    def test_adaptive_context_probe_target_reserves_half_budget_and_heldout_capacity(self) -> None:
        self.assertEqual(_adaptive_context_probe_target(4, 11, 3, 67), 11)
        self.assertEqual(_adaptive_context_probe_target(4, 11, 3, 14), 6)
        self.assertEqual(_adaptive_context_probe_target(4, 11, 3, 5), 4)

    def test_heldout_input_pair_uses_distinct_single_and_joint_contexts(self) -> None:
        prompt = [10, 11, 12, 13, 14, 15, 16]
        units = [[index] for index in range(len(prompt))]
        dependency_holdout = _select_heldout_input_token_indices(prompt, units)
        interaction_holdout = _select_heldout_input_token_indices(
            prompt,
            units,
            excluded_input_token_indices=dependency_holdout,
        )

        self.assertEqual(len(dependency_holdout), 2)
        self.assertEqual(len(interaction_holdout), 2)
        self.assertTrue(set(dependency_holdout).isdisjoint(interaction_holdout))
        trainable_units = [
            unit
            for unit in units
            if not set(unit).intersection(
                {*dependency_holdout, *interaction_holdout}
            )
        ]
        self.assertGreaterEqual(len(trainable_units), 3)

    def test_trace_measures_pair_components_but_withholds_the_joint_probe(self) -> None:
        class Tokenizer:
            all_special_ids: list[int] = []
            eos_token_id = 19

            def decode(
                self,
                token_ids: list[int],
                *,
                skip_special_tokens: bool = False,
                clean_up_tokenization_spaces: bool = False,
            ) -> str:
                del skip_special_tokens, clean_up_tokenization_spaces
                return "".join(f" token-{token_id}" for token_id in token_ids)

        prompt = [10, 11, 12, 13, 14, 15, 16]
        units = [[index] for index in range(len(prompt))]
        dependency_holdout = _select_heldout_input_token_indices(prompt, units)
        interaction_holdout = _select_heldout_input_token_indices(
            prompt,
            units,
            excluded_input_token_indices=dependency_holdout,
        )

        def oracle(_: list[int]) -> list[float]:
            logits = [0.0] * 20
            logits[2] = 5.0
            logits[3] = 4.0
            return logits

        trace = trace_greedy_generation(
            oracle_query_token_ids=oracle,
            tokenizer=Tokenizer(),
            input_token_ids=prompt,
            max_new_tokens=1,
            probes_per_token=7,
            top_token_count=2,
            interaction_probe_count=100,
            editable_input_token_indices=list(range(len(prompt))),
            reserved_input_token_indices=dependency_holdout,
            priority_input_token_indices=interaction_holdout,
            reserved_pairwise_input_token_indices=interaction_holdout,
        )

        self.assertEqual(
            trace["heldOutInputTokenIndices"],
            dependency_holdout,
        )
        self.assertEqual(
            trace["heldOutPairwiseInputTokenIndices"],
            interaction_holdout,
        )
        step = trace["steps"][0]
        measured_singletons = {
            position
            for intervention in step["contextTokenInterventions"]
            if intervention["source"] == "input"
            for position in intervention["contextTokenIndices"]
        }
        self.assertTrue(set(interaction_holdout).issubset(measured_singletons))
        for interaction in step["pairwiseInterventions"]:
            removed_positions = {
                position
                for unit in interaction["deletedUnits"]
                if unit["source"] == "input"
                for position in unit["contextTokenIndices"]
            }
            self.assertNotEqual(removed_positions, set(interaction_holdout))

    def test_interaction_margin_is_unavailable_without_both_candidate_scores(self) -> None:
        self.assertIsNone(
            _predicted_logit_margin(
                {"candidates": [{"tokenId": 16, "relativeLogit": 1.0}]},
                16,
                23,
            )
        )

    def test_heldout_interaction_keeps_component_effects(self) -> None:
        class UniquePromptTokenizer(FakeDecompilerTokenizer):
            def __call__(
                self,
                text: str,
                *,
                add_special_tokens: bool,
            ) -> dict[str, list[int]]:
                del add_special_tokens
                vocabulary = {
                    "alpha": 10,
                    "beta": 11,
                    "gamma": 12,
                    "delta": 13,
                    "epsilon": 14,
                    "zeta": 15,
                    "eta": 16,
                }
                return {
                    "input_ids": [
                        vocabulary.get(
                            word.strip(".,!?"),
                            20 + sum(map(ord, word)) % 70,
                        )
                        for word in text.split()
                    ]
                }

        class FakeOracle:
            query_count = 0

            def query_token_ids(self, token_ids: list[int]) -> list[float]:
                self.query_count += 1
                logits = [0.0] * 100
                if token_ids[-1] in (2, 3):
                    logits[9] = 8.0
                elif 10 in token_ids and 11 in token_ids:
                    logits[2] = 5.0
                    logits[3] = 4.0
                else:
                    logits[3] = 5.0
                return logits

        with tempfile.TemporaryDirectory() as directory:
            report = run_decompilation(
                oracle=FakeOracle(),
                tokenizer=UniquePromptTokenizer(),
                prompt="alpha beta gamma delta epsilon zeta eta",
                model_identity={"id": "synthetic-factorial"},
                session_dir=directory,
                query_budget=200,
                max_new_tokens=2,
                probes_per_token=7,
                explanation_probes=0,
                interaction_probes=1,
                branch_probes=0,
                branch_width=1,
                top_token_count=2,
                response_probes=0,
                resume=False,
                formulation_probes=0,
            )

        trace = report["trace"]
        dependency_holdout = set(trace["heldOutInputTokenIndices"])
        interaction_holdout = set(trace["heldOutPairwiseInputTokenIndices"])
        self.assertEqual(len(dependency_holdout), 2)
        self.assertEqual(len(interaction_holdout), 2)
        self.assertTrue(dependency_holdout.isdisjoint(interaction_holdout))
        interaction_case = next(
            item
            for item in report["validation"]["cases"]
            if item["role"] == "heldout_interaction"
        )
        self.assertEqual(
            set(interaction_case["removedTokenIndices"]),
            interaction_holdout,
        )
        training_ids = set(report["provenance"]["trainingQueryIds"])
        independent_validation_ids = set(
            report["provenance"]["independentValidationQueryIds"]
        )
        evidence_by_context = {
            tuple(item["contextTokenIds"]): item["queryId"]
            for item in interaction_case["oracleEvidence"]
        }
        baseline_context = report["input"]["tokenIds"]
        for removed_position in interaction_holdout:
            single_context = [
                token_id
                for position, token_id in enumerate(baseline_context)
                if position != removed_position
            ]
            self.assertIn(
                evidence_by_context[tuple(single_context)],
                training_ids,
            )
        self.assertIn(
            evidence_by_context[tuple(interaction_case["inputTokenIds"])],
            independent_validation_ids,
        )
        interaction_assessment = interaction_case["interactionAssessment"]
        self.assertTrue(interaction_assessment["residualScoreAvailable"])
        self.assertIsNotNone(interaction_assessment["predictedResidual"])
        self.assertLess(interaction_assessment["absoluteError"], 100.0)
        self.assertTrue(report["provenance"]["queryAccountingMatches"])

    def test_persistent_oracle_session_caches_and_resumes_observations(self) -> None:
        class FakeOracle:
            query_count = 0

            def query_token_ids(self, token_ids: list[int]) -> list[float]:
                self.query_count += 1
                return [float(token_ids[0]), 0.0, 0.0]

        with tempfile.TemporaryDirectory() as directory:
            oracle = FakeOracle()
            request = {
                "identity": {"id": "synthetic"},
                "prompt": "test",
                "parameters": {"maxNewTokens": 1},
            }
            session = PersistentOracleSession(
                oracle,
                directory,
                identity=request["identity"],
                prompt=request["prompt"],
                parameters=request["parameters"],
                query_budget=2,
                resume=False,
            )
            self.assertEqual(session.query_token_ids([1]), [1.0, 0.0, 0.0])
            session.set_training_boundary(["q000001"])
            resumed = PersistentOracleSession(
                oracle,
                directory,
                identity=request["identity"],
                prompt=request["prompt"],
                parameters=request["parameters"],
                query_budget=2,
                resume=True,
            )
            self.assertEqual(resumed.query_token_ids([1]), [1.0, 0.0, 0.0])
            self.assertEqual(oracle.query_count, 1)
            self.assertEqual(resumed.training_query_ids, ["q000001"])

    def test_persistent_oracle_session_enforces_query_budget_and_keeps_checkpoint(self) -> None:
        class FakeOracle:
            def query_token_ids(self, _: list[int]) -> list[float]:
                return [0.0, 1.0]

        with tempfile.TemporaryDirectory() as directory:
            session = PersistentOracleSession(
                FakeOracle(),
                directory,
                identity={"id": "synthetic"},
                prompt="test",
                parameters={},
                query_budget=1,
                resume=False,
            )
            session.query_token_ids([1])
            with self.assertRaisesRegex(OracleBudgetExceeded, "Resume this session"):
                session.query_token_ids([2])
            checkpoint = __import__("json").loads(
                (Path(directory) / "checkpoint.json").read_text(encoding="utf-8")
            )
            self.assertEqual(checkpoint["status"], "budget-exhausted")
            self.assertEqual(checkpoint["oracleQueriesCompleted"], 1)

    def test_prompt_formulation_effect_reports_measured_first_divergence(self) -> None:
        baseline_logits = [0.0] * 7
        baseline_logits[5] = 3.0
        baseline_logits[6] = 1.0
        formulation_logits = [0.0] * 7
        formulation_logits[5] = 1.0
        formulation_logits[6] = 4.0
        effect = _prompt_formulation_effects(
            baseline_prompt_token_ids=[1],
            baseline_generated_token_ids=[5],
            formulation_paths=[{
                "id": "wrapper-a",
                "inputTokenIds": [2],
                "generatedTokenIds": [6],
            }],
            records=[
                {"inputTokenIds": [1], "queryId": "q-baseline"},
                {"inputTokenIds": [2], "queryId": "q-formulation"},
            ],
            cache={
                (1,): baseline_logits,
                (2,): formulation_logits,
            },
        )[0]

        self.assertEqual(effect["sharedGeneratedPrefixLength"], 0)
        self.assertEqual(effect["firstDivergence"]["baselineTokenId"], 5)
        self.assertEqual(effect["firstDivergence"]["formulationTokenId"], 6)
        self.assertEqual(effect["firstDivergence"]["baselinePairwiseMargin"], 2.0)
        self.assertEqual(effect["firstDivergence"]["formulationPairwiseMargin"], 3.0)
        self.assertEqual(effect["firstDivergence"]["pairwiseMarginShift"], 1.0)
        self.assertEqual(
            effect["evidenceQueryIds"],
            ["q-baseline", "q-formulation"],
        )

    def test_formulation_kernel_selects_matching_wrapper_expert(self) -> None:
        self.assertGreater(
            _prompt_wrapper_similarity(
                "Instruction: ",
                "\n\nProvide a concise answer.",
                "Please answer the following request:\n\n",
                "\n\nAnswer concisely.",
            ),
            _prompt_wrapper_similarity(
                "Instruction: ",
                "\n\nProvide a concise answer.",
                "Request:\n",
                "\n\nResponse:",
            ),
        )
        reconstruction = {
            "inputTokenIds": [10],
            "inputPromptText": "alpha",
            "observations": [{
                "contextTokenIds": [99, 10, 98],
                "generatedPrefixTokenIds": [],
                "queryId": "q-neighbor",
                "topCandidates": [{"tokenId": 2, "probability": 1.0}],
            }],
            "promptFormulationExperts": [
                {
                    "id": "answer-concise",
                    "promptPrefix": "Please answer the following request:\n\n",
                    "promptSuffix": "\n\nAnswer concisely.",
                    "observations": [{
                        "generatedPrefixTokenIds": [],
                        "queryId": "q-answer",
                        "topCandidates": [
                            {"tokenId": 5, "probability": 0.6},
                            {"tokenId": 6, "probability": 0.4},
                        ],
                    }],
                },
                {
                    "id": "request-response",
                    "promptPrefix": "Request:\n",
                    "promptSuffix": "\n\nResponse:",
                    "observations": [{
                        "generatedPrefixTokenIds": [],
                        "queryId": "q-response",
                        "topCandidates": [{"tokenId": 7, "probability": 1.0}],
                    }],
                },
            ],
        }

        prediction = predict_next_token(
            reconstruction,
            [90, 10, 91],
            prompt_token_ids=[90, 10, 91],
            prompt_surface=(
                "Instruction: alpha\n\nProvide a concise answer."
            ),
            generated_prefix_token_ids=[],
        )

        self.assertEqual(prediction["method"], "prompt_formulation_kernel_vote")
        self.assertEqual(prediction["tokenId"], 5)
        self.assertEqual(
            prediction["formulationEvidence"][0]["formulationId"],
            "answer-concise",
        )
        self.assertIn("q-answer", prediction["evidenceQueryIds"])

    def test_unseen_pair_uses_shrunk_interaction_residuals_from_similar_effects(self) -> None:
        reconstruction = {
            "inputPromptText": "unused",
            "observations": [],
            "decisionModels": [{
                "generatedPrefixTokenIds": [],
                "baselinePromptTokenIds": [10, 11, 12, 13],
                "baselineLogitsByTokenId": {"2": 0.0, "3": 0.5},
                "baselineQueryId": "q-baseline",
                "inputDeletionEffects": [
                    {
                        "deletedTokenIndices": [0],
                        "deltaByTokenId": {"2": 0.0, "3": 0.0},
                        "queryId": "q-single-0",
                    },
                    {
                        "deletedTokenIndices": [1],
                        "deltaByTokenId": {"2": 0.2, "3": 0.0},
                        "queryId": "q-single-1",
                    },
                    {
                        "deletedTokenIndices": [2],
                        "deltaByTokenId": {"2": 0.2, "3": 0.0},
                        "queryId": "q-single-2",
                    },
                    {
                        "deletedTokenIndices": [3],
                        "deltaByTokenId": {"2": 0.2, "3": 0.0},
                        "queryId": "q-single-3",
                    },
                ],
                "measuredPairCorrections": [
                    {
                        "deletedTokenIndices": [[0], [1]],
                        "residualByTokenId": {"2": -4.0, "3": 0.0},
                        "queryId": "q-pair-01",
                    },
                    {
                        "deletedTokenIndices": [[1], [2]],
                        "residualByTokenId": {"2": -4.0, "3": 0.0},
                        "queryId": "q-pair-12",
                    },
                    {
                        "deletedTokenIndices": [[2], [3]],
                        "residualByTokenId": {"2": -4.0, "3": 0.0},
                        "queryId": "q-pair-23",
                    },
                ],
            }],
        }

        prediction = predict_next_token(
            reconstruction,
            [11, 12],
            prompt_token_ids=[11, 12],
            generated_prefix_token_ids=[],
        )

        self.assertEqual(
            prediction["method"],
            "similarity_weighted_pair_interaction_model",
        )
        self.assertEqual(prediction["tokenId"], 3)
        estimate = prediction["interactionEstimate"]
        self.assertEqual(estimate["method"], "cross-pair-shrunk-kernel")
        self.assertTrue(estimate["applied"])
        self.assertTrue(estimate["calibration"]["accepted"])
        self.assertGreater(estimate["effectiveSupport"], 0.0)
        self.assertGreater(estimate["confidence"], 0.0)
        self.assertLess(estimate["confidence"], 1.0)
        self.assertIn(estimate["evidence"][0]["queryId"], {
            "q-pair-01", "q-pair-12", "q-pair-23",
        })
        self.assertTrue(set(
            item["queryId"] for item in estimate["evidence"]
        ).intersection(prediction["evidenceQueryIds"]))
        self.assertLess(estimate["residualByTokenId"]["2"], 0.0)
        self.assertGreater(estimate["residualByTokenId"]["2"], -4.0)

    def test_heldout_dependency_intervention_prediction(self) -> None:
        baseline_ids = [100, 10, 11, 12, 13, 14, 15, 101]
        reconstruction = {
            "inputPromptText": "unused",
            "observations": [{
                "contextTokenIds": baseline_ids,
                "generatedPrefixTokenIds": [],
                "queryId": "q-baseline",
                "topCandidates": [
                    {"tokenId": 2, "probability": 0.8},
                    {"tokenId": 3, "probability": 0.2},
                ],
            }],
            "decisionModels": [{
                "generatedPrefixTokenIds": [],
                "baselinePromptTokenIds": baseline_ids,
                "baselineLogitsByTokenId": {"2": 1.0, "3": 0.0},
                "baselineQueryId": "q-baseline",
                "candidateTokenIds": [2, 3],
                "inputDeletionEffects": [
                    {
                        "deletedTokenIndices": [index],
                        "deltaByTokenId": {"2": -3.0, "3": 0.0},
                        "queryId": f"q-deletion-{index}",
                    }
                    for index in (1, 2, 4, 5, 6)
                ],
                "measuredPairCorrections": [],
            }],
        }
        counterfactual_ids = baseline_ids[:3] + baseline_ids[4:]

        prediction = predict_next_token(
            reconstruction,
            counterfactual_ids,
            prompt_token_ids=counterfactual_ids,
            generated_prefix_token_ids=[],
        )

        self.assertEqual(prediction["method"], "calibrated_local_intervention_transfer")
        self.assertEqual(prediction["tokenId"], 3)
        estimate = prediction["interventionTransferEstimate"]
        self.assertEqual(estimate["estimatedDeletedTokenIndices"], [[3]])
        self.assertGreater(
            estimate["crossValidationRelativeMSEImprovement"],
            0.0,
        )
        self.assertEqual(
            prediction["confidenceKind"],
            "cross_validation_relative_mse_improvement",
        )
        self.assertIn("q-baseline", prediction["evidenceQueryIds"])
        self.assertTrue(
            set(estimate["evidenceQueryIds"]).issubset({
                f"q-deletion-{index}" for index in (1, 2, 4, 5, 6)
            })
        )

    def test_transferred_span_pair_uses_calibrated_measured_interactions(self) -> None:
        baseline_ids = [100, 10, 11, 12, 13, 14, 15, 101]
        measured_spans = (1, 2, 3, 5)
        pair_groups = (
            [[1], [2]],
            [[2], [3]],
            [[3], [5]],
            [[1], [5]],
        )
        reconstruction = {
            "inputPromptText": "unused",
            "editableInputTokenIndices": list(range(len(baseline_ids))),
            "observations": [],
            "decisionModels": [{
                "generatedPrefixTokenIds": [],
                "baselinePromptTokenIds": baseline_ids,
                "baselineLogitsByTokenId": {"2": 1.0, "3": 0.0},
                "baselineQueryId": "q-baseline",
                "inputDeletionEffects": [
                    {
                        "deletedTokenIndices": [index],
                        "deltaByTokenId": {"2": -0.2, "3": 0.0},
                        "queryId": f"q-deletion-{index}",
                    }
                    for index in measured_spans
                ],
                "measuredPairCorrections": [
                    {
                        "deletedTokenIndices": groups,
                        "residualByTokenId": {"2": -2.0, "3": 0.0},
                        "queryId": f"q-pair-{index}",
                    }
                    for index, groups in enumerate(pair_groups)
                ],
            }],
        }
        counterfactual_ids = [
            token for index, token in enumerate(baseline_ids)
            if index not in {4, 6}
        ]

        prediction = predict_next_token(
            reconstruction,
            counterfactual_ids,
            prompt_token_ids=counterfactual_ids,
            generated_prefix_token_ids=[],
        )

        self.assertEqual(
            prediction["method"],
            "calibrated_local_intervention_and_pair_interaction_transfer",
        )
        self.assertEqual(prediction["tokenId"], 3)
        transfer = prediction["interventionTransferEstimate"]
        self.assertEqual(transfer["estimatedDeletedTokenIndices"], [[4], [6]])
        self.assertEqual(len(transfer["estimatedEffectsBySpan"]), 2)
        interaction = prediction["interactionEstimate"]
        self.assertTrue(interaction["applied"])
        self.assertTrue(interaction["calibration"]["accepted"])
        self.assertEqual(
            [component["effectSource"] for component in interaction["targetComponents"]],
            [
                "calibrated_local_intervention_transfer",
                "calibrated_local_intervention_transfer",
            ],
        )
        evidence = set(prediction["evidenceQueryIds"])
        for component in interaction["targetComponents"]:
            self.assertTrue(component["evidenceQueryIds"])
            self.assertTrue(set(component["evidenceQueryIds"]).issubset(evidence))
        self.assertIn("q-baseline", evidence)
        self.assertTrue(evidence.intersection({
            f"q-pair-{index}" for index in range(len(pair_groups))
        }))
        self.assertTrue(evidence.intersection({
            f"q-deletion-{index}" for index in measured_spans
        }))

    def test_local_intervention_transfer_abstains_without_cv_gain(self) -> None:
        baseline_ids = [100, 10, 11, 12, 13, 14, 101]
        reconstruction = {
            "inputPromptText": "unused",
            "observations": [{
                "contextTokenIds": baseline_ids,
                "generatedPrefixTokenIds": [],
                "queryId": "q-baseline",
                "topCandidates": [{"tokenId": 2, "probability": 1.0}],
            }],
            "decisionModels": [{
                "generatedPrefixTokenIds": [],
                "baselinePromptTokenIds": baseline_ids,
                "baselineLogitsByTokenId": {"2": 1.0, "3": 0.0},
                "baselineQueryId": "q-baseline",
                "inputDeletionEffects": [
                    {
                        "deletedTokenIndices": [index],
                        "deltaByTokenId": {
                            "2": -3.0 if index % 2 else 3.0,
                            "3": 0.0,
                        },
                        "queryId": f"q-deletion-{index}",
                    }
                    for index in (1, 2, 3, 4)
                ],
                "measuredPairCorrections": [],
            }],
        }
        counterfactual_ids = baseline_ids[:5] + baseline_ids[6:]

        prediction = predict_next_token(
            reconstruction,
            counterfactual_ids,
            prompt_token_ids=counterfactual_ids,
            generated_prefix_token_ids=[],
        )

        self.assertEqual(prediction["method"], "similarity_weighted_observation_vote")
        self.assertIsNone(prediction.get("interventionTransferEstimate"))

    def test_local_intervention_transfer_does_not_extrapolate_outside_editable_scope(self) -> None:
        baseline_ids = [100, 10, 11, 12, 13, 14, 101]
        reconstruction = {
            "inputPromptText": "unused",
            "editableInputTokenIndices": [1, 2, 3, 4, 5],
            "observations": [{
                "contextTokenIds": baseline_ids,
                "generatedPrefixTokenIds": [],
                "queryId": "q-baseline",
                "topCandidates": [{"tokenId": 2, "probability": 1.0}],
            }],
            "decisionModels": [{
                "generatedPrefixTokenIds": [],
                "baselinePromptTokenIds": baseline_ids,
                "baselineLogitsByTokenId": {"2": 1.0, "3": 0.0},
                "baselineQueryId": "q-baseline",
                "inputDeletionEffects": [
                    {
                        "deletedTokenIndices": [index],
                        "deltaByTokenId": {"2": -3.0, "3": 0.0},
                        "queryId": f"q-deletion-{index}",
                    }
                    for index in (1, 2, 4, 5)
                ],
                "measuredPairCorrections": [],
            }],
        }
        counterfactual_ids = baseline_ids[1:]

        prediction = predict_next_token(
            reconstruction,
            counterfactual_ids,
            prompt_token_ids=counterfactual_ids,
            generated_prefix_token_ids=[],
        )

        self.assertEqual(prediction["method"], "similarity_weighted_observation_vote")
        self.assertIsNone(prediction.get("interventionTransferEstimate"))

    def test_pair_kernel_falls_back_when_leave_one_pair_out_worsens_zero_model(self) -> None:
        reconstruction = {
            "inputPromptText": "unused",
            "observations": [],
            "decisionModels": [{
                "generatedPrefixTokenIds": [],
                "baselinePromptTokenIds": [10, 11, 12, 13],
                "baselineLogitsByTokenId": {"2": 1.0, "3": 0.5},
                "baselineQueryId": "q-baseline",
                "inputDeletionEffects": [
                    {
                        "deletedTokenIndices": [index],
                        "deltaByTokenId": {"2": 0.2, "3": 0.0},
                        "queryId": f"q-single-{index}",
                    }
                    for index in range(4)
                ],
                "measuredPairCorrections": [
                    {
                        "deletedTokenIndices": [[0], [1]],
                        "residualByTokenId": {"2": -4.0, "3": 0.0},
                        "queryId": "q-pair-01",
                    },
                    {
                        "deletedTokenIndices": [[1], [2]],
                        "residualByTokenId": {"2": 4.0, "3": 0.0},
                        "queryId": "q-pair-12",
                    },
                    {
                        "deletedTokenIndices": [[2], [3]],
                        "residualByTokenId": {"2": -4.0, "3": 0.0},
                        "queryId": "q-pair-23",
                    },
                ],
            }],
        }

        prediction = predict_next_token(
            reconstruction,
            [11, 12],
            prompt_token_ids=[11, 12],
            generated_prefix_token_ids=[],
        )

        self.assertEqual(prediction["method"], "additive_single_intervention_model")
        self.assertEqual(prediction["tokenId"], 2)
        estimate = prediction["interactionEstimate"]
        self.assertFalse(estimate["applied"])
        self.assertFalse(estimate["calibration"]["accepted"])
        self.assertGreater(
            estimate["calibration"]["kernelMeanAbsoluteError"],
            estimate["calibration"]["zeroResidualMeanAbsoluteError"],
        )
        self.assertEqual(estimate["residualByTokenId"]["2"], 0.0)
        self.assertNotEqual(
            estimate["candidateResidualByTokenId"]["2"],
            0.0,
        )

    def test_pair_kernel_applies_only_calibrated_decision_margins(self) -> None:
        decision_model = {
            "baselinePromptTokenIds": [10, 11, 12, 13],
            "baselineLogitsByTokenId": {
                "2": 10.0,
                "3": 9.0,
                "4": 0.0,
                "5": -1.0,
            },
            "baselineTopTokenId": 2,
            "candidateTokenIds": [2, 3, 4, 5],
            "inputDeletionEffects": [
                {
                    "deletedTokenIndices": [index],
                    "deltaByTokenId": {
                        "2": 1.0,
                        "3": 0.0,
                        "4": 0.0,
                        "5": 0.0,
                    },
                    "queryId": f"q-single-{index}",
                }
                for index in range(4)
            ],
            "measuredPairCorrections": [
                {
                    "deletedTokenIndices": pair,
                    "residualByTokenId": residuals,
                    "queryId": f"q-pair-{index}",
                }
                for index, (pair, residuals) in enumerate([
                    ([[0], [1]], {"2": 8.0, "3": 6.0, "4": 27.0, "5": -3.0}),
                    ([[0], [2]], {"2": 6.0, "3": 8.0, "4": 27.0, "5": -3.0}),
                    ([[1], [2]], {"2": 8.0, "3": 6.0, "4": 27.0, "5": -3.0}),
                    ([[1], [3]], {"2": 6.0, "3": 8.0, "4": 27.0, "5": -3.0}),
                ])
            ],
        }
        applied = [
            decision_model["inputDeletionEffects"][index]
            for index in (0, 3)
        ]

        estimate = _infer_pair_residuals(decision_model, applied)

        self.assertIsNotNone(estimate)
        assert estimate is not None
        self.assertTrue(estimate["applied"])
        self.assertTrue(estimate["calibration"]["accepted"])
        self.assertEqual(estimate["calibration"]["acceptedMarginCount"], 2)
        self.assertEqual(estimate["calibration"]["rejectedMarginCount"], 1)
        frontier = {
            str(item["alternativeTokenId"]): item
            for item in estimate["calibration"]["decisionFrontier"]
        }
        self.assertFalse(frontier["3"]["accepted"])
        self.assertTrue(frontier["4"]["accepted"])
        self.assertTrue(frontier["5"]["accepted"])
        applied_residuals = estimate["residualByTokenId"]
        self.assertNotEqual(applied_residuals["2"], 0.0)
        self.assertEqual(applied_residuals["3"], applied_residuals["2"])
        self.assertNotEqual(
            applied_residuals["4"],
            applied_residuals["2"],
        )
        self.assertNotEqual(
            applied_residuals["5"],
            applied_residuals["2"],
        )

    def test_pair_kernel_includes_protected_eos_in_decision_frontier(self) -> None:
        decision_model = {
            "baselinePromptTokenIds": [10, 11, 12, 13],
            "baselineLogitsByTokenId": {
                "2": 10.0,
                "3": 9.0,
                "4": 0.0,
                "5": -1.0,
                "6": -10.0,
            },
            "inputDeletionEffects": [
                {
                    "deletedTokenIndices": [index],
                    "deltaByTokenId": {
                        "2": 0.0,
                        "3": 0.0,
                        "4": 20.0 if index in (0, 3) else 0.0,
                        "5": 0.0,
                    },
                    "queryId": f"q-single-{index}",
                }
                for index in range(4)
            ],
            "measuredPairCorrections": [
                {
                    "deletedTokenIndices": pair,
                    "residualByTokenId": {"2": 0.0, "3": 0.0, "4": 1.0, "5": 0.0},
                    "queryId": f"q-pair-{index}",
                }
                for index, pair in enumerate([
                    [[0], [1]],
                    [[0], [2]],
                    [[1], [2]],
                    [[1], [3]],
                ])
            ],
        }
        applied = [
            decision_model["inputDeletionEffects"][index]
            for index in (0, 3)
        ]

        estimate = _infer_pair_residuals(
            decision_model,
            applied,
            protected_token_ids=[6],
        )

        self.assertIsNotNone(estimate)
        assert estimate is not None
        calibration = estimate["calibration"]
        self.assertEqual(calibration["decisionMarginAnchorTokenId"], 4)
        self.assertEqual(
            calibration["decisionFrontierSource"],
            "baseline_plus_applied_single_deletion_effects",
        )
        self.assertEqual(calibration["alternativeTokenIds"], [2, 3, 5, 6])

    def test_reconstruction_beam_retains_measured_non_greedy_response_route(self) -> None:
        reconstruction = {
            "observations": [
                {
                    "contextTokenIds": [1],
                    "generatedPrefixTokenIds": [],
                    "queryId": "q-root",
                    "topCandidates": [
                        {"tokenId": 2, "probability": 0.6},
                        {"tokenId": 3, "probability": 0.4},
                    ],
                },
                {
                    "contextTokenIds": [1, 2],
                    "generatedPrefixTokenIds": [2],
                    "queryId": "q-greedy",
                    "topCandidates": [{"tokenId": 10, "probability": 1.0}],
                },
                {
                    "contextTokenIds": [1, 3],
                    "generatedPrefixTokenIds": [3],
                    "queryId": "q-alternative",
                    "topCandidates": [{"tokenId": 11, "probability": 1.0}],
                },
            ],
            "decisionModels": [],
            "promptFormulationExperts": [],
        }

        result = generate_reconstruction(
            reconstruction,
            [1],
            FakeGenerationTokenizer(),
            max_new_tokens=2,
            beam_width=2,
        )

        self.assertEqual(result["tokenIds"], [2, 10])
        self.assertEqual(
            [item["tokenIds"] for item in result["candidateResponses"]],
            [[2, 10], [3, 11]],
        )

    def test_integrated_decompilation_emits_executable_graph_and_heldout_metrics(self) -> None:
        class FakeOracle:
            query_count = 0

            def query_token_ids(self, token_ids: list[int]) -> list[float]:
                self.query_count += 1
                logits = [0.0] * 20
                if token_ids[-1] in (2, 3):
                    logits[9] = 8.0
                elif 10 in token_ids and 11 in token_ids:
                    logits[2] = 5.0
                    logits[3] = 4.0
                else:
                    logits[2] = 0.0
                    logits[3] = 5.0
                return logits

        with tempfile.TemporaryDirectory() as directory:
            tokenizer = FakeChatTemplateTokenizer()
            report = run_decompilation(
                oracle=FakeOracle(),
                tokenizer=tokenizer,
                prompt="alpha beta gamma",
                model_identity={"id": "synthetic", "modelType": "synthetic"},
                session_dir=directory,
                query_budget=100,
                max_new_tokens=2,
                probes_per_token=3,
                explanation_probes=0,
                interaction_probes=1,
                branch_probes=1,
                branch_width=1,
                top_token_count=2,
                response_probes=0,
                resume=False,
                prompt_format="auto",
            )
            self.assertEqual(report["kind"], "gcl-blackbox-neural-decompilation")
            self.assertEqual(
                report["executableReconstruction"]["kind"],
                "gcl-blackbox-executable-behavior-reconstruction",
            )
            self.assertTrue(report["behaviorGraph"]["tokenPredictions"])
            self.assertEqual(report["input"]["promptFormat"], "chat-template")
            self.assertEqual(report["input"]["tokenIds"][0], 97)
            self.assertEqual(report["input"]["tokenIds"][-1], 98)
            self.assertTrue(report["trace"]["terminatedByEos"])
            self.assertEqual(report["trace"]["terminationReason"], "eos")
            self.assertGreater(
                report["queryAllocation"]["adaptiveInteractionProbing"][
                    "effectivePerToken"
                ],
                1,
            )
            self.assertEqual(
                report["provenance"]["effectiveProbeSettings"]["interactionProbes"],
                report["queryAllocation"]["adaptiveInteractionProbing"][
                    "effectivePerToken"
                ],
            )
            self.assertTrue(report["validation"]["cases"])
            self.assertIn(
                "beamCandidateResponseCoverage",
                report["validation"],
            )
            self.assertEqual(
                [item["id"] for item in report["executableReconstruction"][
                    "promptFormulationExperts"
                ]],
                [
                    "baseline",
                    "training-formulation-request-wrapper",
                ],
            )
            self.assertEqual(len(report["promptFormulationEffects"]), 1)
            self.assertEqual(
                report["explanation"]["formulationEffects"],
                report["promptFormulationEffects"],
            )
            self.assertTrue(report["provenance"]["queryAccountingMatches"])
            self.assertEqual(len(report["trainingPromptFormulations"]), 1)
            self.assertEqual(
                report["provenance"]["formulationLogicalQueryCount"],
                sum(
                    len(item["generatedTokenIds"])
                    for item in report["trainingPromptFormulations"]
                ),
            )
            formulation = report["trainingPromptFormulations"][0]
            formulation_observation = next(
                item
                for item in report["executableReconstruction"]["observations"]
                if item["contextTokenIds"] == formulation["inputTokenIds"]
            )
            self.assertEqual(formulation_observation["generatedPrefixTokenIds"], [])
            self.assertIn(
                formulation_observation["queryId"],
                report["provenance"]["trainingQueryIds"],
            )
            heldout_interaction = next(
                item for item in report["validation"]["cases"]
                if item["role"] == "heldout_interaction"
            )
            self.assertTrue(heldout_interaction["exactResponseMatch"])
            self.assertTrue(heldout_interaction["actualTerminatedByEos"])
            self.assertTrue(heldout_interaction["fullResponseExactMatch"])
            self.assertIn(
                heldout_interaction["predictionEvidence"][0]["conditional"]["method"],
                {
                    "additive_single_intervention_model",
                    "measured_pair_intervention_model",
                    "similarity_weighted_pair_interaction_model",
                },
            )
            self.assertIsNone(report["validation"]["counterfactualEffectAccuracy"])
            self.assertEqual(report["validation"]["interactionReconstructionAccuracy"], 1.0)
            self.assertEqual(report["validation"]["interactionResidualEvaluations"], 1)
            interaction_assessment = heldout_interaction["interactionAssessment"]
            self.assertIsNotNone(interaction_assessment)
            self.assertTrue(interaction_assessment["evidenceQueryIds"])
            self.assertIn("predictedResidual", interaction_assessment)
            self.assertFalse(report["access"]["weightsUsedBySolver"])
            predicted = generate_reconstruction(
                report["executableReconstruction"],
                report["input"]["tokenIds"],
                tokenizer,
                max_new_tokens=2,
            )
            self.assertTrue(predicted["tokenIds"])

    def test_formulation_probe_budget_rejects_unavailable_training_formulations(self) -> None:
        with self.assertRaisesRegex(ValueError, "2 available training formulations"):
            allocate_query_budget(
                query_budget=100,
                max_new_tokens=2,
                probes_per_token=1,
                explanation_probes=0,
                interaction_probes=0,
                branch_probes=0,
                branch_width=1,
                response_probes=0,
                formulation_probes=3,
            )

    def test_validation_metrics_separate_dependencies_from_pair_effects(self) -> None:
        class FakeOracle:
            query_count = 0

            def query_token_ids(self, token_ids: list[int]) -> list[float]:
                self.query_count += 1
                logits = [0.0] * 20
                if token_ids[-1] in (2, 3):
                    logits[9] = 8.0
                elif 10 in token_ids and 11 in token_ids:
                    logits[2] = 5.0
                    logits[3] = 4.0
                else:
                    logits[3] = 5.0
                return logits

        with tempfile.TemporaryDirectory() as directory:
            report = run_decompilation(
                oracle=FakeOracle(),
                tokenizer=FakeDecompilerTokenizer(),
                prompt="alpha beta gamma delta epsilon",
                model_identity={"id": "synthetic", "modelType": "synthetic"},
                session_dir=directory,
                query_budget=100,
                max_new_tokens=2,
                probes_per_token=1,
                explanation_probes=0,
                interaction_probes=1,
                branch_probes=1,
                branch_width=1,
                top_token_count=2,
                response_probes=0,
                resume=False,
                formulation_probes=0,
            )

        validation = report["validation"]
        context_probing = report["queryAllocation"]["adaptiveContextProbing"]
        self.assertEqual(context_probing["initialEffectivePerToken"], 1)
        self.assertEqual(context_probing["effectivePerToken"], 4)
        self.assertEqual(context_probing["heldOutInputUnitsReserved"], 2)
        self.assertEqual(validation["dependencyEvaluations"], 2)
        self.assertEqual(validation["dependencyReconstructionAccuracy"], 0.5)
        self.assertEqual(
            sum(validation["dependencyPredictionMethodCounts"].values()),
            validation["dependencyEvaluations"],
        )
        self.assertGreaterEqual(
            validation["dependencyInterventionModelCoverage"], 0.0
        )
        self.assertLessEqual(
            validation["dependencyInterventionModelCoverage"], 1.0
        )
        self.assertEqual(validation["counterfactualEffectAccuracy"], 1.0)
        self.assertEqual(validation["interactionEvaluations"], 1)
        self.assertEqual(validation["interactionReconstructionAccuracy"], 1.0)
        heldout_indices = set(report["trace"]["heldOutInputTokenIndices"])
        self.assertEqual(len(heldout_indices), 2)
        for step in report["trace"]["steps"]:
            for intervention in step["contextTokenInterventions"]:
                if intervention["source"] == "input":
                    self.assertTrue(
                        heldout_indices.isdisjoint(
                            intervention["contextTokenIndices"]
                        )
                    )
            for interaction in step["pairwiseInterventions"]:
                for deleted_unit in interaction["deletedUnits"]:
                    if deleted_unit["source"] == "input":
                        self.assertTrue(
                            heldout_indices.isdisjoint(
                                deleted_unit["contextTokenIndices"]
                            )
                        )
            for witness in step["minimalSufficientContext"]["singleTokenDeletionWitnesses"]:
                self.assertTrue(
                    heldout_indices.isdisjoint(
                        witness["removedContextTokenIndices"]
                    )
                )
            sufficient_context = step["minimalSufficientContext"]
            self.assertEqual(
                sufficient_context["minimality"]["scope"],
                "editable_context_units_excluding_reserved_input_holdout",
            )
            self.assertEqual(
                set(sufficient_context["excludedHeldOutInputTokenIndices"]),
                heldout_indices,
            )
            self.assertIn(
                "Reserved held-out input units were excluded from this search.",
                sufficient_context["minimality"]["meaning"],
            )
        self.assertIn("metricDefinitions", validation)
        self.assertIn(
            "dependencyInterventionModelCoverage",
            validation["metricDefinitions"],
        )
        self.assertTrue(report["provenance"]["queryAccountingMatches"])
        training_ids = set(report["provenance"]["trainingQueryIds"])
        heldout = next(
            item for item in validation["cases"]
            if item["role"] == "heldout_counterfactual"
        )
        autoregressive_assessments = heldout[
            "autoregressiveDependencyAssessments"
        ]
        self.assertEqual(
            len(autoregressive_assessments),
            len(heldout["actualTokenIds"]),
        )
        self.assertEqual(
            [item["tokenIndex"] for item in autoregressive_assessments],
            list(range(len(heldout["actualTokenIds"]))),
        )
        self.assertEqual(
            validation["autoregressiveCounterfactualDependencyEvaluations"],
            len(autoregressive_assessments),
        )
        self.assertEqual(
            validation["autoregressiveCounterfactualDependencyAccuracy"],
            1.0,
        )
        if len(autoregressive_assessments) > 1:
            self.assertEqual(
                autoregressive_assessments[1]["generatedPrefixTokenIds"],
                [autoregressive_assessments[0]["actualCounterfactualTokenId"]],
            )
        for assessment in autoregressive_assessments:
            self.assertTrue(assessment["counterfactualOracleQueryIds"])
            self.assertTrue(assessment["baselineOracleQueryIds"])
            self.assertTrue(
                set(assessment["counterfactualOracleQueryIds"]).isdisjoint(
                    training_ids
                )
            )
            self.assertTrue(assessment["counterfactualPredictionMethod"])
            self.assertTrue(assessment["baselinePredictionMethod"])
        for assessment in heldout["dependencyAssessments"]:
            self.assertTrue(assessment["evidenceQueryIds"])
            self.assertTrue(set(assessment["evidenceQueryIds"]).isdisjoint(training_ids))
            self.assertTrue(assessment["reconstructionMethod"])
            self.assertGreaterEqual(assessment["reconstructionConfidence"], 0.0)
            self.assertTrue(assessment["reconstructionEvidenceQueryIds"])
            transfer_estimate = assessment[
                "reconstructionInterventionTransferEstimate"
            ]
            if assessment["reconstructionMethod"] == (
                "calibrated_local_intervention_transfer"
            ):
                self.assertGreater(
                    transfer_estimate["crossValidationRelativeMSEImprovement"],
                    0.0,
                )
                self.assertTrue(transfer_estimate["evidenceQueryIds"])
            self.assertTrue(
                set(assessment["reconstructionEvidenceQueryIds"]).issubset(
                    training_ids
                )
            )
            self.assertTrue(assessment["baselineReconstructionMethod"])
            self.assertTrue(
                assessment["baselineReconstructionEvidenceQueryIds"]
            )

    def test_autoregressive_dependency_validation_catches_later_token_effect(self) -> None:
        class FakeOracle:
            query_count = 0

            def query_token_ids(self, token_ids: list[int]) -> list[float]:
                self.query_count += 1
                logits = [0.0] * 20
                if token_ids[-1] == 2:
                    logits[9 if len(token_ids) >= 6 else 8] = 8.0
                else:
                    logits[2] = 8.0
                return logits

        with tempfile.TemporaryDirectory() as directory:
            report = run_decompilation(
                oracle=FakeOracle(),
                tokenizer=FakeDecompilerTokenizer(),
                prompt="alpha beta gamma delta epsilon",
                model_identity={
                    "id": "later-token-dependency",
                    "modelType": "synthetic",
                },
                session_dir=directory,
                query_budget=100,
                max_new_tokens=2,
                probes_per_token=1,
                explanation_probes=0,
                interaction_probes=1,
                branch_probes=1,
                branch_width=1,
                top_token_count=2,
                response_probes=0,
                resume=False,
                formulation_probes=0,
            )
            explanation_path = Path(directory) / "autoregressive-explanation.md"
            write_explanation_markdown(explanation_path, report)
            explanation_markdown = explanation_path.read_text(encoding="utf-8")

        heldout = next(
            item for item in report["validation"]["cases"]
            if item["role"] == "heldout_counterfactual"
        )
        assessments = heldout["autoregressiveDependencyAssessments"]
        self.assertEqual(len(assessments), 2)
        self.assertEqual(
            [item["actualDependency"] for item in assessments],
            [False, True],
        )
        self.assertEqual(assessments[1]["generatedPrefixTokenIds"], [2])
        self.assertIn(
            "Autoregressive counterfactual dependency decisions",
            explanation_markdown,
        )
        self.assertIn(
            f"`{assessments[1]['actualDependency']}` -> "
            f"`{assessments[1]['reconstructedDependency']}`",
            explanation_markdown,
        )
        self.assertIn("counterfactual `", explanation_markdown)

    def test_dependency_prediction_metrics_include_earlier_validation_cases(self) -> None:
        metrics = _dependency_prediction_metrics([
            {
                "dependencyAssessments": [
                    {"reconstructionMethod": "calibrated_local_intervention_transfer"},
                    {
                        "reconstructionMethod": (
                            "calibrated_local_intervention_and_pair_interaction_transfer"
                        ),
                    },
                    {"reconstructionMethod": "similarity_weighted_observation_vote"},
                ],
            },
            {"dependencyAssessments": []},
        ])

        self.assertEqual(
            metrics["methodCounts"],
            {
                "calibrated_local_intervention_transfer": 1,
                "calibrated_local_intervention_and_pair_interaction_transfer": 1,
                "similarity_weighted_observation_vote": 1,
            },
        )
        self.assertAlmostEqual(metrics["interventionModelCoverage"], 2 / 3)

    def test_deletion_probes_rank_prompt_words_by_observed_margin_effect(self) -> None:
        tokenizer = FakeTokenizer()
        observations = {
            "A B C": [5.0, 3.0, 1.0, 0.0, 0.0, 0.0],
            "B C": [2.0, 4.0, 1.0, 0.0, 0.0, 0.0],
            "A C": [5.0, 0.0, 2.0, 0.0, 0.0, 0.0],
            "A B": [4.0, 2.0, 1.0, 0.0, 0.0, 0.0],
        }
        queries: list[str] = []

        def oracle(prompt: str) -> list[float]:
            queries.append(prompt)
            return observations[prompt]

        result = analyze_next_token_decision(
            oracle_query=oracle,
            tokenizer=tokenizer,
            prompt="A B C",
            target_token="target",
            contrast_tokens=["rival"],
            max_probes=3,
            top_token_count=3,
        )

        by_word = {item["surface"]: item for item in result["counterfactuals"]}
        self.assertEqual(len(queries), 4)
        self.assertEqual(result["queryCount"], len(queries))
        self.assertEqual(tokenizer.encoded_inputs[:2], ["target", "rival"])
        self.assertEqual(by_word["A"]["interpretation"], "supports_target")
        self.assertAlmostEqual(by_word["A"]["marginSupport"], 4.0)
        self.assertEqual(by_word["B"]["interpretation"], "suppresses_target")
        self.assertAlmostEqual(by_word["B"]["marginSupport"], -3.0)
        self.assertEqual(by_word["C"]["prompt"], "A B")
        self.assertGreaterEqual(by_word["A"]["distributionJensenShannonBits"], 0.0)
        self.assertIn("unique causal mechanism", result["claimBoundary"])

    def test_probe_budget_samples_across_long_prompt_and_records_exact_prompts(self) -> None:
        tokenizer = FakeTokenizer()
        outputs = {
            "A B C A B C": [5.0, 2.0, 1.0, 0.0, 0.0, 0.0],
            "B C A B C": [4.0, 2.0, 1.0, 0.0, 0.0, 0.0],
            "A B C A B": [4.0, 2.0, 1.0, 0.0, 0.0, 0.0],
        }
        queried: list[str] = []

        def oracle(prompt: str) -> list[float]:
            queried.append(prompt)
            return outputs[prompt]

        result = analyze_next_token_decision(
            oracle_query=oracle,
            tokenizer=tokenizer,
            prompt="A B C A B C",
            target_token="target",
            max_probes=2,
            top_token_count=2,
        )
        self.assertEqual(len(result["counterfactuals"]), 2)
        self.assertEqual(result["queryCount"], 3)
        self.assertEqual(result["summary"]["eligiblePromptSpanCount"], 6)
        self.assertEqual(result["summary"]["selectedPromptSpanCount"], 2)
        self.assertEqual(result["summary"]["unprobedPromptSpanCount"], 4)
        self.assertEqual(queried[1:], [item["prompt"] for item in result["counterfactuals"]])
        self.assertIn(0, [item["spanIndex"] for item in result["counterfactuals"]])
        self.assertIn(5, [item["spanIndex"] for item in result["counterfactuals"]])

    def test_chat_control_tokens_are_excluded_from_word_deletions(self) -> None:
        tokenizer = FakeTokenizer()
        observations = {
            "<|im_start|>user A B <|im_end|>": [5.0, 3.0, 1.0, 0.0, 0.0, 0.0],
            "<|im_start|>user B <|im_end|>": [4.0, 3.0, 1.0, 0.0, 0.0, 0.0],
            "<|im_start|>user A <|im_end|>": [4.0, 3.0, 1.0, 0.0, 0.0, 0.0],
        }
        queries: list[str] = []

        def oracle(prompt: str) -> list[float]:
            queries.append(prompt)
            return observations[prompt]

        result = analyze_next_token_decision(
            oracle_query=oracle,
            tokenizer=tokenizer,
            prompt="<|im_start|>user A B <|im_end|>",
            target_token="target",
            contrast_tokens=["rival"],
            max_probes=8,
        )
        self.assertEqual(result["summary"]["excludedControlTokenCount"], 2)
        self.assertEqual([item["surface"] for item in result["counterfactuals"]], ["A", "B"])
        self.assertEqual(result["queryCount"], 3)
        self.assertEqual(len(queries), 3)

    def test_rejects_multitoken_targets_and_invalid_budgets_before_querying(self) -> None:
        tokenizer = FakeTokenizer()
        calls = 0

        def oracle(_: str) -> list[float]:
            nonlocal calls
            calls += 1
            return [1.0] * 6

        with self.assertRaisesRegex(ValueError, "exactly one"):
            analyze_next_token_decision(
                oracle_query=oracle,
                tokenizer=tokenizer,
                prompt="A B",
                target_token="not-in-vocabulary",
            )
        with self.assertRaisesRegex(ValueError, "non-negative"):
            analyze_next_token_decision(
                oracle_query=oracle,
                tokenizer=tokenizer,
                prompt="A B",
                target_token="target",
                max_probes=-1,
            )
        self.assertEqual(calls, 0)

    def test_autoregressive_trace_reuses_exact_token_context_and_counts_interventions(self) -> None:
        tokenizer = FakeGenerationTokenizer()
        queried: list[list[int]] = []

        def oracle(token_ids: list[int]) -> list[float]:
            queried.append(list(token_ids))
            logits = [0.0] * 15
            if 2 in token_ids:
                logits[9] = 5.0
                logits[2] = 2.0
            elif 10 in token_ids:
                logits[2] = 5.0
                logits[3] = 4.0
            else:
                logits[3] = 5.0
                logits[2] = 4.0
            return logits

        trace = trace_greedy_generation(
            oracle_query_token_ids=oracle,
            tokenizer=tokenizer,
            input_token_ids=[10, 11, 12],
            max_new_tokens=4,
            probes_per_token=2,
            top_token_count=2,
        )
        self.assertEqual(trace["generatedTokenIds"], [2, 9])
        self.assertEqual(trace["generatedText"], " answer")
        self.assertTrue(trace["outputComplete"])
        self.assertEqual(trace["terminationReason"], "eos")
        self.assertEqual([step["chosenTokenId"] for step in trace["steps"]], [2, 9])
        self.assertEqual(trace["steps"][0]["contextTokenInterventions"][0]["inputToken"], "word-a")
        self.assertEqual(trace["steps"][0]["strongestSupportingToken"]["inputToken"], "word-a")
        self.assertEqual(trace["steps"][1]["contextTokenIds"], [10, 11, 12, 2])
        self.assertEqual(queried[4], [10, 12, 2])
        self.assertEqual(queried[5], [10, 11, 12])
        self.assertEqual(trace["steps"][1]["strongestSupportingToken"]["source"], "generated_prefix")
        self.assertEqual(trace["queryCount"], len(queried))
        self.assertEqual(trace["queryCount"], 6)

    def test_autoregressive_trace_marks_token_cap_as_incomplete(self) -> None:
        class NoEosTokenizer(FakeGenerationTokenizer):
            eos_token_id = None

        trace = trace_greedy_generation(
            oracle_query_token_ids=lambda _: [0.0, 0.0, 5.0, *([0.0] * 12)],
            tokenizer=NoEosTokenizer(),
            input_token_ids=[10],
            max_new_tokens=1,
            probes_per_token=0,
            top_token_count=2,
        )

        self.assertEqual(trace["generatedTokenIds"], [2])
        self.assertFalse(trace["outputComplete"])
        self.assertEqual(trace["terminationReason"], "max_new_tokens")

    def test_predict_next_token_uses_generated_prefix_kernel(self) -> None:
        reconstruction = {
            "observations": [
                {
                    "contextTokenIds": [10, 11, 99],
                    "generatedPrefixTokenIds": [2],
                    "queryId": "prefix-1",
                    "topCandidates": [
                        {"tokenId": 3, "probability": 0.8},
                        {"tokenId": 4, "probability": 0.2},
                    ],
                },
                {
                    "contextTokenIds": [10, 31, 99],
                    "generatedPrefixTokenIds": [2],
                    "queryId": "prefix-2",
                    "topCandidates": [
                        {"tokenId": 5, "probability": 0.7},
                        {"tokenId": 3, "probability": 0.3},
                    ],
                },
            ],
        }
        prediction = predict_next_token(
            reconstruction,
            [10, 11, 2],
            generated_prefix_token_ids=[2],
        )
        self.assertEqual(prediction["method"], "autoregressive_state_kernel")
        self.assertEqual(prediction["tokenId"], 3)
        self.assertGreaterEqual(prediction["confidence"], 0.5)
        self.assertEqual(
            set(prediction["evidenceQueryIds"]),
            {"prefix-1", "prefix-2"},
        )

    def test_validation_excludes_token_capped_outputs_from_full_response_fidelity(self) -> None:
        class NoEosTokenizer(FakeChatTemplateTokenizer):
            eos_token_id = None

        class ConstantOracle:
            query_count = 0

            def query_token_ids(self, token_ids: list[int]) -> list[float]:
                del token_ids
                self.query_count += 1
                logits = [0.0] * 20
                logits[2] = 5.0
                return logits

        with tempfile.TemporaryDirectory() as directory:
            report = run_decompilation(
                oracle=ConstantOracle(),
                tokenizer=NoEosTokenizer(),
                prompt="alpha beta",
                model_identity={"id": "synthetic-no-eos"},
                session_dir=directory,
                query_budget=100,
                max_new_tokens=1,
                probes_per_token=0,
                explanation_probes=0,
                interaction_probes=0,
                branch_probes=0,
                branch_width=1,
                top_token_count=2,
                response_probes=0,
                resume=False,
                formulation_probes=0,
            )
            explanation_path = Path(directory) / "explanation.md"
            write_explanation_markdown(explanation_path, report)
            explanation_markdown = explanation_path.read_text(encoding="utf-8")

        self.assertFalse(report["trace"]["outputComplete"])
        self.assertIsNone(report["validation"]["fullResponseExactMatchRate"])
        self.assertEqual(report["validation"]["completeOracleResponseCount"], 0)
        self.assertIn(
            "not measured (0 complete oracle responses)",
            explanation_markdown,
        )
        self.assertIn(
            "not measured (0 token positions)",
            explanation_markdown,
        )
        self.assertTrue(report["validation"]["cases"])
        self.assertTrue(all(
            not case["actualTerminatedByEos"]
            and case["fullResponseExactMatch"] is None
            for case in report["validation"]["cases"]
        ))

    def test_validation_free_run_stops_at_eos_and_reports_distribution_and_edit_metrics(self) -> None:
        tokenizer = FakeGenerationTokenizer()
        reconstruction = {
            "inputPromptText": "unused",
            "observations": [
                {
                    "contextTokenIds": [10],
                    "generatedPrefixTokenIds": [],
                    "queryId": "q-prompt",
                    "topCandidates": [
                        {"tokenId": 9, "logit": 6.0},
                        {"tokenId": 8, "logit": 5.0},
                        {"tokenId": 7, "logit": 4.0},
                        {"tokenId": 6, "logit": 3.0},
                        {"tokenId": 5, "logit": 2.0},
                    ],
                },
                {
                    "contextTokenIds": [10, 2],
                    "generatedPrefixTokenIds": [2],
                    "queryId": "q-prefix-2",
                    "topCandidates": [
                        {"tokenId": 3, "logit": 6.0},
                        {"tokenId": 2, "logit": 5.0},
                        {"tokenId": 4, "logit": 4.0},
                        {"tokenId": 5, "logit": 3.0},
                        {"tokenId": 6, "logit": 2.0},
                    ],
                },
                {
                    "contextTokenIds": [10, 2, 3],
                    "generatedPrefixTokenIds": [2, 3],
                    "queryId": "q-prefix-2-3",
                    "topCandidates": [
                        {"tokenId": 9, "logit": 6.0},
                        {"tokenId": 3, "logit": 5.0},
                        {"tokenId": 2, "logit": 4.0},
                        {"tokenId": 4, "logit": 3.0},
                        {"tokenId": 5, "logit": 2.0},
                    ],
                },
            ],
        }

        def oracle(context: list[int]) -> list[float]:
            logits = [0.0] * 20
            winner = 2 if context == [10] else 3 if context == [10, 2] else 9
            logits[winner] = 6.0
            logits[3 if winner != 3 else 2] = 5.0
            logits[4] = 4.0
            logits[5] = 3.0
            logits[6] = 2.0
            return logits

        metrics = _validation_metrics(
            reconstruction,
            prompt_cases=[{
                "id": "early-eos",
                "role": "heldout_formulation",
                "inputTokenIds": [10],
                "prompt": "test",
            }],
            query=oracle,
            oracle_records=[],
            tokenizer=tokenizer,
            max_new_tokens=4,
            baseline_token_id=2,
        )
        case = metrics["cases"][0]

        self.assertEqual(case["actualTokenIds"], [2, 3, 9])
        self.assertEqual(case["predictedTokenIds"], [9])
        self.assertTrue(case["predictedTerminatedByEos"])
        self.assertEqual(case["predictedTerminationReason"], "eos")
        self.assertEqual(case["sequenceEditDistance"], 2)
        self.assertAlmostEqual(case["normalizedSequenceEditSimilarity"], 1 / 3)
        self.assertEqual(
            case["predictionEvidence"][0]["conditional"]["topFiveAgreement"],
            0.4,
        )
        self.assertEqual(metrics["teacherForcedNextTokenAccuracy"], 2 / 3)
        self.assertAlmostEqual(metrics["topFivePredictionAgreement"], 11 / 15)
        self.assertEqual(metrics["eosPredictionAccuracy"], 2 / 3)
        self.assertEqual(metrics["freeRunningRolloutFidelity"], 0.0)
        self.assertEqual(metrics["meanSequenceEditDistance"], 2.0)
        self.assertIsNotNone(
            metrics["teacherForcedTopFiveDistributionFidelity"]
        )
        self.assertEqual(_sequence_edit_distance([1, 2], [1, 3, 2]), 1)

    def test_response_counterfactual_regenerates_answer_after_measured_prompt_deletion(self) -> None:
        tokenizer = FakeGenerationTokenizer()
        queried: list[list[int]] = []

        def oracle(token_ids: list[int]) -> list[float]:
            queried.append(list(token_ids))
            logits = [0.0] * 15
            if 2 in token_ids or 3 in token_ids:
                logits[9] = 5.0
                logits[2] = 1.0
            elif 10 in token_ids:
                logits[2] = 5.0
                logits[3] = 4.0
            else:
                logits[3] = 5.0
                logits[2] = 4.0
            return logits

        trace = trace_greedy_generation(
            oracle_query_token_ids=oracle,
            tokenizer=tokenizer,
            input_token_ids=[10, 11, 12],
            max_new_tokens=4,
            probes_per_token=2,
            top_token_count=2,
            response_probe_count=1,
        )

        self.assertEqual(trace["generatedTokenIds"], [2, 9])
        self.assertEqual(len(trace["responseCounterfactuals"]), 1)
        counterfactual = trace["responseCounterfactuals"][0]
        self.assertEqual(counterfactual["deletedInputTokenId"], 10)
        self.assertEqual(counterfactual["counterfactualInputTokenIds"], [11, 12])
        self.assertEqual(counterfactual["counterfactualGeneratedTokenIds"], [3, 9])
        self.assertEqual(counterfactual["counterfactualGeneratedText"], " alternative")
        self.assertEqual(counterfactual["commonPrefixTokenCount"], 0)
        self.assertEqual(counterfactual["firstDivergentTokenIndex"], 0)
        self.assertTrue(counterfactual["responseChanged"])
        self.assertEqual(trace["responseProbeSummary"]["eligibleMeasuredInputTokens"], 2)
        self.assertEqual(trace["responseProbeSummary"]["oracleQueryCount"], 2)
        self.assertEqual(trace["queryCount"], len(queried))
        self.assertEqual(trace["queryCount"], 8)

    def test_response_counterfactual_budget_is_validated_before_oracle_queries(self) -> None:
        calls = 0

        def oracle(_: list[int]) -> list[float]:
            nonlocal calls
            calls += 1
            return [1.0] * 15

        with self.assertRaisesRegex(ValueError, "response_probe_count must be non-negative"):
            trace_greedy_generation(
                oracle_query_token_ids=oracle,
                tokenizer=FakeGenerationTokenizer(),
                input_token_ids=[10, 11],
                response_probe_count=-1,
            )
        self.assertEqual(calls, 0)

    def test_trace_measures_pairwise_nonadditivity_and_accounts_for_query(self) -> None:
        tokenizer = FakeGenerationTokenizer()
        queried: list[list[int]] = []

        def oracle(token_ids: list[int]) -> list[float]:
            queried.append(list(token_ids))
            logits = [0.0] * 15
            has_a = 10 in token_ids
            has_b = 11 in token_ids
            logits[2] = 10.0 if has_a and has_b else 8.0 if has_a or has_b else 10.0
            logits[3] = 5.0 if has_a and has_b else 4.0 if has_a or has_b else 0.0
            return logits

        trace = trace_greedy_generation(
            oracle_query_token_ids=oracle,
            tokenizer=tokenizer,
            input_token_ids=[10, 11, 12],
            max_new_tokens=1,
            probes_per_token=2,
            top_token_count=2,
            interaction_probe_count=1,
        )

        interaction = trace["steps"][0]["pairwiseInterventions"][0]
        self.assertEqual(
            [unit["tokenIds"] for unit in interaction["deletedUnits"]],
            [[10], [11]],
        )
        self.assertEqual(interaction["targetVsContrastMargin"], 10.0)
        self.assertEqual(interaction["additivePredictedMargin"], 3.0)
        self.assertEqual(interaction["interactionResidual"], 7.0)
        self.assertEqual(trace["interactionQueryCount"], 1)
        self.assertEqual(trace["queryCount"], len(queried))
        self.assertEqual(trace["queryCount"], 4)

    def test_negative_interaction_budget_is_rejected_before_oracle_queries(self) -> None:
        calls = 0

        def oracle(_: list[int]) -> list[float]:
            nonlocal calls
            calls += 1
            return [1.0] * 15

        with self.assertRaisesRegex(ValueError, "interaction_probe_count must be non-negative"):
            trace_greedy_generation(
                oracle_query_token_ids=oracle,
                tokenizer=FakeGenerationTokenizer(),
                input_token_ids=[10, 11],
                interaction_probe_count=-1,
            )
        self.assertEqual(calls, 0)

    def test_trace_follows_forced_alternative_with_width_one_beam(self) -> None:
        tokenizer = FakeGenerationTokenizer()
        queried: list[list[int]] = []

        def oracle(token_ids: list[int]) -> list[float]:
            queried.append(list(token_ids))
            logits = [0.0] * 15
            if 2 in token_ids:
                logits[9] = 5.0
            elif 3 in token_ids:
                logits[2] = 5.0
            else:
                logits[2] = 5.0
                logits[3] = 4.0
            return logits

        trace = trace_greedy_generation(
            oracle_query_token_ids=oracle,
            tokenizer=tokenizer,
            input_token_ids=[10],
            max_new_tokens=3,
            probes_per_token=0,
            top_token_count=2,
            response_probe_count=0,
            explanation_probes_per_token=0,
            interaction_probe_count=0,
            branch_probe_count=1,
            branch_width=1,
        )

        self.assertEqual(trace["generatedTokenIds"], [2, 9])
        branch = trace["steps"][0]["alternativeBranches"][0]
        self.assertEqual(branch["forcedTokenId"], 3)
        self.assertEqual(branch["branchTokenIds"], [3, 2, 9])
        self.assertEqual(branch["branchText"], " alternative answer")
        self.assertEqual(branch["firstDivergentTokenIndex"], 0)
        self.assertEqual(branch["oracleQueryCount"], 2)
        self.assertEqual(trace["branchQueryCount"], 2)
        self.assertEqual(trace["queryCount"], len(queried))
        self.assertEqual(trace["queryCount"], 4)

    def test_branch_beam_returns_requested_ranked_continuations(self) -> None:
        tokenizer = FakeGenerationTokenizer()
        queried: list[list[int]] = []

        def oracle(token_ids: list[int]) -> list[float]:
            queried.append(list(token_ids))
            logits = [0.0] * 15
            if token_ids == [10]:
                logits[2] = 5.0
                logits[3] = 4.0
            elif token_ids == [10, 3]:
                logits[4] = 3.0
                logits[5] = 2.0
            else:
                logits[9] = 5.0
            return logits

        trace = trace_greedy_generation(
            oracle_query_token_ids=oracle,
            tokenizer=tokenizer,
            input_token_ids=[10],
            max_new_tokens=2,
            probes_per_token=0,
            top_token_count=2,
            response_probe_count=0,
            explanation_probes_per_token=0,
            interaction_probe_count=0,
            branch_probe_count=1,
            branch_width=2,
        )

        branch = trace["steps"][0]["alternativeBranches"][0]
        self.assertEqual(branch["requestedBeamWidth"], 2)
        self.assertEqual(branch["observedContinuationCount"], 2)
        self.assertEqual(
            [item["tokenIds"] for item in branch["continuations"]],
            [[3, 4], [3, 5]],
        )
        self.assertGreater(
            branch["continuations"][0]["cumulativeLogProbability"],
            branch["continuations"][1]["cumulativeLogProbability"],
        )
        self.assertEqual(branch["oracleQueryCount"], 1)
        self.assertEqual(trace["branchQueryCount"], 1)
        self.assertEqual(trace["queryCount"], len(queried))
        self.assertEqual(queried.count([10, 3]), 1)

    def test_branch_beam_stops_expanding_eos_and_accounts_for_queries(self) -> None:
        tokenizer = FakeGenerationTokenizer()
        queried: list[list[int]] = []

        def oracle(token_ids: list[int]) -> list[float]:
            queried.append(list(token_ids))
            logits = [0.0] * 15
            if token_ids == [10]:
                logits[2] = 5.0
                logits[3] = 4.0
            elif token_ids == [10, 3]:
                logits[9] = 5.0
            else:
                logits[9] = 5.0
            return logits

        trace = trace_greedy_generation(
            oracle_query_token_ids=oracle,
            tokenizer=tokenizer,
            input_token_ids=[10],
            max_new_tokens=3,
            probes_per_token=0,
            top_token_count=2,
            response_probe_count=0,
            explanation_probes_per_token=0,
            interaction_probe_count=0,
            branch_probe_count=1,
            branch_width=1,
        )

        branch = trace["steps"][0]["alternativeBranches"][0]
        self.assertEqual(
            branch["branchContinuationTokenIds"],
            [3, 9],
        )
        self.assertEqual(branch["oracleQueryCount"], 1)
        self.assertEqual(trace["branchQueryCount"], 1)
        self.assertNotIn([10, 3, 9], queried)
        self.assertEqual(trace["queryCount"], len(queried))

    def test_negative_branch_budget_is_rejected_before_oracle_queries(self) -> None:
        calls = 0

        def oracle(_: list[int]) -> list[float]:
            nonlocal calls
            calls += 1
            return [1.0] * 15

        with self.assertRaisesRegex(ValueError, "branch_probe_count must be non-negative"):
            trace_greedy_generation(
                oracle_query_token_ids=oracle,
                tokenizer=FakeGenerationTokenizer(),
                input_token_ids=[10],
                branch_probe_count=-1,
            )
        self.assertEqual(calls, 0)

    def test_nonpositive_branch_width_is_rejected_before_oracle_queries(self) -> None:
        calls = 0

        def oracle(_: list[int]) -> list[float]:
            nonlocal calls
            calls += 1
            return [1.0] * 15

        with self.assertRaisesRegex(ValueError, "branch_width must be positive"):
            trace_greedy_generation(
                oracle_query_token_ids=oracle,
                tokenizer=FakeGenerationTokenizer(),
                input_token_ids=[10],
                branch_width=0,
            )
        self.assertEqual(calls, 0)

    def test_trace_finds_one_minimal_sufficient_context_with_deletion_witnesses(self) -> None:
        tokenizer = ChatFakeGenerationTokenizer()
        queried: list[list[int]] = []

        def oracle(token_ids: list[int]) -> list[float]:
            queried.append(list(token_ids))
            logits = [0.0] * 15
            if 10 in token_ids and 11 in token_ids:
                logits[2] = 5.0
                logits[3] = 4.0
            else:
                logits[2] = 3.0
                logits[3] = 5.0
            return logits

        trace = trace_greedy_generation(
            oracle_query_token_ids=oracle,
            tokenizer=tokenizer,
            input_token_ids=[12, 14, 15, 10, 11, 13, 8],
            max_new_tokens=1,
            probes_per_token=0,
            top_token_count=2,
            explanation_probes_per_token=8,
        )

        evidence = trace["steps"][0]["minimalSufficientContext"]
        self.assertEqual(trace["generatedTokenIds"], [2])
        self.assertEqual(evidence["retainedContextTokenIds"], [12, 14, 15, 10, 11, 8])
        self.assertEqual(evidence["retainedEditableUnitCount"], 2)
        self.assertEqual(
            evidence["minimality"]["scope"],
            "all_editable_context_units",
        )
        self.assertEqual(evidence["excludedHeldOutInputTokenIndices"], [])
        self.assertTrue(evidence["targetRemainsTop1"])
        self.assertTrue(evidence["minimality"]["oneMinimalAmongEditableUnits"])
        self.assertFalse(evidence["minimality"]["globalMinimumProven"])
        self.assertEqual(
            [item["removedContextTokenIds"] for item in evidence["singleTokenDeletionWitnesses"]],
            [[10], [11]],
        )
        self.assertEqual(
            [item["counterfactualWinnerTokenId"] for item in evidence["singleTokenDeletionWitnesses"]],
            [3, 3],
        )
        self.assertEqual(evidence["oracleQueryCount"], 4)
        self.assertEqual(trace["explanationQueryCount"], 4)
        self.assertEqual(trace["queryCount"], len(queried))

    def test_trace_rotates_input_probes_to_cover_unmeasured_context_units(self) -> None:
        tokenizer = FakeGenerationTokenizer()
        queried: list[list[int]] = []

        def oracle(token_ids: list[int]) -> list[float]:
            queried.append(list(token_ids))
            logits = [0.0] * 15
            logits[2] = 5.0
            logits[3] = 4.0
            return logits

        trace = trace_greedy_generation(
            oracle_query_token_ids=oracle,
            tokenizer=tokenizer,
            input_token_ids=[10, 11, 13, 14, 15],
            max_new_tokens=4,
            probes_per_token=2,
            top_token_count=2,
        )

        measured_input_indexes = [
            item["contextTokenIndex"]
            for step in trace["steps"]
            for item in step["contextTokenInterventions"]
            if item["source"] == "input"
        ]
        self.assertEqual(measured_input_indexes, [0, 1, 2, 3, 4])
        self.assertEqual(trace["queryCount"], len(queried))

    def test_sufficient_context_search_reports_budget_limited_result(self) -> None:
        tokenizer = ChatFakeGenerationTokenizer()
        queried: list[list[int]] = []

        def oracle(token_ids: list[int]) -> list[float]:
            queried.append(list(token_ids))
            logits = [0.0] * 15
            logits[2] = 5.0 if 10 in token_ids and 11 in token_ids else 1.0
            logits[3] = 4.0 if 10 in token_ids and 11 in token_ids else 5.0
            return logits

        trace = trace_greedy_generation(
            oracle_query_token_ids=oracle,
            tokenizer=tokenizer,
            input_token_ids=[12, 14, 15, 10, 11, 13, 8],
            max_new_tokens=1,
            probes_per_token=0,
            top_token_count=2,
            explanation_probes_per_token=1,
        )

        evidence = trace["steps"][0]["minimalSufficientContext"]
        self.assertFalse(evidence["minimality"]["oneMinimalAmongEditableUnits"])
        self.assertTrue(evidence["minimality"]["budgetExhausted"])
        self.assertEqual(evidence["oracleQueryCount"], 1)
        self.assertEqual(trace["queryCount"], len(queried))


if __name__ == "__main__":
    unittest.main()
