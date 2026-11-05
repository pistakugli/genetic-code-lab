#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import random
import uuid
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from multiprocessing.connection import Connection
from transformers.utils import logging as transformers_logging

from llm_reader.blackbox_decision import analyze_next_token_decision, trace_greedy_generation
from llm_reader.blackbox_decompiler import (
    _training_formulations,
    encode_prompt,
    generate_reconstruction,
    load_executable_reconstruction,
    run_decompilation,
    write_explanation_markdown,
)

transformers_logging.disable_progress_bar()


FACTS: List[Dict[str, str]] = [
    {"subject": "France", "answer": "Paris"},
    {"subject": "Germany", "answer": "Berlin"},
    {"subject": "Italy", "answer": "Rome"},
    {"subject": "Spain", "answer": "Madrid"},
    {"subject": "Portugal", "answer": "Lisbon"},
    {"subject": "Greece", "answer": "Athens"},
    {"subject": "Egypt", "answer": "Cairo"},
    {"subject": "Japan", "answer": "Tokyo"},
    {"subject": "China", "answer": "Beijing"},
    {"subject": "Russia", "answer": "Moscow"},
    {"subject": "Canada", "answer": "Ottawa"},
    {"subject": "Australia", "answer": "Canberra"},
    {"subject": "Brazil", "answer": "Brasilia"},
    {"subject": "Sweden", "answer": "Stockholm"},
    {"subject": "Norway", "answer": "Oslo"},
    {"subject": "Denmark", "answer": "Copenhagen"},
    {"subject": "Finland", "answer": "Helsinki"},
    {"subject": "Poland", "answer": "Warsaw"},
    {"subject": "Austria", "answer": "Vienna"},
    {"subject": "Belgium", "answer": "Brussels"},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Output-only black-box LLM behavior reader."
    )
    parser.add_argument(
        "--mode",
        choices=["facts", "decision", "trace", "decompile", "predict"],
        default="facts",
    )
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--prompt")
    parser.add_argument("--target-token")
    parser.add_argument("--contrast-token", action="append", default=[])
    parser.add_argument("--max-probes", type=int, default=24)
    parser.add_argument("--top-logits", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--prediction-beam-width", type=int, default=1)
    parser.add_argument("--probes-per-token", type=int, default=4)
    parser.add_argument("--response-probes", type=int, default=2)
    parser.add_argument("--explanation-probes", type=int, default=8)
    parser.add_argument("--interaction-probes", type=int, default=2)
    parser.add_argument("--branch-probes", type=int, default=1)
    parser.add_argument("--branch-width", type=int, default=3)
    parser.add_argument("--query-budget", type=int, default=512)
    parser.add_argument("--formulation-probes", type=int, default=1)
    parser.add_argument(
        "--prompt-format",
        choices=["auto", "raw", "chat-template"],
        default=None,
        help="Prompt encoding; auto uses the tokenizer chat template when available.",
    )
    parser.add_argument("--session-dir")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reconstruction")
    parser.add_argument("--case-count", type=int, default=10)
    parser.add_argument("--min-valid-cases", type=int, default=10)
    parser.add_argument("--only-subjects", default="")
    parser.add_argument("--control-count", type=int, default=8)
    parser.add_argument("--paraphrase-count", type=int, default=3)
    parser.add_argument("--wrong-answer-count", type=int, default=4)
    parser.add_argument("--max-target-rank", type=int, default=100)
    parser.add_argument("--min-pass-rate", type=float, default=1.0)
    parser.add_argument("--min-subject-lift", type=float, default=0.05)
    parser.add_argument("--min-relation-lift", type=float, default=0.02)
    parser.add_argument("--min-paraphrase-retention-rate", type=float, default=0.50)
    parser.add_argument("--control-seed", type=int, default=260709)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output-json")
    parser.add_argument("--output-md")
    return parser.parse_args()


def package_version(name: str) -> Optional[str]:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def clean_token(tokenizer: Any, token_id: int) -> str:
    return tokenizer.decode([int(token_id)]).replace("\n", "\\n")


def percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = min(len(ordered) - 1, max(0.0, (len(ordered) - 1) * q))
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    fraction = position - lower
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction)


def median(values: List[float]) -> float:
    return percentile(values, 0.5)


def encode_single_answer_token(tokenizer: Any, answer: str) -> Optional[Tuple[int, str]]:
    encoded = tokenizer(" " + answer, add_special_tokens=False)["input_ids"]
    if len(encoded) != 1:
        return None
    token_id = int(encoded[0])
    return token_id, clean_token(tokenizer, token_id)


def prompt_for_fact(fact: Dict[str, str]) -> str:
    return fact.get("prompt") or f"The capital of {fact['subject']} is"


def paraphrase_prompts(subject: str, count: int) -> List[str]:
    candidates = [
        f"The capital city of {subject} is",
        f"{subject} has its capital in",
        f"{subject} is governed from",
        f"The city that is the capital of {subject} is",
        f"The main capital of {subject} is",
        f"{subject}'s main capital is",
        f"{subject}'s capital is",
        f"For {subject}, the capital is",
        f"In {subject}, the capital is",
        f"In {subject}, the capital city is",
    ]
    return candidates[: max(0, count)]


def relation_control_prompts(subject: str) -> List[str]:
    return [
        f"The currency of {subject} is",
        f"The main language of {subject} is",
        f"The largest city in {subject} is",
        f"The weather in {subject} is",
    ]


def resolve_device(value: str) -> str:
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    return value


class BlackBoxNextTokenOracle:
    """The only model-facing object. The reader below receives logits only."""

    def __init__(self, model_id: str, *, local_files_only: bool, device: str) -> None:
        self.model_id = model_id
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=local_files_only)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, local_files_only=local_files_only)
        self.model.eval()
        self.model.to(self.device)
        self.query_count = 0
        self.local_files_only = bool(local_files_only)

    def query_logits(self, prompt: str) -> torch.Tensor:
        encoded = self.tokenizer(prompt, return_tensors="pt")
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        self.query_count += 1
        with torch.no_grad():
            return self.model(**encoded).logits.detach().cpu()[0, -1].float()

    def query_token_ids(self, token_ids: List[int]) -> torch.Tensor:
        if not token_ids:
            raise ValueError("At least one input token is required.")
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids)
        self.query_count += 1
        with torch.no_grad():
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits.detach().cpu()[0, -1].float()

    def identity(self) -> Dict[str, Any]:
        config = getattr(self.model, "config", None)
        return {
            "id": self.model_id,
            "class": self.model.__class__.__name__,
            "modelType": getattr(config, "model_type", None),
            "device": str(self.device),
            "localFilesOnly": self.local_files_only,
        }


def _blackbox_oracle_worker(
    connection: Connection,
    model_id: str,
    local_files_only: bool,
    device_name: str,
) -> None:
    try:
        device = torch.device(device_name)
        config = AutoConfig.from_pretrained(model_id, local_files_only=local_files_only)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            local_files_only=local_files_only,
        ).eval().to(device)
    except Exception as error:
        connection.send({
            "status": "error",
            "stage": "initialize",
            "error": f"{type(error).__name__}: {error}",
        })
        connection.close()
        return
    connection.send({
        "status": "ready",
        "identity": {
            "id": model_id,
            "class": model.__class__.__name__,
            "modelType": getattr(config, "model_type", None),
            "device": str(device),
            "localFilesOnly": local_files_only,
        },
    })
    try:
        while True:
            request = connection.recv()
            if request.get("operation") == "close":
                connection.send({"status": "closed"})
                break
            try:
                token_ids = request.get("inputTokenIds")
                if not token_ids:
                    raise ValueError("Oracle worker received an empty token sequence.")
                input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
                attention_mask = torch.ones_like(input_ids)
                with torch.no_grad():
                    logits = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    ).logits[0, -1].float()
                connection.send({"status": "ok", "logits": logits.detach().cpu().tolist()})
            except Exception as error:
                connection.send({
                    "status": "error",
                    "stage": "query",
                    "error": f"{type(error).__name__}: {error}",
                })
    finally:
        connection.close()


class ProcessSeparatedBlackBoxOracle:
    """Keep model weights in the oracle worker; expose only token IDs and logits."""

    def __init__(self, model_id: str, *, local_files_only: bool, device: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            local_files_only=local_files_only,
        )
        context = mp.get_context("spawn")
        parent, child = context.Pipe()
        self._connection = parent
        self._worker = context.Process(
            target=_blackbox_oracle_worker,
            args=(child, model_id, local_files_only, device),
            daemon=True,
        )
        self._worker.start()
        child.close()
        try:
            ready = self._connection.recv()
            if ready.get("status") != "ready":
                raise RuntimeError(f"Oracle worker failed to initialize: {ready!r}")
        except Exception as error:
            self._worker.join()
            self._connection.close()
            if isinstance(error, RuntimeError):
                raise
            raise RuntimeError(
                f"Oracle worker exited during initialization (exit code {self._worker.exitcode})."
            ) from error
        self._identity = ready["identity"]
        self.query_count = 0
        self._closed = False

    def query_logits(self, prompt: str) -> List[float]:
        token_ids = self.tokenizer(prompt, add_special_tokens=True)["input_ids"]
        return self.query_token_ids(token_ids)

    def query_token_ids(self, token_ids: List[int]) -> List[float]:
        if self._closed:
            raise RuntimeError("Oracle worker is closed.")
        self._connection.send({
            "operation": "query",
            "inputTokenIds": [int(token_id) for token_id in token_ids],
        })
        self.query_count += 1
        response = self._connection.recv()
        if response.get("status") != "ok" or not isinstance(response.get("logits"), list):
            raise RuntimeError(f"Oracle query failed: {response!r}")
        return response["logits"]

    def identity(self) -> Dict[str, Any]:
        return dict(self._identity)

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self._worker.is_alive():
                self._connection.send({"operation": "close"})
                response = self._connection.recv()
                if response.get("status") != "closed":
                    raise RuntimeError(f"Oracle worker failed to close cleanly: {response!r}")
        finally:
            self._worker.join()
            self._connection.close()
            self._closed = True

    def __enter__(self) -> "ProcessSeparatedBlackBoxOracle":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        del exc_type, exc_value, traceback
        self.close()


def build_answer_pool(tokenizer: Any, facts: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    pool = []
    seen = set()
    for fact in facts:
        encoded = encode_single_answer_token(tokenizer, fact["answer"])
        if encoded is None:
            continue
        token_id, token = encoded
        if token_id in seen:
            continue
        seen.add(token_id)
        pool.append({
            "subject": fact["subject"],
            "answer": fact["answer"],
            "tokenId": token_id,
            "token": token,
        })
    return pool


def top_logit_items(tokenizer: Any, logits: torch.Tensor, count: int) -> List[Dict[str, Any]]:
    limit = min(max(0, count), int(logits.numel()))
    top = torch.topk(logits.detach().cpu(), k=limit)
    return [
        {
            "rank": rank + 1,
            "tokenId": int(index),
            "token": clean_token(tokenizer, int(index)),
            "logit": float(value),
        }
        for rank, (value, index) in enumerate(zip(top.values, top.indices))
    ]


def token_rank(logits: torch.Tensor, token_id: int) -> int:
    target = float(logits[int(token_id)])
    return int(torch.sum(logits > target).item()) + 1


def answer_margin(logits: torch.Tensor, target_token_id: int, answer_pool: List[Dict[str, Any]]) -> float:
    competitors = [
        float(logits[int(item["tokenId"])])
        for item in answer_pool
        if int(item["tokenId"]) != int(target_token_id)
    ]
    if not competitors:
        return 0.0
    return float(logits[int(target_token_id)]) - max(competitors)


def score_prompt(
    oracle_query: Any,
    prompt: str,
    target_token_id: int,
    answer_pool: List[Dict[str, Any]],
) -> Dict[str, Any]:
    logits = oracle_query(prompt)
    return {
        "prompt": prompt,
        "targetLogit": float(logits[int(target_token_id)]),
        "targetRank": token_rank(logits, target_token_id),
        "answerPoolMargin": answer_margin(logits, target_token_id, answer_pool),
    }


def select_facts(args: argparse.Namespace) -> List[Dict[str, str]]:
    selected = FACTS[: max(0, args.case_count)]
    only = {item.strip() for item in args.only_subjects.split(",") if item.strip()}
    if only:
        selected = [fact for fact in FACTS if fact["subject"] in only][: max(0, args.case_count)]
    return selected


def analyze_case(
    *,
    oracle_query: Any,
    tokenizer: Any,
    fact: Dict[str, str],
    answer_pool: List[Dict[str, Any]],
    all_facts: List[Dict[str, str]],
    args: argparse.Namespace,
    rng: random.Random,
) -> Dict[str, Any]:
    encoded = encode_single_answer_token(tokenizer, fact["answer"])
    if encoded is None:
        return {
            "subject": fact["subject"],
            "answer": fact["answer"],
            "valid": False,
            "skipReason": "answer_not_single_token",
        }

    target_token_id, target_token = encoded
    baseline_prompt = prompt_for_fact(fact)
    baseline = score_prompt(oracle_query, baseline_prompt, target_token_id, answer_pool)

    subject_controls = []
    for other in all_facts:
        if other["subject"] == fact["subject"]:
            continue
        subject_controls.append(score_prompt(
            oracle_query,
            f"The capital of {other['subject']} is",
            target_token_id,
            answer_pool,
        ))
        if len(subject_controls) >= args.control_count:
            break

    relation_controls = [
        score_prompt(oracle_query, prompt, target_token_id, answer_pool)
        for prompt in relation_control_prompts(fact["subject"])
    ]

    paraphrase_candidates = [
        score_prompt(oracle_query, prompt, target_token_id, answer_pool)
        for prompt in paraphrase_prompts(fact["subject"], max(args.paraphrase_count, 10))
    ]
    paraphrases = sorted(
        paraphrase_candidates,
        key=lambda item: float(item["answerPoolMargin"]),
        reverse=True,
    )[: max(0, args.paraphrase_count)]

    wrong_pool = [item for item in answer_pool if item["answer"] != fact["answer"]]
    rng.shuffle(wrong_pool)
    wrong_answer_controls = []
    baseline_logits_for_wrong = oracle_query(baseline_prompt)
    for item in wrong_pool[: max(0, args.wrong_answer_count)]:
        wrong_answer_controls.append({
            "answer": item["answer"],
            "tokenId": int(item["tokenId"]),
            "token": item["token"],
            "logit": float(baseline_logits_for_wrong[int(item["tokenId"])]),
            "rank": token_rank(baseline_logits_for_wrong, int(item["tokenId"])),
        })

    subject_margins = [float(item["answerPoolMargin"]) for item in subject_controls]
    paraphrase_margins = [float(item["answerPoolMargin"]) for item in paraphrases]
    baseline_margin = float(baseline["answerPoolMargin"])
    subject_lift = baseline_margin - percentile(subject_margins, 0.95)
    relation_template_margin = median(paraphrase_margins) if paraphrase_margins else baseline_margin
    paraphrase_retention = 0.0
    if paraphrase_margins:
        paraphrase_retention = sum(1 for value in paraphrase_margins if value > 0.0) / len(paraphrase_margins)

    graph = {
        "nodes": [
            {"id": "prompt:subject", "type": "input-text", "label": fact["subject"]},
            {"id": "prompt:relation", "type": "input-text", "label": "capital-of"},
            {"id": "output:target", "type": "next-token", "label": target_token},
        ],
        "edges": [
            {
                "from": "prompt:subject",
                "to": "output:target",
                "score": subject_lift,
                "method": "same-relation subject-swap output-logit intervention",
            },
            {
                "from": "prompt:relation",
                "to": "output:target",
                "score": relation_template_margin,
                "method": "capital-relation paraphrase stability over output logits",
            },
        ],
    }

    pass_case = (
        int(baseline["targetRank"]) <= int(args.max_target_rank)
        and subject_lift >= float(args.min_subject_lift)
        and baseline_margin >= float(args.min_relation_lift)
        and relation_template_margin >= float(args.min_relation_lift)
        and paraphrase_retention >= float(args.min_paraphrase_retention_rate)
    )

    return {
        "subject": fact["subject"],
        "answer": fact["answer"],
        "targetTokenId": target_token_id,
        "targetToken": target_token,
        "valid": True,
        "pass": pass_case,
        "baseline": baseline,
        "subjectControls": subject_controls,
        "relationControls": relation_controls,
        "paraphrases": paraphrases,
        "paraphraseCandidates": paraphrase_candidates,
        "wrongAnswerControls": wrong_answer_controls,
        "behaviorGraph": graph,
        "metrics": {
            "answerSpecificityMargin": baseline_margin,
            "subjectSpecificityLift": subject_lift,
            "relationTemplateMargin": relation_template_margin,
            "paraphraseRetentionRate": paraphrase_retention,
            "subjectControlP95Margin": percentile(subject_margins, 0.95),
            "relationControlP95TargetLogit": percentile([float(item["targetLogit"]) for item in relation_controls], 0.95),
        },
    }


def summarize(cases: List[Dict[str, Any]], query_count: int, args: argparse.Namespace) -> Dict[str, Any]:
    valid = [case for case in cases if case.get("valid")]
    passed = [case for case in valid if case.get("pass")]
    ranks = [int(case["baseline"]["targetRank"]) for case in valid]
    subject_lifts = [float(case["metrics"]["subjectSpecificityLift"]) for case in valid]
    relation_lifts = [float(case["metrics"]["relationTemplateMargin"]) for case in valid]
    answer_margins = [float(case["metrics"]["answerSpecificityMargin"]) for case in valid]
    paraphrase_rates = [float(case["metrics"]["paraphraseRetentionRate"]) for case in valid]
    pass_rate = len(passed) / max(1, len(valid))
    pass_summary = (
        len(valid) >= int(args.min_valid_cases)
        and pass_rate >= float(args.min_pass_rate)
    )
    return {
        "pass": pass_summary,
        "validCases": len(valid),
        "passedCases": len(passed),
        "failedCases": len(valid) - len(passed),
        "skippedCases": len(cases) - len(valid),
        "passRate": pass_rate,
        "queryCount": query_count,
        "medianTargetRank": median([float(value) for value in ranks]),
        "maxTargetRank": max(ranks) if ranks else None,
        "medianSubjectSpecificityLift": median(subject_lifts),
        "minSubjectSpecificityLift": min(subject_lifts) if subject_lifts else None,
        "medianAnswerSpecificityMargin": median(answer_margins),
        "minAnswerSpecificityMargin": min(answer_margins) if answer_margins else None,
        "medianRelationTemplateMargin": median(relation_lifts),
        "minRelationTemplateMargin": min(relation_lifts) if relation_lifts else None,
        "medianParaphraseRetentionRate": median(paraphrase_rates),
        "thresholds": {
            "minValidCases": int(args.min_valid_cases),
            "minPassRate": float(args.min_pass_rate),
            "maxTargetRank": int(args.max_target_rank),
            "minSubjectLift": float(args.min_subject_lift),
            "minAnswerSpecificityMargin": float(args.min_relation_lift),
            "minRelationTemplateMargin": float(args.min_relation_lift),
            "minParaphraseRetentionRate": float(args.min_paraphrase_retention_rate),
        },
    }


def write_markdown(path: str, report: Dict[str, Any]) -> None:
    summary = report["summary"]
    access = report["access"]
    lines = [
        "# Output-only Black-box LLM Behavior Reader",
        "",
        f"- generatedAt: `{report['generatedAt']}`",
        f"- model: `{report['model']['id']}`",
        f"- pass: `{str(summary['pass']).lower()}`",
        f"- validCases: `{summary['validCases']}`",
        f"- passedCases: `{summary['passedCases']}`",
        f"- passRate: `{summary['passRate']}`",
        f"- queryCount: `{summary['queryCount']}`",
        "",
        "## Access Audit",
        "",
        f"- solverAccess: `{access['solverAccess']}`",
        f"- weightsUsedBySolver: `{str(access['weightsUsedBySolver']).lower()}`",
        f"- hooksUsedBySolver: `{str(access['hooksUsedBySolver']).lower()}`",
        f"- postHocWitnessOnly: `{str(access['postHocWitnessOnly']).lower()}`",
        "",
        "## Aggregate Metrics",
        "",
        f"- medianTargetRank: `{summary['medianTargetRank']}`",
        f"- maxTargetRank: `{summary['maxTargetRank']}`",
        f"- medianSubjectSpecificityLift: `{summary['medianSubjectSpecificityLift']}`",
        f"- minSubjectSpecificityLift: `{summary['minSubjectSpecificityLift']}`",
        f"- medianAnswerSpecificityMargin: `{summary['medianAnswerSpecificityMargin']}`",
        f"- minAnswerSpecificityMargin: `{summary['minAnswerSpecificityMargin']}`",
        f"- medianRelationTemplateMargin: `{summary['medianRelationTemplateMargin']}`",
        f"- minRelationTemplateMargin: `{summary['minRelationTemplateMargin']}`",
        f"- medianParaphraseRetentionRate: `{summary['medianParaphraseRetentionRate']}`",
        "",
        "## Case Summary",
        "",
        "| subject | answer | pass | rank | subjectLift | answerMargin | relationTemplateMargin | paraphraseRetention |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for case in report["cases"]:
        if not case.get("valid"):
            lines.append(f"| {case.get('subject')} | {case.get('answer')} | skip | - | - | - | - |")
            continue
        metrics = case["metrics"]
        lines.append(
            "| "
            + " | ".join([
                str(case["subject"]),
                str(case["answer"]),
                str(case["pass"]).lower(),
                str(case["baseline"]["targetRank"]),
                f"{float(metrics['subjectSpecificityLift']):.6f}",
                f"{float(metrics['answerSpecificityMargin']):.6f}",
                f"{float(metrics['relationTemplateMargin']):.6f}",
                f"{float(metrics['paraphraseRetentionRate']):.3f}",
            ])
            + " |"
        )
    lines.extend([
        "",
        "## Claim Boundary",
        "",
        "This gate is output/logit-only. It builds a behavioral counterfactual graph",
        "from prompt interventions and next-token logits. It does not claim original",
        "neuron identity, hidden activation access, hook-based circuit recovery, or",
        "full LLM decompilation.",
        "",
    ])
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def write_decision_markdown(path: str, report: Dict[str, Any]) -> None:
    summary = report["summary"]
    def cell(value: Any) -> str:
        return str(value).replace("|", r"\|").replace("\r\n", "<br>").replace("\n", "<br>")

    lines = [
        "# Black-box next-token decision evidence",
        "",
        f"- model: `{report['model']['id']}`",
        f"- target: `{report['decision']['target']['text']}`",
        "",
        "## Exact prompt",
        "",
        "```text",
        report["decision"]["prompt"],
        "```",
        "",
        f"- targetRank: `{report['decision']['baseline']['targetRank']}`",
        f"- targetVsContrastMargin: `{report['decision']['baseline']['targetVsContrastMargin']}`",
        f"- oracleQueryCount: `{report['decision']['queryCount']}`",
        "",
        "## Prompt-span counterfactuals",
        "",
        "| span | measured prompt | margin support | target rank | distribution JS (bits) |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for item in report["decision"]["counterfactuals"]:
        lines.append(
            f"| {cell(item['surface'])} | {cell(item['prompt'])} | {item['marginSupport']:.6f} "
            f"| {item['targetRank']} | {item['distributionJensenShannonBits']:.6f} |"
        )
    lines.extend([
        "",
        f"- measuredCounterfactuals: `{summary['measuredCounterfactuals']}`",
        f"- unprobedPromptSpans: `{summary['unprobedPromptSpanCount']}`",
        f"- supportingPromptSpans: `{summary['supportingPromptSpans']}`",
        f"- suppressingPromptSpans: `{summary['suppressingPromptSpans']}`",
        "",
        report["decision"]["claimBoundary"],
        "",
    ])
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def write_trace_markdown(path: str, report: Dict[str, Any]) -> None:
    trace = report["trace"]
    lines = [
        "# Black-box autoregressive decision trace",
        "",
        f"- model: `{report['model']['id']}`",
        f"- decodePolicy: `{trace['decodePolicy']}`",
        f"- generatedText: `{trace['generatedText']}`",
        f"- generatedTokens: `{len(trace['steps'])}`",
        f"- oracleQueryCount: `{trace['queryCount']}`",
        f"- sufficientContextSearchQueries: `{trace['explanationQueryCount']}`",
        f"- pairwiseInteractionQueries: `{trace['interactionQueryCount']}`",
        f"- alternativeBranchQueries: `{trace['branchQueryCount']}`",
        f"- alternativeBranchBeamWidth: `{trace['branchWidth']}`",
        "",
        "| step | token | probability | rank | margin vs alternatives | strongest input-unit effect |",
        "| ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for step in trace["steps"]:
        evidence = step["strongestSupportingToken"] or step["strongestSuppressingToken"]
        effect = (
            f"{evidence['source']} {evidence['inputToken']!r}: {evidence['marginSupport']:.4f}"
            if evidence else "-"
        )
        lines.append(
            f"| {step['stepIndex']} | {step['chosenToken']!r} "
            f"| {step['chosenTokenProbability']:.6g} | {step['chosenTokenRank']} "
            f"| {step['marginVsTopAlternatives']:.6f} | {effect} |"
        )
    lines.extend([
        "",
        "## Minimal sufficient context for each observed next token",
        "",
        "Each context below is a token-ID-preserving subset that retained the same top-1 "
        "next token under oracle queries. `one-minimal` means every remaining editable "
        "unit was individually tested; it does not mean a globally smallest explanation.",
        "",
    ])
    for step in trace["steps"]:
        evidence = step["minimalSufficientContext"]
        retained = evidence["retainedContextTokens"]
        rendered_tokens = " ".join(
            item["token"].replace("|", r"\|")
            for item in retained
        ) or "(empty editable context)"
        status = (
            "one-minimal"
            if evidence["minimality"]["oneMinimalAmongEditableUnits"]
            else "budget-limited"
        )
        lines.extend([
            f"### Step {step['stepIndex']}: `{step['chosenToken']!r}`",
            "",
            f"- retained tokens ({status}): `{rendered_tokens}`",
            f"- token IDs: `{evidence['retainedContextTokenIds']}`",
            f"- oracle queries: `{evidence['oracleQueryCount']}/{evidence['oracleQueryBudget']}`",
        ])
        for witness in evidence["singleTokenDeletionWitnesses"]:
            removed_tokens = "".join(witness["removedContextTokens"])
            lines.append(
                f"- deleting `{removed_tokens!r}` changed top-1 to "
                f"`{witness['counterfactualWinnerToken']!r}` "
                f"(target-vs-best-alternative margin "
                f"`{witness['targetVsBestAlternativeMargin']:.6f}`)"
            )
        lines.append("")
    lines.extend([
        "",
        f"- inputTokens: `{trace['inputTokenCount']}`",
        f"- specialTokensExcluded: `{trace['specialInputTokenCount']}`",
        f"- whitespaceTokensExcluded: `{trace['whitespaceInputTokenCount']}`",
        f"- protectedInputTokens: `{trace['protectedInputTokenCount']}`",
        f"- maximumInterventionsPerStep: `{trace['probesPerToken']}`",
        "",
        "## Full-response prompt-unit counterfactuals",
        "",
        f"- requested: `{trace['responseProbeSummary']['requested']}`",
        f"- selected from measured prompt units: `{trace['responseProbeSummary']['selected']}`",
        f"- eligible measured prompt units: `{trace['responseProbeSummary']['eligibleMeasuredInputUnits']}`",
        f"- oracle queries for full-response counterfactuals: `{trace['responseProbeSummary']['oracleQueryCount']}`",
        "",
        "| deleted prompt unit | counterfactual response | common prefix tokens | first divergence | changed |",
        "| --- | --- | ---: | ---: | --- |",
    ])
    for item in trace["responseCounterfactuals"]:
        first_divergence = item["firstDivergentTokenIndex"]
        lines.append(
            f"| {item['deletedInputToken']!r} | {item['counterfactualGeneratedText']!r} "
            f"| {item['commonPrefixTokenCount']} "
            f"| {first_divergence if first_divergence is not None else '-'} "
            f"| {str(item['responseChanged']).lower()} |"
        )
    lines.extend([
        "",
        "## Forced-token autoregressive branches",
        "",
        "At each measured decision, a selected non-greedy candidate is forced once;",
        "the remaining continuations are searched with bounded beam expansion,",
        "ranked by cumulative token log-probability. These are output counterfactuals,",
        "not transcripts of hidden reasoning.",
        "",
        "| step | forced token (rank) | beam continuation | mean log-probability | first divergence | queries |",
        "| ---: | --- | --- | ---: | ---: | ---: |",
    ])
    for step in trace["steps"]:
        for item in step["alternativeBranches"]:
            for continuation in item["continuations"]:
                lines.append(
                    f"| {step['stepIndex']} | {item['forcedToken']!r} "
                    f"(#{item['forcedTokenRank']}) | {continuation['text']!r} "
                    f"| {continuation['meanTokenLogProbability']:.6f} "
                    f"| {continuation['firstDivergentTokenIndex']} "
                    f"| {item['oracleQueryCount']} |"
                )
    lines.extend([
        "",
        "## Pairwise decision interactions",
        "",
        "The interaction residual is the measured joint-deletion margin minus the",
        "margin predicted by adding the two single-deletion effects. Tested pairs",
        "are prioritized by the summed absolute single-deletion margin changes.",
        "This is local to these exact token-ID interventions, not an internal circuit claim.",
        "",
        "| step | deleted units | joint margin | additive prediction | residual |",
        "| ---: | --- | ---: | ---: | ---: |",
    ])
    for step in trace["steps"]:
        for item in step["pairwiseInterventions"]:
            units = " + ".join(
                f"{unit['source']} {unit['text']!r}"
                for unit in item["deletedUnits"]
            )
            lines.append(
                f"| {step['stepIndex']} | {units} "
                f"| {item['targetVsContrastMargin']:.6f} "
                f"| {item['additivePredictedMargin']:.6f} "
                f"| {item['interactionResidual']:.6f} |"
            )
    lines.extend([
        "",
        trace["claimBoundary"],
        "",
    ])
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.mode == "predict":
        if not args.reconstruction:
            raise ValueError("--reconstruction is required in predict mode.")
        if not args.prompt or not args.prompt.strip():
            raise ValueError("--prompt must contain non-whitespace text in predict mode.")
        if args.max_new_tokens < 1:
            raise ValueError("--max-new-tokens must be positive.")
        if args.prediction_beam_width < 1:
            raise ValueError("--prediction-beam-width must be positive.")
        artifact = load_executable_reconstruction(args.reconstruction)
        if args.model != artifact["tokenizerModelId"]:
            raise ValueError(
                f"Tokenizer model {args.model!r} does not match reconstruction "
                f"tokenizer {artifact['tokenizerModelId']!r}."
            )
        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            local_files_only=args.local_files_only,
        )
        prompt_format = args.prompt_format or artifact.get("inputPromptFormat", "raw")
        input_ids, used_prompt_format = encode_prompt(
            tokenizer,
            args.prompt,
            prompt_format,
        )
        prediction = generate_reconstruction(
            artifact,
            input_ids,
            tokenizer,
            max_new_tokens=args.max_new_tokens,
            prompt_text=args.prompt,
            beam_width=args.prediction_beam_width,
        )
        print(json.dumps({
            "kind": "gcl-blackbox-reconstruction-prediction",
            "reconstruction": str(Path(args.reconstruction).resolve()),
            "modelTokenizer": args.model,
            "inputTokenIds": input_ids,
            "promptFormat": used_prompt_format,
            "prediction": prediction,
            "oracleAccess": False,
        }, indent=2))
        return 0

    if args.mode == "decompile":
        if not args.prompt or not args.prompt.strip():
            raise ValueError("--prompt must contain non-whitespace text in decompile mode.")
        if args.query_budget < 1:
            raise ValueError("--query-budget must be positive.")
        available_formulations = len(_training_formulations(args.prompt))
        if args.formulation_probes < 0:
            raise ValueError("--formulation-probes must be non-negative.")
        if args.formulation_probes > available_formulations:
            raise ValueError(
                "--formulation-probes cannot exceed the "
                f"{available_formulations} available training formulations."
            )
        if args.max_new_tokens < 1:
            raise ValueError("--max-new-tokens must be positive.")
        if args.resume and not args.session_dir:
            raise ValueError("--session-dir is required with --resume.")
        session_dir = args.session_dir
        if not session_dir:
            request_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
            session_dir = str(Path("artifacts/decompilations") / request_id)
        if args.resume and not Path(session_dir).exists():
            raise FileNotFoundError(f"Cannot resume missing decompilation session: {session_dir}")
        with ProcessSeparatedBlackBoxOracle(
            args.model,
            local_files_only=args.local_files_only,
            device=resolve_device(args.device),
        ) as oracle:
            report = run_decompilation(
                oracle=oracle,
                tokenizer=oracle.tokenizer,
                prompt=args.prompt,
                model_identity=oracle.identity(),
                session_dir=session_dir,
                query_budget=args.query_budget,
                max_new_tokens=args.max_new_tokens,
                probes_per_token=args.probes_per_token,
                explanation_probes=args.explanation_probes,
                interaction_probes=args.interaction_probes,
                branch_probes=args.branch_probes,
                branch_width=args.branch_width,
                top_token_count=args.top_logits,
                response_probes=args.response_probes,
                resume=args.resume,
                formulation_probes=args.formulation_probes,
                prompt_format=args.prompt_format or "auto",
            )
        directory = Path(session_dir)
        report_path = Path(args.output_json) if args.output_json else directory / "report.json"
        reconstruction_path = directory / "reconstruction.json"
        explanation_path = Path(args.output_md) if args.output_md else directory / "explanation.md"
        report["executableReconstructionPath"] = str(reconstruction_path.resolve())
        reconstruction_path.parent.mkdir(parents=True, exist_ok=True)
        reconstruction_path.write_text(
            json.dumps(report["executableReconstruction"], indent=2),
            encoding="utf-8",
        )
        rendered = json.dumps(report, indent=2)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(rendered, encoding="utf-8")
        write_explanation_markdown(explanation_path, report)
        print(rendered)
        return 0

    if args.mode == "trace":
        if not args.prompt or not args.prompt.strip():
            raise ValueError("--prompt must contain non-whitespace text when --mode=trace.")
        if args.max_new_tokens < 1:
            raise ValueError("--max-new-tokens must be positive.")
        if args.probes_per_token < 0:
            raise ValueError("--probes-per-token must be non-negative.")
        if args.response_probes < 0:
            raise ValueError("--response-probes must be non-negative.")
        if args.explanation_probes < 0:
            raise ValueError("--explanation-probes must be non-negative.")
        if args.interaction_probes < 0:
            raise ValueError("--interaction-probes must be non-negative.")
        if args.branch_probes < 0:
            raise ValueError("--branch-probes must be non-negative.")
        if args.branch_width < 1:
            raise ValueError("--branch-width must be positive.")
        if args.top_logits < 2:
            raise ValueError("--top-logits must be at least two in trace mode.")
        with ProcessSeparatedBlackBoxOracle(
            args.model,
            local_files_only=args.local_files_only,
            device=resolve_device(args.device),
        ) as oracle:
            prompt_ids = oracle.tokenizer(args.prompt, add_special_tokens=True)["input_ids"]
            trace = trace_greedy_generation(
                oracle_query_token_ids=oracle.query_token_ids,
                tokenizer=oracle.tokenizer,
                input_token_ids=prompt_ids,
                max_new_tokens=args.max_new_tokens,
                probes_per_token=args.probes_per_token,
                top_token_count=args.top_logits,
                response_probe_count=args.response_probes,
                explanation_probes_per_token=args.explanation_probes,
                interaction_probe_count=args.interaction_probes,
                branch_probe_count=args.branch_probes,
                branch_width=args.branch_width,
            )
            report = {
                "schemaVersion": 1,
                "generatedAt": now_iso(),
                "lane": "nn-live-blackbox-llm-behavior-reader",
                "model": oracle.identity(),
                "access": {
                    "solverAccess": "token-id-sequences-to-next-token-logits-only",
                    "weightsUsedBySolver": False,
                    "hooksUsedBySolver": False,
                    "activationsUsedBySolver": False,
                    "postHocWitnessOnly": True,
                },
                "input": {
                    "prompt": args.prompt,
                    "tokenIds": prompt_ids,
                    "tokenCount": len(prompt_ids),
                },
                "trace": trace,
                "summary": {
                    "generatedTokenCount": len(trace["steps"]),
                    "oracleQueryCount": oracle.query_count,
                    "reportedQueryCount": trace["queryCount"],
                    "explanationQueryCount": trace["explanationQueryCount"],
                    "interactionQueryCount": trace["interactionQueryCount"],
                    "branchQueryCount": trace["branchQueryCount"],
                    "branchWidth": trace["branchWidth"],
                    "queryAccountingMatches": oracle.query_count == trace["queryCount"],
                },
            }
        rendered = json.dumps(report, indent=2)
        if args.output_json:
            Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output_json).write_text(rendered, encoding="utf-8")
        if args.output_md:
            Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
            write_trace_markdown(args.output_md, report)
        print(rendered)
        return 0

    if args.mode == "decision":
        if not args.prompt or not args.prompt.strip():
            raise ValueError("--prompt must contain non-whitespace text when --mode=decision.")
        if not args.target_token:
            raise ValueError("--target-token is required when --mode=decision.")
        if args.max_probes < 0:
            raise ValueError("--max-probes must be non-negative.")
        if args.top_logits < 1:
            raise ValueError("--top-logits must be positive.")
        with ProcessSeparatedBlackBoxOracle(
            args.model,
            local_files_only=args.local_files_only,
            device=resolve_device(args.device),
        ) as oracle:
            decision = analyze_next_token_decision(
                oracle_query=oracle.query_logits,
                tokenizer=oracle.tokenizer,
                prompt=args.prompt,
                target_token=args.target_token,
                contrast_tokens=args.contrast_token,
                max_probes=args.max_probes,
                top_token_count=args.top_logits,
            )
            report = {
                "schemaVersion": 1,
                "generatedAt": now_iso(),
                "lane": "nn-live-blackbox-llm-behavior-reader",
                "model": oracle.identity(),
                "access": {
                    "solverAccess": "token-id-sequences-to-next-token-logits-only",
                    "weightsUsedBySolver": False,
                    "hooksUsedBySolver": False,
                    "activationsUsedBySolver": False,
                    "postHocWitnessOnly": True,
                },
                "decision": decision,
                "summary": {
                    **decision["summary"],
                    "oracleQueryCount": oracle.query_count,
                    "reportedQueryCount": decision["queryCount"],
                    "queryAccountingMatches": oracle.query_count == decision["queryCount"],
                },
            }
        rendered = json.dumps(report, indent=2)
        if args.output_json:
            Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output_json).write_text(rendered, encoding="utf-8")
        if args.output_md:
            Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
            write_decision_markdown(args.output_md, report)
        print(rendered)
        return 0

    if not args.output_json or not args.output_md:
        raise ValueError("--output-json and --output-md are required in facts mode.")
    device = resolve_device(args.device)
    oracle = BlackBoxNextTokenOracle(args.model, local_files_only=args.local_files_only, device=device)
    facts = select_facts(args)
    answer_pool = build_answer_pool(oracle.tokenizer, FACTS)
    if not answer_pool:
        raise RuntimeError("No single-token answer pool could be built for this tokenizer.")

    rng = random.Random(int(args.control_seed))

    # The reader below receives only oracle_query(prompt) -> logits.
    oracle_query = oracle.query_logits
    cases = [
        analyze_case(
            oracle_query=oracle_query,
            tokenizer=oracle.tokenizer,
            fact=fact,
            answer_pool=answer_pool,
            all_facts=FACTS,
            args=args,
            rng=rng,
        )
        for fact in facts
    ]

    report = {
        "schemaVersion": 1,
        "generatedAt": now_iso(),
        "lane": "nn-live-blackbox-llm-behavior-reader",
        "variant": "gpt2-output-only-factual-behavior",
        "model": oracle.identity(),
        "runtime": {
            "python": ".".join(map(str, tuple(__import__("sys").version_info[:3]))),
            "torch": package_version("torch"),
            "transformers": package_version("transformers"),
            "cudaAvailable": bool(torch.cuda.is_available()),
        },
        "access": {
            "solverAccess": "prompt-to-next-token-logits-only",
            "weightsUsedBySolver": False,
            "hooksUsedBySolver": False,
            "activationsUsedBySolver": False,
            "postHocWitnessOnly": True,
            "allowedSolverObservation": "prompt string, target token id, next-token logits returned by oracle",
        },
        "summary": summarize(cases, oracle.query_count, args),
        "cases": cases,
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(args.output_md, report)
    print(json.dumps({"pass": report["summary"]["pass"], "summary": report["summary"]}, indent=2))
    return 0 if report["summary"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
