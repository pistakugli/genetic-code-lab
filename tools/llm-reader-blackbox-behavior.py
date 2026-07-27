#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import random
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as transformers_logging

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
        description="Output-only black-box LLM behavior reader for factual next-token prompts."
    )
    parser.add_argument("--model", default="gpt2")
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
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
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

    def identity(self) -> Dict[str, Any]:
        config = getattr(self.model, "config", None)
        return {
            "id": self.model_id,
            "class": self.model.__class__.__name__,
            "modelType": getattr(config, "model_type", None),
            "device": str(self.device),
            "localFilesOnly": self.local_files_only,
        }


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


def main() -> int:
    args = parse_args()
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
