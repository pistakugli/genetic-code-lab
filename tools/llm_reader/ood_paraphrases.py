from __future__ import annotations

from collections import defaultdict
from statistics import fmean
from typing import Any, Dict, Iterable, List


OOD_CAPITAL_PROMPT_TEMPLATES = (
    ("service-question", "Which city serves as the capital of {subject}?"),
    ("possessive-question", "What city is {subject}'s capital?"),
    ("imperative", "Give the name of the capital city of {subject}."),
    (
        "embedded-question",
        "I am checking a country profile. Could you tell me which city is the capital of {subject}?",
    ),
    (
        "sentence-completion",
        "Complete this sentence with a city name: The city that is the capital of {subject} is",
    ),
    (
        "context-prefixed-question",
        "For a geography quiz, answer with a city name only. Which city is the capital of {subject}?",
    ),
)


def ood_paraphrase_cases_for_fact(fact: Dict[str, Any], count: int) -> List[Dict[str, str]]:
    if count < 0:
        raise ValueError("OOD paraphrase count must be non-negative")
    subject = str(fact.get("subject", "")).strip()
    if not subject:
        raise ValueError("OOD capital paraphrases require a non-empty subject")
    return [
        {
            "profile": "ood",
            "family": family,
            "prompt": template.format(subject=subject),
        }
        for family, template in OOD_CAPITAL_PROMPT_TEMPLATES[:count]
    ]


def summarize_paraphrases_by_family(items: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, float | int]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in items:
        family = str(item.get("family", "")).strip()
        if not family:
            raise ValueError("Every paraphrase measurement must have a family")
        grouped[family].append(item)

    return {
        family: {
            "count": len(rows),
            "passed": sum(bool(row["pass"]) for row in rows),
            "passRate": fmean(bool(row["pass"]) for row in rows),
            "meanDropRatio": fmean(float(row["ratioToOriginal"]) for row in rows),
            "meanTokenCount": fmean(int(row["tokenCount"]) for row in rows),
        }
        for family, rows in sorted(grouped.items())
    }
