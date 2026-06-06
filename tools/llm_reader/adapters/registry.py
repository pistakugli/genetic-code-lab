from __future__ import annotations

from .base import LLMOracleAdapter
from .gpt2 import GPT2OracleAdapter
from .gpt_neox import GPTNeoXOracleAdapter


def create_adapter(adapter_id: str) -> LLMOracleAdapter:
    normalized = adapter_id.strip().lower()
    if normalized in {"gpt2", "gpt-2"}:
        return GPT2OracleAdapter()
    if normalized in {"gpt-neox", "neox", "pythia"}:
        return GPTNeoXOracleAdapter()
    raise RuntimeError(f"Unknown LLM oracle adapter: {adapter_id}")


def adapter_choices() -> list[str]:
    return ["gpt2", "gpt-neox"]
