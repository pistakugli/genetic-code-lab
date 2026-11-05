from __future__ import annotations

from .base import LLMOracleAdapter
from .gpt2 import GPT2OracleAdapter
from .gpt_neox import GPTNeoXOracleAdapter
from .llama_swiglu import LlamaSwiGLUOracleAdapter, MistralSwiGLUOracleAdapter
from .qwen2_swiglu import Qwen2SwiGLUOracleAdapter


def create_adapter(adapter_id: str) -> LLMOracleAdapter:
    normalized = adapter_id.strip().lower()
    if normalized in {"gpt2", "gpt-2"}:
        return GPT2OracleAdapter()
    if normalized in {"gpt-neox", "neox", "pythia"}:
        return GPTNeoXOracleAdapter()
    if normalized in {"llama-swiglu", "llama"}:
        return LlamaSwiGLUOracleAdapter()
    if normalized in {"mistral", "mistral-swiglu"}:
        return MistralSwiGLUOracleAdapter()
    if normalized in {"qwen2-swiglu", "qwen2"}:
        return Qwen2SwiGLUOracleAdapter()
    raise RuntimeError(f"Unknown LLM oracle adapter: {adapter_id}")


def create_adapter_for_model_type(model_type: str) -> LLMOracleAdapter:
    normalized = model_type.strip().lower()
    adapter_ids = {
        "gpt2": "gpt2",
        "gpt_neox": "gpt-neox",
        "llama": "llama-swiglu",
        "mistral": "mistral-swiglu",
        "qwen2": "qwen2-swiglu",
    }
    adapter_id = adapter_ids.get(normalized)
    if adapter_id is None:
        raise RuntimeError(
            f"No live LLM reader adapter is registered for model_type={model_type!r}."
        )
    return create_adapter(adapter_id)


def adapter_choices() -> list[str]:
    return ["gpt2", "gpt-neox", "llama-swiglu", "mistral-swiglu", "qwen2-swiglu"]
