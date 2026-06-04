"""LLM oracle adapter implementations."""

from .base import LLMOracleAdapter
from .gpt2 import GPT2OracleAdapter

__all__ = ["LLMOracleAdapter", "GPT2OracleAdapter"]

