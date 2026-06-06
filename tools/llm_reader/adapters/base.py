from __future__ import annotations

from abc import ABC, abstractmethod
import random
from typing import Any, Dict, Iterable, Tuple

import torch


class LLMOracleAdapter(ABC):
    """Typed boundary between circuit search and a concrete transformer family."""

    adapter_id: str
    activation_family: str

    @abstractmethod
    def load_model_and_tokenizer(
        self,
        model_id: str,
        *,
        local_files_only: bool,
        device: str,
    ) -> Tuple[Any, Any]:
        """Load a real model runtime and tokenizer."""

    @abstractmethod
    def mlp_modules(self, model: Any, layers: Iterable[int]) -> Dict[int, Any]:
        """Return layer-indexed MLP components addressable by the circuit finder."""

    @abstractmethod
    def attention_modules(self, model: Any, layers: Iterable[int]) -> Dict[int, Any]:
        """Return layer-indexed attention components addressable by the circuit finder."""

    @abstractmethod
    def attention_head_count(self, attention: Any) -> int:
        """Return the number of independently addressable attention heads."""

    @abstractmethod
    def clean_token(self, tokenizer: Any, token_id: int) -> str:
        """Return a report-safe token string."""

    @abstractmethod
    def unembedding_weight(self, model: Any) -> torch.Tensor:
        """Return the output unembedding matrix indexed by token id."""

    @abstractmethod
    def run_activation_ablation_logits(
        self,
        model: Any,
        mlp: Any,
        encoded: Dict[str, torch.Tensor],
        neuron: int,
        ablate_positions: list[int],
        read_token_index: int,
    ) -> torch.Tensor:
        """Run a real forward pass with one activation channel ablated."""

    @abstractmethod
    def run_attention_head_ablation_logits(
        self,
        model: Any,
        attention: Any,
        encoded: Dict[str, torch.Tensor],
        head: int,
        ablate_positions: list[int],
        read_token_index: int,
    ) -> torch.Tensor:
        """Run a real forward pass with one attention head path ablated."""

    @abstractmethod
    def run_attention_heads_ablation_logits(
        self,
        model: Any,
        attentions: Dict[int, Any],
        encoded: Dict[str, torch.Tensor],
        interventions: list[Dict[str, Any]],
        read_token_index: int,
    ) -> torch.Tensor:
        """Run a real forward pass with multiple attention head paths ablated."""

    @abstractmethod
    def package_version(self, name: str) -> str | None:
        """Return an installed package version if available."""

    @abstractmethod
    def model_identity(self, model: Any, model_id: str, device: str, local_files_only: bool) -> Dict[str, Any]:
        """Return artifact metadata for the loaded model/runtime."""

    @abstractmethod
    def pick_amplitude_matched_controls(
        self,
        activation: torch.Tensor,
        target_neurons: list[int],
        excluded: set[int],
        count: int,
        rng: random.Random,
    ) -> list[Dict[str, Any]]:
        """Select active negative-control neurons with similar activation amplitude."""

    @abstractmethod
    def pick_inactive_controls(
        self,
        activation: torch.Tensor,
        excluded: set[int],
        count: int,
        rng: random.Random,
    ) -> tuple[list[Dict[str, Any]], float]:
        """Select inactive negative-control neurons."""
