from __future__ import annotations

import importlib.util
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as transformers_logging

from .base import LLMOracleAdapter

transformers_logging.disable_progress_bar()


def _load_live_helper() -> Any:
    helper_path = Path(__file__).resolve().parents[2] / "llm-reader-gpt2-live.py"
    spec = importlib.util.spec_from_file_location("gcl_live_reader", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load GPT-2 live helper module from {helper_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GPT2OracleAdapter(LLMOracleAdapter):
    """GPT-2-family adapter used as the first fast real-model proof gate."""

    adapter_id = "gpt2"
    activation_family = "gelu"

    def __init__(self) -> None:
        self._live = _load_live_helper()

    def load_model_and_tokenizer(
        self,
        model_id: str,
        *,
        local_files_only: bool,
        device: str,
    ) -> Tuple[Any, Any]:
        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=local_files_only)
        model = AutoModelForCausalLM.from_pretrained(model_id, local_files_only=local_files_only)
        model.eval()
        model.to(device)
        return tokenizer, model

    def mlp_modules(self, model: Any, layers: Iterable[int]) -> Dict[int, Any]:
        return {int(layer): self._live.ensure_gpt2_mlp_shape(model, int(layer)) for layer in layers}

    def clean_token(self, tokenizer: Any, token_id: int) -> str:
        return self._live.clean_token(tokenizer, int(token_id))

    def run_activation_ablation_logits(
        self,
        model: Any,
        mlp: Any,
        encoded: Dict[str, torch.Tensor],
        neuron: int,
        ablate_positions: list[int],
        read_token_index: int,
    ) -> torch.Tensor:
        return self._live.run_activation_ablation_logits(
            model,
            mlp,
            encoded,
            int(neuron),
            ablate_positions,
            int(read_token_index),
        )

    def package_version(self, name: str) -> str | None:
        return self._live.package_version(name)

    def model_identity(self, model: Any, model_id: str, device: str, local_files_only: bool) -> Dict[str, Any]:
        return {
            "id": model_id,
            "class": model.__class__.__name__,
            "device": device,
            "localFilesOnly": bool(local_files_only),
            "adapter": self.adapter_id,
            "activationFamily": self.activation_family,
        }

    def pick_amplitude_matched_controls(
        self,
        activation: torch.Tensor,
        target_neurons: list[int],
        excluded: set[int],
        count: int,
        rng: random.Random,
    ) -> list[Dict[str, Any]]:
        return self._live.pick_amplitude_matched_controls(activation, target_neurons, excluded, count, rng)

    def pick_inactive_controls(
        self,
        activation: torch.Tensor,
        excluded: set[int],
        count: int,
        rng: random.Random,
    ) -> tuple[list[Dict[str, Any]], float]:
        return self._live.pick_inactive_controls(activation, excluded, count, rng)
