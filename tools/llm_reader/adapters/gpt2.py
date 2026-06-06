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

    def attention_modules(self, model: Any, layers: Iterable[int]) -> Dict[int, Any]:
        blocks = getattr(getattr(model, "transformer", None), "h", None)
        if blocks is None:
            raise RuntimeError("GPT-2 adapter expected model.transformer.h blocks.")
        attentions: Dict[int, Any] = {}
        for layer in layers:
            layer_index = int(layer)
            attention = blocks[layer_index].attn
            for name in ("c_proj", "num_heads", "head_dim"):
                if not hasattr(attention, name):
                    raise RuntimeError(f"GPT-2 attention layer {layer_index} is missing {name}.")
            attentions[layer_index] = attention
        return attentions

    def attention_head_count(self, attention: Any) -> int:
        return int(attention.num_heads)

    def clean_token(self, tokenizer: Any, token_id: int) -> str:
        return self._live.clean_token(tokenizer, int(token_id))

    def unembedding_weight(self, model: Any) -> torch.Tensor:
        if not hasattr(model, "lm_head"):
            raise RuntimeError("GPT-2 adapter expected model.lm_head.")
        return model.lm_head.weight

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

    def run_attention_head_ablation_logits(
        self,
        model: Any,
        attention: Any,
        encoded: Dict[str, torch.Tensor],
        head: int,
        ablate_positions: list[int],
        read_token_index: int,
    ) -> torch.Tensor:
        layer_attentions = {0: attention}
        return self.run_attention_heads_ablation_logits(
            model,
            layer_attentions,
            encoded,
            [{
                "layer": 0,
                "head": int(head),
                "ablateTokenPositions": ablate_positions,
            }],
            int(read_token_index),
        )

    def run_attention_heads_ablation_logits(
        self,
        model: Any,
        attentions: Dict[int, Any],
        encoded: Dict[str, torch.Tensor],
        interventions: list[Dict[str, Any]],
        read_token_index: int,
    ) -> torch.Tensor:
        seq_len = int(encoded["input_ids"].shape[1])
        grouped: Dict[int, Dict[int, list[int]]] = {}
        for intervention in interventions:
            layer = int(intervention["layer"])
            head = int(intervention["head"])
            positions = sorted(set(int(position) for position in intervention.get("ablateTokenPositions", [])))
            valid_positions = [position for position in positions if 0 <= position < seq_len]
            if not valid_positions:
                raise RuntimeError("Attention-head ablation requires at least one valid token position.")
            if layer not in attentions:
                raise RuntimeError(f"Attention-head ablation references unavailable layer {layer}.")
            head_count = self.attention_head_count(attentions[layer])
            if head < 0 or head >= head_count:
                raise RuntimeError(f"Attention-head ablation references invalid head {head} for layer {layer}.")
            grouped.setdefault(layer, {}).setdefault(head, [])
            grouped[layer][head] = sorted(set(grouped[layer][head] + valid_positions))
        if not grouped:
            raise RuntimeError("Attention-head graph ablation requires at least one intervention.")

        handles = []
        for layer, heads in grouped.items():
            attention = attentions[layer]
            head_dim = int(attention.head_dim)

            def patch_attention_path(module: Any, inputs: Any, layer_index: int = layer) -> tuple[torch.Tensor, ...]:
                patched = inputs[0].clone()
                for head, positions in grouped[layer_index].items():
                    start = int(head) * head_dim
                    end = start + head_dim
                    for position in positions:
                        patched[:, position, start:end] = 0
                return (patched,) + tuple(inputs[1:])

            handles.append(attention.c_proj.register_forward_pre_hook(patch_attention_path))
        try:
            with torch.no_grad():
                return model(**encoded).logits.detach().cpu()[0, read_token_index]
        finally:
            for handle in handles:
                handle.remove()

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
