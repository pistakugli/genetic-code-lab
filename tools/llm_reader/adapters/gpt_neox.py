from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
import random
from typing import Any, Dict, Iterable, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as transformers_logging

from .base import LLMOracleAdapter

transformers_logging.disable_progress_bar()


class NeoXMlpView:
    """Compatibility view for GPT-NeoX MLPs using the existing GELU path code."""

    def __init__(self, mlp: Any) -> None:
        self._mlp = mlp
        self.act = mlp.act
        self.c_proj = NeoXMlpProjectionView(mlp.dense_4h_to_h)


class NeoXMlpProjectionView:
    def __init__(self, dense_4h_to_h: Any) -> None:
        self._dense_4h_to_h = dense_4h_to_h

    @property
    def weight(self) -> torch.Tensor:
        # Existing circuit code expects one projection row per hidden MLP channel.
        return self._dense_4h_to_h.weight.transpose(0, 1)


class NeoXAttentionView:
    """Compatibility view for GPT-NeoX attention output-path ablations."""

    def __init__(self, attention: Any, num_heads: int, hidden_size: int) -> None:
        self._attention = attention
        self.c_proj = attention.dense
        self.num_heads = int(num_heads)
        self.head_dim = int(hidden_size) // int(num_heads)


class GPTNeoXOracleAdapter(LLMOracleAdapter):
    """GPT-NeoX/Pythia adapter for real-model circuit/path finding."""

    adapter_id = "gpt-neox"
    activation_family = "gelu"

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

    def _layers(self, model: Any) -> Any:
        root = getattr(model, "gpt_neox", None)
        layers = getattr(root, "layers", None)
        if layers is None:
            raise RuntimeError("GPT-NeoX adapter expected model.gpt_neox.layers.")
        return layers

    def mlp_modules(self, model: Any, layers: Iterable[int]) -> Dict[int, Any]:
        blocks = self._layers(model)
        mlps: Dict[int, Any] = {}
        for layer in layers:
            layer_index = int(layer)
            if layer_index < 0 or layer_index >= len(blocks):
                raise RuntimeError(f"Layer {layer_index} is outside GPT-NeoX block range 0..{len(blocks) - 1}.")
            mlp = blocks[layer_index].mlp
            for name in ("dense_h_to_4h", "dense_4h_to_h", "act"):
                if not hasattr(mlp, name):
                    raise RuntimeError(f"GPT-NeoX MLP layer {layer_index} is missing {name}.")
            mlps[layer_index] = NeoXMlpView(mlp)
        return mlps

    def attention_modules(self, model: Any, layers: Iterable[int]) -> Dict[int, Any]:
        blocks = self._layers(model)
        num_heads = int(getattr(model.config, "num_attention_heads"))
        hidden_size = int(getattr(model.config, "hidden_size"))
        attentions: Dict[int, Any] = {}
        for layer in layers:
            layer_index = int(layer)
            if layer_index < 0 or layer_index >= len(blocks):
                raise RuntimeError(f"Layer {layer_index} is outside GPT-NeoX block range 0..{len(blocks) - 1}.")
            attention = blocks[layer_index].attention
            for name in ("query_key_value", "dense"):
                if not hasattr(attention, name):
                    raise RuntimeError(f"GPT-NeoX attention layer {layer_index} is missing {name}.")
            attentions[layer_index] = NeoXAttentionView(attention, num_heads, hidden_size)
        return attentions

    def attention_head_count(self, attention: Any) -> int:
        return int(attention.num_heads)

    def clean_token(self, tokenizer: Any, token_id: int) -> str:
        return tokenizer.decode([int(token_id)]).replace("\n", "\\n")

    def unembedding_weight(self, model: Any) -> torch.Tensor:
        if hasattr(model, "embed_out"):
            return model.embed_out.weight
        if hasattr(model, "lm_head"):
            return model.lm_head.weight
        raise RuntimeError("GPT-NeoX adapter expected model.embed_out or model.lm_head.")

    def run_activation_ablation_logits(
        self,
        model: Any,
        mlp: Any,
        encoded: Dict[str, torch.Tensor],
        neuron: int,
        ablate_positions: list[int],
        read_token_index: int,
    ) -> torch.Tensor:
        seq_len = int(encoded["input_ids"].shape[1])
        valid_positions = sorted(set(position for position in ablate_positions if 0 <= int(position) < seq_len))
        if not valid_positions:
            raise RuntimeError("Ablation requires at least one valid token position.")

        def zero_neuron_activation(module: Any, inputs: Any, output: torch.Tensor) -> torch.Tensor:
            patched = output.clone()
            for position in valid_positions:
                patched[:, int(position), int(neuron)] = 0
            return patched

        handle = mlp.act.register_forward_hook(zero_neuron_activation)
        try:
            with torch.no_grad():
                return model(**encoded).logits.detach().cpu()[0, int(read_token_index)]
        finally:
            handle.remove()

    def run_attention_head_ablation_logits(
        self,
        model: Any,
        attention: Any,
        encoded: Dict[str, torch.Tensor],
        head: int,
        ablate_positions: list[int],
        read_token_index: int,
    ) -> torch.Tensor:
        return self.run_attention_heads_ablation_logits(
            model,
            {0: attention},
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
                return model(**encoded).logits.detach().cpu()[0, int(read_token_index)]
        finally:
            for handle in handles:
                handle.remove()

    def package_version(self, name: str) -> str | None:
        try:
            return version(name)
        except PackageNotFoundError:
            return None

    def model_identity(self, model: Any, model_id: str, device: str, local_files_only: bool) -> Dict[str, Any]:
        return {
            "id": model_id,
            "class": model.__class__.__name__,
            "modelType": getattr(model.config, "model_type", None),
            "layers": getattr(model.config, "num_hidden_layers", None),
            "heads": getattr(model.config, "num_attention_heads", None),
            "hiddenSize": getattr(model.config, "hidden_size", None),
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
        activation_cpu = activation.detach().cpu().float()
        values = activation_cpu.abs()
        if not target_neurons or count <= 0:
            return []
        choices: list[Dict[str, Any]] = []
        used = set(excluded)
        target_index = 0
        while len(choices) < count:
            target = int(target_neurons[target_index % len(target_neurons)])
            target_index += 1
            target_abs = float(values[target])
            candidates = [
                index for index in range(int(values.numel()))
                if index not in used and float(values[index]) > 0.05
            ]
            if not candidates:
                candidates = [index for index in range(int(values.numel())) if index not in used]
            if not candidates:
                break
            nearest = sorted(candidates, key=lambda index: abs(float(values[index]) - target_abs))[:64]
            chosen = int(rng.choice(nearest))
            used.add(chosen)
            choices.append({
                "neuron": chosen,
                "matchedToNeuron": target,
                "matchedToActivationAbs": target_abs,
                "activationAbs": float(values[chosen]),
                "activation": float(activation_cpu[chosen]),
            })
        return choices

    def pick_inactive_controls(
        self,
        activation: torch.Tensor,
        excluded: set[int],
        count: int,
        rng: random.Random,
    ) -> tuple[list[Dict[str, Any]], float]:
        values = activation.detach().cpu().float().abs()
        threshold = float(torch.quantile(values, 0.10).item()) if int(values.numel()) else 0.0
        candidates = [
            (float(values[index]), index)
            for index in range(int(values.numel()))
            if index not in excluded and float(values[index]) <= threshold
        ]
        if len(candidates) < count:
            candidates = [
                (float(values[index]), index)
                for index in range(int(values.numel()))
                if index not in excluded
            ]
        rng.shuffle(candidates)
        candidates.sort(key=lambda item: item[0])
        return (
            [{"neuron": int(index), "activationAbs": float(value)} for value, index in candidates[:count]],
            threshold,
        )
