from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
import random
from typing import Any, Callable, Dict, Iterable, Tuple

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as transformers_logging

from .base import LLMOracleAdapter

transformers_logging.disable_progress_bar()


def _dequantized_weight(module: Any) -> torch.Tensor:
    weight = module.weight
    quant_state = getattr(weight, "quant_state", None)
    if quant_state is None:
        return weight.detach()
    try:
        import bitsandbytes.functional as bnb_functional
    except ImportError as error:
        raise RuntimeError(
            "Quantized LLaMA/Mistral models require bitsandbytes in the GCL GPU environment."
        ) from error
    return bnb_functional.dequantize_4bit(weight.data, quant_state).detach()


class SwiGLUActivationHook:
    """Expose SiLU(gate_proj(x)) * up_proj(x) through the existing hook API."""

    def __init__(self, down_proj: Any) -> None:
        self._down_proj = down_proj

    def register_forward_hook(
        self,
        callback: Callable[[Any, tuple[Any, ...], torch.Tensor], torch.Tensor | None],
    ) -> Any:
        def translate_to_pre_hook(module: Any, inputs: tuple[Any, ...]) -> tuple[Any, ...] | None:
            activation = inputs[0]
            patched = callback(self, (), activation)
            if patched is None:
                return None
            return (patched,) + tuple(inputs[1:])

        return self._down_proj.register_forward_pre_hook(translate_to_pre_hook)


class SwiGLUProjectionView:
    def __init__(self, down_proj: Any) -> None:
        self._down_proj = down_proj
        self._rows: torch.Tensor | None = None

    @property
    def weight(self) -> torch.Tensor:
        if self._rows is None:
            # Existing scoring expects one projection row per addressable MLP unit.
            self._rows = _dequantized_weight(self._down_proj).transpose(0, 1).contiguous().cpu()
        return self._rows


class SwiGLUMlpView:
    def __init__(self, mlp: Any) -> None:
        self._mlp = mlp
        self.act = SwiGLUActivationHook(mlp.down_proj)
        self.c_proj = SwiGLUProjectionView(mlp.down_proj)


class LlamaAttentionView:
    def __init__(self, attention: Any, num_heads: int, head_dim: int) -> None:
        self._attention = attention
        self.c_proj = attention.o_proj
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)


class LlamaSwiGLUOracleAdapter(LLMOracleAdapter):
    """LLaMA/Mistral adapter addressing real SwiGLU value/gate pair channels."""

    adapter_id = "llama-swiglu"
    activation_family = "swiglu"
    supported_model_types = {"llama"}

    def load_model_and_tokenizer(
        self,
        model_id: str,
        *,
        local_files_only: bool,
        device: str,
    ) -> Tuple[Any, Any]:
        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=local_files_only)
        config = AutoConfig.from_pretrained(model_id, local_files_only=local_files_only)
        model_type = str(getattr(config, "model_type", ""))
        if model_type not in self.supported_model_types:
            raise RuntimeError(
                f"{self.adapter_id} adapter does not support model_type={model_type!r}; "
                f"expected one of {sorted(self.supported_model_types)}."
            )
        if getattr(config, "quantization_config", None):
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                local_files_only=local_files_only,
                device_map=device,
            )
        else:
            dtype = torch.float16 if device == "cuda" else torch.float32
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                local_files_only=local_files_only,
                dtype=dtype,
            )
            model.to(device)
        model.eval()
        return tokenizer, model

    def _layers(self, model: Any) -> Any:
        root = getattr(model, "model", None)
        layers = getattr(root, "layers", None)
        if layers is None:
            raise RuntimeError("LLaMA/Mistral adapter expected model.model.layers.")
        return layers

    def mlp_modules(self, model: Any, layers: Iterable[int]) -> Dict[int, Any]:
        blocks = self._layers(model)
        mlps: Dict[int, Any] = {}
        for layer in layers:
            layer_index = int(layer)
            if layer_index < 0 or layer_index >= len(blocks):
                raise RuntimeError(
                    f"Layer {layer_index} is outside LLaMA/Mistral block range 0..{len(blocks) - 1}."
                )
            mlp = blocks[layer_index].mlp
            for name in ("gate_proj", "up_proj", "down_proj", "act_fn"):
                if not hasattr(mlp, name):
                    raise RuntimeError(f"LLaMA/Mistral MLP layer {layer_index} is missing {name}.")
            mlps[layer_index] = SwiGLUMlpView(mlp)
        return mlps

    def attention_modules(self, model: Any, layers: Iterable[int]) -> Dict[int, Any]:
        blocks = self._layers(model)
        num_heads = int(getattr(model.config, "num_attention_heads"))
        head_dim = int(getattr(model.config, "head_dim", 0) or getattr(blocks[0].self_attn, "head_dim"))
        attentions: Dict[int, Any] = {}
        for layer in layers:
            layer_index = int(layer)
            if layer_index < 0 or layer_index >= len(blocks):
                raise RuntimeError(
                    f"Layer {layer_index} is outside LLaMA/Mistral block range 0..{len(blocks) - 1}."
                )
            attention = blocks[layer_index].self_attn
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                if not hasattr(attention, name):
                    raise RuntimeError(f"LLaMA/Mistral attention layer {layer_index} is missing {name}.")
            attentions[layer_index] = LlamaAttentionView(attention, num_heads, head_dim)
        return attentions

    def attention_head_count(self, attention: Any) -> int:
        return int(attention.num_heads)

    def clean_token(self, tokenizer: Any, token_id: int) -> str:
        return tokenizer.decode([int(token_id)]).replace("\n", "\\n")

    def unembedding_weight(self, model: Any) -> torch.Tensor:
        if hasattr(model, "lm_head"):
            return model.lm_head.weight
        output_embeddings = model.get_output_embeddings()
        if output_embeddings is None:
            raise RuntimeError("LLaMA/Mistral adapter could not locate output embeddings.")
        return output_embeddings.weight

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
        valid_positions = sorted(set(int(position) for position in ablate_positions if 0 <= int(position) < seq_len))
        if not valid_positions:
            raise RuntimeError("SwiGLU ablation requires at least one valid token position.")

        def zero_pair_activation(module: Any, inputs: Any, output: torch.Tensor) -> torch.Tensor:
            patched = output.clone()
            for position in valid_positions:
                patched[:, position, int(neuron)] = 0
            return patched

        handle = mlp.act.register_forward_hook(zero_pair_activation)
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
            [{"layer": 0, "head": int(head), "ablateTokenPositions": ablate_positions}],
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
        for layer in grouped:
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
            "keyValueHeads": getattr(model.config, "num_key_value_heads", None),
            "hiddenSize": getattr(model.config, "hidden_size", None),
            "intermediateSize": getattr(model.config, "intermediate_size", None),
            "device": device,
            "localFilesOnly": bool(local_files_only),
            "quantized": bool(getattr(model, "is_quantized", False)),
            "adapter": self.adapter_id,
            "activationFamily": self.activation_family,
            "addressedMlpUnit": "silu(gate_proj(x))*up_proj(x) pair before down_proj",
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


class MistralSwiGLUOracleAdapter(LlamaSwiGLUOracleAdapter):
    """Strict Mistral-family SwiGLU adapter using the shared pair-channel mechanics."""

    adapter_id = "mistral-swiglu"
    supported_model_types = {"mistral"}
