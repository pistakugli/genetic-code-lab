from __future__ import annotations

from typing import Any, Tuple

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .llama_swiglu import LlamaSwiGLUOracleAdapter


class Qwen2SwiGLUOracleAdapter(LlamaSwiGLUOracleAdapter):
    """Qwen2 SwiGLU adapter with full-precision weights for finite-difference probes."""

    adapter_id = "qwen2-swiglu"
    supported_model_types = {"qwen2"}

    def load_model_and_tokenizer(
        self,
        model_id: str,
        *,
        local_files_only: bool,
        device: str,
    ) -> Tuple[Any, Any]:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            local_files_only=local_files_only,
        )
        config = AutoConfig.from_pretrained(
            model_id,
            local_files_only=local_files_only,
        )
        model_type = str(getattr(config, "model_type", ""))
        if model_type not in self.supported_model_types:
            raise RuntimeError(
                f"{self.adapter_id} adapter does not support model_type={model_type!r}; "
                f"expected one of {sorted(self.supported_model_types)}."
            )
        if getattr(config, "quantization_config", None):
            raise RuntimeError(
                "The Qwen2 finite-difference recovery pilot requires an unquantized "
                "full-precision model."
            )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            local_files_only=local_files_only,
            dtype=torch.float32,
        )
        model.to(device)
        model.eval()
        return tokenizer, model
