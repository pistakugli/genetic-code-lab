#!/usr/bin/env python3
"""List the SwiGLU units that actually reach a chosen down-proj output channel.

In `down-proj-output-channel` oracle mode the scalar the solver sees is

    sum_i  down_proj[channel, unit_i] * silu(gate_i . x) * (up_i . x)

so a unit whose `down_proj[channel, unit]` coefficient is zero contributes nothing
at all. Asking a solver to recover such a unit is not a hard task, it is an
impossible one: the oracle never exposes it, and the recovered atom count can never
match the requested one.

Randomly drawn 16-unit blocks hit this regularly on 4-bit quantised models. Two of
the six cases in the randomised cold-start court were ill-posed for exactly this
reason, with three and two zero-coefficient units respectively.

This helper runs on the teacher side only. It never touches the solver, which still
receives nothing beyond its opaque endpoint and public shape metadata.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.llm_reader.swiglu_oracle_server import load_teacher


def main() -> int:
    parser = argparse.ArgumentParser(description="List units contributing to a down-proj output channel")
    parser.add_argument("--model", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--output-channel", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--min-abs-scale",
        type=float,
        default=1e-6,
        help="smallest down-proj magnitude that still counts as contributing",
    )
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    if args.output_channel < 0:
        raise SystemExit("--output-channel must be non-negative; selected-unit-sum mode needs no filtering")
    if args.min_abs_scale <= 0.0:
        raise SystemExit("--min-abs-scale must be positive")

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.utils import logging as transformers_logging

    from tools.llm_reader.swiglu_oracle_server import dequantized_weight

    transformers_logging.disable_progress_bar()
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    if getattr(config, "quantization_config", None):
        model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True, device_map=args.device)
    else:
        dtype = torch.float16 if args.device == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True, dtype=dtype)
        model.to(args.device)
    model.eval()
    layers = model.model.layers
    if args.layer < 0 or args.layer >= len(layers):
        raise SystemExit(f"layer {args.layer} outside range 0..{len(layers) - 1}")
    down_weight = dequantized_weight(layers[args.layer].mlp.down_proj).detach().float().cpu()
    if args.output_channel >= down_weight.shape[0]:
        raise SystemExit(f"output channel {args.output_channel} outside range 0..{down_weight.shape[0] - 1}")
    scales = down_weight[args.output_channel].abs()
    contributing = [int(index) for index in range(scales.shape[0]) if float(scales[index]) >= args.min_abs_scale]
    del model, down_weight
    torch.cuda.empty_cache()

    payload = {
        "model": args.model,
        "layer": args.layer,
        "outputChannel": args.output_channel,
        "minAbsScale": args.min_abs_scale,
        "unitCount": int(scales.shape[0]),
        "contributingUnitCount": len(contributing),
        "zeroCoefficientUnitCount": int(scales.shape[0]) - len(contributing),
        "contributingUnits": contributing,
    }
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(payload) + "\n", encoding="utf-8")
    else:
        print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
