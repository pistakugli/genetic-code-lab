# GPU Environments for Live Lanes

How the repository obtains and validates GPU compute for live neural lanes
(ordered-path stability/OOD stress, circuit finders, SwiGLU oracles), and what
each environment is allowed to claim. This document is the runbook; the code
entry points are `scripts/gpu/provision-gpu-env.sh` (bootstrap),
`scripts/gpu/gpu-smoke.py` (host validation), and
`scripts/gpu/wrap-lane-suite.py` (suite assembly without Node.js).

## The environment contract

Every live-lane host, rented or owned, satisfies the same contract:

| Piece | Value |
| --- | --- |
| Python virtualenv | `.gcl-python-gpu/` at the repo root (gitignored) |
| GPU dependency pin | `requirements/gcl-python-gpu.txt` (torch cu128, transformers, accelerate, bitsandbytes) |
| CPU dependency pin | `requirements/gcl-python.txt` (torch cpu, for plumbing smokes only) |
| Device override | `GCL_DEVICE` env var — honored by `gpu-smoke.py`, the ordered-path runner, and the TS release orchestrator; overrides the `--device` flag so containers/notebooks need no argument edits |
| Python override | `GCL_PYTHON` (or `GCL_LLM_READER_PYTHON`) env var for the TS orchestrator; if unset it auto-detects `.gcl-python-gpu/bin/python` |
| Model weights | warmed into the Hugging Face cache by provisioning, then runs use `--local-files-only` so evidence runs never touch the network |
| Host validation | `scripts/gpu/gpu-smoke.py` — device fingerprint, forward throughput, same-device determinism (max |Δlogit| ≤ 1e-5), forward-hook capture; writes `benchmarks/scratch/gpu-smoke/report.json` |

## Environment matrix

| Environment | Cost | Setup | Evidence status |
| --- | --- | --- | --- |
| Google Colab free T4 | free (Google account) | open `scripts/gpu/colab-gcl-ordered-path-ood-stress.ipynb`, Runtime → T4 GPU, Run all | full lane + suite + verifier cycle, `device: cuda` artifacts are release-grade |
| Owned/workstation GPU (e.g. RTX 3070) | owned | `bash scripts/gpu/provision-gpu-env.sh`, then any lane or `npm run` script | recorded release evidence was produced this way (see `benchmarks/release/nn-live-ordered-path-stability/`) |
| Any Linux CUDA host (rented: RunPod, Vast, Lambda, …) | hourly | same provision script; check `nvidia-smi` driver ≥ the cu128 wheel requirement first | same as workstation once `gpu-smoke.py` passes |
| CPU-only host | — | `bash scripts/gpu/provision-gpu-env.sh --cpu` | **plumbing smoke only** — the OOD-stress verifier rejects non-CUDA artifacts by rule (see below) |

## Runbooks

### 1. Provision and validate any host

```bash
bash scripts/gpu/provision-gpu-env.sh          # GPU stack
bash scripts/gpu/provision-gpu-env.sh --cpu    # CPU stack (dev smoke)
GCL_MODEL=gpt2 GCL_DEVICE=auto bash scripts/gpu/provision-gpu-env.sh
```

Idempotent. The final step is the smoke check; a host is usable only when it
prints `verdict: OK`. The smoke report records throughput and the device
fingerprint, so runs can cite which host produced them.

### 2. Colab free T4 (no local setup at all)

Open `scripts/gpu/colab-gcl-ordered-path-ood-stress.ipynb` in Colab, pick a T4
runtime, Run all. The notebook clones the public GitHub remote, provisions,
runs the recorded France OOD-stress configuration (6 OOD paraphrases, rescue +
replacement search), assembles the suite with the python wrapper, verifies it
with `tools/llm_reader/verify_ordered_path_ood_stress.py`, and zips the
artifacts. No Node.js and no credentials beyond the Colab account.

### 3. Full release gate (needs Node.js)

```bash
npm install
npm run benchmark:nn-live-ordered-path-ood-stress          # GPT-2 only
npm run benchmark:nn-live-ordered-path-ood-stress:all      # + Pythia, Llama, Mistral
npm run verify:nn-live-ordered-path-ood-stress
```

The `:gpu` npm scripts preset `GCL_PYTHON=.gcl-python-gpu/bin/python`. Every
lane parameter is env-tunable (`GCL_LIVE_STABILITY_*`); the orchestrator passes
`--device` from `GCL_DEVICE` (default `cuda`).

### 4. CPU plumbing smoke (what it can and cannot show)

```bash
GCL_DEVICE=cpu GCL_BENCHMARK_ROOT=benchmarks/scratch \
GCL_BENCHMARK_VARIANT=cpu-plumbing-smoke \
GCL_LIVE_STABILITY_MODELS=gpt2 GCL_LIVE_STABILITY_PARAPHRASE_PROFILE=ood \
GCL_LIVE_STABILITY_GPT2_SUBJECTS=France GCL_LIVE_STABILITY_GPT2_CASE_COUNT=1 \
GCL_LIVE_STABILITY_GPT2_PARAPHRASE_COUNT=1 GCL_LIVE_STABILITY_RESCUE=0 \
node_modules/.bin/ts-node scripts/benchmark-nn-live-ordered-path-stability.ts
```

This proves the orchestration plumbing (env → TS → venv → lane tool → output
layout) on machines with no GPU. It is **not** lane evidence.

## Device rules and evidence grading

- The OOD-stress lane verifier requires each model result to record
  `model.device == "cuda"`. A CPU-generated artifact fails verification with
  `pilot must record CUDA as the model device` — by design: the release lane
  is a GPU-evidence lane.
- CPU runs therefore belong under `benchmarks/scratch/` and may be cited only
  as plumbing/oracle-development smokes, never as OOD-stress evidence.
- Same-device determinism (replay error 0.0 on repeated runs) is checked by
  the lane itself (`maxReplayError`) and by the smoke check; cross-device
  numeric equality is explicitly **not** claimed anywhere.
- Throughput numbers from `gpu-smoke.py` describe one host and one prompt
  length; they are host fingerprints, not benchmark comparisons.

## Verification record (2026-11-06, CPU plumbing sandbox)

| Check | Result |
| --- | --- |
| `gpu-smoke.py` on CPU (device auto-resolve, determinism, hooks) | OK, max abs Δlogit 0.0, exit 0 |
| `provision-gpu-env.sh --cpu` end-to-end | PROVISIONED, exit 0 |
| Live ordered-path OOD stress run, `GCL_DEVICE=cpu`, France, 6 OOD paraphrases, rescue + replacement | completed in 3m34s; sign-flip break, component-rebinding replacement, replay error 0.0; transfer 5/5 |
| `wrap-lane-suite.py` suite assembly | format-identical to TS output; verifier processes it fully |
| Verifier on CPU-generated suite | correctly rejected (`device != cuda`) — negative control for the release rule |
| Verifier on recorded release suite (RTX 3070 artifact) | `ORDERED_PATH_OOD_STRESS_ARTIFACT_VERIFIED`, exit 0 — positive control |
| TS release gate plumbing smoke (`GCL_DEVICE=cpu`, scratch root) | exit 0, full output layout written |

The T4 path itself is exercised by the Colab notebook above; on the CPU
sandbox only its non-GPU steps could be validated, which is what the claim
boundary in each artifact records.
