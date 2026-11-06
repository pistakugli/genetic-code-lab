# Agent Note: Self-provisioned GPU environment lane

Status: implemented

## Context

All recorded live-lane evidence (ordered-path OOD stress, circuit finders) was
produced on the project owner's personal RTX 3070 through a manually built
`.gcl-python-gpu` venv. That single-host dependency had three consequences:
the repository had no reproducible way to stand up a live-lane environment
anywhere else, no host validation step (nothing checked a machine before a
multi-hour lane ran on it), and no runbook for free or rented GPU compute.
The instruction for this round was to remove that dependency by finding a GPU
environment the project controls end to end.

## Decision

Added a self-provisioning GPU environment lane under `scripts/gpu/` plus two
small device-resolution changes, all infrastructure — no lane algorithm or
evidence semantics was touched:

- `scripts/gpu/provision-gpu-env.sh`: idempotent bootstrap for any Linux host.
  Creates `.gcl-python-gpu/` (the exact venv path the TS release orchestrator
  auto-detects and recorded release runs cite via `GCL_PYTHON`), installs the
  pinned stack (`requirements/gcl-python-gpu.txt`, or `--cpu` for the
  `gcl-python.txt` dev pin), warms the Hugging Face cache so evidence runs can
  use `--local-files-only`, then runs the smoke check.
- `scripts/gpu/gpu-smoke.py`: host validator that loads the lane model through
  the repository's own adapter registry, fingerprints the environment (GPU
  name/VRAM/capability, driver, torch/transformers build, RAM, cores),
  measures short- and long-prompt forward throughput, checks same-device
  determinism (max abs Δlogit ≤ 1e-5) and forward-hook capture, and writes a
  JSON report with an explicit claim boundary.
- `scripts/gpu/wrap-lane-suite.py`: python assembly of the release-format
  suite (report.json/report.md/manifest.json, results nested under model-id
  subdirs) from per-model lane reports, so the full lane → suite → verifier
  cycle runs on hosts without Node.js. Field-for-field compatible with the TS
  orchestrator output, including `synthetic: false, surrogate: false`.
- `scripts/gpu/colab-gcl-ordered-path-ood-stress.ipynb`: turns a free Colab T4
  into the GPU environment — clones the public GitHub remote, provisions,
  validates with the smoke, runs the recorded France OOD-stress configuration
  live on the T4, assembles the suite, verifies it with
  `verify_ordered_path_ood_stress.py`, and zips the artifacts.
- Device plumbing: the ordered-path runner and the TS release orchestrator now
  honor a `GCL_DEVICE` environment override (runner: env wins over `--device`;
  orchestrator: `--device` comes from `GCL_DEVICE`, default `cuda`). Defaults
  are unchanged, so existing commands behave identically.
- `docs/gpu-environments.md`: the environment contract, the host matrix with
  runbooks (Colab free T4, owned GPU, rented Linux CUDA hosts, CPU plumbing
  smoke), the evidence-grading rules, and the verification record.

The evidence rule was deliberately left strict: the OOD-stress verifier still
requires `model.device == "cuda"`, so CPU-produced artifacts are rejected as
dev smokes and the free-T4 notebook produces genuine release-grade artifacts
by satisfying the rule rather than relaxing it.

## Alternatives considered

- Weakening the verifier to accept CPU artifacts: rejected, it would downgrade
  the release lane's evidence grade instead of finding real GPU compute.
- Requiring Node.js everywhere and reusing only the TS orchestrator: rejected
  for notebooks and containers; the python suite wrapper keeps the cycle
  portable while the TS gate remains the full multi-model path.
- Adding CUDA batching inside the lane for GPU throughput: rejected this
  round; per-prompt evaluation is the determinism contract behind recorded
  replay errors of 0.0, and padding/batching would change numeric ordering.
  Throughput on GPU is measured, not optimized, until a lane needs it.
- Renting paid GPU compute: unnecessary at the current scale; free T4 covers
  the ordered-path lane, and the provision script covers any future rented
  host without code changes.

## Consequences

- Any machine with a browser and a Google account can now produce
  release-grade ordered-path OOD stress evidence for this repository.
- Any owned or rented Linux CUDA host becomes a lane host with one command and
  a machine-checked verdict, instead of undocumented manual venv setup.
- `GCL_DEVICE` gives containers and notebooks device control without argument
  edits; the artifact still records the resolved device, so provenance is
  preserved.
- Hosts are fingerprinted before lanes run (gpu-smoke report), so throughput
  and device claims in future artifacts can cite a recorded host check.

## Verification

On the CPU plumbing sandbox (2 cores, 4 GB RAM, no CUDA, torch 2.10.0+cpu):

- `gpu-smoke.py` device auto-resolution, determinism (max abs Δlogit 0.0),
  hook capture: verdict OK, exit 0.
- `provision-gpu-env.sh --cpu` end to end: PROVISIONED, exit 0.
- Live ordered-path OOD stress run with `GCL_DEVICE=cpu` (France, 6 OOD
  paraphrases, rescue + replacement): completed in 3m34s, replay error 0.0,
  sign-flip break and component-rebinding replacement — structurally matching
  the recorded RTX 3070 artifact — with the why-report layer populated.
- `wrap-lane-suite.py` output is processed fully by the verifier; the only
  rejection for the CPU artifact is the intended `device != "cuda"` rule
  (negative control), while the recorded RTX 3070 suite verifies green
  (positive control, `ORDERED_PATH_OOD_STRESS_ARTIFACT_VERIFIED`).
- TS release gate plumbing smoke with `GCL_DEVICE=cpu` and a scratch output
  root: exit 0, complete output layout, venv auto-detected through the
  `.gcl-python-gpu` path.

The T4 portion of the Colab notebook cannot be executed from the CPU sandbox;
its cells use only commands proven on the CPU path (provision, lane, wrapper,
verifier) with `GCL_DEVICE=cuda`, and the notebook records its own claim
boundary.

## Cross-references

- `scripts/gpu/provision-gpu-env.sh`
- `scripts/gpu/gpu-smoke.py`
- `scripts/gpu/wrap-lane-suite.py`
- `scripts/gpu/colab-gcl-ordered-path-ood-stress.ipynb`
- `docs/gpu-environments.md`
- `tools/llm-reader-ordered-path-stability.py` (GCL_DEVICE resolution)
- `scripts/benchmark-nn-live-ordered-path-stability.ts` (GCL_DEVICE passthrough)
- `requirements/gcl-python-gpu.txt`, `requirements/gcl-python.txt`
- Prior lane context: `.agents/notes/implemented/feature/2026-11-06-ordered-path-why-report.md`
