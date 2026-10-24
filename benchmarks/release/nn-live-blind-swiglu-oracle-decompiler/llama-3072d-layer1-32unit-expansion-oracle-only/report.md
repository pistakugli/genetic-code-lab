# Blind 3072D SwiGLU Oracle Decompiler Sweep

- pass: `false`
- cases: `0/1`
- min R2: `Infinity`
- max MAE: `-Infinity`
- max worst error: `-Infinity`
- min subspace principal cosine: `Infinity`
- min post-hoc paired atom joint cosine: `Infinity`
- min paired atoms above threshold: `null`
- min source cancellation fraction: `Infinity`
- minimum source pair correlation: `Infinity`
- max unit count: `-Infinity`
- max atom count: `-Infinity`
- total raw oracle calls: `0`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

- layer1-units27-57-413-460-500-618-778-1141-1607-1970-2062-2215-2529-2847-3530-3627-3679-3786-4033-4186-4236-4624-4906-5014-5041-5409-6455-7283-7434-7598-7663-7720: `fail missing report.json; process exit 1; stderr tail: [reserve] holding 4200 MiB on cuda | Traceback (most recent call last): | torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1000.00 MiB. GPU 0 has a total capacity of 7.65 GiB of which 577.94 MiB is free. Including non-PyTorch memory, this process has 4.05 GiB memory in use. Process 2531178 has 2.57 GiB memory in use. Of the allocated memory 3.78 GiB is allocated by PyTorch, and 46.25 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer1-units27-57-413-460-500-618-778-1141-1607-1970-2062-2215-2529-2847-3530-3627-3679-3786-4033-4186-4236-4624-4906-5014-5041-5409-6455-7283-7434-7598-7663-7720-32atom-forward-coordinate-oracle-only/report.json)

Boundary: this is a selected-unit sweep, not full FFN or arbitrary LLM decompilation.
