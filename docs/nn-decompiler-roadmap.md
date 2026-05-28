# NN Decompiler Roadmap

This roadmap keeps the neural-network decompiler work honest. A lane becomes a
release claim only when it has a command, artifact, gate, and clear boundary.

## Engineering Rule

Problems must not be silenced or bypassed. If a benchmark reveals a failure,
the failure becomes an explicit test or benchmark variant, and the implementation
must be fixed or the limitation must remain visible as a failing/non-promoted
gate. Do not make the task easier just to preserve a pass.

## SOTA Seriousness Checklist

The NN decompiler roadmap is now ordered by dependency, not by excitement. A
later phase may not be promoted while an earlier phase has an unresolved hard
failure.

Phase 0: shallow clean/noisy geometry floor.

- Status: release-gated for 1D, 2D, 3D, randomized 3D, clean 4D/5D line probes,
  noisy 4D/5D line probes, generated adversarial noisy ND, and adaptive
  adversarial noisy ND.
- Purpose: prove that the oracle/probing layer can recover observable
  piecewise-linear geometry before deep composition is attempted.

Phase 1: deep 1D and 2D output-equivalent recovery.

- Status: release-gated for `benchmark:nn-deep-blackbox-relu1d-decompile`,
  `benchmark:nn-deep-blackbox-relu2d-region-graph`, and
  `benchmark:nn-deep-blackbox-relu2d-region-graph:randomized`.
- Purpose: recover the observable folded function first. Original hidden
  weights and exact layer attribution remain out of scope for output-only
  access.

Phase 2: deep 2D hardening.

- Status: release-gated for `benchmark:nn-noisy-deep-blackbox-relu2d-region-graph`
  and `benchmark:nn-adaptive-adversarial-deep-blackbox-relu2d-region-graph`.
- Current proof: repeated noisy/quantized oracle calls on deterministic folded
  2D seed sweep plus adaptive generated near-parallel, offset-collision,
  cancellation, low-amplitude, boundary-grazing, and mixed-oblique folded
  teachers, with median aggregation and explicit query-budget accounting.
- Required gates: held-out accuracy, region/adjacency evidence, query budget,
  first-layer-only baseline gap, and failure artifact if a random/adversarial
  teacher breaks the solver.

Phase 3: compact/adaptive/topology boundary extraction.

- Status: compact-boundary extraction, the first adaptive-budget probe
  selector, boundary-refined local switching, subcell-boundary adaptive
  validation, and topology-boundary recovered-adjacency geometry are
  release-gated for
  `benchmark:nn-compact-deep-blackbox-relu2d-boundary` and
  `benchmark:nn-adaptive-budget-deep-blackbox-relu2d-region-graph`, and
  `benchmark:nn-boundary-refined-deep-blackbox-relu2d-region-graph`, and
  `benchmark:nn-subcell-boundary-adaptive-deep-blackbox-relu2d-region-graph`,
  and `benchmark:nn-topology-boundary-deep-blackbox-relu2d-region-graph`.
- Current boundary: oracle-guided subcell validation selects the lowest passing
  boundary-aware grid and records too-coarse failed candidates; topology-
  boundary geometry clips every recovered adjacency edge to an affine equality
  line inside an expanded contact corridor with no fallback segments. It is
  still recovered-adjacency topology, not topology-perfect true-region merging.
- Next target after topology-boundary geometry: the same boundary discipline in
  deep ND folded recovery.
- Required gates: same accuracy as fixed grid with fewer oracle calls or a
  strictly better boundary artifact.

Phase 4: deep ND folded recovery.

- Status: first release-gated foundation via
  `benchmark:nn-deep-blackbox-relu-nd-region-sampler`, now hardened with
  `benchmark:nn-noisy-adaptive-adversarial-deep-blackbox-relu-nd-region-sampler`
  and budget-hardened with
  `benchmark:nn-adaptive-budget-noisy-deep-blackbox-relu-nd-region-sampler`,
  then geometry-hardened with
  `benchmark:nn-nd-adjacency-geometry-deep-blackbox-relu-region-sampler`.
- Current proof: controlled 3D/4D folded two-layer ReLU teachers recovered as
  output-equivalent affine-region samplers from oracle-only random local-gradient
  probes, with query-free kNN region assignment, noisy/quantized repeated-call
  aggregation, adaptive region-vote/local-affine KNN selection, sequential
  noisy holdout budget promotion with rejected-candidate accounting, query-
  budget accounting, recovered ND adjacency geometry from cross-region contact
  witnesses plus affine-equality hyperplanes, a 5D clean/noisy hard-profile
  sweep, a research-only 6D guarded probe, and a global-linear baseline gap.
- Required gates: deterministic randomized ND teachers, noisy/adaptive
  adversarial hard-profile sweeps, query-budget accounting, and baseline
  comparison against shallow/linear approximations. Next hardening step is
  adaptive 6D probe-distribution selection, stronger external extraction
  baselines, and harder topology-perfect ND cell-complex audits.

Phase 5: layer attribution only with extra observability.

- Activation-access lanes may claim layer attribution when hidden activations
  are explicitly available.
- Output-only lanes must keep the claim at equivalent-function recovery unless
  an identifiability assumption is stated and gated.

Phase 6: smooth LLM activation and FFN recovery.

- Status: planned, not release-gated.
- Scope: GELU, SiLU, and SwiGLU-style LLM feed-forward channels are not
  piecewise-linear kink geometry. The 5D ReLU recovered-adjacency gates provide
  oracle discipline, query accounting, and benchmark hygiene, but they do not
  prove GELU/SwiGLU recovery.
- First target command:
  `npm run benchmark:nn-gelu-swiglu-decompiler`.
- Stage A: synthetic single-neuron recovery.
  - GELU teacher: `y = a * gelu(w dot x + b) + c`.
  - SiLU/SwiGLU teacher: `y = (u dot x + bu) * silu(g dot x + bg)`.
  - GCL receives only `f(vector)` oracle calls.
  - Gate: held-out R2 above `0.999` for clean teachers, explicit direction or
    subspace error where identifiable, and a win against linear/ridge baselines.
- Stage B: synthetic FFN block recovery.
  - Multiple GELU/SwiGLU channels are summed into a block output slice.
  - Gate: output-equivalent transparent surrogate with per-channel or subspace
    evidence, held-out accuracy, query budget, and failure accounting for
    collinear/cancelling gates.
- Stage C: local transformer MLP hook.
  - Use a small local transformer block with captured residual-stream inputs and
    one MLP output slice/channel target.
  - Gate: compare GCL against linear/ridge/MLP surrogate baselines on held-out
    activation rows, with all data extraction and tokenization failures treated
    as hard failures.
- Claim boundary: until these gates exist and pass, GCL may not claim LLM neuron
  decompilation, GELU/SwiGLU reverse engineering, or mechanistic understanding
  of production transformer blocks.

The following items are still required before any broad SOTA language:

- Wider randomized and adversarial 3D sweep, beyond the current smoke,
  non-degenerate randomized, and two-teacher cancellation hard-case gates.
  Status: started with `benchmark:nn-noisy-blackbox-relu3d-adversarial`;
  broaden further with generated/adaptive adversarial search before using broad
  SOTA language.
- True general-ND solver, not only 3D. Status: started with the 4D line-probe
  gate and broadened to deterministic clean/noisy 4D/5D seed sweeps plus a
  generated adversarial noisy 4D/5D profile sweep and an adaptive
  search-generated noisy 4D/5D adversary lane.
- Random-direction probing instead of relying on z-slices and surface
  refinement only. Status: started with `benchmark:nn-blackbox-relu-nd-line-probe`.
- Deep ReLU composition recovery, not only shallow hidden-layer recovery.
  Status: started with `benchmark:nn-deep-blackbox-relu1d-decompile` and
  broadened to `benchmark:nn-deep-blackbox-relu2d-region-graph`, which recovers
  an output-equivalent affine-region adjacency graph for a folded two-layer 2D
  ReLU black box.
- Query-budget benchmark: total oracle calls, repeated calls per point, and
  accuracy versus query cost. Status: started with
  `benchmark:nn-adaptive-budget-deep-blackbox-relu2d-region-graph` and
  broadened to `benchmark:nn-boundary-refined-deep-blackbox-relu2d-region-graph`;
  further broadened to
  `benchmark:nn-subcell-boundary-adaptive-deep-blackbox-relu2d-region-graph`,
  which records rejected too-coarse candidate grids before promotion, and to
  `benchmark:nn-topology-boundary-deep-blackbox-relu2d-region-graph`, which
  requires zero fallback topology segments and endpoint residual below
  `1e-9`. Next broaden this boundary discipline beyond 2D before broad SOTA
  language.
- Comparison against strong methods for piecewise-linear model extraction, with
  clear baselines and failure accounting.
- Smooth activation decompilation for GELU/SwiGLU LLM-style FFN channels, with
  separate gates from the ReLU recovered-adjacency geometry lane.

## Current Frontier Target

The current frontier target is deep ND folded recovery, using the now-gated
deep 2D topology-boundary lane as the boundary-geometry floor.

The adaptive adversarial noisy 4D/5D ND shallow-ReLU lane and the adaptive
adversarial noisy folded-2D deep-ReLU lane are now release-gated robustness
floors. GCL now has release-gated lower-budget deep-2D selection, boundary-
refined local switching, subcell-boundary adaptive validation with an explicit
rejected-candidate ledger, topology-preserving recovered-adjacency boundary
geometry, noisy/adaptive adversarial deep-ND folded recovery, deep-ND query-
budget reduction, and recovered-adjacency geometry beyond 2D. The next
capability jump is query-efficiency hardening and stronger external package
baselines for the 6D lane. The 6D lane now has a guarded research probe,
adaptive probe-distribution search, a broader clean/noisy hard-profile sweep
with rejected-candidate accounting, and a dependency-free external-method
baseline comparison. It is not a broad SOTA claim until query cost and stronger
external packages are handled in a promotion gate.

Target commands:

- `npm run benchmark:nn-deep-blackbox-relu-nd-region-sampler`
- `npm run benchmark:nn-noisy-adaptive-adversarial-deep-blackbox-relu-nd-region-sampler`
- `npm run benchmark:nn-adaptive-budget-noisy-deep-blackbox-relu-nd-region-sampler`
- `npm run benchmark:nn-nd-adjacency-geometry-deep-blackbox-relu-region-sampler`
- `npm run benchmark:nn-nd-adjacency-geometry-5d-sweep`
- `npm run benchmark:nn-nd-adjacency-geometry-6d-sweep`
- `npm run benchmark:nn-nd-adjacency-geometry-6d-adaptive-probe`
- `npm run benchmark:nn-nd-adjacency-geometry-6d-baselines`
- `npm run experiment:nn-nd-adjacency-geometry-5d`
- `npm run experiment:nn-nd-adjacency-geometry-6d`
- `npm run benchmark:nn-deep-blackbox-relu2d-region-graph`
- `npm run benchmark:nn-deep-blackbox-relu2d-region-graph:randomized`
- `npm run benchmark:nn-noisy-deep-blackbox-relu2d-region-graph`
- `npm run benchmark:nn-adaptive-adversarial-deep-blackbox-relu2d-region-graph`
- `npm run benchmark:nn-compact-deep-blackbox-relu2d-boundary`
- `npm run benchmark:nn-adaptive-budget-deep-blackbox-relu2d-region-graph`
- `npm run benchmark:nn-boundary-refined-deep-blackbox-relu2d-region-graph`
- `npm run benchmark:nn-subcell-boundary-adaptive-deep-blackbox-relu2d-region-graph`
- `npm run benchmark:nn-topology-boundary-deep-blackbox-relu2d-region-graph`

Current status:

- Direct benchmark PASS on 2026-05-25.
- Teacher: 2-input / 4-hidden / 4-hidden / 1-output ReLU MLP.
- GCL access: oracle calls only, `f(x,y)`.
- Recovered affine regions: 21.
- Region adjacency edges: 46.
- Kink-cell boundaries: 964.
- GCL MAE: `1.485e-4`.
- GCL R2: `0.999999399`.
- Plane baseline MAE: `0.830813`.
- First-layer-only baseline MAE: `0.230876`.
- Raw oracle calls: `51005`.
- Artifact:
  `benchmarks/release/nn-deep-blackbox-relu2d-region-graph/folded-two-layer/report.json`
- The lane passes inside `npm run release:gate` on 2026-05-25.
- Randomized seed-sweep benchmark PASS on 2026-05-25.
- Randomized teachers: 4 deterministic 2-input / 4-hidden / 4-hidden /
  1-output ReLU MLPs.
- Randomized passed teachers: 4/4.
- Randomized minimum recovered regions: 20.
- Randomized minimum deep-only region estimate: 15.
- Randomized minimum adjacency edges: 40.
- Randomized minimum kink-cell boundaries: 839.
- Randomized max MAE: `1.890e-4`.
- Randomized min R2: `0.999996583`.
- Randomized total raw oracle calls: `204020`.
- Randomized artifact:
  `benchmarks/release/nn-deep-blackbox-relu2d-region-graph-randomized/seed-sweep/report.json`
- Noisy/quantized seed-sweep benchmark PASS on 2026-05-25.
- Noisy teachers: 2 deterministic 2-input / 4-hidden / 4-hidden / 1-output
  ReLU MLPs.
- Noisy oracle: Gaussian output noise `0.003`, quantization `0.001`,
  `21` repeated calls per probe point, median aggregation.
- Noisy passed teachers: 2/2.
- Noisy minimum recovered regions: 125.
- Noisy minimum deep-only region estimate: 120.
- Noisy minimum adjacency edges: 304.
- Noisy minimum kink-cell boundaries: 2061.
- Noisy max MAE: `0.002395`.
- Noisy min R2: `0.99994584`.
- Noisy total raw oracle calls: `1377810`.
- Noisy artifact:
  `benchmarks/release/nn-noisy-deep-blackbox-relu2d-region-graph/seed-sweep/report.json`
- Adaptive-adversarial benchmark PASS on 2026-05-25 after a failing first
  attempt exposed near-parallel and weak-signal hard cases.
- Adaptive-adversarial teachers: 12 generated 2-input / 4-hidden / 4-hidden /
  1-output folded ReLU MLPs across near-parallel, offset-collision,
  cancellation, low-amplitude, boundary-grazing, and mixed-oblique profiles.
- Adaptive-adversarial passed final candidates: 12/12.
- Adaptive-adversarial selected hard cases passed: 4/4.
- Adaptive-adversarial minimum selected regions: 75.
- Adaptive-adversarial minimum selected deep-only region estimate: 70.
- Adaptive-adversarial minimum selected adjacency edges: 246.
- Adaptive-adversarial max selected MAE: `0.006062`.
- Adaptive-adversarial min selected R2: `0.99895766`.
- Adaptive-adversarial max selected first-layer MAE ratio: `0.126213`.
- Adaptive-adversarial total raw oracle calls: `9313740`.
- Adaptive-adversarial artifact:
  `benchmarks/release/nn-adaptive-adversarial-deep-blackbox-relu2d-region-graph/adaptive-search/report.json`
- Compact boundary benchmark PASS on 2026-05-25.
- Compact boundary teachers: 3 fixed/randomized 2-input / 4-hidden /
  4-hidden / 1-output folded ReLU MLPs.
- Compact boundary passed teachers: 3/3.
- Compact boundary minimum compression ratio: `19.068182x` versus raw
  cell-edge boundary contacts.
- Compact boundary max fit RMSE: `0.075528`.
- Compact boundary max MAE: `1.775e-4`.
- Compact boundary min R2: `0.999996769`.
- Compact boundary total raw oracle calls: `153015`.
- Compact boundary artifact:
  `benchmarks/release/nn-compact-deep-blackbox-relu2d-boundary/fixed-plus-randomized/report.json`
- Adaptive-budget benchmark PASS on 2026-05-25.
- Adaptive-budget teachers: 3 fixed/randomized 2-input / 4-hidden /
  4-hidden / 1-output folded ReLU MLPs.
- Adaptive-budget passed teachers: 3/3.
- Adaptive-budget selected grid widths: `89, 89, 89`.
- Adaptive-budget minimum oracle reduction versus fixed `101x101`: `20.102%`.
- Adaptive-budget total oracle calls: `122256` versus fixed `153015`.
- Adaptive-budget max MAE: `2.407e-4`.
- Adaptive-budget min R2: `0.999992859`.
- Adaptive-budget artifact:
  `benchmarks/release/nn-adaptive-budget-deep-blackbox-relu2d-region-graph/fixed-plus-randomized/report.json`
- Adaptive-budget boundary: this is a validated whole-grid budget selector
  with fallback, not the final form of local spatial refinement.
- Boundary-refined benchmark PASS on 2026-05-25.
- Boundary-refined teachers: 3 fixed/randomized 2-input / 4-hidden /
  4-hidden / 1-output folded ReLU MLPs.
- Boundary-refined grid: `57x57` plus held-out validation.
- Boundary-refined passed teachers: 3/3.
- Boundary-refined minimum oracle reduction versus fixed `101x101`: `65.901%`.
- Boundary-refined total oracle calls: `52176` versus fixed `153015`.
- Boundary-refined max MAE: `7.994e-5`.
- Boundary-refined min R2: `0.999999013`.
- Boundary-refined minimum held-out coarse-to-refined MAE improvement:
  `7.996471x`.
- Boundary-refined artifact:
  `benchmarks/release/nn-boundary-refined-deep-blackbox-relu2d-region-graph/fixed-plus-randomized/report.json`
- Boundary-refined boundary: compact boundary segments are now an active local
  switching mechanism, but this still does not claim topology-perfect true-region
  merging or original hidden-layer identity.
- Subcell-boundary adaptive benchmark PASS on 2026-05-25.
- Subcell-boundary adaptive teachers: 3 fixed/randomized 2-input / 4-hidden /
  4-hidden / 1-output folded ReLU MLPs.
- Subcell-boundary adaptive selected grid widths: `37, 37, 41`.
- Subcell-boundary adaptive rejected too-coarse attempts: 1.
- Subcell-boundary adaptive total oracle calls: `33528` versus fixed `153015`
  and boundary-refined 57x57 `52176`.
- Subcell-boundary adaptive total oracle reduction: `78.088%` versus fixed
  `101x101`, and `35.741%` versus boundary-refined 57x57.
- Subcell-boundary adaptive max MAE: `2.837e-4`.
- Subcell-boundary adaptive min R2: `0.999990210`.
- Subcell-boundary adaptive minimum held-out coarse-to-subcell MAE improvement:
  `3.550431x`.
- Subcell-boundary adaptive artifact:
  `benchmarks/release/nn-subcell-boundary-adaptive-deep-blackbox-relu2d-region-graph/fixed-plus-randomized/report.json`
- Topology-boundary benchmark PASS on 2026-05-26 after a first randomized
  attempt exposed fallback contact-corridor segments on seed `83`.
- Topology-boundary teachers: 3 fixed/randomized 2-input / 4-hidden /
  4-hidden / 1-output folded ReLU MLPs.
- Topology-boundary passed teachers: 3/3.
- Topology-boundary fallback segments: 0.
- Deep ND region-sampler benchmark PASS on 2026-05-26.
- Deep ND teachers: 4 deterministic 3D/4D folded two-layer ReLU MLPs.
- Deep ND visibility: oracle calls only, `f(vector)`, no hidden activations,
  layer widths, or teacher weights.
- Deep ND passed teachers: 4/4.
- Deep ND minimum recovered regions: `40`.
- Deep ND minimum deep-only region estimate: `35`.
- Deep ND max MAE: `0.011095250643373489`.
- Deep ND min R2: `0.998534063321796`.
- Deep ND min global-linear MAE improvement ratio: `42.81846369086524`.
- Deep ND total raw oracle calls: `315800`.
- Deep ND artifact:
  `benchmarks/release/nn-deep-blackbox-relu-nd-region-sampler/3d-4d-seed-sweep/report.json`
- Noisy adaptive-adversarial deep ND benchmark PASS on 2026-05-27.
- Noisy adaptive-adversarial deep ND teachers: 7 generated 3D/4D folded
  two-layer ReLU MLPs across near-parallel, cancellation, low-amplitude, and
  mixed-oblique hard profiles.
- Noisy adaptive-adversarial deep ND visibility: noisy/quantized oracle calls
  only, `f(vector)`, no hidden activations, layer widths, or teacher weights.
- Noisy adaptive-adversarial deep ND passed teachers: 7/7.
- Noisy adaptive-adversarial selected local-affine KNN attempts: `2`.
- Noisy adaptive-adversarial minimum recovered regions: `6`.
- Noisy adaptive-adversarial median deep-only region estimate: `25`.
- Noisy adaptive-adversarial max MAE: `0.026450213096249696`.
- Noisy adaptive-adversarial min R2: `0.9830270507435355`.
- Noisy adaptive-adversarial min global-linear MAE improvement ratio:
  `7.392219720751436`.
- Noisy adaptive-adversarial total selected raw oracle calls: `4274160`.
- Noisy adaptive-adversarial total adaptive-search raw oracle calls:
  `14132880`.
- Noisy adaptive-adversarial artifact:
  `benchmarks/release/nn-noisy-adaptive-adversarial-deep-blackbox-relu-nd-region-sampler/3d-4d-adaptive-search/report.json`
- Adaptive-budget noisy deep ND benchmark PASS on 2026-05-27.
- Adaptive-budget noisy deep ND teachers: same 7 generated 3D/4D folded
  two-layer ReLU MLP hard profiles as the noisy adaptive-adversarial gate.
- Adaptive-budget selection: sequential noisy holdout promotion over low/mid/high
  region-vote/local-affine candidates, with rejected candidates written into the
  report.
- Adaptive-budget passed teachers: 7/7.
- Adaptive-budget selected low/mid/high budgets: `6/0/1`.
- Adaptive-budget selected local-affine KNN attempts: `3`.
- Adaptive-budget rejected candidates recorded: `7`.
- Adaptive-budget max attempted candidates per teacher: `5`.
- Adaptive-budget max MAE: `0.03291800518211379`.
- Adaptive-budget min R2: `0.9865538047691716`.
- Adaptive-budget selected raw oracle calls: `1862760`.
- Adaptive-budget adaptive-search raw oracle calls: `3404120`.
- Adaptive-budget selected-call reduction vs previous noisy deep-ND release:
  `56.418%`.
- Adaptive-budget search-call reduction vs previous noisy adaptive-search cost:
  `75.913%`.
- Adaptive-budget artifact:
  `benchmarks/release/nn-adaptive-budget-noisy-deep-blackbox-relu-nd-region-sampler/3d-4d-budget-search/report.json`
- ND adjacency-geometry benchmark PASS on 2026-05-27.
- ND adjacency-geometry teachers: clean/noisy 3D/4D folded two-layer ReLU MLP
  profiles, with no hidden teacher state exposed to GCL.
- ND adjacency-geometry mechanism: recovered affine-region assignments are
  scanned for cross-region nearest-neighbor contact witnesses; every passing
  edge stores an affine-equality hyperplane and bounded projection residual.
- ND adjacency-geometry passed teachers: 4/4.
- ND adjacency-geometry min passing edges: `62`.
- ND adjacency-geometry min largest-component ratio: `0.9622641509433962`.
- ND adjacency-geometry max normalized projection RMSE:
  `0.9926122288520228`.
- ND adjacency-geometry max MAE: `0.035446586421537286`.
- ND adjacency-geometry min R2: `0.9973558277920783`.
- ND adjacency-geometry total raw oracle calls: `805100`.
- ND adjacency-geometry artifact:
  `benchmarks/release/nn-nd-adjacency-geometry-deep-blackbox-relu-region-sampler/3d-4d-clean-noisy/report.json`
- ND adjacency-geometry stabilization check on 2026-05-28: the existing 3D/4D
  clean/noisy release benchmark was run three times consecutively and passed
  all three cycles with identical gate metrics: `4/4` teachers, min passing
  edges `62`, min largest-component ratio `0.9622641509433962`, max normalized
  projection RMSE `0.9926122288520228`, max MAE `0.035446586421537286`, min
  R2 `0.9973558277920783`.
- 5D single-probe experiment PASS on 2026-05-28.
- 5D probe criteria were deliberately narrow: R2 > `0.99`, regionCount > `50`,
  and raw oracle calls < `2000000`; it was not promoted to release capability.
- 5D probe result: R2 `0.9987241546608961`, MAE
  `0.01894093946992718`, regionCount `57`, raw oracle calls `198000`,
  geometry pass `true`, passing edges `432`, largest-component ratio
  `0.9649122807017544`, max normalized projection RMSE
  `0.5429795030091397`.
- 5D probe artifact:
  `benchmarks/research/nn-nd-adjacency-geometry-5d/single-probe/report.json`
- 5D clean/noisy sweep PASS on 2026-05-28.
- First 5D sweep attempt failed correctly instead of being hidden: clean/noisy
  accuracy passed, but low-amplitude and near-parallel profiles collapsed below
  the `regionCount > 50` gate. The repaired benchmark tightened per-profile
  noisy clustering and increased sample budgets while keeping each teacher
  below `2000000` raw oracle calls.
- 5D sweep result: `6/6` teachers passed across clean, low-amplitude,
  cancellation, near-parallel, and weak-support profiles; min region count
  `57`, min passing edges `462`, min largest-component ratio
  `0.9672131147540983`, max normalized projection RMSE `1.2786982674727818`,
  max MAE `0.07984871493240715`, min R2 `0.985430220526229`, max raw oracle
  calls per teacher `1188000`.
- 5D sweep artifact:
  `benchmarks/release/nn-nd-adjacency-geometry-5d-sweep/clean-noisy-hard-profiles/report.json`
- 6D guarded research probe PASS on 2026-05-28.
- 6D probe result: `4/4` teachers passed across clean, noisy cancellation, and
  noisy near-parallel profiles; min region count `63`, min passing edges `374`,
  min largest-component ratio `0.9850746268656716`, max normalized projection
  RMSE `0.939255366086159`, max MAE `0.09099788983192539`, min R2
  `0.9674291733376437`, max raw oracle calls per teacher `1989000`.
- 6D probe artifact:
  `benchmarks/research/nn-nd-adjacency-geometry-6d/guarded-probe/report.json`
- 6D adaptive probe-distribution search PASS on 2026-05-28.
- 6D adaptive search result: `4/4` teachers passed across clean, noisy
  cancellation, and noisy near-parallel profiles; `9` probe candidates
  evaluated; `5` rejected candidates were genuine gate failures; min region
  count `63`, min passing edges `374`, max MAE `0.09099788983192539`, min R2
  `0.9674291733376437`, max selected raw oracle calls per teacher `1989000`,
  total selected raw oracle calls `4537000`, and total adaptive search raw
  oracle calls `14482000`.
- 6D adaptive artifact:
  `benchmarks/research/nn-nd-adjacency-geometry-6d-adaptive-probe/adaptive-probe-search/report.json`
- 6D release-candidate hard-profile sweep PASS on 2026-05-28.
- First 6D sweep attempt failed correctly instead of being hidden: weak-support
  accuracy passed, but region count was only `49` against the `56` floor, and
  the total search budget was `24836500` raw oracle calls against the
  `24000000` gate. The repaired sweep tightened weak-support clustering and
  shortened the near-parallel candidate list while keeping the failure ledger.
- 6D sweep result: `6/6` teachers passed across clean, cancellation,
  near-parallel, low-amplitude, and weak-support profiles; `12` candidates
  evaluated; `4` rejected candidates were genuine gate failures; min region
  count `57`, min passing edges `374`, max MAE `0.09099788983192539`, min R2
  `0.9674291733376437`, max selected raw oracle calls per teacher `1989000`,
  total selected raw oracle calls `8105500`, and total adaptive search raw
  oracle calls `19220500`.
- 6D sweep artifact:
  `benchmarks/research/nn-nd-adjacency-geometry-6d-sweep/release-candidate-hard-profiles/report.json`
- 6D dependency-free external-method baseline comparison PASS on 2026-05-28.
- 6D baseline comparison covered global linear ridge, quadratic ridge,
  random-ReLU ridge, local-affine kNN, and ExtraTrees-style forest baselines.
  GCL won MAE on `6/6` teachers; the best measured baseline on every teacher
  was local-affine kNN. GCL max MAE was `0.09099788983192539`, min R2 was
  `0.9674291733376437`, and max GCL MAE ratio versus the best baseline was
  `0.8155502106737931`.
- 6D baseline artifact:
  `benchmarks/research/nn-nd-adjacency-geometry-6d-baselines/selected-sweep-external-baselines/report.json`
- 6D is stronger now, but still not a broad SOTA claim. The baseline comparison
  shows an accuracy win against the dependency-free methods, while also showing
  the query-cost boundary: GCL used up to `1989000` raw oracle calls per
  teacher, while these direct-sample baselines used at most `18000`. Next
  target: query-efficiency hardening plus stronger external package baselines
  before release-gate promotion.
- Topology-boundary max endpoint line residual: `4.441e-16`.
- Topology-boundary minimum topology segment length: `0.070194`.
- Topology-boundary minimum boundary compression ratio: `10.022222x`.
- Topology-boundary max MAE: `6.393e-4`.
- Topology-boundary min R2: `0.999979703`.
- Topology-boundary total raw oracle calls: `48735`.
- Topology-boundary artifact:
  `benchmarks/release/nn-topology-boundary-deep-blackbox-relu2d-region-graph/fixed-plus-randomized/report.json`
- Topology-boundary passes inside `npm run release:gate` on 2026-05-26.
- Fixed failure: the first topology-boundary corridor used only raw grid
  contact midpoints and missed the affine equality line for three seed-83
  segments. The solver now expands the contact corridor with projected contact
  points, so valid recovered equality lines clip without fallback while the
  prediction segments remain unchanged.
- Direct boundary-offset/root-localization prototypes were not promoted because
  they regressed held-out accuracy on randomized teachers. The release-gated
  fix is lower-risk oracle-guided subcell grid validation with explicit failed
  candidate accounting.
- The first adaptive run was not silenced: it failed on near-parallel /
  low-amplitude cases, then the solver added per-region affine refitting and
  the generator was adjusted to keep weak profiles non-degenerate under the
  benchmark noise model.
- Fixed failure: the first noisy attempt failed because the solver capped
  explicit `gradientStep` to 10% of a grid cell, shrinking the requested
  noise-tolerant step `0.075` to `0.00494` and amplifying noise into false
  gradients. The solver now preserves explicit gradient steps up to a domain
  safety cap, and the regression test checks repeated-query median probing.

Required behavior:

- Treat the deep black box as an output function first: recover the observable
  piecewise-linear region graph accurately enough to predict held-out points.
- Prove that the recovered surface is not just a plane or first-layer-only
  ReLU basis by gating on region count, adjacency, kink-cell boundaries, and
  strong baseline improvements.
- Keep the boundary explicit: output-equivalent region-graph recovery is a
  valid black-box claim; original hidden-weight recovery and exact layer
  attribution are not identifiable from output-only access without more
  assumptions.

Next exits:

- Move from 2D folded region graphs to ND deep folded region recovery.
- Keep topology-boundary geometry as a non-negotiable recovered-adjacency gate
  for future deep-ND folded recovery.
- Add an activation-access control lane that can validate layer attribution
  when hidden activations are explicitly available.

## Previous Frontier: Adaptive Adversarial ND Teacher Generation

The previous frontier target was adaptive adversarial ND teacher generation.

The generated adversarial noisy 4D/5D line-probe lane proved that GCL can
recover shallow ReLU hyperplanes under noisy and quantized oracle-only access
for fixed hard profiles: near-parallel, offset-collision, low-amplitude, and
oblique weak-support. The adaptive lane makes the hard cases search-generated
instead of hand-selected.

Target command:

- `npm run benchmark:nn-noisy-blackbox-relu-nd-adaptive-adversary`

Current status:

- Direct benchmark PASS on 2026-05-24.
- Candidate pool: 24 generated 4D/5D teachers.
- Selected hard teachers: 4.
- Passed teachers: 4/4.
- Max MAE: `0.001696`.
- Min R2: `0.99982863`.
- Max plane error: `0.008613`.
- Total raw oracle calls: `3708980`.
- Artifact:
  `benchmarks/release/nn-noisy-blackbox-relu-nd-adaptive-adversary/adaptive-search/report.json`
- The lane passes inside `npm run release:gate` on 2026-05-24.

Required behavior:

- Generate a large candidate pool of 4D/5D shallow ReLU teachers with
  adversarial knobs: near-parallel normals, offset collisions, weak readout
  amplitudes, weak first-axis line intersections, low support near domain
  boundaries, coarse quantization, and stochastic oracle noise.
- Score candidates before the expensive solve by estimated difficulty: minimum
  pairwise normal distance, offset distance, expected line delta slope, support
  fraction, boundary proximity, and cancellation risk.
- Select the hardest candidates and run the real noisy ND line-probe solver
  against them with repeated oracle calls only.
- Save the selected teachers, difficulty scores, solver metrics, query counts,
  and failures under `benchmarks/release/`.
- Gate on output-equivalent recovery: low MAE, high R2, bounded plane error,
  support thresholds, and explicit raw oracle-call budget.
- If an adaptive case fails, do not lower the gate and do not hide the failure.
  Either repair the solver and keep the case as a regression fixture, or leave
  the lane non-promoted with the failing artifact visible.

Likely solver upgrades if adaptive search exposes a failure:

- Multi-threshold kink extraction so weak line kinks are collected without
  letting noise dominate.
- RANSAC plane refit from all inliers after a candidate plane is selected.
- Weighted inlier scoring by kink confidence and absolute line delta slope.
- Residual-driven second pass that looks for missed weak hyperplanes after the
  first symbolic circuit is fitted.
- Query-budget controller that raises repeats only where line probes are
  ambiguous instead of uniformly spending calls everywhere.

Fixed failure case:

- The first adaptive run solved all selected teachers accurately, but one 5D
  mixed case exceeded the raw oracle-call budget because the generator increased
  `lineCount` once per hidden term instead of once per teacher.
- The generator now applies that budget increase once per teacher. The same
  adaptive search passes with the 5D mixed case reduced from 560 to 410 line
  probes while preserving accuracy.

Promotion boundary:

- This target can promote a stronger shallow noisy ND black-box ReLU recovery
  claim for the gated shallow 4D/5D setting.
- It still must not claim deep ReLU composition recovery, arbitrary production
  model extraction, or exact hidden-unit identity for functionally
  indistinguishable cancellation pairs.

## Direction 1: Noise-Tolerant Oracle Decompilation

Status: started and release-lane smoke gated for 1D, 2D, and 3D.

Why first:

- Real black-box APIs are not clean mathematical oracles.
- Production outputs may be stochastic, rounded, quantized, batched, or served
  through low precision inference.
- ND and deep recovery are not useful if the base oracle layer breaks under
  small output noise.

Current implemented gate:

- Command: `npm run benchmark:nn-noisy-blackbox-relu-decompile`
- Teacher: fixed 1-input/5-hidden ReLU MLP
- GCL access: repeated noisy `f(x)` only
- Noise: Gaussian output noise plus quantization
- Required: recover all kinks within tolerance, beat linear strongly, maintain
  high R2 against the clean teacher.
- Command: `npm run benchmark:nn-noisy-blackbox-relu2d-decompile`
- Teacher: fixed 2-input/3-hidden non-crossing non-vertical ReLU MLP
- GCL access: repeated noisy `f(x,y)` only
- Noise: Gaussian output noise plus quantization
- Required: recover kink-line geometry within tolerance, beat a plane baseline,
  maintain high R2 against the clean teacher.
- Command: `npm run benchmark:nn-noisy-blackbox-relu3d-decompile`
- Teacher: fixed 3-input/3-hidden non-vertical ReLU MLP
- GCL access: repeated noisy `f(x,y,z)` only
- Noise: Gaussian output noise plus quantization
- Required: recover hyperplane geometry within tolerance, beat a linear
  hyperplane baseline, maintain high R2 against the clean teacher.
- Command: `npm run benchmark:nn-noisy-blackbox-relu3d-randomized`
- Teacher: deterministic non-degenerate randomized 3-input/3-hidden ReLU MLP
  seed sweep.
- GCL access: repeated noisy `f(x,y,z)` only.
- Noise: Gaussian output noise plus quantization.
- Required: every seed recovers 3/3 hyperplanes within tolerance, beats a
  linear hyperplane baseline, and maintains high R2 against the clean teacher.
- Command: `npm run benchmark:nn-noisy-blackbox-relu3d-cancellation`
- Teacher: deterministic 3-input/3-hidden ReLU MLP hard cases with close
  opposite-sign kink pairs.
- GCL access: repeated noisy `f(x,y,z)` only.
- Method: noisy slice stitching seeds the model, then a global parametric
  ReLU-surface refiner fits the full 3D piecewise-linear surface.
- Required: recover an output-equivalent symbolic surface on every hard
  teacher, beat slice-only stitching by at least 70%, beat a linear hyperplane
  baseline by at least 85%, and maintain high R2 against the clean teacher.
- Boundary: this gate proves functional transparent surface recovery, not
  unique hidden-unit identity for cancelling near-collision pairs that can be
  functionally indistinguishable under noisy black-box output.
- Command: `npm run benchmark:nn-noisy-blackbox-relu3d-adversarial`
- Teacher: six adversarial shallow-3D ReLU MLP profiles: near-parallel,
  beta-collision, cancellation, oblique-wide, and low-amplitude mixed.
- GCL access: repeated noisy `f(x,y,z)` only.
- Method: noisy slice stitching plus global parametric ReLU-surface refinement.
- Required: every profile reconstructs an output-equivalent symbolic surface,
  beats a linear hyperplane baseline, avoids regression against slice-only
  stitching, and reports actual oracle query counts.

Next exits:

- Expand query-budget reporting beyond the current adversarial 3D sweep into
  the older 1D/2D/3D smoke and randomized lanes.
- Add adversarial precision variants: low query repeat, coarse quantization,
  and biased API rounding.
- Add harder noisy 2D cases: low-amplitude hidden units.

Fixed failure case:

- The original noisy 2D crossing-line teacher produced good MAE but wrong
  geometry for one weak kink line because bad per-slice detections polluted the
  line fit.
- The stitching layer now uses robust kink-line fitting with slice outlier
  rejection.
- Command: `npm run benchmark:nn-noisy-blackbox-relu2d-crossing`
- Required: recover crossing-line geometry and maintain high held-out R2.
- Command: `npm run benchmark:nn-noisy-blackbox-relu2d-near-parallel`
- Required: separate near-parallel kink lines without merging and maintain high
  held-out R2.
- Command: `npm run benchmark:nn-noisy-blackbox-relu2d-near-collision`
- Required: separate kink lines when two collide on an observed slice and
  maintain high held-out R2.
- The unrestricted noisy 3D randomized attempt exposed near-collision and
  cancellation cases where close opposite-sign hyperplanes were merged or missed
  by slice-only stitching.
- The 3D noisy hard case is now covered by the global refiner gate.
- Command: `npm run benchmark:nn-noisy-blackbox-relu3d-cancellation`
- Required: output-equivalent symbolic surface recovery on the hard cases and
  strong improvement over the slice-only baseline.
- The wider adversarial 3D sweep now covers near-parallel, beta-collision,
  cancellation, oblique-wide, and low-amplitude mixed profiles.
- Command: `npm run benchmark:nn-noisy-blackbox-relu3d-adversarial`
- Required: 6/6 profiles pass with explicit oracle query-budget accounting.

## Direction 2: 3D / General-ND Hyperplane Stitching

Status: started and release-lane gated for clean/noisy 3D teachers, the first
clean 4D random-direction line-probe teacher, and a deterministic randomized
4D/5D seed sweep.

Why second:

- The current 2D black-box solver already recovers non-vertical ReLU kink
  lines from horizontal slices.
- The natural generalization is to probe many 1D lines through the domain,
  recover kink locations on those lines, then stitch those points into
  hyperplanes.

Proposed architecture:

- Active line sampler: choose lines across coordinate axes and random
  directions.
- Kink point extraction: reuse robust 1D/noisy 1D decompilers per line.
- Hyperplane clustering: group recovered kink points by compatible normals.
- Region adjacency graph: maintain which linear region lies on each side of
  each recovered hyperplane.
- Verification pass: hold out random points and compare reconstructed symbolic
  circuit against the oracle.

Current implemented gate:

- Command: `npm run benchmark:nn-blackbox-relu3d-decompile`
- Teacher: fixed 3-input/3-hidden ReLU MLP with non-degenerate non-vertical
  hyperplanes.
- GCL access: `f(x,y,z)` oracle only.
- Method: recover 2D kink lines on z-slices, then stitch matched lines through
  z into hyperplanes.
- Required: recover 3/3 hyperplanes with geometry error below `1e-8`, beat a
  linear hyperplane baseline by at least 1000x, and keep held-out MAE below
  `1e-9`.
- Command: `npm run benchmark:nn-blackbox-relu3d-randomized`
- Teacher: deterministic randomized 3-input/4-hidden ReLU MLP seed sweep.
- GCL access: `f(x,y,z)` oracle only.
- Method: clean 2D slice recovery with denser `y` slicing per z-slice, then
  hyperplane stitching.
- Required: every seed recovers all hyperplanes with geometry error below
  `1e-8`, and every seed keeps held-out MAE below `1e-9`.
- Command: `npm run benchmark:nn-blackbox-relu-nd-line-probe`
- Teacher: fixed 4-input/4-hidden shallow ReLU MLP.
- GCL access: `f(vector)` oracle only.
- Method: random-direction line probes recover 1D kink points; RANSAC stitches
  kink points into N-D hyperplanes; a final linear readout fit reconstructs an
  equivalent symbolic ReLU circuit.
- Required: recover 4/4 hyperplanes, keep max plane error below `1e-3`, beat a
  linear hyperplane baseline by at least `1000x`, keep held-out MAE below
  `1e-8`, and report oracle query count.
- Command: `npm run benchmark:nn-blackbox-relu-nd-line-probe-randomized`
- Teacher: deterministic randomized 4D/5D shallow ReLU MLP seed sweep.
- GCL access: `f(vector)` oracle only.
- Method: same random-direction line probing, RANSAC hyperplane stitching, and
  linear readout fitting as the fixed 4D lane.
- Required: all randomized 4D/5D teachers recover every hyperplane, keep max
  plane error below `1e-3`, beat linear by at least `1000x`, keep MAE below
  `1e-8`, satisfy support thresholds, and stay under the query-budget cap.
- Command: `npm run benchmark:nn-noisy-blackbox-relu-nd-line-probe`
- Teacher: deterministic randomized 4D/5D shallow ReLU MLP seed sweep with
  stochastic output noise and quantization.
- GCL access: repeated noisy `f(vector)` oracle only.
- Method: random-direction line probing uses median-aggregated 1D noisy ReLU
  segmentation; weak noisy line kinks are filtered before RANSAC stitching; a
  median-aggregated final readout fit reconstructs an equivalent symbolic ReLU
  circuit.
- Required: all noisy 4D/5D teachers recover every hyperplane, keep max plane
  error below `0.06`, beat a linear hyperplane baseline by at least 75%, keep
  MAE below `0.025`, keep R2 above `0.998`, satisfy support thresholds, and
  stay under the query-budget cap.
- Command: `npm run benchmark:nn-noisy-blackbox-relu3d-decompile`
- Teacher: fixed 3-input/3-hidden ReLU MLP with noisy and quantized oracle
  output.
- GCL access: repeated noisy `f(x,y,z)` oracle only.
- Method: noisy 2D slice recovery per z, followed by robust `beta(z)`
  hyperplane stitching.
- Required: recover 3/3 hyperplanes within `0.12` geometry tolerance, beat a
  linear hyperplane baseline by at least 65%, and keep held-out MAE below
  `0.08`.
- Command: `npm run benchmark:nn-noisy-blackbox-relu3d-cancellation`
- Teacher: noisy 3D near-collision/cancellation hard cases with close
  opposite-sign hidden units.
- GCL access: repeated noisy `f(x,y,z)` oracle only.
- Method: noisy slice recovery seeds a global parametric refiner that optimizes
  the full 3D ReLU surface and readout coefficients.
- Required: output-equivalent surface MAE below `0.012`, R2 above `0.999`, and
  strong improvement over both linear and slice-only baselines.
- Command: `npm run benchmark:nn-noisy-blackbox-relu3d-adversarial`
- Teacher: wider noisy adversarial 3D profile sweep over shallow ReLU MLPs.
- GCL access: repeated noisy `f(x,y,z)` oracle only.
- Method: slice stitching seeds a global refiner; benchmark counts actual oracle
  calls.
- Required: 6/6 profiles pass, max MAE below `0.020`, R2 above `0.9985`, and
  no bad regression against slice-only stitching.

Boundary:

- This should not claim arbitrary high-dimensional network recovery until it
  survives adaptive search-generated ND profiles and deep composition recovery.
  The clean, noisy, generated adversarial, and adaptive-adversary 4D/5D shallow
  lanes now pass inside the release gate.

Next exits:

- Broaden adaptive adversarial search to larger candidate pools and stricter
  budget tiers.
- Add residual-driven second-pass recovery for missed weak hyperplanes.
- Start the deep/staked ReLU composition recovery lane.

## Direction 3: Deep ReLU Composition Recovery

Status: started with a release-lane folded 1D output-equivalent recovery gate.

Why third:

- Deep ReLU networks fold space. The visible function is still piecewise
  linear, but internal layers are not identifiable from output alone without
  extra assumptions.
- A correct deep lane needs to distinguish equivalent surface recovery from
  exact hidden-layer recovery.

Proposed architecture:

- Start with equivalent-function recovery, not exact original weights.
- Recover piecewise-linear regions and adjacency graph.
- Search for compact factorizations of the recovered surface into layered ReLU
  circuits.
- Add an activation-access control lane for validating factorization when
  hidden layers are available.

First gates:

- Command: `npm run benchmark:nn-deep-blackbox-relu1d-decompile`
- Teacher: fixed two-layer ReLU network over 1D.
- GCL access: output-only equivalent surface recovery.
- Required: held-out reconstruction accuracy, recovered canonical kink count
  greater than the first-layer kink count, at least 5 deep-only kinks, and
  a GCL MAE advantage over both a linear baseline and a first-layer-kink-only
  baseline.
- Command: `npm run benchmark:nn-deep-blackbox-relu2d-region-graph`
- Teacher: fixed two-layer ReLU network over 2D.
- GCL access: output-only equivalent surface recovery.
- Required: held-out reconstruction accuracy, recovered affine-region
  adjacency graph, region count greater than the first-layer structure, and
  a GCL MAE advantage over both plane and first-layer-only ReLU baselines.

Boundary:

- Do not call this "exact deep reverse engineering" unless hidden activations
  or strict identifiability assumptions are part of the gate.
