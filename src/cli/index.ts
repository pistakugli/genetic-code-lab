#!/usr/bin/env node

/**
 * Genetic Code Lab - CLI Interface
 */

import { Command } from 'commander';
import chalk from 'chalk';
import ora from 'ora';
import fs from 'fs-extra';
import path from 'path';
import { spawnSync } from 'child_process';

import { EvolutionEngine, EvolutionResult } from '../core/evolution';
import { EvolutionEngineTree } from '../core/evolution-tree';
import {
  createMathProblem,
  executeMathGenome,
  genomeToString
} from '../solvers/math-expression';
import { createSortingProblem, genomeToSortingNetwork } from '../solvers/sorting-network';
import {
  createPathfindingProblem,
  extractMoves,
  executePathfinding,
  visualizePath,
  PathfindingTestCase
} from '../solvers/pathfinding';
import { EvolutionConfig, PopulationConfig } from '../core/types';
import { formatTime } from '../utils/helpers';
import { compileMathExpression } from '../utils/math-expression-parser';

type BuiltInProblemType = 'math' | 'sort' | 'path';
type Spinner = ReturnType<typeof ora>;
type CapabilityStatus = 'production' | 'preview' | 'diagnostic';

type CapabilityCommand = {
  label: string;
  command: string;
  args: string[];
};

type Capability = {
  id: string;
  name: string;
  status: CapabilityStatus;
  summary: string;
  scope: string;
  run: CapabilityCommand;
  proofs: CapabilityCommand[];
  artifacts: string[];
};

type CliRunConfig = {
  problem?: {
    type?: BuiltInProblemType;
    target?: string;
    variableCount?: number;
    testCount?: number;
    size?: number;
    width?: number;
    height?: number;
  };
  evolution?: Partial<EvolutionConfig> & {
    generations?: number;
    populationSize?: number;
    eliteCount?: number;
    crossoverRate?: number;
    mutationRate?: number;
    maxGenomeDepth?: number;
    maxGenomeSize?: number;
    tournamentSize?: number;
  };
  output?: string;
};

type ResultMetadata = {
  command: string;
  problemType: BuiltInProblemType;
  problem: Record<string, unknown>;
  config: Record<string, unknown>;
  solution: Record<string, unknown>;
};

type MathRunOptions = {
  command: string;
  spinner: Spinner;
  output?: string;
  target: string;
  variableCount: number;
  testCount: number;
  config: Partial<EvolutionConfig>;
  showSampleTable: boolean;
};

type SortRunOptions = {
  command: string;
  spinner: Spinner;
  output?: string;
  size: number;
  config: Partial<EvolutionConfig>;
};

type PathRunOptions = {
  command: string;
  spinner: Spinner;
  output?: string;
  width: number;
  height: number;
  testCount: number;
  config: Partial<EvolutionConfig>;
};

const DEFAULT_MATH_TARGET = 'x*x + 2*x + 1';

const MATH_DEFAULTS = {
  population: {
    size: 100,
    eliteCount: 2,
    crossoverRate: 0.8,
    mutationRate: 0.2,
    maxGenomeDepth: 8,
    maxGenomeSize: 50,
    tournamentSize: 3
  },
  maxGenerations: 100,
  targetFitness: 0.99,
  stagnationLimit: 30
};

const SORT_DEFAULTS = {
  population: {
    size: 200,
    eliteCount: 4,
    crossoverRate: 0.7,
    mutationRate: 0.3,
    maxGenomeDepth: 6,
    maxGenomeSize: 16,
    tournamentSize: 5
  },
  maxGenerations: 200,
  targetFitness: 0.95,
  stagnationLimit: 40
};

const PATH_DEFAULTS = {
  population: {
    size: 150,
    eliteCount: 3,
    crossoverRate: 0.75,
    mutationRate: 0.25,
    maxGenomeDepth: 10,
    maxGenomeSize: 50,
    tournamentSize: 4
  },
  maxGenerations: 150,
  targetFitness: 0.9,
  stagnationLimit: 30
};

const CAPABILITIES: Capability[] = [
  {
    id: 'threshold-adaptive-duel',
    name: 'Threshold adaptive-duel compose',
    status: 'production',
    summary: 'Default threshold compose policy that duels upper-boundary strategies against a paired-witness incumbent and promotes guarded repairs when they pass audit checks.',
    scope: 'Majority-style monotone threshold runs with n >= 9 and default weighted selection.',
    run: { label: 'default threshold policy invariant', command: 'npm', args: ['run', 'verify:default-threshold-policy'] },
    proofs: [
      { label: 'default threshold policy invariant', command: 'npm', args: ['run', 'verify:default-threshold-policy'] },
      { label: 'adaptive-duel positive-use-case proof', command: 'npm', args: ['run', 'release:promotion'] }
    ],
    artifacts: ['/tmp/gcl-promotion-gate/promotion-calibration.json', '/tmp/gcl-promotion-gate/promotion-quality.json']
  },
  {
    id: 'semantic-archive',
    name: 'Semantic archive composition',
    status: 'production',
    summary: 'Stores behavior-space elites and synthesizes composite candidates from multiple useful subexpressions.',
    scope: 'Symbolic regression archive-composition gauntlet with train-only composition checks.',
    run: { label: 'semantic archive benchmark', command: 'npm', args: ['run', 'benchmark:semantic-archive'] },
    proofs: [
      { label: 'semantic archive benchmark', command: 'npm', args: ['run', 'benchmark:semantic-archive'] }
    ],
    artifacts: [
      'benchmarks/release/semantic-archive-composition/positive-use-case/report.json',
      'benchmarks/release/semantic-archive-composition/positive-use-case/report.md'
    ]
  },
  {
    id: 'archive-emergence',
    name: 'Closed-loop archive emergence',
    status: 'production',
    summary: 'Runs the real engine with archive-off versus archive-on to prove a curated closed-loop archive positive-use-case.',
    scope: 'Curated deterministic symbolic-regression seeds used by the release gate; this is not a universal archive superiority claim.',
    run: { label: 'closed-loop archive emergence benchmark', command: 'npm', args: ['run', 'benchmark:archive-emergence'] },
    proofs: [
      { label: 'closed-loop archive emergence benchmark', command: 'npm', args: ['run', 'benchmark:archive-emergence'] }
    ],
    artifacts: [
      'benchmarks/release/archive-emergence/positive-use-case/report.json',
      'benchmarks/release/archive-emergence/positive-use-case/report.md'
    ]
  },
  {
    id: 'distributed-island-nn',
    name: 'Distributed worker-island NN policy',
    status: 'production',
    summary: 'Runs worker-thread islands through a migration hub and uses an MLP policy with robust validation/guard fallback for final candidate selection.',
    scope: '24-seed release sweep with timeout/error accounting and non-regression checks against the single baseline. Historical 24-seed proof exists; verify regenerates the isolated wide report.',
    run: { label: 'distributed worker-island NN wide benchmark', command: 'npm', args: ['run', 'benchmark:distributed-island-nn:wide'] },
    proofs: [
      { label: 'distributed worker-island NN wide benchmark', command: 'npm', args: ['run', 'benchmark:distributed-island-nn:wide'] }
    ],
    artifacts: [
      'benchmarks/release/distributed-island-nn/wide/report.json',
      'benchmarks/release/distributed-island-nn/wide/report.md'
    ]
  },
  {
    id: 'nn-symbolic-distillation',
    name: 'NN symbolic distillation',
    status: 'production',
    summary: 'Fits a transparent symbolic residual ensemble to input-output samples from a fixed tanh MLP teacher and gates the result against a linear baseline.',
    scope: 'Controlled 3-input/3-hidden neural teacher distillation benchmark; proves transparent input-output surrogate behavior, not exact weight recovery.',
    run: { label: 'NN symbolic distillation benchmark', command: 'npm', args: ['run', 'benchmark:nn-distill'] },
    proofs: [
      { label: 'NN symbolic distillation benchmark', command: 'npm', args: ['run', 'benchmark:nn-distill'] }
    ],
    artifacts: [
      'benchmarks/release/nn-symbolic-distillation/smoke/report.json',
      'benchmarks/release/nn-symbolic-distillation/smoke/report.md'
    ]
  },
  {
    id: 'nn-circuit-distillation',
    name: 'NN circuit distillation',
    status: 'production',
    summary: 'Recovers symbolic formulas for hidden tanh activations and reconstructs the output readout as a transparent circuit.',
    scope: 'Activation-access neural teacher benchmark; proves circuit-level symbolic recovery for tanh-linear neurons, not arbitrary black-box internals without activation access.',
    run: { label: 'NN circuit distillation benchmark', command: 'npm', args: ['run', 'benchmark:nn-circuit-distill'] },
    proofs: [
      { label: 'NN circuit distillation benchmark', command: 'npm', args: ['run', 'benchmark:nn-circuit-distill'] }
    ],
    artifacts: [
      'benchmarks/release/nn-circuit-distillation/smoke/report.json',
      'benchmarks/release/nn-circuit-distillation/smoke/report.md'
    ]
  },
  {
    id: 'nn-relu-circuit-distillation',
    name: 'NN ReLU circuit distillation',
    status: 'production',
    summary: 'Recovers symbolic formulas for hidden ReLU activations and reconstructs the output readout as a transparent piecewise-linear circuit.',
    scope: 'Activation-access neural teacher benchmark; proves circuit-level symbolic recovery for ReLU-linear neurons, not arbitrary black-box internals without activation access.',
    run: { label: 'NN ReLU circuit distillation benchmark', command: 'npm', args: ['run', 'benchmark:nn-relu-circuit-distill'] },
    proofs: [
      { label: 'NN ReLU circuit distillation benchmark', command: 'npm', args: ['run', 'benchmark:nn-relu-circuit-distill'] }
    ],
    artifacts: [
      'benchmarks/release/nn-relu-circuit-distillation/smoke/report.json',
      'benchmarks/release/nn-relu-circuit-distillation/smoke/report.md'
    ]
  },
  {
    id: 'nn-blackbox-relu-decompile',
    name: 'NN black-box ReLU decompile',
    status: 'production',
    summary: 'Reconstructs an equivalent symbolic piecewise-ReLU circuit from oracle calls to a black-box 1D ReLU teacher.',
    scope: 'Black-box neural teacher benchmark; proves 1D ReLU kink recovery without hidden activations or teacher weights, not arbitrary high-dimensional network decompilation.',
    run: { label: 'NN black-box ReLU decompile benchmark', command: 'npm', args: ['run', 'benchmark:nn-blackbox-relu-decompile'] },
    proofs: [
      { label: 'NN black-box ReLU decompile benchmark', command: 'npm', args: ['run', 'benchmark:nn-blackbox-relu-decompile'] }
    ],
    artifacts: [
      'benchmarks/release/nn-blackbox-relu-decompile/smoke/report.json',
      'benchmarks/release/nn-blackbox-relu-decompile/smoke/report.md'
    ]
  },
  {
    id: 'nn-deep-blackbox-relu1d-decompile',
    name: 'NN deep black-box ReLU 1D decompile',
    status: 'production',
    summary: 'Recovers an output-equivalent canonical piecewise-ReLU spline from oracle calls to a folded two-layer 1D ReLU black box.',
    scope: 'Black-box deep ReLU teacher benchmark; proves equivalent-function recovery for a controlled 1-input/5-hidden/5-hidden ReLU MLP, not unique original hidden-layer weights or arbitrary deep ND extraction.',
    run: { label: 'NN deep black-box ReLU 1D decompile benchmark', command: 'npm', args: ['run', 'benchmark:nn-deep-blackbox-relu1d-decompile'] },
    proofs: [
      { label: 'NN deep black-box ReLU 1D decompile benchmark', command: 'npm', args: ['run', 'benchmark:nn-deep-blackbox-relu1d-decompile'] }
    ],
    artifacts: [
      'benchmarks/release/nn-deep-blackbox-relu1d-decompile/folded-two-layer/report.json',
      'benchmarks/release/nn-deep-blackbox-relu1d-decompile/folded-two-layer/report.md'
    ]
  },
  {
    id: 'nn-deep-blackbox-relu2d-region-graph',
    name: 'NN deep black-box ReLU 2D region graph',
    status: 'production',
    summary: 'Recovers an output-equivalent affine-region adjacency graph plus compact, adaptive, and topology-preserving boundary geometry from fixed, randomized, noisy, and adversarial folded two-layer 2D ReLU black boxes.',
    scope: 'Black-box deep ReLU teacher benchmarks; prove equivalent-function region-graph recovery for controlled 2-input/4-hidden/4-hidden ReLU MLPs, including deterministic randomized, noisy/quantized, adaptive adversarial seed sweeps, compact boundary segment extraction, budgeted adaptive grid selection, boundary-aware local switching on lower-cost grids, oracle-guided subcell validation that records rejected too-coarse grids, and line-clipped contact-corridor topology geometry with no fallback segments. This does not claim unique original hidden-layer weights, exact layer attribution, arbitrary deep ND extraction, topology-perfect true-region merging, or production API extraction beyond the benchmark noise model.',
    run: { label: 'NN deep black-box ReLU 2D region-graph benchmark', command: 'npm', args: ['run', 'benchmark:nn-deep-blackbox-relu2d-region-graph'] },
    proofs: [
      { label: 'NN deep black-box ReLU 2D region-graph benchmark', command: 'npm', args: ['run', 'benchmark:nn-deep-blackbox-relu2d-region-graph'] },
      { label: 'NN randomized deep black-box ReLU 2D region-graph benchmark', command: 'npm', args: ['run', 'benchmark:nn-deep-blackbox-relu2d-region-graph:randomized'] },
      { label: 'NN noisy deep black-box ReLU 2D region-graph benchmark', command: 'npm', args: ['run', 'benchmark:nn-noisy-deep-blackbox-relu2d-region-graph'] },
      { label: 'NN adaptive-adversarial deep black-box ReLU 2D region-graph benchmark', command: 'npm', args: ['run', 'benchmark:nn-adaptive-adversarial-deep-blackbox-relu2d-region-graph'] },
      { label: 'NN compact deep black-box ReLU 2D boundary benchmark', command: 'npm', args: ['run', 'benchmark:nn-compact-deep-blackbox-relu2d-boundary'] },
      { label: 'NN adaptive-budget deep black-box ReLU 2D region-graph benchmark', command: 'npm', args: ['run', 'benchmark:nn-adaptive-budget-deep-blackbox-relu2d-region-graph'] },
      { label: 'NN boundary-refined deep black-box ReLU 2D region-graph benchmark', command: 'npm', args: ['run', 'benchmark:nn-boundary-refined-deep-blackbox-relu2d-region-graph'] },
      { label: 'NN subcell-boundary adaptive deep black-box ReLU 2D region-graph benchmark', command: 'npm', args: ['run', 'benchmark:nn-subcell-boundary-adaptive-deep-blackbox-relu2d-region-graph'] },
      { label: 'NN topology-boundary deep black-box ReLU 2D region-graph benchmark', command: 'npm', args: ['run', 'benchmark:nn-topology-boundary-deep-blackbox-relu2d-region-graph'] }
    ],
    artifacts: [
      'benchmarks/release/nn-deep-blackbox-relu2d-region-graph/folded-two-layer/report.json',
      'benchmarks/release/nn-deep-blackbox-relu2d-region-graph/folded-two-layer/report.md',
      'benchmarks/release/nn-deep-blackbox-relu2d-region-graph-randomized/seed-sweep/report.json',
      'benchmarks/release/nn-deep-blackbox-relu2d-region-graph-randomized/seed-sweep/report.md',
      'benchmarks/release/nn-noisy-deep-blackbox-relu2d-region-graph/seed-sweep/report.json',
      'benchmarks/release/nn-noisy-deep-blackbox-relu2d-region-graph/seed-sweep/report.md',
      'benchmarks/release/nn-adaptive-adversarial-deep-blackbox-relu2d-region-graph/adaptive-search/report.json',
      'benchmarks/release/nn-adaptive-adversarial-deep-blackbox-relu2d-region-graph/adaptive-search/report.md',
      'benchmarks/release/nn-compact-deep-blackbox-relu2d-boundary/fixed-plus-randomized/report.json',
      'benchmarks/release/nn-compact-deep-blackbox-relu2d-boundary/fixed-plus-randomized/report.md',
      'benchmarks/release/nn-adaptive-budget-deep-blackbox-relu2d-region-graph/fixed-plus-randomized/report.json',
      'benchmarks/release/nn-adaptive-budget-deep-blackbox-relu2d-region-graph/fixed-plus-randomized/report.md',
      'benchmarks/release/nn-boundary-refined-deep-blackbox-relu2d-region-graph/fixed-plus-randomized/report.json',
      'benchmarks/release/nn-boundary-refined-deep-blackbox-relu2d-region-graph/fixed-plus-randomized/report.md',
      'benchmarks/release/nn-subcell-boundary-adaptive-deep-blackbox-relu2d-region-graph/fixed-plus-randomized/report.json',
      'benchmarks/release/nn-subcell-boundary-adaptive-deep-blackbox-relu2d-region-graph/fixed-plus-randomized/report.md',
      'benchmarks/release/nn-topology-boundary-deep-blackbox-relu2d-region-graph/fixed-plus-randomized/report.json',
      'benchmarks/release/nn-topology-boundary-deep-blackbox-relu2d-region-graph/fixed-plus-randomized/report.md'
    ]
  },
  {
    id: 'nn-deep-blackbox-relu-nd-region-sampler',
    name: 'NN deep black-box ReLU ND region sampler',
    status: 'production',
    summary: 'Recovers output-equivalent affine-region samplers for controlled 3D/4D folded two-layer ReLU black boxes using random local-gradient oracle probes and query-free kNN region assignment.',
    scope: 'Black-box deep ReLU teacher benchmark; proves benchmark-scoped equivalent-function recovery for controlled 3D/4D folded ReLU MLPs with query-budget accounting and a global-linear baseline gap. This does not claim unique hidden-layer weights, exact layer attribution, topology-perfect region merging, noisy ND deep recovery, or arbitrary production API extraction.',
    run: { label: 'NN deep black-box ReLU ND region-sampler benchmark', command: 'npm', args: ['run', 'benchmark:nn-deep-blackbox-relu-nd-region-sampler'] },
    proofs: [
      { label: 'NN deep black-box ReLU ND region-sampler benchmark', command: 'npm', args: ['run', 'benchmark:nn-deep-blackbox-relu-nd-region-sampler'] }
    ],
    artifacts: [
      'benchmarks/release/nn-deep-blackbox-relu-nd-region-sampler/3d-4d-seed-sweep/report.json',
      'benchmarks/release/nn-deep-blackbox-relu-nd-region-sampler/3d-4d-seed-sweep/report.md'
    ]
  },
  {
    id: 'nn-noisy-adaptive-adversarial-deep-blackbox-relu-nd-region-sampler',
    name: 'NN noisy adaptive-adversarial deep black-box ReLU ND region sampler',
    status: 'production',
    summary: 'Adapts between region-vote and local-affine KNN prediction to recover output-equivalent 3D/4D folded two-layer ReLU behavior from noisy quantized oracle calls.',
    scope: 'Black-box deep ReLU teacher benchmark; proves benchmark-scoped noisy/adaptive output-equivalent recovery on generated near-parallel, cancellation, low-amplitude, and mixed-oblique 3D/4D folded teachers. This does not claim unique hidden-layer weights, exact layer attribution, topology-perfect ND region adjacency, arbitrary production API extraction, or superiority over all model-extraction algorithms.',
    run: { label: 'NN noisy adaptive-adversarial deep black-box ReLU ND region-sampler benchmark', command: 'npm', args: ['run', 'benchmark:nn-noisy-adaptive-adversarial-deep-blackbox-relu-nd-region-sampler'] },
    proofs: [
      { label: 'NN noisy adaptive-adversarial deep black-box ReLU ND region-sampler benchmark', command: 'npm', args: ['run', 'benchmark:nn-noisy-adaptive-adversarial-deep-blackbox-relu-nd-region-sampler'] }
    ],
    artifacts: [
      'benchmarks/release/nn-noisy-adaptive-adversarial-deep-blackbox-relu-nd-region-sampler/3d-4d-adaptive-search/report.json',
      'benchmarks/release/nn-noisy-adaptive-adversarial-deep-blackbox-relu-nd-region-sampler/3d-4d-adaptive-search/report.md'
    ]
  },
  {
    id: 'nn-adaptive-budget-noisy-deep-blackbox-relu-nd-region-sampler',
    name: 'NN adaptive-budget noisy deep black-box ReLU ND region sampler',
    status: 'production',
    summary: 'Uses noisy held-out promotion to select the lowest sufficient region-sampler budget for 3D/4D folded ReLU black boxes while recording rejected candidates.',
    scope: 'Black-box deep ReLU teacher benchmark; proves benchmark-scoped query-budget reduction for noisy/adaptive 3D/4D folded recovery with explicit failed-candidate accounting. This does not claim globally optimal query complexity, unique hidden-layer weights, exact layer attribution, topology-perfect ND region adjacency, or arbitrary production API extraction.',
    run: { label: 'NN adaptive-budget noisy deep black-box ReLU ND region-sampler benchmark', command: 'npm', args: ['run', 'benchmark:nn-adaptive-budget-noisy-deep-blackbox-relu-nd-region-sampler'] },
    proofs: [
      { label: 'NN adaptive-budget noisy deep black-box ReLU ND region-sampler benchmark', command: 'npm', args: ['run', 'benchmark:nn-adaptive-budget-noisy-deep-blackbox-relu-nd-region-sampler'] }
    ],
    artifacts: [
      'benchmarks/release/nn-adaptive-budget-noisy-deep-blackbox-relu-nd-region-sampler/3d-4d-budget-search/report.json',
      'benchmarks/release/nn-adaptive-budget-noisy-deep-blackbox-relu-nd-region-sampler/3d-4d-budget-search/report.md'
    ]
  },
  {
    id: 'nn-nd-adjacency-geometry-deep-blackbox-relu-region-sampler',
    name: 'NN ND adjacency-geometry deep black-box ReLU region sampler',
    status: 'production',
    summary: 'Builds recovered ND region-adjacency geometry from affine-region sampler assignments using cross-region contact witnesses and affine-equality hyperplanes.',
    scope: 'Black-box deep ReLU teacher benchmark; proves benchmark-scoped recovered ND adjacency geometry for clean and noisy 3D/4D folded ReLU region samplers. This does not claim topology-perfect full cell-complex recovery, unique hidden-layer weights, exact layer attribution, arbitrary production API extraction, or globally optimal query complexity.',
    run: { label: 'NN ND adjacency-geometry deep black-box ReLU region-sampler benchmark', command: 'npm', args: ['run', 'benchmark:nn-nd-adjacency-geometry-deep-blackbox-relu-region-sampler'] },
    proofs: [
      { label: 'NN ND adjacency-geometry deep black-box ReLU region-sampler benchmark', command: 'npm', args: ['run', 'benchmark:nn-nd-adjacency-geometry-deep-blackbox-relu-region-sampler'] }
    ],
    artifacts: [
      'benchmarks/release/nn-nd-adjacency-geometry-deep-blackbox-relu-region-sampler/3d-4d-clean-noisy/report.json',
      'benchmarks/release/nn-nd-adjacency-geometry-deep-blackbox-relu-region-sampler/3d-4d-clean-noisy/report.md'
    ]
  },
  {
    id: 'nn-nd-adjacency-geometry-5d-sweep',
    name: 'NN ND adjacency-geometry 5D sweep',
    status: 'production',
    summary: 'Extends recovered ND adjacency geometry to a clean/noisy 5D folded-ReLU sweep with explicit region-count and query-budget gates.',
    scope: 'Black-box deep ReLU teacher benchmark; proves benchmark-scoped recovered 5D adjacency geometry across clean, low-amplitude, cancellation, near-parallel, and weak-support folded teachers. This does not claim 6D readiness, topology-perfect full cell-complex recovery, unique hidden-layer weights, exact layer attribution, arbitrary production API extraction, or superiority over all model-extraction algorithms.',
    run: { label: 'NN ND adjacency-geometry 5D sweep benchmark', command: 'npm', args: ['run', 'benchmark:nn-nd-adjacency-geometry-5d-sweep'] },
    proofs: [
      { label: 'NN ND adjacency-geometry 5D sweep benchmark', command: 'npm', args: ['run', 'benchmark:nn-nd-adjacency-geometry-5d-sweep'] }
    ],
    artifacts: [
      'benchmarks/release/nn-nd-adjacency-geometry-5d-sweep/clean-noisy-hard-profiles/report.json',
      'benchmarks/release/nn-nd-adjacency-geometry-5d-sweep/clean-noisy-hard-profiles/report.md'
    ]
  },
  {
    id: 'nn-nd-adjacency-geometry-6d-robustness',
    name: 'NN ND adjacency-geometry 6D robustness',
    status: 'production',
    summary: 'Extends recovered ND adjacency geometry to a 12-teacher 6D clean/noisy randomized hard-profile gate with zero high-tier selections and explicit query-budget accounting.',
    scope: 'Black-box deep ReLU teacher benchmark; proves benchmark-scoped recovered 6D adjacency geometry across clean, low-amplitude, cancellation, near-parallel, and weak-support folded teachers. This does not claim broad SOTA status, topology-perfect full cell-complex recovery, unique hidden-layer weights, exact layer attribution, GELU/SwiGLU recovery, arbitrary production API extraction, or superiority over all extraction algorithms.',
    run: { label: 'NN ND adjacency-geometry 6D robustness benchmark', command: 'npm', args: ['run', 'benchmark:nn-nd-adjacency-geometry-6d-robustness'] },
    proofs: [
      { label: 'NN ND adjacency-geometry 6D robustness benchmark', command: 'npm', args: ['run', 'benchmark:nn-nd-adjacency-geometry-6d-robustness'] }
    ],
    artifacts: [
      'benchmarks/release/nn-nd-adjacency-geometry-6d-robustness/randomized-hard-profiles/report.json',
      'benchmarks/release/nn-nd-adjacency-geometry-6d-robustness/randomized-hard-profiles/report.md'
    ]
  },
  {
    id: 'nn-nd-adjacency-geometry-6d-baseline-court',
    name: 'NN ND adjacency-geometry 6D baseline court',
    status: 'production',
    summary: 'Compares the 12-teacher 6D robustness manifest against dependency-free baseline families and fails promotion if a measured baseline beats GCL by MAE.',
    scope: 'Black-box deep ReLU teacher benchmark; proves benchmark-scoped recovered 6D adjacency geometry beats the measured linear ridge, quadratic ridge, random ReLU ridge, local-affine kNN, and ExtraTrees-style dependency-free baselines on the exact robustness manifest. This does not claim superiority over all external packages, topology-perfect full cell-complex recovery, unique hidden-layer weights, exact layer attribution, GELU/SwiGLU recovery, or arbitrary production API extraction.',
    run: { label: 'NN ND adjacency-geometry 6D baseline court benchmark', command: 'npm', args: ['run', 'benchmark:nn-nd-adjacency-geometry-6d-baselines'] },
    proofs: [
      { label: 'NN ND adjacency-geometry 6D baseline court benchmark', command: 'npm', args: ['run', 'benchmark:nn-nd-adjacency-geometry-6d-baselines'] }
    ],
    artifacts: [
      'benchmarks/release/nn-nd-adjacency-geometry-6d-baselines/robustness-manifest-baseline-court/report.json',
      'benchmarks/release/nn-nd-adjacency-geometry-6d-baselines/robustness-manifest-baseline-court/report.md'
    ]
  },
  {
    id: 'nn-active-subspace-ffn-decompiler',
    name: 'NN active-subspace FFN decompiler',
    status: 'production',
    summary: 'Discovers low-dimensional active subspaces in controlled 256D/1024D GELU and SwiGLU-style black-box FFN channels, then fits an output-equivalent smooth surrogate with query-budget and baseline-court accounting.',
    scope: 'Black-box smooth activation benchmark; proves benchmark-scoped active-subspace discovery and faithful output surrogate recovery for controlled high-dimensional single-channel GELU and two-direction SwiGLU-style teachers. This does not claim real LLM decompilation, arbitrary 4096D production model extraction, exact GELU/SwiGLU parameter recovery, tokenizer/model integration, or broad SOTA status.',
    run: { label: 'NN active-subspace FFN decompiler benchmark', command: 'npm', args: ['run', 'benchmark:nn-active-subspace-ffn-decompiler'] },
    proofs: [
      { label: 'NN active-subspace FFN decompiler benchmark', command: 'npm', args: ['run', 'benchmark:nn-active-subspace-ffn-decompiler'] }
    ],
    artifacts: [
      'benchmarks/release/nn-active-subspace-ffn-decompiler/gelu-swiglu-256d-1024d/report.json',
      'benchmarks/release/nn-active-subspace-ffn-decompiler/gelu-swiglu-256d-1024d/report.md'
    ]
  },
  {
    id: 'nn-active-subspace-ffn-4096d-stress',
    name: 'NN active-subspace FFN 4096D stress',
    status: 'production',
    summary: 'Stress-tests the active-subspace FFN decompiler at 4096 input dimensions on controlled GELU and SwiGLU-style black-box channels with explicit query-budget and baseline-court gates.',
    scope: 'Black-box smooth activation benchmark; proves benchmark-scoped active-subspace discovery and faithful output surrogate recovery for controlled 4096D single-channel GELU and two-direction SwiGLU-style teachers. This does not claim real LLM decompilation, production transformer hook extraction, exact GELU/SwiGLU parameter recovery, multi-channel FFN block recovery, tokenizer/model integration, or broad SOTA status.',
    run: { label: 'NN active-subspace FFN 4096D stress benchmark', command: 'npm', args: ['run', 'benchmark:nn-active-subspace-ffn-4096d-stress'] },
    proofs: [
      { label: 'NN active-subspace FFN 4096D stress benchmark', command: 'npm', args: ['run', 'benchmark:nn-active-subspace-ffn-4096d-stress'] }
    ],
    artifacts: [
      'benchmarks/release/nn-active-subspace-ffn-4096d-stress/gelu-swiglu-single-and-rank2/report.json',
      'benchmarks/release/nn-active-subspace-ffn-4096d-stress/gelu-swiglu-single-and-rank2/report.md'
    ]
  },
  {
    id: 'nn-active-subspace-ffn-noisy',
    name: 'NN noisy active-subspace FFN',
    status: 'production',
    summary: 'Recovers rank-1 GELU active directions and output-faithful surrogates from stochastic noisy high-dimensional oracle calls, with held-out noiseless validation and baseline-court accounting.',
    scope: 'Black-box smooth activation benchmark; proves benchmark-scoped noisy active-subspace recovery for controlled 128D/256D rank-1 GELU teachers. This does not claim real LLM decompilation, arbitrary stochastic API extraction, exact GELU parameter recovery, mixed multi-channel FFN block recovery, or broad SOTA status.',
    run: { label: 'NN noisy active-subspace FFN benchmark', command: 'npm', args: ['run', 'benchmark:nn-active-subspace-ffn-noisy'] },
    proofs: [
      { label: 'NN noisy active-subspace FFN benchmark', command: 'npm', args: ['run', 'benchmark:nn-active-subspace-ffn-noisy'] }
    ],
    artifacts: [
      'benchmarks/release/nn-active-subspace-ffn-noisy/rank1-gelu-noisy-128d-256d/report.json',
      'benchmarks/release/nn-active-subspace-ffn-noisy/rank1-gelu-noisy-128d-256d/report.md'
    ]
  },
  {
    id: 'nn-gelu-ffn-structure-finder',
    name: 'NN GELU FFN structure finder',
    status: 'production',
    summary: 'Recovers explicit GELU FFN atom structure from oracle calls only: input directions, input biases, output weights, validation metrics, and query budget for controlled high-dimensional channels.',
    scope: 'Black-box smooth activation structure benchmark; proves benchmark-scoped explicit GELU atom recovery for controlled 96D single-channel and two-channel FFN teachers. This is not a local KNN or polynomial surrogate. It does not claim full GPT-2 FFN extraction, arbitrary channel-count recovery, SwiGLU structure recovery, exact identity under non-identifiability, tokenizer/model integration, or broad SOTA status.',
    run: { label: 'NN GELU FFN structure finder benchmark', command: 'npm', args: ['run', 'benchmark:nn-gelu-ffn-structure-finder'] },
    proofs: [
      { label: 'NN GELU FFN structure finder benchmark', command: 'npm', args: ['run', 'benchmark:nn-gelu-ffn-structure-finder'] }
    ],
    artifacts: [
      'benchmarks/release/nn-gelu-ffn-structure-finder/controlled-single-and-two-channel/report.json',
      'benchmarks/release/nn-gelu-ffn-structure-finder/controlled-single-and-two-channel/report.md'
    ]
  },
  {
    id: 'nn-gelu-ffn-mixed-structure',
    name: 'NN rank-adaptive mixed GELU FFN structure',
    status: 'production',
    summary: 'Recovers explicit mixed GELU FFN atoms and effective channel count from oracle calls only when GCL is given a maximum channel budget but not the true channel count.',
    scope: 'Black-box smooth activation structure benchmark; proves benchmark-scoped rank-adaptive mixed GELU atom recovery for a controlled 32D three-channel FFN teacher with an overcomplete max-channel budget. This is not a local KNN or polynomial surrogate. It does not claim full GPT-2 FFN extraction, arbitrary 3072-channel recovery, arbitrary 4096D production LLM block decompilation, SwiGLU mixed-block recovery, tokenizer/model integration, or broad SOTA status.',
    run: { label: 'NN rank-adaptive mixed GELU FFN structure benchmark', command: 'npm', args: ['run', 'benchmark:nn-gelu-ffn-mixed-structure'] },
    proofs: [
      { label: 'NN rank-adaptive mixed GELU FFN structure benchmark', command: 'npm', args: ['run', 'benchmark:nn-gelu-ffn-mixed-structure'] }
    ],
    artifacts: [
      'benchmarks/release/nn-gelu-ffn-mixed-structure/rank-adaptive-three-channel/report.json',
      'benchmarks/release/nn-gelu-ffn-mixed-structure/rank-adaptive-three-channel/report.md'
    ]
  },
  {
    id: 'nn-gelu-ffn-near-parallel-family',
    name: 'NN near-parallel mixed GELU FFN family',
    status: 'production',
    summary: 'Stress-tests mixed GELU structure recovery as two channels become nearly parallel, measuring direction recovery, effective channel count, duplicate diagnostics, near-parallel diagnostics, output fit, and query budget.',
    scope: 'Black-box smooth activation structure benchmark; proves benchmark-scoped stability on a controlled clean 16D three-channel GELU family with epsilons 1e-1, 1e-2, 1e-3, and 1e-4. It does not claim noisy mixed-block recovery, full GPT-2 FFN extraction, arbitrary 3072-channel recovery, arbitrary 4096D production LLM block decompilation, SwiGLU mixed-block recovery, tokenizer/model integration, or broad SOTA status.',
    run: { label: 'NN near-parallel mixed GELU FFN family benchmark', command: 'npm', args: ['run', 'benchmark:nn-gelu-ffn-near-parallel-family'] },
    proofs: [
      { label: 'NN near-parallel mixed GELU FFN family benchmark', command: 'npm', args: ['run', 'benchmark:nn-gelu-ffn-near-parallel-family'] }
    ],
    artifacts: [
      'benchmarks/release/nn-gelu-ffn-near-parallel-family/epsilon-sweep/report.json',
      'benchmarks/release/nn-gelu-ffn-near-parallel-family/epsilon-sweep/report.md'
    ]
  },
  {
    id: 'nn-gelu-ffn-cross-decomposition-invariance',
    name: 'NN GELU FFN cross-decomposition invariance',
    status: 'production',
    summary: 'Checks whether two mathematically equivalent hidden GELU decompositions recover to the same canonical structure from oracle calls only.',
    scope: 'Black-box smooth activation identifiability benchmark; proves benchmark-scoped canonical structure invariance for a clean controlled 16D GELU FFN pair with a minimal representation and an overcomplete equivalent representation. It does not claim semantic identity of original hidden neurons, noisy invariance, full GPT-2 FFN extraction, arbitrary 3072-channel recovery, arbitrary 4096D production LLM block decompilation, SwiGLU mixed-block recovery, tokenizer/model integration, or broad SOTA status.',
    run: { label: 'NN GELU FFN cross-decomposition invariance benchmark', command: 'npm', args: ['run', 'benchmark:nn-gelu-ffn-cross-decomposition-invariance'] },
    proofs: [
      { label: 'NN GELU FFN cross-decomposition invariance benchmark', command: 'npm', args: ['run', 'benchmark:nn-gelu-ffn-cross-decomposition-invariance'] }
    ],
    artifacts: [
      'benchmarks/release/nn-gelu-ffn-cross-decomposition-invariance/equivalent-reparameterizations/report.json',
      'benchmarks/release/nn-gelu-ffn-cross-decomposition-invariance/equivalent-reparameterizations/report.md'
    ]
  },
  {
    id: 'nn-gelu-ffn-negative-cases',
    name: 'NN GELU FFN formalization negative cases',
    status: 'production',
    summary: 'Executes negative-case guardrails for the neural decompiler formalization: duplicate/cancellation canonicalization, weak-under-noise non-identifiability, weak-above-noise exact recovery, activation-family mismatch rejection, and minimum equivalent structure under overcomplete budgets.',
    scope: 'Black-box smooth activation structure benchmark; proves benchmark-scoped guardrails that prevent unsupported structure promotion in selected failure modes. It does not claim semantic identity of original hidden neurons, noisy full mixed-block recovery, full GPT-2 FFN extraction, arbitrary 3072-channel recovery, arbitrary 4096D/16384D production LLM decompilation, or broad SOTA status.',
    run: { label: 'NN GELU FFN formalization negative-case benchmark', command: 'npm', args: ['run', 'benchmark:nn-gelu-ffn-negative-cases'] },
    proofs: [
      { label: 'NN GELU FFN formalization negative-case benchmark', command: 'npm', args: ['run', 'benchmark:nn-gelu-ffn-negative-cases'] }
    ],
    artifacts: [
      'benchmarks/release/nn-gelu-ffn-negative-cases/formalization-negative-cases/report.json',
      'benchmarks/release/nn-gelu-ffn-negative-cases/formalization-negative-cases/report.md'
    ]
  },
  {
    id: 'nn-gelu-ffn-rank-cap-negative',
    name: 'NN GELU FFN rank-cap negative guard',
    status: 'production',
    summary: 'Prevents unsupported full-slice structure promotion when a GELU FFN rank cap is below the observable source-group count.',
    scope: 'Black-box smooth activation negative guard; proves benchmark-scoped refusal of a synthetic 768D/3072-channel full-slice structure claim under an insufficient rank cap. It strengthens the neural decompiler by recording insufficient-rank evidence instead of promoting a weak structure claim. It does not claim full GPT-2 FFN extraction, arbitrary 3072-channel blind decompilation, semantic output-only source identity, or broad SOTA status.',
    run: { label: 'NN GELU FFN rank-cap negative guard benchmark', command: 'npm', args: ['run', 'benchmark:nn-gelu-ffn-rank-cap-negative'] },
    proofs: [
      { label: 'NN GELU FFN rank-cap negative guard benchmark', command: 'npm', args: ['run', 'benchmark:nn-gelu-ffn-rank-cap-negative'] }
    ],
    artifacts: [
      'benchmarks/release/nn-gelu-ffn-rank-cap-negative/synthetic-3072-rank-cap/report.json',
      'benchmarks/release/nn-gelu-ffn-rank-cap-negative/synthetic-3072-rank-cap/report.md'
    ]
  },
  {
    id: 'nn-gelu-ffn-noisy-mixed-structure',
    name: 'NN noisy mixed GELU FFN structure',
    status: 'production',
    summary: 'Recovers explicit mixed GELU atom structure under controlled stochastic output noise and finite-precision quantization, with separate structure and accuracy-court gates.',
    scope: 'Black-box smooth activation structure benchmark; proves benchmark-scoped noisy/quantized mixed GELU recovery for controlled balanced, near-parallel, and weak-but-resolvable three-channel FFN teachers. This is not a local KNN or polynomial surrogate. It does not claim full GPT-2 FFN extraction, arbitrary 3072-channel recovery, arbitrary 4096D/16384D production LLM block decompilation, noisy SwiGLU mixed-block recovery, semantic original-neuron identity from output-only access, or broad SOTA status.',
    run: { label: 'NN noisy mixed GELU FFN structure benchmark', command: 'npm', args: ['run', 'benchmark:nn-gelu-ffn-noisy-mixed-structure'] },
    proofs: [
      { label: 'NN noisy mixed GELU FFN structure benchmark', command: 'npm', args: ['run', 'benchmark:nn-gelu-ffn-noisy-mixed-structure'] }
    ],
    artifacts: [
      'benchmarks/release/nn-gelu-ffn-noisy-mixed-structure/quantized-noisy-family/report.json',
      'benchmarks/release/nn-gelu-ffn-noisy-mixed-structure/quantized-noisy-family/report.md'
    ]
  },
  {
    id: 'nn-swiglu-ffn-mixed-structure',
    name: 'NN mixed SwiGLU FFN structure',
    status: 'production',
    summary: 'Recovers explicit value/gate SwiGLU FFN pairs and effective pair count from oracle calls only when GCL is given a maximum pair budget but not the true pair count.',
    scope: 'Black-box smooth activation structure benchmark; proves benchmark-scoped rank-adaptive mixed SwiGLU value/gate pair recovery for a controlled 24D two-block FFN teacher with an overcomplete max-pair budget. This is not a local KNN or polynomial surrogate. It does not claim noisy SwiGLU recovery, full GPT-2/LLM FFN extraction, arbitrary 3072-channel recovery, arbitrary 4096D production transformer block decompilation, semantic output-only source identity, or broad SOTA status.',
    run: { label: 'NN mixed SwiGLU FFN structure benchmark', command: 'npm', args: ['run', 'benchmark:nn-swiglu-ffn-mixed-structure'] },
    proofs: [
      { label: 'NN mixed SwiGLU FFN structure benchmark', command: 'npm', args: ['run', 'benchmark:nn-swiglu-ffn-mixed-structure'] }
    ],
    artifacts: [
      'benchmarks/release/nn-swiglu-ffn-mixed-structure/rank-adaptive-two-block/report.json',
      'benchmarks/release/nn-swiglu-ffn-mixed-structure/rank-adaptive-two-block/report.md'
    ]
  },
  {
    id: 'nn-swiglu-ffn-noisy-mixed-structure',
    name: 'NN noisy mixed SwiGLU FFN structure',
    status: 'production',
    summary: 'Recovers explicit mixed SwiGLU value/gate pair structure and effective pair counts from repeated noisy/quantized oracle calls across balanced, near-parallel-value, and weak-resolvable profiles.',
    scope: 'Black-box smooth activation structure benchmark; proves benchmark-scoped noisy/quantized mixed SwiGLU value/gate pair recovery for controlled 16D/18D/24D FFN output slices with hidden true pair counts and overcomplete max-pair budgets. This is not a local KNN or polynomial surrogate. Shared-gate/cancellation-heavy canonicalization is covered by the separate nn-swiglu-ffn-shared-gate-canonical gate. It does not claim full GPT-2/LLM FFN extraction, arbitrary 3072-channel recovery, arbitrary 4096D/16384D production transformer block decompilation, exact hidden weights under non-identifiability, semantic output-only source identity, XGBoost/RF/MLP structure-decompiler superiority, or broad SOTA status.',
    run: { label: 'NN noisy mixed SwiGLU FFN structure benchmark', command: 'npm', args: ['run', 'benchmark:nn-swiglu-ffn-noisy-mixed-structure'] },
    proofs: [
      { label: 'NN noisy mixed SwiGLU FFN structure benchmark', command: 'npm', args: ['run', 'benchmark:nn-swiglu-ffn-noisy-mixed-structure'] }
    ],
    artifacts: [
      'benchmarks/release/nn-swiglu-ffn-noisy-mixed-structure/quantized-adversarial-family/report.json',
      'benchmarks/release/nn-swiglu-ffn-noisy-mixed-structure/quantized-adversarial-family/report.md'
    ]
  },
  {
    id: 'nn-swiglu-ffn-shared-gate-canonical',
    name: 'NN shared-gate cancellation SwiGLU canonical',
    status: 'production',
    summary: 'Canonicalizes shared-gate/cancellation-heavy SwiGLU source channels into the minimal functionally equivalent value/gate pair instead of promoting duplicate non-identifiable channels.',
    scope: 'Black-box smooth activation structure benchmark; proves benchmark-scoped shared-gate/cancellation-heavy SwiGLU canonicalization for controlled clean and noisy/quantized FFN output slices. Multiple hidden source value channels share one gate; GCL must recover the canonical aggregate value direction, the shared gate direction, and effective pair count 1 from oracle calls only. This is not a local KNN or polynomial surrogate. It does not claim output-only original source-neuron identity, exact hidden weights under non-identifiability, full GPT-2/LLM FFN extraction, arbitrary 3072-channel recovery, arbitrary 4096D/16384D production transformer block decompilation, or broad SOTA status.',
    run: { label: 'NN shared-gate cancellation SwiGLU canonical benchmark', command: 'npm', args: ['run', 'benchmark:nn-swiglu-ffn-shared-gate-canonical'] },
    proofs: [
      { label: 'NN shared-gate cancellation SwiGLU canonical benchmark', command: 'npm', args: ['run', 'benchmark:nn-swiglu-ffn-shared-gate-canonical'] }
    ],
    artifacts: [
      'benchmarks/release/nn-swiglu-ffn-shared-gate-canonical/canonical-collapse/report.json',
      'benchmarks/release/nn-swiglu-ffn-shared-gate-canonical/canonical-collapse/report.md'
    ]
  },
  {
    id: 'nn-gelu-ffn-provenance-verifier',
    name: 'NN GELU FFN source-neuron provenance verifier',
    status: 'production',
    summary: 'Maps recovered GELU FFN atoms to exact source neuron ids, layers, indices, duplicate source groups, and why-reacts evidence when a source-neuron manifest or equivalent instrumentation is available.',
    scope: 'Provenance-mode smooth activation structure benchmark; proves benchmark-scoped source-neuron identity mapping for controlled distinguishable and duplicate-split GELU FFN source neurons. It does not claim exact source identity from output-only access without source provenance, mathematical uniqueness for arbitrary neural networks, full GPT-2 FFN extraction, arbitrary 3072-channel recovery, tokenizer/model integration, or broad SOTA status.',
    run: { label: 'NN GELU FFN source-neuron provenance verifier benchmark', command: 'npm', args: ['run', 'benchmark:nn-gelu-ffn-provenance-verifier'] },
    proofs: [
      { label: 'NN GELU FFN source-neuron provenance verifier benchmark', command: 'npm', args: ['run', 'benchmark:nn-gelu-ffn-provenance-verifier'] }
    ],
    artifacts: [
      'benchmarks/release/nn-gelu-ffn-provenance-verifier/source-neuron-mapping/report.json',
      'benchmarks/release/nn-gelu-ffn-provenance-verifier/source-neuron-mapping/report.md'
    ]
  },
  {
    id: 'nn-local-transformer-mlp-hook',
    name: 'NN local transformer MLP hook',
    status: 'production',
    summary: 'Recovers explicit GELU source structure from a hooked local transformer MLP output slice and maps recovered atoms back to source expansion neurons with why-reacts evidence.',
    scope: 'Provenance-mode local transformer benchmark; uses captured residual input, MLP input, fc preactivation, GELU activation, per-source contribution, and target output rows from a deterministic local transformer-style block. GCL must recover the explicit GELU source atoms from the hooked MLP-input oracle and verify source ids through the hook/source manifest. This is not full GPT-2/LLM extraction, arbitrary transformer block decompilation, output-only source identity without hooks, or broad SOTA status.',
    run: { label: 'NN local transformer MLP hook benchmark', command: 'npm', args: ['run', 'benchmark:nn-local-transformer-mlp-hook'] },
    proofs: [
      { label: 'NN local transformer MLP hook benchmark', command: 'npm', args: ['run', 'benchmark:nn-local-transformer-mlp-hook'] }
    ],
    artifacts: [
      'benchmarks/release/nn-local-transformer-mlp-hook/hooked-gelu-slice/report.json',
      'benchmarks/release/nn-local-transformer-mlp-hook/hooked-gelu-slice/report.md'
    ]
  },
  {
    id: 'nn-gpt2-gelu-ffn-real-structure',
    name: 'NN real GPT-2 GELU FFN source structure',
    status: 'production',
    summary: 'Recovers explicit GELU atoms from compact real GPT-2 layer-0 FFN top-3/top-5/top-10 output slices using oracle-only recovery, then maps recovered atoms to exact GPT-2 source channel ids through functional provenance.',
    scope: 'Provenance-mode real GPT-2 benchmark; uses compact fixtures extracted from real GPT-2 layer-0 FFN weights for output channel 0 top-3, top-5, and top-10 source channels. During recovery GCL receives only f(vector) oracle calls; the source manifest is used after recovery to verify source ids through direction, activation trace, contribution trace, why-reacts evidence, and query budget. This is not full GPT-2 FFN extraction, arbitrary 3072-channel recovery, exact parameter identity under GELU gauge drift, tokenizer/model execution, output-only source identity without post-hoc provenance, or broad SOTA status.',
    run: { label: 'NN real GPT-2 GELU FFN source-structure benchmark', command: 'npm', args: ['run', 'benchmark:nn-gpt2-gelu-ffn-real-structure'] },
    proofs: [
      { label: 'NN real GPT-2 GELU FFN source-structure benchmark', command: 'npm', args: ['run', 'benchmark:nn-gpt2-gelu-ffn-real-structure'] },
      { label: 'NN real GPT-2 GELU FFN top-5 blind source-structure benchmark', command: 'npm', args: ['run', 'benchmark:nn-gpt2-gelu-ffn-real-structure:top5'] },
      { label: 'NN real GPT-2 GELU FFN top-10 blind source-structure benchmark', command: 'npm', args: ['run', 'benchmark:nn-gpt2-gelu-ffn-real-structure:top10'] }
    ],
    artifacts: [
      'benchmarks/fixtures/gpt2-layer0-output0-top3-gelu.json',
      'benchmarks/fixtures/gpt2-layer0-output0-top5-gelu.json',
      'benchmarks/fixtures/gpt2-layer0-output0-top8-gelu.json',
      'benchmarks/fixtures/gpt2-layer0-output0-top10-gelu.json',
      'benchmarks/release/nn-gpt2-gelu-ffn-real-structure/layer0-output0-topk-source-mapping/report.json',
      'benchmarks/release/nn-gpt2-gelu-ffn-real-structure/layer0-output0-topk-source-mapping/report.md',
      'benchmarks/release/nn-gpt2-gelu-ffn-real-structure/layer0-output0-top5-blind-source-mapping/report.json',
      'benchmarks/release/nn-gpt2-gelu-ffn-real-structure/layer0-output0-top5-blind-source-mapping/report.md',
      'benchmarks/release/nn-gpt2-gelu-ffn-real-structure/layer0-output0-top8-blind-source-mapping/report.json',
      'benchmarks/release/nn-gpt2-gelu-ffn-real-structure/layer0-output0-top8-blind-source-mapping/report.md',
      'benchmarks/release/nn-gpt2-gelu-ffn-real-structure/layer0-output0-top10-blind-source-mapping/report.json',
      'benchmarks/release/nn-gpt2-gelu-ffn-real-structure/layer0-output0-top10-blind-source-mapping/report.md'
    ]
  },
  {
    id: 'nn-gpt2-gelu-ffn-source-extractor',
    name: 'NN real GPT-2 GELU FFN source extractor',
    status: 'production',
    summary: 'Builds an exact explicit GELU source graph for a real GPT-2 layer-0 FFN top-20 output slice when source weights or equivalent hook manifests are available.',
    scope: 'Instrumented real GPT-2 benchmark; uses a compact fixture extracted from real GPT-2 layer-0 FFN weights for output channel 0 top-20 source channels. GCL builds the explicit GELU source graph directly from source provenance, validates exact held-out outputs across IID and OOD activation distributions, and verifies every atom against exact GPT-2 channel ids with strict parameter provenance. This is not output-only black-box recovery, full 3072-channel GPT-2 FFN extraction, tokenizer/model execution, or broad SOTA status.',
    run: { label: 'NN real GPT-2 GELU FFN source-extractor benchmark', command: 'npm', args: ['run', 'benchmark:nn-gpt2-gelu-ffn-source-extractor'] },
    proofs: [
      { label: 'NN real GPT-2 GELU FFN source-extractor benchmark', command: 'npm', args: ['run', 'benchmark:nn-gpt2-gelu-ffn-source-extractor'] }
    ],
    artifacts: [
      'benchmarks/fixtures/gpt2-layer0-output0-top20-gelu.json',
      'benchmarks/release/nn-gpt2-gelu-ffn-source-extractor/layer0-output0-top20-exact-source-graph/report.json',
      'benchmarks/release/nn-gpt2-gelu-ffn-source-extractor/layer0-output0-top20-exact-source-graph/report.md'
    ]
  },
  {
    id: 'nn-blackbox-relu2d-decompile',
    name: 'NN 2D black-box ReLU decompile',
    status: 'production',
    summary: 'Reconstructs an equivalent symbolic piecewise-ReLU surface from oracle calls to a 2D black-box ReLU teacher.',
    scope: 'Black-box neural teacher benchmark; proves 2D non-vertical ReLU kink-line recovery from f(x,y) probes only, not arbitrary high-dimensional network decompilation.',
    run: { label: 'NN 2D black-box ReLU decompile benchmark', command: 'npm', args: ['run', 'benchmark:nn-blackbox-relu2d-decompile'] },
    proofs: [
      { label: 'NN 2D black-box ReLU decompile benchmark', command: 'npm', args: ['run', 'benchmark:nn-blackbox-relu2d-decompile'] }
    ],
    artifacts: [
      'benchmarks/release/nn-blackbox-relu2d-decompile/smoke/report.json',
      'benchmarks/release/nn-blackbox-relu2d-decompile/smoke/report.md'
    ]
  },
  {
    id: 'nn-blackbox-relu3d-decompile',
    name: 'NN 3D black-box ReLU decompile',
    status: 'production',
    summary: 'Reconstructs equivalent symbolic 3D ReLU hyperplanes from oracle calls to a black-box ReLU teacher.',
    scope: 'Black-box neural teacher benchmark; proves 3D non-vertical ReLU hyperplane recovery from f(x,y,z) probes only, not arbitrary high-dimensional or deep network decompilation.',
    run: { label: 'NN 3D black-box ReLU decompile benchmark', command: 'npm', args: ['run', 'benchmark:nn-blackbox-relu3d-decompile'] },
    proofs: [
      { label: 'NN 3D black-box ReLU decompile benchmark', command: 'npm', args: ['run', 'benchmark:nn-blackbox-relu3d-decompile'] }
    ],
    artifacts: [
      'benchmarks/release/nn-blackbox-relu3d-decompile/smoke/report.json',
      'benchmarks/release/nn-blackbox-relu3d-decompile/smoke/report.md'
    ]
  },
  {
    id: 'nn-blackbox-relu3d-randomized',
    name: 'NN randomized 3D black-box ReLU decompile',
    status: 'production',
    summary: 'Reconstructs symbolic 3D ReLU hyperplanes across deterministic randomized black-box teachers.',
    scope: 'Black-box neural teacher seed-sweep benchmark; reduces fixed-teacher risk for 3D non-vertical hyperplane recovery, not arbitrary ND or deep network decompilation.',
    run: { label: 'NN randomized 3D black-box ReLU decompile benchmark', command: 'npm', args: ['run', 'benchmark:nn-blackbox-relu3d-randomized'] },
    proofs: [
      { label: 'NN randomized 3D black-box ReLU decompile benchmark', command: 'npm', args: ['run', 'benchmark:nn-blackbox-relu3d-randomized'] }
    ],
    artifacts: [
      'benchmarks/release/nn-blackbox-relu3d-randomized/seed-sweep/report.json',
      'benchmarks/release/nn-blackbox-relu3d-randomized/seed-sweep/report.md'
    ]
  },
  {
    id: 'nn-noisy-blackbox-relu-decompile',
    name: 'NN noisy black-box ReLU decompile',
    status: 'production',
    summary: 'Reconstructs a useful symbolic piecewise-ReLU circuit from repeated noisy and quantized oracle calls.',
    scope: 'Noise-tolerant black-box neural teacher benchmark; proves 1D ReLU recovery under stochastic output and finite precision, not arbitrary production API decompilation.',
    run: { label: 'NN noisy black-box ReLU decompile benchmark', command: 'npm', args: ['run', 'benchmark:nn-noisy-blackbox-relu-decompile'] },
    proofs: [
      { label: 'NN noisy black-box ReLU decompile benchmark', command: 'npm', args: ['run', 'benchmark:nn-noisy-blackbox-relu-decompile'] }
    ],
    artifacts: [
      'benchmarks/release/nn-noisy-blackbox-relu-decompile/smoke/report.json',
      'benchmarks/release/nn-noisy-blackbox-relu-decompile/smoke/report.md'
    ]
  },
  {
    id: 'nn-noisy-blackbox-relu2d-decompile',
    name: 'NN noisy 2D black-box ReLU decompile',
    status: 'production',
    summary: 'Reconstructs useful symbolic 2D piecewise-ReLU geometry from repeated noisy and quantized oracle calls.',
    scope: 'Noise-tolerant black-box neural teacher benchmark; proves 2D non-vertical ReLU kink-line recovery under stochastic output and finite precision, not arbitrary high-dimensional network decompilation.',
    run: { label: 'NN noisy 2D black-box ReLU decompile benchmark', command: 'npm', args: ['run', 'benchmark:nn-noisy-blackbox-relu2d-decompile'] },
    proofs: [
      { label: 'NN noisy 2D black-box ReLU decompile benchmark', command: 'npm', args: ['run', 'benchmark:nn-noisy-blackbox-relu2d-decompile'] }
    ],
    artifacts: [
      'benchmarks/release/nn-noisy-blackbox-relu2d-decompile/smoke/report.json',
      'benchmarks/release/nn-noisy-blackbox-relu2d-decompile/smoke/report.md'
    ]
  },
  {
    id: 'nn-noisy-blackbox-relu3d-decompile',
    name: 'NN noisy 3D black-box ReLU decompile',
    status: 'production',
    summary: 'Reconstructs useful symbolic 3D piecewise-ReLU hyperplane geometry from repeated noisy and quantized oracle calls.',
    scope: 'Noise-tolerant black-box neural teacher benchmark; proves 3D non-vertical ReLU hyperplane stitching under stochastic output and finite precision, not arbitrary high-dimensional or deep network decompilation.',
    run: { label: 'NN noisy 3D black-box ReLU decompile benchmark', command: 'npm', args: ['run', 'benchmark:nn-noisy-blackbox-relu3d-decompile'] },
    proofs: [
      { label: 'NN noisy 3D black-box ReLU decompile benchmark', command: 'npm', args: ['run', 'benchmark:nn-noisy-blackbox-relu3d-decompile'] }
    ],
    artifacts: [
      'benchmarks/release/nn-noisy-blackbox-relu3d-decompile/smoke/report.json',
      'benchmarks/release/nn-noisy-blackbox-relu3d-decompile/smoke/report.md'
    ]
  },
  {
    id: 'nn-blackbox-relu-nd-line-probe',
    name: 'NN black-box ReLU ND line probe',
    status: 'production',
    summary: 'Recovers 4D shallow ReLU hyperplanes from random-direction oracle line probes instead of axis-aligned 2D/3D slicing.',
    scope: 'Oracle-only shallow ReLU teacher benchmark; proves the first general-ND random-direction hyperplane stitching lane on a 4D teacher, not noisy ND, deep composition recovery, or arbitrary black-box model extraction.',
    run: { label: 'NN 4D random-direction line-probe ReLU benchmark', command: 'npm', args: ['run', 'benchmark:nn-blackbox-relu-nd-line-probe'] },
    proofs: [
      { label: 'NN 4D random-direction line-probe ReLU benchmark', command: 'npm', args: ['run', 'benchmark:nn-blackbox-relu-nd-line-probe'] }
    ],
    artifacts: [
      'benchmarks/release/nn-blackbox-relu-nd-line-probe/four-dimensional/report.json',
      'benchmarks/release/nn-blackbox-relu-nd-line-probe/four-dimensional/report.md'
    ]
  },
  {
    id: 'nn-blackbox-relu-nd-line-probe-randomized',
    name: 'NN randomized black-box ReLU ND line probe',
    status: 'production',
    summary: 'Runs a deterministic 4D/5D random-direction line-probe seed sweep with query-budget accounting.',
    scope: 'Oracle-only shallow ReLU teacher benchmark; proves clean random-direction hyperplane stitching across 4D and 5D teachers, not noisy ND, deep composition recovery, or arbitrary production model extraction.',
    run: { label: 'NN randomized 4D/5D line-probe ReLU benchmark', command: 'npm', args: ['run', 'benchmark:nn-blackbox-relu-nd-line-probe-randomized'] },
    proofs: [
      { label: 'NN randomized 4D/5D line-probe ReLU benchmark', command: 'npm', args: ['run', 'benchmark:nn-blackbox-relu-nd-line-probe-randomized'] }
    ],
    artifacts: [
      'benchmarks/release/nn-blackbox-relu-nd-line-probe-randomized/4d-5d-seed-sweep/report.json',
      'benchmarks/release/nn-blackbox-relu-nd-line-probe-randomized/4d-5d-seed-sweep/report.md'
    ]
  },
  {
    id: 'nn-noisy-blackbox-relu-nd-line-probe',
    name: 'NN noisy black-box ReLU ND line probe',
    status: 'production',
    summary: 'Runs a deterministic noisy/quantized 4D/5D random-direction line-probe seed sweep with query-budget accounting.',
    scope: 'Noise-tolerant oracle-only shallow ReLU teacher benchmark; proves repeated noisy f(vector) hyperplane stitching across 4D and 5D teachers, not deep composition recovery, adversarial weak-support ND, or arbitrary production model extraction.',
    run: { label: 'NN noisy 4D/5D line-probe ReLU benchmark', command: 'npm', args: ['run', 'benchmark:nn-noisy-blackbox-relu-nd-line-probe'] },
    proofs: [
      { label: 'NN noisy 4D/5D line-probe ReLU benchmark', command: 'npm', args: ['run', 'benchmark:nn-noisy-blackbox-relu-nd-line-probe'] }
    ],
    artifacts: [
      'benchmarks/release/nn-noisy-blackbox-relu-nd-line-probe/4d-5d-seed-sweep/report.json',
      'benchmarks/release/nn-noisy-blackbox-relu-nd-line-probe/4d-5d-seed-sweep/report.md'
    ]
  },
  {
    id: 'nn-noisy-blackbox-relu-nd-adversarial',
    name: 'NN noisy adversarial black-box ReLU ND line probe',
    status: 'production',
    summary: 'Runs generated noisy/quantized 4D/5D adversarial ND line-probe profiles with query-budget accounting.',
    scope: 'Noise-tolerant oracle-only shallow ReLU adversarial benchmark; proves generated near-parallel, offset-collision, low-amplitude, and oblique weak-support 4D/5D hyperplane stitching, not deep composition recovery or arbitrary production model extraction.',
    run: { label: 'NN noisy adversarial ND line-probe ReLU benchmark', command: 'npm', args: ['run', 'benchmark:nn-noisy-blackbox-relu-nd-adversarial'] },
    proofs: [
      { label: 'NN noisy adversarial ND line-probe ReLU benchmark', command: 'npm', args: ['run', 'benchmark:nn-noisy-blackbox-relu-nd-adversarial'] }
    ],
    artifacts: [
      'benchmarks/release/nn-noisy-blackbox-relu-nd-adversarial/generated-sweep/report.json',
      'benchmarks/release/nn-noisy-blackbox-relu-nd-adversarial/generated-sweep/report.md'
    ]
  },
  {
    id: 'nn-noisy-blackbox-relu-nd-adaptive-adversary',
    name: 'NN noisy adaptive-adversary black-box ReLU ND line probe',
    status: 'production',
    summary: 'Generates a pool of noisy/quantized 4D/5D shallow-ReLU teachers, selects the hardest profiles, and runs the ND line-probe solver with query-budget accounting.',
    scope: 'Noise-tolerant oracle-only shallow ReLU adaptive-adversary benchmark; proves selected near-parallel, offset-collision, weak-kink, and cancellation-risk 4D/5D hyperplane stitching, not deep composition recovery or arbitrary production model extraction.',
    run: { label: 'NN noisy adaptive-adversary ND line-probe ReLU benchmark', command: 'npm', args: ['run', 'benchmark:nn-noisy-blackbox-relu-nd-adaptive-adversary'] },
    proofs: [
      { label: 'NN noisy adaptive-adversary ND line-probe ReLU benchmark', command: 'npm', args: ['run', 'benchmark:nn-noisy-blackbox-relu-nd-adaptive-adversary'] }
    ],
    artifacts: [
      'benchmarks/release/nn-noisy-blackbox-relu-nd-adaptive-adversary/adaptive-search/report.json',
      'benchmarks/release/nn-noisy-blackbox-relu-nd-adaptive-adversary/adaptive-search/report.md'
    ]
  },
  {
    id: 'nn-noisy-blackbox-relu3d-randomized',
    name: 'NN noisy randomized 3D black-box ReLU decompile',
    status: 'production',
    summary: 'Reconstructs 3D ReLU hyperplanes across non-degenerate randomized teachers using repeated noisy and quantized oracle calls.',
    scope: 'Noise-tolerant black-box neural teacher seed-sweep benchmark; proves non-degenerate randomized 3D hyperplane stitching under stochastic output and finite precision, not adversarial near-collision, arbitrary ND, or deep network decompilation.',
    run: { label: 'NN noisy randomized 3D black-box ReLU decompile benchmark', command: 'npm', args: ['run', 'benchmark:nn-noisy-blackbox-relu3d-randomized'] },
    proofs: [
      { label: 'NN noisy randomized 3D black-box ReLU decompile benchmark', command: 'npm', args: ['run', 'benchmark:nn-noisy-blackbox-relu3d-randomized'] }
    ],
    artifacts: [
      'benchmarks/release/nn-noisy-blackbox-relu3d-randomized/seed-sweep/report.json',
      'benchmarks/release/nn-noisy-blackbox-relu3d-randomized/seed-sweep/report.md'
    ]
  },
  {
    id: 'nn-noisy-blackbox-relu3d-cancellation',
    name: 'NN noisy 3D cancellation ReLU decompile',
    status: 'production',
    summary: 'Uses global parametric refinement to recover an output-equivalent symbolic 3D ReLU surface when close opposite-sign noisy kinks partially cancel.',
    scope: 'Noise-tolerant black-box neural teacher hard-case benchmark; proves functional symbolic surface recovery for near-collision/cancellation cases, not unique hidden-unit identity for indistinguishable cancelling pairs.',
    run: { label: 'NN noisy 3D cancellation hard-case benchmark', command: 'npm', args: ['run', 'benchmark:nn-noisy-blackbox-relu3d-cancellation'] },
    proofs: [
      { label: 'NN noisy 3D cancellation hard-case benchmark', command: 'npm', args: ['run', 'benchmark:nn-noisy-blackbox-relu3d-cancellation'] }
    ],
    artifacts: [
      'benchmarks/release/nn-noisy-blackbox-relu3d-cancellation/hard-case/report.json',
      'benchmarks/release/nn-noisy-blackbox-relu3d-cancellation/hard-case/report.md'
    ]
  },
  {
    id: 'nn-noisy-blackbox-relu3d-adversarial',
    name: 'NN noisy 3D adversarial ReLU decompile',
    status: 'production',
    summary: 'Runs a wider noisy 3D adversarial sweep with global refinement and explicit oracle query-budget accounting.',
    scope: 'Noise-tolerant shallow 3D black-box ReLU teacher sweep covering near-parallel, beta-collision, cancellation, oblique-wide, and low-amplitude mixed profiles; not a general-ND or deep-network claim.',
    run: { label: 'NN noisy 3D adversarial sweep benchmark', command: 'npm', args: ['run', 'benchmark:nn-noisy-blackbox-relu3d-adversarial'] },
    proofs: [
      { label: 'NN noisy 3D adversarial sweep benchmark', command: 'npm', args: ['run', 'benchmark:nn-noisy-blackbox-relu3d-adversarial'] }
    ],
    artifacts: [
      'benchmarks/release/nn-noisy-blackbox-relu3d-adversarial/sweep/report.json',
      'benchmarks/release/nn-noisy-blackbox-relu3d-adversarial/sweep/report.md'
    ]
  },
  {
    id: 'nn-noisy-blackbox-relu2d-crossing',
    name: 'NN noisy 2D crossing ReLU decompile',
    status: 'production',
    summary: 'Recovers 2D piecewise-ReLU kink-line geometry when noisy slice detections include crossing-line outliers.',
    scope: 'Noise-tolerant black-box neural teacher benchmark; proves robust stitching for crossing non-vertical ReLU kink lines from noisy f(x,y) probes, not arbitrary high-dimensional network decompilation.',
    run: { label: 'NN noisy 2D crossing black-box ReLU decompile benchmark', command: 'npm', args: ['run', 'benchmark:nn-noisy-blackbox-relu2d-crossing'] },
    proofs: [
      { label: 'NN noisy 2D crossing black-box ReLU decompile benchmark', command: 'npm', args: ['run', 'benchmark:nn-noisy-blackbox-relu2d-crossing'] }
    ],
    artifacts: [
      'benchmarks/release/nn-noisy-blackbox-relu2d-decompile/crossing/report.json',
      'benchmarks/release/nn-noisy-blackbox-relu2d-decompile/crossing/report.md'
    ]
  },
  {
    id: 'nn-noisy-blackbox-relu2d-near-parallel',
    name: 'NN noisy 2D near-parallel ReLU decompile',
    status: 'production',
    summary: 'Recovers 2D piecewise-ReLU kink-line geometry when noisy lines are near-parallel and at risk of merging.',
    scope: 'Noise-tolerant black-box neural teacher benchmark; proves robust stitching for near-parallel non-vertical ReLU kink lines from noisy f(x,y) probes, not arbitrary high-dimensional network decompilation.',
    run: { label: 'NN noisy 2D near-parallel black-box ReLU decompile benchmark', command: 'npm', args: ['run', 'benchmark:nn-noisy-blackbox-relu2d-near-parallel'] },
    proofs: [
      { label: 'NN noisy 2D near-parallel black-box ReLU decompile benchmark', command: 'npm', args: ['run', 'benchmark:nn-noisy-blackbox-relu2d-near-parallel'] }
    ],
    artifacts: [
      'benchmarks/release/nn-noisy-blackbox-relu2d-decompile/near-parallel/report.json',
      'benchmarks/release/nn-noisy-blackbox-relu2d-decompile/near-parallel/report.md'
    ]
  },
  {
    id: 'nn-noisy-blackbox-relu2d-near-collision',
    name: 'NN noisy 2D near-collision ReLU decompile',
    status: 'production',
    summary: 'Recovers 2D piecewise-ReLU kink-line geometry when two noisy kink lines collide on an observed slice.',
    scope: 'Noise-tolerant black-box neural teacher benchmark; proves robust stitching when a slice sees merged/colliding kink detections, not arbitrary high-dimensional network decompilation.',
    run: { label: 'NN noisy 2D near-collision black-box ReLU decompile benchmark', command: 'npm', args: ['run', 'benchmark:nn-noisy-blackbox-relu2d-near-collision'] },
    proofs: [
      { label: 'NN noisy 2D near-collision black-box ReLU decompile benchmark', command: 'npm', args: ['run', 'benchmark:nn-noisy-blackbox-relu2d-near-collision'] }
    ],
    artifacts: [
      'benchmarks/release/nn-noisy-blackbox-relu2d-decompile/near-collision/report.json',
      'benchmarks/release/nn-noisy-blackbox-relu2d-decompile/near-collision/report.md'
    ]
  },
  {
    id: 'island-nn-archive',
    name: 'Island NN archive',
    status: 'production',
    summary: 'Runs real multi-island engine histories through an MLP candidate gate with archive telemetry, robust holdout selection, and single-baseline fallback.',
    scope: '24-seed release sweep with timeout/error accounting and non-regression checks against the single baseline.',
    run: { label: 'island NN archive wide benchmark', command: 'npm', args: ['run', 'benchmark:island-nn-archive:wide'] },
    proofs: [
      { label: 'island NN archive wide benchmark', command: 'npm', args: ['run', 'benchmark:island-nn-archive:wide'] }
    ],
    artifacts: [
      'benchmarks/release/island-nn-archive/wide/report.json',
      'benchmarks/release/island-nn-archive/wide/report.md'
    ]
  },
  {
    id: 'open-source-baseline',
    name: 'Open-source symbolic-regression baseline',
    status: 'diagnostic',
    summary: 'Compares pure GCL against open-source gplearn and DEAP baselines, with claim accounting separated from hybrid preview claims.',
    scope: 'Small symbolic-regression smoke suite with external Python baseline dependencies and pure-GCL claim accounting.',
    run: { label: 'open-source benchmark', command: 'npm', args: ['run', 'benchmark:opensource'] },
    proofs: [
      { label: 'open-source smoke gate', command: 'npm', args: ['run', 'benchmark:opensource:gate'] }
    ],
    artifacts: [
      'benchmarks/release/open-source-symbolic-regression/smoke/report.json',
      'benchmarks/release/open-source-symbolic-regression/smoke/report.md'
    ]
  },
  {
    id: 'gcl-hybrid-basis',
    name: 'GCL hybrid sparse basis',
    status: 'production',
    summary: 'Uses a train-only sparse basis gate over grammar-derived symbolic terms and audits the result against pure GCL without folding it into pure-GCL claims.',
    scope: 'Eight-task symbolic-regression gate with no hybrid MAE/success regressions, 100% solved rate, and separate pure-GCL claim accounting.',
    run: { label: 'GCL hybrid production gate', command: 'npm', args: ['run', 'benchmark:gcl-hybrid:gate'] },
    proofs: [
      { label: 'GCL hybrid production gate', command: 'npm', args: ['run', 'benchmark:gcl-hybrid:gate'] }
    ],
    artifacts: [
      'benchmarks/release/open-source-symbolic-regression/hybrid-wide/report.json',
      'benchmarks/release/open-source-symbolic-regression/hybrid-wide/report.md'
    ]
  }
];

const program = new Command();

program
  .name('gclab')
  .description('Genetic Code Lab - Evolve code through genetic programming')
  .version('1.0.0');

program
  .command('math')
  .description('Evolve a mathematical expression')
  .option('-g, --generations <number>', 'Maximum generations', '100')
  .option('-p, --population <number>', 'Population size', '100')
  .option('-t, --target <expression>', 'Target expression (e.g., "x*x + 2*x + 1")')
  .option('-o, --output <file>', 'Save results to file')
  .option('-s, --seed <number>', 'Random seed')
  .action(async (options) => {
    console.log(chalk.cyan.bold('\n🧬 Genetic Code Lab - Math Expression Evolution\n'));
    const spinner = ora('Setting up problem...').start();

    try {
      const generations = positiveInt(options.generations, MATH_DEFAULTS.maxGenerations, 'generations');
      const populationSize = positiveInt(options.population, MATH_DEFAULTS.population.size, 'population');
      const seed = options.seed === undefined ? undefined : positiveInt(options.seed, 1, 'seed');
      const config = buildEvolutionConfig({ generations, populationSize, seed }, MATH_DEFAULTS);

      await runMathEvolution({
        command: 'math',
        spinner,
        output: options.output,
        target: options.target || DEFAULT_MATH_TARGET,
        variableCount: 1,
        testCount: 30,
        config,
        showSampleTable: true
      });
    } catch (error) {
      failCli(spinner, error, error instanceof Error && error.message.startsWith('Invalid') ? 'Invalid target expression' : 'Invalid CLI option');
    }
  });

program
  .command('sort')
  .description('Evolve a sorting network')
  .option('-n, --size <number>', 'Array size to sort', '4')
  .option('-g, --generations <number>', 'Maximum generations', '200')
  .option('-p, --population <number>', 'Population size', '200')
  .option('-o, --output <file>', 'Save results to file')
  .action(async (options) => {
    console.log(chalk.cyan.bold('\n🧬 Genetic Code Lab - Sorting Network Evolution\n'));
    const spinner = ora('Creating problem...').start();

    try {
      const size = positiveInt(options.size, 4, 'size');
      const generations = positiveInt(options.generations, SORT_DEFAULTS.maxGenerations, 'generations');
      const populationSize = positiveInt(options.population, SORT_DEFAULTS.population.size, 'population');
      const config = buildEvolutionConfig({ generations, populationSize }, sortingDefaults(size));

      await runSortEvolution({
        command: 'sort',
        spinner,
        output: options.output,
        size,
        config
      });
    } catch (error) {
      failCli(spinner, error, 'Invalid CLI option');
    }
  });

program
  .command('path')
  .description('Evolve a pathfinding algorithm')
  .option('-w, --width <number>', 'Grid width', '8')
  .option('-h, --height <number>', 'Grid height', '8')
  .option('-g, --generations <number>', 'Maximum generations', '150')
  .option('-p, --population <number>', 'Population size', '150')
  .option('-o, --output <file>', 'Save results to file')
  .action(async (options) => {
    console.log(chalk.cyan.bold('\n🧬 Genetic Code Lab - Pathfinding Evolution\n'));
    const spinner = ora('Creating problem...').start();

    try {
      const width = positiveInt(options.width, 8, 'width');
      const height = positiveInt(options.height, 8, 'height');
      const generations = positiveInt(options.generations, PATH_DEFAULTS.maxGenerations, 'generations');
      const populationSize = positiveInt(options.population, PATH_DEFAULTS.population.size, 'population');
      const config = buildEvolutionConfig({ generations, populationSize }, PATH_DEFAULTS);

      await runPathEvolution({
        command: 'path',
        spinner,
        output: options.output,
        width,
        height,
        testCount: 5,
        config
      });
    } catch (error) {
      failCli(spinner, error, 'Invalid CLI option');
    }
  });

program
  .command('run <config>')
  .description('Run evolution from config file')
  .option('-o, --output <file>', 'Save results to file')
  .action(async (configPath, options) => {
    console.log(chalk.cyan.bold('\n🧬 Genetic Code Lab - Custom Evolution\n'));
    const spinner = ora('Loading config...').start();

    try {
      const runConfig = await fs.readJson(configPath) as CliRunConfig;
      const problemConfig = runConfig.problem || {};
      const type = problemConfig.type;
      if (!type || !['math', 'sort', 'path'].includes(type)) {
        throw new Error('Config must set problem.type to one of: math, sort, path');
      }

      if (type === 'math') {
        const target = problemConfig.target || DEFAULT_MATH_TARGET;
        const variableCount = positiveInt(problemConfig.variableCount, 1, 'problem.variableCount');
        const testCount = positiveInt(problemConfig.testCount, 30, 'problem.testCount');
        const config = buildEvolutionConfig(runConfig.evolution, MATH_DEFAULTS);
        await runMathEvolution({
          command: 'run',
          spinner,
          output: options.output || runConfig.output,
          target,
          variableCount,
          testCount,
          config,
          showSampleTable: variableCount === 1
        });
        return;
      }

      if (type === 'sort') {
        const size = positiveInt(problemConfig.size, 4, 'problem.size');
        const config = buildEvolutionConfig(runConfig.evolution, sortingDefaults(size));
        await runSortEvolution({
          command: 'run',
          spinner,
          output: options.output || runConfig.output,
          size,
          config
        });
        return;
      }

      const width = positiveInt(problemConfig.width, 8, 'problem.width');
      const height = positiveInt(problemConfig.height, 8, 'problem.height');
      const testCount = positiveInt(problemConfig.testCount, 5, 'problem.testCount');
      const config = buildEvolutionConfig(runConfig.evolution, PATH_DEFAULTS);
      await runPathEvolution({
        command: 'run',
        spinner,
        output: options.output || runConfig.output,
        width,
        height,
        testCount,
        config
      });
    } catch (error) {
      failCli(spinner, error, 'Config run failed');
    }
  });

program
  .command('capabilities [id]')
  .description('List innovation capabilities or inspect one capability')
  .option('--json', 'Print machine-readable JSON')
  .action((id, options) => {
    const capability = id ? findCapability(id) : null;
    if (id && !capability) {
      console.error(chalk.red(`Unknown capability: ${id}`));
      console.error(chalk.white(`Known capabilities: ${CAPABILITIES.map(item => item.id).join(', ')}`));
      process.exitCode = 1;
      return;
    }

    if (options.json) {
      console.log(JSON.stringify(capability || CAPABILITIES, null, 2));
      return;
    }

    if (capability) {
      printCapability(capability);
      return;
    }

    console.log(chalk.cyan.bold('\nGenetic Code Lab Capabilities\n'));
    for (const item of CAPABILITIES) {
      console.log(`${statusLabel(item.status)} ${chalk.bold(item.id)} - ${item.name}`);
      console.log(chalk.white(`  ${item.summary}`));
      console.log(chalk.dim(`  verify: gclab capability ${item.id} --verify`));
    }
  });

program
  .command('capability <id>')
  .description('Run or verify one innovation capability')
  .option('--verify', 'Run the capability proof command(s)')
  .option('--run', 'Run the capability primary command')
  .option('--json', 'Print capability metadata as JSON')
  .action((id, options) => {
    const capability = findCapability(id);
    if (!capability) {
      console.error(chalk.red(`Unknown capability: ${id}`));
      console.error(chalk.white(`Known capabilities: ${CAPABILITIES.map(item => item.id).join(', ')}`));
      process.exitCode = 1;
      return;
    }

    if (options.json) {
      console.log(JSON.stringify(capability, null, 2));
      return;
    }

    if (options.verify) {
      runCapabilityCommands(capability, capability.proofs);
      return;
    }

    if (options.run) {
      runCapabilityCommands(capability, [capability.run]);
      return;
    }

    printCapability(capability);
  });

program
  .command('explain-llm')
  .description('Read a live GPT-2/LLM forward pass and explain hooked MLP neuron reactions')
  .requiredOption('--prompt <text>', 'Prompt to run through the live model')
  .option('--model <id-or-path>', 'HuggingFace causal LM id or local model path', 'gpt2')
  .option('--layer <number>', 'Transformer block index to inspect', '0')
  .option('--token-index <number>', 'Prompt token index to explain; negative indexes are supported', '-1')
  .option('--top-neurons <number>', 'Number of top MLP source neurons to report', '12')
  .option('--top-logits <number>', 'Number of next-token logits to report', '8')
  .option('--ablation-mode <selected-token|all-tokens|none>', 'Run real causal neuron ablations for reported neurons', 'selected-token')
  .option('--negative-controls', 'Run negative-control ablations against reported neurons', true)
  .option('--no-negative-controls', 'Disable negative-control ablations')
  .option('--control-count <number>', 'Number of random/control neurons per negative-control family', '3')
  .option('--control-prompt <text>', 'Different real prompt used to select off-prompt top neurons', 'The weather in Berlin is')
  .option('--control-seed <number>', 'Deterministic seed for random negative-control neuron selection', '1337')
  .option('--device <cpu|cuda|auto>', 'Runtime device', 'cpu')
  .option('--output <file>', 'Write JSON report to file')
  .option('--json', 'Print machine-readable JSON')
  .option('--allow-download', 'Allow HuggingFace downloads if the model is not already cached')
  .action((options) => {
    const repoRoot = path.resolve(__dirname, '..', '..');
    const scriptPath = path.join(repoRoot, 'tools', 'llm-reader-gpt2-live.py');
    const args = [
      scriptPath,
      '--model', options.model,
      '--prompt', options.prompt,
      '--layer', String(nonNegativeInt(options.layer, 0, 'layer')),
      '--token-index', String(integerOption(options.tokenIndex, -1, 'token-index')),
      '--top-neurons', String(positiveInt(options.topNeurons, 12, 'top-neurons')),
      '--top-logits', String(positiveInt(options.topLogits, 8, 'top-logits')),
      '--ablation-mode', String(options.ablationMode),
      options.negativeControls ? '--negative-controls' : '--no-negative-controls',
      '--control-count', String(positiveInt(options.controlCount, 3, 'control-count')),
      '--control-prompt', String(options.controlPrompt),
      '--control-seed', String(integerOption(options.controlSeed, 1337, 'control-seed')),
      '--device', options.device,
      '--format', options.json ? 'json' : 'text'
    ];
    if (!options.allowDownload) args.push('--local-files-only');
    if (options.output) args.push('--output', options.output);

    if (!fs.pathExistsSync(scriptPath)) {
      console.error(chalk.red(`Missing LLM reader tool: ${scriptPath}`));
      process.exitCode = 1;
      return;
    }

    const pythonCommand = resolveGclPython(repoRoot);
    const result = spawnSync(pythonCommand, args, {
      cwd: repoRoot,
      stdio: 'inherit',
      env: process.env
    });
    if (result.error) {
      console.error(chalk.red(`Failed to start live LLM reader: ${result.error.message}`));
      process.exitCode = 1;
      return;
    }
    process.exitCode = result.status ?? 1;
  });

program
  .command('doctor')
  .description('Verify local installation and optionally run the full release gate')
  .option('--release', 'Run the full release gate: build, tests, routing gate, and promotion safety gate')
  .action(async (options) => {
    console.log(chalk.cyan.bold('\n🧬 Genetic Code Lab - Doctor\n'));

    const repoRoot = path.resolve(__dirname, '..', '..');
    const exampleConfigPath = path.join(repoRoot, 'configs', 'math-quadratic.json');
    const devMode = await fs.pathExists(path.join(repoRoot, 'src'));
    const checks = [
      { name: 'package.json', pass: await fs.pathExists(path.join(repoRoot, 'package.json')) },
      { name: 'compiled CLI artifact', pass: await fs.pathExists(path.join(repoRoot, 'dist', 'cli', 'index.js')) },
      { name: 'example config', pass: await fs.pathExists(exampleConfigPath) },
      { name: 'math expression parser', pass: doctorCheckMathParser() },
      { name: 'config runner schema', pass: await doctorCheckConfigRunnerSchema(exampleConfigPath) },
      ...(devMode ? [
        { name: 'TypeScript config', pass: await fs.pathExists(path.join(repoRoot, 'tsconfig.json')) },
        { name: 'release gate script', pass: await fs.pathExists(path.join(repoRoot, 'scripts', 'release-gate.ts')) }
      ] : [])
    ];

    for (const check of checks) {
      console.log(`${check.pass ? chalk.green('PASS') : chalk.red('FAIL')} ${check.name}`);
    }

    if (checks.some(check => !check.pass)) {
      console.log(chalk.yellow('\nRun `npm install` and `npm run build`, then retry `npm run start -- doctor`.'));
      process.exitCode = 1;
      return;
    }

    if (!options.release) {
      console.log(chalk.green('\nQuick doctor passed.'));
      console.log(chalk.white('Run `npm run start -- doctor --release` for the full release gate.'));
      return;
    }

    console.log(chalk.yellow('\nRunning full release gate. This can take several minutes.\n'));
    const result = spawnSync('npm', ['run', 'release:gate'], {
      cwd: repoRoot,
      stdio: 'inherit',
      env: process.env
    });

    if (result.error) {
      console.error(chalk.red(`Release gate failed to start: ${result.error.message}`));
      process.exitCode = 1;
      return;
    }

    process.exitCode = result.status ?? 1;
  });

async function runMathEvolution(options: MathRunOptions): Promise<void> {
  const { command, spinner, output, target, variableCount, testCount, config, showSampleTable } = options;
  spinner.text = 'Creating math problem...';

  const compiledTarget = compileMathExpression(target);
  const targetFn = (inputs: number[]) => {
    const scope: Record<string, number> = { x: inputs[0], x0: inputs[0] };
    inputs.forEach((value, idx) => {
      scope[`x${idx}`] = value;
    });
    return compiledTarget(scope);
  };
  const problem = createMathProblem(targetFn, variableCount, testCount);
  const engine = new EvolutionEngine(problem, {
    ...config,
    onGeneration: (_, stats) => {
      if (stats.generation % 10 === 0) {
        spinner.text = `Generation ${stats.generation} | Best: ${stats.bestFitness.toFixed(4)}`;
      }
    }
  });

  spinner.text = 'Evolving...';
  const result = await engine.evolve();
  spinner.succeed('Evolution complete!');

  displayResult(result, 'math');
  const expression = genomeToString(result.bestGenome);
  console.log(chalk.yellow('\n📊 Evolved Expression:'));
  console.log(chalk.white(expression));

  if (showSampleTable) {
    printMathSampleTable(result, targetFn);
  }

  await saveResultIfRequested(result, output, buildResultMetadata(command, 'math', {
    target,
    variableCount,
    testCount
  }, config, {
    expression
  }));
}

async function runSortEvolution(options: SortRunOptions): Promise<void> {
  const { command, spinner, output, size, config } = options;
  spinner.text = 'Creating sorting problem...';

  const problem = createSortingProblem(size);
  const engine = new EvolutionEngineTree(problem, {
    ...config,
    onGeneration: (_, stats) => {
      if (stats.generation % 20 === 0) {
        spinner.text = `Generation ${stats.generation} | Best: ${stats.bestFitness.toFixed(4)}`;
      }
    }
  });

  spinner.text = 'Evolving sorting network...';
  const result = await engine.evolve();
  spinner.succeed('Evolution complete!');

  displayResult(result, 'sort');
  const network = genomeToSortingNetwork(result.bestGenome, size);
  console.log(chalk.yellow('\n📊 Evolved Sorting Network:'));
  console.log(chalk.white(network));

  await saveResultIfRequested(result, output, buildResultMetadata(command, 'sort', {
    size
  }, config, {
    network
  }));
}

async function runPathEvolution(options: PathRunOptions): Promise<void> {
  const { command, spinner, output, width, height, testCount, config } = options;
  spinner.text = 'Creating pathfinding problem...';

  const problem = createPathfindingProblem(width, height, testCount);
  const engine = new EvolutionEngineTree(problem, {
    ...config,
    onGeneration: (_, stats) => {
      if (stats.generation % 15 === 0) {
        spinner.text = `Generation ${stats.generation} | Best: ${stats.bestFitness.toFixed(4)}`;
      }
    }
  });

  spinner.text = 'Evolving pathfinding algorithm...';
  const result = await engine.evolve();
  spinner.succeed('Evolution complete!');

  displayResult(result, 'path');
  const moves = extractMoves(result.bestGenome);
  console.log(chalk.yellow('\n📊 Evolved Moves:'));
  console.log(chalk.white(`  Sequence: ${moves.join(' -> ')}`));
  console.log(chalk.white(`  Total moves: ${moves.length}`));

  const firstTest = problem.testCases[0] as PathfindingTestCase | undefined;
  if (firstTest) {
    const { grid, start, goal } = firstTest.input;
    const pathResult = executePathfinding(grid, start, goal, moves);
    console.log(chalk.yellow('\n🗺️ Sample Path:'));
    console.log(chalk.white(visualizePath(grid, pathResult.path, start, goal)));
    console.log(chalk.white(`\n  Reached goal: ${pathResult.reached ? chalk.green('Yes') : chalk.red('No')}`));
    console.log(chalk.white(`  Steps taken: ${pathResult.steps}`));
  }

  await saveResultIfRequested(result, output, buildResultMetadata(command, 'path', {
    width,
    height,
    testCount
  }, config, {
    moves
  }));
}

function sortingDefaults(size: number): typeof SORT_DEFAULTS {
  return {
    ...SORT_DEFAULTS,
    population: {
      ...SORT_DEFAULTS.population,
      maxGenomeSize: size * size
    }
  };
}

function positiveInt(value: unknown, fallback: number, name: string): number {
  if (value === undefined || value === null) return fallback;
  const parsed = typeof value === 'number' ? value : Number(value);
  if (!Number.isInteger(parsed) || parsed <= 0) {
    throw new Error(`${name} must be a positive integer`);
  }
  return parsed;
}

function nonNegativeInt(value: unknown, fallback: number, name: string): number {
  if (value === undefined || value === null) return fallback;
  const parsed = typeof value === 'number' ? value : Number(value);
  if (!Number.isInteger(parsed) || parsed < 0) {
    throw new Error(`${name} must be a non-negative integer`);
  }
  return parsed;
}

function integerOption(value: unknown, fallback: number, name: string): number {
  if (value === undefined || value === null) return fallback;
  const parsed = typeof value === 'number' ? value : Number(value);
  if (!Number.isInteger(parsed)) {
    throw new Error(`${name} must be an integer`);
  }
  return parsed;
}

function resolveGclPython(repoRoot: string): string {
  if (process.env.GCL_PYTHON) return process.env.GCL_PYTHON;
  if (process.env.GCL_LLM_READER_PYTHON) return process.env.GCL_LLM_READER_PYTHON;
  const projectPython = path.join(repoRoot, '.gcl-python', 'bin', 'python');
  if (fs.pathExistsSync(projectPython)) return projectPython;
  return 'python3';
}

function finiteNumber(value: unknown, fallback: number, name: string): number {
  if (value === undefined || value === null) return fallback;
  const parsed = typeof value === 'number' ? value : Number(value);
  if (!Number.isFinite(parsed)) {
    throw new Error(`${name} must be a finite number`);
  }
  return parsed;
}

function probability(value: unknown, fallback: number, name: string): number {
  const parsed = finiteNumber(value, fallback, name);
  if (parsed < 0 || parsed > 1) {
    throw new Error(`${name} must be between 0 and 1`);
  }
  return parsed;
}

function buildEvolutionConfig(
  overrides: CliRunConfig['evolution'] = {},
  defaults: {
    population: PopulationConfig;
    maxGenerations: number;
    targetFitness: number;
    stagnationLimit: number;
  }
): Partial<EvolutionConfig> {
  const populationOverrides = (overrides.population || {}) as Partial<PopulationConfig>;
  const population: PopulationConfig = {
    ...defaults.population,
    ...populationOverrides,
    size: positiveInt(overrides.populationSize ?? populationOverrides.size, defaults.population.size, 'evolution.population.size'),
    eliteCount: positiveInt(overrides.eliteCount ?? populationOverrides.eliteCount, defaults.population.eliteCount, 'evolution.population.eliteCount'),
    crossoverRate: probability(overrides.crossoverRate ?? populationOverrides.crossoverRate, defaults.population.crossoverRate, 'evolution.population.crossoverRate'),
    mutationRate: probability(overrides.mutationRate ?? populationOverrides.mutationRate, defaults.population.mutationRate, 'evolution.population.mutationRate'),
    maxGenomeDepth: positiveInt(overrides.maxGenomeDepth ?? populationOverrides.maxGenomeDepth, defaults.population.maxGenomeDepth, 'evolution.population.maxGenomeDepth'),
    maxGenomeSize: positiveInt(overrides.maxGenomeSize ?? populationOverrides.maxGenomeSize, defaults.population.maxGenomeSize, 'evolution.population.maxGenomeSize'),
    tournamentSize: positiveInt(overrides.tournamentSize ?? populationOverrides.tournamentSize, defaults.population.tournamentSize, 'evolution.population.tournamentSize')
  };

  return {
    ...overrides,
    population,
    maxGenerations: positiveInt(overrides.maxGenerations ?? overrides.generations, defaults.maxGenerations, 'evolution.maxGenerations'),
    targetFitness: probability(overrides.targetFitness, defaults.targetFitness, 'evolution.targetFitness'),
    stagnationLimit: positiveInt(overrides.stagnationLimit, defaults.stagnationLimit, 'evolution.stagnationLimit'),
    parallelEvaluators: positiveInt(overrides.parallelEvaluators, 1, 'evolution.parallelEvaluators')
  };
}

function doctorCheckMathParser(): boolean {
  try {
    const expression = compileMathExpression(DEFAULT_MATH_TARGET);
    return expression({ x: 3, x0: 3 }) === 16;
  } catch {
    return false;
  }
}

function findCapability(id: string): Capability | undefined {
  return CAPABILITIES.find(item => item.id === id);
}

function statusLabel(status: CapabilityStatus): string {
  if (status === 'production') return chalk.green('[production]');
  if (status === 'preview') return chalk.yellow('[preview]');
  return chalk.gray('[diagnostic]');
}

function formatCapabilityCommand(command: CapabilityCommand): string {
  return [command.command, ...command.args].join(' ');
}

function printCapability(capability: Capability): void {
  console.log(chalk.cyan.bold(`\n${capability.name}\n`));
  console.log(`${chalk.bold('ID:')} ${capability.id}`);
  console.log(`${chalk.bold('Status:')} ${statusLabel(capability.status)}`);
  console.log(`${chalk.bold('Summary:')} ${capability.summary}`);
  console.log(`${chalk.bold('Scope:')} ${capability.scope}`);
  console.log(`${chalk.bold('Run:')} ${formatCapabilityCommand(capability.run)}`);
  console.log(chalk.bold('Proofs:'));
  for (const proof of capability.proofs) {
    console.log(`  - ${proof.label}: ${formatCapabilityCommand(proof)}`);
  }
  if (capability.artifacts.length > 0) {
    console.log(chalk.bold('Artifacts:'));
    for (const artifact of capability.artifacts) {
      console.log(`  - ${artifact}`);
    }
  }
}

function runCapabilityCommands(capability: Capability, commands: CapabilityCommand[]): void {
  const repoRoot = path.resolve(__dirname, '..', '..');
  console.log(chalk.cyan.bold(`\n${capability.name}\n`));
  for (const command of commands) {
    console.log(chalk.yellow(`[capability] ${command.label}`));
    console.log(chalk.dim(`$ ${formatCapabilityCommand(command)}`));
    const result = spawnSync(command.command, command.args, {
      cwd: repoRoot,
      stdio: 'inherit',
      env: process.env
    });
    if (result.error) {
      console.error(chalk.red(`Failed to start ${command.label}: ${result.error.message}`));
      process.exitCode = 1;
      return;
    }
    if (result.status !== 0) {
      process.exitCode = result.status ?? 1;
      return;
    }
  }
}

async function doctorCheckConfigRunnerSchema(configPath: string): Promise<boolean> {
  try {
    const config = await fs.readJson(configPath) as CliRunConfig;
    const type = config.problem?.type;
    if (!type || !['math', 'sort', 'path'].includes(type)) return false;
    if (config.evolution?.populationSize !== undefined) {
      positiveInt(config.evolution.populationSize, 1, 'evolution.populationSize');
    }
    if (config.evolution?.generations !== undefined) {
      positiveInt(config.evolution.generations, 1, 'evolution.generations');
    }
    if (type === 'math' && config.problem?.target) {
      compileMathExpression(config.problem.target);
    }
    return true;
  } catch {
    return false;
  }
}

function displayResult(result: EvolutionResult, type: string): void {
  console.log(chalk.green('\n✨ Evolution Results:'));
  console.log(chalk.white(`  Status: ${result.success ? chalk.green('Success!') : chalk.yellow('Partial')}`));
  console.log(chalk.white(`  Best Fitness: ${result.bestGenome.fitness.toFixed(6)}`));
  console.log(chalk.white(`  Generations: ${result.generationsRun}`));
  console.log(chalk.white(`  Total Evaluations: ${result.totalEvaluations}`));
  console.log(chalk.white(`  Time: ${formatTime(result.totalTimeMs)}`));

  const stats = result.stats;
  if (stats.length > 0) {
    const first = stats[0].bestFitness;
    const last = stats[stats.length - 1].bestFitness;
    const improvement = last - first;
    console.log(chalk.white(`  Fitness Improvement: ${improvement.toFixed(6)}`));
  }
}

function printMathSampleTable(result: EvolutionResult, targetFn: (inputs: number[]) => number): void {
  console.log(chalk.yellow('\n🧪 Test Results:'));
  const testValues = [-2, -1, 0, 1, 2, 3];
  console.log('  x   | Expected | Got      | Error');
  console.log('------|----------|----------|--------');

  for (const x of testValues) {
    const expected = targetFn([x]);
    const got = executeMathGenome(result.bestGenome, { x });
    const error = Math.abs(expected - got).toFixed(4);
    console.log(`  ${x.toString().padEnd(4)} | ${expected.toFixed(4).padEnd(8)} | ${got.toFixed(4).padEnd(8)} | ${error}`);
  }
}

function buildResultMetadata(
  command: string,
  problemType: BuiltInProblemType,
  problem: Record<string, unknown>,
  config: Partial<EvolutionConfig>,
  solution: Record<string, unknown>
): ResultMetadata {
  return { command, problemType, problem, config, solution };
}

async function saveResultIfRequested(result: EvolutionResult, outputPath?: string, metadata?: ResultMetadata): Promise<void> {
  if (!outputPath) return;
  await saveResult(result, outputPath, metadata);
  console.log(chalk.green(`\n💾 Results saved to ${outputPath}`));
}

async function saveResult(result: EvolutionResult, outputPath: string, metadata?: ResultMetadata): Promise<void> {
  const output = {
    schemaVersion: 1,
    timestamp: new Date().toISOString(),
    summary: {
      success: result.success,
      bestFitness: result.bestGenome.fitness,
      generations: result.generationsRun,
      evaluations: result.totalEvaluations,
      timeMs: result.totalTimeMs,
      bestGenomeSize: result.bestGenome.genes.length
    },
    cli: metadata ? { command: metadata.command } : undefined,
    problem: metadata ? { type: metadata.problemType, ...metadata.problem } : undefined,
    config: metadata?.config,
    solution: metadata?.solution,
    success: result.success,
    bestFitness: result.bestGenome.fitness,
    generations: result.generationsRun,
    evaluations: result.totalEvaluations,
    timeMs: result.totalTimeMs,
    bestGenome: result.bestGenome,
    stats: result.stats
  };

  await fs.ensureDir(path.dirname(outputPath));
  await fs.writeJson(outputPath, output, { spaces: 2 });
}

function failCli(spinner: Spinner, error: unknown, label: string): void {
  spinner.fail(label);
  console.error(chalk.red(error instanceof Error ? error.message : String(error)));
  process.exitCode = 1;
}

program.parse();
