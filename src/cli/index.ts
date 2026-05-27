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
