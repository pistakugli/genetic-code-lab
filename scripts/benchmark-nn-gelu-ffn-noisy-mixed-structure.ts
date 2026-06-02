#!/usr/bin/env ts-node

import { spawnSync } from 'child_process';
import * as fs from 'fs';
import * as path from 'path';
import {
  canonicalizeGeluFfnStructure,
  decompileRankAdaptiveGeluFfnStructure,
  predictGeluFfnStructureBatch
} from '../src/solvers/gelu-ffn-structure-finder';
import { regressionMetrics } from '../src/solvers/symbolic-residual-ensemble';
import { resolveBenchmarkOutput, writeBenchmarkManifest } from './benchmark-paths';

type TeacherAtom = {
  direction: number[];
  inputScale: number;
  inputBias: number;
  outputWeight: number;
};

type TeacherCase = {
  id: string;
  profile: 'balanced-noisy' | 'near-parallel-noisy' | 'weak-resolvable-noisy';
  dimension: number;
  atoms: TeacherAtom[];
  outputBias: number;
  noiseStd: number;
  quantization: number;
  querySamples: number;
  randomSeed: number;
  ranges: Array<[number, number]>;
  solver: {
    gradientProbeCount: number;
    trainingSampleCount: number;
    validationSampleCount: number;
    restarts: number;
    iterations: number;
    learningRate: number;
    atomRelativeStrengthThreshold: number;
    atomAbsoluteStrengthThreshold: number;
    pruneValidationToleranceRatio: number;
    pruneValidationToleranceAbsolute: number;
    nearParallelDirectionCosine: number;
    duplicateDirectionCosine: number;
    duplicateBiasTolerance: number;
  };
};

type Predictor = (input: number[]) => number;
type EvaluationSet = {
  id: 'iid-uniform' | 'wide-ood' | 'shifted-ood';
  kind: 'iid' | 'ood';
  inputs: number[][];
  actual: number[];
};
type EvaluationMetrics = {
  id: EvaluationSet['id'];
  kind: EvaluationSet['kind'];
  metrics: ReturnType<typeof regressionMetrics>;
  worstError: number;
};
type PredictionBaselineRow = {
  id: string;
  family: string;
  metrics: ReturnType<typeof regressionMetrics>;
  worstError: number;
  evaluations: EvaluationMetrics[];
  trainingRawOracleCalls: number;
  package?: string;
};

const repoRoot = path.resolve(__dirname, '..');
const output = resolveBenchmarkOutput(
  repoRoot,
  'nn-gelu-ffn-noisy-mixed-structure',
  'quantized-noisy-family'
);
const strongBaselineVenvPath = process.env.GCL_NEURAL_BASELINE_PYTHON_VENV ?? '/tmp/gcl-neural-baseline-venv';
const strongBaselineRequirementsPath = path.join(repoRoot, 'scripts', 'neural-baseline-python-requirements.txt');
const requiredStrongBaselineModules = ['numpy', 'sklearn', 'xgboost'];
let cachedStrongBaselinePythonBin: string | null = null;

function makeRandom(seed: number): () => number {
  let state = Math.floor(seed) % 2147483647;
  if (state <= 0) state += 2147483646;
  return () => {
    state = (state * 16807) % 2147483647;
    return (state - 1) / 2147483646;
  };
}

function gaussian(random: () => number): number {
  const u1 = Math.max(1e-12, random());
  const u2 = random();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

function dot(a: number[], b: number[]): number {
  return a.reduce((sum, value, index) => sum + value * b[index], 0);
}

function norm(values: number[]): number {
  return Math.sqrt(Math.max(0, dot(values, values)));
}

function normalize(values: number[]): number[] {
  const length = Math.max(1e-12, norm(values));
  return values.map(value => value / length);
}

function subtractProjection(vector: number[], basis: number[][]): number[] {
  let outputVector = [...vector];
  for (const direction of basis) {
    const projection = dot(outputVector, direction);
    outputVector = outputVector.map((value, index) => value - projection * direction[index]);
  }
  return outputVector;
}

function gelu(value: number): number {
  return 0.5 * value * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (value + 0.044715 * value ** 3)));
}

function makeDirections(dimension: number, count: number, seed: number): number[][] {
  const random = makeRandom(seed);
  const directions: number[][] = [];
  for (let index = 0; index < count; index++) {
    directions.push(normalize(subtractProjection(
      Array.from({ length: dimension }, () => gaussian(random)),
      directions
    )));
  }
  return directions;
}

function makeInputs(seed: number, dimension: number, count: number, range: [number, number] = [-1.5, 1.5]): number[][] {
  const random = makeRandom(seed);
  return Array.from({ length: count }, () => (
    Array.from({ length: dimension }, () => range[0] + (range[1] - range[0]) * random())
  ));
}

function makeShiftedOodInputs(seed: number, dimension: number, count: number): number[][] {
  const random = makeRandom(seed);
  return Array.from({ length: count }, () => (
    Array.from({ length: dimension }, (_unused, axis) => {
      const shift = axis % 3 === 0 ? 0.55 : axis % 3 === 1 ? -0.45 : 0.25;
      return -1.35 + 2.7 * random() + shift;
    })
  ));
}

function predictTeacher(teacher: TeacherCase, input: number[]): number {
  return teacher.outputBias + teacher.atoms.reduce((sum, atom) => (
    sum + atom.outputWeight * gelu(atom.inputScale * dot(atom.direction, input) + atom.inputBias)
  ), 0);
}

function makeNoisyOracle(teacher: TeacherCase, seedOffset: number): Predictor {
  const random = makeRandom(teacher.randomSeed + seedOffset);
  return input => {
    const noisy = predictTeacher(teacher, input) + gaussian(random) * teacher.noiseStd;
    return Math.round(noisy / teacher.quantization) * teacher.quantization;
  };
}

function aggregate(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}

function queryRepeated(oracle: Predictor, input: number[], count: number): number {
  return aggregate(Array.from({ length: count }, () => oracle(input)));
}

function bestDirectionProjection(trueDirection: number[], recoveredWeights: number[][]): number {
  return Math.max(0, ...recoveredWeights.map(weight => Math.abs(dot(trueDirection, normalize(weight)))));
}

function worstAbsoluteError(actual: number[], predicted: number[]): number {
  return Math.max(0, ...actual.map((value, index) => Math.abs(value - predicted[index])));
}

function solveSystem(matrix: number[][], rhs: number[]): number[] {
  const n = rhs.length;
  const augmented = matrix.map((row, index) => [...row, rhs[index]]);
  for (let column = 0; column < n; column++) {
    let pivot = column;
    for (let row = column + 1; row < n; row++) {
      if (Math.abs(augmented[row][column]) > Math.abs(augmented[pivot][column])) pivot = row;
    }
    [augmented[column], augmented[pivot]] = [augmented[pivot], augmented[column]];
    const divisor = augmented[column][column];
    if (Math.abs(divisor) < 1e-12) throw new Error('baseline linear system is singular');
    for (let index = column; index <= n; index++) augmented[column][index] /= divisor;
    for (let row = 0; row < n; row++) {
      if (row === column) continue;
      const factor = augmented[row][column];
      if (Math.abs(factor) < 1e-20) continue;
      for (let index = column; index <= n; index++) augmented[row][index] -= factor * augmented[column][index];
    }
  }
  return augmented.map(row => row[n]);
}

function trainRidge(
  inputs: number[][],
  targets: number[],
  feature: (input: number[]) => number[],
  ridge: number
): Predictor {
  const width = feature(inputs[0]).length;
  const xtx = Array.from({ length: width }, () => new Array(width).fill(0));
  const xty = new Array(width).fill(0);
  for (let rowIndex = 0; rowIndex < inputs.length; rowIndex++) {
    const row = feature(inputs[rowIndex]);
    for (let i = 0; i < width; i++) {
      xty[i] += row[i] * targets[rowIndex];
      for (let j = 0; j < width; j++) xtx[i][j] += row[i] * row[j];
    }
  }
  for (let index = 0; index < width; index++) xtx[index][index] += ridge;
  const weights = solveSystem(xtx, xty);
  return input => dot(weights, feature(input));
}

function makeQuadraticFeatures(input: number[]): number[] {
  return [1, ...input, ...input.map(value => value ** 2)];
}

function makeRandomGeluFeature(seed: number, dimension: number, count: number): (input: number[]) => number[] {
  const random = makeRandom(seed);
  const directions = Array.from({ length: count }, () => (
    normalize(Array.from({ length: dimension }, () => gaussian(random)))
  ));
  const biases = Array.from({ length: count }, () => -0.6 + 1.2 * random());
  return input => [
    1,
    ...input,
    ...directions.map((direction, index) => gelu(dot(direction, input) + biases[index]))
  ];
}

function probeStrongBaselinePythonEnv(pythonExecutable: string): { ok: boolean; message: string } {
  const result = spawnSync(pythonExecutable, ['-c', [
    'import importlib',
    `required = ${JSON.stringify(requiredStrongBaselineModules)}`,
    'missing = []',
    'versions = {}',
    'for name in required:',
    '    try:',
    '        module = importlib.import_module(name)',
    "        versions[name] = getattr(module, '__version__', 'unknown')",
    '    except Exception as exc:',
    "        missing.append(f'{name}: {exc!r}')",
    'if missing:',
    "    raise SystemExit('\\n'.join(missing))",
    "print('neural-baseline-python-ok ' + ' '.join(f'{k}={v}' for k, v in sorted(versions.items())))"
  ].join('\n')], {
    cwd: repoRoot,
    encoding: 'utf8',
    timeout: 30 * 1000,
    maxBuffer: 2 * 1024 * 1024
  });

  if (result.status === 0) return { ok: true, message: (result.stdout || '').trim() };
  return {
    ok: false,
    message: [
      (result.stderr || '').trim(),
      (result.stdout || '').trim(),
      result.error ? String(result.error) : ''
    ].filter(Boolean).join('\n')
  };
}

function bootstrapStrongBaselinePythonEnv(venvPath: string, venvPython: string, reason: string): void {
  if (!fs.existsSync(strongBaselineRequirementsPath)) {
    throw new Error(`neural baseline Python requirements file is missing: ${strongBaselineRequirementsPath}`);
  }
  console.log(`[nn-gelu-baseline-court] preparing managed Python baseline env at ${venvPath}`);
  if (reason) console.log(`[nn-gelu-baseline-court] reason: ${reason}`);

  if (!fs.existsSync(venvPython)) {
    const create = spawnSync('python3', ['-m', 'venv', '--system-site-packages', venvPath], {
      cwd: repoRoot,
      encoding: 'utf8',
      timeout: 2 * 60 * 1000,
      maxBuffer: 8 * 1024 * 1024
    });
    if (create.stdout) process.stdout.write(create.stdout);
    if (create.stderr) process.stderr.write(create.stderr);
    if (create.error) throw create.error;
    if (create.status !== 0) throw new Error(`could not create neural baseline venv at ${venvPath}`);
  }

  const install = spawnSync(venvPython, [
    '-m',
    'pip',
    'install',
    '--disable-pip-version-check',
    '--quiet',
    '-r',
    strongBaselineRequirementsPath
  ], {
    cwd: repoRoot,
    encoding: 'utf8',
    timeout: 8 * 60 * 1000,
    maxBuffer: 128 * 1024 * 1024
  });
  if (install.stdout) process.stdout.write(install.stdout);
  if (install.stderr) process.stderr.write(install.stderr);
  if (install.error) throw install.error;
  if (install.status !== 0) throw new Error(`could not install neural baseline requirements into ${venvPath}`);

  const finalProbe = probeStrongBaselinePythonEnv(venvPython);
  if (!finalProbe.ok) throw new Error(`managed neural baseline env is still invalid after install:\n${finalProbe.message}`);
  console.log(`[nn-gelu-baseline-court] ${finalProbe.message}`);
}

function resolveStrongBaselinePythonBin(): string {
  if (cachedStrongBaselinePythonBin) return cachedStrongBaselinePythonBin;
  const configuredPython = process.env.GCL_NEURAL_BASELINE_PYTHON;
  if (configuredPython) {
    const probe = probeStrongBaselinePythonEnv(configuredPython);
    if (!probe.ok) {
      throw new Error([
        `GCL_NEURAL_BASELINE_PYTHON is missing required neural baseline modules: ${configuredPython}`,
        probe.message,
        'Install the requirements in that environment or unset GCL_NEURAL_BASELINE_PYTHON so GCL can use its managed benchmark venv.'
      ].filter(Boolean).join('\n'));
    }
    cachedStrongBaselinePythonBin = configuredPython;
    return cachedStrongBaselinePythonBin;
  }

  const venvPython = path.join(strongBaselineVenvPath, 'bin', 'python');
  const probe = fs.existsSync(venvPython)
    ? probeStrongBaselinePythonEnv(venvPython)
    : { ok: false, message: 'managed neural baseline venv does not exist' };
  if (!probe.ok) bootstrapStrongBaselinePythonEnv(strongBaselineVenvPath, venvPython, probe.message);
  cachedStrongBaselinePythonBin = venvPython;
  return cachedStrongBaselinePythonBin;
}

const sklearnBaselineProgram = String.raw`
import json
import math
import sys

try:
    import numpy as np
    from xgboost import XGBRegressor
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
except Exception as exc:
    print(json.dumps({"ok": False, "error": f"missing strong baseline package: {type(exc).__name__}: {exc}"}))
    sys.exit(2)

payload = json.loads(sys.stdin.read())
X_train = np.asarray(payload["trainInputs"], dtype=float)
y_train = np.asarray(payload["trainTargets"], dtype=float)
evaluation_sets = payload["evaluationSets"]
seed = int(payload["seed"])
raw_calls = int(payload["trainingRawOracleCalls"])

def score(y_true, pred):
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    pred = np.asarray(pred, dtype=float).reshape(-1)
    diff = y_true - pred
    mae = float(np.mean(np.abs(diff)))
    rmse = float(math.sqrt(np.mean(diff * diff)))
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - float(np.sum(diff * diff)) / denom if denom > 0 else 1.0
    worst = float(np.max(np.abs(diff))) if len(diff) else 0.0
    return {"mae": mae, "rmse": rmse, "r2": r2}, worst

def eval_model(model):
    evaluations = []
    for item in evaluation_sets:
        X_eval = np.asarray(item["inputs"], dtype=float)
        y_eval = np.asarray(item["actual"], dtype=float)
        metrics, worst = score(y_eval, model.predict(X_eval))
        evaluations.append({
            "id": item["id"],
            "kind": item["kind"],
            "metrics": metrics,
            "worstError": worst
        })
    return evaluations

models = [
    (
        "sklearn-ridge",
        "sklearn linear ridge",
        make_pipeline(StandardScaler(), Ridge(alpha=1e-4))
    ),
    (
        "sklearn-poly2-ridge",
        "sklearn polynomial ridge",
        make_pipeline(PolynomialFeatures(degree=2, include_bias=True), StandardScaler(with_mean=False), Ridge(alpha=1e-3))
    ),
    (
        "sklearn-random-forest",
        "sklearn random forest",
        RandomForestRegressor(n_estimators=140, max_depth=12, min_samples_leaf=2, random_state=seed, n_jobs=1)
    ),
    (
        "sklearn-extra-trees",
        "sklearn extra trees",
        ExtraTreesRegressor(n_estimators=180, max_depth=14, min_samples_leaf=2, random_state=seed + 11, n_jobs=1)
    ),
    (
        "sklearn-hist-gradient-boosting",
        "sklearn histogram gradient boosting",
        HistGradientBoostingRegressor(max_iter=180, max_leaf_nodes=31, learning_rate=0.055, l2_regularization=1e-4, random_state=seed + 23)
    ),
    (
        "xgboost-hist",
        "XGBoost histogram gradient trees",
        XGBRegressor(
            objective="reg:squarederror",
            n_estimators=220,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=seed + 29,
            n_jobs=1,
            tree_method="hist",
            verbosity=0
        )
    ),
    (
        "sklearn-mlp-regressor",
        "sklearn neural network MLP regressor",
        make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(96, 64),
                activation="relu",
                solver="adam",
                alpha=1e-5,
                learning_rate_init=0.002,
                max_iter=350,
                tol=1e-5,
                random_state=seed + 31
            )
        )
    )
]

rows = []
for ident, family, model in models:
    model.fit(X_train, y_train)
    evaluations = eval_model(model)
    primary = next(item for item in evaluations if item["id"] == "iid-uniform")
    package = "XGBoost" if ident.startswith("xgboost") else "scikit-learn"
    rows.append({
        "id": ident,
        "family": family,
        "metrics": primary["metrics"],
        "worstError": primary["worstError"],
        "evaluations": evaluations,
        "trainingRawOracleCalls": raw_calls,
        "package": package
    })

component_count = max(1, min(4, X_train.shape[1], X_train.shape[0] - 1))
pls = PLSRegression(n_components=component_count)
Z_train = pls.fit_transform(X_train, y_train)[0]
pls_eval_sets = []
active_model = make_pipeline(PolynomialFeatures(degree=3, include_bias=True), StandardScaler(with_mean=False), Ridge(alpha=1e-4))
active_model.fit(Z_train, y_train)
for item in evaluation_sets:
    X_eval = np.asarray(item["inputs"], dtype=float)
    y_eval = np.asarray(item["actual"], dtype=float)
    Z_eval = pls.transform(X_eval)
    metrics, worst = score(y_eval, active_model.predict(Z_eval))
    pls_eval_sets.append({
        "id": item["id"],
        "kind": item["kind"],
        "metrics": metrics,
        "worstError": worst
    })
primary = next(item for item in pls_eval_sets if item["id"] == "iid-uniform")
rows.append({
    "id": "sklearn-pls-active-subspace-poly-ridge",
    "family": "sklearn active-subspace surrogate",
    "metrics": primary["metrics"],
    "worstError": primary["worstError"],
    "evaluations": pls_eval_sets,
    "trainingRawOracleCalls": raw_calls,
    "package": "scikit-learn"
})

print(json.dumps({"ok": True, "rows": rows}))
`;

function makeTeacherCases(): TeacherCase[] {
  const balancedDirections = makeDirections(24, 3, 300101);
  const nearBase = normalize(Array.from({ length: 16 }, (_, index) => Math.sin(index * 1.7 + 0.2)));
  const nearPerturbation = normalize(subtractProjection(
    Array.from({ length: 16 }, (_, index) => Math.cos(index * 2.1 - 0.4)),
    [nearBase]
  ));
  const nearThird = makeDirections(16, 1, 300202)[0];
  const nearSecond = normalize(nearBase.map((value, index) => value + 0.018 * nearPerturbation[index]));
  const weakDirections = makeDirections(12, 3, 300303);

  return [
    {
      id: 'balanced-three-channel-noisy-24d',
      profile: 'balanced-noisy',
      dimension: 24,
      atoms: [
        { direction: balancedDirections[0], inputScale: 1.18, inputBias: -0.22, outputWeight: 0.82 },
        { direction: balancedDirections[1], inputScale: 0.96, inputBias: 0.18, outputWeight: -0.64 },
        { direction: balancedDirections[2], inputScale: 0.88, inputBias: 0.34, outputWeight: 0.48 }
      ],
      outputBias: 0.04,
      noiseStd: 0.0025,
      quantization: 0.001,
      querySamples: 15,
      randomSeed: 301001,
      ranges: Array.from({ length: 24 }, () => [-1.5, 1.5] as [number, number]),
      solver: {
        gradientProbeCount: 22,
        trainingSampleCount: 1450,
        validationSampleCount: 420,
        restarts: 8,
        iterations: 540,
        learningRate: 0.03,
        atomRelativeStrengthThreshold: 0.025,
        atomAbsoluteStrengthThreshold: 0.004,
        pruneValidationToleranceRatio: 0.22,
        pruneValidationToleranceAbsolute: 0.014,
        nearParallelDirectionCosine: 0.999,
        duplicateDirectionCosine: 0.998,
        duplicateBiasTolerance: 0.12
      }
    },
    {
      id: 'near-parallel-three-channel-noisy-16d',
      profile: 'near-parallel-noisy',
      dimension: 16,
      atoms: [
        { direction: nearBase, inputScale: 1.16, inputBias: -0.36, outputWeight: 0.74 },
        { direction: nearSecond, inputScale: 1.02, inputBias: 0.26, outputWeight: -0.58 },
        { direction: nearThird, inputScale: 0.9, inputBias: 0.28, outputWeight: 0.5 }
      ],
      outputBias: -0.03,
      noiseStd: 0.002,
      quantization: 0.001,
      querySamples: 11,
      randomSeed: 302001,
      ranges: Array.from({ length: 16 }, () => [-1.5, 1.5] as [number, number]),
      solver: {
        gradientProbeCount: 28,
        trainingSampleCount: 1900,
        validationSampleCount: 550,
        restarts: 12,
        iterations: 900,
        learningRate: 0.028,
        atomRelativeStrengthThreshold: 0.02,
        atomAbsoluteStrengthThreshold: 0.004,
        pruneValidationToleranceRatio: 0.45,
        pruneValidationToleranceAbsolute: 0.02,
        nearParallelDirectionCosine: 0.999,
        duplicateDirectionCosine: 0.9994,
        duplicateBiasTolerance: 0.16
      }
    },
    {
      id: 'weak-resolvable-three-channel-noisy-12d',
      profile: 'weak-resolvable-noisy',
      dimension: 12,
      atoms: [
        { direction: weakDirections[0], inputScale: 1.12, inputBias: -0.2, outputWeight: 0.78 },
        { direction: weakDirections[1], inputScale: 0.94, inputBias: 0.24, outputWeight: -0.57 },
        { direction: weakDirections[2], inputScale: 1.08, inputBias: -0.08, outputWeight: 0.09 }
      ],
      outputBias: 0.025,
      noiseStd: 0.003,
      quantization: 0.001,
      querySamples: 13,
      randomSeed: 303001,
      ranges: Array.from({ length: 12 }, () => [-1.5, 1.5] as [number, number]),
      solver: {
        gradientProbeCount: 32,
        trainingSampleCount: 2300,
        validationSampleCount: 680,
        restarts: 14,
        iterations: 1050,
        learningRate: 0.028,
        atomRelativeStrengthThreshold: 0.015,
        atomAbsoluteStrengthThreshold: 0.004,
        pruneValidationToleranceRatio: 0.2,
        pruneValidationToleranceAbsolute: 0.01,
        nearParallelDirectionCosine: 0.999,
        duplicateDirectionCosine: 0.998,
        duplicateBiasTolerance: 0.12
      }
    }
  ];
}

function makeEvaluationSets(teacher: TeacherCase): EvaluationSet[] {
  const specs: Array<{ id: EvaluationSet['id']; kind: EvaluationSet['kind']; inputs: number[][] }> = [
    {
      id: 'iid-uniform',
      kind: 'iid',
      inputs: makeInputs(teacher.randomSeed + 501, teacher.dimension, 850)
    },
    {
      id: 'wide-ood',
      kind: 'ood',
      inputs: makeInputs(teacher.randomSeed + 502, teacher.dimension, 850, [-2.05, 2.05])
    },
    {
      id: 'shifted-ood',
      kind: 'ood',
      inputs: makeShiftedOodInputs(teacher.randomSeed + 503, teacher.dimension, 850)
    }
  ];
  return specs.map(spec => ({
    ...spec,
    actual: spec.inputs.map(input => predictTeacher(teacher, input))
  }));
}

function scorePredictorOnEvaluations(predict: Predictor, evaluationSets: EvaluationSet[]): EvaluationMetrics[] {
  return evaluationSets.map(evaluation => {
    const predicted = evaluation.inputs.map(predict);
    return {
      id: evaluation.id,
      kind: evaluation.kind,
      metrics: regressionMetrics(evaluation.actual, predicted),
      worstError: worstAbsoluteError(evaluation.actual, predicted)
    };
  });
}

function runPredictionBaselines(teacher: TeacherCase, evaluationSets: EvaluationSet[]) {
  const trainInputs = makeInputs(teacher.randomSeed + 401, teacher.dimension, 1250);
  const noisyOracle = makeNoisyOracle(teacher, 402);
  let rawOracleCalls = 0;
  const trainTargets = trainInputs.map(input => {
    rawOracleCalls += teacher.querySamples;
    return queryRepeated(noisyOracle, input, teacher.querySamples);
  });
  const mean = trainTargets.reduce((sum, value) => sum + value, 0) / trainTargets.length;
  const baselines = [
    { id: 'mean-output', family: 'constant', predict: (() => mean) as Predictor, trainingRawOracleCalls: 0 },
    {
      id: 'linear-ridge',
      family: 'linear ridge',
      predict: trainRidge(trainInputs, trainTargets, input => [1, ...input], 1e-4),
      trainingRawOracleCalls: rawOracleCalls
    },
    {
      id: 'quadratic-ridge',
      family: 'linear + squared coordinates ridge',
      predict: trainRidge(trainInputs, trainTargets, makeQuadraticFeatures, 1e-3),
      trainingRawOracleCalls: rawOracleCalls
    },
    {
      id: 'random-gelu-ridge',
      family: 'random GELU feature ridge surrogate',
      predict: trainRidge(
        trainInputs,
        trainTargets,
        makeRandomGeluFeature(teacher.randomSeed + 403, teacher.dimension, 48),
        1e-3
      ),
      trainingRawOracleCalls: rawOracleCalls
    }
  ];
  const rows: PredictionBaselineRow[] = baselines.map(baseline => {
    const evaluations = scorePredictorOnEvaluations(baseline.predict, evaluationSets);
    const primary = evaluations.find(evaluation => evaluation.id === 'iid-uniform') ?? evaluations[0];
    return {
      id: baseline.id,
      family: baseline.family,
      metrics: primary.metrics,
      worstError: primary.worstError,
      evaluations,
      trainingRawOracleCalls: baseline.trainingRawOracleCalls
    };
  });
  const pythonBin = resolveStrongBaselinePythonBin();
  const sklearn = spawnSync(pythonBin, ['-c', sklearnBaselineProgram], {
    input: JSON.stringify({
      trainInputs,
      trainTargets,
      evaluationSets: evaluationSets.map(evaluation => ({
        id: evaluation.id,
        kind: evaluation.kind,
        inputs: evaluation.inputs,
        actual: evaluation.actual
      })),
      seed: teacher.randomSeed + 404,
      trainingRawOracleCalls: rawOracleCalls
    }),
    encoding: 'utf8',
    maxBuffer: 64 * 1024 * 1024
  });
  if (sklearn.status !== 0) {
    throw new Error(`sklearn prediction baseline court failed for ${teacher.id}: ${sklearn.stderr || sklearn.stdout}`);
  }
  const parsed = JSON.parse(sklearn.stdout);
  if (!parsed.ok) throw new Error(`sklearn prediction baseline court failed for ${teacher.id}: ${parsed.error}`);
  rows.push(...parsed.rows);
  return rows;
}

fs.mkdirSync(output.outputDir, { recursive: true });
const started = Date.now();
const rows = makeTeacherCases().map((teacher, index) => {
  const noisyOracle = makeNoisyOracle(teacher, 101);
  const model = decompileRankAdaptiveGeluFfnStructure(noisyOracle, {
    dimension: teacher.dimension,
    ranges: teacher.ranges,
    maxChannelCount: 4,
    minEffectiveChannelCount: 2,
    selectionRestarts: 1,
    gradientProbeCount: teacher.solver.gradientProbeCount,
    trainingSampleCount: teacher.solver.trainingSampleCount,
    validationSampleCount: teacher.solver.validationSampleCount,
    randomSeed: teacher.randomSeed + 201,
    restarts: teacher.solver.restarts,
    iterations: teacher.solver.iterations,
    learningRate: teacher.solver.learningRate,
    querySamples: teacher.querySamples,
    aggregation: 'median',
    atomRelativeStrengthThreshold: teacher.solver.atomRelativeStrengthThreshold,
    atomAbsoluteStrengthThreshold: teacher.solver.atomAbsoluteStrengthThreshold,
    pruneValidationToleranceRatio: teacher.solver.pruneValidationToleranceRatio,
    pruneValidationToleranceAbsolute: teacher.solver.pruneValidationToleranceAbsolute,
    nearParallelDirectionCosine: teacher.solver.nearParallelDirectionCosine,
    duplicateDirectionCosine: teacher.solver.duplicateDirectionCosine,
    duplicateBiasTolerance: teacher.solver.duplicateBiasTolerance
  });
  const evaluationSets = makeEvaluationSets(teacher);
  const iidEvaluation = evaluationSets.find(evaluation => evaluation.id === 'iid-uniform') ?? evaluationSets[0];
  const actual = iidEvaluation.actual;
  const predicted = predictGeluFfnStructureBatch(model, iidEvaluation.inputs);
  const metrics = regressionMetrics(actual, predicted);
  const worstError = worstAbsoluteError(actual, predicted);
  const gclEvaluations = evaluationSets.map(evaluation => {
    const evaluationPredicted = predictGeluFfnStructureBatch(model, evaluation.inputs);
    return {
      id: evaluation.id,
      kind: evaluation.kind,
      metrics: regressionMetrics(evaluation.actual, evaluationPredicted),
      worstError: worstAbsoluteError(evaluation.actual, evaluationPredicted)
    };
  });
  const signature = canonicalizeGeluFfnStructure(model, {
    atomRelativeStrengthThreshold: teacher.profile === 'weak-resolvable-noisy' ? 0.012 : 0.018,
    atomAbsoluteStrengthThreshold: teacher.profile === 'weak-resolvable-noisy' ? 0.003 : 0.004,
    duplicateDirectionCosine: teacher.solver.duplicateDirectionCosine,
    duplicateBiasTolerance: teacher.solver.duplicateBiasTolerance
  });
  const recoveredWeights = model.atoms.map(atom => atom.inputWeight);
  const projectionScores = teacher.atoms.map(atom => bestDirectionProjection(atom.direction, recoveredWeights));
  const minDirectionProjection = Math.min(...projectionScores);
  const predictionBaselines = runPredictionBaselines(teacher, evaluationSets);
  const bestPredictionBaseline = predictionBaselines.reduce((best, baseline) => (
    baseline.metrics.mae < best.metrics.mae ? baseline : best
  ), predictionBaselines[0]);
  const distributionCourt = gclEvaluations.map(evaluation => {
    const predictionBaselineEvaluations = predictionBaselines.map(baseline => {
      const baselineEvaluation = baseline.evaluations.find(item => item.id === evaluation.id);
      if (!baselineEvaluation) throw new Error(`prediction baseline ${baseline.id} missing evaluation ${evaluation.id}`);
      return {
        id: baseline.id,
        family: baseline.family,
        package: baseline.package,
        metrics: baselineEvaluation.metrics,
        worstError: baselineEvaluation.worstError
      };
    });
    const best = predictionBaselineEvaluations.reduce((candidateBest, baseline) => (
      baseline.metrics.mae < candidateBest.metrics.mae ? baseline : candidateBest
    ), predictionBaselineEvaluations[0]);
    return {
      id: evaluation.id,
      kind: evaluation.kind,
      gcl: evaluation,
      bestPredictionBaseline: best,
      gclVsBestPredictionBaselineMaeRatio: evaluation.metrics.mae / Math.max(1e-12, best.metrics.mae),
      beatsBestPredictionBaselineByMae: evaluation.metrics.mae < best.metrics.mae
    };
  });
  const strongOnlyPredict = (input: number[]): number => teacher.outputBias + teacher.atoms.slice(0, 2).reduce((sum, atom) => (
    sum + atom.outputWeight * gelu(atom.inputScale * dot(atom.direction, input) + atom.inputBias)
  ), 0);
  const strongOnlyMetrics = regressionMetrics(actual, iidEvaluation.inputs.map(strongOnlyPredict));
  const weakSignalResolvable = teacher.profile !== 'weak-resolvable-noisy'
    || strongOnlyMetrics.mae > Math.max(0.015, metrics.mae * 2.25);
  const maxOodMae = Math.max(...distributionCourt.filter(item => item.kind === 'ood').map(item => item.gcl.metrics.mae));
  const minOodR2 = Math.min(...distributionCourt.filter(item => item.kind === 'ood').map(item => item.gcl.metrics.r2));
  const maxOodWorstError = Math.max(...distributionCourt.filter(item => item.kind === 'ood').map(item => item.gcl.worstError));
  const gates = {
    outputR2AboveThreshold: metrics.r2 > (teacher.profile === 'weak-resolvable-noisy' ? 0.997 : 0.9985),
    outputMaeBelowThreshold: metrics.mae < (teacher.profile === 'weak-resolvable-noisy' ? 0.022 : 0.018),
    worstErrorBelowThreshold: worstError < (teacher.profile === 'weak-resolvable-noisy' ? 0.105 : 0.085),
    oodR2AboveThreshold: minOodR2 > (teacher.profile === 'weak-resolvable-noisy' ? 0.994 : 0.996),
    oodMaeBelowThreshold: maxOodMae < (teacher.profile === 'weak-resolvable-noisy' ? 0.035 : 0.028),
    oodWorstErrorBelowThreshold: maxOodWorstError < (teacher.profile === 'weak-resolvable-noisy' ? 0.19 : 0.15),
    effectiveChannelCountMatches: model.effectiveChannelCount === teacher.atoms.length,
    canonicalAtomCountMatches: signature.canonicalAtomCount === teacher.atoms.length,
    directionRecoveryAboveThreshold: minDirectionProjection > (teacher.profile === 'near-parallel-noisy' ? 0.985 : 0.992),
    noDuplicateCollapse: (model.adaptiveSelection?.duplicatePairCount ?? 0) === 0,
    weakResolvableSignalNotPruned: weakSignalResolvable && (teacher.profile !== 'weak-resolvable-noisy' || (model.effectiveChannelCount ?? 0) === 3),
    beatsBestPredictionBaselineByMae: metrics.mae < bestPredictionBaseline.metrics.mae,
    beatsBestPredictionBaselineOnEveryDistribution: distributionCourt.every(item => item.beatsBestPredictionBaselineByMae),
    packageBackedPredictionBaselinesCovered: predictionBaselines.some(baseline => baseline.package === 'scikit-learn'),
    xgboostPredictionBaselineCovered: predictionBaselines.some(baseline => baseline.package === 'XGBoost'),
    nnPredictionBaselineCovered: predictionBaselines.some(baseline => baseline.id === 'sklearn-mlp-regressor'),
    activeSubspacePredictionBaselineCovered: predictionBaselines.some(baseline => baseline.id === 'sklearn-pls-active-subspace-poly-ridge'),
    nonlinearPackagePredictionBaselinesCovered: predictionBaselines.some(baseline => baseline.id === 'sklearn-extra-trees')
      && predictionBaselines.some(baseline => baseline.id === 'sklearn-random-forest')
      && predictionBaselines.some(baseline => baseline.id === 'sklearn-hist-gradient-boosting')
      && predictionBaselines.some(baseline => baseline.id === 'xgboost-hist')
      && predictionBaselines.some(baseline => baseline.id === 'sklearn-mlp-regressor'),
    rawOracleCallsBelowBudget: model.queryBudget.rawOracleCalls < 220000
  };
  return {
    id: teacher.id,
    profile: teacher.profile,
    teacher: {
      dimension: teacher.dimension,
      trueChannelCount: teacher.atoms.length,
      noiseStd: teacher.noiseStd,
      quantization: teacher.quantization,
      querySamples: teacher.querySamples
    },
    summary: {
      pass: Object.values(gates).every(Boolean),
      mae: metrics.mae,
      rmse: metrics.rmse,
      r2: metrics.r2,
      worstError,
      effectiveChannelCount: model.effectiveChannelCount,
      canonicalAtomCount: signature.canonicalAtomCount,
      minDirectionProjection,
      duplicatePairCount: model.adaptiveSelection?.duplicatePairCount ?? 0,
      nearParallelPairCount: model.adaptiveSelection?.nearParallelPairCount ?? 0,
      prunedWeakAtomCount: model.adaptiveSelection?.prunedWeakAtomCount ?? 0,
      bestPredictionBaselineId: bestPredictionBaseline.id,
      bestPredictionBaselineMae: bestPredictionBaseline.metrics.mae,
      gclVsBestPredictionBaselineMaeRatio: metrics.mae / Math.max(1e-12, bestPredictionBaseline.metrics.mae),
      maxDistributionMae: Math.max(...distributionCourt.map(item => item.gcl.metrics.mae)),
      minDistributionR2: Math.min(...distributionCourt.map(item => item.gcl.metrics.r2)),
      maxDistributionWorstError: Math.max(...distributionCourt.map(item => item.gcl.worstError)),
      maxDistributionGclVsBestPredictionBaselineMaeRatio: Math.max(...distributionCourt.map(item => item.gclVsBestPredictionBaselineMaeRatio)),
      worstDistributionId: distributionCourt.reduce((worst, item) => (
        item.gcl.metrics.mae > worst.gcl.metrics.mae ? item : worst
      ), distributionCourt[0]).id,
      strongOnlyMae: strongOnlyMetrics.mae,
      rawOracleCalls: model.queryBudget.rawOracleCalls
    },
    gates,
    predictionBaselines,
    distributionCourt,
    gcl: {
      metrics,
      worstError,
      evaluations: gclEvaluations,
      projectionScores,
      canonicalSignature: signature,
      adaptiveSelection: model.adaptiveSelection,
      queryBudget: model.queryBudget,
      expression: model.expression,
      atoms: model.atoms.map(atom => ({
        inputWeightNorm: atom.inputWeightNorm,
        inputBias: atom.inputBias,
        outputWeight: atom.outputWeight,
        strength: Math.abs(atom.outputWeight) * atom.inputWeightNorm
      }))
    }
  };
});

const pass = rows.every(row => row.summary.pass);
const report = {
  generatedAt: new Date().toISOString(),
  task: 'noisy quantized mixed GELU FFN structure recovery',
  blackBox: {
    visibilityToGcl: 'oracle calls only: f(vector); true channel count, directions, biases, readout weights, noise seed, and clean target are hidden from the solver',
    oracleNoise: 'stochastic Gaussian output noise plus finite-precision quantization; solver uses repeated median queries'
  },
  claimScope: {
    provesIfPasses: 'GCL can recover explicit mixed GELU atom structure under controlled stochastic noise and quantization, including balanced, near-parallel, and weak-but-resolvable channel profiles; the recovered structure is also prediction-sanity checked against dependency-free, scikit-learn, XGBoost, and sklearn-MLP fit baselines across IID and OOD held-out noiseless truth',
    doesNotClaim: 'full GPT-2 FFN extraction, arbitrary 3072-channel recovery, arbitrary 4096D/16384D production LLM block decompilation, noisy SwiGLU mixed-block recovery, PyTorch deep baseline superiority, XGBoost/MLP/RF as structure-decompiler competitors, semantic original-neuron identity from output-only access, or broad SOTA status'
  },
  predictionBaselineScope: {
    role: 'prediction-sanity only',
    purpose: 'checks that the recovered explicit GELU structure is not merely worse than strong black-box output fitting on held-out truth',
    notAClaim: 'XGBoost, random forests, and sklearn MLP are not structure decompilers and are not treated as direct rivals for recovered channel identity, directions, or canonical atoms'
  },
  config: {
    maxChannelBudgetGivenToGcl: 4,
    caseCount: rows.length
  },
  gates: {
    pass,
    structureRecoveryPass: rows.every(row => (
      row.gates.effectiveChannelCountMatches
      && row.gates.canonicalAtomCountMatches
      && row.gates.directionRecoveryAboveThreshold
      && row.gates.noDuplicateCollapse
      && row.gates.weakResolvableSignalNotPruned
    )),
    predictionSanityCourtPass: rows.every(row => (
      row.gates.outputR2AboveThreshold
      && row.gates.outputMaeBelowThreshold
      && row.gates.worstErrorBelowThreshold
      && row.gates.oodR2AboveThreshold
      && row.gates.oodMaeBelowThreshold
      && row.gates.oodWorstErrorBelowThreshold
      && row.gates.beatsBestPredictionBaselineByMae
      && row.gates.beatsBestPredictionBaselineOnEveryDistribution
      && row.gates.packageBackedPredictionBaselinesCovered
      && row.gates.xgboostPredictionBaselineCovered
      && row.gates.nnPredictionBaselineCovered
      && row.gates.activeSubspacePredictionBaselineCovered
      && row.gates.nonlinearPackagePredictionBaselinesCovered
    )),
    allCasesPassed: rows.every(row => row.summary.pass),
    allEffectiveCountsMatch: rows.every(row => row.gates.effectiveChannelCountMatches),
    allCanonicalCountsMatch: rows.every(row => row.gates.canonicalAtomCountMatches),
    allDirectionRecoveryAboveThreshold: rows.every(row => row.gates.directionRecoveryAboveThreshold),
    allBeatBestPredictionBaselineByMae: rows.every(row => row.gates.beatsBestPredictionBaselineByMae),
    allBeatBestPredictionBaselineOnEveryDistribution: rows.every(row => row.gates.beatsBestPredictionBaselineOnEveryDistribution),
    allPackageBackedPredictionBaselinesCovered: rows.every(row => row.gates.packageBackedPredictionBaselinesCovered),
    allXgboostPredictionBaselinesCovered: rows.every(row => row.gates.xgboostPredictionBaselineCovered),
    allNnPredictionBaselinesCovered: rows.every(row => row.gates.nnPredictionBaselineCovered),
    allActiveSubspacePredictionBaselinesCovered: rows.every(row => row.gates.activeSubspacePredictionBaselineCovered),
    allNonlinearPackagePredictionBaselinesCovered: rows.every(row => row.gates.nonlinearPackagePredictionBaselinesCovered),
    allRawOracleCallsBelowBudget: rows.every(row => row.gates.rawOracleCallsBelowBudget),
    coversBalancedNoise: rows.some(row => row.profile === 'balanced-noisy' && row.summary.pass),
    coversNearParallelNoise: rows.some(row => row.profile === 'near-parallel-noisy' && row.summary.pass),
    coversWeakResolvableNoise: rows.some(row => row.profile === 'weak-resolvable-noisy' && row.summary.pass),
    coversIidUniformDistribution: rows.every(row => row.distributionCourt.some(item => item.id === 'iid-uniform')),
    coversWideOodDistribution: rows.every(row => row.distributionCourt.some(item => item.id === 'wide-ood')),
    coversShiftedOodDistribution: rows.every(row => row.distributionCourt.some(item => item.id === 'shifted-ood'))
  },
  summary: {
    pass,
    caseCount: rows.length,
    passedCases: rows.filter(row => row.summary.pass).length,
    failedCases: rows.filter(row => !row.summary.pass).map(row => row.id),
    maxMae: Math.max(...rows.map(row => row.summary.mae)),
    minR2: Math.min(...rows.map(row => row.summary.r2)),
    maxWorstError: Math.max(...rows.map(row => row.summary.worstError)),
    minDirectionProjection: Math.min(...rows.map(row => row.summary.minDirectionProjection)),
    effectiveChannelCounts: rows.map(row => row.summary.effectiveChannelCount),
    canonicalAtomCounts: rows.map(row => row.summary.canonicalAtomCount),
    maxGclVsBestPredictionBaselineMaeRatio: Math.max(...rows.map(row => row.summary.gclVsBestPredictionBaselineMaeRatio)),
    maxDistributionMae: Math.max(...rows.map(row => row.summary.maxDistributionMae)),
    minDistributionR2: Math.min(...rows.map(row => row.summary.minDistributionR2)),
    maxDistributionWorstError: Math.max(...rows.map(row => row.summary.maxDistributionWorstError)),
    maxDistributionGclVsBestPredictionBaselineMaeRatio: Math.max(...rows.map(row => row.summary.maxDistributionGclVsBestPredictionBaselineMaeRatio)),
    distributionIds: ['iid-uniform', 'wide-ood', 'shifted-ood'],
    maxRawOracleCalls: Math.max(...rows.map(row => row.summary.rawOracleCalls))
  },
  rows,
  runtimeMs: Date.now() - started
};

const markdown = [
  '# Noisy Mixed GELU FFN Structure Benchmark',
  '',
  `Generated: ${report.generatedAt}`,
  '',
  '## Claim Scope',
  '',
  `- Proves if pass: ${report.claimScope.provesIfPasses}`,
  `- Does not claim: ${report.claimScope.doesNotClaim}`,
  '',
  '## Prediction Baseline Scope',
  '',
  `- Role: ${report.predictionBaselineScope.role}`,
  `- Purpose: ${report.predictionBaselineScope.purpose}`,
  `- Not a claim: ${report.predictionBaselineScope.notAClaim}`,
  '',
  '## Summary',
  '',
  `- Pass: ${pass}`,
  `- Cases: ${report.summary.caseCount}`,
  `- Passed cases: ${report.summary.passedCases}`,
  `- Failed cases: ${report.summary.failedCases.join(', ') || 'none'}`,
  `- Max MAE: ${report.summary.maxMae}`,
  `- Min R2: ${report.summary.minR2}`,
  `- Max worst error: ${report.summary.maxWorstError}`,
  `- Min direction projection: ${report.summary.minDirectionProjection}`,
  `- Effective channel counts: ${report.summary.effectiveChannelCounts.join(', ')}`,
  `- Canonical atom counts: ${report.summary.canonicalAtomCounts.join(', ')}`,
  `- Max GCL/best-prediction-baseline MAE ratio: ${report.summary.maxGclVsBestPredictionBaselineMaeRatio}`,
  `- Distribution ids: ${report.summary.distributionIds.join(', ')}`,
  `- Max distribution MAE: ${report.summary.maxDistributionMae}`,
  `- Min distribution R2: ${report.summary.minDistributionR2}`,
  `- Max distribution worst error: ${report.summary.maxDistributionWorstError}`,
  `- Max distribution GCL/best-prediction-baseline MAE ratio: ${report.summary.maxDistributionGclVsBestPredictionBaselineMaeRatio}`,
  `- Max raw oracle calls: ${report.summary.maxRawOracleCalls}`,
  '',
  '## Cases',
  '',
  '| Case | Profile | Pass | MAE | R2 | Canonical atoms | Best prediction baseline | GCL/prediction-baseline MAE |',
  '| --- | --- | --- | --- | --- | --- | --- | --- |',
  ...rows.map(row => (
    `| ${row.id} | ${row.profile} | ${row.summary.pass} | ${row.summary.mae} | ${row.summary.r2} | ${row.summary.canonicalAtomCount} | ${row.summary.bestPredictionBaselineId} | ${row.summary.gclVsBestPredictionBaselineMaeRatio} |`
  )),
  '',
  '## Distribution Prediction-Sanity Court',
  '',
  '| Case | Distribution | Kind | GCL MAE | GCL R2 | Best prediction baseline | Best prediction baseline MAE | GCL/prediction-baseline MAE |',
  '| --- | --- | --- | --- | --- | --- | --- | --- |',
  ...rows.flatMap(row => row.distributionCourt.map(item => (
    `| ${row.id} | ${item.id} | ${item.kind} | ${item.gcl.metrics.mae} | ${item.gcl.metrics.r2} | ${item.bestPredictionBaseline.id} | ${item.bestPredictionBaseline.metrics.mae} | ${item.gclVsBestPredictionBaselineMaeRatio} |`
  ))),
  '',
  '## Prediction Baseline Families',
  '',
  ...rows.slice(0, 1).flatMap(row => row.predictionBaselines.map(baseline => (
    `- ${baseline.id}: ${baseline.family}${baseline.package ? ` (${baseline.package})` : ' (local)'}`
  ))),
  '',
  '## Gate Details',
  '',
  ...rows.flatMap(row => [
    `### ${row.id}`,
    '',
    `- Profile: ${row.profile}`,
    `- Noise std: ${row.teacher.noiseStd}`,
    `- Quantization: ${row.teacher.quantization}`,
    `- Query samples: ${row.teacher.querySamples}`,
    `- Summary: \`${JSON.stringify(row.summary)}\``,
    ...Object.entries(row.gates).map(([key, value]) => `- ${key}: ${value}`),
    ''
  ]),
  '## Conclusion',
  '',
  pass
    ? 'The noisy mixed GELU structure gate passed. GCL recovered explicit atom structure under stochastic noise/quantization, and that recovered structure also beat the dependency-free, scikit-learn, XGBoost, and sklearn-MLP prediction-sanity baselines across IID and OOD held-out noiseless truth. These prediction baselines are not structure-decompiler competitors.'
    : 'The noisy mixed GELU structure gate failed. Do not promote noisy mixed GELU recovery until the failing case is repaired.'
].join('\n');

fs.writeFileSync(output.jsonOutputPath, `${JSON.stringify(report, null, 2)}\n`);
fs.writeFileSync(output.markdownOutputPath, `${markdown}\n`);
writeBenchmarkManifest(repoRoot, output, {
  lane: 'nn-gelu-ffn-noisy-mixed-structure',
  status: pass ? 'pass' : 'fail',
  summary: report.summary,
  notes: [
    'Release-gated noisy/quantized mixed GELU structure recovery family.',
    'Fails if output fidelity is good but effective/canonical atom structure is wrong.',
    'Includes IID, wide-OOD, and shifted-OOD evaluation distributions.',
    'Includes dependency-free, scikit-learn, XGBoost, and sklearn-MLP package-backed prediction-sanity baselines on held-out noiseless truth.',
    'Prediction baselines are not treated as structure-decompiler competitors.',
    'Keeps full GPT-2 FFN extraction, arbitrary 3072-channel recovery, and broad SOTA out of scope.'
  ]
});

console.log(JSON.stringify({
  pass,
  report: path.relative(repoRoot, output.jsonOutputPath),
  summary: report.summary,
  gates: report.gates
}, null, 2));

if (!pass) process.exit(1);
