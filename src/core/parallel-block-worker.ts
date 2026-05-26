import { parentPort } from 'worker_threads';
import { BlockEvaluator } from './block-evaluator';

if (!parentPort) {
  throw new Error('This file must be run as a worker thread.');
}
const port = parentPort;

type SharedEvalTask = {
  geneBuffer: SharedArrayBuffer;
  constBuffer: SharedArrayBuffer;
  offsets: SharedArrayBuffer;
  lengths: SharedArrayBuffer;
  fitness: SharedArrayBuffer;
  caseErrors: SharedArrayBuffer;
  caseWeights: SharedArrayBuffer;
  inputs: SharedArrayBuffer;
  expected: SharedArrayBuffer;
  startIndex: number;
  count: number;
  maxTestCases: number;
  inputSize: number;
  fitnessMode: 'regression' | 'classification';
  classificationWeight: number;
};

port.on('message', (task: SharedEvalTask) => {
  const buffer = {
    geneBuffer: new Uint32Array(task.geneBuffer),
    constBuffer: new Float64Array(task.constBuffer),
    offsets: new Uint32Array(task.offsets),
    lengths: new Uint32Array(task.lengths),
    fitness: new Float64Array(task.fitness),
    caseErrors: new Float64Array(task.caseErrors),
    caseWeights: new Float64Array(task.caseWeights),
    maxTestCases: task.maxTestCases
  };

  const inputs = new Float64Array(task.inputs);
  const expected = new Float64Array(task.expected);

  BlockEvaluator.evaluateBlock(
    buffer as any,
    task.startIndex,
    task.count,
    inputs,
    expected,
    task.inputSize,
    task.fitnessMode,
    task.classificationWeight
  );

  port.postMessage({ startIndex: task.startIndex, count: task.count });
});
