import { parentPort } from 'worker_threads';
import { BlockEvaluator } from './block-evaluator';

if (!parentPort) {
  throw new Error('This file must be run as a worker thread.');
}
const port = parentPort;

type WorkerRequest = {
  startIndex: number;
  count: number;
  maxTestCases: number;
  geneBuffer: SharedArrayBuffer;
  constBuffer: SharedArrayBuffer;
  offsets: SharedArrayBuffer;
  lengths: SharedArrayBuffer;
  fitness: SharedArrayBuffer;
  caseErrors: SharedArrayBuffer;
  caseWeights: SharedArrayBuffer;
  inputs: SharedArrayBuffer;
  expected: SharedArrayBuffer;
  inputSize: number;
  fitnessMode: 'regression' | 'classification';
  classificationWeight: number;
};

port.on('message', (data: WorkerRequest) => {
  const buffer = {
    geneBuffer: new Uint32Array(data.geneBuffer),
    constBuffer: new Float64Array(data.constBuffer),
    offsets: new Uint32Array(data.offsets),
    lengths: new Uint32Array(data.lengths),
    fitness: new Float64Array(data.fitness),
    caseErrors: new Float64Array(data.caseErrors),
    caseWeights: new Float64Array(data.caseWeights),
    maxTestCases: data.maxTestCases
  };

  const inputs = new Float64Array(data.inputs);
  const expected = new Float64Array(data.expected);

  BlockEvaluator.evaluateBlock(
    buffer as any,
    data.startIndex,
    data.count,
    inputs,
    expected,
    data.inputSize,
    data.fitnessMode,
    data.classificationWeight
  );

  port.postMessage({ ok: true, startIndex: data.startIndex, count: data.count });
});
