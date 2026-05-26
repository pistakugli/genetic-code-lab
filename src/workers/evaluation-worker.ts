import { parentPort } from 'worker_threads';
import { BlockEvaluator } from '../core/block-evaluator';
import { PopulationBuffer } from '../core/pop-buffer';
import {
  EvaluationErrorEnvelope,
  EvaluationRequestEnvelope,
  EvaluationResponseEnvelope,
  isEvaluationRequestEnvelope,
  SerializedPopulationSnapshot
} from './evaluation-protocol';

function rehydratePopulationBuffer(snapshot: SerializedPopulationSnapshot): PopulationBuffer {
  const buffer = new PopulationBuffer(
    snapshot.maxGenomes,
    snapshot.maxNodesPerGenome,
    snapshot.maxConstantsPerGenome ?? 10,
    snapshot.maxTestCases
  );

  buffer.geneBuffer = new Uint32Array(snapshot.geneBuffer);
  buffer.constBuffer = new Float64Array(snapshot.constBuffer);
  buffer.offsets = new Uint32Array(snapshot.offsets);
  buffer.lengths = new Uint32Array(snapshot.lengths);
  buffer.caseWeights = new Float64Array(snapshot.caseWeights);
  buffer.searchCaseWeights = new Float64Array(snapshot.searchCaseWeights ?? snapshot.caseWeights);
  buffer.fitness = new Float64Array(snapshot.maxGenomes).fill(-Infinity);
  buffer.searchFitness = new Float64Array(snapshot.maxGenomes).fill(-Infinity);
  buffer.caseErrors = new Float64Array(snapshot.maxGenomes * snapshot.maxTestCases).fill(0);
  buffer.ancestorMap = new Uint8Array(snapshot.maxGenomes).fill(0);
  buffer.nicheMap = new Uint32Array(1024).fill(0);

  return buffer;
}

function flattenTestCases(testCases: EvaluationRequestEnvelope['batch']['testCases'], inputSize: number): {
  inputs: Float64Array;
  expected: Float64Array;
} {
  const inputs = new Float64Array(testCases.length * inputSize);
  const expected = new Float64Array(testCases.length);

  for (let i = 0; i < testCases.length; i++) {
    const tc = testCases[i];
    expected[i] = tc.expectedOutput;
    for (let j = 0; j < inputSize; j++) {
      inputs[i * inputSize + j] = tc.input[j] ?? 0;
    }
  }

  return { inputs, expected };
}

function evaluateBatch(message: EvaluationRequestEnvelope): EvaluationResponseEnvelope {
  const { batch } = message;
  const buffer = rehydratePopulationBuffer(batch.population);
  const { inputs, expected } = flattenTestCases(batch.testCases, batch.inputSize);

  const results = BlockEvaluator.evaluateBlock(
    buffer,
    batch.startIndex,
    batch.count,
    inputs,
    expected,
    batch.inputSize,
    batch.fitnessMode,
    batch.classificationWeight ?? 0.8
  );

  const rows = Array.from({ length: batch.count }, (_, i) => {
    const genomeIdx = batch.startIndex + i;
    const errBase = genomeIdx * buffer.maxTestCases;
    const caseErrors = Array.from(buffer.caseErrors.subarray(errBase, errBase + batch.testCases.length));
    return {
      id: genomeIdx,
      fitness: results[i],
      caseErrors
    };
  });

  return {
    requestId: message.requestId,
    kind: 'evaluate-result',
    results: rows
  };
}

function toEvaluationError(message: EvaluationRequestEnvelope, error: unknown): EvaluationErrorEnvelope {
  const err = error instanceof Error ? error : new Error(String(error));
  return {
    requestId: message.requestId,
    kind: 'evaluate-error',
    error: {
      message: err.message,
      stack: err.stack
    }
  };
}

const port = parentPort;
if (port) {
  port.on('message', (message: unknown) => {
    if (!isEvaluationRequestEnvelope(message)) return;
    try {
      port.postMessage(evaluateBatch(message));
    } catch (error) {
      port.postMessage(toEvaluationError(message, error));
    }
  });
}

export { rehydratePopulationBuffer, flattenTestCases, evaluateBatch };
