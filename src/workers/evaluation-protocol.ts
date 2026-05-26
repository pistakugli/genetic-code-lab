export type EvaluationFitnessMode = 'regression' | 'classification';

export interface SerializedTestCase {
  input: number[];
  expectedOutput: number;
  weight?: number;
}

/**
 * Read-only buffers that can be transferred to a worker.
 * Output/scratch arrays are allocated inside the worker so the payload stays minimal.
 */
export interface SerializedPopulationSnapshot {
  geneBuffer: ArrayBuffer;
  constBuffer: ArrayBuffer;
  offsets: ArrayBuffer;
  lengths: ArrayBuffer;
  caseWeights: ArrayBuffer;
  searchCaseWeights?: ArrayBuffer;
  maxGenomes: number;
  maxNodesPerGenome: number;
  maxConstantsPerGenome?: number;
  maxTestCases: number;
}

export interface PackedEvaluationBatch {
  population: SerializedPopulationSnapshot;
  testCases: SerializedTestCase[];
  startIndex: number;
  count: number;
  inputSize: number;
  solverType: string;
  fitnessMode: EvaluationFitnessMode;
  classificationWeight?: number;
}

export interface EvaluationRequestEnvelope {
  requestId: string;
  kind: 'evaluate';
  batch: PackedEvaluationBatch;
}

export interface EvaluationResultRow {
  id: number;
  fitness: number;
  caseErrors: number[];
}

export interface EvaluationResponseEnvelope {
  requestId: string;
  kind: 'evaluate-result';
  results: EvaluationResultRow[];
}

export interface EvaluationErrorEnvelope {
  requestId: string;
  kind: 'evaluate-error';
  error: {
    message: string;
    stack?: string;
  };
}

export type EvaluationEnvelope =
  | EvaluationRequestEnvelope
  | EvaluationResponseEnvelope
  | EvaluationErrorEnvelope;

export function createEvaluationRequestId(prefix: string = 'eval'): string {
  return `${prefix}_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 10)}`;
}

export function isEvaluationRequestEnvelope(message: unknown): message is EvaluationRequestEnvelope {
  if (!message || typeof message !== 'object') return false;
  const candidate = message as Partial<EvaluationRequestEnvelope>;
  return candidate.kind === 'evaluate'
    && typeof candidate.requestId === 'string'
    && typeof candidate.batch === 'object'
    && !!candidate.batch;
}

export function isEvaluationResponseEnvelope(message: unknown): message is EvaluationResponseEnvelope {
  if (!message || typeof message !== 'object') return false;
  const candidate = message as Partial<EvaluationResponseEnvelope>;
  return candidate.kind === 'evaluate-result'
    && typeof candidate.requestId === 'string'
    && Array.isArray(candidate.results);
}

export function isEvaluationErrorEnvelope(message: unknown): message is EvaluationErrorEnvelope {
  if (!message || typeof message !== 'object') return false;
  const candidate = message as Partial<EvaluationErrorEnvelope>;
  return candidate.kind === 'evaluate-error'
    && typeof candidate.requestId === 'string'
    && !!candidate.error
    && typeof candidate.error.message === 'string';
}
