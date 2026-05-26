import * as path from 'path';
import { PopulationBuffer } from './pop-buffer';
import { WorkerPool } from './worker-pool';

type SharedBuffers = {
  geneBuffer: Uint32Array;
  constBuffer: Float64Array;
  offsets: Uint32Array;
  lengths: Uint32Array;
  fitness: Float64Array;
  caseErrors: Float64Array;
  caseWeights: Float64Array;
  inputs: Float64Array;
  expected: Float64Array;
};

export class ParallelBlockEvaluator {
  private readonly workerCount: number;
  private readonly maxGenomes: number;
  private readonly maxNodesPerGenome: number;
  private readonly maxConstantsPerGenome: number;
  private readonly maxTestCases: number;
  private readonly pool: WorkerPool;
  private shared?: SharedBuffers;
  private sharedInputSize = 0;

  constructor(
    workerCount: number,
    maxGenomes: number,
    maxNodesPerGenome: number,
    maxTestCases: number,
    maxConstantsPerGenome: number = 10
  ) {
    this.workerCount = Math.max(1, workerCount);
    this.maxGenomes = maxGenomes;
    this.maxNodesPerGenome = maxNodesPerGenome;
    this.maxConstantsPerGenome = maxConstantsPerGenome;
    this.maxTestCases = maxTestCases;

    const isCompiled = __filename.endsWith('.js');
    const workerExt = isCompiled ? '.js' : '.ts';
    this.pool = new WorkerPool(path.join(__dirname, 'block-eval-worker' + workerExt), this.workerCount);
    this.pool.init();
  }

  private createSharedBuffers(inputSize: number): SharedBuffers {
    return {
      geneBuffer: new Uint32Array(new SharedArrayBuffer(Uint32Array.BYTES_PER_ELEMENT * this.maxGenomes * this.maxNodesPerGenome)),
      constBuffer: new Float64Array(new SharedArrayBuffer(Float64Array.BYTES_PER_ELEMENT * this.maxGenomes * this.maxConstantsPerGenome)),
      offsets: new Uint32Array(new SharedArrayBuffer(Uint32Array.BYTES_PER_ELEMENT * this.maxGenomes)),
      lengths: new Uint32Array(new SharedArrayBuffer(Uint32Array.BYTES_PER_ELEMENT * this.maxGenomes)),
      fitness: new Float64Array(new SharedArrayBuffer(Float64Array.BYTES_PER_ELEMENT * this.maxGenomes)),
      caseErrors: new Float64Array(new SharedArrayBuffer(Float64Array.BYTES_PER_ELEMENT * this.maxGenomes * this.maxTestCases)),
      caseWeights: new Float64Array(new SharedArrayBuffer(Float64Array.BYTES_PER_ELEMENT * this.maxTestCases)),
      inputs: new Float64Array(new SharedArrayBuffer(Float64Array.BYTES_PER_ELEMENT * this.maxTestCases * inputSize)),
      expected: new Float64Array(new SharedArrayBuffer(Float64Array.BYTES_PER_ELEMENT * this.maxTestCases))
    };
  }

  private ensureShared(inputSize: number): SharedBuffers {
    if (!this.shared || this.sharedInputSize !== inputSize) {
      this.shared = this.createSharedBuffers(inputSize);
      this.sharedInputSize = inputSize;
    }
    return this.shared;
  }

  public async evaluate(
    buffer: PopulationBuffer,
    startIndex: number,
    count: number,
    inputs: Float64Array,
    expected: Float64Array,
    inputSize: number,
    fitnessMode: 'regression' | 'classification',
    classificationWeight: number
  ): Promise<Float64Array> {
    if (count <= 0) return new Float64Array(0);
    if (count === 1 || this.workerCount <= 1) {
      const { BlockEvaluator } = await import('./block-evaluator');
      return BlockEvaluator.evaluateBlock(buffer, startIndex, count, inputs, expected, inputSize, fitnessMode, classificationWeight);
    }

    const shared = this.ensureShared(inputSize);
    shared.geneBuffer.set(buffer.geneBuffer);
    shared.constBuffer.set(buffer.constBuffer);
    shared.offsets.set(buffer.offsets);
    shared.lengths.set(buffer.lengths);
    shared.caseWeights.set(buffer.caseWeights);
    shared.inputs.set(inputs);
    shared.expected.set(expected);

    const batchSize = Math.ceil(count / this.workerCount);
    const jobs: Promise<unknown>[] = [];

    for (let workerIdx = 0; workerIdx < this.workerCount; workerIdx++) {
      const localStart = workerIdx * batchSize;
      const localCount = Math.min(batchSize, count - localStart);
      if (localCount <= 0) continue;

      jobs.push(this.pool.execute({
        geneBuffer: shared.geneBuffer.buffer,
        constBuffer: shared.constBuffer.buffer,
        offsets: shared.offsets.buffer,
        lengths: shared.lengths.buffer,
        fitness: shared.fitness.buffer,
        caseErrors: shared.caseErrors.buffer,
        caseWeights: shared.caseWeights.buffer,
        inputs: shared.inputs.buffer,
        expected: shared.expected.buffer,
        startIndex: startIndex + localStart,
        count: localCount,
        maxTestCases: buffer.maxTestCases,
        inputSize,
        fitnessMode,
        classificationWeight
      }));
    }

    await Promise.all(jobs);

    const resultStart = startIndex;
    const resultEnd = startIndex + count;
    buffer.fitness.set(shared.fitness.subarray(resultStart, resultEnd), resultStart);

    const caseStart = resultStart * buffer.maxTestCases;
    const caseEnd = resultEnd * buffer.maxTestCases;
    buffer.caseErrors.set(shared.caseErrors.subarray(caseStart, caseEnd), caseStart);

    return new Float64Array(shared.fitness.slice(resultStart, resultEnd));
  }

  public async shutdown(): Promise<void> {
    await this.pool.shutdown();
  }
}
