import * as path from 'path';
import { Genome, TestCase, FunctionDefinition, TerminalDefinition } from './types';
import { WorkerPool } from './worker-pool';
import { FlatGenomeManager } from './flat-genome';

export class ParallelEvaluator {
  private workerCount: number;
  private pool: WorkerPool;

  constructor(workerCount: number) {
    this.workerCount = workerCount;
    const isCompiled = __filename.endsWith('.js');
    const workerExt = isCompiled ? '.js' : '.ts';
    this.pool = new WorkerPool(path.join(__dirname, 'worker' + workerExt), workerCount);
    this.pool.init();
  }

  /**
   * Evaluate a batch of genomes in parallel.
   */
  async evaluate(
    genomes: Genome[],
    testCases: TestCase<any, any>[],
    solverType: string,
    funcSet: FunctionDefinition[],
    termSet: TerminalDefinition[],
    multiObjective: boolean = false,
    lexicase: boolean = false
  ): Promise<Map<string | number, any>> {
    const results = new Map<string | number, any>();
    const funcNames = funcSet.map(f => f.name);
    const termNames = termSet.map(t => t.name);

    if (genomes.length > 0 && solverType === 'santa-fe') {
      const maxNodes = 150;
      const flatManager = new FlatGenomeManager(genomes.length, maxNodes);

      const behaviorBuffer = new SharedArrayBuffer(genomes.length * 1024);
      const behaviorView = new Uint8Array(behaviorBuffer);
      behaviorView.fill(0);

      const funcMap = new Map<string, number>();
      funcSet.forEach((f, idx) => funcMap.set(f.name, idx));
      const termMap = new Map<string, number>();
      termSet.forEach((t, idx) => termMap.set(t.name, idx));
      const funcSetLen = funcSet.length;

      for (let i = 0; i < genomes.length; i++) {
        flatManager.pack(genomes[i], i, maxNodes, funcMap, termMap, funcSetLen);
      }

      const sharedBuffer = flatManager.getSharedBuffer();

      // Sequentially evaluate each map in the ensemble
      for (const tc of testCases) {
        const sharedBufferView = new Uint8Array(sharedBuffer);
        // Map is passed as a 2D array, flatten it
        const flatMap = (tc.input as number[][]).reduce((acc, val) => acc.concat(val), [] as number[]);
        sharedBufferView.set(new Uint8Array(flatMap));

        const evaluationPromises: Promise<any>[] = [];
        const batchSize = Math.ceil(genomes.length / this.workerCount);

        for (let i = 0; i < this.workerCount; i++) {
          const startIdx = i * batchSize;
          const count = Math.min(batchSize, genomes.length - startIdx);
          if (count <= 0) continue;

          evaluationPromises.push(this.pool.execute({
            sharedBuffer,
            behaviorBuffer,
            batchInfo: {
              startIdx,
              count,
              maxNodes,
              nodeSize: FlatGenomeManager.NODE_SIZE
            },
            testCases: [tc],
            solverType,
            funcNames,
            termNames,
            lexicase,
            multiObjective
          }));
        }

        const batchResults = await Promise.all(evaluationPromises);
        
        for (const msg of batchResults) {
          for (const res of msg.results) {
            const originalGenome = genomes[res.id as number];
            if (!originalGenome) continue;
            
            const existing = results.get(originalGenome.id) || { fitness: 0, caseErrors: [] };
            existing.fitness += res.fitness / testCases.length;
            if (res.caseErrors) existing.caseErrors.push(...res.caseErrors);
            
            if (res.visitationMap) {
               if (!existing.visitationMap) existing.visitationMap = new Uint8Array(res.visitationMap.length);
               for (let j = 0; j < res.visitationMap.length; j++) if (res.visitationMap[j]) existing.visitationMap[j] = 1;
            }
            results.set(originalGenome.id, existing);
          }
        }
      }
      return results;
    } else if (genomes.length > 0 && (solverType === 'math' || solverType === 'math-classify' || solverType === 'math-quant')) {
      const maxNodes = 64; // Math trees are usually smaller
      const flatManager = new FlatGenomeManager(genomes.length, maxNodes);
      
      const funcMap = new Map<string, number>();
      funcSet.forEach((f, idx) => funcMap.set(f.name, idx));
      const termMap = new Map<string, number>();
      termSet.forEach((t, idx) => termMap.set(t.name, idx));
      const funcSetLen = funcSet.length;

      for (let i = 0; i < genomes.length; i++) {
        flatManager.pack(genomes[i], i, maxNodes, funcMap, termMap, funcSetLen);
      }

      const sharedBuffer = flatManager.getSharedBuffer();
      const batchSize = Math.ceil(genomes.length / this.workerCount);
      const evaluationPromises: Promise<any>[] = [];

      for (let i = 0; i < this.workerCount; i++) {
        const startIdx = i * batchSize;
        const count = Math.min(batchSize, genomes.length - startIdx);
        if (count <= 0) continue;

        evaluationPromises.push(this.pool.execute({
          sharedBuffer,
          batchInfo: {
            startIdx,
            count,
            maxNodes,
            nodeSize: FlatGenomeManager.NODE_SIZE
          },
          testCases,
          solverType,
          funcNames,
          termNames,
          lexicase,
          multiObjective
        }));
      }

      const allBatchResults = await Promise.all(evaluationPromises);
      for (const msg of allBatchResults) {
        for (const res of msg.results) {
          const originalId = genomes[res.id as number].id;
          results.set(originalId, { ...res, id: originalId });
        }
      }
      return results;
    } else if (genomes.length > 0) {
      // IPC Path fallback
      const batchSize = Math.ceil(genomes.length / this.workerCount);
      const evaluationPromises: Promise<any>[] = [];

      for (let i = 0; i < this.workerCount; i++) {
        const startIdx = i * batchSize;
        const count = Math.min(batchSize, genomes.length - startIdx);
        if (count <= 0) continue;

        evaluationPromises.push(this.pool.execute({
          genomes: genomes.slice(startIdx, startIdx + count),
          testCases,
          solverType,
          funcNames,
          termNames,
          lexicase,
          multiObjective
        }));
      }

      const allBatchResults = await Promise.all(evaluationPromises);
      for (const msg of allBatchResults) {
        for (const res of msg.results) {
          results.set(res.id, res);
        }
      }
      return results;
    }

    return results;
  }

  /**
   * Performs JIT-accelerated nudging in parallel across workers.
   */
  async nudge(
    genomes: Genome[],
    testCases: TestCase<any, any>[],
    termNames: string[],
    iterations: number = 5
  ): Promise<Map<string, { constants: number[], fitness: number }>> {
    const results = new Map<string, { constants: number[], fitness: number }>();
    if (genomes.length === 0) return results;

    const batchSize = Math.ceil(genomes.length / this.workerCount);
    const optimizationPromises: Promise<any>[] = [];

    for (let i = 0; i < this.workerCount; i++) {
        const startIdx = i * batchSize;
        const count = Math.min(batchSize, genomes.length - startIdx);
        if (count <= 0) continue;

        optimizationPromises.push(this.pool.execute({
            type: 'NUDGE',
            genomes: genomes.slice(startIdx, startIdx + count),
            testCases,
            termNames,
            iterations,
            solverType: 'math' // Nudge is only for math-based JIT
        }));
    }

    const allBatchResults = await Promise.all(optimizationPromises);
    for (const msg of allBatchResults) {
        if (msg.results) {
            for (const res of msg.results) {
                results.set(res.id, { constants: res.constants, fitness: res.fitness });
            }
        }
    }
    return results;
  }

  shutdown() {
    this.pool.shutdown();
  }
}
