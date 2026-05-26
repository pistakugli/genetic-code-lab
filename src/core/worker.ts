/**
 * Multi-core Worker for Genetic GP Evaluations
 */

import { parentPort } from 'worker_threads';
import { Genome, TestCase, MultiObjectiveFitness, FunctionDefinition, EvolutionConfig, ProblemDefinition } from './types';
import { EvolutionEngine } from './evolution';
import { MigrationHub } from './migration-hub';
import { PopulationBuffer } from './pop-buffer';
import { BlockEvaluator } from './block-evaluator';

if (!parentPort) {
  throw new Error('This file must be run as a worker thread.');
}

let engine: EvolutionEngine<any, any> | null = null;

parentPort.on('message', (data: any) => {
  const { type, islandId, config, problem, hubBuffer, preferredSource } = data;

  if (type === 'START_ISLAND' && config && problem && hubBuffer) {
      console.error(`[Island-Worker] Starting Actor for ID: ${islandId}`);
      const hub = new MigrationHub(hubBuffer);
      
      // Inject seed into config for deterministic island search
      const islandConfig = { ...config, seed: data.seed };
      
      if (islandConfig.quotientMode) {
          BlockEvaluator.divisionMode = 'strict-quotient';
      }

      engine = new EvolutionEngine(problem, islandConfig);
      if (preferredSource !== undefined) engine.preferredSource = preferredSource;
      
      engine.onStats = (stats) => parentPort!.postMessage({
          ...stats,
          semanticArchive: engine?.getSemanticArchiveDebugStats(),
          hubStats: hub.getStats()
      });
      engine.evolveIsland(hub, islandId || 0).then(() => {
          if (engine) {
              const resultData = {
                  type: 'FINAL_RESULT',
                  islandId,
                  bestGenome: engine.getBestBinary(),
                  bestGenomeTree: engine.getBestGenome(),
                  bestPredictions: Array.from(engine.getBestPredictions() || []),
                  semanticArchive: engine.getSemanticArchiveDebugStats(),
                  hubStats: hub.getStats()
              };
              
              // Optional FS fallback for callers that need worker result persistence.
              try {
                  if (process.env.GCL_WORKER_RESULT_DIR) {
                      const fs = require('fs');
                      const path = require('path');
                      fs.mkdirSync(process.env.GCL_WORKER_RESULT_DIR, { recursive: true });
                      const resultPath = path.join(process.env.GCL_WORKER_RESULT_DIR, `result_island_${islandId}.json`);
                      fs.writeFileSync(resultPath, JSON.stringify(resultData));
                  }
              } catch (e) {
                  // Silent fail, rely on parentPort
              }

              parentPort!.postMessage(resultData);
          }
      }).catch(err => {
          const message = err instanceof Error ? err.message : String(err);
          console.error(`[Island-Worker ${islandId}] FATAL:`, err);
          parentPort!.postMessage({
              type: 'FINAL_ERROR',
              islandId,
              error: message,
              stack: err instanceof Error ? err.stack : undefined
          });
      });
      return;
  }

  if (type === 'UPDATE_TOPOLOGY' && engine) {
      engine.preferredSource = preferredSource;
  }

  if (type === 'ADAPTIVE_SIGNAL' && engine) {
      engine.setMutationIntensity(data.intensity);
      if (data.command) {
          engine.applyCommand(data.command);
      }
  }

  if (type === 'AGENT_CHAMPION' && engine) {
      const { opcodes, constants, fitness } = data;
      const safeFit = (typeof fitness === 'number' && isFinite(fitness)) ? fitness : 0;
      console.error(`[PROOF] Worker ${islandId} received AGENT_CHAMPION (Fitness: ${safeFit.toFixed(6)})`);
      engine.injectGenome(new Uint32Array(opcodes), new Float64Array(constants), safeFit);
  }

  // Phase 61.3: Protocol Restoration - Batch Evaluation (used by ParallelEvaluator)
  if (data.sharedBuffer && data.batchInfo) {
      const { sharedBuffer, batchInfo, testCases, solverType } = data;
      const { startIdx, count } = batchInfo;
      
      // Initialize a lightweight buffer for evaluation
      const buffer = new PopulationBuffer(sharedBuffer, count + startIdx, testCases.length);
      
      // Extract inputs/outputs
      const inputSize = (testCases[0].input as number[]).length;
      const inputs = new Float64Array(testCases.length * inputSize);
      const expected = new Float64Array(testCases.length);
      testCases.forEach((tc: any, i: number) => {
          (tc.input as number[]).forEach((val, j) => inputs[i * inputSize + j] = val);
          expected[i] = tc.expectedOutput as number;
      });

      // Run Block Evaluator
      const fitnessMode = solverType === 'math-classify' ? 'classification' : 'regression';
      const results = BlockEvaluator.evaluateBlock(buffer, startIdx, count, inputs, expected, inputSize, fitnessMode);
      
      // Return protocol compatible with ParallelEvaluator.ts
      const responseResults = [];
      for (let i = 0; i < count; i++) {
          responseResults.push({
              id: startIdx + i,
              fitness: results[i],
              caseErrors: [] // Minimal telemetry to reduce IPC overhead
          });
      }

      parentPort!.postMessage({ results: responseResults });
  }
});
