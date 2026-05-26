import { Worker } from 'worker_threads';
import { MigrationHub } from './migration-hub';
import { ProblemDefinition, EvolutionConfig } from './types';
import { AsyncAgentBridge } from './async-agent-bridge';
import path from 'path';

export type IslandLaunchSpec = {
    problem?: ProblemDefinition<any, any>;
    config?: Partial<EvolutionConfig>;
    preferredSource?: number;
};

/**
 * PHASE 38/40: DISTRIBUTED ISLAND MANAGER
 * Orchestrates multi-core islands and global agent steering with real-time causal tracking.
 */
export class DistributedManager {
    private workers: Worker[] = [];
    private hub: MigrationHub;
    private numIslands: number;
    private config: EvolutionConfig;
    private bridge: AsyncAgentBridge;
    private globalStats: Map<number, { gen: number, bestFitness: number, entropy: number, causalUtility: number, bestSize: number, bestGenome?: any }> = new Map();
    private migrationMesh: Map<number, number> = new Map(); 

    private bestOfRun: any = null;
    private history: any[] = [];
    private finalResults: Map<number, any> = new Map();

    constructor(numIslands: number, config: EvolutionConfig, bridge: AsyncAgentBridge) {
        this.numIslands = numIslands;
        this.config = config;
        this.bridge = bridge;
        this.hub = new MigrationHub();
    }

    public getBestOfRun() {
        return this.bestOfRun;
    }

    public getHistory() {
        return [...this.history];
    }

    public getFinalResults() {
        return Array.from(this.finalResults.values());
    }

    public getMigrationStats() {
        return this.hub.getStats();
    }

    public async shutdown(): Promise<void> {
        for (const worker of this.workers) {
            await worker.terminate();
        }
    }

    public start(problem: ProblemDefinition<any, any>, seed: number = 42, islandSpecs: IslandLaunchSpec[] = []): Promise<any> {
        return new Promise((resolve, reject) => {
            console.error(`[Dist-Manager] Launching ${this.numIslands} Island Actors...`);
            
            const isCompiled = __filename.endsWith('.js');
            const workerExt = isCompiled ? '.js' : '.ts';
            const workerPath = path.resolve(__dirname, 'worker' + workerExt);
            
            let finishedCount = 0;
            let settled = false;

            const sanitize = (obj: any): any => {
                if (Array.isArray(obj)) return obj.map(v => sanitize(v));
                if (obj !== null && typeof obj === 'object') {
                    const res: any = {};
                    for (const k in obj) {
                        if (typeof obj[k] === 'function') continue;
                        res[k] = sanitize(obj[k]);
                    }
                    return res;
                }
                return obj;
            };

            const intraIslandEvaluators = Math.max(
                1,
                this.config.islandParallelEvaluators
                    || this.config.parallelEvaluators
                    || 1
            );
            const islandConfig = sanitize({
                ...this.config,
                parallelEvaluators: intraIslandEvaluators
            });
            const serializedProblem = sanitize(problem);

            for (let i = 0; i < this.numIslands; i++) {
                const spec = islandSpecs[i] || {};
                const perIslandConfig = sanitize({
                    ...islandConfig,
                    ...(spec.config || {})
                });
                const perIslandProblem = sanitize(spec.problem || problem);
                const worker = new Worker(workerPath, { 
                    workerData: { islandId: i },
                    execArgv: isCompiled ? [] : ['-r', 'ts-node/register'],
                    env: { ...process.env, TS_NODE_TRANSPILE_ONLY: 'true' },
                    stderr: true, stdout: true
                });
                
                this.workers.push(worker);

                worker.on('message', async (msg) => {
                    if (msg.type === 'ISLAND_STATS') {
                        this.history.push({ islandId: i, ...msg });
                        this.globalStats.set(i, { 
                            gen: msg.gen, 
                            bestFitness: msg.bestFitness, 
                            entropy: msg.entropy,
                            causalUtility: msg.causalUtility || 0,
                            bestSize: msg.bestSize || 0,
                            bestGenome: msg.bestGenome 
                        });

                        if (msg.bestGenome && msg.bestGenome.opcodes && msg.bestGenome.opcodes.length > 0) {
                            if (!this.bestOfRun || msg.bestGenome.fitness > this.bestOfRun.fitness) {
                                this.bestOfRun = {
                                    opcodes: new Uint32Array(msg.bestGenome.opcodes),
                                    constants: new Float64Array(msg.bestGenome.constants),
                                    fitness: msg.bestGenome.fitness
                                };
                            }
                        }
                    }

                    if (msg.type === 'FINAL_ERROR') {
                        if (!settled) {
                            settled = true;
                            reject(new Error(`Island ${msg.islandId} failed: ${msg.error || 'unknown worker error'}`));
                        }
                    }

                    if (msg.type === 'FINAL_RESULT') {
                        if (settled) return;
                        this.finalResults.set(i, msg);
                        if (msg.bestGenome && msg.bestGenome.opcodes && msg.bestGenome.opcodes.length > 0) {
                            if (!this.bestOfRun || msg.bestGenome.fitness > this.bestOfRun.fitness) {
                                this.bestOfRun = {
                                    opcodes: new Uint32Array(msg.bestGenome.opcodes),
                                    constants: new Float64Array(msg.bestGenome.constants),
                                    fitness: msg.bestGenome.fitness
                                };
                            }
                        }
                        finishedCount++;
                        if (finishedCount === this.numIslands) {
                            settled = true;
                            console.error(`[Dist-Manager] Finished. Final Global Best: ${this.bestOfRun?.fitness}`);
                            resolve(this.bestOfRun);
                        }
                    }
                });

                worker.on('error', (err) => {
                    if (!settled) {
                        settled = true;
                        reject(err);
                    }
                });

                worker.on('exit', (code) => {
                    if (!settled && code !== 0) {
                        settled = true;
                        reject(new Error(`Island worker exited with code ${code}`));
                    }
                });

                worker.postMessage({
                    type: 'START_ISLAND',
                    islandId: i,
                    config: perIslandConfig,
                    problem: perIslandProblem || serializedProblem,
                    hubBuffer: this.hub.getSharedBuffer(),
                    seed: seed + i,
                    preferredSource: spec.preferredSource ?? ((i + this.numIslands - 1) % this.numIslands)
                });
            }
        });
    }

    public getGlobalAnalytics() {
        return Array.from(this.globalStats.entries()).map(([islandId, stats]) => ({ islandId, ...stats }));
    }

    public async stop(): Promise<void> {
        for (const worker of this.workers) await worker.terminate();
    }
}
