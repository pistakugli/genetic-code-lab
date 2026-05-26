/**
 * 🚀 Genetic Code Lab - Super Engine (The Orchestrator)
 * Unifies Distributed Islands, JIT Compilation, and Agent Steering.
 */

import { DistributedManager } from './distributed-manager';
import { AsyncAgentBridge } from './async-agent-bridge';
import { DiscoveryAgent } from '../agents/discovery-agent';
import { ProblemDefinition, EvolutionConfig, EvolutionResult, Genome } from './types';
import { PostfixBridge } from './postfix-bridge';
import { OpCode } from './pop-buffer';

export class SuperEngine {
    private manager: DistributedManager;
    private bridge: AsyncAgentBridge;
    private agent: DiscoveryAgent;
    private config: EvolutionConfig;
    private problem: ProblemDefinition<any, any>;
    private numIslands: number;
    private islandParallelEvaluators: number;

    constructor(problem: ProblemDefinition<any, any>, config: Partial<EvolutionConfig> = {}) {
        this.problem = problem;
        this.config = {
            maxGenerations: 100,
            targetFitness: 1.0,
            stagnationLimit: 25,
            parallelEvaluators: 4,
            ...config,
            selectionMethod: 'epsilon-lexicase', 
            alps: {
                enabled: true,
                numLayers: 8,
                ageGap: 20,
                innovationInterval: 15
            },
            population: {
                size: 5000,
                eliteCount: 10,
                crossoverRate: 0.8,
                mutationRate: 0.3,
                maxGenomeSize: 200,
                maxGenomeDepth: 12,
                tournamentSize: 7,
                ...config.population
            }
        };

        const explicitDistributedIslands = this.config.distributedIslands;
        this.numIslands = Math.max(1, explicitDistributedIslands || this.config.parallelEvaluators || 1);
        this.islandParallelEvaluators = Math.max(
            1,
            this.config.islandParallelEvaluators
                || (explicitDistributedIslands ? (this.config.parallelEvaluators || 1) : 1)
        );

        this.agent = new DiscoveryAgent();
        this.bridge = new AsyncAgentBridge(this.agent);
        this.manager = new DistributedManager(this.numIslands, {
            ...this.config,
            islandParallelEvaluators: this.islandParallelEvaluators
        } as EvolutionConfig, this.bridge);
    }

    public async evolve(): Promise<EvolutionResult> {
        const startTime = Date.now();
        console.error(`[SuperEngine] Strategy: ${this.config.selectionMethod} | ALPS Layers: ${this.config.alps?.numLayers}`);
        
        if (this.numIslands === 1) {
            // Local fallback for stability
            const EvolutionEngine = require('./evolution').EvolutionEngine;
            const localEngine = new EvolutionEngine(this.problem, {
                ...this.config,
                parallelEvaluators: this.islandParallelEvaluators
            });
            if (localEngine.setDiscoveryAgent) localEngine.setDiscoveryAgent(this.agent);
            const result = await localEngine.evolve();
            return result;
        }

        // Start multi-island evolution and capture result directly from the promise
        const bestOfRun = await this.manager.start(this.problem, this.config.seed || 42);
        console.error(`[SuperEngine] Manager returned bestOfRun: ${!!bestOfRun} | Type: ${typeof bestOfRun}`);
        if (bestOfRun) console.error(`[SuperEngine] bestOfRun keys: ${Object.keys(bestOfRun)}`);
        
        let bestGenome: Genome;
        if (bestOfRun && bestOfRun.opcodes && bestOfRun.opcodes.length > 0) {
            const binaryOpcodes = bestOfRun.opcodes;
            const opcodes: OpCode[] = [];
            const data: number[] = [];
            for (let i = 0; i < binaryOpcodes.length; i++) {
                const raw = binaryOpcodes[i];
                opcodes.push(raw >>> 24);
                data.push(raw & 0xFFFFFF);
            }

            const varMap = (this.problem.terminalSet.filter(t => t.type === 'variable').map(t => t.name));
            bestGenome = PostfixBridge.unbake(opcodes, data, Array.from(bestOfRun.constants), varMap);
            bestGenome.fitness = bestOfRun.fitness;
        } else {
            console.error('[SuperEngine] ERROR: No valid bestOfRun retrieved from DistributedManager.');
            bestGenome = { id: 'empty', genes: [], fitness: 0, generation: 0, age: 0 };
        }

        return {
            bestGenome,
            generationsRun: this.config.maxGenerations,
            totalEvaluations: this.config.maxGenerations * this.config.population.size * this.numIslands,
            totalTimeMs: Date.now() - startTime,
            success: bestGenome.fitness >= (this.config.targetFitness || 0.99),
            stats: [], 
            archive: []
        };
    }

    public async shutdown() {
        await this.manager.stop();
    }
}
