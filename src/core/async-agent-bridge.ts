import { Genome } from './types';
import { DiscoveryAgent } from '../agents/discovery-agent';
import { DomainProfile } from './domain-analyzer';

export interface RefinedCandidate {
    genome: Genome;
    sourceGenId: number;
    fitness: number;
    lineage: string;
    timestamp: number;
    problemKey?: string;
}

/**
 * AsyncAgentBridge - Decoupled Communication for Co-Evolution
 * 
 * Manages bounded queues between the EvolutionEngine and the DiscoveryAgent.
 * Enforces staleness policies and generation boundary re-injection.
 */
export class AsyncAgentBridge {
    private resultQueue: RefinedCandidate[] = [];
    private inFlightSnapshots = 0;
    private totalProcessed = 0;
    private totalDiscarded = 0;
    
    private readonly MAX_IN_FLIGHT = 2; // Reduced for event loop stability
    private readonly MAX_PENDING = 500;
    private currentEntropy: number = 0;
    private diversityBias: number = 0;
    private globalIntensity: number = 1.0;
    private readonly MAX_STALENESS = 50; // Balanced for production throughput
    
    private agent: DiscoveryAgent;
    private domainProfile?: DomainProfile;
    private currentProblemKey: string | null = null;
    private pendingCommand: { action: string, intensity?: number, regime?: string } | null = null;

    constructor(agent: DiscoveryAgent) {
        this.agent = agent;
    }

    public setDomainProfile(profile: DomainProfile): void {
        this.domainProfile = profile;
        this.agent.setDomainProfile(profile);
    }

    public setEvaluationData(testX: Float64Array, testY: Float64Array, inputSize: number, fitnessMode: 'regression' | 'classification', caseWeights?: Float64Array, problemKey?: string): void {
        if (problemKey && this.currentProblemKey !== problemKey) {
            this.resultQueue = [];
            this.pendingCommand = null;
            this.inFlightSnapshots = 0;
        }
        this.currentProblemKey = problemKey || null;
        this.agent.setEvaluationData(testX, testY, inputSize, fitnessMode, caseWeights, problemKey);
    }

    /**
     * Push a population snapshot to the agent asynchronously.
     * Fire-and-forget pattern.
     */
    public async pushSnapshot(elites: Genome[], currentGen: number): Promise<void> {
        if (this.inFlightSnapshots >= this.MAX_IN_FLIGHT) {
            // Drop snapshot if agent is backlogged
            return;
        }

        this.inFlightSnapshots++;
        console.error(`[Agent-Bridge] Received snapshot for Gen ${currentGen}. In-flight: ${this.inFlightSnapshots}`);
        
        // Start refinement in background (No await here)
        this.processRefinement(elites, currentGen).finally(() => {
            if (this.inFlightSnapshots > 0) this.inFlightSnapshots--;
        });
    }

    public pushGlobalAnalytics(stats: any[], problemName: string = 'GLOBAL') {
        const { avgBias, globalIntensity } = this.agent.analyzeMultiIslandDynamics(stats);
        this.diversityBias = avgBias;
        this.globalIntensity = globalIntensity;

        // Phase 62.6: Strategic Feedback Loop
        const command = this.agent.decideGlobalAction(stats, problemName);
        if (command) {
            this.pendingCommand = command;
            if (typeof command.intensity === 'number' && command.intensity > 0) {
                this.globalIntensity = command.intensity;
            }
            const currentGen = stats.reduce((max, item) => Math.max(max, item.gen ?? 0), 0);
            this.captureAgentOutputs(currentGen);
        }
    }

    public pollCommand(): { action: string, intensity?: number, regime?: string } | null {
        const command = this.pendingCommand;
        this.pendingCommand = null;
        return command;
    }

    public getDiversityBias(): number {
        return this.diversityBias;
    }

    public getMutationIntensity(): number {
        return this.globalIntensity;
    }

    /**
     * Poll for refined candidates that are ready for re-injection.
     * Enforces staleness guard.
     */
    public poll(currentGen: number): RefinedCandidate[] {
        this.captureAgentOutputs(currentGen);

        // 1. Filter stale candidates and count them
        const initialCount = this.resultQueue.length;
        this.resultQueue = this.resultQueue.filter(c => 
            (!this.currentProblemKey || c.problemKey === this.currentProblemKey)
            && currentGen >= c.sourceGenId
            && (currentGen - c.sourceGenId) <= this.MAX_STALENESS
        );
        this.totalDiscarded += (initialCount - this.resultQueue.length);

        // 2. Return pending results
        const results = [...this.resultQueue];
        this.totalProcessed += results.length;
        this.resultQueue = [];
        return results;
    }

    /**
     * Internal background refinement loop.
     */
    private async processRefinement(elites: Genome[], currentGen: number): Promise<void> {
        try {
            // Push to agent AST loop
            await this.agent.pushPopulationSnapshot(elites, currentGen);
            this.captureAgentOutputs(currentGen);
        } catch (e: any) {
            console.error(`[Agent-Bridge] CRITICAL FAILURE in refinement: ${e.message}`);
        }
    }

    private captureAgentOutputs(currentGen: number): void {
        const agentResults = this.agent.pullAgentCandidates(currentGen);

        for (const res of agentResults) {
            const sourceGenId = typeof res.log?.gen === 'number' ? res.log.gen : currentGen;
            const improvement = (res.log?.fitnessAfter || 0) - (res.log?.fitnessBefore || 0);
            console.error(`[Agent-Bridge] Background refinement complete for Source Gen ${sourceGenId}. Improvement: +${improvement.toFixed(5)}`);
            if (this.resultQueue.length >= this.MAX_PENDING) {
                this.resultQueue.shift();
                this.totalDiscarded++;
            }

            this.resultQueue.push({
                genome: res.genome,
                sourceGenId,
                fitness: (res.genome as any).fitness,
                lineage: res.log.actionApplied || res.log.strategy || res.log.sourceId || 'UNKNOWN_AGENT_RESULT',
                timestamp: Date.now(),
                problemKey: this.currentProblemKey || undefined
            });
        }
    }

    public getStaleness(currentGen: number): number {
        if (this.resultQueue.length === 0) return 0;
        const totalStaleness = this.resultQueue.reduce((acc, r) => acc + (currentGen - r.sourceGenId), 0);
        const avg = totalStaleness / this.resultQueue.length;
        return Math.min(1.0, avg / this.MAX_STALENESS);
    }

    public getStatus() {
        return {
            inFlight: this.inFlightSnapshots,
            pending: this.resultQueue.length,
            totalProcessed: this.totalProcessed,
            totalDiscarded: this.totalDiscarded,
            stalenessRate: this.totalProcessed > 0 ? (this.totalDiscarded / (this.totalProcessed + this.totalDiscarded)) : 0
        };
    }
}
