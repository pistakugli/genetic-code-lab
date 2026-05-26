import { Genome, Gene } from '../core/types';
import { deepClone, generateId, randomInt, randomChoice } from '../utils/helpers';
import { NumericalOptimizer } from '../core/optimizer';
import { PostfixBridge } from '../core/postfix-bridge';
import { OpCode, PopulationBuffer } from '../core/pop-buffer';

export interface RefinedCandidate {
    genome: Genome | any;
    log: any;
}

type LogicMotifType = 'parity' | 'majority' | 'mux';

interface LogicMotifSeed {
    opcodes: Uint32Array;
    sourceId: string;
    motifType: string;
    variables: number[];
}

type BoundaryAudit = {
    lowerAccuracy: number;
    upperAccuracy: number;
};

export class DiscoveryAgent {
    private problemMemory: Map<string, any[]> = new Map();
    private lastBestFitPerProblem: Map<string, number> = new Map();
    private currentAction: string | null = null;
    private actionCooldown: number = 10;
    private lastActionGen: number = -100;
    private inputSize: number = 1;
    private testX: Float64Array | null = null;
    private testY: Float64Array | null = null;
    private fitnessMode: 'regression' | 'classification' = 'regression';
    private caseWeights: Float64Array | null = null;
    private domainProfile: any | null = null;
    private currentProblemKey: string | null = null;
    private issuedLogicBootstraps: Map<string, Set<string>> = new Map();

    private outboundQueue: RefinedCandidate[] = [];

    private signatureHash(values?: Float64Array, limit: number = 256): string {
        if (!values || values.length === 0) return '0';
        let hash = 2166136261 >>> 0;
        const step = Math.max(1, Math.floor(values.length / limit));
        for (let i = 0; i < values.length; i += step) {
            const value = Number.isFinite(values[i]) ? Math.round(values[i] * 1e6) : 0x7f4a7c15;
            hash ^= value >>> 0;
            hash = Math.imul(hash, 16777619) >>> 0;
        }
        return hash.toString(16);
    }

    public setEvaluationData(testX: Float64Array | undefined, testY: Float64Array | undefined, inputSize: number, mode: string, caseWeights?: Float64Array, problemKey?: string): void {
        const nextProblemKey = problemKey || `${inputSize}:${mode}:${testY?.length || 0}:${testX?.length || 0}:${this.signatureHash(testX)}:${this.signatureHash(testY)}:${this.signatureHash(caseWeights)}`;
        if (this.currentProblemKey !== nextProblemKey) {
            this.problemMemory.clear();
            this.lastBestFitPerProblem.clear();
            this.currentAction = null;
            this.lastActionGen = -100;
            this.outboundQueue = [];
            this.issuedLogicBootstraps.clear();
            this.currentProblemKey = nextProblemKey;
        }
        this.testX = testX || null;
        this.testY = testY || null;
        this.inputSize = inputSize;
        this.fitnessMode = mode as any;
        this.caseWeights = caseWeights || null;
    }

    public setDomainProfile(profile: any): void {
        this.domainProfile = profile;
    }

    public setCaseWeights(weights: Float64Array): void {
        this.caseWeights = weights;
    }

    public pullAgentCandidates(_gen?: number): RefinedCandidate[] {
        const candidates = [...this.outboundQueue];
        this.outboundQueue = [];
        return candidates;
    }

    public decodeBinaryCandidate(candidate: any): Genome {
        // Phase 62.5: Bridge between packed binary and AST
        // Candidate opcodes are packed (OpCode << 24) | data
        const opcodes: number[] = [];
        const data: number[] = [];
        for (let i = 0; i < candidate.opcodes.length; i++) {
            const unpacked = PopulationBuffer.unpack(candidate.opcodes[i]);
            opcodes.push(unpacked.op);
            data.push(unpacked.data);
        }
        return PostfixBridge.unbake(opcodes, data, Array.from(candidate.constants), ['x0', 'x1', 'x2', 'x3']);
    }

    public generateDiscoverySkeleton(): Genome {
        // Phase 62.5: Discovery Agent Isolation Proof Skeleton
        // Target: (x^2 + 1) / (x - 2)
        const genes: Gene[] = [
            { id: 'root', type: 'function', operation: 'div', operands: ['num', 'den'] },
            { id: 'num', type: 'function', operation: 'add', operands: ['sq', 'c1'] },
            { id: 'sq', type: 'function', operation: 'mul', operands: ['x0', 'x0'] },
            { id: 'c1', type: 'terminal', operation: 'const', operands: [1.0] }, // exact target
            { id: 'den', type: 'function', operation: 'sub', operands: ['x0', 'c2'] },
            { id: 'x0', type: 'terminal', operation: 'x0', operands: [] },
            { id: 'c2', type: 'terminal', operation: 'const', operands: [2.0] }  // exact target
        ];
        const genome = {
            id: 'skeleton_' + generateId(),
            genes,
            fitness: -Infinity,
            generation: 0,
            age: 0
        };
        // Evaluate initial fitness if data is available
        if (this.testX && this.testY) {
            NumericalOptimizer.refineIndividual(genome, this.testX, this.testY, this.inputSize, 0, this.fitnessMode);
        }
        return genome;
    }

    private evalBinaryCandidate(opcodes: Uint32Array, constants: Float64Array, input: number[]): number {
        const stack: number[] = [];
        for (let i = 0; i < opcodes.length; i++) {
            const raw = opcodes[i];
            const op = raw >>> 24;
            const data = raw & 0xFFFFFF;
            switch (op) {
                case OpCode.XN:
                    stack.push(input[data] || 0);
                    break;
                case OpCode.CONST:
                    stack.push(constants[data] || 0);
                    break;
                case OpCode.AND: {
                    const b = stack.pop() || 0;
                    const a = stack.pop() || 0;
                    stack.push(a > 0.5 && b > 0.5 ? 1 : 0);
                    break;
                }
                case OpCode.OR: {
                    const b = stack.pop() || 0;
                    const a = stack.pop() || 0;
                    stack.push(a > 0.5 || b > 0.5 ? 1 : 0);
                    break;
                }
                case OpCode.NAND: {
                    const b = stack.pop() || 0;
                    const a = stack.pop() || 0;
                    stack.push(a > 0.5 && b > 0.5 ? 0 : 1);
                    break;
                }
                case OpCode.NOR: {
                    const b = stack.pop() || 0;
                    const a = stack.pop() || 0;
                    stack.push(a > 0.5 || b > 0.5 ? 0 : 1);
                    break;
                }
                case OpCode.XOR: {
                    const b = stack.pop() || 0;
                    const a = stack.pop() || 0;
                    const aBit = a > 0.5 ? 1 : 0;
                    const bBit = b > 0.5 ? 1 : 0;
                    stack.push(aBit + bBit === 1 ? 1 : 0);
                    break;
                }
                case OpCode.ITE: {
                    const whenFalse = stack.pop() || 0;
                    const whenTrue = stack.pop() || 0;
                    const cond = stack.pop() || 0;
                    stack.push(cond > 0.5 ? whenTrue : whenFalse);
                    break;
                }
                default:
                    stack.push(0);
                    break;
            }
        }
        return stack.length > 0 ? stack[stack.length - 1] : 0;
    }

    private enqueueContextualPositiveLift(action: string, baseCandidate: any, motifs: LogicMotifSeed[], gen: number): void {
        if (!baseCandidate?.opcodes || !(baseCandidate.opcodes instanceof Uint32Array)) return;
        const baseConstants = baseCandidate.constants instanceof Float64Array
            ? baseCandidate.constants
            : new Float64Array(baseCandidate.constants || []);

        for (const motif of motifs) {
            const opcodes = new Uint32Array(baseCandidate.opcodes.length + motif.opcodes.length + 1);
            opcodes.set(baseCandidate.opcodes, 0);
            opcodes.set(motif.opcodes, baseCandidate.opcodes.length);
            opcodes[opcodes.length - 1] = (OpCode.OR << 24);

            this.outboundQueue.push({
                genome: { opcodes, constants: new Float64Array(baseConstants) },
                log: {
                    actionApplied: action,
                    strategy: action,
                    sourceId: `CONTEXTUAL_${motif.sourceId}`,
                    motifType: motif.motifType,
                    variables: motif.variables,
                    gen,
                    fitnessBefore: baseCandidate.fitness || 0,
                    fitnessAfter: 0
                }
            });
        }
    }

    private auditThresholdBoundary(candidate: any, threshold: number): BoundaryAudit | null {
        if (!this.testX || !this.testY || this.inputSize <= 0) return null;
        if (!candidate?.opcodes || !(candidate.opcodes instanceof Uint32Array)) return null;

        const constants = candidate.constants instanceof Float64Array
            ? candidate.constants
            : new Float64Array(candidate.constants || []);

        let lowerTotal = 0;
        let lowerCorrect = 0;
        let upperTotal = 0;
        let upperCorrect = 0;

        for (let t = 0; t < this.testY.length; t++) {
            const input: number[] = [];
            let ones = 0;
            const offset = t * this.inputSize;
            for (let i = 0; i < this.inputSize; i++) {
                const value = this.testX[offset + i] > 0.5 ? 1 : 0;
                input.push(value);
                ones += value;
            }

            const pred = this.evalBinaryCandidate(candidate.opcodes, constants, input) > 0.5 ? 1 : 0;
            const expected = this.testY[t] > 0.5 ? 1 : 0;
            if (ones === threshold - 1) {
                lowerTotal++;
                if (pred === expected) lowerCorrect++;
            } else if (ones === threshold) {
                upperTotal++;
                if (pred === expected) upperCorrect++;
            }
        }

        if (lowerTotal === 0 || upperTotal === 0) return null;
        return {
            lowerAccuracy: lowerCorrect / lowerTotal,
            upperAccuracy: upperCorrect / upperTotal
        };
    }

    public async pushPopulationSnapshot(elites: Genome[], _gen: number): Promise<void> {
        if (elites.length === 0 || !this.domainProfile || this.domainProfile.domainClass !== 'discrete') return;
        const threshold = this.domainProfile?.signals?.logicThreshold;
        if (typeof threshold !== 'number' || threshold <= 1) return;

        const problemName = this.currentProblemKey || 'CURRENT_PROBLEM';
        if (!this.issuedLogicBootstraps.has(problemName)) this.issuedLogicBootstraps.set(problemName, new Set());
        const issued = this.issuedLogicBootstraps.get(problemName)!;
        if (issued.has('MAJORITY_UPPER_LIFT')) return;

        const bestElite = elites[0] as any;
        const boundaryAudit = this.auditThresholdBoundary(bestElite, threshold);
        if (!boundaryAudit) return;

        if (_gen >= 5 && boundaryAudit.upperAccuracy + 0.15 < boundaryAudit.lowerAccuracy) {
            const motifs = this.generateLogicMotifs('majority')
                .filter(motif => motif.motifType === 'thr2of4' || motif.motifType === 'thr3of4')
                .sort((a, b) => a.motifType === 'thr2of4' ? -1 : b.motifType === 'thr2of4' ? 1 : 0);
            if (motifs.length === 0) return;
            issued.add('MAJORITY_UPPER_LIFT');
            this.currentAction = 'MAJORITY_UPPER_LIFT';
            console.error(`[Director] UPPER THRESHOLD SIDE WEAK (${boundaryAudit.upperAccuracy.toFixed(4)} < ${boundaryAudit.lowerAccuracy.toFixed(4)}). Executing MAJORITY_UPPER_LIFT.`);
            this.enqueueContextualPositiveLift('MAJORITY_UPPER_LIFT', bestElite, motifs.slice(0, 4), _gen);
        }
    }

    public analyzeMultiIslandDynamics(stats: any[]): { avgBias: number, globalIntensity: number } {
        return { avgBias: 0.5, globalIntensity: 1.0 };
    }

    private generateLogicMotifs(type: LogicMotifType, selectorBits: number[] = [], dataBits: number[] = []): LogicMotifSeed[] {
        switch (type) {
            case 'majority': {
                const motifs: LogicMotifSeed[] = [];
                for (let i = 0; i < Math.max(0, this.inputSize - 1); i++) {
                    motifs.push({
                        opcodes: new Uint32Array([
                            (OpCode.XN << 24) | i,
                            (OpCode.XN << 24) | (i + 1),
                            (OpCode.AND << 24)
                        ]),
                        sourceId: 'MAJORITY_MOTIF',
                        motifType: 'and2',
                        variables: [i, i + 1]
                    });
                }

                for (let i = 0; i < Math.max(0, this.inputSize - 2); i++) {
                    const a = i;
                    const b = i + 1;
                    const c = i + 2;
                    motifs.push({
                        opcodes: new Uint32Array([
                            (OpCode.XN << 24) | a, (OpCode.XN << 24) | b, (OpCode.AND << 24),
                            (OpCode.XN << 24) | a, (OpCode.XN << 24) | c, (OpCode.AND << 24),
                            (OpCode.XN << 24) | b, (OpCode.XN << 24) | c, (OpCode.AND << 24),
                            (OpCode.OR << 24),
                            (OpCode.OR << 24)
                        ]),
                        sourceId: 'MAJORITY3_MOTIF',
                        motifType: 'maj3',
                        variables: [a, b, c]
                    });
                }

                for (let i = 0; i < Math.max(0, this.inputSize - 3); i++) {
                    const a = i;
                    const b = i + 1;
                    const c = i + 2;
                    const d = i + 3;
                    motifs.push({
                        opcodes: new Uint32Array([
                            (OpCode.XN << 24) | a, (OpCode.XN << 24) | b, (OpCode.AND << 24),
                            (OpCode.XN << 24) | a, (OpCode.XN << 24) | c, (OpCode.AND << 24),
                            (OpCode.OR << 24),
                            (OpCode.XN << 24) | a, (OpCode.XN << 24) | d, (OpCode.AND << 24),
                            (OpCode.OR << 24),
                            (OpCode.XN << 24) | b, (OpCode.XN << 24) | c, (OpCode.AND << 24),
                            (OpCode.OR << 24),
                            (OpCode.XN << 24) | b, (OpCode.XN << 24) | d, (OpCode.AND << 24),
                            (OpCode.OR << 24),
                            (OpCode.XN << 24) | c, (OpCode.XN << 24) | d, (OpCode.AND << 24),
                            (OpCode.OR << 24)
                        ]),
                        sourceId: 'THRESHOLD2OF4_MOTIF',
                        motifType: 'thr2of4',
                        variables: [a, b, c, d]
                    });
                    motifs.push({
                        opcodes: new Uint32Array([
                            (OpCode.XN << 24) | a, (OpCode.XN << 24) | b, (OpCode.AND << 24), (OpCode.XN << 24) | c, (OpCode.AND << 24),
                            (OpCode.XN << 24) | a, (OpCode.XN << 24) | b, (OpCode.AND << 24), (OpCode.XN << 24) | d, (OpCode.AND << 24),
                            (OpCode.OR << 24),
                            (OpCode.XN << 24) | a, (OpCode.XN << 24) | c, (OpCode.AND << 24), (OpCode.XN << 24) | d, (OpCode.AND << 24),
                            (OpCode.OR << 24),
                            (OpCode.XN << 24) | b, (OpCode.XN << 24) | c, (OpCode.AND << 24), (OpCode.XN << 24) | d, (OpCode.AND << 24),
                            (OpCode.OR << 24)
                        ]),
                        sourceId: 'THRESHOLD3OF4_MOTIF',
                        motifType: 'thr3of4',
                        variables: [a, b, c, d]
                    });
                }

                return motifs;
            }

            case 'parity': {
                const motifs: LogicMotifSeed[] = [];
                for (let i = 0; i < Math.max(0, this.inputSize - 1); i++) {
                    const left = i;
                    const right = i + 1;
                    motifs.push({
                        opcodes: new Uint32Array([
                            (OpCode.XN << 24) | left,
                            (OpCode.XN << 24) | right,
                            (OpCode.XOR << 24)
                        ]),
                        sourceId: 'PARITY_MOTIF',
                        motifType: 'xor2',
                        variables: [left, right]
                    });
                }
                return motifs;
            }

            case 'mux': {
                if (this.inputSize < 3) return [];

                const activeSelectors = selectorBits.length > 0 ? selectorBits : [0];
                const fallbackDataBits = Array.from({ length: this.inputSize }, (_, i) => i).filter(i => !activeSelectors.includes(i));
                const resolvedDataBits = (dataBits.length > 0 ? dataBits : fallbackDataBits).filter(i => !activeSelectors.includes(i));
                if (resolvedDataBits.length < 2) return [];

                const motifs: LogicMotifSeed[] = [];
                for (const selector of activeSelectors) {
                    for (let i = 0; i + 1 < resolvedDataBits.length; i += 2) {
                        const d0 = resolvedDataBits[i];
                        const d1 = resolvedDataBits[i + 1];
                        motifs.push({
                            opcodes: new Uint32Array([
                                (OpCode.XN << 24) | selector,
                                (OpCode.XN << 24) | d1,
                                (OpCode.XN << 24) | d0,
                                (OpCode.ITE << 24)
                            ]),
                            sourceId: 'MUX_ITE_MOTIF',
                            motifType: 'mux1-ite',
                            variables: [selector, d0, d1]
                        });
                    }
                }

                return motifs;
            }
        }
    }

    private enqueueLogicMotifs(action: 'PARITY_BOOTSTRAP' | 'MAJORITY_BOOTSTRAP' | 'MUX_BOOTSTRAP', motifs: LogicMotifSeed[], gen: number): void {
        for (const motif of motifs) {
            this.outboundQueue.push({
                genome: { opcodes: motif.opcodes, constants: new Float64Array([]) },
                log: {
                    actionApplied: action,
                    strategy: action,
                    sourceId: motif.sourceId,
                    motifType: motif.motifType,
                    variables: motif.variables,
                    gen,
                    fitnessBefore: 0,
                    fitnessAfter: 0
                }
            });
        }
    }

    public decideGlobalAction(stats: any[], problemName: string): { action: string, intensity: number, regime: string } | null {
        if (stats.length === 0) return null;

        const avgGen = stats.reduce((acc, s) => acc + s.gen, 0) / stats.length;
        const avgStagnation = stats.reduce((acc, s) => acc + s.stagnationCount, 0) / stats.length;
        const avgSize = stats.reduce((acc, s) => acc + s.bestSize, 0) / stats.length;
        const bestFit = Math.max(...stats.map(s => s.bestFitness));

        const isAtTarget = bestFit >= 0.999999;

        if (this.currentAction && !this.currentAction.startsWith('LOGIC')) {
            const prevBest = this.lastBestFitPerProblem.get(problemName) || 0;
            const delta = bestFit - prevBest;

            if (!this.problemMemory.has(problemName)) this.problemMemory.set(problemName, []);
            this.problemMemory.get(problemName)!.push({ action: this.currentAction, deltaFit: delta, gen: avgGen });

            if (delta > 0) console.error(`[Director] Action ${this.currentAction} SUCCESS (ΔFit: +${delta.toFixed(6)})`);
            this.currentAction = null;
        }

        if (isAtTarget) return null;

        this.lastBestFitPerProblem.set(problemName, bestFit);
        if (avgGen < this.lastActionGen + this.actionCooldown) return null;

        // Domain Specific Bootstrapping for Boolean problems
        if (this.domainProfile && this.domainProfile.domainClass === 'discrete' && avgStagnation > 4) {
            const sig = this.domainProfile.signals;
            if (!this.issuedLogicBootstraps.has(problemName)) this.issuedLogicBootstraps.set(problemName, new Set());
            const issued = this.issuedLogicBootstraps.get(problemName)!;
            const muxLike = !!(sig.selectorMap && sig.selectorMap.some((v: number) => v === 1) && sig.selectorMap.some((v: number) => v === 0));
            
            // MUX DETECTION
            if (!issued.has('MUX_BOOTSTRAP') && muxLike) {
                this.lastActionGen = avgGen;
                this.currentAction = 'MUX_BOOTSTRAP';
                console.error(`[Director] SELECTOR/DATA INFLUENCE ASYMMETRY DETECTED. Executing MUX_BOOTSTRAP (Motif Injection).`);
                
                const selBits: number[] = [];
                const dataBits: number[] = [];
                for (let i = 0; i < this.inputSize; i++) {
                    if (sig.selectorMap[i] === 1) selBits.push(i);
                    else dataBits.push(i);
                }

                const motifs = this.generateLogicMotifs('mux', selBits, dataBits);
                if (motifs.length === 0) return null;
                issued.add('MUX_BOOTSTRAP');
                this.enqueueLogicMotifs('MUX_BOOTSTRAP', motifs, avgGen);
                return { action: 'MUX_BOOTSTRAP', intensity: 1.5, regime: 'EXPLORE' };
            }

            // MAJORITY DETECTION
            if (!issued.has('MAJORITY_BOOTSTRAP') && !muxLike && sig.logicThreshold !== undefined && sig.monotonicityScore && sig.monotonicityScore > 0.9) {
                this.lastActionGen = avgGen;
                this.currentAction = 'MAJORITY_BOOTSTRAP';
                console.error(`[Director] MAJORITY THRESHOLD DETECTED. Executing MAJORITY_BOOTSTRAP (Motif Injection).`);

                const motifs = this.generateLogicMotifs('majority');
                if (motifs.length === 0) return null;
                issued.add('MAJORITY_BOOTSTRAP');
                this.enqueueLogicMotifs('MAJORITY_BOOTSTRAP', motifs, avgGen);
                return { action: 'MAJORITY_BOOTSTRAP', intensity: 1.5, regime: 'EXPLORE' };
            }

            // PARITY DETECTION
            if (!issued.has('PARITY_BOOTSTRAP') && sig.alternationScore && sig.alternationScore > 0.8) {
                this.lastActionGen = avgGen;
                this.currentAction = 'PARITY_BOOTSTRAP';
                console.error(`[Director] EXTREME ALTERNATION DETECTED. Executing PARITY_BOOTSTRAP (Motif Injection).`);

                const motifs = this.generateLogicMotifs('parity');
                if (motifs.length === 0) return null;
                issued.add('PARITY_BOOTSTRAP');
                this.enqueueLogicMotifs('PARITY_BOOTSTRAP', motifs, avgGen);
                return { action: 'PARITY_BOOTSTRAP', intensity: 1.5, regime: 'EXPLORE' };
            }

        }

        const isDiscreteProblem = this.domainProfile?.domainClass === 'discrete' || (this.fitnessMode as string) === 'classification';
        if (isDiscreteProblem) {
            if (avgStagnation > 10) {
                this.lastActionGen = avgGen;
                this.currentAction = 'MUTATION_KICK';
                const intensity = Math.min(6.0, 1.5 + Math.floor(avgStagnation / 5) + Math.max(0, this.inputSize - 5) * 0.35);
                console.error(`[Director] DISCRETE PLATEAU DETECTED. Escalating MUTATION_KICK to ${intensity.toFixed(2)}.`);
                return { action: 'MUTATION_KICK', intensity, regime: 'EXPLORE' };
            }
            return null;
        }

        // Phase 67: General Interaction Induction (Frontier Protocol)
        const isClassification = (this.fitnessMode as string) === 'classification';
        if (!isClassification && (!this.domainProfile || this.domainProfile.domainClass !== 'discrete') && avgStagnation > 7 && this.inputSize > 1) {
            this.lastActionGen = avgGen;
            this.currentAction = 'INTERACTION_INDUCTION';
            console.error(`[Director] VARIABLE BLINDNESS DETECTED. Injecting general interaction seeds.`);
            
            const v1 = randomInt(0, this.inputSize - 1);
            const v2 = (v1 + randomInt(1, this.inputSize - 1)) % this.inputSize;
            const isLogic = this.fitnessMode === 'classification' || (this.domainProfile && this.domainProfile.domainClass === 'discrete');
            const ops = isLogic ? [OpCode.AND, OpCode.OR, OpCode.NAND, OpCode.NOR] : [OpCode.MUL, OpCode.DIV, OpCode.ADD, OpCode.SUB];
            const op = randomChoice(ops);

            const opcodes = new Uint32Array([
                (OpCode.XN << 24) | v1,
                (OpCode.XN << 24) | v2,
                (op << 24)
            ]);
            const constants = new Float64Array([]);
            
            this.outboundQueue.push({
                genome: { opcodes, constants },
                log: {
                    actionApplied: 'INTERACTION_INDUCTION',
                    strategy: 'INTERACTION_INDUCTION',
                    sourceId: 'INTERACTION_MOTIF',
                    op: op,
                    vars: [v1, v2],
                    gen: avgGen,
                    fitnessBefore: 0,
                    fitnessAfter: 0
                }
            });

            return { action: 'STRUCTURAL_PIVOT', intensity: 2.0, regime: 'STRUCTURAL_PIVOT' };
        }



        const characteristics = this.detectDomainCharacteristics();
        if (characteristics.hasAsymptote && (avgGen > 20 || avgStagnation > 5)) {
            this.lastActionGen = avgGen;
            this.currentAction = 'STRUCTURAL_PIVOT';
            console.error(`[Director] ASYMPTOTE DETECTED at x=${characteristics.pivotX!.toFixed(3)}. Proposing STRUCTURAL_PIVOT.`);
            return { action: 'STRUCTURAL_PIVOT', intensity: 2.0, regime: 'STRUCTURAL_PIVOT' };
        }

        if (avgSize < 6 && avgStagnation > 15) {
            this.lastActionGen = avgGen;
            this.currentAction = 'STRUCTURE_FLUSH';
            console.error(`[Director] COLLAPSE DETECTED. Triggering STRUCTURE_FLUSH.`);
            return { action: 'WAIT', intensity: 1.0, regime: 'STRUCTURE_FLUSH' };
        }

        return null;
    }

    public analyzePopulationDynamics(entropy: number, _opcodes: Uint32Array): { intensity: number } {
        const intensity = entropy < 0.2 ? 2.5 : 1.0;
        return { intensity };
    }

    private detectDomainCharacteristics(): { hasAsymptote: boolean, isOscillatory: boolean, pivotX?: number } {
        if (!this.testY || this.testY.length < 10) return { hasAsymptote: false, isOscillatory: false };
        
        let hasAsymptote = false;
        let pivotX = 0;
        for (let i = 1; i < this.testY.length; i++) {
            const diff = Math.abs(this.testY[i] - this.testY[i - 1]);
            if (diff > 50.0) {
                hasAsymptote = true;
                pivotX = this.testX ? this.testX[i] : i;
                break;
            }
        }

        let crossings = 0;
        for (let i = 1; i < this.testY.length; i++) {
            if ((this.testY[i] >= 0 && this.testY[i - 1] < 0) || (this.testY[i] < 0 && this.testY[i - 1] >= 0)) crossings++;
        }
        return { hasAsymptote, isOscillatory: crossings > 15, pivotX };
    }
}
