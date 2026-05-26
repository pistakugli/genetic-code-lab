import {
  EvolutionConfig,
  EvolutionResult,
  EvolutionStats,
  Population,
  Genome,
  FitnessFunction,
  TestCase,
  FunctionDefinition,
  TerminalDefinition,
  ProblemDefinition,
  PopulationConfig
} from './types';
import { ParallelBlockEvaluator } from './parallel-block-evaluator';
import { VectorizedEvaluator } from './vectorized-evaluator';
import { PopulationManager, DEFAULT_POPULATION_CONFIG } from './population';
import {
  getSelectionOperator,
  getCrossoverOperator
} from './operators';
import { PopulationBuffer, OpCode } from './pop-buffer';
import { MigrationHub } from './migration-hub';
import { BlockEvaluator } from './block-evaluator';
import { LinearOperators } from './linear-operators';
import { PostfixBridge } from './postfix-bridge';
import {
  setSeed,
  random,
  getRandomState,
  restoreRandomState,
  generateId,
  deepClone
} from '../utils/helpers';
import { ALPSManager } from './alps';
import { NumericalOptimizer } from './optimizer';
import { SemanticArchive, SemanticArchiveDiagnostics } from './semantic-archive';
import { SemanticRefiner } from './semantic-refiner';
import { SymbolicSimplifier } from './simplifier';
import { BooleanRefiner } from '../solvers/boolean-refiner';
import { applyThresholdCourtFinalizer } from '../solvers/threshold-court-finalizer';
import { DiscoveryAgent, RefinedCandidate } from '../agents/discovery-agent';
import { AsyncAgentBridge } from './async-agent-bridge';
import { evaluateSPEA2Buffer } from './spea2';
import { DomainAnalyzer, DomainProfile } from './domain-analyzer';

export type { EvolutionResult, EvolutionStats, EvolutionConfig, PopulationConfig };

type RegressionBasisCandidate = {
  key: string;
  opcodes: number[];
  constants: number[];
  values: Float64Array;
  complexity: number;
};

export const DEFAULT_EVOLUTION_CONFIG: EvolutionConfig = {
  population: DEFAULT_POPULATION_CONFIG,
  maxGenerations: 100,
  targetFitness: 1.0,
  stagnationLimit: 20,
  parallelEvaluators: 1
};

export class EvolutionEngine<TInput = unknown, TOutput = unknown> {
  private config: EvolutionConfig;
  private populationManager: PopulationManager;
  private functionSet: FunctionDefinition[];
  private terminalSet: TerminalDefinition[];
  private fitnessFunction: FitnessFunction<TInput, TOutput>;
  private testCases: TestCase<TInput, TOutput>[];
  private parallelBlockEvaluator?: ParallelBlockEvaluator;
  private stats: EvolutionStats[] = [];
  private startTime: number = 0;
  private currentGeneration: number = 0;
  private evaluationsCount: number = 0;
  private stagnationCount: number = 0;
  private cataclysmCount: number = 0;
  private selectionIndices!: Uint32Array;
  private bestFitnessSoFar: number = -Infinity;
  private alpsManager?: ALPSManager;
  private discoveryAgent?: DiscoveryAgent;
  private agentBridge?: AsyncAgentBridge;
  private testXBuffer?: Float64Array;
  private testYBuffer?: Float64Array;
  private inputSize: number = 1;
  private currentBuffer?: PopulationBuffer;
  private nextBuffer?: PopulationBuffer;
  private semanticArchive: SemanticArchive = new SemanticArchive();
  private semanticArchiveRecordCycles: number = 0;
  private semanticArchiveInjectionAttempts: number = 0;
  private semanticArchiveInjectionSuccesses: number = 0;
  private opCodes: OpCode[] = [OpCode.ADD, OpCode.SUB, OpCode.MUL, OpCode.DIV, OpCode.SIN, OpCode.COS];
  private domainProfile?: DomainProfile;
  private fitnessMode: 'regression' | 'classification' = 'regression';
  private classificationWeight: number = 0.8;
  private currentRegime: string = 'GLOBAL';

  public onStats?: (stats: any) => void;
  public preferredSource?: number;
  private currentMutationIntensity: number = 1.0;
  private caseScratch: Uint32Array = new Uint32Array(0);
  private casePressure: Float64Array = new Float64Array(0);
  private boundaryCaseMask: Uint8Array = new Uint8Array(0);
  private discretePressureScores: Float64Array = new Float64Array(0);
  private discreteBoundaryScores: Float64Array = new Float64Array(0);
  private discreteLowerBoundaryScores: Float64Array = new Float64Array(0);
  private discreteUpperBoundaryScores: Float64Array = new Float64Array(0);
  private discreteExactAccuracyScores: Float64Array = new Float64Array(0);
  private discreteLowerExactAccuracyScores: Float64Array = new Float64Array(0);
  private discreteUpperExactAccuracyScores: Float64Array = new Float64Array(0);
  private thresholdComposeFocusCount: number = 0;
  private thresholdRoleComposeCount: number = 0;
  private thresholdUpperComposeOpportunityCount: number = 0;
  private thresholdAdaptiveDuelCount: number = 0;
  private thresholdAdaptiveCrossCount: number = 0;
  private thresholdAdaptiveMirroredCrossCount: number = 0;
  private thresholdAdaptiveTriadWitnessCount: number = 0;
  private thresholdAdaptivePairedWitnessCount: number = 0;
  private thresholdAdaptiveGuardedCrossCount: number = 0;
  private thresholdAdaptiveGuardedUpperRepairCount: number = 0;
  private thresholdAdaptiveWitnessedOrCount: number = 0;
  private thresholdAdaptivePairedWitnessIncumbentCount: number = 0;
  private thresholdAdaptivePairedWitnessVetoCount: number = 0;
  private thresholdAdaptiveCrossVetoCount: number = 0;
  private thresholdAdaptiveMirroredCrossVetoCount: number = 0;
  private thresholdAdaptiveTriadWitnessVetoCount: number = 0;
  private thresholdAdaptiveGuardedCrossVetoCount: number = 0;
  private thresholdAdaptiveGuardedUpperRepairVetoCount: number = 0;
  private thresholdAdaptiveWitnessedOrVetoCount: number = 0;
  private thresholdAdaptiveVetoExactRegressionCount: number = 0;
  private thresholdAdaptiveVetoUpperExactRegressionCount: number = 0;
  private thresholdAdaptiveVetoLowerExactRegressionCount: number = 0;
  private thresholdAdaptiveVetoUpperBoundaryRegressionCount: number = 0;
  private thresholdAdaptiveVetoLowerBoundaryRegressionCount: number = 0;
  private thresholdAdaptiveVetoBoundaryRegressionCount: number = 0;
  private thresholdAdaptiveVetoPressureRegressionCount: number = 0;
  private thresholdAdaptiveVetoWeightedRegressionCount: number = 0;
  private thresholdAdaptiveVetoNoMaterialGainCount: number = 0;
  private thresholdAdaptiveShadowWinnerCount: number = 0;
  private thresholdAdaptiveShadowCrossCount: number = 0;
  private thresholdAdaptiveShadowMirroredCrossCount: number = 0;
  private thresholdAdaptiveShadowTriadWitnessCount: number = 0;
  private thresholdAdaptiveShadowGuardedUpperRepairCount: number = 0;
  private thresholdAdaptiveShadowWitnessedOrCount: number = 0;
  private thresholdAdaptivePromotedWinnerCount: number = 0;
  private thresholdAdaptivePromotedCrossCount: number = 0;
  private thresholdAdaptivePromotedMirroredCrossCount: number = 0;
  private thresholdAdaptivePromotedTriadWitnessCount: number = 0;
  private thresholdAdaptivePromotedGuardedUpperRepairCount: number = 0;
  private thresholdAdaptivePromotedWitnessedOrCount: number = 0;
  private thresholdAdaptivePromotionChampionBlockCount: number = 0;
  private thresholdStaticPairedWitnessComposeCount: number = 0;
  private lastAdaptiveUpperComposeVetoCount: number = 0;
  private lastAdaptiveUpperComposeShadowWinner: 'cross' | 'mirrored-cross' | 'triad-witness' | 'guarded-upper-repair' | 'witnessed-or' | null = null;
  private thresholdComposeProbeBuffer?: PopulationBuffer;
  private readonly spea2Warmup: number = 30;
  private problemKey?: string;

  private evaluateBinaryProgram(opcodes: ArrayLike<number>, constants: Float64Array, input: number[]): number {
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
          stack.push((a > 0.5 ? 1 : 0) + (b > 0.5 ? 1 : 0) === 1 ? 1 : 0);
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

  private getBestIndex(): number | null {
    if (!this.currentBuffer) return null;
    return this.getChampionIndex(this.currentBuffer);
  }

  private finalizeBinaryOutput(binary: { opcodes: Uint32Array, constants: Float64Array, fitness: number, generation?: number }): { opcodes: Uint32Array, constants: Float64Array, fitness: number, generation?: number } {
    const threshold = this.domainProfile?.signals.logicThreshold;
    if (!this.isMonotoneThresholdRun() || typeof threshold !== 'number' || this.testCases.length === 0) {
      return binary;
    }
    const finalized = applyThresholdCourtFinalizer(
      this.testCases as unknown as TestCase<number[], number>[],
      threshold,
      binary,
      Math.max(576, this.config.population.maxGenomeSize || 128)
    );
    return {
      ...finalized,
      generation: binary.generation
    };
  }

  constructor(problem: ProblemDefinition<TInput, TOutput>, config: Partial<EvolutionConfig> = {}) {
    const baseConfig = { ...DEFAULT_EVOLUTION_CONFIG, ...config };
    const basePopulation = { ...DEFAULT_POPULATION_CONFIG, ...config.population };
    this.config = { ...baseConfig, population: basePopulation };

    this.functionSet = problem.functionSet;
    this.terminalSet = problem.terminalSet;
    this.fitnessFunction = problem.fitnessFunction;
    this.testCases = problem.testCases;
    this.fitnessMode = problem.fitnessMode || 'regression';
    this.classificationWeight = problem.classificationWeight ?? 0.8;
    this.populationManager = new PopulationManager(this.functionSet, this.terminalSet, this.config.population as PopulationConfig);

    if (this.config.parallelEvaluators > 1 && !this.parallelBlockEvaluator) {
      this.parallelBlockEvaluator = new ParallelBlockEvaluator(
        this.config.parallelEvaluators,
        this.config.population.size,
        this.config.population.maxGenomeSize || 128,
        this.testCases.length,
        10
      );
    }
    if (this.config.alps?.enabled) this.alpsManager = new ALPSManager(this.config.alps);
    if (this.config.seed !== undefined) setSeed(this.config.seed);

    const size = this.config.population.size as number;
    this.selectionIndices = new Uint32Array(size);
    for (let i = 0; i < size; i++) this.selectionIndices[i] = i;
  }

  private hashFloatArray(values?: Float64Array, limit: number = 256): string {
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

  private buildCaseWeightBuffer(): Float64Array {
    const weights = new Float64Array(this.testCases.length);
    for (let i = 0; i < this.testCases.length; i++) {
      weights[i] = this.testCases[i].weight || 1.0;
    }
    return weights;
  }

  private async evaluatePopulationBlock(
    buffer: PopulationBuffer,
    startIndex: number,
    count: number
  ): Promise<Float64Array> {
    if (
      this.parallelBlockEvaluator
      && count > 1
      && this.testXBuffer
      && this.testYBuffer
    ) {
      return this.parallelBlockEvaluator.evaluate(
        buffer,
        startIndex,
        count,
        this.testXBuffer,
        this.testYBuffer,
        this.inputSize,
        this.fitnessMode,
        this.classificationWeight
      );
    }

    return BlockEvaluator.evaluateBlock(
      buffer,
      startIndex,
      count,
      this.testXBuffer!,
      this.testYBuffer!,
      this.inputSize,
      this.fitnessMode,
      this.classificationWeight
    );
  }

  private buildRuntimeCaseWeightBuffer(): Float64Array {
    const weights = this.buildCaseWeightBuffer();
    const threshold = this.domainProfile?.signals.logicThreshold;
    const useAdaptiveWeights =
      this.isMonotoneThresholdRun()
      && this.inputSize >= 9
      && typeof threshold === 'number'
      && this.bestFitnessSoFar >= 0.9;

    if (!useAdaptiveWeights) return weights;

    this.ensureCasePressure(this.testCases.length);
    const alpha = this.stagnationCount >= 18 ? 0.9 : this.stagnationCount >= 10 ? 0.65 : 0.45;
    let sum = 0;
    for (let i = 0; i < weights.length; i++) {
      weights[i] *= 1 + ((this.casePressure[i] - 1) * alpha);
      sum += weights[i];
    }

    sum = 0;
    for (let i = 0; i < weights.length; i++) sum += weights[i];
    const mean = sum / (weights.length || 1);
    if (mean > 0) {
      for (let i = 0; i < weights.length; i++) weights[i] /= mean;
    }

    return weights;
  }

  private applyRuntimeCaseWeights(): void {
    const weights = this.buildRuntimeCaseWeightBuffer();
    if (this.currentBuffer) this.currentBuffer.caseWeights.set(weights);
    if (this.nextBuffer) this.nextBuffer.caseWeights.set(weights);
  }

  private getProblemKey(): string {
    if (!this.problemKey) {
      const weights = this.buildCaseWeightBuffer();
      this.problemKey = `${this.inputSize}:${this.fitnessMode}:${this.testCases.length}:${this.hashFloatArray(this.testXBuffer)}:${this.hashFloatArray(this.testYBuffer)}:${this.hashFloatArray(weights)}`;
    }
    return this.problemKey;
  }

  private isDiscreteRun(): boolean {
    return this.domainProfile?.domainClass === 'discrete' || this.fitnessMode === 'classification';
  }

  private isMonotoneThresholdRun(): boolean {
    const signals = this.domainProfile?.signals;
    return this.isDiscreteRun()
      && typeof signals?.logicThreshold === 'number'
      && (signals.monotonicityScore ?? 0) >= 0.95
      && (signals.symmetryScore ?? 0) >= 0.75;
  }

  private shouldUseSpea2(gen: number): boolean {
    const method = this.config.selectionMethod;
    if (this.isDiscreteRun()) return false;
    if (method === 'lexicase' || method === 'epsilon-lexicase') return false;
    return gen >= this.spea2Warmup;
  }

  private ensureCasePressure(totalCases: number): void {
    if (this.casePressure.length !== totalCases) {
      this.casePressure = new Float64Array(totalCases);
      this.boundaryCaseMask = new Uint8Array(totalCases);
    }

    const threshold = this.domainProfile?.signals.logicThreshold;
    const useThresholdPressure =
      this.isDiscreteRun()
      && this.inputSize >= 7
      && typeof threshold === 'number'
      && this.testCases.length === totalCases;

    if (!useThresholdPressure) {
      for (let i = 0; i < totalCases; i++) {
        this.casePressure[i] = 1;
        this.boundaryCaseMask[i] = 0;
      }
      return;
    }

    const symmetry = Math.max(0, Math.min(1, this.domainProfile?.signals.symmetryScore ?? 0));
    const monotonicity = Math.max(0, Math.min(1, this.domainProfile?.signals.monotonicityScore ?? 0));
    const weightProfile = this.domainProfile?.signals.weightProfile || [];

    for (let i = 0; i < totalCases; i++) {
      const input = this.testCases[i].input as number[];
      let ones = 0;
      for (let j = 0; j < input.length; j++) if ((input[j] as number) > 0.5) ones++;

      const dist = Math.abs(ones - threshold);
      const boundaryKernel = dist === 0 ? 5.0 : dist === 1 ? 3.25 : dist === 2 ? 1.85 : 1 / (1 + dist);
      const symmetryKernel = 1 + symmetry * (dist <= 1 ? 0.9 : dist === 2 ? 0.35 : 0.1);
      const monotonicKernel = 1 + monotonicity * (dist <= 1 ? 0.35 : dist === 2 ? 0.18 : 0.05);
      const profile = ones < weightProfile.length ? weightProfile[ones] : 0.5;
      const ambiguity = 1 - Math.min(1, Math.abs(profile - 0.5) * 2);
      const ambiguityKernel = 1 + ambiguity * (dist <= 1 ? 0.75 : 0.2);

      this.casePressure[i] = boundaryKernel * symmetryKernel * monotonicKernel * ambiguityKernel;
      this.boundaryCaseMask[i] = dist <= 1 ? 1 : 0;
    }
  }

  private buildCaseOrder(totalCases: number): void {
    if (this.caseScratch.length !== totalCases) {
      this.caseScratch = new Uint32Array(totalCases);
      for (let i = 0; i < totalCases; i++) this.caseScratch[i] = i;
    }
    this.ensureCasePressure(totalCases);

    const threshold = this.domainProfile?.signals.logicThreshold;
    const useThresholdOrdering =
      this.shouldUseThresholdSelectionBias()
      && this.inputSize >= 7
      && typeof threshold === 'number'
      && this.testCases.length === totalCases;

    if (!useThresholdOrdering) {
      for (let i = totalCases - 1; i > 0; i--) {
        const j = Math.floor(random() * (i + 1));
        const tmp = this.caseScratch[i];
        this.caseScratch[i] = this.caseScratch[j];
        this.caseScratch[j] = tmp;
      }
      return;
    }

    const buckets = new Map<number, { below: number[]; above: number[] }>();
    for (let i = 0; i < totalCases; i++) {
      const input = this.testCases[i].input as number[];
      let ones = 0;
      for (let j = 0; j < input.length; j++) if ((input[j] as number) > 0.5) ones++;
      const dist = Math.abs(ones - threshold);
      const bucket = buckets.get(dist) || { below: [], above: [] };
      if (ones < threshold) bucket.below.push(i);
      else bucket.above.push(i);
      buckets.set(dist, bucket);
    }

    const shuffle = (arr: number[]) => {
      for (let i = arr.length - 1; i > 0; i--) {
        const j = Math.floor(random() * (i + 1));
        const tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
      }
    };

    const distances = Array.from(buckets.keys()).sort((a, b) => a - b);
    const weakSide = this.getThresholdWeakSide();
    let write = 0;
    for (const dist of distances) {
      const bucket = buckets.get(dist)!;
      shuffle(bucket.below);
      shuffle(bucket.above);
      let bi = 0;
      let ai = 0;
      const startAbove = weakSide === 'upper' ? true : weakSide === 'lower' ? false : random() < 0.5;
      while (bi < bucket.below.length || ai < bucket.above.length) {
        if ((startAbove && ai < bucket.above.length) || bi >= bucket.below.length) {
          this.caseScratch[write++] = bucket.above[ai++];
        }
        if (bi < bucket.below.length) {
          this.caseScratch[write++] = bucket.below[bi++];
        }
        if (!startAbove && ai < bucket.above.length) {
          this.caseScratch[write++] = bucket.above[ai++];
        }
      }
    }
  }

  private refreshDiscreteSelectionScores(buffer: PopulationBuffer): void {
    const size = this.config.population.size;
    if (this.discretePressureScores.length !== size) {
      this.discretePressureScores = new Float64Array(size);
      this.discreteBoundaryScores = new Float64Array(size);
      this.discreteLowerBoundaryScores = new Float64Array(size);
      this.discreteUpperBoundaryScores = new Float64Array(size);
      this.discreteExactAccuracyScores = new Float64Array(size);
      this.discreteLowerExactAccuracyScores = new Float64Array(size);
      this.discreteUpperExactAccuracyScores = new Float64Array(size);
    }
    this.ensureCasePressure(this.testCases.length);

    const threshold = this.domainProfile?.signals.logicThreshold;
    const useThresholdPressure =
      this.isDiscreteRun()
      && this.inputSize >= 7
      && typeof threshold === 'number'
      && this.casePressure.length === this.testCases.length;

    if (!useThresholdPressure) {
      for (let i = 0; i < size; i++) {
        this.discretePressureScores[i] = 0;
        this.discreteBoundaryScores[i] = 0;
        this.discreteLowerBoundaryScores[i] = 0;
        this.discreteUpperBoundaryScores[i] = 0;
        this.discreteExactAccuracyScores[i] = 0;
        this.discreteLowerExactAccuracyScores[i] = 0;
        this.discreteUpperExactAccuracyScores[i] = 0;
      }
      return;
    }

    let totalWeight = 0;
    let boundaryWeight = 0;
    let lowerBoundaryWeight = 0;
    let upperBoundaryWeight = 0;
    let lowerBoundaryCases = 0;
    let upperBoundaryCases = 0;
    for (let t = 0; t < this.testCases.length; t++) {
      const input = this.testCases[t].input as number[];
      let ones = 0;
      for (let j = 0; j < input.length; j++) if ((input[j] as number) > 0.5) ones++;
      totalWeight += this.casePressure[t];
      if (this.boundaryCaseMask[t]) {
        boundaryWeight += this.casePressure[t];
        if (ones < threshold) {
          lowerBoundaryWeight += this.casePressure[t];
          lowerBoundaryCases++;
        } else {
          upperBoundaryWeight += this.casePressure[t];
          upperBoundaryCases++;
        }
      }
    }

    for (let idx = 0; idx < size; idx++) {
      const errBase = idx * buffer.maxTestCases;
      let weightedErr = 0;
      let boundaryErr = 0;
      let lowerBoundaryErr = 0;
      let upperBoundaryErr = 0;
      let correctCount = 0;
      let lowerCorrectCount = 0;
      let upperCorrectCount = 0;
      for (let t = 0; t < this.testCases.length; t++) {
        const err = buffer.caseErrors[errBase + t];
        const pressure = this.casePressure[t];
        weightedErr += err * pressure;
        if (err <= 1e-9) correctCount++;
        if (this.boundaryCaseMask[t]) {
          boundaryErr += err * pressure;
          const input = this.testCases[t].input as number[];
          let ones = 0;
          for (let j = 0; j < input.length; j++) if ((input[j] as number) > 0.5) ones++;
          if (ones < threshold) {
            lowerBoundaryErr += err * pressure;
            if (err <= 1e-9) lowerCorrectCount++;
          } else {
            upperBoundaryErr += err * pressure;
            if (err <= 1e-9) upperCorrectCount++;
          }
        }
      }

      this.discretePressureScores[idx] = weightedErr / (totalWeight || 1);
      this.discreteBoundaryScores[idx] = boundaryErr / (boundaryWeight || totalWeight || 1);
      this.discreteLowerBoundaryScores[idx] = lowerBoundaryErr / (lowerBoundaryWeight || boundaryWeight || totalWeight || 1);
      this.discreteUpperBoundaryScores[idx] = upperBoundaryErr / (upperBoundaryWeight || boundaryWeight || totalWeight || 1);
      this.discreteExactAccuracyScores[idx] = correctCount / Math.max(1, this.testCases.length);
      this.discreteLowerExactAccuracyScores[idx] = lowerCorrectCount / Math.max(1, lowerBoundaryCases);
      this.discreteUpperExactAccuracyScores[idx] = upperCorrectCount / Math.max(1, upperBoundaryCases);
    }
  }

  private getThresholdSelectionPolicy(): 'weighted' | 'hybrid' | 'exact' {
    if (!this.isMonotoneThresholdRun()) return 'weighted';
    return this.config.thresholdSelectionPolicy || 'weighted';
  }

  private shouldUseThresholdSelectionBias(): boolean {
    return this.isMonotoneThresholdRun() && this.getThresholdSelectionPolicy() !== 'weighted';
  }

  private compareWeightedCandidates(buffer: PopulationBuffer, aIdx: number, bIdx: number): number {
    const fitnessDelta = buffer.fitness[bIdx] - buffer.fitness[aIdx];
    if (Math.abs(fitnessDelta) > 1e-9) return fitnessDelta > 0 ? 1 : -1;
    return buffer.lengths[aIdx] - buffer.lengths[bIdx];
  }

  private compareThresholdCandidates(buffer: PopulationBuffer, aIdx: number, bIdx: number): number {
    const policy = this.getThresholdSelectionPolicy();
    if (policy === 'weighted') return this.compareWeightedCandidates(buffer, aIdx, bIdx);

    const exactDelta = this.discreteExactAccuracyScores[bIdx] - this.discreteExactAccuracyScores[aIdx];
    if (Math.abs(exactDelta) > 1e-9) return exactDelta > 0 ? 1 : -1;

    const weightedAnchor = this.getWeightedBestIndex(buffer);
    const weakSide = this.getThresholdWeakSide(weightedAnchor);
    const weakA = weakSide === 'lower'
      ? this.discreteLowerExactAccuracyScores[aIdx]
      : weakSide === 'upper'
        ? this.discreteUpperExactAccuracyScores[aIdx]
        : Math.min(this.discreteLowerExactAccuracyScores[aIdx], this.discreteUpperExactAccuracyScores[aIdx]);
    const weakB = weakSide === 'lower'
      ? this.discreteLowerExactAccuracyScores[bIdx]
      : weakSide === 'upper'
        ? this.discreteUpperExactAccuracyScores[bIdx]
        : Math.min(this.discreteLowerExactAccuracyScores[bIdx], this.discreteUpperExactAccuracyScores[bIdx]);
    const weakDelta = weakB - weakA;
    if (Math.abs(weakDelta) > 1e-9) return weakDelta > 0 ? 1 : -1;

    if (policy === 'hybrid') {
      const weightedDelta = buffer.fitness[bIdx] - buffer.fitness[aIdx];
      if (Math.abs(weightedDelta) > 1e-9) return weightedDelta > 0 ? 1 : -1;
    }

    return buffer.lengths[aIdx] - buffer.lengths[bIdx];
  }

  private getThresholdComposeProbeBuffer(): PopulationBuffer {
    const maxNodes = this.config.population.maxGenomeSize || 128;
    if (
      !this.thresholdComposeProbeBuffer
      || this.thresholdComposeProbeBuffer.maxGenomes !== 6
      || this.thresholdComposeProbeBuffer.getMaxNodesPerGenome() !== maxNodes
      || this.thresholdComposeProbeBuffer.maxTestCases !== this.testCases.length
    ) {
      this.thresholdComposeProbeBuffer = new PopulationBuffer(6, maxNodes, 10, this.testCases.length);
    }
    return this.thresholdComposeProbeBuffer;
  }

  private measureThresholdCandidateMetrics(
    buffer: PopulationBuffer,
    idx: number,
    caseScope: 'all' | 'promotion-audit' = 'all'
  ): {
    fitness: number;
    length: number;
    exactAccuracy: number;
    pressureScore: number;
    boundaryScore: number;
    lowerBoundaryScore: number;
    upperBoundaryScore: number;
    lowerExactAccuracy: number;
    upperExactAccuracy: number;
  } {
    this.ensureCasePressure(this.testCases.length);

    const threshold = this.domainProfile?.signals.logicThreshold;
    const metrics = {
      fitness: buffer.fitness[idx] || 0,
      length: buffer.lengths[idx] || 0,
      exactAccuracy: 0,
      pressureScore: 0,
      boundaryScore: 0,
      lowerBoundaryScore: 0,
      upperBoundaryScore: 0,
      lowerExactAccuracy: 0,
      upperExactAccuracy: 0
    };

    if (
      !this.isDiscreteRun()
      || this.inputSize < 7
      || typeof threshold !== 'number'
      || this.casePressure.length !== this.testCases.length
    ) {
      return metrics;
    }

    let totalWeight = 0;
    let boundaryWeight = 0;
    let lowerBoundaryWeight = 0;
    let upperBoundaryWeight = 0;
    let lowerBoundaryCases = 0;
    let upperBoundaryCases = 0;
    let weightedErr = 0;
    let boundaryErr = 0;
    let lowerBoundaryErr = 0;
    let upperBoundaryErr = 0;
    let consideredCases = 0;
    let correctCount = 0;
    let lowerCorrectCount = 0;
    let upperCorrectCount = 0;
    const errBase = idx * buffer.maxTestCases;

    for (let t = 0; t < this.testCases.length; t++) {
      const input = this.testCases[t].input as number[];
      let ones = 0;
      for (let j = 0; j < input.length; j++) if ((input[j] as number) > 0.5) ones++;
      if (
        caseScope === 'promotion-audit'
        && !this.isAdaptivePromotionAuditCase(t, input, ones, threshold)
      ) {
        continue;
      }

      const pressure = this.casePressure[t];
      const err = buffer.caseErrors[errBase + t];

      consideredCases++;
      totalWeight += pressure;
      weightedErr += err * pressure;
      if (err <= 1e-9) correctCount++;

      if (this.boundaryCaseMask[t]) {
        boundaryWeight += pressure;
        boundaryErr += err * pressure;
        if (ones < threshold) {
          lowerBoundaryWeight += pressure;
          lowerBoundaryErr += err * pressure;
          lowerBoundaryCases++;
          if (err <= 1e-9) lowerCorrectCount++;
        } else {
          upperBoundaryWeight += pressure;
          upperBoundaryErr += err * pressure;
          upperBoundaryCases++;
          if (err <= 1e-9) upperCorrectCount++;
        }
      }
    }

    metrics.exactAccuracy = correctCount / Math.max(1, consideredCases);
    metrics.pressureScore = weightedErr / (totalWeight || 1);
    metrics.boundaryScore = boundaryErr / (boundaryWeight || totalWeight || 1);
    metrics.lowerBoundaryScore = lowerBoundaryErr / (lowerBoundaryWeight || boundaryWeight || totalWeight || 1);
    metrics.upperBoundaryScore = upperBoundaryErr / (upperBoundaryWeight || boundaryWeight || totalWeight || 1);
    metrics.lowerExactAccuracy = lowerCorrectCount / Math.max(1, lowerBoundaryCases);
    metrics.upperExactAccuracy = upperCorrectCount / Math.max(1, upperBoundaryCases);
    return metrics;
  }

  private compareThresholdComposeCandidateMetrics(
    a: ReturnType<EvolutionEngine<TInput, TOutput>['measureThresholdCandidateMetrics']>,
    b: ReturnType<EvolutionEngine<TInput, TOutput>['measureThresholdCandidateMetrics']>,
    weakSide: 'lower' | 'upper'
  ): number {
    const exactDelta = b.exactAccuracy - a.exactAccuracy;
    if (Math.abs(exactDelta) > 1e-9) return exactDelta > 0 ? 1 : -1;

    const weakExactA = weakSide === 'lower' ? a.lowerExactAccuracy : a.upperExactAccuracy;
    const weakExactB = weakSide === 'lower' ? b.lowerExactAccuracy : b.upperExactAccuracy;
    const weakExactDelta = weakExactB - weakExactA;
    if (Math.abs(weakExactDelta) > 1e-9) return weakExactDelta > 0 ? 1 : -1;

    const weakBoundaryA = weakSide === 'lower' ? a.lowerBoundaryScore : a.upperBoundaryScore;
    const weakBoundaryB = weakSide === 'lower' ? b.lowerBoundaryScore : b.upperBoundaryScore;
    const weakBoundaryDelta = weakBoundaryA - weakBoundaryB;
    if (Math.abs(weakBoundaryDelta) > 1e-9) return weakBoundaryDelta > 0 ? 1 : -1;

    const boundaryDelta = a.boundaryScore - b.boundaryScore;
    if (Math.abs(boundaryDelta) > 1e-9) return boundaryDelta > 0 ? 1 : -1;

    const pressureDelta = a.pressureScore - b.pressureScore;
    if (Math.abs(pressureDelta) > 1e-9) return pressureDelta > 0 ? 1 : -1;

    const weightedDelta = b.fitness - a.fitness;
    if (Math.abs(weightedDelta) > 1e-9) return weightedDelta > 0 ? 1 : -1;

    if (a.length !== b.length) return a.length > b.length ? 1 : -1;
    return 0;
  }

  private shouldOfferGuardedUpperRepairCandidate(
    source: ReturnType<EvolutionEngine<TInput, TOutput>['measureThresholdCandidateMetrics']>,
    guarded: ReturnType<EvolutionEngine<TInput, TOutput>['measureThresholdCandidateMetrics']>
  ): boolean {
    const lowerBoundaryGain = source.lowerBoundaryScore - guarded.lowerBoundaryScore;
    const upperBoundaryGain = source.upperBoundaryScore - guarded.upperBoundaryScore;
    const lowerExactGain = guarded.lowerExactAccuracy - source.lowerExactAccuracy;
    const upperExactGain = guarded.upperExactAccuracy - source.upperExactAccuracy;
    const upperExactRegression = source.upperExactAccuracy - guarded.upperExactAccuracy;
    const upperBoundaryRegression = guarded.upperBoundaryScore - source.upperBoundaryScore;
    const exactRegression = source.exactAccuracy - guarded.exactAccuracy;

    if (
      lowerBoundaryGain < 0.02
      && lowerExactGain <= 1e-9
      && upperBoundaryGain < 0.02
      && upperExactGain <= 1e-9
    ) return false;
    if (upperExactRegression > 1e-9) return false;
    if (upperBoundaryRegression > 1e-9) return false;
    if (exactRegression > 1e-9) return false;
    if (guarded.boundaryScore > source.boundaryScore + 1e-9) return false;
    return true;
  }

  private evaluateAdaptiveUpperComposeChallenger(
    incumbent: ReturnType<EvolutionEngine<TInput, TOutput>['measureThresholdCandidateMetrics']>,
    challenger: ReturnType<EvolutionEngine<TInput, TOutput>['measureThresholdCandidateMetrics']>
  ): { accepted: boolean; reasons: string[] } {
    const exactGain = challenger.exactAccuracy - incumbent.exactAccuracy;
    const upperExactGain = challenger.upperExactAccuracy - incumbent.upperExactAccuracy;
    const lowerExactGain = challenger.lowerExactAccuracy - incumbent.lowerExactAccuracy;
    const upperBoundaryGain = incumbent.upperBoundaryScore - challenger.upperBoundaryScore;
    const lowerBoundaryGain = incumbent.lowerBoundaryScore - challenger.lowerBoundaryScore;
    const boundaryGain = incumbent.boundaryScore - challenger.boundaryScore;
    const pressureGain = incumbent.pressureScore - challenger.pressureScore;
    const weightedGain = challenger.fitness - incumbent.fitness;
    const reasons: string[] = [];

    if (exactGain < -1e-9) reasons.push('exact-regression');
    if (upperExactGain < -1e-9) reasons.push('upper-exact-regression');
    if (lowerExactGain < -1e-9) reasons.push('lower-exact-regression');
    if (upperBoundaryGain < -1e-9) reasons.push('upper-boundary-regression');
    if (lowerBoundaryGain < -1e-9) reasons.push('lower-boundary-regression');
    if (boundaryGain < -1e-9) reasons.push('boundary-regression');
    if (pressureGain < -0.005) reasons.push('pressure-regression');
    if (weightedGain < -0.005 && upperExactGain <= 1e-9) reasons.push('weighted-regression');

    if (reasons.length > 0) return { accepted: false, reasons };

    const strongBoundaryGain =
      upperBoundaryGain >= 0.03
      && boundaryGain >= 0.02
      && pressureGain >= 0.01
      && weightedGain >= -1e-9;
    const accepted =
      exactGain > 1e-9
      || (
        upperExactGain > 1e-9
        && boundaryGain >= 0.005
        && pressureGain >= -1e-9
        && weightedGain >= -1e-9
      )
      || strongBoundaryGain;

    return accepted ? { accepted, reasons: [] } : { accepted, reasons: ['no-material-gain'] };
  }

  private shouldAcceptAdaptiveUpperComposeChallenger(
    incumbent: ReturnType<EvolutionEngine<TInput, TOutput>['measureThresholdCandidateMetrics']>,
    challenger: ReturnType<EvolutionEngine<TInput, TOutput>['measureThresholdCandidateMetrics']>
  ): boolean {
    return this.evaluateAdaptiveUpperComposeChallenger(incumbent, challenger).accepted;
  }

  private shouldCommitAdaptiveUpperComposeWinner(): boolean {
    const raw = (process.env.GCL_THRESHOLD_ADAPTIVE_DUEL_COMMIT || '').trim().toLowerCase();
    return raw === '1' || raw === 'true' || raw === 'yes';
  }

  private shouldUseAdaptiveUpperComposePromotion(): boolean {
    const raw = (process.env.GCL_THRESHOLD_ADAPTIVE_DUEL_PROMOTION || '').trim().toLowerCase();
    if (raw === '0' || raw === 'false' || raw === 'no' || raw === 'off') return false;
    if (raw === '1' || raw === 'true' || raw === 'yes' || raw === 'on') return true;
    return true;
  }

  private isAdaptivePromotionAuditCase(caseIndex: number, input: number[], ones: number, threshold: number): boolean {
    const mask = this.getCaseMaskFromInput(input);
    const hashed = Math.imul(mask ^ Math.imul(caseIndex + 1, 0x9e3779b1), 0x85ebca6b) >>> 0;
    if (ones === threshold - 1 || ones === threshold) return hashed % 3 === 0;
    return hashed % 11 === 0;
  }

  private shouldPromoteAdaptiveUpperComposeWinner(
    incumbent: ReturnType<EvolutionEngine<TInput, TOutput>['measureThresholdCandidateMetrics']>,
    challenger: ReturnType<EvolutionEngine<TInput, TOutput>['measureThresholdCandidateMetrics']>,
    incumbentAudit: ReturnType<EvolutionEngine<TInput, TOutput>['measureThresholdCandidateMetrics']>,
    challengerAudit: ReturnType<EvolutionEngine<TInput, TOutput>['measureThresholdCandidateMetrics']>,
    champion: ReturnType<EvolutionEngine<TInput, TOutput>['measureThresholdCandidateMetrics']>,
    championAudit: ReturnType<EvolutionEngine<TInput, TOutput>['measureThresholdCandidateMetrics']>
  ): boolean {
    const globalDecision = this.evaluateAdaptiveUpperComposeChallenger(incumbent, challenger);
    const auditDecision = this.evaluateAdaptiveUpperComposeChallenger(incumbentAudit, challengerAudit);
    if (!globalDecision.accepted || !auditDecision.accepted) return false;

    const globalExactGain = challenger.exactAccuracy - incumbent.exactAccuracy;
    const auditExactGain = challengerAudit.exactAccuracy - incumbentAudit.exactAccuracy;
    const globalBoundaryGain = incumbent.boundaryScore - challenger.boundaryScore;
    const auditBoundaryGain = incumbentAudit.boundaryScore - challengerAudit.boundaryScore;
    const globalPressureGain = incumbent.pressureScore - challenger.pressureScore;
    const auditPressureGain = incumbentAudit.pressureScore - challengerAudit.pressureScore;
    const globalWeightedGain = challenger.fitness - incumbent.fitness;
    const championExactGain = challenger.exactAccuracy - champion.exactAccuracy;
    const championAuditExactGain = challengerAudit.exactAccuracy - championAudit.exactAccuracy;
    const championBoundaryGain = champion.boundaryScore - challenger.boundaryScore;
    const championAuditBoundaryGain = championAudit.boundaryScore - challengerAudit.boundaryScore;
    const championPressureGain = champion.pressureScore - challenger.pressureScore;
    const championWeightedGain = challenger.fitness - champion.fitness;

    const championSafe =
      championExactGain >= -1e-9
      && championAuditExactGain >= -1e-9
      && championBoundaryGain >= 0.005
      && championAuditBoundaryGain >= 0.005
      && championPressureGain >= -1e-9
      && championWeightedGain >= -1e-9;
    const championImproves =
      championExactGain > 1e-9
      || championAuditExactGain > 1e-9
      || (championBoundaryGain >= 0.02 && championAuditBoundaryGain >= 0.02);
    if (!championSafe || !championImproves) {
      this.thresholdAdaptivePromotionChampionBlockCount++;
      return false;
    }

    if (globalExactGain > 1e-9 && auditExactGain >= -1e-9 && auditBoundaryGain >= -1e-9) return true;
    if (auditExactGain > 1e-9 && globalExactGain >= -1e-9 && globalBoundaryGain >= 0.005) return true;
    return (
      globalBoundaryGain >= 0.03
      && auditBoundaryGain >= 0.03
      && globalPressureGain >= 0.01
      && auditPressureGain >= 0.01
      && globalWeightedGain >= -1e-9
    );
  }

  private recordAdaptiveUpperComposeShadowWinner(
    kind: 'cross' | 'mirrored-cross' | 'triad-witness' | 'guarded-upper-repair' | 'witnessed-or'
  ): void {
    this.thresholdAdaptiveShadowWinnerCount++;
    if (kind === 'cross') this.thresholdAdaptiveShadowCrossCount++;
    else if (kind === 'mirrored-cross') this.thresholdAdaptiveShadowMirroredCrossCount++;
    else if (kind === 'triad-witness') this.thresholdAdaptiveShadowTriadWitnessCount++;
    else if (kind === 'guarded-upper-repair') this.thresholdAdaptiveShadowGuardedUpperRepairCount++;
    else this.thresholdAdaptiveShadowWitnessedOrCount++;
  }

  private recordAdaptiveUpperComposePromotion(
    kind: 'cross' | 'mirrored-cross' | 'triad-witness' | 'guarded-upper-repair' | 'witnessed-or'
  ): void {
    this.thresholdAdaptivePromotedWinnerCount++;
    if (kind === 'cross') this.thresholdAdaptivePromotedCrossCount++;
    else if (kind === 'mirrored-cross') this.thresholdAdaptivePromotedMirroredCrossCount++;
    else if (kind === 'triad-witness') this.thresholdAdaptivePromotedTriadWitnessCount++;
    else if (kind === 'guarded-upper-repair') this.thresholdAdaptivePromotedGuardedUpperRepairCount++;
    else this.thresholdAdaptivePromotedWitnessedOrCount++;
  }

  private recordAdaptiveUpperComposeVeto(
    kind: 'cross' | 'mirrored-cross' | 'triad-witness' | 'guarded-upper-repair' | 'witnessed-or',
    reasons: string[]
  ): void {
    if (kind === 'cross') this.thresholdAdaptiveCrossVetoCount++;
    else if (kind === 'mirrored-cross') this.thresholdAdaptiveMirroredCrossVetoCount++;
    else if (kind === 'triad-witness') this.thresholdAdaptiveTriadWitnessVetoCount++;
    else if (kind === 'guarded-upper-repair') this.thresholdAdaptiveGuardedUpperRepairVetoCount++;
    else this.thresholdAdaptiveWitnessedOrVetoCount++;

    for (const reason of reasons) {
      if (reason === 'exact-regression') this.thresholdAdaptiveVetoExactRegressionCount++;
      else if (reason === 'upper-exact-regression') this.thresholdAdaptiveVetoUpperExactRegressionCount++;
      else if (reason === 'lower-exact-regression') this.thresholdAdaptiveVetoLowerExactRegressionCount++;
      else if (reason === 'upper-boundary-regression') this.thresholdAdaptiveVetoUpperBoundaryRegressionCount++;
      else if (reason === 'lower-boundary-regression') this.thresholdAdaptiveVetoLowerBoundaryRegressionCount++;
      else if (reason === 'boundary-regression') this.thresholdAdaptiveVetoBoundaryRegressionCount++;
      else if (reason === 'pressure-regression') this.thresholdAdaptiveVetoPressureRegressionCount++;
      else if (reason === 'weighted-regression') this.thresholdAdaptiveVetoWeightedRegressionCount++;
      else if (reason === 'no-material-gain') this.thresholdAdaptiveVetoNoMaterialGainCount++;
    }
  }

  private getCaseMaskFromInput(input: number[]): number {
    let mask = 0;
    for (let i = 0; i < Math.min(input.length, 31); i++) {
      if (input[i] > 0.5) mask |= (1 << i);
    }
    return mask >>> 0;
  }

  private appendNegativeGuardFromInactiveMask(candidate: number[], inactiveMask: number): boolean {
    const inactiveVars: number[] = [];
    for (let i = 0; i < this.inputSize; i++) {
      if ((inactiveMask & (1 << i)) !== 0) inactiveVars.push(i);
    }
    if (inactiveVars.length === 0) return false;
    for (const variable of inactiveVars) candidate.push(PopulationBuffer.pack(OpCode.XN, variable));
    for (let i = 1; i < inactiveVars.length; i++) candidate.push(PopulationBuffer.pack(OpCode.OR, 0));
    candidate.push(PopulationBuffer.pack(OpCode.AND, 0));
    return true;
  }

  private appendPositiveTermFromActiveMask(candidate: number[], activeMask: number): boolean {
    const activeVars: number[] = [];
    for (let i = 0; i < this.inputSize; i++) {
      if ((activeMask & (1 << i)) !== 0) activeVars.push(i);
    }
    if (activeVars.length === 0) return false;
    for (const variable of activeVars) candidate.push(PopulationBuffer.pack(OpCode.XN, variable));
    for (let i = 1; i < activeVars.length; i++) candidate.push(PopulationBuffer.pack(OpCode.AND, 0));
    candidate.push(PopulationBuffer.pack(OpCode.OR, 0));
    return true;
  }

  private collectWorstLowerFalsePositiveInactiveMask(buffer: PopulationBuffer, idx: number): number | null {
    const threshold = this.domainProfile?.signals.logicThreshold;
    if (typeof threshold !== 'number') return null;

    const limitMask = this.inputSize >= 31 ? 0x7fffffff : ((1 << this.inputSize) - 1);
    const errBase = idx * buffer.maxTestCases;
    let bestMask: number | null = null;
    let bestPressure = -Infinity;

    for (let t = 0; t < this.testCases.length; t++) {
      const err = buffer.caseErrors[errBase + t];
      if (err <= 0.5) continue;

      const input = this.testCases[t].input as number[];
      let ones = 0;
      for (let i = 0; i < input.length; i++) if (input[i] > 0.5) ones++;
      if (ones !== threshold - 1) continue;
      if ((this.testCases[t].expectedOutput as unknown as number) > 0.5) continue;

      const pressure = this.casePressure[t] || 1;
      if (pressure > bestPressure) {
        bestPressure = pressure;
        const caseMask = this.getCaseMaskFromInput(input);
        bestMask = ((~caseMask) & limitMask) >>> 0;
      }
    }

    return bestMask;
  }

  private collectWorstUpperFalseNegativeActiveMasks(buffer: PopulationBuffer, idx: number, limit: number = 3): number[] {
    const threshold = this.domainProfile?.signals.logicThreshold;
    if (typeof threshold !== 'number') return [];

    const errBase = idx * buffer.maxTestCases;
    const candidates: Array<{ mask: number; pressure: number }> = [];
    const seen = new Set<number>();

    for (let t = 0; t < this.testCases.length; t++) {
      const err = buffer.caseErrors[errBase + t];
      if (err <= 0.5) continue;

      const input = this.testCases[t].input as number[];
      let ones = 0;
      for (let i = 0; i < input.length; i++) if (input[i] > 0.5) ones++;
      if (ones !== threshold) continue;
      if ((this.testCases[t].expectedOutput as unknown as number) <= 0.5) continue;

      const pressure = this.casePressure[t] || 1;
      const mask = this.getCaseMaskFromInput(input);
      if (!seen.has(mask)) {
        seen.add(mask);
        candidates.push({ mask, pressure });
      }
    }

    return candidates
      .sort((a, b) => b.pressure - a.pressure)
      .slice(0, Math.max(1, limit))
      .map(candidate => candidate.mask);
  }

  private writeAdaptiveUpperComposeWinner(
    parent1Idx: number,
    parent2Idx: number,
    offspringIdx: number
  ): 'cross' | 'mirrored-cross' | 'triad-witness' | 'guarded-upper-repair' | 'witnessed-or' | 'paired-witness' {
    this.lastAdaptiveUpperComposeShadowWinner = null;
    if (!this.currentBuffer || !this.nextBuffer || !this.testXBuffer || !this.testYBuffer) {
      LinearOperators.crossoverBooleanThresholdCompose(
        this.currentBuffer!,
        parent1Idx,
        parent2Idx,
        this.nextBuffer!,
        offspringIdx,
        this.config.population.maxGenomeSize || 128,
        this.inputSize,
        'upper',
        'paired-witness'
      );
      return 'paired-witness';
    }

    const probeRandomState = getRandomState();
    const probeBuffer = this.getThresholdComposeProbeBuffer();
    probeBuffer.caseWeights.set(this.currentBuffer.caseWeights);

    LinearOperators.crossoverBooleanThresholdCompose(
      this.currentBuffer,
      parent1Idx,
      parent2Idx,
      probeBuffer,
      0,
      this.config.population.maxGenomeSize || 128,
      this.inputSize,
      'upper',
      'cross'
    );
    LinearOperators.crossoverBooleanThresholdCompose(
      this.currentBuffer,
      parent1Idx,
      parent2Idx,
      probeBuffer,
      1,
      this.config.population.maxGenomeSize || 128,
      this.inputSize,
      'upper',
      'mirrored-cross'
    );
    LinearOperators.crossoverBooleanThresholdCompose(
      this.currentBuffer,
      parent1Idx,
      parent2Idx,
      probeBuffer,
      2,
      this.config.population.maxGenomeSize || 128,
      this.inputSize,
      'upper',
      'triad-witness'
    );
    LinearOperators.crossoverBooleanThresholdCompose(
      this.currentBuffer,
      parent1Idx,
      parent2Idx,
      probeBuffer,
      3,
      this.config.population.maxGenomeSize || 128,
      this.inputSize,
      'upper',
      'witnessed-or'
    );
    LinearOperators.crossoverBooleanThresholdCompose(
      this.currentBuffer,
      parent1Idx,
      parent2Idx,
      probeBuffer,
      4,
      this.config.population.maxGenomeSize || 128,
      this.inputSize,
      'upper',
      'paired-witness'
    );

    BlockEvaluator.evaluateBlock(
      probeBuffer,
      0,
      5,
      this.testXBuffer,
      this.testYBuffer,
      this.inputSize,
      this.fitnessMode,
      this.classificationWeight
    );

    let guardedUpperRepairAvailable = false;
    let crossMetrics = this.measureThresholdCandidateMetrics(probeBuffer, 0);
    let mirroredCrossMetrics = this.measureThresholdCandidateMetrics(probeBuffer, 1);
    let triadWitnessMetrics = this.measureThresholdCandidateMetrics(probeBuffer, 2);
    let witnessedMetrics = this.measureThresholdCandidateMetrics(probeBuffer, 3);
    let pairedWitnessMetrics = this.measureThresholdCandidateMetrics(probeBuffer, 4);

    const baseChallengers: Array<{
      idx: number;
      kind: 'cross' | 'mirrored-cross' | 'triad-witness' | 'witnessed-or';
      metrics: ReturnType<EvolutionEngine<TInput, TOutput>['measureThresholdCandidateMetrics']>;
    }> = [
      { idx: 0, kind: 'cross' as const, metrics: crossMetrics },
      { idx: 1, kind: 'mirrored-cross' as const, metrics: mirroredCrossMetrics },
      { idx: 2, kind: 'triad-witness' as const, metrics: triadWitnessMetrics },
      { idx: 3, kind: 'witnessed-or' as const, metrics: witnessedMetrics }
    ];
    let guardedSource = baseChallengers[0];
    for (const challenger of baseChallengers.slice(1)) {
      if (this.compareThresholdComposeCandidateMetrics(guardedSource.metrics, challenger.metrics, 'upper') > 0) {
        guardedSource = challenger;
      }
    }

    const guardedSourceDecision = this.evaluateAdaptiveUpperComposeChallenger(pairedWitnessMetrics, guardedSource.metrics);
    const inactiveMask = guardedSourceDecision.accepted
      ? this.collectWorstLowerFalsePositiveInactiveMask(probeBuffer, guardedSource.idx)
      : null;
    const activeMasks = guardedSourceDecision.accepted
      ? this.collectWorstUpperFalseNegativeActiveMasks(probeBuffer, guardedSource.idx, 2)
      : [];
    if (inactiveMask !== null || activeMasks.length > 0) {
      const guardedCandidate = Array.from(probeBuffer.getOpcodes(guardedSource.idx));
      let repaired = false;
      if (inactiveMask !== null) {
        repaired = this.appendNegativeGuardFromInactiveMask(guardedCandidate, inactiveMask) || repaired;
      }
      for (const activeMask of activeMasks) {
        repaired = this.appendPositiveTermFromActiveMask(guardedCandidate, activeMask) || repaired;
      }
      if (
        repaired
        && guardedCandidate.length <= (this.config.population.maxGenomeSize || 128)
      ) {
        probeBuffer.writeGenome(5, guardedCandidate, new Uint32Array(guardedCandidate.length), probeBuffer.getConstants(guardedSource.idx));
        BlockEvaluator.evaluateBlock(
          probeBuffer,
          5,
          1,
          this.testXBuffer,
          this.testYBuffer,
          this.inputSize,
          this.fitnessMode,
          this.classificationWeight
        );
        const guardedMetrics = this.measureThresholdCandidateMetrics(probeBuffer, 5);
        guardedUpperRepairAvailable = this.shouldOfferGuardedUpperRepairCandidate(guardedSource.metrics, guardedMetrics);
      }
    }

    const candidateMetrics: Array<{
      idx: number;
      kind: 'cross' | 'mirrored-cross' | 'triad-witness' | 'guarded-upper-repair' | 'witnessed-or' | 'paired-witness';
      metrics: ReturnType<EvolutionEngine<TInput, TOutput>['measureThresholdCandidateMetrics']>;
    }> = [
      ...baseChallengers,
      { idx: 4, kind: 'paired-witness' as const, metrics: pairedWitnessMetrics }
    ];
    if (guardedUpperRepairAvailable) {
      candidateMetrics.push({ idx: 5, kind: 'guarded-upper-repair' as const, metrics: this.measureThresholdCandidateMetrics(probeBuffer, 5) });
    }

    let winner = candidateMetrics[4];
    this.lastAdaptiveUpperComposeVetoCount = 0;
    for (let i = 0; i < candidateMetrics.length; i++) {
      const challenger = candidateMetrics[i];
      if (challenger.kind === 'paired-witness') continue;
      const wouldBeatIncumbent = this.compareThresholdComposeCandidateMetrics(pairedWitnessMetrics, challenger.metrics, 'upper') > 0;
      const decision = this.evaluateAdaptiveUpperComposeChallenger(pairedWitnessMetrics, challenger.metrics);
      const accepted = decision.accepted;
      if (wouldBeatIncumbent && !accepted) {
        this.lastAdaptiveUpperComposeVetoCount++;
        this.recordAdaptiveUpperComposeVeto(challenger.kind, decision.reasons);
      }
      if (
        accepted
        && this.compareThresholdComposeCandidateMetrics(winner.metrics, challenger.metrics, 'upper') > 0
      ) {
        winner = challenger;
      }
    }

    restoreRandomState(probeRandomState);
    if (!this.shouldCommitAdaptiveUpperComposeWinner() && winner.kind !== 'paired-witness') {
      const incumbentAuditMetrics = this.measureThresholdCandidateMetrics(probeBuffer, 4, 'promotion-audit');
      const winnerAuditMetrics = this.measureThresholdCandidateMetrics(probeBuffer, winner.idx, 'promotion-audit');
      const championIdx = this.selectionIndices.length > 0 ? this.selectionIndices[0] : 0;
      const championMetrics = this.measureThresholdCandidateMetrics(this.currentBuffer, championIdx);
      const championAuditMetrics = this.measureThresholdCandidateMetrics(this.currentBuffer, championIdx, 'promotion-audit');
      if (
        this.shouldUseAdaptiveUpperComposePromotion()
        && this.shouldPromoteAdaptiveUpperComposeWinner(
        pairedWitnessMetrics,
        winner.metrics,
        incumbentAuditMetrics,
        winnerAuditMetrics,
        championMetrics,
        championAuditMetrics
        )
      ) {
        this.recordAdaptiveUpperComposePromotion(winner.kind);
      } else {
      this.lastAdaptiveUpperComposeShadowWinner = winner.kind;
      this.recordAdaptiveUpperComposeShadowWinner(winner.kind);
      LinearOperators.crossoverBooleanThresholdCompose(
        this.currentBuffer,
        parent1Idx,
        parent2Idx,
        this.nextBuffer,
        offspringIdx,
        this.config.population.maxGenomeSize || 128,
        this.inputSize,
        'upper',
        'paired-witness'
      );
      return 'paired-witness';
      }
    }

    if (winner.kind !== 'guarded-upper-repair') {
      LinearOperators.crossoverBooleanThresholdCompose(
        this.currentBuffer,
        parent1Idx,
        parent2Idx,
        this.nextBuffer,
        offspringIdx,
        this.config.population.maxGenomeSize || 128,
        this.inputSize,
        'upper',
        winner.kind
      );
      return winner.kind;
    }

    this.nextBuffer.copyGenome(winner.idx, offspringIdx, probeBuffer);
    return winner.kind;
  }

  private sortPopulationSelection(buffer: PopulationBuffer, useSpea2: boolean): void {
    if (useSpea2) {
      this.selectionIndices.sort((a, b) => buffer.spea2Score[a] - buffer.spea2Score[b]);
      return;
    }

    if (this.getThresholdSelectionPolicy() !== 'weighted' && this.isMonotoneThresholdRun()) {
      this.refreshDiscreteSelectionScores(buffer);
      this.selectionIndices.sort((a, b) => this.compareThresholdCandidates(buffer, a, b));
      return;
    }

    this.selectionIndices.sort((a, b) => this.compareWeightedCandidates(buffer, a, b));
  }

  private getWeightedBestIndex(buffer: PopulationBuffer): number {
    let bestIdx = 0;
    for (let i = 1; i < this.config.population.size; i++) {
      if (buffer.fitness[i] > buffer.fitness[bestIdx]) bestIdx = i;
    }
    return bestIdx;
  }

  private getChampionIndex(buffer: PopulationBuffer): number {
    if (this.getThresholdSelectionPolicy() !== 'weighted' && this.isMonotoneThresholdRun()) {
      this.refreshDiscreteSelectionScores(buffer);
      let bestIdx = 0;
      for (let i = 1; i < this.config.population.size; i++) {
        if (this.compareThresholdCandidates(buffer, i, bestIdx) < 0) bestIdx = i;
      }
      return bestIdx;
    }
    return this.getWeightedBestIndex(buffer);
  }

  private getChampionScore(buffer: PopulationBuffer, idx: number): number {
    if (this.getThresholdSelectionPolicy() !== 'weighted' && this.isMonotoneThresholdRun()) {
      return this.discreteExactAccuracyScores[idx] || 0;
    }
    return buffer.fitness[idx] || 0;
  }

  private getThresholdWeakSide(referenceIdx?: number): 'lower' | 'upper' | null {
    if (!this.isMonotoneThresholdRun()) return null;
    if (
      this.discreteLowerBoundaryScores.length !== this.config.population.size
      || this.discreteUpperBoundaryScores.length !== this.config.population.size
    ) {
      return null;
    }

    const idx = typeof referenceIdx === 'number'
      ? referenceIdx
      : (this.selectionIndices.length > 0 ? this.selectionIndices[0] : -1);
    if (idx < 0 || idx >= this.config.population.size) return null;

    const lower = this.discreteLowerBoundaryScores[idx];
    const upper = this.discreteUpperBoundaryScores[idx];
    if (!Number.isFinite(lower) || !Number.isFinite(upper)) return null;
    if (lower > upper + 1e-9) return 'lower';
    if (upper > lower + 1e-9) return 'upper';
    return null;
  }

  private getThresholdBreederPoolLimit(): number {
    return Math.min(this.selectionIndices.length, Math.max(16, this.config.population.eliteCount * 6));
  }

  private selectBoundaryRoleParent(
    buffer: PopulationBuffer,
    side: 'lower' | 'upper',
    excludeIdx?: number,
    anchorIdx?: number
  ): number | null {
    const thresholdRun = this.isMonotoneThresholdRun();
    if (!thresholdRun || this.inputSize < 9) return null;
    if (this.stagnationCount <= 5 && this.currentMutationIntensity <= 1.5) return null;
    if (this.selectionIndices.length === 0) return null;

    const currentBest = this.getChampionScore(buffer, this.selectionIndices[0]) || 0;
    if (currentBest < 0.9 || currentBest >= 0.999999) return null;

    const sideScores = side === 'lower' ? this.discreteLowerBoundaryScores : this.discreteUpperBoundaryScores;
    const oppositeScores = side === 'lower' ? this.discreteUpperBoundaryScores : this.discreteLowerBoundaryScores;
    if (sideScores.length !== this.config.population.size) return null;

    let bestIdx = -1;
    let bestScore = Infinity;
    let bestComposite = -Infinity;
    let bestFitness = -Infinity;
    let bestLen = Infinity;
    let bestOpposite = Infinity;

    const anchorSideScore = typeof anchorIdx === 'number' ? sideScores[anchorIdx] : Infinity;
    const anchorLowerScore = typeof anchorIdx === 'number' ? this.discreteLowerBoundaryScores[anchorIdx] : Infinity;
    const anchorUpperScore = typeof anchorIdx === 'number' ? this.discreteUpperBoundaryScores[anchorIdx] : Infinity;
    const poolLimit = this.getThresholdBreederPoolLimit();
    for (let rank = 0; rank < poolLimit; rank++) {
      const idx = this.selectionIndices[rank];
      if (idx === excludeIdx) continue;
      if (idx === anchorIdx) continue;

      const sideErr = sideScores[idx];
      const oppositeErr = oppositeScores[idx];
      const pressure = this.discretePressureScores[idx];
      const fitness = buffer.fitness[idx];
      const len = buffer.lengths[idx];
      if (typeof anchorIdx === 'number' && sideErr > anchorSideScore + 1e-9) continue;

      const profileDistance = typeof anchorIdx === 'number'
        ? Math.abs(this.discreteLowerBoundaryScores[idx] - anchorLowerScore)
          + Math.abs(this.discreteUpperBoundaryScores[idx] - anchorUpperScore)
        : 0;
      const sideGain = typeof anchorIdx === 'number' ? Math.max(0, anchorSideScore - sideErr) : 0;
      const compositeScore = typeof anchorIdx === 'number'
        ? (sideGain * 6.0)
          + ((1 - oppositeErr) * 1.5)
          + Math.min(0.75, profileDistance * 1.5)
          + (fitness * 1.0)
          - (len * 0.0004)
        : (-sideErr * 5.0)
          - (pressure * 1.25)
          + (fitness * 1.0)
          - (len * 0.0004);

      if (compositeScore > bestComposite + 1e-9) {
        bestIdx = idx;
        bestComposite = compositeScore;
        bestScore = sideErr;
        bestOpposite = oppositeErr;
        bestFitness = fitness;
        bestLen = len;
        continue;
      }

      if (Math.abs(compositeScore - bestComposite) <= 1e-9) {
        const sideBetter = sideErr < bestScore - 1e-9;
        const oppositeBetter =
          Math.abs(sideErr - bestScore) <= 1e-9
          && oppositeErr < bestOpposite - 1e-9;
        const fitnessBetter =
          Math.abs(sideErr - bestScore) <= 1e-9
          && Math.abs(oppositeErr - bestOpposite) <= 1e-9
          && fitness > bestFitness + 1e-9;
        const lenBetter =
          Math.abs(sideErr - bestScore) <= 1e-9
          && Math.abs(oppositeErr - bestOpposite) <= 1e-9
          && Math.abs(fitness - bestFitness) <= 1e-9
          && len < bestLen;
        if (sideBetter || oppositeBetter || fitnessBetter || lenBetter) {
          bestIdx = idx;
          bestComposite = compositeScore;
          bestScore = sideErr;
          bestOpposite = oppositeErr;
          bestFitness = fitness;
          bestLen = len;
        }
      }
    }

    return bestIdx >= 0 ? bestIdx : null;
  }

  private selectDiscreteLexicaseWinner(buffer: PopulationBuffer, pool: Uint32Array, poolLength: number): number {
    let best = pool[0];
    const useThresholdTieBreak =
      this.shouldUseThresholdSelectionBias()
      && this.inputSize >= 7
      && typeof this.domainProfile?.signals.logicThreshold === 'number'
      && this.discretePressureScores.length === this.config.population.size;
    const weakSide = useThresholdTieBreak ? this.getThresholdWeakSide() : null;
    const weakSideScores = weakSide === 'lower'
      ? this.discreteLowerBoundaryScores
      : weakSide === 'upper'
        ? this.discreteUpperBoundaryScores
        : null;

    let bestWeakBoundary = weakSideScores ? weakSideScores[best] : Infinity;
    let bestBoundary = useThresholdTieBreak ? this.discreteBoundaryScores[best] : Infinity;
    let bestPressure = useThresholdTieBreak ? this.discretePressureScores[best] : Infinity;
    let bestFitness = buffer.fitness[best];
    let bestLen = buffer.lengths[best];

    for (let i = 1; i < poolLength; i++) {
      const idx = pool[i];
      const weakBoundary = weakSideScores ? weakSideScores[idx] : Infinity;
      const boundary = useThresholdTieBreak ? this.discreteBoundaryScores[idx] : Infinity;
      const pressure = useThresholdTieBreak ? this.discretePressureScores[idx] : Infinity;
      const fitness = buffer.fitness[idx];
      const len = buffer.lengths[idx];

      const weakBoundaryBetter = weakBoundary < bestWeakBoundary - 1e-9;
      const boundaryBetter =
        Math.abs(weakBoundary - bestWeakBoundary) <= 1e-9
        && boundary < bestBoundary - 1e-9;
      const pressureBetter =
        Math.abs(weakBoundary - bestWeakBoundary) <= 1e-9
        && 
        Math.abs(boundary - bestBoundary) <= 1e-9
        && pressure < bestPressure - 1e-9;
      const fitnessBetter =
        Math.abs(weakBoundary - bestWeakBoundary) <= 1e-9
        &&
        Math.abs(boundary - bestBoundary) <= 1e-9
        && Math.abs(pressure - bestPressure) <= 1e-9
        && fitness > bestFitness + 1e-9;
      const lengthBetter =
        Math.abs(weakBoundary - bestWeakBoundary) <= 1e-9
        &&
        Math.abs(boundary - bestBoundary) <= 1e-9
        && Math.abs(pressure - bestPressure) <= 1e-9
        && Math.abs(fitness - bestFitness) <= 1e-9
        && len < bestLen;

      if (weakBoundaryBetter || boundaryBetter || pressureBetter || fitnessBetter || lengthBetter) {
        best = idx;
        bestWeakBoundary = weakBoundary;
        bestBoundary = boundary;
        bestPressure = pressure;
        bestFitness = fitness;
        bestLen = len;
      }
    }

    return best;
  }

  private syncAgentEvaluationContext(): void {
    if (!this.testXBuffer || !this.testYBuffer) return;
    const weights = this.buildRuntimeCaseWeightBuffer();
    const problemKey = this.getProblemKey();
    if (this.agentBridge) {
      this.agentBridge.setEvaluationData(this.testXBuffer, this.testYBuffer, this.inputSize, this.fitnessMode, weights, problemKey);
      if (this.domainProfile) this.agentBridge.setDomainProfile(this.domainProfile);
    }
    if (this.discoveryAgent) {
      this.discoveryAgent.setEvaluationData(this.testXBuffer, this.testYBuffer, this.inputSize, this.fitnessMode, weights, problemKey);
      if (this.domainProfile) this.discoveryAgent.setDomainProfile(this.domainProfile);
    }
  }

  private isLogicSafeGenome(opcodes: Uint32Array): boolean {
    const logicSafeOps = new Set<OpCode>([OpCode.XN, OpCode.CONST, OpCode.AND, OpCode.OR, OpCode.NAND, OpCode.NOR, OpCode.XOR, OpCode.ITE]);
    for (let i = 0; i < opcodes.length; i++) {
      const opcode = (opcodes[i] >>> 24) as OpCode;
      if (!logicSafeOps.has(opcode)) return false;
    }
    return true;
  }

  private hasValidInputReferences(opcodes: Uint32Array): boolean {
    for (let i = 0; i < opcodes.length; i++) {
      const opcode = (opcodes[i] >>> 24) as OpCode;
      const data = opcodes[i] & 0xFFFFFF;
      if (opcode === OpCode.XN && data >= this.inputSize) return false;
    }
    return true;
  }

  /**
   * PHASE 64.2.3: Early Domain Analysis (Centralized)
   * This is triggered BEFORE Gen 0 randomization to anchor the opCodes.
   */
  private performDomainAnalysis(): void {
    if (this.testXBuffer && this.testYBuffer && !this.domainProfile) {
      this.domainProfile = DomainAnalyzer.analyze(this.testXBuffer, this.testYBuffer, this.inputSize);
      // console.error(`[EvolutionEngine] DOMAIN AUDIT: ${this.domainProfile.domainClass.toUpperCase()} (Confidence: ${this.domainProfile.confidence.toFixed(2)})`);
      this.syncAgentEvaluationContext();

      // Lock initial opCodes to the domain invariants
      this.opCodes = this.getEffectiveOperatorSet();
    }
  }

  private buildRegressionBasisSeed(): { opcodes: Uint32Array; constants: Float64Array } | null {
    if (
      process.env.GCL_REGRESSION_BASIS_SEED === '0'
      || this.fitnessMode !== 'regression'
      || this.domainProfile?.domainClass === 'discrete'
      || !this.testXBuffer
      || !this.testYBuffer
      || this.testYBuffer.length < 3
    ) {
      return null;
    }

    const candidates = this.buildRegressionBasisCandidates();
    if (candidates.length === 0) return null;
    const selectedTerms = this.selectRegressionBasisTerms(candidates, this.testYBuffer);
    if (selectedTerms.length === 0) return null;

    const opcodes: number[] = [];
    const constants: number[] = [];
    let emittedTerms = 0;

    for (const term of selectedTerms) {
      if (constants.length + 1 + term.candidate.constants.length > 10) continue;
      this.appendRegressionBasisTerm(opcodes, constants, term.candidate, term.coefficient);
      emittedTerms++;
      if (emittedTerms > 1) opcodes.push(PopulationBuffer.pack(OpCode.ADD, 0));
    }

    if (emittedTerms === 0) {
      opcodes.push(PopulationBuffer.pack(OpCode.CONST, 0));
      constants.push(0);
    }

    if (opcodes.length > (this.config.population.maxGenomeSize || 128)) return null;
    const paddedConstants = new Float64Array(10);
    for (let i = 0; i < constants.length && i < paddedConstants.length; i++) paddedConstants[i] = constants[i];
    return { opcodes: new Uint32Array(opcodes), constants: paddedConstants };
  }

  private buildRegressionBasisCandidates(): RegressionBasisCandidate[] {
    const allowed = new Set(this.functionSet.map(fn => fn.name.toLowerCase()));
    if (!allowed.has('add')) return [];

    const candidates: RegressionBasisCandidate[] = [];
    const seen = new Set<string>();
    const caseCount = this.testYBuffer?.length || 0;
    const maxComplexity = Math.max(3, Math.min(8, (this.config.population.maxGenomeDepth || 5) + 1));

    const addCandidate = (candidate: RegressionBasisCandidate) => {
      if (candidates.length >= 2048) return;
      if (seen.has(candidate.key)) return;
      if (candidate.opcodes.length > Math.max(1, (this.config.population.maxGenomeSize || 128) - 3)) return;
      if (!this.valuesAreUsable(candidate.values, candidate.key === '1')) return;
      seen.add(candidate.key);
      candidates.push(candidate);
    };

    addCandidate({
      key: '1',
      opcodes: [PopulationBuffer.pack(OpCode.CONST, 0)],
      constants: [1],
      values: new Float64Array(caseCount).fill(1),
      complexity: 1
    });

    for (let variableIndex = 0; variableIndex < this.inputSize; variableIndex++) {
      const values = new Float64Array(caseCount);
      for (let row = 0; row < caseCount; row++) values[row] = this.testXBuffer![row * this.inputSize + variableIndex];
      addCandidate({
        key: this.inputSize === 1 ? 'x' : `x${variableIndex}`,
        opcodes: [PopulationBuffer.pack(OpCode.XN, variableIndex)],
        constants: [],
        values,
        complexity: 1
      });
    }

    for (let round = 0; round < maxComplexity; round++) {
      const snapshot = [...candidates];
      const unaryOps: Array<{ name: string; opcode: OpCode; apply: (value: number) => number; skipConstant?: boolean }> = [
        { name: 'sin', opcode: OpCode.SIN, apply: Math.sin, skipConstant: true },
        { name: 'cos', opcode: OpCode.COS, apply: Math.cos, skipConstant: true },
        { name: 'sqrt', opcode: OpCode.SQRT, apply: value => value >= 0 ? Math.sqrt(value) : 0 },
        { name: 'log', opcode: OpCode.LOG, apply: value => Math.log(Math.abs(value) + 1e-12), skipConstant: true },
        { name: 'exp', opcode: OpCode.EXP, apply: value => Math.exp(Math.max(-40, Math.min(40, value))), skipConstant: true }
      ];

      for (const unary of unaryOps) {
        if (!allowed.has(unary.name)) continue;
        for (const base of snapshot) {
          if ((unary.skipConstant && base.key === '1') || base.complexity + 1 > maxComplexity) continue;
          const values = new Float64Array(caseCount);
          for (let i = 0; i < caseCount; i++) values[i] = unary.apply(base.values[i]);
          addCandidate({
            key: `${unary.name}(${base.key})`,
            opcodes: [...base.opcodes, PopulationBuffer.pack(unary.opcode, 0)],
            constants: [...base.constants],
            values,
            complexity: base.complexity + 1
          });
        }
      }
      const binaryOps: Array<{ name: string; opcode: OpCode; apply: (left: number, right: number) => number; commutative: boolean; skipUnit?: boolean }> = [
        { name: 'div', opcode: OpCode.DIV, apply: (left, right) => left / (right >= 0 ? right + 1e-4 : right - 1e-4), commutative: false, skipUnit: true },
        { name: 'mul', opcode: OpCode.MUL, apply: (left, right) => left * right, commutative: true },
        { name: 'add', opcode: OpCode.ADD, apply: (left, right) => left + right, commutative: true },
        { name: 'sub', opcode: OpCode.SUB, apply: (left, right) => left - right, commutative: false }
      ];

      for (const binary of binaryOps) {
        if (!allowed.has(binary.name)) continue;
        for (let i = 0; i < snapshot.length; i++) {
          const left = snapshot[i];
          if (binary.skipUnit && left.key === '1') continue;
          const jStart = binary.commutative ? i : 0;
          for (let j = jStart; j < snapshot.length; j++) {
            const right = snapshot[j];
            if (binary.skipUnit && right.key === '1') continue;
            const complexity = left.complexity + right.complexity + 1;
            if (complexity > maxComplexity) continue;
            const ordered = binary.commutative && left.key > right.key ? [right, left] : [left, right];
            const values = new Float64Array(caseCount);
            for (let row = 0; row < caseCount; row++) values[row] = binary.apply(ordered[0].values[row], ordered[1].values[row]);
            addCandidate({
              key: `${binary.name}(${ordered[0].key},${ordered[1].key})`,
              opcodes: [...ordered[0].opcodes, ...this.shiftCandidateConstants(ordered[1].opcodes, ordered[0].constants.length), PopulationBuffer.pack(binary.opcode, 0)],
              constants: [...ordered[0].constants, ...ordered[1].constants],
              values,
              complexity
            });
          }
        }
      }
      if (candidates.length >= 2048) break;
    }

    return candidates.sort((a, b) => a.complexity - b.complexity || a.key.localeCompare(b.key));
  }

  private valuesAreUsable(values: Float64Array, allowConstant: boolean): boolean {
    let min = Number.POSITIVE_INFINITY;
    let max = Number.NEGATIVE_INFINITY;
    for (let i = 0; i < values.length; i++) {
      const value = values[i];
      if (!Number.isFinite(value) || Math.abs(value) > 1e9) return false;
      min = Math.min(min, value);
      max = Math.max(max, value);
    }
    return allowConstant || max - min > 1e-10;
  }

  private shiftCandidateConstants(opcodes: number[], shift: number): number[] {
    if (shift === 0) return [...opcodes];
    return opcodes.map(instruction => {
      const op = (instruction >>> 24) as OpCode;
      const data = instruction & 0xFFFFFF;
      return op === OpCode.CONST ? PopulationBuffer.pack(OpCode.CONST, data + shift) : instruction;
    });
  }

  private selectRegressionBasisTerms(candidates: RegressionBasisCandidate[], target: Float64Array): Array<{ candidate: RegressionBasisCandidate; coefficient: number }> {
    const selected: RegressionBasisCandidate[] = [];
    let selectedCoefficients: number[] = [];
    let bestMae = Number.POSITIVE_INFINITY;
    const maxTerms = Math.min(8, candidates.length);

    for (let step = 0; step < maxTerms; step++) {
      let bestTrial: { candidate: RegressionBasisCandidate; coefficients: number[]; mae: number } | null = null;
      for (const candidate of candidates) {
        if (selected.some(term => term.key === candidate.key)) continue;
        const trial = [...selected, candidate];
        const coefficients = this.fitLinearBasis(trial.map(term => term.values), target);
        const mae = this.measureLinearBasisMae(trial, coefficients, target);
        if (!Number.isFinite(mae)) continue;
        const adjustedMae = mae
          + candidate.complexity * 1e-8
          + this.measureBasisStabilityPenalty(trial, coefficients) * 0.05;
        if (!bestTrial || adjustedMae < bestTrial.mae) {
          bestTrial = { candidate, coefficients, mae: adjustedMae };
        }
      }
      if (!bestTrial) break;
      if (bestMae - bestTrial.mae < 1e-10 && selected.length > 0) break;
      selected.push(bestTrial.candidate);
      selectedCoefficients = bestTrial.coefficients;
      bestMae = bestTrial.mae;
      if (bestMae < 1e-10) break;
    }

    return selected
      .map((candidate, index) => ({ candidate, coefficient: selectedCoefficients[index] || 0 }))
      .filter(term => Number.isFinite(term.coefficient) && Math.abs(term.coefficient) > 1e-9);
  }

  private measureLinearBasisMae(candidates: RegressionBasisCandidate[], coefficients: number[], target: Float64Array): number {
    let total = 0;
    for (let row = 0; row < target.length; row++) {
      let predicted = 0;
      for (let col = 0; col < candidates.length; col++) predicted += (coefficients[col] || 0) * candidates[col].values[row];
      total += Math.abs(predicted - target[row]);
    }
    return total / Math.max(1, target.length);
  }

  private measureBasisStabilityPenalty(candidates: RegressionBasisCandidate[], coefficients: number[]): number {
    if (!this.testXBuffer || !this.testYBuffer || this.inputSize !== 1 || this.testYBuffer.length < 2) return 0;
    let minX = Number.POSITIVE_INFINITY;
    let maxX = Number.NEGATIVE_INFINITY;
    for (let i = 0; i < this.testYBuffer.length; i++) {
      const x = this.testXBuffer[i * this.inputSize];
      minX = Math.min(minX, x);
      maxX = Math.max(maxX, x);
    }
    const span = maxX - minX;
    if (!(span > 0)) return 0;

    const samples = 9;
    let penalty = 0;
    for (let i = 0; i < samples; i++) {
      const x = minX + ((i + 0.5) * span) / samples;
      const context = new Map<string, number>();
      for (let c = 0; c < candidates.length; c++) {
        const value = this.evaluateBasisCandidateAt(candidates[c], x, context);
        if (!Number.isFinite(value) || Math.abs(value) > 1e6) {
          penalty += 1;
          continue;
        }
        const contribution = Math.abs((coefficients[c] || 0) * value);
        if (contribution > 1e4) penalty += Math.min(10, contribution / 1e4);
      }
    }
    return penalty / samples;
  }

  private evaluateBasisCandidateAt(candidate: RegressionBasisCandidate, x: number, context: Map<string, number>): number {
    const cached = context.get(candidate.key);
    if (cached !== undefined) return cached;

    const stack: number[] = [];
    for (let i = 0; i < candidate.opcodes.length; i++) {
      const instruction = candidate.opcodes[i];
      const op = (instruction >>> 24) as OpCode;
      const data = instruction & 0xFFFFFF;
      switch (op) {
        case OpCode.XN:
          stack.push(data === 0 ? x : 0);
          break;
        case OpCode.CONST:
          stack.push(candidate.constants[data] ?? 0);
          break;
        case OpCode.ADD: {
          const b = stack.pop() || 0;
          const a = stack.pop() || 0;
          stack.push(a + b);
          break;
        }
        case OpCode.SUB: {
          const b = stack.pop() || 0;
          const a = stack.pop() || 0;
          stack.push(a - b);
          break;
        }
        case OpCode.MUL: {
          const b = stack.pop() || 0;
          const a = stack.pop() || 0;
          stack.push(a * b);
          break;
        }
        case OpCode.DIV: {
          const b = stack.pop() || 0;
          const a = stack.pop() || 0;
          stack.push(a / (b >= 0 ? b + 1e-4 : b - 1e-4));
          break;
        }
        case OpCode.SIN:
          stack.push(Math.sin(stack.pop() || 0));
          break;
        case OpCode.COS:
          stack.push(Math.cos(stack.pop() || 0));
          break;
        case OpCode.SQRT: {
          const value = stack.pop() || 0;
          stack.push(value >= 0 ? Math.sqrt(value) : 0);
          break;
        }
        case OpCode.LOG:
          stack.push(Math.log(Math.abs(stack.pop() || 0) + 1e-12));
          break;
        case OpCode.EXP:
          stack.push(Math.exp(Math.max(-40, Math.min(40, stack.pop() || 0))));
          break;
      }
    }
    const result = stack.length > 0 ? stack[stack.length - 1] : 0;
    context.set(candidate.key, result);
    return result;
  }

  private appendRegressionBasisTerm(opcodes: number[], constants: number[], candidate: RegressionBasisCandidate, coefficient: number): void {
    if (candidate.key === '1') {
      opcodes.push(PopulationBuffer.pack(OpCode.CONST, constants.length));
      constants.push(coefficient);
      return;
    }

    const cIdx = constants.length;
    constants.push(coefficient);
    opcodes.push(PopulationBuffer.pack(OpCode.CONST, cIdx));
    opcodes.push(...this.shiftCandidateConstants(candidate.opcodes, constants.length));
    constants.push(...candidate.constants);
    opcodes.push(PopulationBuffer.pack(OpCode.MUL, 0));
  }

  private fitLinearBasis(columns: Float64Array[], target: Float64Array): number[] {
    const n = columns.length;
    const matrix = Array.from({ length: n }, () => new Array<number>(n).fill(0));
    const rhs = new Array<number>(n).fill(0);

    for (let row = 0; row < target.length; row++) {
      for (let i = 0; i < n; i++) {
        const vi = columns[i][row];
        rhs[i] += vi * target[row];
        for (let j = 0; j < n; j++) matrix[i][j] += vi * columns[j][row];
      }
    }
    for (let i = 1; i < n; i++) matrix[i][i] += 1e-10;
    return this.solveLinearSystem(matrix, rhs);
  }

  private solveLinearSystem(matrix: number[][], rhs: number[]): number[] {
    const n = rhs.length;
    const a = matrix.map((row, i) => [...row, rhs[i]]);

    for (let pivot = 0; pivot < n; pivot++) {
      let best = pivot;
      for (let row = pivot + 1; row < n; row++) {
        if (Math.abs(a[row][pivot]) > Math.abs(a[best][pivot])) best = row;
      }
      if (Math.abs(a[best][pivot]) < 1e-12) continue;
      if (best !== pivot) [a[pivot], a[best]] = [a[best], a[pivot]];

      const divisor = a[pivot][pivot];
      for (let col = pivot; col <= n; col++) a[pivot][col] /= divisor;

      for (let row = 0; row < n; row++) {
        if (row === pivot) continue;
        const factor = a[row][pivot];
        for (let col = pivot; col <= n; col++) a[row][col] -= factor * a[pivot][col];
      }
    }

    return a.map(row => Number.isFinite(row[n]) ? row[n] : 0);
  }

  private injectRegressionBasisSeed(): void {
    if (!this.currentBuffer) return;
    const seed = this.buildRegressionBasisSeed();
    if (!seed) return;
    const targetIdx = Math.max(0, this.config.population.size - 1);
    this.currentBuffer.writeGenome(targetIdx, seed.opcodes, [], seed.constants);
  }

  private injectAnalyticScaffoldSeeds(): void {
    if (
      process.env.GCL_ANALYTIC_SCAFFOLD_SEED === '0'
      || !this.currentBuffer
      || this.fitnessMode !== 'regression'
      || this.inputSize !== 1
      || this.domainProfile?.domainClass === 'discrete'
    ) {
      return;
    }

    const allowed = new Set(this.getEffectiveOperatorSet());
    const hasOps = (...ops: OpCode[]) => ops.every(op => allowed.has(op));
    const x = () => PopulationBuffer.pack(OpCode.XN, 0);
    const one = () => PopulationBuffer.pack(OpCode.CONST, 0);
    const add = () => PopulationBuffer.pack(OpCode.ADD, 0);
    const sub = () => PopulationBuffer.pack(OpCode.SUB, 0);
    const mul = () => PopulationBuffer.pack(OpCode.MUL, 0);
    const div = () => PopulationBuffer.pack(OpCode.DIV, 0);
    const sin = () => PopulationBuffer.pack(OpCode.SIN, 0);
    const cos = () => PopulationBuffer.pack(OpCode.COS, 0);

    const scaffolds: number[][] = [];
    const addScaffold = (required: OpCode[], opcodes: number[]) => {
      if (!hasOps(...required)) return;
      if (opcodes.length > (this.config.population.maxGenomeSize || 128)) return;
      scaffolds.push(opcodes);
    };

    if (hasOps(OpCode.ADD, OpCode.MUL)) {
      addScaffold([OpCode.ADD, OpCode.MUL], [x(), x(), mul()]);
      addScaffold([OpCode.ADD, OpCode.MUL], [one(), x(), x(), mul(), add()]);
    }

    if (hasOps(OpCode.ADD, OpCode.MUL, OpCode.DIV)) {
      const denom = [one(), x(), x(), mul(), add()];
      addScaffold([OpCode.ADD, OpCode.MUL, OpCode.DIV], [one(), ...denom, div()]);
      addScaffold([OpCode.ADD, OpCode.MUL, OpCode.DIV], [x(), ...denom, div()]);
      if (hasOps(OpCode.SUB)) {
        addScaffold([OpCode.ADD, OpCode.SUB, OpCode.MUL, OpCode.DIV], [one(), ...denom, div(), one(), sub()]);
      }
    }

    if (hasOps(OpCode.ADD, OpCode.MUL, OpCode.DIV, OpCode.SIN)) {
      const rational = [x(), one(), x(), x(), mul(), add(), div()];
      addScaffold([OpCode.ADD, OpCode.MUL, OpCode.DIV, OpCode.SIN], [x(), sin(), ...rational, add()]);
      addScaffold([OpCode.ADD, OpCode.MUL, OpCode.DIV, OpCode.SIN], [x(), sin(), one(), one(), x(), x(), mul(), add(), div(), add()]);
    }

    if (hasOps(OpCode.ADD, OpCode.SIN, OpCode.COS)) {
      addScaffold([OpCode.ADD, OpCode.SIN, OpCode.COS], [x(), sin(), x(), cos(), add()]);
    }

    const size = this.config.population.size;
    let injected = 0;
    for (const opcodes of scaffolds) {
      const targetIdx = size - 2 - injected;
      if (targetIdx < 0) break;
      this.currentBuffer.writeGenome(targetIdx, opcodes, [], new Float64Array([1]));
      injected++;
    }
  }

  public async step(gen: number, islandId: number, hub?: MigrationHub): Promise<boolean> {
    this.currentGeneration = gen;
    const size = this.config.population.size;

    // Gen 0 Invariant: Analysis -> Randomize -> Evaluate -> Agent Poll
    if (gen === 0 && !this.currentBuffer) {
      this.ensureTestDataBuffers();
      if (this.config.quotientMode) {
          BlockEvaluator.divisionMode = 'strict-quotient';
      }
      this.performDomainAnalysis();

      this.currentBuffer = new PopulationBuffer(size, this.config.population.maxGenomeSize || 128, 10, this.testCases.length);
      this.nextBuffer = new PopulationBuffer(size, this.config.population.maxGenomeSize || 128, 10, this.testCases.length);

      // Zero-leakage randomization
      const effectiveOps = this.getEffectiveOperatorSet();
      this.currentBuffer.randomize(
        this.inputSize,
        effectiveOps,
        3,
        this.config.population.maxGenomeSize || 128,
        this.config.population.maxGenomeDepth || 8,
        this.config.quotientMode
      );
      this.injectAnalyticScaffoldSeeds();
      this.injectRegressionBasisSeed();

      this.applyRuntimeCaseWeights();

      await this.evaluatePopulationBlock(this.currentBuffer, 0, size);
    }

    if (gen > 0 && this.currentBuffer) {
      this.applyRuntimeCaseWeights();
      if (this.isMonotoneThresholdRun() && this.inputSize >= 9 && this.bestFitnessSoFar >= 0.94) {
        await this.evaluatePopulationBlock(this.currentBuffer, 0, size);
      }
    }

    const effectiveOps = this.getEffectiveOperatorSet();
    for (let i = 0; i < size; i++) this.selectionIndices[i] = i;
    if (this.shouldUseSpea2(gen)) evaluateSPEA2Buffer(this.currentBuffer!, size);
    this.sortPopulationSelection(this.currentBuffer!, this.shouldUseSpea2(gen));

    // Migration logic
    if (hub && gen > 0 && gen % 20 === 0) {
      const champIdx = this.selectionIndices[0];
      hub.dropMigrant(islandId, this.currentBuffer!.getOpcodes(champIdx), this.currentBuffer!.getConstants(champIdx), this.currentBuffer!.fitness[champIdx], this.currentGeneration);
      const migrant = hub.pullMigrant(this.preferredSource);
      if (migrant) {
        const targetIdx = this.selectionIndices[size - 1];
        if (this.isMigrationCompatible(migrant.opcodes, effectiveOps)) {
          this.currentBuffer!.writeGenome(targetIdx, migrant.opcodes, new Uint32Array(migrant.opcodes.length), migrant.constants);
          this.currentBuffer!.fitness[targetIdx] = migrant.fitness;
          this.currentBuffer!.ancestorMap[targetIdx] = 1;
          this.currentBuffer!.generationMap[targetIdx] = migrant.sourceGeneration;
        }
      }
    }

    const useExpandedDiscreteRefiner = this.isDiscreteRun() && this.inputSize >= 7;
    const monotoneThresholdRun = this.isMonotoneThresholdRun();
    
    // Beyond SOTA Upgrade: High-Dimensional Continuous Refinement Scaling
    // For complex tabular datasets (inputSize >= 6), we dynamically scale the constant tuning cadence to 2
    // and elite count to 16, ensuring that feature-couplings are optimized to high precision early.
    const refinementCadence = this.inputSize >= 6 ? 2 : (monotoneThresholdRun && this.inputSize >= 9 ? 4 : (useExpandedDiscreteRefiner ? 5 : 8));
    const refinementEliteCount = this.inputSize >= 6 
      ? Math.min(16, size) 
      : (monotoneThresholdRun 
        ? Math.min(this.inputSize >= 9 ? 12 : 10, size)
        : (useExpandedDiscreteRefiner ? Math.min(8, size) : Math.min(5, size)));

    // Optimization & Simplification (Restricted to non-discrete)
    // Phase 71: Hyper-Orchestrated AutoML Control
    if (gen > 0 && gen % refinementCadence === 0) {
      const elites: number[] = [];
      for (let i = 0; i < refinementEliteCount; i++) elites.push(this.selectionIndices[i]);
      const variableMap = this.domainProfile?.domainClass === 'discrete' 
        ? Array.from({ length: 32 }, (_, i) => `b${i}`) 
        : Array.from({ length: this.inputSize }, (_, i) => i === 0 ? 'x' : `x${i}`);

      if (this.domainProfile?.domainClass !== 'discrete') {
        for (const idx of elites) SymbolicSimplifier.simplify(this.currentBuffer!, idx, variableMap);
        
        // AutoML Directive: Check if optimizer should run based on Discovery Agent's guidance
        const shouldOptimize = this.currentRegime !== 'STRUCTURE_FLUSH';
        if (shouldOptimize) {
            NumericalOptimizer.refineConstants(this.currentBuffer!, elites, this.testXBuffer!, this.testYBuffer!, this.inputSize, 15, this.fitnessMode, this.classificationWeight);
            if (this.fitnessMode === 'regression') {
              const targetSlots: number[] = [];
              for (let i = size - 1; i >= 0 && targetSlots.length < elites.length; i--) {
                const idx = this.selectionIndices[i];
                if (!elites.includes(idx)) targetSlots.push(idx);
              }
              this.semanticArchive.recordElites(
                this.currentBuffer!,
                elites,
                this.testXBuffer!,
                this.testYBuffer!,
                this.inputSize
              );
              if (process.env.GCL_SEMANTIC_ARCHIVE !== '0') this.semanticArchiveRecordCycles++;
              const archiveTargetSlot = targetSlots.pop();
              const injected = SemanticRefiner.refineRegressionElites(
                this.currentBuffer!,
                elites,
                targetSlots,
                this.testXBuffer!,
                this.testYBuffer!,
                this.inputSize,
                this.config.population.maxGenomeSize || 128
              );
              if (process.env.GCL_SEMANTIC_ARCHIVE !== '0' && archiveTargetSlot !== undefined && this.selectionIndices[0] !== archiveTargetSlot) {
                this.semanticArchiveInjectionAttempts++;
                const archiveInjected = this.semanticArchive.injectCandidate(
                  this.currentBuffer!,
                  this.selectionIndices[0],
                  archiveTargetSlot,
                  this.testXBuffer!,
                  this.testYBuffer!,
                  this.inputSize,
                  this.config.population.maxGenomeSize || 128
                );
                if (archiveInjected) this.semanticArchiveInjectionSuccesses++;
              }
            }
        }
      } else {
        // Boolean Domain Check
        // Try simple boolean variations to climb discrete structures
        BooleanRefiner.refineDiscrete(
          this.currentBuffer!,
          elites,
          this.testXBuffer!,
          this.testYBuffer!,
          this.inputSize,
          this.fitnessMode,
          this.classificationWeight,
          monotoneThresholdRun,
          monotoneThresholdRun ? this.domainProfile?.signals.logicThreshold : undefined
        );
      }
      await this.evaluatePopulationBlock(this.currentBuffer!, 0, size);
      if (this.shouldUseSpea2(gen)) evaluateSPEA2Buffer(this.currentBuffer!, size);
      this.sortPopulationSelection(this.currentBuffer!, this.shouldUseSpea2(gen));
    }

    // Autonomous Discovery Agent Loop
    if (this.discoveryAgent || this.agentBridge) {
      if (gen % 5 === 0) {
        const problemStats = {
          gen,
          bestFitness: this.bestFitnessSoFar,
          stagnationCount: this.stagnationCount,
          populationEntropy: this.currentBuffer?.calculateEntropy() || 0,
          bestSize: this.currentBuffer?.lengths[this.selectionIndices[0]] || 0
        };
        if (this.agentBridge) {
          this.agentBridge.pushGlobalAnalytics([problemStats], 'CURRENT_PROBLEM');
          const bridgeCommand = this.agentBridge.pollCommand();
          if (bridgeCommand) this.applyCommand(bridgeCommand);
        }
        else if (this.discoveryAgent) this.discoveryAgent.decideGlobalAction([problemStats], 'CURRENT_PROBLEM');
      }

      const elitesSnap: Genome[] = [];
      const elitesCount = Math.min(3, size);
      for (let i = 0; i < elitesCount; i++) {
        const idx = this.selectionIndices[i];
        elitesSnap.push({
          id: `gen${gen}-elite${i}`,
          genes: [],
          opcodes: this.currentBuffer!.getOpcodes(idx),
          constants: this.currentBuffer!.getConstants(idx),
          fitness: this.currentBuffer!.fitness[idx],
          generation: gen,
          age: 0
        } as any);
      }

      if (this.agentBridge) {
        this.agentBridge.pushSnapshot(elitesSnap, gen);
        const completions = this.agentBridge.poll(gen);
        this.applyAgentCompletions(completions, gen);
      } else if (this.discoveryAgent) {
        this.discoveryAgent.pushPopulationSnapshot(elitesSnap, gen);
        const completions = this.discoveryAgent.pullAgentCandidates(gen);
        this.applyAgentCompletions(completions, gen);
      }
    }

    this.refreshDiscreteSelectionScores(this.currentBuffer!);

    const bestRawFitnessIdx = this.getChampionIndex(this.currentBuffer!);
    const bestChampionScore = this.getChampionScore(this.currentBuffer!, bestRawFitnessIdx);
    const hasImproved = bestChampionScore > this.bestFitnessSoFar + 1e-6;
    if (hasImproved) {
      this.bestFitnessSoFar = bestChampionScore;
      this.stagnationCount = 0;
    } else {
      this.stagnationCount++;
    }

    const paretoChampIdx = this.selectionIndices[0];
    const causalUtility = this.calculateCausalUtility();
    const shouldSendHeavy = gen % 10 === 0 && (hasImproved || gen === 0);
    const bestGenome = shouldSendHeavy ? this.getBestGenome() : undefined;

    this.onStats?.({
      type: 'ISLAND_STATS', gen,
      bestFitness: bestChampionScore,
      bestSize: this.currentBuffer!.lengths[paretoChampIdx],
      entropy: this.currentBuffer!.calculateEntropy(),
      causalUtility,
      stagnationCount: this.stagnationCount,
      bestGenome,
      islandId
    });

    if (bestChampionScore >= this.config.targetFitness) return true;
    if (monotoneThresholdRun) {
      const finalizedBest = this.getBestBinary();
      if (finalizedBest && finalizedBest.fitness >= this.config.targetFitness) return true;
    }

    // Phase 71: Neural-Assisted Structural Selection
    // If we have a Discovery Agent and we are in a stagnant state, 
    // we use the NN to validate the 'Honesty' of the best individuals.
    if (this.discoveryAgent && gen > 0 && gen % 15 === 0 && this.testXBuffer) {
        const topIdx = this.selectionIndices[0];
        const predictions = BlockEvaluator.getPredictions(this.currentBuffer!, topIdx, this.testXBuffer, this.inputSize);
        
        let divCount = 0;
        const len = this.currentBuffer!.lengths[topIdx];
        const offset = this.currentBuffer!.offsets[topIdx];
        for (let i = 0; i < len; i++) {
            if ((this.currentBuffer!.geneBuffer[offset + i] >>> 24) === OpCode.DIV) divCount++;
        }

        // Let the agent audit the predictions vs targets
        // If it detects mimicry, it returns a 'STRUCTURAL_PIVOT' or 'QUOTIENT_MODE' signal
        const audit = (this.discoveryAgent as any).decideGlobalAction([{
            gen,
            bestFitness: this.currentBuffer!.fitness[topIdx],
            bestSize: this.currentBuffer!.lengths[topIdx],
            stagnationCount: this.stagnationCount,
            divCount
        }], 'CURRENT_PROBLEM');

        if (audit && audit.regime === 'STRUCTURAL_PIVOT') {
            this.currentRegime = 'STRUCTURAL_PIVOT';
            this.currentMutationIntensity = audit.intensity;
            console.error(`[EvolutionEngine] AutoML Pivot: Intensity=${audit.intensity}`);
        }
    }

    // NEXT GENERATION: Hyper-Orchestration Search
    const variableMap = this.domainProfile?.domainClass === 'discrete' 
      ? Array.from({ length: 32 }, (_, i) => `b${i}`) 
      : Array.from({ length: this.inputSize }, (_, i) => i === 0 ? 'x' : `x${i}`);

    const rawUpperCompose = (process.env.GCL_THRESHOLD_UPPER_COMPOSE || '').trim().toLowerCase();
    const explicitThresholdUpperCompose =
      rawUpperCompose === 'cross' ? 'cross'
      : rawUpperCompose === 'mirrored-cross' || rawUpperCompose === 'mirrored_cross' || rawUpperCompose === 'mirroredcross'
        ? 'mirrored-cross'
      : rawUpperCompose === 'triad-witness' || rawUpperCompose === 'triad_witness' || rawUpperCompose === 'triadwitness'
        ? 'triad-witness'
      : rawUpperCompose === 'coalition' ? 'coalition'
      : rawUpperCompose === 'paired-witness' || rawUpperCompose === 'paired_witness' || rawUpperCompose === 'pairedwitness'
        ? 'paired-witness'
      : rawUpperCompose === 'witnessed-or' || rawUpperCompose === 'witnessed_or' || rawUpperCompose === 'witnessedor'
        ? 'witnessed-or'
        : null;

    const eliteCount = this.config.population.eliteCount;
    for (let i = 0; i < eliteCount; i++) {
        this.nextBuffer!.copyGenome(this.selectionIndices[i], i, this.currentBuffer!);
        // Force simplification on all elites every generation to ensure structural honesty
        SymbolicSimplifier.simplify(this.nextBuffer!, i, variableMap);
    }
    
    for (let i = eliteCount; i < size; i++) {
      let p1 = this.selectParentIndex(this.currentBuffer!, gen);
      
      // Structural Pressure: If we are close to target, favor smaller genomes
      const currentBestFit = this.currentBuffer!.fitness[this.selectionIndices[0]];
      const parsimonyActive = currentBestFit > 0.95;

      if (random() < this.config.population.crossoverRate) {
        const booleanComposeRate = monotoneThresholdRun && this.inputSize >= 9 ? 0.65 : 0.45;
        const rawThresholdComposeExperiment = (process.env.GCL_THRESHOLD_COMPOSE_EXPERIMENT || '').trim().toLowerCase();
        const thresholdComposeExperiment = !(
          rawThresholdComposeExperiment === '0'
          || rawThresholdComposeExperiment === 'false'
          || rawThresholdComposeExperiment === 'no'
          || rawThresholdComposeExperiment === 'off'
        );
        const rawAdaptiveDuelExperiment = (process.env.GCL_THRESHOLD_ADAPTIVE_DUEL_EXPERIMENT || '').trim().toLowerCase();
        const thresholdAdaptiveDuelEnabled = !(
          rawAdaptiveDuelExperiment === '0'
          || rawAdaptiveDuelExperiment === 'false'
          || rawAdaptiveDuelExperiment === 'no'
          || rawAdaptiveDuelExperiment === 'off'
        );
        const thresholdComposeEligible =
          thresholdComposeExperiment
          && 
          monotoneThresholdRun
          && this.inputSize >= 9
          && currentBestFit >= 0.82;
        const useBooleanCompose =
          this.isDiscreteRun()
          && this.inputSize >= 7
          && (
            this.stagnationCount > 5
            || this.currentMutationIntensity > 1.5
            || thresholdComposeEligible
          )
          && random() < booleanComposeRate;
        const useThresholdCompose = useBooleanCompose && monotoneThresholdRun;
        const thresholdRoleArraysReady =
          useThresholdCompose
          && this.discreteLowerBoundaryScores.length === size
          && this.discreteUpperBoundaryScores.length === size;
        let p2 = this.selectParentIndex(this.currentBuffer!, gen);
        let thresholdComposeFocus: 'lower' | 'upper' | null = null;
        let thresholdComposeStrategy: 'coalition' | 'cross' | 'mirrored-cross' | 'triad-witness' | 'witnessed-or' | 'paired-witness' | null = null;
        let adaptiveUpperComposeCommitted = false;

        if (thresholdRoleArraysReady) {
          const champIdx = this.selectionIndices[0];
          const lowerScore = this.discreteLowerBoundaryScores[champIdx];
          const upperScore = this.discreteUpperBoundaryScores[champIdx];
          const lowerExact = this.discreteLowerExactAccuracyScores[champIdx];
          const upperExact = this.discreteUpperExactAccuracyScores[champIdx];
          const lowerDominant = lowerScore > upperScore + 1e-9;
          const upperDominant = upperScore > lowerScore + 1e-9;
          const sideDelta = Math.abs(lowerScore - upperScore);
          const exactGap = Math.abs(lowerExact - upperExact);
          const weakSideForCompose: 'lower' | 'upper' | null =
            lowerDominant ? 'lower' : upperDominant ? 'upper' : null;
          if (thresholdComposeExperiment && weakSideForCompose && currentBestFit >= 0.84 && sideDelta >= 0.01) {
            thresholdComposeFocus = weakSideForCompose;
            this.thresholdComposeFocusCount++;
            if (explicitThresholdUpperCompose && weakSideForCompose === 'upper') {
              thresholdComposeStrategy = explicitThresholdUpperCompose;
            }
          }
          const thresholdRoleCompose = thresholdComposeExperiment
            ? currentBestFit >= 0.86
              && (currentBestFit >= 0.92 || sideDelta >= 0.01)
              && (lowerDominant || upperDominant || currentBestFit >= 0.94)
            : currentBestFit >= 0.9
              && (currentBestFit >= 0.94 || sideDelta >= 0.015)
              && (lowerDominant || upperDominant || currentBestFit >= 0.94);
          if (!thresholdRoleCompose) {
            // Below the late plateau, only force role-specialized compose when the
            // champion already shows a meaningful one-sided boundary weakness.
          } else {
          const primarySide: 'lower' | 'upper' =
            lowerDominant ? 'lower' : upperDominant ? 'upper' : (random() < 0.5 ? 'lower' : 'upper');
          const secondarySide: 'lower' | 'upper' = primarySide === 'lower' ? 'upper' : 'lower';
          thresholdComposeFocus = primarySide;
          this.thresholdRoleComposeCount++;
          const primaryParent = this.selectBoundaryRoleParent(this.currentBuffer!, primarySide, undefined, champIdx);
          const secondaryParent = this.selectBoundaryRoleParent(
            this.currentBuffer!,
            secondarySide,
            primaryParent ?? undefined,
            primaryParent ?? champIdx
          );
          if (primaryParent !== null) p1 = primaryParent;
          if (secondaryParent !== null) p2 = secondaryParent;
          }
        }

        if (
          thresholdComposeExperiment
          && thresholdComposeFocus === 'upper'
        ) {
          this.thresholdUpperComposeOpportunityCount++;
          if (explicitThresholdUpperCompose) {
            thresholdComposeStrategy = explicitThresholdUpperCompose;
            if (explicitThresholdUpperCompose === 'paired-witness') {
              this.thresholdStaticPairedWitnessComposeCount++;
            }
          } else if (thresholdAdaptiveDuelEnabled) {
            const adaptiveWinner = this.writeAdaptiveUpperComposeWinner(p1, p2, i);
            adaptiveUpperComposeCommitted = true;
            this.thresholdAdaptiveDuelCount++;
            if (adaptiveWinner === 'cross') this.thresholdAdaptiveCrossCount++;
            else if (adaptiveWinner === 'mirrored-cross') this.thresholdAdaptiveMirroredCrossCount++;
            else if (adaptiveWinner === 'triad-witness') this.thresholdAdaptiveTriadWitnessCount++;
            else if (adaptiveWinner === 'guarded-upper-repair') this.thresholdAdaptiveGuardedUpperRepairCount++;
            else if (adaptiveWinner === 'paired-witness') {
              this.thresholdAdaptivePairedWitnessCount++;
              if (!this.lastAdaptiveUpperComposeShadowWinner) {
                this.thresholdAdaptivePairedWitnessIncumbentCount++;
              }
            }
            else this.thresholdAdaptiveWitnessedOrCount++;
            this.thresholdAdaptivePairedWitnessVetoCount += this.lastAdaptiveUpperComposeVetoCount;
          } else {
            thresholdComposeStrategy = 'paired-witness';
            this.thresholdStaticPairedWitnessComposeCount++;
          }
        }

        if (useBooleanCompose) {
          if (useThresholdCompose) {
            if (!adaptiveUpperComposeCommitted) {
              LinearOperators.crossoverBooleanThresholdCompose(
                this.currentBuffer!,
                p1,
                p2,
                this.nextBuffer!,
                i,
                this.config.population.maxGenomeSize || 128,
                this.inputSize,
                thresholdComposeFocus,
                thresholdComposeStrategy
              );
            }
          } else {
            LinearOperators.crossoverBooleanCompose(
              this.currentBuffer!,
              p1,
              p2,
              this.nextBuffer!,
              i,
              this.config.population.maxGenomeSize || 128,
              this.inputSize
            );
          }
        } else {
          LinearOperators.crossover(this.currentBuffer!, p1, p2, this.nextBuffer!, i, this.config.population.maxGenomeSize || 128, this.config.quotientMode);
        }
      } else {
        this.nextBuffer!.copyGenome(p1, i, this.currentBuffer!);
      }

      if (random() < this.config.population.mutationRate) {
        const mutationIntensity = parsimonyActive
          ? Math.max(1, this.currentMutationIntensity * 0.75)
          : Math.max(1, this.currentMutationIntensity);
        const mutationDepth = this.config.population.maxGenomeDepth || 8;

        if (this.domainProfile?.domainClass === 'discrete' || this.fitnessMode === 'classification' || random() < 0.7) {
          LinearOperators.mutateSubtree(
            this.nextBuffer!,
            i,
            effectiveOps,
            this.config.population.maxGenomeSize || 128,
            this.inputSize,
            mutationDepth,
            mutationIntensity,
            this.config.quotientMode
          );
        } else {
          LinearOperators.mutateAdaptive(this.nextBuffer!, i, effectiveOps, mutationIntensity, this.config.quotientMode);
        }
      }
    }

    await this.evaluatePopulationBlock(this.nextBuffer!, 0, size);
    if (this.shouldUseSpea2(gen)) evaluateSPEA2Buffer(this.nextBuffer!, size);
    [this.currentBuffer, this.nextBuffer] = [this.nextBuffer, this.currentBuffer];

    return false;
  }

  private isMigrationCompatible(opcodes: ArrayLike<number>, effectiveOps: OpCode[]): boolean {
    const allowedOps = new Set<OpCode>([
      OpCode.NONE,
      OpCode.XN,
      OpCode.CONST,
      OpCode.TERMINATOR,
      ...effectiveOps
    ]);
    for (let i = 0; i < opcodes.length; i++) {
      const op = (opcodes[i] >>> 24) as OpCode;
      if (!allowedOps.has(op)) return false;
    }
    return true;
  }

  public async evolve(): Promise<EvolutionResult> {
    this.startTime = Date.now();
    let gen = 0;
    try {
      while (gen < this.config.maxGenerations) {
        if (await this.step(gen, 0)) break;
        gen++;
      }
      const best = this.getBestGenome();
      return {
        bestGenome: (best as any) || { id: 'empty', genes: [], fitness: 0, generation: gen, age: 0 },
        generationsRun: gen,
        totalEvaluations: gen * this.config.population.size,
        totalTimeMs: Date.now() - this.startTime,
        success: (best?.fitness || 0) >= this.config.targetFitness,
        stats: this.stats,
        archive: []
      };
    } finally {
      await this.dispose();
    }
  }

  public async dispose(): Promise<void> {
    if (!this.parallelBlockEvaluator) return;
    const evaluator = this.parallelBlockEvaluator;
    this.parallelBlockEvaluator = undefined;
    await evaluator.shutdown();
  }

  public getEffectiveOperatorSet(): OpCode[] {
    // Phase 66: Strict Operator Enforcement
    const opMap: Record<string, OpCode> = {
        'add': OpCode.ADD, 'sub': OpCode.SUB, 'mul': OpCode.MUL, 'div': OpCode.DIV,
        'sin': OpCode.SIN, 'cos': OpCode.COS, 'exp': OpCode.EXP, 'log': OpCode.LOG,
        'sqrt': OpCode.SQRT, 'and': OpCode.AND, 'or': OpCode.OR, 'nand': OpCode.NAND, 'nor': OpCode.NOR,
        'xor': OpCode.XOR, 'if_then_else': OpCode.ITE
    };

    const problemOps = this.functionSet.map(f => f.name.toLowerCase());
    const mapped = problemOps.map(name => opMap[name]).filter(op => op !== undefined);
    if (this.isMonotoneThresholdRun()) return [OpCode.AND, OpCode.OR];
    
    if (mapped.length > 0) return mapped;

    if (this.domainProfile?.domainClass === 'discrete' || this.fitnessMode === 'classification') return [OpCode.AND, OpCode.OR, OpCode.NAND, OpCode.NOR, OpCode.XOR];
    if (this.currentRegime === 'STRUCTURAL_PIVOT') return [OpCode.ADD, OpCode.SUB, OpCode.MUL, OpCode.DIV];
    return [OpCode.ADD, OpCode.SUB, OpCode.MUL, OpCode.DIV, OpCode.SIN, OpCode.COS];
  }

  public setDiscoveryAgent(agent: DiscoveryAgent): void {
    this.discoveryAgent = agent;
    if (this.testXBuffer && this.testYBuffer) this.syncAgentEvaluationContext();
  }

  public setMutationIntensity(intensity: number): void {
    this.currentMutationIntensity = intensity;
  }

  public async evolveIsland(hub: MigrationHub, islandId: number): Promise<void> {
    this.startTime = Date.now();
    try {
      for (let gen = 0; gen < this.config.maxGenerations; gen++) {
        if (await this.step(gen, islandId, hub)) break;
        
        // Heartbeat stats
        if (gen % 5 === 0 && this.onStats) {
          this.onStats({
            type: 'ISLAND_STATS',
            islandId,
            gen,
            bestFitness: this.bestFitnessSoFar,
            entropy: this.currentBuffer?.calculateEntropy() || 0,
            bestGenome: this.getBestGenome(),
            bestPredictions: this.getBestPredictions()
          });
        }
      }
    } finally {
      await this.dispose();
    }
  }
  private ensureTestDataBuffers() {
    if (!this.testXBuffer && this.testCases.length > 0) {
      this.inputSize = (this.testCases[0].input as number[]).length;
      this.testXBuffer = new Float64Array(this.testCases.length * this.inputSize);
      this.testYBuffer = new Float64Array(this.testCases.length);
      for (let i = 0; i < this.testCases.length; i++) {
        for (let j = 0; j < this.inputSize; j++) this.testXBuffer[i * this.inputSize + j] = (this.testCases[i].input as number[])[j];
        this.testYBuffer[i] = (this.testCases[i].expectedOutput as number);
      }
    }
  }

  public applyRegime(regime: string): void {
    this.currentRegime = regime;
    this.opCodes = this.getEffectiveOperatorSet();
    console.error(`[EvolutionEngine] REGIME PIVOT: ${regime} (Operator Set Locked to Domain)`);
  }

  public applyCommand(cmd: any): void {
    const { action, intensity, regime } = cmd;
    if (regime) this.applyRegime(regime);
    const size = this.config.population.size;
    const effectiveOps = this.getEffectiveOperatorSet();
    switch (action) {
      case 'MUTATION_KICK':
        this.currentMutationIntensity = Math.max(this.currentMutationIntensity, intensity || 5.0);
        break;
      case 'MAJORITY_BOOTSTRAP':
      case 'PARITY_BOOTSTRAP':
      case 'MUX_BOOTSTRAP':
        if (typeof intensity === 'number' && intensity > 0) {
          this.currentMutationIntensity = Math.max(this.currentMutationIntensity, intensity);
        }
        break;
      case 'CATACLYSM':
        if (this.currentBuffer) {
          const keepCount = Math.max(2, Math.floor(size * 0.1));
          for (let i = keepCount; i < size; i++) {
            const idx = this.selectionIndices[i];
            this.currentBuffer.randomize(this.inputSize, effectiveOps, 3, 16, Math.min(4, this.config.population.maxGenomeDepth || 8));
          }
          console.error(`[EvolutionEngine] CATACLYSM AUDIT: Wiped ${size - keepCount} individuals while maintaining Domain Lock.`);
        }
        break;
    }
  }

  public getBestGenome(): Genome | null {
    const binary = this.getBestBinary();
    if (!binary) return null;

    const varMap = this.domainProfile?.domainClass === 'discrete' 
      ? Array.from({ length: 32 }, (_, i) => `b${i}`) 
      : Array.from({ length: this.inputSize }, (_, i) => i === 0 ? 'x' : `x${i}`);

    const opcodes: OpCode[] = [];
    const data: number[] = [];
    for (let i = 0; i < binary.opcodes.length; i++) {
        const unpacked = PopulationBuffer.unpack(binary.opcodes[i]);
        opcodes.push(unpacked.op);
        data.push(unpacked.data);
    }

    const genome = PostfixBridge.unbake(opcodes, data, Array.from(binary.constants), varMap);
    genome.fitness = binary.fitness;
    genome.generation = (binary as any).generation ?? this.currentGeneration;
    
    return genome;
  }

  public getBestBinary(): { opcodes: Uint32Array, constants: Float64Array, fitness: number, generation?: number } | null {
    const raw = this.getBestBinaryRaw();
    return raw ? this.finalizeBinaryOutput(raw) : null;
  }

  public getBestBinaryRaw(): { opcodes: Uint32Array, constants: Float64Array, fitness: number, generation?: number } | null {
    if (!this.currentBuffer) return null;
    const bestIdx = this.getBestIndex();
    if (bestIdx === null) return null;
    return {
        opcodes: new Uint32Array(this.currentBuffer.getOpcodes(bestIdx)),
        constants: new Float64Array(this.currentBuffer.getConstants(bestIdx)),
        fitness: this.currentBuffer.fitness[bestIdx],
        generation: this.currentBuffer.generationMap[bestIdx]
    };
  }

  public getComposeDebugStats(): {
    thresholdComposeFocusCount: number;
    thresholdRoleComposeCount: number;
    thresholdUpperComposeOpportunityCount: number;
    thresholdAdaptiveDuelCount: number;
    thresholdAdaptiveCrossCount: number;
    thresholdAdaptiveMirroredCrossCount: number;
    thresholdAdaptiveTriadWitnessCount: number;
    thresholdAdaptivePairedWitnessCount: number;
    thresholdAdaptiveGuardedCrossCount: number;
    thresholdAdaptiveGuardedUpperRepairCount: number;
    thresholdAdaptiveWitnessedOrCount: number;
    thresholdAdaptivePairedWitnessIncumbentCount: number;
    thresholdAdaptivePairedWitnessVetoCount: number;
    thresholdAdaptiveCrossVetoCount: number;
    thresholdAdaptiveMirroredCrossVetoCount: number;
    thresholdAdaptiveTriadWitnessVetoCount: number;
    thresholdAdaptiveGuardedCrossVetoCount: number;
    thresholdAdaptiveGuardedUpperRepairVetoCount: number;
    thresholdAdaptiveWitnessedOrVetoCount: number;
    thresholdAdaptiveVetoExactRegressionCount: number;
    thresholdAdaptiveVetoUpperExactRegressionCount: number;
    thresholdAdaptiveVetoLowerExactRegressionCount: number;
    thresholdAdaptiveVetoUpperBoundaryRegressionCount: number;
    thresholdAdaptiveVetoLowerBoundaryRegressionCount: number;
    thresholdAdaptiveVetoBoundaryRegressionCount: number;
    thresholdAdaptiveVetoPressureRegressionCount: number;
    thresholdAdaptiveVetoWeightedRegressionCount: number;
    thresholdAdaptiveVetoNoMaterialGainCount: number;
    thresholdAdaptiveShadowWinnerCount: number;
    thresholdAdaptiveShadowCrossCount: number;
    thresholdAdaptiveShadowMirroredCrossCount: number;
    thresholdAdaptiveShadowTriadWitnessCount: number;
    thresholdAdaptiveShadowGuardedUpperRepairCount: number;
    thresholdAdaptiveShadowWitnessedOrCount: number;
    thresholdAdaptivePromotedWinnerCount: number;
    thresholdAdaptivePromotedCrossCount: number;
    thresholdAdaptivePromotedMirroredCrossCount: number;
    thresholdAdaptivePromotedTriadWitnessCount: number;
    thresholdAdaptivePromotedGuardedUpperRepairCount: number;
    thresholdAdaptivePromotedWitnessedOrCount: number;
    thresholdAdaptivePromotionChampionBlockCount: number;
    thresholdStaticPairedWitnessComposeCount: number;
  } {
    return {
      thresholdComposeFocusCount: this.thresholdComposeFocusCount,
      thresholdRoleComposeCount: this.thresholdRoleComposeCount,
      thresholdUpperComposeOpportunityCount: this.thresholdUpperComposeOpportunityCount,
      thresholdAdaptiveDuelCount: this.thresholdAdaptiveDuelCount,
      thresholdAdaptiveCrossCount: this.thresholdAdaptiveCrossCount,
      thresholdAdaptiveMirroredCrossCount: this.thresholdAdaptiveMirroredCrossCount,
      thresholdAdaptiveTriadWitnessCount: this.thresholdAdaptiveTriadWitnessCount,
      thresholdAdaptivePairedWitnessCount: this.thresholdAdaptivePairedWitnessCount,
      thresholdAdaptiveGuardedCrossCount: this.thresholdAdaptiveGuardedCrossCount,
      thresholdAdaptiveGuardedUpperRepairCount: this.thresholdAdaptiveGuardedUpperRepairCount,
      thresholdAdaptiveWitnessedOrCount: this.thresholdAdaptiveWitnessedOrCount,
      thresholdAdaptivePairedWitnessIncumbentCount: this.thresholdAdaptivePairedWitnessIncumbentCount,
      thresholdAdaptivePairedWitnessVetoCount: this.thresholdAdaptivePairedWitnessVetoCount,
      thresholdAdaptiveCrossVetoCount: this.thresholdAdaptiveCrossVetoCount,
      thresholdAdaptiveMirroredCrossVetoCount: this.thresholdAdaptiveMirroredCrossVetoCount,
      thresholdAdaptiveTriadWitnessVetoCount: this.thresholdAdaptiveTriadWitnessVetoCount,
      thresholdAdaptiveGuardedCrossVetoCount: this.thresholdAdaptiveGuardedCrossVetoCount,
      thresholdAdaptiveGuardedUpperRepairVetoCount: this.thresholdAdaptiveGuardedUpperRepairVetoCount,
      thresholdAdaptiveWitnessedOrVetoCount: this.thresholdAdaptiveWitnessedOrVetoCount,
      thresholdAdaptiveVetoExactRegressionCount: this.thresholdAdaptiveVetoExactRegressionCount,
      thresholdAdaptiveVetoUpperExactRegressionCount: this.thresholdAdaptiveVetoUpperExactRegressionCount,
      thresholdAdaptiveVetoLowerExactRegressionCount: this.thresholdAdaptiveVetoLowerExactRegressionCount,
      thresholdAdaptiveVetoUpperBoundaryRegressionCount: this.thresholdAdaptiveVetoUpperBoundaryRegressionCount,
      thresholdAdaptiveVetoLowerBoundaryRegressionCount: this.thresholdAdaptiveVetoLowerBoundaryRegressionCount,
      thresholdAdaptiveVetoBoundaryRegressionCount: this.thresholdAdaptiveVetoBoundaryRegressionCount,
      thresholdAdaptiveVetoPressureRegressionCount: this.thresholdAdaptiveVetoPressureRegressionCount,
      thresholdAdaptiveVetoWeightedRegressionCount: this.thresholdAdaptiveVetoWeightedRegressionCount,
      thresholdAdaptiveVetoNoMaterialGainCount: this.thresholdAdaptiveVetoNoMaterialGainCount,
      thresholdAdaptiveShadowWinnerCount: this.thresholdAdaptiveShadowWinnerCount,
      thresholdAdaptiveShadowCrossCount: this.thresholdAdaptiveShadowCrossCount,
      thresholdAdaptiveShadowMirroredCrossCount: this.thresholdAdaptiveShadowMirroredCrossCount,
      thresholdAdaptiveShadowTriadWitnessCount: this.thresholdAdaptiveShadowTriadWitnessCount,
      thresholdAdaptiveShadowGuardedUpperRepairCount: this.thresholdAdaptiveShadowGuardedUpperRepairCount,
      thresholdAdaptiveShadowWitnessedOrCount: this.thresholdAdaptiveShadowWitnessedOrCount,
      thresholdAdaptivePromotedWinnerCount: this.thresholdAdaptivePromotedWinnerCount,
      thresholdAdaptivePromotedCrossCount: this.thresholdAdaptivePromotedCrossCount,
      thresholdAdaptivePromotedMirroredCrossCount: this.thresholdAdaptivePromotedMirroredCrossCount,
      thresholdAdaptivePromotedTriadWitnessCount: this.thresholdAdaptivePromotedTriadWitnessCount,
      thresholdAdaptivePromotedGuardedUpperRepairCount: this.thresholdAdaptivePromotedGuardedUpperRepairCount,
      thresholdAdaptivePromotedWitnessedOrCount: this.thresholdAdaptivePromotedWitnessedOrCount,
      thresholdAdaptivePromotionChampionBlockCount: this.thresholdAdaptivePromotionChampionBlockCount,
      thresholdStaticPairedWitnessComposeCount: this.thresholdStaticPairedWitnessComposeCount
    };
  }

  public getSemanticArchiveDebugStats(): {
    recordCycles: number;
    injectionAttempts: number;
    injectionSuccesses: number;
    size: number;
    diagnostics: SemanticArchiveDiagnostics[];
  } {
    return {
      recordCycles: this.semanticArchiveRecordCycles,
      injectionAttempts: this.semanticArchiveInjectionAttempts,
      injectionSuccesses: this.semanticArchiveInjectionSuccesses,
      size: this.semanticArchive.size(),
      diagnostics: this.semanticArchive.diagnostics()
    };
  }

  /**
   * Extract the predictions vector from the best individual
   */
  public getBestPredictions(): Float64Array | null {
    if (!this.currentBuffer || !this.testXBuffer) return null;
    const bestIdx = this.getBestIndex();
    if (bestIdx === null) return null;

    const raw = this.getBestBinaryRaw();
    const finalized = raw ? this.finalizeBinaryOutput(raw) : null;
    if (!raw || !finalized || finalized === raw) {
      return BlockEvaluator.getPredictions(this.currentBuffer, bestIdx, this.testXBuffer, this.inputSize);
    }

    const predictions = new Float64Array(this.testCases.length);
    for (let i = 0; i < this.testCases.length; i++) {
      predictions[i] = this.evaluateBinaryProgram(finalized.opcodes, finalized.constants, this.testCases[i].input as number[]);
    }
    return predictions;
  }

  public injectGenome(opcodes: Uint32Array, constants: Float64Array, _fit: number, targetIdx?: number): boolean {
    if (!this.currentBuffer || !this.testXBuffer || !this.testYBuffer) return false;
    if (!this.hasValidInputReferences(opcodes)) {
      console.error('[EvolutionEngine] REJECTED INJECTION: invalid variable reference.');
      return false;
    }
    if (this.isDiscreteRun() && !this.isLogicSafeGenome(opcodes)) {
      console.error('[EvolutionEngine] REJECTED INJECTION: non-logic opcode in discrete run.');
      return false;
    }
    const injectIdx = targetIdx ?? this.selectionIndices[this.config.population.size - 1];
    const previousOpcodes = new Uint32Array(this.currentBuffer.getOpcodes(injectIdx));
    const previousConstants = new Float64Array(this.currentBuffer.getConstants(injectIdx));
    const previousFitness = this.currentBuffer.fitness[injectIdx];
    const previousSpea2 = this.currentBuffer.spea2Score[injectIdx];
    const previousAncestor = this.currentBuffer.ancestorMap[injectIdx];
    const previousGeneration = this.currentBuffer.generationMap[injectIdx];
    const caseOffset = injectIdx * this.currentBuffer.maxTestCases;
    const previousCaseErrors = this.currentBuffer.caseErrors.slice(caseOffset, caseOffset + this.currentBuffer.maxTestCases);

    this.currentBuffer.writeGenome(injectIdx, opcodes, [], constants);
    this.currentBuffer.generationMap[injectIdx] = this.currentGeneration;
    BlockEvaluator.evaluateBlock(this.currentBuffer, injectIdx, 1, this.testXBuffer, this.testYBuffer, this.inputSize, this.fitnessMode, this.classificationWeight);
    const injectedFitness = this.currentBuffer.fitness[injectIdx];

    if (injectedFitness + 1e-9 < previousFitness) {
      this.currentBuffer.writeGenome(injectIdx, previousOpcodes, [], previousConstants);
      this.currentBuffer.fitness[injectIdx] = previousFitness;
      this.currentBuffer.spea2Score[injectIdx] = previousSpea2;
      this.currentBuffer.ancestorMap[injectIdx] = previousAncestor;
      this.currentBuffer.generationMap[injectIdx] = previousGeneration;
      this.currentBuffer.caseErrors.set(previousCaseErrors, caseOffset);
      return false;
    }

    this.currentBuffer.ancestorMap[injectIdx] = 1;
    console.error(`[EvolutionEngine] AGENT INJECTION: Slot ${injectIdx} updated (Fitness: ${injectedFitness.toFixed(8)})`);
    return true;
  }

  private applyAgentCompletions(completions: Array<RefinedCandidate | { genome: Genome, fitness: number }>, gen: number): void {
    if (!completions || completions.length === 0 || !this.currentBuffer) return;

    const completionAction = (entry: any): string => {
      if (typeof entry?.lineage === 'string' && entry.lineage.length > 0) return entry.lineage;
      if (typeof entry?.log?.actionApplied === 'string' && entry.log.actionApplied.length > 0) return entry.log.actionApplied;
      if (typeof entry?.log?.strategy === 'string' && entry.log.strategy.length > 0) return entry.log.strategy;
      return '';
    };
    const hasMajorityUpperLift = completions.some(entry => completionAction(entry) === 'MAJORITY_UPPER_LIFT');

    let maxInsertions = Math.min(
      completions.length,
      Math.max(4, Math.min(16, Math.ceil(this.config.population.size * 0.08)))
    );
    if (this.isDiscreteRun() && this.inputSize >= 7) {
      const currentBest = this.currentBuffer.fitness[this.selectionIndices[0]];
      const monotoneThresholdRun = this.isMonotoneThresholdRun();

      if (monotoneThresholdRun && this.inputSize >= 9) {
        if (currentBest >= 0.94) return;
        if (currentBest >= 0.9) maxInsertions = Math.min(maxInsertions, 1);
        else if (currentBest >= 0.82) maxInsertions = Math.min(maxInsertions, 2);
        else if (currentBest >= 0.7) maxInsertions = Math.min(maxInsertions, 3);
        else maxInsertions = Math.min(maxInsertions, 5);

        if (hasMajorityUpperLift && currentBest >= 0.82) {
          maxInsertions = Math.max(maxInsertions, Math.min(completions.length, 4));
        }
      } else if (monotoneThresholdRun) {
        if (currentBest >= 0.9) return;
        if (currentBest >= 0.82) maxInsertions = Math.min(maxInsertions, 1);
        else if (currentBest >= 0.7) maxInsertions = Math.min(maxInsertions, 2);
        else maxInsertions = Math.min(maxInsertions, 4);
      } else {
        if (currentBest >= 0.85) return;
        if (currentBest >= 0.75) maxInsertions = Math.min(maxInsertions, 2);
        else if (currentBest >= 0.6) maxInsertions = Math.min(maxInsertions, 4);
        else maxInsertions = Math.min(maxInsertions, 6);
      }
    }
    const injectionSlots: number[] = [];
    if (hasMajorityUpperLift && this.isMonotoneThresholdRun() && this.inputSize >= 9) {
      const breederPoolLimit = this.getThresholdBreederPoolLimit();
      const breederWindow = Array.from(
        this.selectionIndices.slice(this.config.population.eliteCount, breederPoolLimit)
      );
      breederWindow.sort((a, b) => {
        const upperDelta = this.discreteUpperBoundaryScores[b] - this.discreteUpperBoundaryScores[a];
        if (Math.abs(upperDelta) > 1e-9) return upperDelta > 0 ? 1 : -1;
        return this.currentBuffer!.fitness[a] - this.currentBuffer!.fitness[b];
      });
      for (const slot of breederWindow) {
        if (!injectionSlots.includes(slot)) injectionSlots.push(slot);
        if (injectionSlots.length >= maxInsertions) break;
      }
    }
    for (let i = this.config.population.size - 1; i >= 0 && injectionSlots.length < maxInsertions; i--) {
      const slot = this.selectionIndices[i];
      if (!injectionSlots.includes(slot)) injectionSlots.push(slot);
    }

    let anyAccepted = false;
    for (let i = 0; i < completions.length && i < injectionSlots.length; i++) {
      anyAccepted = this.applyInternalInjection(completions[i], gen, injectionSlots[i]) || anyAccepted;
    }

    if (anyAccepted) {
      if (this.shouldUseSpea2(gen)) evaluateSPEA2Buffer(this.currentBuffer, this.config.population.size);
      this.sortPopulationSelection(this.currentBuffer, this.shouldUseSpea2(gen));
    }
  }

  private applyInternalInjection(res: RefinedCandidate | { genome: Genome, fitness: number }, gen: number, targetIdx?: number): boolean {
    const genome = (res as any).genome;
    const fitness = (res as any).fitness !== undefined ? (res as any).fitness : ((res as any).log?.fitnessAfter || 0);

    // Direct binary injection if agent provides it
    if (genome.opcodes && genome.constants) {
        return this.injectGenome(genome.opcodes, genome.constants, fitness, targetIdx);
    }

    // AST fallback
    const varMap = this.domainProfile?.domainClass === 'discrete' 
      ? Array.from({ length: 32 }, (_, i) => `b${i}`) 
      : Array.from({ length: this.inputSize }, (_, i) => i === 0 ? 'x' : `x${i}`);
      
    const bridgeBake = PostfixBridge.bake(genome, varMap);
    const binaryOpcodes = new Uint32Array(bridgeBake.opcodes.length);
    for (let i = 0; i < bridgeBake.opcodes.length; i++) binaryOpcodes[i] = (bridgeBake.opcodes[i] << 24) | (bridgeBake.data[i] & 0xFFFFFF);
    return this.injectGenome(binaryOpcodes, new Float64Array(bridgeBake.constants), fitness, targetIdx);
  }

  private calculateCausalUtility(): number {
    if (!this.currentBuffer) return 0;
    let count = 0;
    for (let i = 0; i < this.config.population.size; i++) if (this.currentBuffer.ancestorMap[i] === 1) count++;
    return count / this.config.population.size;
  }

  private selectAgentLineageParent(buffer: PopulationBuffer): number | null {
    if (!this.isDiscreteRun() || (!this.discoveryAgent && !this.agentBridge)) return null;

    const thresholdRun = this.isMonotoneThresholdRun();
    const selectorMap = this.domainProfile?.signals.selectorMap;
    const muxLike = !!(selectorMap && selectorMap.some(v => v === 1) && selectorMap.some(v => v === 0));
    if ((!thresholdRun || this.inputSize < 9) && !muxLike) return null;

    const currentBest = this.getChampionScore(buffer, this.selectionIndices[0]) || 0;
    let lineageRate = muxLike ? 0.12 : 0.14;
    if (this.stagnationCount > 5 || this.currentMutationIntensity > 1.5) lineageRate *= 1.25;
    if (currentBest >= 0.96) lineageRate *= 0.45;
    if (random() >= Math.min(0.22, lineageRate)) return null;

    let bestIdx = -1;
    let bestScore = -Infinity;
    for (let i = 0; i < this.config.population.size; i++) {
      if (buffer.ancestorMap[i] !== 1) continue;
      const fitness = buffer.fitness[i];
      if (!Number.isFinite(fitness) || fitness <= 0) continue;

      const boundaryPenalty = this.shouldUseThresholdSelectionBias()
        && this.discreteBoundaryScores.length === this.config.population.size
        ? this.discreteBoundaryScores[i] * 0.35
        : 0;
      const lengthPenalty = buffer.lengths[i] * (this.inputSize >= 9 ? 0.00006 : 0.000035);
      const score = fitness - boundaryPenalty - lengthPenalty;
      if (score > bestScore) {
        bestScore = score;
        bestIdx = i;
      }
    }

    return bestIdx >= 0 ? bestIdx : null;
  }

  private selectParentIndex(buffer: PopulationBuffer, gen: number): number {
    const agentLineageParent = this.selectAgentLineageParent(buffer);
    if (agentLineageParent !== null) return agentLineageParent;

    const method = this.config.selectionMethod
      || ((this.domainProfile?.domainClass === 'discrete' || this.fitnessMode === 'classification') ? 'lexicase' : 'tournament');
    const thresholdAwareLexicase =
      this.isDiscreteRun()
      && this.inputSize >= 7
      && typeof this.domainProfile?.signals.logicThreshold === 'number';

    if (method === 'spea2' && gen >= 30) return this.binaryTournamentSelection(buffer, 10, true);
    if (method === 'lexicase') return this.binaryLexicaseSelection(buffer, thresholdAwareLexicase);
    if (method === 'epsilon-lexicase') return this.binaryLexicaseSelection(buffer, true);
    return this.binaryTournamentSelection(buffer, 10, false);
  }

  private binaryTournamentSelection(buffer: PopulationBuffer, k: number, useSpea2: boolean = false): number {
    let best = Math.floor(random() * this.config.population.size);
    for (let i = 1; i < k; i++) {
      const cand = Math.floor(random() * this.config.population.size);
      if (useSpea2) { if (buffer.spea2Score[cand] < buffer.spea2Score[best]) best = cand; }
      else { if (buffer.fitness[cand] > buffer.fitness[best]) best = cand; }
    }
    return best;
  }

  private binaryLexicaseSelection(buffer: PopulationBuffer, epsilonMode: boolean): number {
    const populationSize = this.config.population.size;
    const isDiscrete = this.isDiscreteRun();
    const subsetFloor = isDiscrete ? populationSize : 24;
    const subsetScale = isDiscrete ? populationSize : 4;
    const subsetSize = Math.min(populationSize, Math.max(this.config.population.tournamentSize * subsetScale, subsetFloor));
    const pool = new Uint32Array(subsetSize);
    if (isDiscrete && subsetSize === populationSize) {
      for (let i = 0; i < subsetSize; i++) pool[i] = i;
    } else {
      for (let i = 0; i < subsetSize; i++) pool[i] = Math.floor(random() * populationSize);
    }
    let poolLength = subsetSize;

    const totalCases = this.testCases.length;
    if (totalCases === 0) return this.binaryTournamentSelection(buffer, 10, false);

    this.buildCaseOrder(totalCases);

    const maxCases = isDiscrete ? totalCases : Math.min(totalCases, 64);
    for (let ci = 0; ci < maxCases && poolLength > 1; ci++) {
      const caseIdx = this.caseScratch[ci];
      let bestErr = Infinity;
      for (let i = 0; i < poolLength; i++) {
        const err = buffer.caseErrors[pool[i] * buffer.maxTestCases + caseIdx];
        if (err < bestErr) bestErr = err;
      }

      let epsilon = 0;
      if (epsilonMode) {
        if (typeof this.config.epsilon === 'number') epsilon = this.config.epsilon;
        else if (this.config.population.lexicaseEpsilon !== undefined) epsilon = this.config.population.lexicaseEpsilon;
        else epsilon = 0.05;
      }

      const threshold = bestErr + epsilon;
      let write = 0;
      for (let i = 0; i < poolLength; i++) {
        const idx = pool[i];
        const err = buffer.caseErrors[idx * buffer.maxTestCases + caseIdx];
        if (err <= threshold) pool[write++] = idx;
      }
      poolLength = Math.max(1, write);
    }

    if (isDiscrete) return this.selectDiscreteLexicaseWinner(buffer, pool, poolLength);

    let best = pool[Math.floor(random() * poolLength)];
    let bestLen = buffer.lengths[best];
    for (let i = 0; i < poolLength; i++) {
      const idx = pool[i];
      if (buffer.lengths[idx] < bestLen) {
        best = idx;
        bestLen = buffer.lengths[idx];
      }
    }
    return best;
  }

  public async evaluateIndividual(genome: Genome): Promise<number> {
    const tempBuffer = new PopulationBuffer(1, genome.genes.length, 1, this.testCases.length);
    const opcodes = new Uint32Array(genome.genes.length);
    const constants = new Float64Array(10);
    let cIdx = 0;
    for (let i = 0; i < genome.genes.length; i++) {
      if (genome.genes[i].operation === 'constant') {
        constants[cIdx] = genome.genes[i].operands[0] as number;
        opcodes[i] = (11 << 24) | (cIdx++ & 0xFFFFFF);
      }
    }
    tempBuffer.writeGenome(0, opcodes, [], constants);
    this.ensureTestDataBuffers();
    BlockEvaluator.evaluateBlock(tempBuffer, 0, 1, this.testXBuffer!, this.testYBuffer!, this.inputSize, this.fitnessMode, this.classificationWeight);
    return tempBuffer.fitness[0];
  }
}
