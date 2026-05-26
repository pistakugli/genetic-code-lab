/**
 * Genetic Code Lab - Core Types
 * Production-ready genetic programming engine
 */

// ============================================================================
// GENOME TYPES
// ============================================================================

/**
 * A single gene in the genome - represents a code operation
 */
export interface Gene {
  id: string;
  type: GeneType;
  operation: string; // Restored from 'name'
  operands: (number | string | Gene)[];
  metadata?: Record<string, unknown>;
}

/**
 * Types of genes supported
 */
export type GeneType =
  | 'function'      // Built-in function (add, subtract, etc.)
  | 'terminal'      // Terminal value (constant or variable)
  | 'conditional'   // If-then-else
  | 'loop'          // Loop construct
  | 'logic'         // Logic gate
  | 'void'          // Action with no return value
  | 'custom';       // User-defined function

/**
 * A complete genome (individual in population)
 */
export interface Genome {
  id: string;
  genes: Gene[];
  fitness: number;
  generation: number;
  parents?: [string, string];  // IDs of parent genomes
  age: number;                 // Current age (Generations since birth/refresh)
  noveltyScore?: number;       // For diversity-aware selection
  fitnessTrace?: string;       // Structural trace of the best individual
  multiObjectiveFitness?: MultiObjectiveFitness;
  /** Per-case errors for Lexicase / Epsilon-Lexicase selection */
  caseErrors?: number[];
  /** 1024-bit visitation map (32x32 grid coverage) for high-fidelity Novelty Search */
  visitationMap?: Uint8Array;
}

// ============================================================================
// POPULATION TYPES
// ============================================================================

/**
 * Population configuration
 */
export interface PopulationConfig {
  size: number;
  eliteCount: number;          // Number of top individuals to preserve
  crossoverRate: number;       // 0-1 probability
  mutationRate: number;        // 0-1 probability
  maxGenomeDepth: number;      // Maximum tree depth
  maxGenomeSize: number;       // Maximum number of genes
  tournamentSize: number;      // For tournament selection
  lexicaseRate?: number;       // Probability of using Lexicase vs Tournament (0.0 - 1.0)
  lexicaseEpsilon?: number;    // Epsilon threshold for Best-in-Class filtering
  parsimonyCoefficient?: number; // Penalty for large genomes (0 to ignore)
  quotientMode?: boolean;      // Phase 72.4: Root contract enforcement flag
}

/**
 * A population of genomes
 */
export interface Population {
  id: string;
  generation: number;
  genomes: Genome[];
  bestFitness: number;
  averageFitness: number;
  worstFitness: number;
  diversity: number;           // Genetic diversity metric
  config: PopulationConfig;
}

// ============================================================================
// EVOLUTION TYPES
// ============================================================================

/**
 * Evolution configuration
 */
export interface EvolutionConfig {
  population: PopulationConfig;
  maxGenerations: number;
  targetFitness: number;
  stagnationLimit: number;     // Stop if no improvement for N generations
  parallelEvaluators: number;  // Number of parallel fitness evaluations
  distributedIslands?: number; // Number of distributed islands when using multi-process orchestration
  islandParallelEvaluators?: number; // Intra-island evaluator count for distributed runs
  thresholdSelectionPolicy?: 'weighted' | 'hybrid' | 'exact';
  selectionMethod?: 'tournament' | 'roulette' | 'rank' | 'sus' | 'spea2' | 'lexicase' | 'epsilon-lexicase';
  archiveSize?: number;        // Size of the non-dominated archive (for SPEA2)
  multiObjective?: boolean;    // Enable multi-objective optimization
  crossoverMethod?: 'subtree' | 'onePoint' | 'uniform';
  solverType?: 'math' | 'math-classify' | 'boolean' | 'santa-fe' | 'financial' | 'math-quant'; // For parallel worker routing
  seed?: number;               // Random seed for reproducibility
  islandMode?: boolean;        // When true, disables HARD STOP on stagnation
  quotientMode?: boolean;      // Phase 72.3: Core support for Quotient representation
  strictQuotientValidation?: boolean; // Phase 72.3: Denominator viability and coprime checks
  /**
   * Epsilon tolerance for epsilon-lexicase (continuous domains).
   * If not set, defaults to the Median Absolute Deviation (MAD) per case.
   */
  epsilon?: number | 'mad';
  /**
   * Age-Layered Population Structure (ALPS) configuration.
   * Prevents premature convergence by segregating individuals by age.
   */
  alps?: ALPSConfig;
  /**
   * Hybrid Numerical Optimizer (Local Search).
   * Fine-tunes constants in the best individuals.
   */
  optimizer?: OptimizerConfig;
  /** Pre-built seed genomes to inject into initial population and ALPS Layer 0. */
  seedGenomes?: Genome[];
  onGeneration?: (pop: Population, stats: EvolutionStats) => void;
  onImprovement?: (genome: Genome, generation: number) => void;
}

/**
 * ALPS configuration
 */
export interface ALPSConfig {
  enabled: boolean;
  numLayers: number;        // Number of age layers
  ageGap: number;           // Generations between layers (e.g., 20)
  innovationInterval: number; // Generations between Layer 0 refreshes (e.g., 10)
}

/**
 * Gradient/Numerical Optimizer configuration
 */
export interface OptimizerConfig {
  enabled: boolean;
  iterations: number;       // Steps of hill-climbing
  learningRate: number;     // Step size for adjustment
  interval: number;         // Run every N generations
  targetCount: number;      // Number of top individuals to optimize
}

/**
 * Evolution statistics
 */
export interface EvolutionStats {
  generation: number;
  bestFitness: number;
  averageFitness: number;
  worstFitness: number;
  diversity: number;
  improvementRate: number;     // Fitness improvement per generation
  timeElapsed: number;         // ms
  evaluationsCount: number;
  stagnationCount: number;
  averageSize?: number;        // Average number of genes in population
  phaseTimes?: {
    selection: number;
    reproduction: number;
    evaluation: number;
    agent: number;
  };
  agentUtility?: number;    // % of top 100 genomes with agent ancestry
  agentStaleness?: number; // % of refined candidates discarded as stale
}

/**
 * Evolution result
 */
export interface EvolutionResult {
  bestGenome: Genome;
  generationsRun: number;
  totalEvaluations: number;
  totalTimeMs: number;
  success: boolean;
  stats: EvolutionStats[];
  finalPopulation?: Population;
  archive?: Genome[];
}

// ============================================================================
// META-CONTROL TYPES (PHASE 45: ADAPTIVE DIRECTOR)
// ============================================================================

/**
 * Snapshot of population dynamics for the Adaptive Director to observe.
 */
export interface PopulationState {
  generation: number;
  bestFitness: number;
  fitnessTrend: number;   // Slope of improvement over last N gens
  entropy: number;        // Genetic diversity
  avgSize: number;        // Complexity
  stagnationCount: number;
  islandId: number;
}

/**
 * Strategic actions the Director can dispatch to shift the search regime.
 */
export enum EvolutionAction {
  WAIT = 'WAIT',
  MUTATION_KICK = 'MUTATION_KICK', // Intense structural noise
  CATACLYSM = 'CATACLYSM',         // Mass extinction event
  PARSIMONY_SHIFT = 'PARSIMONY_SHIFT', // Focus on size vs accuracy
  OPERATOR_SHIFT = 'OPERATOR_SHIFT'    // Change active grammar
}

/**
 * Command envelope sent from Agent back to the Engine.
 */
export interface EvolutionCommand {
  action: EvolutionAction;
  intensity: number;      // e.g., 5.0 for mutation kick
  duration: number;       // How many generations the effect lasts
}

// ============================================================================
// FITNESS TYPES
// ============================================================================

/**
 * Multi-objective fitness values (e.g. [error, size])
 */
export interface MultiObjectiveFitness {
  values: number[];
  strength?: number;      // SPEA2: How many individuals this one dominates
  rawFitness?: number;    // SPEA2: Sum of strengths of individuals that dominate this one
  density?: number;       // SPEA2: k-th nearest neighbor distance
  score?: number;         // SPEA2: R(i) + D(i)
}

/**
 * Fitness function interface
 */
export type FitnessFunction<TInput, TOutput> = (
  genome: Genome,
  testCases: TestCase<TInput, TOutput>[]
) => number | MultiObjectiveFitness;

/**
 * Test case for fitness evaluation
 */
export interface TestCase<TInput, TOutput> {
  input: TInput;
  expectedOutput: TOutput;
  weight?: number;             // Importance of this test case
}

// ============================================================================
// OPERATOR TYPES
// ============================================================================

/**
 * Selection operator
 */
export type SelectionOperator = (
  population: Population,
  count: number
) => Genome[];

/**
 * Crossover operator
 */
export type CrossoverOperator = (
  parent1: Genome,
  parent2: Genome,
  config?: PopulationConfig
) => [Genome, Genome];

/**
 * Mutation operator
 */
export type MutationOperator = (
  genome: Genome,
  config: PopulationConfig
) => Genome;

// ============================================================================
// PROBLEM SOLVER TYPES
// ============================================================================

/**
 * Problem definition for the solver
 */
export interface ProblemDefinition<TInput, TOutput> {
  name?: string;
  description?: string;
  functionSet: FunctionDefinition[];   // Available functions
  terminalSet: TerminalDefinition[];   // Available terminals
  testCases: TestCase<TInput, TOutput>[];
  fitnessFunction: FitnessFunction<TInput, TOutput>;
  config?: Partial<EvolutionConfig>;
  /** Explicit fitness evaluation mode. Default: 'regression' */
  fitnessMode?: 'regression' | 'classification';
  /** Weight of classification accuracy in blended fitness (0-1). Default: 0.8 */
  classificationWeight?: number;
}

/**
 * Function definition for the problem
 */
export interface FunctionDefinition {
  name: string;
  arity: number;               // Number of arguments
  implementation: (...args: unknown[]) => unknown;
  returnType: string;
  description?: string;
}

/**
 * Terminal definition for the problem
 */
export interface TerminalDefinition {
  name: string;
  type: 'constant' | 'variable' | 'input' | 'void';
  value?: unknown;
  description?: string;
}

// ============================================================================
// PERSISTENCE TYPES
// ============================================================================

/**
 * Run record for persistence
 */
export interface RunRecord {
  id: string;
  problem: string;
  config: EvolutionConfig;
  startTime: string;
  endTime?: string;
  status: 'running' | 'completed' | 'failed';
  result?: EvolutionResult;
}

// ============================================================================
// ERROR TYPES
// ============================================================================

export class GeneticCodeLabError extends Error {
  constructor(message: string, public code: string, public details?: unknown) {
    super(message);
    this.name = 'GeneticCodeLabError';
  }
}

export class FitnessEvaluationError extends GeneticCodeLabError {
  constructor(message: string, details?: unknown) {
    super(message, 'FITNESS_EVALUATION_ERROR', details);
    this.name = 'FitnessEvaluationError';
  }
}

export class GenomeValidationError extends GeneticCodeLabError {
  constructor(message: string, details?: unknown) {
    super(message, 'GENOME_VALIDATION_ERROR', details);
    this.name = 'GenomeValidationError';
  }
}

// Re-export for convenience
export type { EvolutionResult as EvolutionResultType };
export type { EvolutionStats as EvolutionStatsType };

// Default population config
export const DEFAULT_POPULATION_CONFIG: PopulationConfig = {
  size: 100,
  eliteCount: 2,
  crossoverRate: 0.8,
  mutationRate: 0.2,
  maxGenomeDepth: 8,
  maxGenomeSize: 100,
  tournamentSize: 3,
  parsimonyCoefficient: 0.002
};

export class EvolutionStagnationError extends GeneticCodeLabError {
  constructor(message: string, details?: unknown) {
    super(message, 'EVOLUTION_STAGNATION_ERROR', details);
    this.name = 'EvolutionStagnationError';
  }
}
