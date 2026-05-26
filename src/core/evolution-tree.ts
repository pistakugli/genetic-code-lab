/**
 * Genetic Code Lab - Tree-based Evolution Engine
 * Original tree-based GP implementation for non-vectorizable problems.
 */

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

export type { EvolutionResult, EvolutionStats, EvolutionConfig };

import { PopulationManager, DEFAULT_POPULATION_CONFIG } from './population';
import {
  getSelectionOperator,
  getCrossoverOperator,
  applyMutation
} from './operators';
import {
  setSeed,
  random,
  mean,
  generateId
} from '../utils/helpers';
import { SymbolicSimplifier } from './simplifier';

export const DEFAULT_EVOLUTION_CONFIG: EvolutionConfig = {
  population: DEFAULT_POPULATION_CONFIG,
  maxGenerations: 100,
  targetFitness: 1.0,
  stagnationLimit: 20,
  parallelEvaluators: 1
};

export class EvolutionEngineTree<TInput = unknown, TOutput = unknown> {
  private config: EvolutionConfig;
  private populationManager: PopulationManager;
  private functionSet: FunctionDefinition[];
  private terminalSet: TerminalDefinition[];
  private fitnessFunction: FitnessFunction<TInput, TOutput>;
  private testCases: TestCase<TInput, TOutput>[];

  private stats: EvolutionStats[] = [];
  private startTime: number = 0;
  private evaluationsCount: number = 0;
  private stagnationCount: number = 0;
  private bestFitnessSoFar: number = -Infinity;
  private agent: any = null;

  constructor(
    problem: ProblemDefinition<TInput, TOutput>,
    config: Partial<EvolutionConfig> = {}
  ) {
    this.config = { ...DEFAULT_EVOLUTION_CONFIG, ...config };

    // Merge population config
    this.config.population = {
      ...DEFAULT_EVOLUTION_CONFIG.population,
      ...config.population,
      quotientMode: this.config.quotientMode // Ensure root contract is inherited
    };

    this.functionSet = problem.functionSet;
    this.terminalSet = problem.terminalSet;
    this.fitnessFunction = problem.fitnessFunction;
    this.testCases = problem.testCases;

    this.populationManager = new PopulationManager(
      this.functionSet,
      this.terminalSet,
      this.config.population as PopulationConfig
    );

    // Set random seed if provided
    if (this.config.seed !== undefined) {
      setSeed(this.config.seed);
    }
  }

  public setDiscoveryAgent(agent: any) {
    this.agent = agent;
  }

  /**
   * Run the evolution
   */
  async evolve(): Promise<EvolutionResult> {
    this.startTime = Date.now();
    this.stats = [];
    this.evaluationsCount = 0;
    this.stagnationCount = 0;
    this.bestFitnessSoFar = -Infinity;

    // Initialize population
    let population = this.populationManager.initialize();

    // Evaluate initial population
    await this.evaluatePopulation(population);

    // Record initial stats
    this.recordStats(population);

    // Main evolution loop
    for (let gen = 0; gen < this.config.maxGenerations; gen++) {
      // Check termination conditions
      const bestGenome = this.populationManager.getBest(population);

      // Target fitness reached?
      if (bestGenome.fitness >= this.config.targetFitness) {
        return this.createResult(population, true);
      }

      // Stagnation check
      if (bestGenome.fitness > this.bestFitnessSoFar) {
        this.bestFitnessSoFar = bestGenome.fitness;
        this.stagnationCount = 0;

        // Callback for improvement
        if (this.config.onImprovement) {
          this.config.onImprovement(bestGenome, gen);
        }
      } else {
        this.stagnationCount++;
      }

      if (this.stagnationCount >= this.config.stagnationLimit) {
        return this.createResult(population, false);
      }

      // Generate next generation
      population = this.nextGeneration(population);

      // Evaluate new population
      await this.evaluatePopulation(population);

      // Record stats
      this.recordStats(population);

      // Callback for generation
      if (this.config.onGeneration) {
        this.config.onGeneration(population, this.stats[this.stats.length - 1]);
      }

      // Phase 74.1: Structural Diversity Injection
      if (this.config.quotientMode && this.stagnationCount > 0 && this.stagnationCount % 10 === 0) {
        // Replace the bottom 10% of the population with fresh quotient trees to break flat optima
        const replaceCount = Math.floor(population.genomes.length * 0.1);
        const freshGenomes = this.populationManager.factory.createPopulation(replaceCount);
        
        // Sort population by fitness descending (best first)
        population.genomes.sort((a, b) => b.fitness - a.fitness);
        
        // Replace worst individuals
        for (let i = 0; i < replaceCount; i++) {
             population.genomes[population.genomes.length - 1 - i] = freshGenomes[i];
        }
      }
    }

    return this.createResult(population, false);
  }

  private evaluateAST(genome: Genome, nodeId: string, inputs: number[], nodeMap: Map<string, any>): number {
    const node = nodeMap.get(nodeId);
    if (!node) return 0;

    if (node.type === 'terminal') {
        if (node.operation === 'const' || node.operation === 'constant') return node.operands[0] as number;
        const idx = parseInt(node.operation.replace('x', ''));
        return inputs[idx] || 0;
    }

    const left = this.evaluateAST(genome, node.operands[0] as string, inputs, nodeMap);
    const right = node.operands[1] ? this.evaluateAST(genome, node.operands[1] as string, inputs, nodeMap) : 0;

    switch (node.operation) {
        case 'add': return left + right;
        case 'sub': return left - right;
        case 'mul': return left * right;
        case 'div': return right === 0 ? NaN : left / right;
        default: return 0;
    }
  }

  /**
   * Evaluate fitness for all genomes in population
   */
  private async evaluatePopulation(population: Population): Promise<void> {
    for (const genome of population.genomes) {
      if (genome.fitness === -Infinity) {
        
        // Phase 72.3: Core Quotient Integration
        if (this.config.quotientMode) {
            genome.fitness = this.evaluateQuotient(genome);
        } else {
            const result = this.fitnessFunction(genome, this.testCases);
            
            if (typeof result === 'number') {
              genome.fitness = result;
            } else {
              // Handle MultiObjectiveFitness
              genome.multiObjectiveFitness = result;
              // Use SPEA2 score if available, otherwise first value (assumed error/fitness)
              genome.fitness = result.score !== undefined ? 1 / (1 + result.score) : result.values[0];
            }
        }

        this.evaluationsCount++;

        // Check for invalid fitness
        if (!Number.isFinite(genome.fitness)) {
          genome.fitness = -Infinity;
        }
      }
    }
  }

  private evaluateQuotient(genome: Genome): number {
      if (!genome || genome.genes.length === 0) return 0.000001;
      
      const root = genome.genes[0];
      if (root.type === 'terminal' || !root.operands[1]) return 0.000001;

      const nodeMap = new Map();
      for (const g of genome.genes) nodeMap.set(g.id, g);

      let penalty = 0;
      
      if (this.config.strictQuotientValidation) {
          const probePoints = [
              [0, 0.5], [0.5, 0], [0, 0], [0.1, 0.1], [0.9, 0.9], [0.5, 0.5]
          ];
          
          for (const probe of probePoints) {
              const numProbe = this.evaluateAST(genome, root.operands[0] as string, probe, nodeMap);
              const denProbe = this.evaluateAST(genome, root.operands[1] as string, probe, nodeMap);
              
              const absNum = Math.abs(numProbe);
              const absDen = Math.abs(denProbe);

              if (absNum < 0.1 && absDen < 0.1) {
                  penalty += (0.1 - absNum) + (0.1 - absDen); 
              }
              
              if (absDen < 0.1) {
                  penalty += (0.1 - absDen) * 10; 
              }
          }
      }

      let totalError = 0;
      genome.caseErrors = [];
      for (const tc of this.testCases) {
          const num = this.evaluateAST(genome, root.operands[0] as string, tc.input as number[], nodeMap);
          const rawDenom = this.evaluateAST(genome, root.operands[1] as string, tc.input as number[], nodeMap);
          
          let safeDenom = rawDenom;
          if (Math.abs(rawDenom) < 1e-3) {
              safeDenom = rawDenom >= 0 ? 1e-3 : -1e-3;
              penalty += 1.0; 
          }
          
          const pred = num / safeDenom;
          let tcError = 0;
          if (isNaN(pred) || !isFinite(pred)) {
              tcError = 10;
          } else {
              tcError = Math.abs(pred - (tc.expectedOutput as number));
          }
          
          genome.caseErrors.push(tcError + penalty); // Include penalty in case error for lexicase
          totalError += tcError;
      }
      
      const mae = (totalError / this.testCases.length) + penalty;
      return 1 / (1 + mae);
  }

  /**
   * Create next generation
   */
  private nextGeneration(currentPopulation: Population): Population {
    const config = this.config.population;
    const newGenomes: Genome[] = [];

    // Get elites
    const elites = this.populationManager.getElites(currentPopulation);

    // Select parents
    const selectionOp = getSelectionOperator(this.config.selectionMethod || 'tournament');
    const parents = selectionOp(currentPopulation, config.size - config.eliteCount);

    // Get crossover operator
    const crossoverOp = getCrossoverOperator(this.config.crossoverMethod || 'subtree');

    // Create offspring
    for (let i = 0; i < parents.length - 1; i += 2) {
      const parent1 = parents[i];
      const parent2 = parents[i + 1];

      // Crossover
      if (random() < config.crossoverRate) {
        const [child1, child2] = crossoverOp(parent1, parent2, { ...config, quotientMode: this.config.quotientMode } as PopulationConfig);
        newGenomes.push(child1, child2);
      } else {
        newGenomes.push(parent1, parent2);
      }
    }

    // Handle odd parent
    if (parents.length % 2 !== 0) {
      newGenomes.push(parents[parents.length - 1]);
    }

    // Mutation and Simplification
    for (let i = 0; i < newGenomes.length; i++) {
      if (random() < config.mutationRate) {
        newGenomes[i] = applyMutation(
          newGenomes[i],
          { ...config, quotientMode: this.config.quotientMode } as PopulationConfig,
          this.functionSet,
          this.terminalSet
        );
      }
      // Phase 74.1: Prevent Bloat through Native AST Simplification
      SymbolicSimplifier.simplifyAST(newGenomes[i], !!this.config.quotientMode);
    }

    // Create new population
    return this.populationManager.replacePopulation(
      currentPopulation,
      newGenomes.slice(0, config.size - config.eliteCount),
      elites
    );
  }

  /**
   * Record statistics for current generation
   */
  private recordStats(population: Population): void {
    const fitnesses = population.genomes.map(g => g.fitness);
    const prevBest = this.stats.length > 0
      ? this.stats[this.stats.length - 1].bestFitness
      : -Infinity;

    const stats: EvolutionStats = {
      generation: population.generation,
      bestFitness: population.bestFitness,
      averageFitness: population.averageFitness,
      worstFitness: population.worstFitness,
      diversity: population.diversity,
      improvementRate: population.bestFitness - prevBest,
      timeElapsed: Date.now() - this.startTime,
      evaluationsCount: this.evaluationsCount,
      stagnationCount: this.stagnationCount
    };

    this.stats.push(stats);
  }

  /**
   * Create final result
   */
  private createResult(population: Population, success: boolean): EvolutionResult {
    const bestGenome = this.populationManager.getBest(population);

    return {
      success,
      bestGenome,
      finalPopulation: population,
      stats: this.stats,
      generationsRun: population.generation,
      totalEvaluations: this.evaluationsCount,
      totalTimeMs: Date.now() - this.startTime
    };
  }
}
