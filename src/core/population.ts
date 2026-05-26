/**
 * Genetic Code Lab - Population
 * Manages a population of genomes
 */

import {
  Population,
  Genome,
  PopulationConfig,
  FunctionDefinition,
  TerminalDefinition,
  DEFAULT_POPULATION_CONFIG
} from './types';
import { GenomeFactory } from './genome';
import {
  generateId,
  mean,
  stdDev,
  sortBy,
  deepClone
} from '../utils/helpers';

// Re-export for backward compatibility
export { DEFAULT_POPULATION_CONFIG };

// ============================================================================
// POPULATION CLASS
// ============================================================================

export class PopulationManager {
  private config: PopulationConfig;
  public factory: GenomeFactory;

  constructor(
    functionSet: FunctionDefinition[],
    terminalSet: TerminalDefinition[],
    config: Partial<PopulationConfig> = {}
  ) {
    this.config = { ...DEFAULT_POPULATION_CONFIG, ...config };
    this.factory = new GenomeFactory(functionSet, terminalSet, this.config);
  }

  /**
   * Initialize a new random population, optionally seeded with structural templates
   */
  initialize(seedGenomes?: Genome[]): Population {
    const genomes = this.factory.createPopulation(this.config.size);
    
    // Inject seed genomes by replacing the last N random individuals
    if (seedGenomes && seedGenomes.length > 0) {
      const seedCount = Math.min(seedGenomes.length, Math.floor(this.config.size * 0.1)); // Max 10% seeds
      for (let i = 0; i < seedCount; i++) {
        genomes[genomes.length - 1 - i] = seedGenomes[i];
      }
      console.log(`[PopManager] Injected ${seedCount} seed template genomes into initial population.`);
    }
    
    return this.createPopulationRecord(genomes, 0);
  }

  /**
   * Create population from existing genomes
   */
  createPopulationRecord(genomes: Genome[], generation: number): Population {
    const fitnesses = genomes.map(g => g.fitness);

    return {
      id: generateId(),
      generation,
      genomes,
      bestFitness: Math.max(...fitnesses),
      averageFitness: mean(fitnesses),
      worstFitness: Math.min(...fitnesses),
      diversity: this.calculateDiversity(genomes),
      config: this.config
    };
  }

  /**
   * Calculate genetic diversity of the population
   */
  private calculateDiversity(genomes: Genome[]): number {
    if (genomes.length < 2) return 0;

    // Calculate diversity based on structural differences
    let totalDistance = 0;
    let comparisons = 0;

    for (let i = 0; i < Math.min(genomes.length, 50); i++) {
      for (let j = i + 1; j < Math.min(genomes.length, 50); j++) {
        totalDistance += this.genomeDistance(genomes[i], genomes[j]);
        comparisons++;
      }
    }

    return comparisons > 0 ? totalDistance / comparisons : 0;
  }

  /**
   * Calculate distance between two genomes
   */
  private genomeDistance(g1: Genome, g2: Genome): number {
    // Simple structural distance based on size and depth differences
    const sizeDiff = Math.abs(g1.genes.length - g2.genes.length);
    
    // Handle -Infinity fitness safely to avoid NaN
    const f1 = Number.isFinite(g1.fitness) ? g1.fitness : -1e10;
    const f2 = Number.isFinite(g2.fitness) ? g2.fitness : -1e10;
    const fitnessDiff = Math.abs(f1 - f2);

    return sizeDiff / 10 + (fitnessDiff / 1000); // Scale down fitness impact
  }

  /**
   * Get elite genomes (best performers)
   */
  getElites(population: Population, count?: number): Genome[] {
    const eliteCount = count ?? this.config.eliteCount;
    const sorted = sortBy(population.genomes, g => g.fitness, true);
    return sorted.slice(0, eliteCount).map(g => deepClone(g));
  }

  /**
   * Get best genome
   */
  getBest(population: Population): Genome {
    return population.genomes.reduce((best, current) =>
      current.fitness > best.fitness ? current : best
    );
  }

  /**
   * Get worst genome
   */
  getWorst(population: Population): Genome {
    return population.genomes.reduce((worst, current) =>
      current.fitness < worst.fitness ? current : worst
    );
  }

  /**
   * Get population statistics
   */
  getStats(population: Population): {
    best: number;
    worst: number;
    average: number;
    stdDev: number;
    median: number;
    diversity: number;
    size: number;
  } {
    const fitnesses = population.genomes.map(g => g.fitness);
    const sorted = [...fitnesses].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);

    return {
      best: population.bestFitness,
      worst: population.worstFitness,
      average: population.averageFitness,
      stdDev: stdDev(fitnesses),
      median: sorted.length % 2 !== 0
        ? sorted[mid]
        : (sorted[mid - 1] + sorted[mid]) / 2,
      diversity: population.diversity,
      size: population.genomes.length
    };
  }

  /**
   * Replace population with new generation
   */
  replacePopulation(
    oldPopulation: Population,
    newGenomes: Genome[],
    elites: Genome[]
  ): Population {
    // Combine elites with new genomes
    const combined = [...elites, ...newGenomes];

    // Trim to population size if necessary
    const finalGenomes = combined.slice(0, this.config.size);

    return this.createPopulationRecord(finalGenomes, oldPopulation.generation + 1);
  }

  /**
   * Update fitness values in population
   */
  updateFitness(population: Population, fitnessMap: Map<string, number>): Population {
    const updatedGenomes = population.genomes.map(g => ({
      ...g,
      fitness: fitnessMap.get(g.id) ?? g.fitness
    }));

    return this.createPopulationRecord(updatedGenomes, population.generation);
  }

  /**
   * Check if population has converged
   */
  hasConverged(population: Population, threshold: number = 0.01): boolean {
    return population.diversity < threshold;
  }

  /**
   * Filter genomes to only include those on the Pareto front (fitness vs size)
   * To prevent bloat: remove individuals dominated by others (same/better fitness + smaller size)
   */
  public getParetoFront(genomes: Genome[]): Genome[] {
    if (genomes.length === 0) return [];
    
    return genomes.filter(g => {
      const gSize = g.genes.length;
      const gFit = g.fitness;

      return !genomes.some(other => {
        if (other.id === g.id) return false;
        const oSize = other.genes.length;
        const oFit = other.fitness;
        
        // "other" dominates "g" if it's better in one and not worse in the other
        // BUT: avoid pruning structurally diverse genomes in early search (fitness 0)
        if (oFit === 0 && gFit === 0) return false;

        return (oFit >= gFit && oSize < gSize) || (oFit > gFit && oSize <= gSize);
      });
    });
  }

  /**
   * Cataclysmic Recovery (Beyond SOTA)
   * Execute Mass Extinction to break extreme stagnation.
   * - 5% Elites are preserved
   * - 45% Hyper-Mutated Elites (Macro structural mutations)
   * - 50% Pure Random Immigrants
   */
  triggerCataclysm(population: Population): Population {
    const totalSize = population.genomes.length;
    const eliteCount = Math.max(1, Math.floor(totalSize * 0.05));
    const mutatedCount = Math.floor(totalSize * 0.45);
    const randomCount = totalSize - eliteCount - mutatedCount;

    // 1. Preserve absolute top Elites
    const sorted = sortBy(population.genomes, g => g.fitness, true);
    const elites = sorted.slice(0, eliteCount).map(g => deepClone(g));

    // 2. Hyper-Mutate Elites
    const mutatedGenomes: Genome[] = [];
    const factory = this.factory;
    for (let i = 0; i < mutatedCount; i++) {
        const baseElite = elites[i % eliteCount];
        // Apply heavy structural mutation (Subtree mutation)
        let mutated = deepClone(baseElite);
        mutated.id = generateId();
        mutated.fitness = -Infinity;
        mutated.generation = population.generation + 1;
        
        // Find a random function node to replace with a completely new tree
        const functionIndices = mutated.genes
          .map((g, idx) => g.type === 'function' ? idx : -1)
          .filter(idx => idx !== -1);
          
        if (functionIndices.length > 0) {
            const targetIdx = functionIndices[Math.floor(Math.random() * functionIndices.length)];
            const targetId = mutated.genes[targetIdx].id;
            // Generate a heavy new subtree (depth 5)
            const newSubtreeGenes = factory.createRandomTree(5);
            
            // Inline replacement logic (simplified for Cataclysm)
            try {
              // Temporary safe-guard, if it fails, just use the original elite
              const GenomeModule = require('./genome');
              mutated.genes = GenomeModule.replaceSubtree(mutated.genes, targetId, newSubtreeGenes);
            } catch (e) {
               // Fallback: mutate a terminal
               const termIndices = mutated.genes.map((g, idx) => g.type === 'terminal' ? idx : -1).filter(idx => idx !== -1);
               if (termIndices.length > 0) {
                   const tIdx = termIndices[Math.floor(Math.random() * termIndices.length)];
                   mutated.genes[tIdx] = factory.createRandomTree(1)[0];
               }
            }
        }
        
        // Ensure size limits are respected
        if (mutated.genes.length > (this.config.maxGenomeSize || 500)) {
            mutated.genes = deepClone(baseElite.genes);
        }
        
        mutatedGenomes.push(mutated);
    }

    // 3. Generate pure random immigrants
    const randomGenomes = this.factory.createPopulation(randomCount).map(g => {
        g.generation = population.generation + 1;
        return g;
    });

    console.error(`[Cataclysm] Executed Mass Extinction. Preserved: ${eliteCount}. Mutated Elites: ${mutatedCount}. Fresh: ${randomCount}.`);

    return this.createPopulationRecord(
      [...elites, ...mutatedGenomes, ...randomGenomes],
      population.generation
    );
  }
}

// ============================================================================
// POPULATION SERIALIZATION
// ============================================================================

/**
 * Serialize population to JSON
 */
export function serializePopulation(population: Population): string {
  return JSON.stringify(population, null, 2);
}

/**
 * Deserialize population from JSON
 */
export function deserializePopulation(json: string): Population {
  return JSON.parse(json) as Population;
}
