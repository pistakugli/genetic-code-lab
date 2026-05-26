/**
 * Genetic Code Lab - Genetic Operators
 * Selection, Crossover, and Mutation operators
 */

import {
  Genome,
  Gene,
  Population,
  PopulationConfig,
  SelectionOperator,
  CrossoverOperator,
  MutationOperator,
  FunctionDefinition,
  TerminalDefinition
} from './types';
import {
  generateId,
  random,
  randomChoice,
  randomInt,
  randomBool,
  deepClone
} from '../utils/helpers';
import { getGenomeDepth, validateGenome, GenomeValidationError, collectSubtree, replaceSubtree, GenomeFactory, rebindGeneIds } from './genome';

// ============================================================================
// SELECTION OPERATORS
// ============================================================================

/**
 * Tournament selection - select best from random subset
 */
export function tournamentSelection(
  population: Population,
  count: number
): Genome[] {
  const selected: Genome[] = [];
  const tournamentSize = population.config.tournamentSize;

  for (let i = 0; i < count; i++) {
    // Random selection with replacement
    const tournament: Genome[] = [];
    for (let j = 0; j < tournamentSize; j++) {
      const idx = randomInt(0, population.genomes.length - 1);
      tournament.push(population.genomes[idx]);
    }

    // Select best from tournament
    const winner = tournament.reduce((best, current) => {
      if (current.fitness > best.fitness) return current;
      
      // Anti-Bloat Tiebreaker: Prefer smaller genomes, BUT ONLY if fitness is actually meaningful (> 0)
      // OR probabilistically (20% of the time) to prevent culling all large trees in Generation 0
      if (current.fitness === best.fitness && current.genes.length < best.genes.length) {
        if (current.fitness > 0 || random() < 0.2) {
          return current;
        }
      }
      return best;
    });

    selected.push(deepClone(winner));
  }

  return selected;
}

/**
 * Roulette wheel selection (fitness proportionate)
 */
export function rouletteSelection(
  population: Population,
  count: number
): Genome[] {
  const selected: Genome[] = [];

  // Normalize fitnesses (handle negative values)
  const minFitness = Math.min(...population.genomes.map(g => g.fitness));
  const adjustedFitnesses = population.genomes.map(g => g.fitness - minFitness + 1);
  const totalFitness = adjustedFitnesses.reduce((sum, f) => sum + f, 0);

  for (let i = 0; i < count; i++) {
    let cumulative = 0;
    const threshold = random() * totalFitness;

    for (let j = 0; j < population.genomes.length; j++) {
      cumulative += adjustedFitnesses[j];
      if (cumulative >= threshold) {
        selected.push(deepClone(population.genomes[j]));
        break;
      }
    }
  }

  return selected;
}

/**
 * Rank selection
 */
export function rankSelection(
  population: Population,
  count: number
): Genome[] {
  const selected: Genome[] = [];

  // Sort by fitness
  const sorted = [...population.genomes].sort((a, b) => b.fitness - a.fitness);

  // Calculate rank probabilities
  const n = sorted.length;
  const totalRank = (n * (n + 1)) / 2;

  for (let i = 0; i < count; i++) {
    let cumulative = 0;
    const threshold = random() * totalRank;

    for (let j = 0; j < n; j++) {
      cumulative += n - j; // Higher rank = higher probability
      if (cumulative >= threshold) {
        selected.push(deepClone(sorted[j]));
        break;
      }
    }
  }

  return selected;
}

/**
 * Stochastic universal sampling
 */
export function stochasticUniversalSampling(
  population: Population,
  count: number
): Genome[] {
  const selected: Genome[] = [];

  // Normalize fitnesses
  const minFitness = Math.min(...population.genomes.map(g => g.fitness));
  const adjustedFitnesses = population.genomes.map(g => g.fitness - minFitness + 1);
  const totalFitness = adjustedFitnesses.reduce((sum, f) => sum + f, 0);

  const distance = totalFitness / count;
  let start = random() * distance;

  const pointers: number[] = [];
  for (let i = 0; i < count; i++) {
    pointers.push(start + i * distance);
  }

  let cumulative = 0;
  let genomeIdx = 0;

  for (const pointer of pointers) {
    while (cumulative < pointer && genomeIdx < population.genomes.length) {
      cumulative += adjustedFitnesses[genomeIdx];
      genomeIdx++;
    }
    if (genomeIdx > 0) {
      selected.push(deepClone(population.genomes[genomeIdx - 1]));
    }
  }

  return selected;
}

/**
 * Epsilon-Lexicase Selection
 * Evaluates individuals based on specific test cases randomly ordered.
 * Filters out individuals that are not within epsilon of the elite performance on that case.
 */
export function epsilonLexicaseSelection(
  population: Population,
  count: number
): Genome[] {
  const selected: Genome[] = [];
  
  // Epsilon value could come from config, but let's default to a small tolerance for structural discovery
  const epsilon = population.config.parsimonyCoefficient || 1e-4; 
  
  for (let i = 0; i < count; i++) {
    // Optimizing Lexicase Selection via Stochastic Sampling
    // Instead of reducing 10,000 individuals repeatedly, we draw a random
    // subset of 50 individuals to form an 'Epsilon-Lexicase Tournament'.
    const subsetSize = Math.max(10, Math.floor(population.genomes.length * 0.015));
    let candidates: Genome[] = [];
    for (let j = 0; j < subsetSize; j++) {
       const idx = randomInt(0, population.genomes.length - 1);
       candidates.push(population.genomes[idx]);
    }
    
    // We need case errors to run lexicase. If missing, fallback to tournament
    if (!candidates[0].caseErrors || candidates[0].caseErrors.length === 0) {
      selected.push(tournamentSelection(population, 1)[0]);
      continue;
    }

    const numCases = candidates[0].caseErrors.length;
    // Create random test case ordering using stable seeded Shuffle
    const caseOrder = Array.from({length: numCases}, (_, idx) => idx);
    for (let j = caseOrder.length - 1; j > 0; j--) {
      const k = Math.floor(random() * (j + 1));
      [caseOrder[j], caseOrder[k]] = [caseOrder[k], caseOrder[j]];
    }

    for (const testIdx of caseOrder) {
      // Find the best (lowest) error among candidates on this specific test case
      let bestError = Infinity;
      for (const cand of candidates) {
          if (!cand.caseErrors) continue;
          const err = cand.caseErrors[testIdx];
          if (err !== undefined && err < bestError) bestError = err;
      }
      
      const threshold = bestError + epsilon;
      
      // Filter candidates that are outside the epsilon threshold
      candidates = candidates.filter(c => c.caseErrors && c.caseErrors[testIdx] !== undefined && c.caseErrors[testIdx] <= threshold);
      
      if (candidates.length === 1) break;
    }
    
    // Select randomly from remaining tied candidates
    selected.push(deepClone(randomChoice(candidates)));
  }

  return selected;
}

// ============================================================================
// CROSSOVER OPERATORS
// ============================================================================

/**
 * Subtree crossover — swaps entire subtrees between two parents.
 * Correctly handles tree references by collecting all descendant genes.
 */
export function subtreeCrossover(
  parent1: Genome,
  parent2: Genome,
  config?: PopulationConfig
): [Genome, Genome] {
  // Clone parents deeply
  const child1Genes = deepClone(parent1.genes);
  const child2Genes = deepClone(parent2.genes);

  if (child1Genes.length < 1 || child2Genes.length < 1) {
    return [
      { id: generateId(), genes: child1Genes, fitness: -Infinity, generation: Math.max(parent1.generation, parent2.generation) + 1, parents: [parent1.id, parent2.id], age: 0 },
      { id: generateId(), genes: child2Genes, fitness: -Infinity, generation: Math.max(parent1.generation, parent2.generation) + 1, parents: [parent1.id, parent2.id], age: 0 }
    ];
  }

  // Select crossover points (random gene indices, avoid root for safety)
  let point1Idx, point2Idx;
  
  if (config?.quotientMode) {
    // Phase 72.4: Quotient Root Contract Enforcement
    // 1. Choose branch: Numerator (0) or Denominator (1)
    const branchIdx = randomInt(0, 1); 
    
    // 2. Identify root operands (Numerator/Denominator IDs)
    const root1 = child1Genes[0];
    const root2 = child2Genes[0];
    
    if (!root1.operands[1] || !root2.operands[1]) {
       // Fallback if structure is broken
       point1Idx = child1Genes.length > 1 ? randomInt(1, child1Genes.length - 1) : 0;
       point2Idx = child2Genes.length > 1 ? randomInt(1, child2Genes.length - 1) : 0;
    } else {
       // 3. Find all indices belonging to the chosen branch for both parents
       const branchId1 = root1.operands[branchIdx] as string;
       const branchId2 = root2.operands[branchIdx] as string;
       
       const subtreeIds1 = collectSubtree(parent1, branchId1).map(g => g.id);
       const subtreeIds2 = collectSubtree(parent2, branchId2).map(g => g.id);
       
       const branchIndices1 = child1Genes.map((g, i) => subtreeIds1.includes(g.id) ? i : -1).filter(i => i !== -1);
       const branchIndices2 = child2Genes.map((g, i) => subtreeIds2.includes(g.id) ? i : -1).filter(i => i !== -1);
       
       point1Idx = branchIndices1.length > 0 ? randomChoice(branchIndices1) : 0;
       point2Idx = branchIndices2.length > 0 ? randomChoice(branchIndices2) : 0;
    }
  } else {
    point1Idx = child1Genes.length > 1 ? randomInt(1, child1Genes.length - 1) : 0;
    point2Idx = child2Genes.length > 1 ? randomInt(1, child2Genes.length - 1) : 0;
  }

  const targetId1 = child1Genes[point1Idx].id;
  const targetId2 = child2Genes[point2Idx].id;

  // Collect full subtrees from each child and HYGIENICALLY rebind their IDs,
  // otherwise validateGenome will throw 'Duplicate gene ID' when swapping into parents
  // that might share lineage, causing silent evolutionary failure and tree shrinkage.
  const tempGenome1: Genome = { id: '', genes: child1Genes, fitness: 0, generation: 0, age: 0 };
  const tempGenome2: Genome = { id: '', genes: child2Genes, fitness: 0, generation: 0, age: 0 };

  const subtree1 = rebindGeneIds(collectSubtree(tempGenome1, targetId1));
  const subtree2 = rebindGeneIds(collectSubtree(tempGenome2, targetId2));

  // Swap: replace subtree1 in child1 with subtree2, and vice versa
  const newChild1Genes = replaceSubtree(child1Genes, targetId1, subtree2);
  const newChild2Genes = replaceSubtree(child2Genes, targetId2, subtree1);

  // Size enforcement
  const maxPixels = config?.maxGenomeSize || 500;
  const final1 = newChild1Genes.length > maxPixels ? deepClone(parent1.genes) : newChild1Genes;
  const final2 = newChild2Genes.length > maxPixels ? deepClone(parent2.genes) : newChild2Genes;

  const child1: Genome = {
    id: generateId(),
    genes: final1,
    fitness: -Infinity,
    generation: Math.max(parent1.generation, parent2.generation) + 1,
    parents: [parent1.id, parent2.id],
    age: 0
  };

  const child2: Genome = {
    id: generateId(),
    genes: final2,
    fitness: -Infinity,
    generation: Math.max(parent1.generation, parent2.generation) + 1,
    parents: [parent1.id, parent2.id],
    age: 0
  };

  return [child1, child2];
}

/**
 * Deterministic Crowding (DC)
 * ──────────────────────────
 * A niching technique that preserves diversity by having offspring compete 
 * with their most similar parents. 
 * Note: Full DC requires mid-crossover evaluation. For this engine, we use 
 * 'Size-Aware Crowding' which uses structural similarity and size as 
 * proxies for behavioral niches when fitness is not yet known.
 */
export function crowdingCrossover(
  parent1: Genome,
  parent2: Genome,
  config?: PopulationConfig
): [Genome, Genome] {
  const [child1, child2] = subtreeCrossover(parent1, parent2, config);
  
  // In a standard crossover, we just return the children.
  // In Crowding, we only return a child if it is 'better' than its most similar parent.
  // Since we haven't evaluated fitness yet, we use a 'Diversity Guard':
  // If a child is structurally identical to a parent, we force a mutation 
  // to push it into a new niche.
  
  const isIdentical = (g1: Genome, g2: Genome) => 
    g1.genes.length === g2.genes.length && 
    g1.genes.every((g, i) => g.operation === g2.genes[i].operation);

  if (isIdentical(child1, parent1) || isIdentical(child1, parent2)) {
     child1.noveltyScore = (child1.noveltyScore || 0) - 0.1; // Penalty for lack of innovation
  }
  
  return [child1, child2];
}

/**
 * Geometric Semantic Crossover (GSGP)
 * T_off = add(mul(r, T1), mul(1-r, T2))
 * This ensures child semantics are between parent semantics.
 */
export function geometricSemanticCrossover(
  p1: Genome,
  p2: Genome
): [Genome, Genome] {
  const r = 0.1 + random() * 0.8;
  
  const buildChild = (parent1: Genome, parent2: Genome, weight: number): Genome => {
    // We need to merge the gene arrays and add the structural roots
    const genes1 = deepClone(parent1.genes);
    const genes2 = deepClone(parent2.genes);
    
    // Constant r
    const rGene: Gene = {
      id: generateId(),
      type: 'terminal',
      operation: 'const',
      operands: [weight]
    };
    
    // Constant (1-r)
    const invRGene: Gene = {
      id: generateId(),
      type: 'terminal',
      operation: 'const',
      operands: [1 - weight]
    };

    // mul(r, p1)
    const mul1Id = generateId();
    const mul1: Gene = {
      id: mul1Id,
      type: 'function',
      operation: 'mul',
      operands: [rGene.id, genes1[0].id]
    };
    
    // mul(1-r, p2)
    const mul2Id = generateId();
    const mul2: Gene = {
      id: mul2Id,
      type: 'function',
      operation: 'mul',
      operands: [invRGene.id, genes2[0].id]
    };
    
    // add(mul1, mul2)
    const rootId = generateId();
    const root: Gene = {
      id: rootId,
      type: 'function',
      operation: 'add',
      operands: [mul1Id, mul2Id]
    };

    // Combined genes array [root, mul1, mul2, r, invR, ...p1, ...p2]
    const combinedGenes = [
      root,
      mul1,
      mul2,
      rGene,
      invRGene,
      ...genes1,
      ...genes2
    ];

    return {
      id: generateId(),
      genes: combinedGenes,
      fitness: -Infinity,
      generation: Math.max(p1.generation, p2.generation) + 1,
      parents: [p1.id, p2.id],
      age: 0
    };
  };

  return [
    buildChild(p1, p2, r),
    buildChild(p1, p2, 1 - r)
  ];
}



// ============================================================================
// MUTATION OPERATORS
// ============================================================================

/**
 * Create a mutation operator factory
 */
export function createMutationOperator(
  functionSet: FunctionDefinition[],
  terminalSet: TerminalDefinition[]
): MutationOperator {
  return (genome: Genome, config: PopulationConfig): Genome => {
    const mutated = deepClone(genome);

    if (mutated.genes.length === 0) {
      return mutated;
    }

    // Select random gene to mutate
    const geneIdx = randomInt(0, mutated.genes.length - 1);
    const gene = mutated.genes[geneIdx];

    // Apply mutation based on gene type
    if (gene.type === 'terminal') {
      // Terminal mutation: change value or type
      if (randomBool(0.5) && terminalSet.length > 0) {
        const newTerminal = randomChoice(terminalSet);
        gene.operation = newTerminal.name;
        if (newTerminal.type === 'constant') {
          gene.operands = [typeof newTerminal.value === 'function' ? newTerminal.value() : newTerminal.value as number];
        } else {
          gene.operands = [];
        }
      } else if (gene.operands.length > 0) {
        // Mutate constant value
        gene.operands[0] = (gene.operands[0] as number) + (random() - 0.5) * 10;
      }
    } else if (gene.type === 'function') {
      // Function mutation: change function type
      const sameArityFunctions = functionSet.filter(
        f => f.arity === gene.metadata?.arity
      );
      if (sameArityFunctions.length > 1) {
        const newFunc = randomChoice(sameArityFunctions);
        gene.operation = newFunc.name;
      }
    }

    mutated.id = generateId();
    mutated.fitness = -Infinity;

    // Validate after mutation
    try {
      validateGenome(mutated);
    } catch (e) {
      if (e instanceof GenomeValidationError) {
        // Return original if mutation produced invalid genome
        return deepClone(genome);
      }
      throw e;
    }

    return mutated;
  };
}

/**
 * Point mutation - mutate a single gene's internal state (not its structure)
 */
export function pointMutation(
  genome: Genome,
  config: PopulationConfig,
  functionSet: FunctionDefinition[]
): Genome {
  const mutated = deepClone(genome);

  if (mutated.genes.length === 0 || random() >= config.mutationRate) {
    return mutated;
  }

  // Select random gene
  let idx;
  if (config?.quotientMode && mutated.genes.length > 1) {
     idx = randomInt(1, mutated.genes.length - 1); // Avoid root index 0
  } else {
     idx = randomInt(0, Math.max(0, mutated.genes.length - 1));
  }
  const gene = mutated.genes[idx];
  if (!gene) return mutated;

  if (gene.type === 'function') {
      // Phase 74: Stabilize structure by allowing operation swaps dynamically
      const sameArityFunctions = functionSet.filter(
        f => f.arity === (gene.metadata?.arity ?? 2)
      );
      if (sameArityFunctions.length > 0) {
        gene.operation = randomChoice(sameArityFunctions).name;
      }
  } else if (gene.type === 'terminal' && gene.operation === 'const') {
      // Mutate ONLY non-ID operand values (constants)
      gene.operands = gene.operands.map(op => {
        if (typeof op === 'number') {
          return op + (random() - 0.5) * 2;
        }
        return op; // Preserve string IDs
      });
  }

  mutated.id = generateId();
  mutated.fitness = -Infinity;

  return mutated;
}

/**
 * Subtree mutation - replaces a random subtree with a newly generated one
 */
export function subtreeMutation(
  genome: Genome,
  config: PopulationConfig,
  functionSet: FunctionDefinition[],
  terminalSet: TerminalDefinition[]
): Genome {
  const mutated = deepClone(genome);

  if (mutated.genes.length === 0 || random() >= config.mutationRate) {
    return mutated;
  }

  // Select random point to replace
  let idx;
  if (config?.quotientMode && mutated.genes.length > 1) {
    // Avoid root (index 0) to preserve quotient contract
    idx = randomInt(1, mutated.genes.length - 1);
  } else {
    idx = randomInt(0, Math.max(0, mutated.genes.length - 1));
  }
  const targetId = mutated.genes[idx]?.id;
  if (!targetId) return mutated;

  // Create a factory to generate the new subtree
  const factory = new GenomeFactory(functionSet, terminalSet, config);
  const newSubtreeGenes = factory.createRandomTree(config.maxGenomeDepth / 2);

  const newGenes = replaceSubtree(mutated.genes, targetId, newSubtreeGenes);
  
  // Size enforcement
  const maxPixels = config.maxGenomeSize || 500;
  if (newGenes.length > maxPixels) {
     // If mutation makes it too big, try shrink mutation instead to keep it under control
     return shrinkMutation(genome, terminalSet);
  }

  mutated.genes = newGenes;
  mutated.id = generateId();
  mutated.fitness = -Infinity;

  return mutated;
}

/**
 * Hoist mutation - create child from a random subtree of the parent
 */
export function hoistMutation(genome: Genome, config?: PopulationConfig): Genome {
  if (genome.genes.length < 2) {
    return deepClone(genome);
  }

  // Select random subtree to hoist (avoid root index 0 if quotientMode is active)
  let idx;
  if (config?.quotientMode) {
     idx = randomInt(1, genome.genes.length - 1);
  } else {
     idx = randomInt(1, genome.genes.length - 1); // Avoid root index 0 anyway
  }
  const rootId = genome.genes[idx].id;

  const hoistedGenes = collectSubtree(genome, rootId);

  return {
    id: generateId(),
    genes: rebindGeneIds(collectSubtree(genome, rootId)),
    fitness: -Infinity,
    generation: (genome.generation || 0) + 1,
    age: 0
  };
}

/**
 * Shrink mutation - replaces a function branch with a terminal
 */
export function shrinkMutation(
  genome: Genome,
  terminalSet: TerminalDefinition[],
  config?: PopulationConfig
): Genome {
  if (genome.genes.length < 2) {
    return deepClone(genome);
  }

  const mutated = deepClone(genome);
  // Find all function genes
  const functionIndices = mutated.genes
    .map((g, i) => g.type === 'function' ? i : -1)
    .filter(i => {
        if (config?.quotientMode) {
           return i > 0; // Avoid root index 0 in quotient mode
        }
        return i !== -1;
    });

  if (functionIndices.length === 0) return mutated;

  const idx = randomChoice(functionIndices);
  const targetId = mutated.genes[idx].id;

  // Create a random terminal replacement
  const terminal = randomChoice(terminalSet);
  const newTerminalGene: Gene = {
    id: generateId(),
    type: 'terminal',
    operation: terminal?.name || 'const',
    operands: terminal?.type === 'constant' ? [typeof terminal.value === 'function' ? terminal.value() : terminal.value as number] : []
  };

  mutated.genes = replaceSubtree(mutated.genes, targetId, [newTerminalGene]);
  mutated.id = generateId();
  mutated.fitness = -Infinity;

  return mutated;
}

// ============================================================================
// COMBINED OPERATORS
// ============================================================================

export interface OperatorConfig {
  selectionMethod: 'tournament' | 'roulette' | 'rank' | 'sus' | 'lexicase' | 'epsilon-lexicase';
  crossoverMethod: 'subtree' | 'onePoint' | 'uniform' | 'crowding' | 'geometric-semantic';
  mutationMethods: ('point' | 'subtree' | 'hoist' | 'shrink')[];
  functionSet: FunctionDefinition[];
  terminalSet: TerminalDefinition[];
}

/**
 * Get selection operator by name
 */
export function getSelectionOperator(method: string): SelectionOperator {
  switch (method) {
    case 'tournament':
      return tournamentSelection;
    case 'roulette':
      return rouletteSelection;
    case 'rank':
      return rankSelection;
    case 'sus':
      return stochasticUniversalSampling;
    case 'lexicase':
    case 'epsilon-lexicase':
      return epsilonLexicaseSelection;
    default:
      return tournamentSelection;
  }
}

/**
 * Get crossover operator by name
 */
export function getCrossoverOperator(method: string): CrossoverOperator {
  switch (method) {
    case 'subtree':
      return subtreeCrossover;
    case 'crowding':
      return crowdingCrossover;
    case 'geometric-semantic':
      return geometricSemanticCrossover;
    default:
      return subtreeCrossover;
  }
}

/**
 * Apply mutation with random method selection
 */
export function applyMutation(
  genome: Genome,
  config: PopulationConfig,
  functionSet: FunctionDefinition[],
  terminalSet: TerminalDefinition[]
): Genome {
  const mutationTypes = ['point', 'subtree', 'hoist', 'shrink'];
  const type = randomChoice(mutationTypes);

  switch (type) {
    case 'point':
      return pointMutation(genome, config, functionSet);
    case 'subtree':
      return subtreeMutation(genome, config, functionSet, terminalSet);
    case 'hoist':
      return hoistMutation(genome, config);
    case 'shrink':
      return shrinkMutation(genome, terminalSet, config);
    default:
      return pointMutation(genome, config, functionSet);
  }
}
