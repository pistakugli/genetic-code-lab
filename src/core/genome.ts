/**
 * Genetic Code Lab - Genome
 * Represents an individual in the population
 */

import {
  Gene,
  Genome,
  GeneType,
  FunctionDefinition,
  TerminalDefinition,
  PopulationConfig,
  GenomeValidationError as GenomeValidationErrorClass
} from './types';

// Re-export for convenience
export { GenomeValidationErrorClass as GenomeValidationError };
import {
  generateId,
  random,
  randomChoice,
  randomInt,
  randomBool,
  deepClone
} from '../utils/helpers';

// ============================================================================
// GENOME BUILDER
// ============================================================================

export class GenomeBuilder {
  private genes: Gene[] = [];
  private generation: number = 0;
  private parents?: [string, string];

  /**
   * Add a gene to the genome
   */
  addGene(gene: Gene): this {
    this.genes.push(gene);
    return this;
  }

  /**
   * Set the generation
   */
  setGeneration(gen: number): this {
    this.generation = gen;
    return this;
  }

  /**
   * Set parent IDs
   */
  setParents(parent1: string, parent2: string): this {
    this.parents = [parent1, parent2];
    return this;
  }

  /**
   * Build the genome
   */
  build(): Genome {
    return {
      id: generateId(),
      genes: [...this.genes],
      fitness: -Infinity,
      generation: this.generation,
      parents: this.parents,
      age: 0
    };
  }
}

// ============================================================================
// GENOME FACTORY
// ============================================================================

export class GenomeFactory {
  private functionSet: FunctionDefinition[];
  private terminalSet: TerminalDefinition[];
  private config: PopulationConfig;

  constructor(
    functionSet: FunctionDefinition[],
    terminalSet: TerminalDefinition[],
    config: PopulationConfig
  ) {
    this.functionSet = functionSet;
    this.terminalSet = terminalSet;
    this.config = config;
  }

  /**
   * Create a random genome using ramped half-and-half method
   */
  createRandom(depth?: number): Genome {
    const maxDepth = depth ?? randomInt(2, this.config.maxGenomeDepth);
    const method = random() < 0.5 ? 'grow' : 'full';
    
    let genes;
    if (this.config.quotientMode) {
      genes = this.createQuotientTree(maxDepth, method);
    } else {
      genes = this.createTree(maxDepth, method);
    }

    return {
      id: generateId(),
      genes,
      fitness: -Infinity,
      generation: 0,
      age: 0
    };
  }

  /**
   * Create a population of random genomes
   */
  createPopulation(size: number): Genome[] {
    const genomes: Genome[] = [];

    // Use ramped half-and-half for initial population
    for (let i = 0; i < size; i++) {
      const depth = 2 + Math.floor((i / size) * (this.config.maxGenomeDepth - 2));
      const method = i % 2 === 0 ? 'grow' : 'full';
      
      let genes;
      if (this.config.quotientMode) {
        genes = this.createQuotientTree(depth, method);
      } else {
        genes = this.createTree(depth, method);
      }

      genomes.push({
        id: generateId(),
        genes,
        fitness: -Infinity,
        generation: 0,
        age: 0
      });
    }

    return genomes;
  }

  /**
   * Create a quotient tree (Numerator / Denominator)
   */
  private createQuotientTree(depth: number, method: 'grow' | 'full'): Gene[] {
    // 1. Create Numerator
    const numSubtree = this.createTree(depth - 1, method);
    
    // 2. Create Denominator
    const denSubtree = this.createTree(depth - 1, method);
    
    // 3. Create Root 'div'
    const rootId = generateId();
    const rootGene: Gene = {
      id: rootId,
      type: 'function',
      operation: 'div',
      operands: [numSubtree[0].id, denSubtree[0].id],
      metadata: { arity: 2, returnType: 'number' }
    };

    return [rootGene, ...numSubtree, ...denSubtree];
  }

  /**
   * Create a random tree of genes (useful for mutation)
   */
  public createRandomTree(maxDepth?: number): Gene[] {
    const depth = maxDepth !== undefined ? maxDepth : randomInt(1, this.config.maxGenomeDepth / 2);
    const method = randomBool(0.5) ? 'grow' : 'full';
    return this.createTree(depth, method);
  }

  /**
   * Create a tree of genes
   */
  private createTree(depth: number, method: 'grow' | 'full'): Gene[] {
    if (depth <= 0) {
      // Terminal node
      return [this.createTerminalGene()];
    }

    if (method === 'full' || random() < 0.5) {
      // Function node
      return this.createFunctionGene(depth, method);
    }

    // Grow method: terminal choice
    return [this.createTerminalGene()];
  }

  /**
   * Create a terminal gene
   */
  private createTerminalGene(): Gene {
    const terminal = randomChoice(this.terminalSet);
    
    // Evaluate value if it's a generator function (common for random constants)
    const value = typeof terminal.value === 'function' ? terminal.value() : terminal.value;

    return {
      id: generateId(),
      type: 'terminal',
      operation: terminal.name,
      operands: terminal.type === 'constant' ? [value as number] : [],
      metadata: { terminalType: terminal.type }
    };
  }

  /**
   * Create a function gene with its subtree
   */
  private createFunctionGene(depth: number, method: 'grow' | 'full'): Gene[] {
    const func = randomChoice(this.functionSet);
    const operands: (number | string | Gene)[] = [];
    const genes: Gene[] = [];

    const mainGene: Gene = {
      id: generateId(),
      type: 'function',
      operation: func.name,
      operands: [],
      metadata: { arity: func.arity, returnType: func.returnType }
    };

    genes.push(mainGene);

    // Create child genes for each argument
    for (let i = 0; i < func.arity; i++) {
      const childGenes = this.createTree(depth - 1, method);
      if (childGenes.length > 0) {
        operands.push(childGenes[0].id);
        genes.push(...childGenes);
      }
    }

    mainGene.operands = operands;
    return genes;
  }
}

// ============================================================================
// GENOME UTILITIES
// ============================================================================

/**
 * Get the depth of a genome tree
 */
export function getGenomeDepth(genome: Genome): number {
  if (genome.genes.length === 0) return 0;

  const geneMap = new Map<string, Gene>();
  genome.genes.forEach(g => geneMap.set(g.id, g));

  function getDepth(gene: Gene): number {
    if (gene.type === 'terminal') return 1;

    let maxChildDepth = 0;
    for (const op of gene.operands) {
      if (typeof op === 'string') {
        const childGene = geneMap.get(op);
        if (childGene) {
          maxChildDepth = Math.max(maxChildDepth, getDepth(childGene));
        }
      }
    }
    return 1 + maxChildDepth;
  }

  return getDepth(genome.genes[0]);
}

/**
 * Collect a gene and all its descendants by following operand ID references
 */
export function collectSubtree(genome: Genome, geneId: string): Gene[] {
  const geneMap = new Map<string, Gene>();
  genome.genes.forEach(g => geneMap.set(g.id, g));

  const collected: Gene[] = [];
  const visited = new Set<string>();

  function collect(id: string): void {
    if (visited.has(id)) return;
    visited.add(id);

    const gene = geneMap.get(id);
    if (!gene) return;

    collected.push(gene);

    for (const op of gene.operands) {
      if (typeof op === 'string') {
        collect(op);
      }
    }
  }

  collect(geneId);
  return collected;
}

/**
 * Replace a subtree in a genome: remove old subtree rooted at targetId,
 * insert new subtree genes, and update the parent's operand reference.
 */
export function replaceSubtree(
  genes: Gene[],
  targetId: string,
  newSubtreeGenes: Gene[]
): Gene[] {
  // Build gene map
  const geneMap = new Map<string, Gene>();
  genes.forEach(g => geneMap.set(g.id, g));

  // Collect all IDs in the old subtree to remove
  const oldIds = new Set<string>();
  const visited = new Set<string>();

  function collectIds(id: string): void {
    if (visited.has(id)) return;
    visited.add(id);
    const gene = geneMap.get(id);
    if (!gene) return;
    oldIds.add(id);
    for (const op of gene.operands) {
      if (typeof op === 'string') collectIds(op);
    }
  }
  collectIds(targetId);

  // Remove old subtree genes
  const remaining = genes.filter(g => !oldIds.has(g.id));

  // Update any parent that referenced targetId to point to new root
  const newRootId = newSubtreeGenes.length > 0 ? newSubtreeGenes[0].id : undefined;
  let isRootReplacement = false;

  if (genes.length > 0 && genes[0].id === targetId) {
    isRootReplacement = true;
  } else if (newRootId) {
    for (const gene of remaining) {
      gene.operands = gene.operands.map(op =>
        op === targetId ? newRootId : op
      );
    }
  }

  // Add new subtree genes.
  // CRITICAL FIX: If we are replacing the root node (genes[0].id === targetId), 
  // the new subtree's root MUST be at index 0 of the returned array so that 
  // FlatGenomeManager.pack() correctly identifies the true root of the genome.
  if (isRootReplacement) {
    return [...newSubtreeGenes, ...remaining];
  } else {
    // If not root, standard append is fine, parents point to it by ID.
    // However, to be extra safe and maintain deterministic order, we keep remaining first.
    return [...remaining, ...newSubtreeGenes];
  }
}

/**
 * Get the size (number of nodes) of a genome
 */
export function getGenomeSize(genome: Genome): number {
  return genome.genes.length;
}

/**
 * Validate a genome
 */
export function validateGenome(genome: Genome): boolean {
  // Check for empty genome
  if (genome.genes.length === 0) {
    throw new GenomeValidationErrorClass('Genome has no genes');
  }

  // Check for duplicate gene IDs
  const ids = new Set<string>();
  for (const gene of genome.genes) {
    if (ids.has(gene.id)) {
      throw new GenomeValidationErrorClass(`Duplicate gene ID: ${gene.id}`);
    }
    ids.add(gene.id);
  }

  // Check for valid gene types
  for (const gene of genome.genes) {
    if (!['function', 'terminal', 'conditional', 'loop', 'custom'].includes(gene.type)) {
      throw new GenomeValidationErrorClass(`Invalid gene type: ${gene.type}`);
    }
  }

  return true;
}

/**
 * Clone a genome with a new ID
 */
export function cloneGenome(genome: Genome): Genome {
  return {
    ...deepClone(genome),
    id: generateId(),
    age: genome.age + 1
  };
}

/**
 * Hygienically rebinds all IDs in a sub-array of genes.
 * Used during crossover to prevent duplicate ID collisions when merging trees.
 */
export function rebindGeneIds(genes: Gene[]): Gene[] {
  if (!genes || genes.length === 0) return [];

  const cloned = deepClone(genes);
  const idMap = new Map<string, string>();

  // 1. Generate new IDs for all nodes
  for (const gene of cloned) {
    const newId = generateId();
    idMap.set(gene.id, newId);
    gene.id = newId;
  }

  // 2. Update intra-tree references (operands)
  for (const gene of cloned) {
    gene.operands = gene.operands.map(op => {
      // If the operand is a string reference to another node in the subtree, rewrite it
      if (typeof op === 'string' && idMap.has(op)) {
        return idMap.get(op)!;
      }
      return op; // Constants and external references stay the same
    });
  }

  return cloned;
}

/**
 * Get a specific gene by ID
 */
export function getGeneById(genome: Genome, id: string): Gene | undefined {
  return genome.genes.find(g => g.id === id);
}

/**
 * Get all terminal genes
 */
export function getTerminalGenes(genome: Genome): Gene[] {
  return genome.genes.filter(g => g.type === 'terminal');
}

/**
 * Get all function genes
 */
export function getFunctionGenes(genome: Genome): Gene[] {
  return genome.genes.filter(g => g.type === 'function');
}

/**
 * Convert genome to string representation
 */
export function genomeToString(genome: Genome): string {
  if (genome.genes.length === 0) return 'EMPTY';

  const geneMap = new Map<string, Gene>();
  genome.genes.forEach(g => geneMap.set(g.id, g));

  function geneToString(gene: Gene): string {
    if (gene.type === 'terminal') {
      return gene.operands.length > 0
        ? `${gene.operation}(${gene.operands[0]})`
        : gene.operation;
    }

    const childStrs = gene.operands.map(op => {
      if (typeof op === 'string') {
        const child = geneMap.get(op);
        return child ? geneToString(child) : '?';
      }
      return String(op);
    });

    return `(${gene.operation} ${childStrs.join(' ')})`;
  }

  return geneToString(genome.genes[0]);
}

/**
 * Get genome statistics
 */
export function getGenomeStats(genome: Genome): {
  size: number;
  depth: number;
  terminals: number;
  functions: number;
  fitness: number;
} {
  return {
    size: getGenomeSize(genome),
    depth: getGenomeDepth(genome),
    terminals: getTerminalGenes(genome).length,
    functions: getFunctionGenes(genome).length,
    fitness: genome.fitness
  };
}
