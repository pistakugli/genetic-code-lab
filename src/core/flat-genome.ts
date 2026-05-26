/**
 * Flat Genome Representation
 * ─────────────────────────
 * To enable Beyond SOTA scaling (100k+ individuals), we avoid JSON serialization.
 * Genomes are packed into a SharedArrayBuffer for zero-copy worker access.
 *
 * Format per node (4-byte aligned):
 * [0]: Operation Type (mapped to index in function/terminal set)
 * [1]: Arity
 * [2]: First Child Pointer (absolute offset in buffer)
 * [3]: Second Child Pointer (if any)
 */

import { Genome, Gene, FunctionDefinition, TerminalDefinition } from './types';

export class FlatGenomeManager {
  public static readonly NODE_SIZE = 5; // [op, arity, c1, c2, c3]
  private buffer: SharedArrayBuffer;
  private view: Int32Array;
  private maxNodes: number;

  constructor(maxPopulation: number, maxNodesPerGenome: number) {
    this.maxNodes = maxPopulation * maxNodesPerGenome;
    this.buffer = new SharedArrayBuffer(this.maxNodes * FlatGenomeManager.NODE_SIZE * 4);
    this.view = new Int32Array(this.buffer);
    this.view.fill(-1); // Initialize with null pointers
  }

  getSharedBuffer(): SharedArrayBuffer {
    return this.buffer;
  }

  /**
   * Pack a standard Genome into the flat buffer at a specific offset.
   * Enforces strict pre-order traversal with pre-allocated node indexes 
   * to ensure hierarchical pointers point to correct contiguous memory.
   */
  pack(genome: Genome, genomeIdx: number, maxNodesPerGenome: number, 
       funcMap: Map<string, number>, termMap: Map<string, number>, funcSetLen: number): number {
    const offset = genomeIdx * maxNodesPerGenome * FlatGenomeManager.NODE_SIZE;
    let nextNodeIdx = 0; // Tracks the next available contiguous node slot

    // Map for ID-based resolution (handles flat array genome representation)
    const idToGene = new Map<string, Gene>();
    if (genome.genes.length > 1) {
      for (const g of genome.genes) idToGene.set(g.id, g);
    }

    const packNode = (gene: Gene): number => {
      if (nextNodeIdx >= maxNodesPerGenome) return -1;

      // 1. Claim the current node slot FIRST
      const currentIdx = nextNodeIdx;
      const nodeStart = offset + (currentIdx * FlatGenomeManager.NODE_SIZE);
      nextNodeIdx++;

      // 2. Resolve operation ID
      let opIdx = funcMap.get(gene.operation);
      if (opIdx === undefined) {
        const tIdx = termMap.get(gene.operation);
        opIdx = tIdx !== undefined ? tIdx + funcSetLen : 0;
      }

      this.view[nodeStart + 0] = opIdx;
      
      // 3. Handle Constants (Beyond SOTA Math support)
      if (gene.operation === 'const' && gene.operands.length > 0) {
        const value = gene.operands[0] as number;
        // Use Float32 view to write bit-perfect float into the child1 slot
        const fView = new Float32Array(this.view.buffer, nodeStart * 4 + 2 * 4, 1);
        fView[0] = value;
        this.view[nodeStart + 1] = 0; // Arity 0
        return nodeStart;
      }

      // 4. Process children recursively (Functions or variables)
      const operands = gene.operands;
      let childCount = 0;
      for (let i = 0; i < operands.length; i++) {
        const op = operands[i];
        let childGene: Gene | null = null;

        if (typeof op === 'object' && op !== null) {
          childGene = op as Gene;
        } else if (typeof op === 'string') {
          childGene = idToGene.get(op) || null;
        }

        if (childGene) {
          // Recursive call will claim the NEXT available slot, ensuring contiguous tree logic
          const childPtr = packNode(childGene);
          if (childCount < 3) {
            this.view[nodeStart + 2 + childCount] = childPtr;
          }
          childCount++;
        }
      }
      this.view[nodeStart + 1] = childCount;

      return nodeStart; // Return absolute pointer
    };

    if (genome.genes.length > 0) {
      packNode(genome.genes[0]);
    }
    return offset;
  }

  /**
   * Static helper for workers to interpret the buffer.
   */
  static getNode(view: Int32Array, ptr: number) {
    return {
      opIdx: view[ptr + 0],
      arity: view[ptr + 1],
      child1: view[ptr + 2],
      child2: view[ptr + 3],
      child3: view[ptr + 4]
    };
  }
}
