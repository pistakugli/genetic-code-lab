import { Genome, Population } from './types';
import { random } from '../utils/helpers';

/**
 * Novelty Search Manager
 * ──────────────────────
 * In Frontier GP, we don't just want high fitness; we want HIGH NOVELTY.
 * This manager computes the 'behavioral' distance between individuals
 * to identify unique strategies that might otherwise be discarded.
 */
export class NoveltyManager {
  /**
   * Update novelty scores for an entire population.
   * Uses random sampling for efficiency on large populations (Beyond SOTA Scaling).
   */
  public static updateNoveltyScores(population: Population, k: number = 5): void {
    const genomes = population.genomes.filter(g => g.caseErrors || g.visitationMap);
    if (genomes.length < k) return;

    // SCALING LOGIC: Use sparser sampling for massive populations (Beyond SOTA)
    const sampleSize = genomes.length > 5000 ? 50 : Math.min(genomes.length, 250);
    const sampleNodes = Array.from({ length: sampleSize }, () => 
      genomes[Math.floor(random() * genomes.length)]
    );

    for (const genome of genomes) {
      // Fast path: if k=1 or sample is small, avoid heavy sorting if possible
      const distances = sampleNodes
        .filter(other => other.id !== genome.id)
        .map(other => this.calculateBehavioralDistance(genome, other));
      
      distances.sort((a, b) => a - b);
      
      const knn = distances.slice(0, k);
      const avgKnn = knn.reduce((a, b) => a + b, 0) / knn.length;
      genome.noveltyScore = avgKnn;
    }
  }

  /**
   * Behavioral distance calculation.
   * Prefers bitset-based Hamming distance (Visitation Map) for higher fidelity.
   */
  private static calculateBehavioralDistance(a: Genome, b: Genome): number {
    if (a.visitationMap && b.visitationMap) {
      return this.calculateBitsetDistance(a.visitationMap, b.visitationMap);
    }

    if (!a.caseErrors || !b.caseErrors) return 1.0;
    
    let sum = 0;
    const len = Math.min(a.caseErrors.length, b.caseErrors.length);
    for (let i = 0; i < len; i++) {
      const diff = a.caseErrors[i] - b.caseErrors[i];
      sum += diff * diff;
    }
    return Math.sqrt(sum);
  }

  /**
   * Hamming distance between visitation bitsets.
   */
  public static calculateBitsetDistance(a: Uint8Array, b: Uint8Array): number {
    let diff = 0;
    const len = Math.min(a.length, b.length);
    for (let i = 0; i < len; i++) {
      if (a[i] !== b[i]) diff++;
    }
    return diff / len;
  }
}
