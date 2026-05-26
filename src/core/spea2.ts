/**
 * SPEA2 (Strength Pareto Evolutionary Algorithm 2) Implementation
 * High-performance multi-objective selection for Genetic Programming.
 */

import { Genome, MultiObjectiveFitness } from './types';

/**
 * Check if a dominates b (multi-objective)
 * Minimal values are better (e.g. error, size)
 */
export function dominates(a: Genome, b: Genome): boolean {
  if (!a.multiObjectiveFitness || !b.multiObjectiveFitness) return false;
  
  let betterInOne = false;
  let worseInOne = false;

  for (let i = 0; i < a.multiObjectiveFitness.values.length; i++) {
    if (a.multiObjectiveFitness.values[i] < b.multiObjectiveFitness.values[i]) {
      betterInOne = true;
    } else if (a.multiObjectiveFitness.values[i] > b.multiObjectiveFitness.values[i]) {
      worseInOne = true;
    }
  }

  return betterInOne && !worseInOne;
}

/**
 * Calculate SPEA2 Fitness for a combined population (Population + Archive)
 */
export function calculateSPEA2Fitness(individuals: Genome[], k: number = 1): void {
  const size = individuals.length;
  
  // 1. Calculate Strength S(i)
  // S(i) = number of individuals that i dominates
  const strengths = new Array(size).fill(0);
  for (let i = 0; i < size; i++) {
    for (let j = 0; j < size; j++) {
      if (i === j) continue;
      const fitI = individuals[i];
      const fitJ = individuals[j];
      if (fitI.multiObjectiveFitness && fitJ.multiObjectiveFitness) {
        if (dominates(fitI, fitJ)) {
          strengths[i]++;
        }
      }
    }
  }

  // 2. Calculate Raw Fitness R(i)
  // R(i) = sum of strengths of individuals that dominate i
  for (let i = 0; i < individuals.length; i++) {
    const ind = individuals[i];
    if (ind.multiObjectiveFitness) {
      ind.multiObjectiveFitness.rawFitness = 0;
      for (let j = 0; j < individuals.length; j++) {
        if (i === j) continue;
        const other = individuals[j];
        if (dominates(other, ind)) {
          ind.multiObjectiveFitness.rawFitness++;
        }
      }
    }
  }

  // 3. Calculate Density D(i)
  // D(i) = 1 / (distance to k-th nearest neighbor + 2)
  for (let i = 0; i < size; i++) {
    const distances: number[] = [];
    for (let j = 0; j < size; j++) {
      if (i === j) continue;
      distances.push(calculateEuclideanDistance(individuals[i], individuals[j]));
    }
    distances.sort((a, b) => a - b);
    
    // k is usually sqrt(N)
    const kthDistance = distances[Math.min(k - 1, distances.length - 1)] || 0;
    const density = 1 / (kthDistance + 2);
    
    if (individuals[i].multiObjectiveFitness) {
      individuals[i].multiObjectiveFitness!.density = density;
      
      // Final SPEA2 Score = R(i) + D(i)
      const score = individuals[i].multiObjectiveFitness!.rawFitness! + density;
      individuals[i].multiObjectiveFitness!.score = score;
      
      // Invert it so higher fitness is better (required by tournament/rank selection)
      individuals[i].fitness = 1 / (1 + score);
    }
  }
}

/**
 * Calculate Euclidean distance between two individuals in objective space
 */
function calculateEuclideanDistance(a: Genome, b: Genome): number {
  if (!a.multiObjectiveFitness || !b.multiObjectiveFitness) return Infinity;
  
  let sum = 0;
  for (let i = 0; i < a.multiObjectiveFitness.values.length; i++) {
    const diff = a.multiObjectiveFitness.values[i] - b.multiObjectiveFitness.values[i];
    sum += diff * diff;
  }
  return Math.sqrt(sum);
}

/**
 * Environmental Selection (Archive Update)
 */
export function environmentalSelection(
  combined: Genome[],
  archiveSize: number
): Genome[] {
  // 1. Individuals with R(i) === 0 are non-dominated
  let nextArchive = combined.filter(ind => ind.multiObjectiveFitness?.rawFitness === 0);

  if (nextArchive.length < archiveSize) {
    // 2. Not enough: Add individuals with best score (lowest score)
    const dominated = combined.filter(ind => ind.multiObjectiveFitness!.rawFitness! > 0)
      .sort((a, b) => a.multiObjectiveFitness!.score! - b.multiObjectiveFitness!.score!);
    
    nextArchive = nextArchive.concat(dominated.slice(0, archiveSize - nextArchive.length));
  } else if (nextArchive.length > archiveSize) {
    // 3. Too many: Remove based on density (Euclidean distance to k-th neighbor)
    // Frontier Optimization: Don't recalculate entire density map in a loop (O(T * N^2))
    // Instead, calculate once and remove worst performers, or use incremental updates.
    
    // For very large populations, we use a single-pass truncation to maintain throughput
    const k = Math.floor(Math.sqrt(nextArchive.length));
    const densityScores = nextArchive.map(ind => {
      const distances = nextArchive
        .filter(other => other.id !== ind.id)
        .map(other => calculateEuclideanDistance(ind, other))
        .sort((a, b) => a - b);
      return (distances[k - 1] || 0);
    });

    // Pair individuals with their k-th neighbor distance
    const paired = nextArchive.map((ind, i) => ({ ind, dist: densityScores[i] }));
    // Sort by distance (largest first to keep most isolated individuals)
    paired.sort((a, b) => b.dist - a.dist);
    
    nextArchive = paired.slice(0, archiveSize).map(p => p.ind);
  }

  return nextArchive;
}

// ============================================================================
// BUFFER-LEVEL SPEA2 (Phase 42 — DOGP Hot Path)
// ============================================================================

import { PopulationBuffer } from './pop-buffer';
/**
 * Returns true if individual `a` Pareto-dominates individual `b`.
 * Objectives: [error, complexity] — both minimized.
 * 
 * Phase 43: Adaptive Lexicographic Pareto Dominance.
 * LEX_EPSILON and SIZE_TOLERANCE are computed per-generation from the
 * population's actual fitness range, set by evaluateSPEA2Buffer before
 * dominance comparisons begin.
 */

// Adaptive hyperparameters — set each generation by evaluateSPEA2Buffer
let _lexEpsilon = 0.001;
let _sizeTolerance = 10;

/** Called once per generation to calibrate dominance thresholds */
export function calibrateDominance(fitnessArray: Float64Array, lengthsArray: Uint32Array, size: number): void {
    // Compute fitness range from a sample
    let minFit = Infinity, maxFit = -Infinity;
    let minLen = Infinity, maxLen = -Infinity;
    const sampleN = Math.min(size, 500);
    const step = Math.max(1, Math.floor(size / sampleN));
    for (let i = 0; i < size; i += step) {
        const f = fitnessArray[i];
        if (f > -Infinity && f < Infinity) {
            if (f < minFit) minFit = f;
            if (f > maxFit) maxFit = f;
        }
        const l = lengthsArray[i];
        if (l < minLen) minLen = l;
        if (l > maxLen) maxLen = l;
    }
    const fitRange = maxFit - minFit;
    const lenRange = maxLen - minLen;
    
    // Adaptive LEX_EPSILON: 0.1% of the population's actual fitness range
    // Phase 43: Tightened from 1% to 0.1% to preserve signal in symbolic regression.
    // Floor at 1e-7 to avoid floating point noise.
    _lexEpsilon = Math.max(1e-7, Math.min(0.005, fitRange * 0.001));
    
    // Adaptive SIZE_TOLERANCE: 5% of the population's size range, minimum 3
    _sizeTolerance = Math.max(3, Math.floor(lenRange * 0.05));
}

function bufferDominates(
    fitnessA: number, lenA: number,
    fitnessB: number, lenB: number
): boolean {
    const errA = 1.0 - fitnessA;
    const errB = 1.0 - fitnessB;
    
    // Phase 67: Frontier Discovery Dominance
    // We strictly penalize 'trivial collapse' (size < 5) if fitness is high.
    // This forces the engine to explore deeper trees before settling.
    const isTrivialA = lenA < 5 && fitnessA > 0.8;
    const isTrivialB = lenB < 5 && fitnessB > 0.8;

    if (isTrivialB && !isTrivialA) return true; // Non-trivial dominates trivial
    if (isTrivialA && !isTrivialB) return false;

    // Standard Lexicographic Pareto: Accuracy is King.
    if (errA < errB - _lexEpsilon) return true;
    
    // Accuracy is effectively identical
    if (errA <= errB) {
        // Phase 67: If we are in the 'Discovery Zone' (fitness < 0.99)
        // We actually PREFER slightly larger structures to allow growth
        if (fitnessA < 0.95) {
            if (lenA > lenB + 2) return true; 
        } else {
            // High fitness zone: Switch back to Parsimony (Size reduction)
            if (lenA < lenB - _sizeTolerance) return true;
        }
    }
    
    return false;
}

/**
 * Evaluate the entire population with SPEA2 and write composite scores
 * into `buffer.spea2Score`.  Lower score = better individual.
 *
 * Called once per generation after BlockEvaluator.evaluateBlock.
 *
 * @param buffer   The population buffer (already fitness-evaluated).
 * @param size     Number of active individuals.
 */
export function evaluateSPEA2Buffer(buffer: PopulationBuffer, size: number): void {
    if (size <= 1) {
        buffer.spea2Score[0] = 0;
        return;
    }

    // Phase 43: Calibrate adaptive dominance thresholds from current population state
    calibrateDominance(buffer.fitness, buffer.lengths, size);

    const useSampling = size > 2000;
    const sampleSize = useSampling ? 200 : size;
    const stride = useSampling ? Math.floor(size / sampleSize) : 1;
    const kNeighbor = Math.max(1, Math.floor(Math.sqrt(sampleSize)));

    // 1. Strength S(i) = |{ j | i dominates j }|  (calculated against sample)
    const strength = new Float64Array(size);
    for (let i = 0; i < size; i++) {
        let domCount = 0;
        for (let s = 0; s < sampleSize; s++) {
            const j = s * stride;
            if (i === j) continue;
            if (bufferDominates(buffer.fitness[i], buffer.lengths[i],
                                buffer.fitness[j], buffer.lengths[j])) {
                domCount++;
            }
        }
        strength[i] = domCount;
    }

    // 2. Raw fitness R(i) = Σ S(j) for all sample j that dominate i
    const rawFitness = new Float64Array(size);
    for (let i = 0; i < size; i++) {
        let rFit = 0;
        for (let s = 0; s < sampleSize; s++) {
            const j = s * stride;
            if (i === j) continue;
            if (bufferDominates(buffer.fitness[j], buffer.lengths[j],
                                buffer.fitness[i], buffer.lengths[i])) {
                rFit += strength[j];
            }
        }
        // Normalize raw fitness so it scales correctly regardless of sampling
        rawFitness[i] = useSampling ? rFit * (size / sampleSize) : rFit;
    }

    // 3. Density D(i) = 1 / (σ_k + 2)
    let minErr = Infinity, maxErr = -Infinity;
    let minLen = Infinity, maxLen = -Infinity;
    // Find min/max purely from the sample to be O(S) instead of O(N)
    for (let s = 0; s < sampleSize; s++) {
        const idx = s * stride;
        const err = 1.0 - buffer.fitness[idx];
        const len = buffer.lengths[idx];
        if (err < minErr) minErr = err;
        if (err > maxErr) maxErr = err;
        if (len < minLen) minLen = len;
        if (len > maxLen) maxLen = len;
    }
    const errRange = maxErr - minErr || 1;
    const lenRange = maxLen - minLen || 1;

    const density = new Float64Array(size);
    const ACCURACY_BIAS = 5.0; // Favor accuracy-based isolation

    const sampleDists = new Float64Array(sampleSize);
    for (let i = 0; i < size; i++) {
        const errI = (1.0 - buffer.fitness[i] - minErr) / errRange;
        const lenI = (buffer.lengths[i] - minLen) / lenRange;
        for (let s = 0; s < sampleSize; s++) {
            const j = s * stride;
            if (j === i) { sampleDists[s] = Infinity; continue; }
            const de = (errI - (1.0 - buffer.fitness[j] - minErr) / errRange) * ACCURACY_BIAS;
            const dl = lenI - (buffer.lengths[j] - minLen) / lenRange;
            sampleDists[s] = Math.sqrt(de * de + dl * dl);
        }
        // Quickselect/Sort for kth shortest
        const sorted = Array.from(sampleDists).sort((a, b) => a - b);
        density[i] = 1.0 / (sorted[Math.min(kNeighbor, sorted.length - 1)] + 2.0);
    }

    // 4. Composite SPEA2 score F(i) = R(i) + D(i)
    for (let i = 0; i < size; i++) {
        buffer.spea2Score[i] = rawFitness[i] + density[i];
    }
}
