/**
 * Lexicase Selection & Epsilon-Lexicase Selection
 * ─────────────────────────────────────────────────
 * Standard tournament selection is known to cause premature convergence
 * in symbolic regression because it aggregates all test case errors into a
 * single scalar—discarding which specific cases a genome handles well.
 *
 * Lexicase selection instead:
 *   1. Shuffles test cases into a random order each parent selection.
 *   2. Filters the current pool to only genomes that are best (or within ε)
 *      on the *first* case in the shuffled sequence.
 *   3. Repeats with the next case until one genome remains or all cases are
 *      exhausted.
 *
 * Epsilon-Lexicase extends this to continuous domains by using an ε tolerance
 * so near-best performers are not prematurely eliminated.
 *
 * References:
 *   Helmuth et al. (2014) – "Solving Uncompromising Problems with Lexicase Selection"
 *   La Croix et al. (2016) – "Epsilon-Lexicase Selection for Regression"
 */

import { Genome, Population } from './types';
import { deepClone, random } from '../utils/helpers';

// ============================================================================
// UTILITIES
// ============================================================================

/**
 * Compute Median Absolute Deviation (MAD) per test case across the population.
 * Used as the default epsilon for epsilon-lexicase.
 */
export function computeMAD(individuals: Genome[]): number[] {
  if (individuals.length === 0 || !individuals[0].caseErrors) return [];
  const numCases = individuals[0].caseErrors.length;
  const mads: number[] = new Array(numCases).fill(0);

  for (let c = 0; c < numCases; c++) {
    const errors = individuals
      .map(ind => ind.caseErrors?.[c] ?? 0)
      .sort((a, b) => a - b);
    const mid = Math.floor(errors.length / 2);
    const median = errors.length % 2 === 0
      ? (errors[mid - 1] + errors[mid]) / 2
      : errors[mid];
    const absDeviations = errors.map(e => Math.abs(e - median)).sort((a, b) => a - b);
    const madMid = Math.floor(absDeviations.length / 2);
    mads[c] = absDeviations.length % 2 === 0
      ? (absDeviations[madMid - 1] + absDeviations[madMid]) / 2
      : absDeviations[madMid];
  }

  return mads;
}

// ============================================================================
// STANDARD LEXICASE (for discrete problems: boolean, santa-fe)
// ============================================================================

/**
 * Select one parent by lexicase selection from the pool.
 * Enhanced for SOTA performance with case sampling.
 */
export function lexicaseSelectOne(individuals: Genome[], maxCases: number = 50): Genome {
  let pool = individuals.filter(g => g.caseErrors && g.caseErrors.length > 0);
  if (pool.length === 0) return deepClone(individuals[Math.floor(random() * individuals.length)]);

  const totalCases = pool[0].caseErrors!.length;
  const numCasesToUse = Math.min(totalCases, maxCases);
  
  // High-entropy random subset of cases
  const caseOrder = Array.from({ length: totalCases }, (_, i) => i)
    .sort(() => random() - 0.5)
    .slice(0, numCasesToUse);

  for (const ci of caseOrder) {
    if (pool.length === 1) break;
    
    // Fast find-best loop
    let best = Infinity;
    for (const g of pool) {
      const err = g.caseErrors![ci];
      if (err < best) best = err;
    }
    
    // Filter pool in-place (avoiding new array allocations)
    const nextPool = [];
    for (const g of pool) {
      if (g.caseErrors![ci] <= best) nextPool.push(g);
    }
    pool = nextPool;
  }

  // PARSIMONY TIE-BREAKING
  if (pool.length > 1) {
    let minSize = Infinity;
    for (const g of pool) if (g.genes.length < minSize) minSize = g.genes.length;
    pool = pool.filter(g => g.genes.length === minSize);
  }

  return deepClone(pool[Math.floor(random() * pool.length)]);
}

/**
 * Select `count` parents using standard Lexicase Selection.
 * Enhanced with Tournament Lexicase for large populations.
 */
export function lexicaseSelection(population: Population, count: number): Genome[] {
  const selected: Genome[] = [];
  const individuals = population.genomes;
  const tourneySize = Math.min(individuals.length, 500); // Tournament-Lexicase Optimization

  for (let i = 0; i < count; i++) {
    // Pick 500 random for tournament
    const tournament = Array.from({ length: tourneySize }, () => 
      individuals[Math.floor(random() * individuals.length)]
    );
    selected.push(lexicaseSelectOne(tournament));
  }
  return selected;
}

// ============================================================================
// EPSILON-LEXICASE (for continuous problems: math, financial)
// ============================================================================

/**
 * Select one parent by epsilon-lexicase selection.
 * Enhanced for SOTA performance with case sampling.
 */
export function epsilonLexicaseSelectOne(
  individuals: Genome[],
  epsilon: number[],
  maxCases: number = 50
): Genome {
  let pool = individuals.filter(g => g.caseErrors && g.caseErrors.length > 0);
  if (pool.length === 0) return deepClone(individuals[Math.floor(random() * individuals.length)]);

  const totalCases = pool[0].caseErrors!.length;
  const numCasesToUse = Math.min(totalCases, maxCases);
  
  const caseOrder = Array.from({ length: totalCases }, (_, i) => i)
    .sort(() => random() - 0.5)
    .slice(0, numCasesToUse);

  for (const ci of caseOrder) {
    if (pool.length === 1) break;
    const eps = epsilon[ci] ?? 0;
    
    let best = Infinity;
    for (const g of pool) {
      if (g.caseErrors![ci] < best) best = g.caseErrors![ci];
    }
    
    const nextPool = [];
    const threshold = best + eps;
    for (const g of pool) {
      if (g.caseErrors![ci] <= threshold) nextPool.push(g);
    }
    pool = nextPool;
  }

  // PARSIMONY TIE-BREAKING
  if (pool.length > 1) {
    let minSize = Infinity;
    for (const g of pool) if (g.genes.length < minSize) minSize = g.genes.length;
    pool = pool.filter(g => g.genes.length === minSize);
  }

  return deepClone(pool[Math.floor(random() * pool.length)]);
}

/**
 * Select `count` parents using Epsilon-Lexicase Selection.
 * Enhanced with Tournament Lexicase for large populations.
 */
export function epsilonLexicaseSelection(
  population: Population,
  count: number,
  fixedEpsilon?: number
): Genome[] {
  const individuals = population.genomes;
  const tourneySize = Math.min(individuals.length, 500); // Optimization
  let epsilon: number[];

  if (fixedEpsilon !== undefined) {
    const numCases = individuals.find(g => g.caseErrors)?.caseErrors?.length ?? 0;
    epsilon = new Array(numCases).fill(fixedEpsilon);
  } else {
    epsilon = computeMAD(individuals);
  }

  const selected: Genome[] = [];
  for (let i = 0; i < count; i++) {
    const tournament = Array.from({ length: tourneySize }, () => 
      individuals[Math.floor(random() * individuals.length)]
    );
    selected.push(epsilonLexicaseSelectOne(tournament, epsilon));
  }
  return selected;
}
