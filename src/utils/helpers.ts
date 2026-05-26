/**
 * Genetic Code Lab - Utility Functions
 */

import { v4 as uuidv4 } from 'uuid';

// ============================================================================
// RANDOM UTILITIES
// ============================================================================

let seedState: number | null = null;

/**
 * Set random seed for reproducibility
 */
export function setSeed(seed: number): void {
  seedState = seed % 2147483647;
  if (seedState <= 0) seedState += 2147483646;
}

export function getRandomState(): number | null {
  return seedState;
}

export function restoreRandomState(state: number | null): void {
  seedState = state;
}

/**
 * Get next random number using seeded random
 */
export function random(): number {
  if (seedState === null) {
    return Math.random();
  }

  // Linear congruential generator
  seedState = (seedState * 16807) % 2147483647;
  return (seedState - 1) / 2147483646;
}

/**
 * Get random integer in range [min, max]
 */
export function randomInt(min: number, max: number): number {
  return Math.floor(random() * (max - min + 1)) + min;
}

/**
 * Get random float in range [min, max]
 */
export function randomFloat(min: number, max: number): number {
  return random() * (max - min) + min;
}

/**
 * Get random element from array
 */
export function randomChoice<T>(array: T[]): T {
  return array[Math.floor(random() * array.length)];
}

/**
 * Get N random elements from array (without replacement)
 */
export function randomSample<T>(array: T[], n: number): T[] {
  const copy = [...array];
  const result: T[] = [];

  for (let i = 0; i < Math.min(n, copy.length); i++) {
    const idx = Math.floor(random() * copy.length);
    result.push(copy.splice(idx, 1)[0]);
  }

  return result;
}

/**
 * Shuffle array in place
 */
export function shuffle<T>(array: T[]): T[] {
  const result = [...array];
  for (let i = result.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [result[i], result[j]] = [result[j], result[i]];
  }
  return result;
}

/**
 * Random boolean with probability p
 */
export function randomBool(p: number = 0.5): boolean {
  return random() < p;
}

// ============================================================================
// ID GENERATION
// ============================================================================

/**
 * Generate unique ID
 */
export function generateId(): string {
  return uuidv4();
}

/**
 * Generate short ID (8 characters)
 */
export function generateShortId(): string {
  return uuidv4().split('-')[0];
}

// ============================================================================
// MATH UTILITIES
// ============================================================================

/**
 * Clamp value between min and max
 */
export function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

/**
 * Linear interpolation
 */
export function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

/**
 * Calculate mean of array
 */
export function mean(values: number[]): number {
  if (values.length === 0) return 0;
  return values.reduce((sum, v) => sum + v, 0) / values.length;
}

/**
 * Calculate standard deviation
 */
export function stdDev(values: number[]): number {
  if (values.length === 0) return 0;
  const avg = mean(values);
  const squareDiffs = values.map(v => Math.pow(v - avg, 2));
  return Math.sqrt(mean(squareDiffs));
}

/**
 * Calculate median
 */
export function median(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 !== 0
    ? sorted[mid]
    : (sorted[mid - 1] + sorted[mid]) / 2;
}

/**
 * Check if number is finite and valid
 */
export function isValidNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value);
}

/**
 * Safe division (returns 0 if divisor is 0)
 */
export function safeDivide(a: number, b: number): number {
  return b === 0 ? 0 : a / b;
}

// ============================================================================
// ARRAY UTILITIES
// ============================================================================

/**
 * Flatten nested array
 */
export function flatten<T>(array: (T | T[])[]): T[] {
  return array.reduce<T[]>((acc, val) =>
    Array.isArray(val) ? acc.concat(val) : acc.concat(val), []
  );
}

/**
 * Get unique values from array
 */
export function unique<T>(array: T[]): T[] {
  return [...new Set(array)];
}

/**
 * Group array by key function
 */
export function groupBy<T, K extends string | number>(
  array: T[],
  keyFn: (item: T) => K
): Record<K, T[]> {
  return array.reduce((acc, item) => {
    const key = keyFn(item);
    if (!acc[key]) acc[key] = [];
    acc[key].push(item);
    return acc;
  }, {} as Record<K, T[]>);
}

/**
 * Sort array by key function
 */
export function sortBy<T>(array: T[], keyFn: (item: T) => number, desc: boolean = false): T[] {
  const sorted = [...array].sort((a, b) => keyFn(a) - keyFn(b));
  return desc ? sorted.reverse() : sorted;
}

// ============================================================================
// OBJECT UTILITIES
// ============================================================================

/**
 * Deep clone object
 */
export function deepClone<T>(obj: T): T {
  if (obj === null || typeof obj !== 'object') {
    return obj;
  }
  if (Array.isArray(obj)) {
    const arr = new Array(obj.length);
    for (let i = 0; i < obj.length; i++) {
      arr[i] = deepClone(obj[i]);
    }
    return arr as any;
  }
  const clonedObj: any = {};
  for (const key in obj) {
    if (Object.prototype.hasOwnProperty.call(obj, key)) {
      clonedObj[key] = deepClone((obj as any)[key]);
    }
  }
  return clonedObj;
}

/**
 * Deep equal comparison
 */
export function deepEqual(a: unknown, b: unknown): boolean {
  return JSON.stringify(a) === JSON.stringify(b);
}

// ============================================================================
// STRING UTILITIES
// ============================================================================

/**
 * Pad string to length
 */
export function padLeft(str: string, length: number, char: string = ' '): string {
  return str.padStart(length, char);
}

/**
 * Truncate string with ellipsis
 */
export function truncate(str: string, maxLength: number): string {
  if (str.length <= maxLength) return str;
  return str.slice(0, maxLength - 3) + '...';
}

// ============================================================================
// TIME UTILITIES
// ============================================================================

/**
 * Format milliseconds to human readable string
 */
export function formatTime(ms: number): string {
  if (ms < 1000) return `${ms.toFixed(0)}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(2)}s`;
  if (ms < 3600000) return `${(ms / 60000).toFixed(2)}m`;
  return `${(ms / 3600000).toFixed(2)}h`;
}

/**
 * Sleep for specified milliseconds
 */
export function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// ============================================================================
// VALIDATION UTILITIES
// ============================================================================

/**
 * Assert condition is true
 */
export function assert(condition: boolean, message: string): asserts condition {
  if (!condition) {
    throw new Error(`Assertion failed: ${message}`);
  }
}

/**
 * Check if value is defined (not null or undefined)
 */
export function isDefined<T>(value: T | null | undefined): value is T {
  return value !== null && value !== undefined;
}
