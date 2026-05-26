import { BlockEvaluator } from './block-evaluator';
import { PopulationBuffer } from './pop-buffer';
import { SemanticBasis, SemanticRefiner } from './semantic-refiner';

type ArchiveMetrics = {
  accuracy: number;
  localNovelty: number;
  globalNovelty: number;
  stability: number;
  recombinationGain: number;
  complexityPenalty: number;
};

type ArchiveEntry = {
  basis: SemanticBasis;
  fingerprint: Float64Array;
  metrics: ArchiveMetrics;
  price: number;
  hits: number;
  lastSeen: number;
};

type ScoredBasis = {
  basis: SemanticBasis;
  fingerprint: Float64Array;
  metrics: ArchiveMetrics;
  price: number;
};

export type SemanticArchiveDiagnostics = {
  key: string;
  price: number;
  hits: number;
  lastSeen: number;
  metrics: ArchiveMetrics;
};

/**
 * Problem-local market of useful semantic building blocks.
 *
 * Entries are scored in behavior space, not syntax space. That means equivalent
 * forms such as x*x and 2*(x*x)/2 compete as near-duplicates, while bases with
 * distinct behavior can coexist even when their raw correlation is lower.
 */
export class SemanticArchive {
  private entries = new Map<string, ArchiveEntry>();
  private generation = 0;

  constructor(private readonly maxEntries: number = 256) {}

  public recordElites(
    buffer: PopulationBuffer,
    eliteIndices: number[],
    inputs: Float64Array,
    expected: Float64Array,
    inputSize: number
  ): void {
    if (process.env.GCL_SEMANTIC_ARCHIVE === '0') return;
    this.generation++;

    const observed = new Map<string, SemanticBasis>();
    for (const idx of eliteIndices) {
      const bases = SemanticRefiner.extractSemanticBases(buffer, idx, inputs, inputSize);
      for (const basis of bases) observed.set(basis.key, basis);
    }

    const scored = this.scoreObservedBases([...observed.values()], expected);
    for (const item of scored) this.recordScoredBasis(item);
    this.prune();
  }

  public injectCandidate(
    buffer: PopulationBuffer,
    incumbentIdx: number,
    targetIdx: number,
    inputs: Float64Array,
    expected: Float64Array,
    inputSize: number,
    maxGenomeSize: number
  ): boolean {
    if (process.env.GCL_SEMANTIC_ARCHIVE === '0') return false;
    const bases = this.snapshotBases();
    if (bases.length < 2) return false;

    const baseline = BlockEvaluator.getPredictions(buffer, incumbentIdx, inputs, inputSize);
    const candidate = SemanticRefiner.synthesizeFromBases(bases, expected, baseline, maxGenomeSize);
    if (!candidate) return false;

    buffer.writeGenome(targetIdx, candidate.opcodes, [], candidate.constants);
    return true;
  }

  public size(): number {
    return this.entries.size;
  }

  public diagnostics(): SemanticArchiveDiagnostics[] {
    return [...this.entries.entries()]
      .map(([key, entry]) => ({
        key,
        price: entry.price,
        hits: entry.hits,
        lastSeen: entry.lastSeen,
        metrics: { ...entry.metrics }
      }))
      .sort((a, b) => b.price - a.price || a.key.localeCompare(b.key));
  }

  private scoreObservedBases(bases: SemanticBasis[], expected: Float64Array): ScoredBasis[] {
    const scored: ScoredBasis[] = [];
    const fingerprints = new Map<string, Float64Array>();

    for (const basis of bases) {
      const fingerprint = this.makeFingerprint(basis.values);
      if (!fingerprint) continue;
      fingerprints.set(basis.key, fingerprint);
    }

    for (const basis of bases) {
      const fingerprint = fingerprints.get(basis.key);
      if (!fingerprint) continue;

      const accuracy = this.absCorrelation(basis.values, expected);
      if (!(accuracy > 0)) continue;

      const localFingerprints = [...fingerprints.entries()]
        .filter(([key]) => key !== basis.key)
        .map(([, value]) => value);
      const globalFingerprints = [...this.entries.values()].map(entry => entry.fingerprint);
      const trainRows = this.rows(expected.length, false);
      const validationRows = this.rows(expected.length, true);
      const trainCorrelation = this.absCorrelation(basis.values, expected, trainRows);
      const validationCorrelation = this.absCorrelation(basis.values, expected, validationRows);
      const stability = 1 - Math.min(1, Math.abs(trainCorrelation - validationCorrelation));
      const recombinationGain = this.recombinationGain(basis, expected);
      const complexityPenalty = 1 / Math.sqrt(Math.max(1, basis.complexity));
      const metrics: ArchiveMetrics = {
        accuracy,
        localNovelty: this.noveltyAgainst(fingerprint, localFingerprints),
        globalNovelty: this.noveltyAgainst(fingerprint, globalFingerprints),
        stability,
        recombinationGain,
        complexityPenalty
      };

      scored.push({
        basis,
        fingerprint,
        metrics,
        price: this.scalarPrice(metrics)
      });
    }

    return scored;
  }

  private recordScoredBasis(item: ScoredBasis): void {
    const existing = this.entries.get(item.basis.key);
    if (existing) {
      existing.price = Math.max(existing.price * 0.985, item.price) + Math.min(0.02, existing.hits * 0.001);
      existing.metrics = item.metrics;
      existing.fingerprint = item.fingerprint;
      existing.hits++;
      existing.lastSeen = this.generation;
      existing.basis = item.basis;
      return;
    }

    this.entries.set(item.basis.key, {
      basis: item.basis,
      fingerprint: item.fingerprint,
      metrics: item.metrics,
      price: item.price,
      hits: 1,
      lastSeen: this.generation
    });
  }

  private snapshotBases(): SemanticBasis[] {
    return [...this.entries.values()]
      .sort((a, b) => b.price - a.price || a.basis.complexity - b.basis.complexity)
      .slice(0, 96)
      .map(entry => entry.basis);
  }

  private prune(): void {
    if (this.entries.size <= this.maxEntries) return;
    const ranked = this.paretoRank([...this.entries.entries()]);
    this.entries.clear();
    for (const [key, entry] of ranked.slice(0, this.maxEntries)) this.entries.set(key, entry);
  }

  private paretoRank(entries: Array<[string, ArchiveEntry]>): Array<[string, ArchiveEntry]> {
    const remaining = [...entries];
    const ranked: Array<[string, ArchiveEntry]> = [];

    while (remaining.length > 0) {
      const front: Array<[string, ArchiveEntry]> = [];
      for (const candidate of remaining) {
        const dominated = remaining.some(other => other !== candidate && this.dominates(other[1], candidate[1]));
        if (!dominated) front.push(candidate);
      }

      front.sort((a, b) => this.temporalPrice(b[1]) - this.temporalPrice(a[1]));
      ranked.push(...front);

      for (const item of front) {
        const index = remaining.indexOf(item);
        if (index >= 0) remaining.splice(index, 1);
      }
    }

    return ranked;
  }

  private dominates(left: ArchiveEntry, right: ArchiveEntry): boolean {
    const leftValues = this.metricVector(left);
    const rightValues = this.metricVector(right);
    let strictlyBetter = false;

    for (let i = 0; i < leftValues.length; i++) {
      if (leftValues[i] + 1e-12 < rightValues[i]) return false;
      if (leftValues[i] > rightValues[i] + 1e-12) strictlyBetter = true;
    }

    return strictlyBetter;
  }

  private metricVector(entry: ArchiveEntry): number[] {
    const metrics = entry.metrics;
    return [
      metrics.accuracy,
      metrics.localNovelty,
      metrics.globalNovelty,
      metrics.stability,
      metrics.recombinationGain,
      metrics.complexityPenalty
    ];
  }

  private scalarPrice(metrics: ArchiveMetrics): number {
    return (
      metrics.accuracy * 0.32
      + Math.max(metrics.localNovelty, metrics.globalNovelty) * 0.22
      + metrics.recombinationGain * 0.22
      + metrics.stability * 0.14
      + metrics.complexityPenalty * 0.1
    );
  }

  private temporalPrice(entry: ArchiveEntry): number {
    const age = this.generation - entry.lastSeen;
    return entry.price + Math.min(0.05, entry.hits * 0.002) - age * 0.003;
  }

  private recombinationGain(basis: SemanticBasis, expected: Float64Array): number {
    const companions = [...this.entries.values()]
      .filter(entry => entry.basis.key !== basis.key)
      .sort((a, b) => b.price - a.price)
      .slice(0, 6)
      .map(entry => entry.basis.values);

    if (companions.length === 0) return this.absCorrelation(basis.values, expected);

    const coefficients = this.fitLinearBasis(companions, expected);
    const residual = new Float64Array(expected.length);
    for (let row = 0; row < expected.length; row++) {
      let predicted = 0;
      for (let i = 0; i < companions.length; i++) predicted += (coefficients[i] || 0) * companions[i][row];
      residual[row] = expected[row] - predicted;
    }
    return this.absCorrelation(basis.values, residual);
  }

  private makeFingerprint(values: Float64Array): Float64Array | null {
    if (!this.channelHasSignal(values)) return null;

    const slopes = new Float64Array(values.length);
    const curvature = new Float64Array(values.length);
    for (let i = 0; i < values.length; i++) {
      const prev = i === 0 ? values[i] : values[i - 1];
      const next = i === values.length - 1 ? values[i] : values[i + 1];
      slopes[i] = values[i] - prev;
      curvature[i] = next - 2 * values[i] + prev;
    }

    const fingerprint = new Float64Array(values.length * 3);
    this.writeNormalizedChannel(values, fingerprint, 0);
    this.writeNormalizedChannel(slopes, fingerprint, values.length);
    this.writeNormalizedChannel(curvature, fingerprint, values.length * 2);
    return fingerprint;
  }

  private channelHasSignal(values: Float64Array): boolean {
    let min = Number.POSITIVE_INFINITY;
    let max = Number.NEGATIVE_INFINITY;
    for (let i = 0; i < values.length; i++) {
      min = Math.min(min, values[i]);
      max = Math.max(max, values[i]);
    }
    return max - min > 1e-12 || Math.abs(max) > 1e-12;
  }

  private writeNormalizedChannel(values: Float64Array, target: Float64Array, offset: number): void {
    let mean = 0;
    for (let i = 0; i < values.length; i++) mean += values[i];
    mean /= Math.max(1, values.length);

    let variance = 0;
    for (let i = 0; i < values.length; i++) {
      const centered = values[i] - mean;
      variance += centered * centered;
    }
    const std = Math.sqrt(variance / Math.max(1, values.length));

    if (!(std > 1e-12)) {
      for (let i = 0; i < values.length; i++) target[offset + i] = 0;
      return;
    }

    for (let i = 0; i < values.length; i++) {
      const normalized = (values[i] - mean) / std;
      target[offset + i] = Math.max(-4, Math.min(4, normalized)) / 4;
    }
  }

  private noveltyAgainst(fingerprint: Float64Array, others: Float64Array[]): number {
    if (others.length === 0) return 1;
    const distances = others
      .map(other => this.fingerprintDistance(fingerprint, other))
      .sort((a, b) => a - b)
      .slice(0, Math.min(5, others.length));
    const avg = distances.reduce((sum, value) => sum + value, 0) / distances.length;
    return Math.max(0, Math.min(1, avg));
  }

  private fingerprintDistance(left: Float64Array, right: Float64Array): number {
    const length = Math.min(left.length, right.length);
    if (length === 0) return 0;
    let total = 0;
    for (let i = 0; i < length; i++) {
      const diff = left[i] - right[i];
      total += diff * diff;
    }
    return Math.sqrt(total / length);
  }

  private absCorrelation(values: Float64Array, expected: Float64Array, rows?: number[]): number {
    const indices = rows && rows.length > 0 ? rows : Array.from({ length: expected.length }, (_, i) => i);
    let meanX = 0;
    let meanY = 0;
    for (const i of indices) {
      meanX += values[i];
      meanY += expected[i];
    }
    meanX /= indices.length;
    meanY /= indices.length;

    let covariance = 0;
    let varianceX = 0;
    let varianceY = 0;
    for (const i of indices) {
      const dx = values[i] - meanX;
      const dy = expected[i] - meanY;
      covariance += dx * dy;
      varianceX += dx * dx;
      varianceY += dy * dy;
    }
    if (!(varianceX > 1e-18) || !(varianceY > 1e-18)) return 0;
    return Math.abs(covariance / Math.sqrt(varianceX * varianceY));
  }

  private rows(length: number, validation: boolean): number[] {
    const rows: number[] = [];
    for (let i = 0; i < length; i++) {
      if ((i % 4 === 3) === validation) rows.push(i);
    }
    return rows;
  }

  private fitLinearBasis(columns: Float64Array[], target: Float64Array): number[] {
    const n = columns.length;
    const matrix = Array.from({ length: n }, () => new Array<number>(n).fill(0));
    const rhs = new Array<number>(n).fill(0);
    for (let row = 0; row < target.length; row++) {
      for (let i = 0; i < n; i++) {
        const vi = columns[i][row];
        rhs[i] += vi * target[row];
        for (let j = 0; j < n; j++) matrix[i][j] += vi * columns[j][row];
      }
    }
    for (let i = 0; i < n; i++) matrix[i][i] += 1e-10;
    return this.solveLinearSystem(matrix, rhs);
  }

  private solveLinearSystem(matrix: number[][], rhs: number[]): number[] {
    const n = rhs.length;
    const a = matrix.map((row, i) => [...row, rhs[i]]);
    for (let pivot = 0; pivot < n; pivot++) {
      let best = pivot;
      for (let row = pivot + 1; row < n; row++) {
        if (Math.abs(a[row][pivot]) > Math.abs(a[best][pivot])) best = row;
      }
      if (Math.abs(a[best][pivot]) < 1e-12) continue;
      if (best !== pivot) [a[pivot], a[best]] = [a[best], a[pivot]];
      const divisor = a[pivot][pivot];
      for (let col = pivot; col <= n; col++) a[pivot][col] /= divisor;
      for (let row = 0; row < n; row++) {
        if (row === pivot) continue;
        const factor = a[row][pivot];
        for (let col = pivot; col <= n; col++) a[row][col] -= factor * a[pivot][col];
      }
    }
    return a.map(row => Number.isFinite(row[n]) ? row[n] : 0);
  }
}
