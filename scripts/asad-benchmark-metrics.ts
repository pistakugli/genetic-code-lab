export const ASAD_MAE_COMPARISON_TOLERANCE = 0.05;
export const ASAD_MAE_ABSOLUTE_FLOOR = 1e-8;

export function maeValuesAreComparable(first: number, second: number): boolean {
  const scale = Math.max(Math.abs(first), Math.abs(second), ASAD_MAE_ABSOLUTE_FLOOR);
  return Math.abs(first - second) <= ASAD_MAE_COMPARISON_TOLERANCE * scale;
}

export function exceedsAllowedMaeRegression(asadMae: number, baselineMae: number): boolean {
  return asadMae > baselineMae * (1 + ASAD_MAE_COMPARISON_TOLERANCE) + ASAD_MAE_ABSOLUTE_FLOOR;
}

export function oracleCallReduction(asadCalls: number, comparableBaselineCalls: number[]): number | null {
  if (!comparableBaselineCalls.length) return null;
  const bestComparableCalls = Math.min(...comparableBaselineCalls);
  if (!Number.isFinite(asadCalls) || asadCalls < 0 || !Number.isFinite(bestComparableCalls) || bestComparableCalls <= 0) {
    throw new Error('Oracle-call comparison requires finite non-negative ASAD calls and positive baseline calls');
  }
  return 1 - asadCalls / bestComparableCalls;
}
