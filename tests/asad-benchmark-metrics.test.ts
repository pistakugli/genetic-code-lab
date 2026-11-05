import {
  exceedsAllowedMaeRegression,
  maeValuesAreComparable,
  oracleCallReduction
} from '../scripts/asad-benchmark-metrics';

describe('ASAD benchmark MAE comparisons', () => {
  it('accepts a five-percent difference but not a larger relative difference', () => {
    expect(maeValuesAreComparable(1, 1.05)).toBe(true);
    expect(maeValuesAreComparable(1, 1.06)).toBe(false);
  });

  it('does not call near-zero and materially larger errors comparable', () => {
    expect(maeValuesAreComparable(0, 4e-10)).toBe(true);
    expect(maeValuesAreComparable(0, 5e-5)).toBe(false);
  });

  it('applies the allowed-regression threshold relative to the baseline', () => {
    expect(exceedsAllowedMaeRegression(1.05, 1)).toBe(false);
    expect(exceedsAllowedMaeRegression(1.06, 1)).toBe(true);
    expect(exceedsAllowedMaeRegression(5e-9, 0)).toBe(false);
    expect(exceedsAllowedMaeRegression(5e-5, 0)).toBe(true);
  });

  it('does not report oracle savings when no baseline has comparable MAE', () => {
    expect(oracleCallReduction(100, [])).toBeNull();
    expect(oracleCallReduction(100, [500])).toBeCloseTo(0.8);
  });
});
