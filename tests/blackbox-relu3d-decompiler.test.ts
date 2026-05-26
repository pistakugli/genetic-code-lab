import {
  decompileBlackBoxRelu3D,
  predictBlackBoxRelu3DBatch
} from '../src/solvers/blackbox-relu3d-decompiler';
import { regressionMetrics } from '../src/solvers/symbolic-residual-ensemble';

type ExpectedTerm = {
  alphaY: number;
  alphaZ: number;
  beta: number;
  deltaSlopeX: number;
};

describe('black-box 3D ReLU decompiler', () => {
  function teacher(x: number, y: number, z: number): number {
    return 0.18
      + 0.22 * x
      - 0.11 * y
      + 0.07 * z
      + 0.82 * Math.max(0, x + 0.25 * y - 0.20 * z + 1.10)
      - 0.48 * Math.max(0, x - 0.35 * y + 0.30 * z - 0.15)
      + 0.31 * Math.max(0, x + 0.10 * y + 0.22 * z - 1.20);
  }

  function maxGeometryError(recoveredTerms: ExpectedTerm[]): number {
    const expected = [
      { alphaY: -0.25, alphaZ: 0.20, beta: -1.10, deltaSlopeX: 0.82 },
      { alphaY: 0.35, alphaZ: -0.30, beta: 0.15, deltaSlopeX: -0.48 },
      { alphaY: -0.10, alphaZ: -0.22, beta: 1.20, deltaSlopeX: 0.31 }
    ].sort((a, b) => a.deltaSlopeX - b.deltaSlopeX);
    const recovered = [...recoveredTerms].sort((a, b) => a.deltaSlopeX - b.deltaSlopeX);
    return expected.reduce((max, item, index) => {
      const term = recovered[index];
      return Math.max(
        max,
        Math.abs(item.deltaSlopeX - (term?.deltaSlopeX ?? Infinity)),
        Math.abs(item.alphaY - (term?.alphaY ?? Infinity)),
        Math.abs(item.alphaZ - (term?.alphaZ ?? Infinity)),
        Math.abs(item.beta - (term?.beta ?? Infinity))
      );
    }, 0);
  }

  it('stitches 2D slice kink lines into 3D ReLU hyperplanes from oracle probes', () => {
    const model = decompileBlackBoxRelu3D(teacher, {
      xRange: [-3, 3],
      yRange: [-1.4, 1.4],
      zRange: [-1.2, 1.2],
      sliceCount: 7,
      samplesPerSlice: 801,
      slopeTolerance: 1e-8,
      clusterTolerance: 1e-7
    });
    const inputs: Array<[number, number, number]> = [];
    for (let i = 0; i < 29; i++) {
      for (let j = 0; j < 19; j++) {
        for (let k = 0; k < 15; k++) {
          inputs.push([
            -2.8 + (5.6 * i) / 28,
            -1.3 + (2.6 * j) / 18,
            -1.1 + (2.2 * k) / 14
          ]);
        }
      }
    }
    const actual = inputs.map(([x, y, z]) => teacher(x, y, z));
    const predicted = predictBlackBoxRelu3DBatch(model, inputs);
    const metrics = regressionMetrics(actual, predicted);

    expect(model.terms).toHaveLength(3);
    expect(maxGeometryError(model.terms)).toBeLessThan(1e-8);
    expect(model.expression).toContain('max(0');
    expect(metrics.mae).toBeLessThan(1e-10);
    expect(metrics.r2).toBeGreaterThan(0.999999999);
  });
});
