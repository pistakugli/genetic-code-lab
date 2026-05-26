import {
  decompileDeepBlackBoxRelu1D,
  predictDeepBlackBoxRelu1DBatch
} from '../src/solvers/deep-blackbox-relu1d-decompiler';
import { regressionMetrics } from '../src/solvers/symbolic-residual-ensemble';

describe('deep black-box ReLU 1D decompiler', () => {
  const teacher = {
    l1: [
      { w: -1.9999060854314883, b: -0.526933179262032 },
      { w: -0.932736135025263, b: -0.6202480226012395 },
      { w: 0.3932068481978095, b: 0.7843877941224608 },
      { w: -1.4354646070259294, b: -2.3170292492183195 },
      { w: -1.8484431317527248, b: 1.5203934386562494 }
    ],
    l2: [
      { w: [-1.3979750717039918, 0.23299646399262963, -0.028415847596168442, -1.5851346753399211, -1.3584603810296025], b: 0.2673017394424433 },
      { w: [-1.9462073929106884, -1.907621767285859, -1.3990121925240495, -1.1978931512664008, -0.9901683078987229], b: -1.319045590068256 },
      { w: [1.0677197138478234, 1.1652379363451506, 0.15400268524326632, 0.3231453302531926, -0.8964213122580398], b: 1.3852706606362677 },
      { w: [-1.0080077191889358, -1.5857128683344581, 0.9238499653747771, -0.8536235251032036, 1.149435923573967], b: -1.0728193778291526 },
      { w: [-1.1670173864504485, 1.9388107135322046, 1.5916628144603808, -0.9230741699441096, 1.8924486263584797], b: -1.2119519647322148 }
    ],
    out: {
      w: [0.7233499700421002, 1.3429525758539818, -0.996056394182198, -0.7197974861783882, 1.3636671727184835],
      b: -0.16916541091088708,
      base: 0.15796166170198628
    }
  };

  function relu(value: number): number {
    return Math.max(0, value);
  }

  function deepTeacher(x: number): number {
    const h1 = teacher.l1.map(unit => relu(unit.w * x + unit.b));
    const h2 = teacher.l2.map(unit => relu(unit.b + unit.w.reduce((sum, weight, index) => sum + weight * h1[index], 0)));
    return teacher.out.b + teacher.out.base * x + teacher.out.w.reduce((sum, weight, index) => sum + weight * h2[index], 0);
  }

  it('recovers an output-equivalent canonical spline for a two-layer folded ReLU teacher', () => {
    const model = decompileDeepBlackBoxRelu1D(deepTeacher, {
      min: -3,
      max: 3,
      samples: 12001,
      slopeTolerance: 1e-7,
      minPlateauIntervals: 4,
      kinkMergeTolerance: 0.002
    });
    const xs = Array.from({ length: 1001 }, (_, index) => -2.99 + (5.98 * index) / 1000);
    const actual = xs.map(deepTeacher);
    const predicted = predictDeepBlackBoxRelu1DBatch(model, xs);
    const metrics = regressionMetrics(actual, predicted);
    const firstLayerKinks = teacher.l1
      .map(unit => -unit.b / unit.w)
      .filter(kink => kink > -3 && kink < 3);

    expect(model.recoveryKind).toBe('output-equivalent-canonical-spline');
    expect(model.kinkCount).toBe(11);
    expect(model.kinkCount).toBeGreaterThan(firstLayerKinks.length);
    expect(model.regionCount).toBe(12);
    expect(metrics.mae).toBeLessThan(1e-10);
    expect(metrics.r2).toBeGreaterThan(0.999999999);
  });
});
