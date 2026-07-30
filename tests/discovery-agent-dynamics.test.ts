import { DiscoveryAgent } from '../src/agents/discovery-agent';

/**
 * analyzeMultiIslandDynamics used to return { avgBias: 0.5, globalIntensity: 1.0 }
 * unconditionally without reading its argument, so a function named "analyze" on a live
 * evolution path analysed nothing, and the numbers it fed into the mutation-intensity and
 * diversity-bias signals were constants.
 *
 * These tests pin the directions rather than the constants: a converged population must
 * pull bias up, a diverse one must pull it down, sustained stagnation must raise mutation
 * intensity, and progress must leave it neutral. That is the behaviour the consumers rely
 * on, and it is what a constant cannot satisfy.
 */
describe('DiscoveryAgent.analyzeMultiIslandDynamics', () => {
  const agent = () => new DiscoveryAgent();

  test('diversity bias rises as population entropy falls', () => {
    const converged = agent().analyzeMultiIslandDynamics([{ populationEntropy: 0.05, stagnationCount: 0 }]);
    const diverse = agent().analyzeMultiIslandDynamics([{ populationEntropy: 0.95, stagnationCount: 0 }]);

    expect(converged.avgBias).toBeGreaterThan(diverse.avgBias);
    expect(converged.avgBias).toBeGreaterThan(0.9);
    expect(diverse.avgBias).toBeLessThan(0.1);
  });

  test('mutation intensity rises with sustained stagnation and stays neutral while improving', () => {
    const improving = agent().analyzeMultiIslandDynamics([{ populationEntropy: 0.5, stagnationCount: 0 }]);
    const stalled = agent().analyzeMultiIslandDynamics([{ populationEntropy: 0.5, stagnationCount: 40 }]);

    expect(improving.globalIntensity).toBe(1.0);
    expect(stalled.globalIntensity).toBeGreaterThan(improving.globalIntensity);
  });

  test('mutation intensity is capped so a long plateau cannot become a random walk', () => {
    const extreme = agent().analyzeMultiIslandDynamics([{ populationEntropy: 0.5, stagnationCount: 100000 }]);

    expect(extreme.globalIntensity).toBeLessThanOrEqual(3.0);
  });

  test('the worst-stalled island sets the pressure rather than the average', () => {
    const averaged = agent().analyzeMultiIslandDynamics([
      { populationEntropy: 0.5, stagnationCount: 0 },
      { populationEntropy: 0.5, stagnationCount: 0 },
      { populationEntropy: 0.5, stagnationCount: 30 }
    ]);
    const onlyStalled = agent().analyzeMultiIslandDynamics([{ populationEntropy: 0.5, stagnationCount: 30 }]);

    expect(averaged.globalIntensity).toBe(onlyStalled.globalIntensity);
  });

  test('bias averages across islands instead of tracking one of them', () => {
    const mixed = agent().analyzeMultiIslandDynamics([
      { populationEntropy: 0.0, stagnationCount: 0 },
      { populationEntropy: 1.0, stagnationCount: 0 }
    ]);

    expect(mixed.avgBias).toBeCloseTo(0.5, 6);
  });

  test('malformed rows fall back to neutral instead of producing NaN', () => {
    const empty = agent().analyzeMultiIslandDynamics([]);
    const garbage = agent().analyzeMultiIslandDynamics([
      { populationEntropy: Number.NaN, stagnationCount: Number.POSITIVE_INFINITY }
    ]);

    expect(empty).toEqual({ avgBias: 0.5, globalIntensity: 1.0 });
    expect(Number.isFinite(garbage.avgBias)).toBe(true);
    expect(Number.isFinite(garbage.globalIntensity)).toBe(true);
    expect(garbage.globalIntensity).toBeLessThanOrEqual(3.0);
  });

  test('out-of-range entropy cannot push the bias outside its declared range', () => {
    const observed = agent().analyzeMultiIslandDynamics([
      { populationEntropy: -5, stagnationCount: 0 },
      { populationEntropy: 12, stagnationCount: 0 }
    ]);

    expect(observed.avgBias).toBeGreaterThanOrEqual(0);
    expect(observed.avgBias).toBeLessThanOrEqual(1);
  });
});
