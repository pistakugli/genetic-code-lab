import { setSeed, random } from '../src/utils/helpers';
import { MLP } from '../src/core/neural-network';
import { createPathfindingProblem } from '../src/solvers/pathfinding';

/**
 * setSeed existed and was respected by utils/helpers, but nine call sites in the engine and
 * solvers called Math.random() directly and bypassed it entirely: novelty selection,
 * population motif injection, MLP weight initialisation and shuffling, the numerical
 * optimizer's escape jump, sorting-network shuffling, math-expression constants and sample
 * generation, and pathfinding grid generation. Seeding a run therefore fixed only part of
 * it, and two runs with the same seed could still diverge.
 *
 * These tests pin the property that fix was for: with a seed set, the same seed reproduces
 * the same result and a different seed does not. Each covers a component that used to
 * bypass the seed, so a regression to Math.random() in any of them fails here.
 */
describe('seeded determinism across components that used to bypass the seed', () => {
  test('the helper itself is reproducible and seed-dependent', () => {
    setSeed(1234);
    const a = [random(), random(), random()];
    setSeed(1234);
    const b = [random(), random(), random()];
    setSeed(5678);
    const c = [random(), random(), random()];

    expect(a).toEqual(b);
    expect(a).not.toEqual(c);
  });

  test('MLP weight initialisation follows the seed', () => {
    setSeed(777);
    const first = new MLP([3, 4, 1]).predict([[0.1, 0.2, 0.3]]);
    setSeed(777);
    const same = new MLP([3, 4, 1]).predict([[0.1, 0.2, 0.3]]);
    setSeed(778);
    const different = new MLP([3, 4, 1]).predict([[0.1, 0.2, 0.3]]);

    expect(same).toEqual(first);
    expect(different).not.toEqual(first);
  });

  test('pathfinding grid generation follows the seed', () => {
    const grid = () => JSON.stringify(createPathfindingProblem(8, 8, 3));

    setSeed(4321);
    const first = grid();
    setSeed(4321);
    const same = grid();
    setSeed(9999);
    const different = grid();

    expect(same).toEqual(first);
    expect(different).not.toEqual(first);
  });

  test('no component reachable here reintroduces an unseeded source', () => {
    // A direct Math.random() call anywhere in these paths would make the two runs diverge
    // even though the seed is identical, which is exactly what this catches.
    setSeed(2026);
    const runA = { mlp: new MLP([2, 3, 1]).predict([[0.4, 0.6]]), path: createPathfindingProblem(6, 6, 2) };
    setSeed(2026);
    const runB = { mlp: new MLP([2, 3, 1]).predict([[0.4, 0.6]]), path: createPathfindingProblem(6, 6, 2) };

    expect(JSON.stringify(runB)).toEqual(JSON.stringify(runA));
  });
});
