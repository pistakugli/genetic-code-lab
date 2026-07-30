import { generateTranscendentalSeeds } from '../src/core/seed-templates';
import { PostfixBridge } from '../src/core/postfix-bridge';
import { setSeed } from '../src/utils/helpers';

/**
 * generateTranscendentalSeeds was exported and then imported by nothing: dead code holding
 * damped-oscillator and decay skeletons that the rational and trigonometric scaffolds do
 * not cover. It also drew its constants from unseeded Math.random(), which would have
 * broken run-to-run reproducibility the moment it was connected.
 *
 * It is now wired into generation-0 injection behind GCL_TRANSCENDENTAL_SEED. These tests
 * cover the two things that make that wiring real rather than nominal: the generator is
 * reproducible under a seed, and every skeleton it produces survives the tree-to-opcode
 * bake that the injection path depends on. Without the second, an enabled flag would
 * silently inject nothing and "no measured change" would mean "not connected" rather than
 * "connected and neutral".
 */
describe('generateTranscendentalSeeds', () => {
  const varMap = ['x'];

  test('is reproducible under a seed', () => {
    setSeed(4242);
    const first = generateTranscendentalSeeds(5);
    setSeed(4242);
    const second = generateTranscendentalSeeds(5);

    const shape = (genomes: ReturnType<typeof generateTranscendentalSeeds>) =>
      genomes.map(genome => genome.genes.map(gene => `${gene.type}:${gene.operation}`).join('|'));
    const constants = (genomes: ReturnType<typeof generateTranscendentalSeeds>) =>
      genomes.map(genome => PostfixBridge.bake(genome, varMap).constants);

    expect(shape(first)).toEqual(shape(second));
    expect(constants(first)).toEqual(constants(second));
  });

  test('different seeds produce different constants, so the seeds are not fixed answers', () => {
    setSeed(1);
    const a = generateTranscendentalSeeds(5).map(genome => PostfixBridge.bake(genome, varMap).constants);
    setSeed(2);
    const b = generateTranscendentalSeeds(5).map(genome => PostfixBridge.bake(genome, varMap).constants);

    expect(JSON.stringify(a)).not.toEqual(JSON.stringify(b));
  });

  test('every generated skeleton bakes into opcodes, which is what the injection path needs', () => {
    setSeed(99);
    const genomes = generateTranscendentalSeeds(8);

    expect(genomes.length).toBeGreaterThan(0);
    for (const genome of genomes) {
      const baked = PostfixBridge.bake(genome, varMap);
      expect(baked.opcodes.length).toBeGreaterThan(0);
      expect(baked.data.length).toBe(baked.opcodes.length);
      expect(baked.opcodes.every(op => Number.isInteger(op))).toBe(true);
      expect(baked.constants.every(value => Number.isFinite(value))).toBe(true);
    }
  });

  test('respects the requested count', () => {
    setSeed(7);
    expect(generateTranscendentalSeeds(3).length).toBe(3);
    setSeed(7);
    expect(generateTranscendentalSeeds(1).length).toBe(1);
  });
});
