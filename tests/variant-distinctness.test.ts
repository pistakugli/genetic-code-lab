import { execFileSync } from 'child_process';
import fs from 'fs';
import os from 'os';
import path from 'path';

/**
 * A variant name claims a method. 153 variants across 51 groups share an environment and a
 * commit with a differently named peer, and every one of those runs happened over a dirty
 * working tree: `statusShort` recorded which files were modified and nothing about how, so a
 * run named adam-refine and one named orthogonal-cover, minutes apart over different edits to
 * the same files, left byte-identical provenance.
 *
 * `uncommittedDiffSha256` is what makes those runs distinguishable going forward. These tests
 * pin both halves: a real difference in the diff separates two variants, and an identical
 * recorded state does not stop being a collision just because the names differ.
 */
describe('variant distinctness verifier', () => {
  const repoRoot = path.resolve(__dirname, '..');
  const script = path.join(repoRoot, 'scripts', 'verify-variant-distinctness.ts');
  let root: string;

  const run = (extraEnv: Record<string, string> = {}): { status: number; output: string } => {
    try {
      const output = execFileSync('npx', ['ts-node', script], {
        cwd: repoRoot,
        encoding: 'utf8',
        env: { ...process.env, GCL_BENCHMARK_ROOT: root, ...extraEnv }
      });
      return { status: 0, output };
    } catch (error) {
      const failure = error as { status?: number; stdout?: string; stderr?: string };
      return { status: failure.status ?? 1, output: `${failure.stdout ?? ''}${failure.stderr ?? ''}` };
    }
  };

  const variant = (
    name: string,
    env: Record<string, string>,
    git: { commit: string; uncommittedDiffSha256?: string | null }
  ): void => {
    const dir = path.join(root, 'a-lane', name);
    fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(
      path.join(dir, 'manifest.json'),
      `${JSON.stringify({ lane: 'a-lane', variant: name, env, git }, null, 2)}\n`
    );
  };

  beforeEach(() => {
    root = fs.mkdtempSync(path.join(os.tmpdir(), 'gcl-distinctness-'));
  });

  afterEach(() => {
    fs.rmSync(root, { recursive: true, force: true });
  });

  test('variants that differ in a describing env key do not collide', () => {
    variant('alpha', { GCL_BENCHMARK_VARIANT: 'alpha', GCL_BLIND_SWIGLU_FIT_STEPS: '100' }, { commit: 'abc' });
    variant('beta', { GCL_BENCHMARK_VARIANT: 'beta', GCL_BLIND_SWIGLU_FIT_STEPS: '200' }, { commit: 'abc' });

    const result = run();
    expect(result.status).toBe(0);
    // The script prints a summary block and then a baseline block, so read the field rather
    // than parsing the whole stream as one object.
    expect(result.output).toMatch(/"collidingGroups": 0/);
  });

  test('the variant name alone cannot make two runs distinct', () => {
    variant('adam-refine', { GCL_BENCHMARK_VARIANT: 'adam-refine', GCL_BLIND_SWIGLU_FIT_STEPS: '100' }, { commit: 'abc' });
    variant('orthogonal-cover', { GCL_BENCHMARK_VARIANT: 'orthogonal-cover', GCL_BLIND_SWIGLU_FIT_STEPS: '100' }, { commit: 'abc' });

    const result = run();
    expect(result.status).toBe(1);
    expect(result.output).toContain('adam-refine');
    expect(result.output).toContain('only differ by name');
  });

  test('a different uncommitted diff separates two runs on the same commit', () => {
    const env = { GCL_BLIND_SWIGLU_FIT_STEPS: '100' };
    variant('adam-refine', { ...env, GCL_BENCHMARK_VARIANT: 'adam-refine' }, { commit: 'abc', uncommittedDiffSha256: 'aaa' });
    variant('orthogonal-cover', { ...env, GCL_BENCHMARK_VARIANT: 'orthogonal-cover' }, { commit: 'abc', uncommittedDiffSha256: 'bbb' });

    // This is the case the digest exists for: same commit, same env, genuinely different code.
    expect(run().status).toBe(0);
  });

  test('the same uncommitted diff on the same commit still collides', () => {
    const env = { GCL_BLIND_SWIGLU_FIT_STEPS: '100' };
    variant('one', { ...env, GCL_BENCHMARK_VARIANT: 'one' }, { commit: 'abc', uncommittedDiffSha256: 'aaa' });
    variant('two', { ...env, GCL_BENCHMARK_VARIANT: 'two' }, { commit: 'abc', uncommittedDiffSha256: 'aaa' });

    expect(run().status).toBe(1);
  });

  test('a variant that declares itself indistinguishable is disclosed, not counted', () => {
    const env = { GCL_BLIND_SWIGLU_FIT_STEPS: '100' };
    variant('one', { ...env, GCL_BENCHMARK_VARIANT: 'one' }, { commit: 'abc' });
    const dir = path.join(root, 'a-lane', 'two');
    fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(
      path.join(dir, 'manifest.json'),
      `${JSON.stringify({
        lane: 'a-lane',
        variant: 'two',
        env: { ...env, GCL_BENCHMARK_VARIANT: 'two' },
        git: { commit: 'abc' },
        variantDistinctness: { state: 'indistinguishable-from', peers: ['one'] }
      }, null, 2)}\n`
    );

    expect(run().status).toBe(0);
  });

  test('a recorded baseline holds the known collisions and only fails on new ones', () => {
    const env = { GCL_BLIND_SWIGLU_FIT_STEPS: '100' };
    variant('one', { ...env, GCL_BENCHMARK_VARIANT: 'one' }, { commit: 'abc' });
    variant('two', { ...env, GCL_BENCHMARK_VARIANT: 'two' }, { commit: 'abc' });
    expect(run({ GCL_VARIANT_DISTINCTNESS_BASELINE_WRITE: '1' }).status).toBe(0);
    expect(run().status).toBe(0);

    variant('three', { ...env, GCL_BENCHMARK_VARIANT: 'three' }, { commit: 'abc' });
    const result = run();
    expect(result.status).toBe(1);
    expect(result.output).toContain('new group');
  });
  test('two cases of one randomized sweep are told apart by what they were run against', () => {
    // A randomized sweep picks its layer and units from a seed, so its cases share every
    // environment variable and differ only in the report's own summary. Reading that as one
    // run under two names was a defect in this check, not in those artifacts.
    const env = { GCL_BLIND_SWIGLU_SWEEP_SEED: '7' };
    for (const [name, layer] of [['case-layer1', 1], ['case-layer19', 19]] as const) {
      const dir = path.join(root, 'a-lane', name);
      fs.mkdirSync(dir, { recursive: true });
      fs.writeFileSync(
        path.join(dir, 'manifest.json'),
        `${JSON.stringify({ lane: 'a-lane', variant: name, env: { ...env, GCL_BENCHMARK_VARIANT: name }, git: { commit: 'abc' } }, null, 2)}\n`
      );
      fs.writeFileSync(
        path.join(dir, 'report.json'),
        `${JSON.stringify({ summary: { layer, units: [layer * 10], atomCount: 1 }, solver: { kind: 'x' } }, null, 2)}\n`
      );
    }

    expect(run().status).toBe(0);
  });

  test('a sweep aggregate does not collide with the case it summarises', () => {
    const env = { GCL_BLIND_SWIGLU_SWEEP_SEED: '7', GCL_BENCHMARK_VARIANT: 'x' };
    const write = (name: string, report: unknown): void => {
      const dir = path.join(root, 'a-lane', name);
      fs.mkdirSync(dir, { recursive: true });
      fs.writeFileSync(path.join(dir, 'manifest.json'), `${JSON.stringify({ lane: 'a-lane', variant: name, env, git: { commit: 'abc' } }, null, 2)}\n`);
      fs.writeFileSync(path.join(dir, 'report.json'), `${JSON.stringify(report, null, 2)}\n`);
    };
    write('the-sweep', { summary: { layer: 1 }, cases: [{ pass: true }] });
    write('the-case', { summary: { layer: 1 }, solver: { kind: 'x' } });

    // Two levels of one run, not two methods.
    expect(run().status).toBe(0);
  });
});
