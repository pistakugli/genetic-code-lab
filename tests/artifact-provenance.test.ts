import { execFileSync } from 'child_process';
import fs from 'fs';
import os from 'os';
import path from 'path';

/**
 * Ten variant directories in the release tree were completely empty, and two carried a
 * report.json with no manifest.json. An empty directory reads as a lane that ran; an
 * unmanifested report reads as evidence while recording neither the command that produced
 * it nor when. Nothing detected either state, because every other check starts from
 * report.json and asks about its contents rather than about the shape of the tree.
 *
 * The verifier is run here against a scratch tree so the assertions are about its rules,
 * not about whatever the real release tree happens to hold today.
 */
describe('artifact provenance verifier', () => {
  const repoRoot = path.resolve(__dirname, '..');
  const script = path.join(repoRoot, 'scripts', 'verify-artifact-provenance.ts');
  let root: string;

  const run = (): { status: number; output: string } => {
    try {
      const output = execFileSync('npx', ['ts-node', script], {
        cwd: repoRoot,
        encoding: 'utf8',
        env: { ...process.env, GCL_BENCHMARK_ROOT: root }
      });
      return { status: 0, output };
    } catch (error) {
      const failure = error as { status?: number; stdout?: string; stderr?: string };
      return { status: failure.status ?? 1, output: `${failure.stdout ?? ''}${failure.stderr ?? ''}` };
    }
  };

  const makeVariant = (relative: string, files: Record<string, unknown>): void => {
    const dir = path.join(root, relative);
    fs.mkdirSync(dir, { recursive: true });
    for (const [name, body] of Object.entries(files)) {
      fs.writeFileSync(path.join(dir, name), `${JSON.stringify(body, null, 2)}\n`);
    }
  };

  beforeEach(() => {
    root = fs.mkdtempSync(path.join(os.tmpdir(), 'gcl-provenance-'));
  });

  afterEach(() => {
    fs.rmSync(root, { recursive: true, force: true });
  });

  test('a complete variant passes', () => {
    makeVariant('a-lane/good-variant', { 'report.json': { summary: { pass: true } }, 'manifest.json': { status: 'pass' } });

    const result = run();
    expect(result.status).toBe(0);
    expect(JSON.parse(result.output.slice(result.output.indexOf('{'))).violations).toBe(0);
  });

  test('an empty variant directory fails', () => {
    fs.mkdirSync(path.join(root, 'a-lane/empty-variant'), { recursive: true });
    // Backdate it past the in-flight window, otherwise it reads as a run in progress.
    const old = new Date(Date.now() - 60 * 60 * 1000);
    fs.utimesSync(path.join(root, 'a-lane/empty-variant'), old, old);

    const result = run();
    expect(result.status).toBe(1);
    expect(result.output).toContain('empty-directory');
  });

  test('a report with no manifest fails', () => {
    makeVariant('a-lane/unmanifested', { 'report.json': { summary: { pass: true } } });

    const result = run();
    expect(result.status).toBe(1);
    expect(result.output).toContain('missing-manifest');
  });

  test('a per-case report under a manifested variant is covered by its parent', () => {
    makeVariant('a-lane/court', { 'report.json': { summary: { pass: true } }, 'manifest.json': { status: 'pass' } });
    makeVariant('a-lane/court/cases/case-one', { 'report.json': { summary: { pass: true } } });

    const result = run();
    expect(result.status).toBe(0);
  });

  test('a directory a run is currently rewriting is not counted against the tree', () => {
    // A run deletes its output directory before writing into it, so a scan racing one sees
    // an empty directory that is neither abandoned nor a defect.
    fs.mkdirSync(path.join(root, 'a-lane/in-flight'), { recursive: true });

    const result = run();
    expect(result.status).toBe(0);
    expect(result.output).toContain('in flight, not counted');
  });
});
