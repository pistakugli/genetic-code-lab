import path from 'path';
import { spawnSync } from 'child_process';

const repoRoot = path.resolve(__dirname, '..');
const tsNode = path.join(repoRoot, 'node_modules', '.bin', 'ts-node');
const cliPath = path.join(repoRoot, 'src', 'cli', 'index.ts');

function runCli(args: string[]) {
  return spawnSync(tsNode, [cliPath, ...args], {
    cwd: repoRoot,
    encoding: 'utf8',
    env: { ...process.env, FORCE_COLOR: '0' },
    maxBuffer: 16 * 1024 * 1024
  });
}

describe('CLI capabilities', () => {
  test('lists innovation capabilities as a first-class product surface', () => {
    const result = runCli(['capabilities', '--json']);

    expect(result.status).toBe(0);
    const capabilities = JSON.parse(result.stdout);
    expect(capabilities).toEqual(expect.arrayContaining([
      expect.objectContaining({
        id: 'threshold-adaptive-duel',
        status: 'production'
      }),
      expect.objectContaining({
        id: 'semantic-archive',
        status: 'production'
      }),
      expect.objectContaining({
        id: 'distributed-island-nn',
        status: 'production'
      }),
      expect.objectContaining({
        id: 'island-nn-archive',
        status: 'production'
      }),
      expect.objectContaining({
        id: 'gcl-hybrid-basis',
        status: 'production'
      }),
      expect.objectContaining({
        id: 'nn-deep-blackbox-relu-nd-region-sampler',
        status: 'production'
      }),
      expect.objectContaining({
        id: 'nn-noisy-adaptive-adversarial-deep-blackbox-relu-nd-region-sampler',
        status: 'production'
      }),
      expect.objectContaining({
        id: 'nn-adaptive-budget-noisy-deep-blackbox-relu-nd-region-sampler',
        status: 'production'
      }),
      expect.objectContaining({
        id: 'nn-nd-adjacency-geometry-deep-blackbox-relu-region-sampler',
        status: 'production'
      }),
      expect.objectContaining({
        id: 'nn-nd-adjacency-geometry-5d-sweep',
        status: 'production'
      }),
      expect.objectContaining({
        id: 'nn-nd-adjacency-geometry-6d-robustness',
        status: 'production'
      }),
      expect.objectContaining({
        id: 'nn-nd-adjacency-geometry-6d-baseline-court',
        status: 'production'
      }),
      expect.objectContaining({
        id: 'nn-active-subspace-ffn-decompiler',
        status: 'production'
      })
    ]));
  });

  test('shows proof commands for a single capability', () => {
    const result = runCli(['capabilities', 'threshold-adaptive-duel', '--json']);

    expect(result.status).toBe(0);
    const capability = JSON.parse(result.stdout);
    expect(capability).toMatchObject({
      id: 'threshold-adaptive-duel',
      status: 'production',
      run: {
        command: 'npm',
        args: ['run', 'verify:default-threshold-policy']
      }
    });
    expect(capability.proofs).toEqual(expect.arrayContaining([
      expect.objectContaining({
        args: ['run', 'release:promotion']
      })
    ]));
  });

  test('rejects unknown capabilities', () => {
    const result = runCli(['capability', 'missing-feature']);

    expect(result.status).toBe(1);
    expect(result.stderr).toContain('Unknown capability');
  });
});
