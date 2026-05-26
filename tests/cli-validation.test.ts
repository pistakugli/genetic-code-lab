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

describe('CLI option validation', () => {
  test('math rejects non-numeric generations before evolution starts', () => {
    const result = runCli(['math', '--generations', 'abc', '--population', '10']);
    expect(result.status).toBe(1);
    expect(result.stderr).toContain('generations must be a positive integer');
  });

  test('sort rejects invalid array size', () => {
    const result = runCli(['sort', '--size', '0', '--generations', '1', '--population', '10']);
    expect(result.status).toBe(1);
    expect(result.stderr).toContain('size must be a positive integer');
  });

  test('path rejects invalid dimensions', () => {
    const result = runCli(['path', '--width', '-1', '--height', '8', '--generations', '1', '--population', '10']);
    expect(result.status).toBe(1);
    expect(result.stderr).toContain('width must be a positive integer');
  });
});
