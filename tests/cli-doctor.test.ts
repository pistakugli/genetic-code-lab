import path from 'path';
import { spawnSync } from 'child_process';

const repoRoot = path.resolve(__dirname, '..');
const tsNode = path.join(repoRoot, 'node_modules', '.bin', 'ts-node');
const cliPath = path.join(repoRoot, 'src', 'cli', 'index.ts');

describe('CLI doctor', () => {
  test('checks compiled artifact, example config, parser, and config schema', () => {
    const result = spawnSync(tsNode, [cliPath, 'doctor'], {
      cwd: repoRoot,
      encoding: 'utf8',
      env: { ...process.env, FORCE_COLOR: '0' }
    });

    expect(result.status).toBe(0);
    expect(result.stdout).toContain('PASS compiled CLI artifact');
    expect(result.stdout).toContain('PASS example config');
    expect(result.stdout).toContain('PASS math expression parser');
    expect(result.stdout).toContain('PASS config runner schema');
  });
});
