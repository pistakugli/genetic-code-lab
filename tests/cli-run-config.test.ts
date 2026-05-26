import fs from 'fs';
import os from 'os';
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

describe('CLI config runner', () => {
  test('runs a math config and writes a result file', () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'gcl-cli-config-'));
    const configPath = path.join(dir, 'math.json');
    const outputPath = path.join(dir, 'result.json');
    fs.writeFileSync(configPath, JSON.stringify({
      problem: {
        type: 'math',
        target: 'x*x + 1',
        testCount: 6
      },
      evolution: {
        generations: 1,
        populationSize: 10,
        seed: 7
      },
      output: outputPath
    }));

    const result = runCli(['run', configPath]);
    expect(result.status).toBe(0);
    expect(result.stdout).toContain('Evolution Results');
    expect(fs.existsSync(outputPath)).toBe(true);
    const output = JSON.parse(fs.readFileSync(outputPath, 'utf8'));
    expect(output.schemaVersion).toBe(1);
    expect(output.summary).toMatchObject({
      success: expect.any(Boolean),
      bestFitness: expect.any(Number),
      evaluations: expect.any(Number)
    });
    expect(output.cli).toMatchObject({ command: 'run' });
    expect(output.problem).toMatchObject({
      type: 'math',
      target: 'x*x + 1',
      testCount: 6
    });
    expect(output.solution).toMatchObject({
      expression: expect.any(String)
    });
    expect(typeof output.bestFitness).toBe('number');
    expect(output.generations).toBeGreaterThanOrEqual(0);
  });

  test('rejects unsupported problem types with non-zero exit', () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'gcl-cli-config-'));
    const configPath = path.join(dir, 'bad.json');
    fs.writeFileSync(configPath, JSON.stringify({ problem: { type: 'unknown' } }));

    const result = runCli(['run', configPath]);
    expect(result.status).toBe(1);
    expect(result.stderr).toContain('problem.type');
  });
});
