#!/usr/bin/env ts-node

import fs from 'fs';
import os from 'os';
import path from 'path';
import { spawnSync } from 'child_process';

function run(command: string, args: string[], cwd: string, env: NodeJS.ProcessEnv = process.env): string {
  const result = spawnSync(command, args, {
    cwd,
    env,
    encoding: 'utf8',
    maxBuffer: 64 * 1024 * 1024
  });
  if (result.status !== 0) {
    if (result.stdout) process.stdout.write(result.stdout);
    if (result.stderr) process.stderr.write(result.stderr);
    throw new Error(`${command} ${args.join(' ')} failed with exit code ${result.status ?? 'unknown'}`);
  }
  return result.stdout || '';
}

function main(): void {
  const repoRoot = process.cwd();
  const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'gcl-install-check-'));
  const packDir = path.join(tempRoot, 'pack');
  const appDir = path.join(tempRoot, 'app');
  fs.mkdirSync(packDir, { recursive: true });
  fs.mkdirSync(appDir, { recursive: true });

  const packRaw = run('npm', ['pack', '--json', '--pack-destination', packDir], repoRoot);
  const packReport = JSON.parse(packRaw)[0] as { filename: string };
  if (!packReport?.filename) throw new Error('npm pack did not report a filename');
  const tarball = path.join(packDir, packReport.filename);
  if (!fs.existsSync(tarball)) throw new Error(`Packed tarball not found: ${tarball}`);

  run('npm', ['init', '-y'], appDir);
  run('npm', ['install', '--ignore-scripts', tarball], appDir);

  const bin = path.join(appDir, 'node_modules', '.bin', process.platform === 'win32' ? 'gclab.cmd' : 'gclab');
  const doctor = run(bin, ['doctor'], appDir, { ...process.env, FORCE_COLOR: '0' });
  if (!doctor.includes('Quick doctor passed')) throw new Error('installed gclab doctor did not pass');

  const apiCheck = run('node', ['-e', "const g=require('genetic-code-lab'); const fn=g.compileMathExpression('x*x+1'); if (fn({x:3,x0:3})!==10) process.exit(2); console.log('api-ok')"], appDir);
  if (!apiCheck.includes('api-ok')) throw new Error('installed API import check failed');
  const exportCheck = run('node', ['-e', `
    const pkg = require('genetic-code-lab/package.json');
    if (!pkg.exports || !pkg.exports['.']) process.exit(2);
    try {
      require('genetic-code-lab/dist/core/evolution');
      process.exit(3);
    } catch (error) {
      if (error.code !== 'ERR_PACKAGE_PATH_NOT_EXPORTED') throw error;
    }
    console.log('exports-ok');
  `], appDir);
  if (!exportCheck.includes('exports-ok')) throw new Error('installed package exports check failed');

  const consumerPath = path.join(appDir, 'consumer.ts');
  const tsconfigPath = path.join(appDir, 'tsconfig.json');
  fs.writeFileSync(consumerPath, `
    import { EvolutionEngine, createMathProblem, compileMathExpression } from 'genetic-code-lab';

    const compiled = compileMathExpression('x*x+1');
    const value: number = compiled({ x: 2, x0: 2 });
    const problem = createMathProblem((inputs: number[]) => inputs[0] + value, 1, 3);
    const engine = new EvolutionEngine(problem, {
      maxGenerations: 1,
      targetFitness: 0.9,
      stagnationLimit: 1,
      population: {
        size: 10,
        eliteCount: 1,
        crossoverRate: 0.7,
        mutationRate: 0.2,
        maxGenomeDepth: 4,
        maxGenomeSize: 20,
        tournamentSize: 2
      }
    });
    void engine;
  `);
  fs.writeFileSync(tsconfigPath, JSON.stringify({
    compilerOptions: {
      target: 'ES2022',
      module: 'commonjs',
      moduleResolution: 'node',
      strict: true,
      esModuleInterop: true,
      skipLibCheck: true
    },
    include: ['consumer.ts']
  }, null, 2));
  const tsc = path.join(repoRoot, 'node_modules', '.bin', process.platform === 'win32' ? 'tsc.cmd' : 'tsc');
  run(tsc, ['--noEmit', '-p', tsconfigPath], appDir);

  const mathOut = path.join(tempRoot, 'math-result.json');
  run(bin, ['math', '--generations', '1', '--population', '10', '--seed', '1', '--target', 'x*x+1', '--output', mathOut], appDir, { ...process.env, FORCE_COLOR: '0' });
  const resultJson = JSON.parse(fs.readFileSync(mathOut, 'utf8')) as { schemaVersion?: number; cli?: { command?: string }; problem?: { type?: string } };
  if (resultJson.schemaVersion !== 1 || resultJson.cli?.command !== 'math' || resultJson.problem?.type !== 'math') {
    throw new Error('installed CLI math output schema check failed');
  }

  const capabilitiesRaw = run(bin, ['capabilities', '--json'], appDir, { ...process.env, FORCE_COLOR: '0' });
  const capabilities = JSON.parse(capabilitiesRaw) as Array<{ id?: string; status?: string }>;
  if (!capabilities.some(item => item.id === 'threshold-adaptive-duel' && item.status === 'production')) {
    throw new Error('installed CLI capabilities registry missing threshold-adaptive-duel production capability');
  }

  const serverPath = path.join(appDir, 'node_modules', 'genetic-code-lab', 'dist', 'server', 'index.js');
  const server = spawnSync('node', ['-e', `
    const { spawn } = require('child_process');
    const http = require('http');
    const child = spawn(process.execPath, [${JSON.stringify(serverPath)}], { env: { ...process.env, GCLAB_PORT: '4301' }, stdio: 'ignore' });
    const started = Date.now();
    function fail(msg) { child.kill(); console.error(msg); process.exit(1); }
    function poll() {
      http.get('http://127.0.0.1:4301/api/health', res => {
        let data = '';
        res.on('data', chunk => data += chunk);
        res.on('end', () => {
          child.kill();
          try {
            const parsed = JSON.parse(data);
            if (parsed.status !== 'ok') fail('bad health payload');
            console.log('server-ok');
          } catch (error) { fail(String(error)); }
        });
      }).on('error', () => {
        if (Date.now() - started > 5000) fail('server did not start');
        setTimeout(poll, 150);
      });
    }
    poll();
  `], { cwd: appDir, encoding: 'utf8', maxBuffer: 16 * 1024 * 1024 });
  if (server.status !== 0 || !server.stdout.includes('server-ok')) {
    if (server.stdout) process.stdout.write(server.stdout);
    if (server.stderr) process.stderr.write(server.stderr);
    throw new Error('installed server health check failed');
  }

  console.log(JSON.stringify({
    pass: true,
    tarball,
    checks: ['install', 'gclab doctor', 'api import', 'package exports', 'typescript declarations', 'cli math output', 'capability registry', 'server health']
  }, null, 2));
}

try {
  main();
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
}
