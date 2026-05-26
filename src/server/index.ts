#!/usr/bin/env node

/**
 * Genetic Code Lab — HTTP API Server
 * Exposes the GP engine as a REST service for real-world symbolic regression.
 *
 * Endpoints:
 *   POST /api/evolve           — Start a new evolution run from JSON test cases
 *   POST /api/evolve/csv       — Upload a CSV; the server parses it and evolves
 *   GET  /api/status/:runId    — Poll the status / result of an ongoing run
 *   GET  /api/health           — Liveness probe
 *   GET  /                     — Dashboard UI
 */

import http from 'http';
import fs from 'fs';
import path from 'path';
import { EvolutionEngine } from '../core/evolution';
import { DiscoveryAgent } from '../agents/discovery-agent';
import { AsyncAgentBridge } from '../core/async-agent-bridge';
import { createSymbolicRegressionProblem } from '../solvers/math-expression';
import { createMajorityProblem, createMuxProblem } from '../solvers/boolean-library';
import { ProblemDefinition, EvolutionConfig } from '../core/types';
import { PostfixBridge } from '../core/postfix-bridge';

// ─── Run store ──────────────────────────────────────────────────────────────

interface RunRecord {
  id: string;
  status: 'running' | 'completed' | 'failed';
  startedAt: string;
  finishedAt?: string;
  progress: { gen: number; bestFitness: number; bestSize: number };
  result?: {
    fitness: number;
    expression: string;
    generations: number;
    timeMs: number;
  };
  error?: string;
}

const runs = new Map<string, RunRecord>();

// ─── CSV parsing ────────────────────────────────────────────────────────────

function parseCSV(raw: string): { headers: string[]; rows: number[][] } {
  const lines = raw.trim().split(/\r?\n/).filter(l => l.trim().length > 0);
  if (lines.length < 2) throw new Error('CSV must have a header row and at least one data row');

  const headers = lines[0].split(',').map(h => h.trim());
  const rows: number[][] = [];
  for (let i = 1; i < lines.length; i++) {
    const values = lines[i].split(',').map(v => {
      const n = parseFloat(v.trim());
      return Number.isFinite(n) ? n : 0;
    });
    if (values.length === headers.length) rows.push(values);
  }
  return { headers, rows };
}

// ─── Problem builder from CSV ───────────────────────────────────────────────

function buildProblemFromCSV(
  headers: string[],
  rows: number[][],
  targetColumn?: string,
  featureColumns?: string[],
  maxRows?: number
): {
  problem: ProblemDefinition<number[], number>;
  targetCol: string;
  featureCols: string[];
} {
  const targetIdx = targetColumn
    ? headers.indexOf(targetColumn)
    : headers.length - 1;
  if (targetIdx < 0) throw new Error(`Target column "${targetColumn}" not found in CSV headers`);

  const featureIdxs = featureColumns
    ? featureColumns.map(f => {
        const idx = headers.indexOf(f);
        if (idx < 0) throw new Error(`Feature column "${f}" not found`);
        return idx;
      })
    : headers.map((_, i) => i).filter(i => i !== targetIdx);

  // Optionally subsample
  let data = rows;
  if (maxRows && data.length > maxRows) {
    // Stratified shuffle-sample
    const step = Math.max(1, Math.floor(data.length / maxRows));
    data = data.filter((_, i) => i % step === 0).slice(0, maxRows);
  }

  const dataPoints = data.map(row => ({
    x: row[featureIdxs[0]],
    y: row[targetIdx]
  }));

  // For multi-variate, we build a custom problem
  if (featureIdxs.length === 1) {
    return {
      problem: createSymbolicRegressionProblem(dataPoints),
      targetCol: headers[targetIdx],
      featureCols: [headers[featureIdxs[0]]]
    };
  }

  // Multi-variate: build problem manually
  const testCases = data.map(row => ({
    input: featureIdxs.map(fi => row[fi]),
    expectedOutput: row[targetIdx],
    weight: 1
  }));

  const { createMathTerminals, mathFitnessFunction, MATH_FUNCTIONS } = require('../solvers/math-expression');
  const variables = featureIdxs.map((_, i) => `x${i}`);

  return {
    problem: {
      name: 'CSV Symbolic Regression',
      description: `Evolving f(${variables.join(',')}) → ${headers[targetIdx]}`,
      functionSet: MATH_FUNCTIONS.slice(0, 8),
      terminalSet: createMathTerminals(variables),
      testCases,
      fitnessFunction: mathFitnessFunction
    },
    targetCol: headers[targetIdx],
    featureCols: featureIdxs.map(i => headers[i])
  };
}

// ─── Evolution runner ───────────────────────────────────────────────────────

async function startEvolution(
  runId: string,
  problem: ProblemDefinition<number[], number>,
  config: Partial<EvolutionConfig>
): Promise<void> {
  const record = runs.get(runId)!;

  try {
    const agent = new DiscoveryAgent();
    const bridge = new AsyncAgentBridge(agent);
    const engine = new EvolutionEngine(problem, {
      population: {
        size: 300,
        eliteCount: 12,
        maxGenomeSize: 200,
        crossoverRate: 0.82,
        mutationRate: 0.2,
        maxGenomeDepth: 10,
        tournamentSize: 5,
        ...config.population
      },
      maxGenerations: config.maxGenerations || 120,
      targetFitness: config.targetFitness || 0.995,
      selectionMethod: config.selectionMethod || 'tournament',
      seed: config.seed,
      ...config
    });

    engine.setDiscoveryAgent(agent);
    (engine as any).agentBridge = bridge;

    engine.onStats = (stats: any) => {
      record.progress = {
        gen: stats.gen,
        bestFitness: stats.bestFitness,
        bestSize: stats.bestSize
      };
    };

    const result = await engine.evolve();

    // Extract readable expression
    let expression = '(unable to decode)';
    try {
      const best = result.bestGenome;
      if (best && best.genes && best.genes.length > 0) {
        const { genomeToString } = require('../core/genome');
        expression = genomeToString(best);
      }
    } catch { /* ignore decode failures */ }

    record.status = 'completed';
    record.finishedAt = new Date().toISOString();
    record.result = {
      fitness: result.bestGenome.fitness,
      expression,
      generations: result.generationsRun,
      timeMs: result.totalTimeMs
    };
  } catch (err: any) {
    record.status = 'failed';
    record.finishedAt = new Date().toISOString();
    record.error = err.message || String(err);
  }
}

// ─── Request body reader ────────────────────────────────────────────────────

function readBody(req: http.IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    req.on('data', (c: Buffer) => chunks.push(c));
    req.on('end', () => resolve(Buffer.concat(chunks).toString()));
    req.on('error', reject);
  });
}

// ─── JSON helpers ───────────────────────────────────────────────────────────

function json(res: http.ServerResponse, data: unknown, status = 200) {
  res.writeHead(status, {
    'Content-Type': 'application/json',
    'Access-Control-Allow-Origin': '*'
  });
  res.end(JSON.stringify(data, null, 2));
}

function error(res: http.ServerResponse, msg: string, status = 400) {
  json(res, { error: msg }, status);
}

// ─── Dashboard HTML ─────────────────────────────────────────────────────────

function serveDashboard(res: http.ServerResponse) {
  const htmlPath = [
    path.join(__dirname, 'dashboard.html'),
    path.join(__dirname, '..', '..', 'src', 'server', 'dashboard.html')
  ].find(candidate => fs.existsSync(candidate));
  if (htmlPath) {
    res.writeHead(200, { 'Content-Type': 'text/html' });
    res.end(fs.readFileSync(htmlPath, 'utf-8'));
  } else {
    res.writeHead(200, { 'Content-Type': 'text/html' });
    res.end(`<!DOCTYPE html>
<html><head><title>GCLab Server</title></head>
<body style="font-family:sans-serif;padding:2rem">
  <h1>🧬 Genetic Code Lab — API Server</h1>
  <p>Server is running. Use the API endpoints:</p>
  <ul>
    <li><code>POST /api/evolve/csv</code> — Upload CSV for symbolic regression</li>
    <li><code>GET /api/status/:id</code> — Check run status</li>
    <li><code>GET /api/health</code> — Health check</li>
  </ul>
</body></html>`);
  }
}

// ─── HTTP Server ────────────────────────────────────────────────────────────

const PORT = parseInt(process.env.GCLAB_PORT || '4200', 10);

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url || '/', `http://localhost:${PORT}`);
  const pathname = url.pathname;

  // CORS pre-flight
  if (req.method === 'OPTIONS') {
    res.writeHead(204, {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type'
    });
    return res.end();
  }

  // ── GET / ─ Dashboard
  if (req.method === 'GET' && pathname === '/') {
    return serveDashboard(res);
  }

  // ── GET /api/health
  if (req.method === 'GET' && pathname === '/api/health') {
    return json(res, {
      status: 'ok',
      engine: 'genetic-code-lab',
      version: '1.0.0',
      activeRuns: [...runs.values()].filter(r => r.status === 'running').length,
      totalRuns: runs.size,
      uptime: process.uptime()
    });
  }

  // ── GET /api/status/:id
  if (req.method === 'GET' && pathname.startsWith('/api/status/')) {
    const runId = pathname.split('/api/status/')[1];
    const record = runs.get(runId);
    if (!record) return error(res, 'Run not found', 404);
    return json(res, record);
  }

  // ── GET /api/runs — list all runs
  if (req.method === 'GET' && pathname === '/api/runs') {
    return json(res, [...runs.values()].reverse().slice(0, 50));
  }

  // ── POST /api/evolve/csv
  if (req.method === 'POST' && pathname === '/api/evolve/csv') {
    try {
      const body = await readBody(req);
      let csvData: string;
      let config: any = {};

      const contentType = req.headers['content-type'] || '';

      if (contentType.includes('application/json')) {
        const payload = JSON.parse(body);
        csvData = payload.csv;
        config = payload.config || {};
      } else {
        // Raw CSV body
        csvData = body;
      }

      if (!csvData || csvData.trim().length === 0) {
        return error(res, 'No CSV data provided');
      }

      const { headers, rows } = parseCSV(csvData);
      const { problem, targetCol, featureCols } = buildProblemFromCSV(
        headers,
        rows,
        config.targetColumn,
        config.featureColumns,
        config.maxRows || 100
      );

      const runId = `run_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
      const record: RunRecord = {
        id: runId,
        status: 'running',
        startedAt: new Date().toISOString(),
        progress: { gen: 0, bestFitness: 0, bestSize: 0 }
      };
      runs.set(runId, record);

      // Fire and forget — client polls /api/status/:id
      startEvolution(runId, problem, config).catch(() => {});

      return json(res, {
        runId,
        status: 'running',
        target: targetCol,
        features: featureCols,
        dataRows: rows.length,
        pollUrl: `/api/status/${runId}`
      }, 202);

    } catch (err: any) {
      return error(res, err.message || 'Failed to process CSV');
    }
  }

  // ── POST /api/evolve/file — Read CSV from a local file path
  if (req.method === 'POST' && pathname === '/api/evolve/file') {
    try {
      const body = JSON.parse(await readBody(req));
      const filePath = body.path;
      if (!filePath) return error(res, 'Missing "path" field');
      if (!fs.existsSync(filePath)) return error(res, `File not found: ${filePath}`, 404);

      const csvData = fs.readFileSync(filePath, 'utf-8');
      const { headers, rows } = parseCSV(csvData);
      const { problem, targetCol, featureCols } = buildProblemFromCSV(
        headers,
        rows,
        body.targetColumn,
        body.featureColumns,
        body.maxRows || 100
      );

      const runId = `run_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
      const record: RunRecord = {
        id: runId,
        status: 'running',
        startedAt: new Date().toISOString(),
        progress: { gen: 0, bestFitness: 0, bestSize: 0 }
      };
      runs.set(runId, record);

      startEvolution(runId, problem, body.config || {}).catch(() => {});

      return json(res, {
        runId,
        status: 'running',
        target: targetCol,
        features: featureCols,
        dataRows: rows.length,
        pollUrl: `/api/status/${runId}`
      }, 202);

    } catch (err: any) {
      return error(res, err.message || 'Failed to read file');
    }
  }

  // ── 404
  error(res, 'Not found', 404);
});

server.listen(PORT, () => {
  console.log(`\n🧬 Genetic Code Lab — API Server`);
  console.log(`   ├─ Port:      ${PORT}`);
  console.log(`   ├─ Dashboard: http://localhost:${PORT}/`);
  console.log(`   ├─ Health:    http://localhost:${PORT}/api/health`);
  console.log(`   └─ Evolve:    POST http://localhost:${PORT}/api/evolve/csv\n`);
});
