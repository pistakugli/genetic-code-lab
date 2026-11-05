import { ChildProcessWithoutNullStreams, spawn } from 'node:child_process';
import readline from 'node:readline';
import path from 'node:path';

export type GfdCudaDiagnostics = {
  totalOracleCalls: number;
  sampleCount: number;
  trainingSamples: number;
  validationSamples: number;
  restartsAttempted: number;
  selectedRestart: number;
  trainingMae: number | null;
  validationMae: number | null;
  reconstructionComplete: boolean;
  computeDevice: string;
  cudaMemoryAllocatedBytes: number;
  trainingTimeMs: number;
  dtype: string;
};

export type GfdCudaModel = {
  dimension: number;
  baseIntercept: number;
  baseWeights: number[];
  terms: Array<{ normal: number[]; offset: number; readout: number }>;
  diagnostics: GfdCudaDiagnostics;
  reconstructionComplete: boolean;
};

export type GfdCudaRequest = {
  inputs: number[][];
  outputs: number[];
  dimension: number;
  termCount: number;
  randomSeed: number;
  restarts: number;
  optimizationSteps: number;
  batchSize: number;
  profileOutputLayer: boolean;
};

type WorkerReady = {
  ready: true;
  computeDevice: string;
  deviceName: string;
  torchVersion: string;
  cudaVersion: string;
};

type WorkerResponse = { ok: true; model: GfdCudaModel } | { ok: false; error: string };

export class GfdCudaClient {
  private readonly child: ChildProcessWithoutNullStreams;
  private readonly lines: readline.Interface;
  private readonly pending: Array<{
    resolve: (line: string) => void;
    reject: (error: Error) => void;
  }> = [];
  private readonly stderr: string[] = [];
  private closed = false;
  private terminalError: Error | null = null;
  device: WorkerReady = {
    ready: true,
    computeDevice: '',
    deviceName: '',
    torchVersion: '',
    cudaVersion: ''
  };

  private constructor(child: ChildProcessWithoutNullStreams) {
    this.child = child;
    this.lines = readline.createInterface({ input: child.stdout });
    this.lines.on('line', line => {
      const waiter = this.pending.shift();
      if (waiter) waiter.resolve(line);
    });
    child.stderr.on('data', chunk => {
      this.stderr.push(String(chunk));
      while (this.stderr.join('').length > 8192) this.stderr.shift();
    });
    child.once('error', error => this.fail(error));
    child.once('exit', code => {
      if (code !== 0 || !this.closed) {
        this.fail(new Error(`GFD-VP CUDA worker exited with code ${code}: ${this.stderr.join('')}`));
      }
    });
  }

  static async start(repositoryRoot: string): Promise<GfdCudaClient> {
    const python = path.join(repositoryRoot, '.gcl-python-gpu/bin/python');
    const worker = path.join(repositoryRoot, 'scripts/gfd-vp-cuda-worker.py');
    const child = spawn(python, [worker], { cwd: repositoryRoot, stdio: 'pipe' });
    const client = new GfdCudaClient(child);
    const line = await client.nextLine();
    const parsed = JSON.parse(line) as WorkerReady;
    if (parsed.ready !== true || parsed.computeDevice !== 'cuda:0') {
      child.kill();
      throw new Error(`GFD-VP CUDA worker did not initialize a CUDA device: ${line}`);
    }
    client.device = parsed;
    return client;
  }

  async fit(request: GfdCudaRequest): Promise<GfdCudaModel> {
    if (this.closed) throw new Error('GFD-VP CUDA worker is closed');
    const responseLine = this.nextLine();
    const accepted = this.child.stdin.write(`${JSON.stringify(request)}\n`);
    if (!accepted) {
      await new Promise<void>(resolve => this.child.stdin.once('drain', resolve));
    }
    const response = JSON.parse(await responseLine) as WorkerResponse;
    if (!response.ok) {
      throw new Error(`GFD-VP CUDA worker failed: ${response.error}`);
    }
    if (response.model.diagnostics.computeDevice !== this.device.computeDevice
      || response.model.diagnostics.cudaMemoryAllocatedBytes <= 0) {
      throw new Error('GFD-VP worker returned evidence inconsistent with its CUDA runtime');
    }
    return response.model;
  }

  async close(): Promise<void> {
    if (this.closed) return;
    this.closed = true;
    const exited = new Promise<void>((resolve, reject) => {
      this.child.once('exit', code => {
        if (code === 0) resolve();
        else reject(new Error(`GFD-VP CUDA worker exited with code ${code}: ${this.stderr.join('')}`));
      });
    });
    this.child.stdin.end(`${JSON.stringify({ action: 'close' })}\n`);
    await exited;
    this.lines.close();
  }

  private nextLine(): Promise<string> {
    return new Promise((resolve, reject) => {
      if (this.terminalError) {
        reject(this.terminalError);
        return;
      }
      this.pending.push({ resolve, reject });
    });
  }

  private fail(error: Error): void {
    if (this.terminalError) return;
    this.terminalError = error;
    for (const waiter of this.pending.splice(0)) waiter.reject(error);
  }
}
