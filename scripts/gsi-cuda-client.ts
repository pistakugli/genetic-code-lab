import { ChildProcessWithoutNullStreams, spawn } from 'node:child_process';
import readline from 'node:readline';
import path from 'node:path';
import {
  GlobalSpectralMomentBackend,
  GlobalSpectralMomentBackendResult,
  GlobalSpectralMomentSamples
} from '../src/solvers/global-spectral-moment-inversion';

type WorkerReady = {
  ready: true;
  computeDevice: string;
  deviceName: string;
  torchVersion: string;
  cudaVersion: string;
};

type WorkerModel = GlobalSpectralMomentBackendResult;

type WorkerResponse = { ok: true; model: WorkerModel } | { ok: false; error: string };

export class GsiCudaClient implements GlobalSpectralMomentBackend {
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
        this.fail(new Error(`GSMI CUDA worker exited with code ${code}: ${this.stderr.join('')}`));
      }
    });
  }

  static async start(repositoryRoot: string): Promise<GsiCudaClient> {
    const python = path.join(repositoryRoot, '.gcl-python-gpu/bin/python');
    const worker = path.join(repositoryRoot, 'scripts/gsi-spectral-worker.py');
    const child = spawn(python, [worker], { cwd: repositoryRoot, stdio: 'pipe' });
    const client = new GsiCudaClient(child);
    const line = await client.nextLine();
    const parsed = JSON.parse(line) as WorkerReady;
    if (parsed.ready !== true || parsed.computeDevice !== 'cuda:0') {
      child.kill();
      throw new Error(`GSMI worker did not initialize CUDA: ${line}`);
    }
    client.device = parsed;
    return client;
  }

  async invert(
    samples: GlobalSpectralMomentSamples,
    config: {
      dimension: number;
      sampleSigma: number;
      maximumAbsoluteInput: number;
      momentOrder: 'second-and-third' | 'second-only';
      randomSeed?: number;
    }
  ): Promise<GlobalSpectralMomentBackendResult> {
    if (this.closed) throw new Error('GSMI CUDA worker is closed');
    const responseLine = this.nextLine();
    const accepted = this.child.stdin.write(`${JSON.stringify({
      inputs: samples.inputs,
      outputs: samples.outputs,
      dimension: config.dimension,
      sampleSigma: config.sampleSigma,
      maximumAbsoluteInput: config.maximumAbsoluteInput,
      momentOrder: config.momentOrder,
      randomSeed: config.randomSeed ?? 1
    })}\n`);
    if (!accepted) await new Promise<void>(resolve => this.child.stdin.once('drain', resolve));
    const response = JSON.parse(await responseLine) as WorkerResponse;
    if (!response.ok) throw new Error(`GSMI CUDA worker failed: ${response.error}`);
    const model = response.model;
    if (model.diagnostics.computeDevice !== this.device.computeDevice
      || model.diagnostics.cudaMemoryAllocatedBytes <= 0
      || model.diagnostics.momentAndInversionTimeMs <= 0) {
      throw new Error('GSMI worker returned evidence inconsistent with its CUDA runtime');
    }
    return model;
  }

  async close(): Promise<void> {
    if (this.closed) return;
    this.closed = true;
    const exited = new Promise<void>((resolve, reject) => {
      this.child.once('exit', code => {
        if (code === 0) resolve();
        else reject(new Error(`GSMI CUDA worker exited with code ${code}: ${this.stderr.join('')}`));
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
