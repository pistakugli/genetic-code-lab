import * as path from 'path';
import { Worker } from 'worker_threads';
import {
  createEvaluationRequestId,
  EvaluationEnvelope,
  EvaluationRequestEnvelope,
  EvaluationResponseEnvelope,
  isEvaluationErrorEnvelope,
  isEvaluationResponseEnvelope,
  PackedEvaluationBatch
} from './evaluation-protocol';

interface PendingTask {
  resolve: (value: EvaluationResponseEnvelope) => void;
  reject: (reason?: unknown) => void;
}

/**
 * Isolated parallel-evaluation pool.
 * This is intentionally not wired into EvolutionEngine yet.
 */
export class EvaluationWorkerPool {
  private readonly workerCount: number;
  private readonly workers: Worker[] = [];
  private readonly idleWorkers: Worker[] = [];
  private readonly pending = new Map<string, PendingTask>();
  private readonly workerToRequest = new Map<Worker, string>();
  private readonly workerPath: string;
  private shuttingDown = false;

  constructor(workerCount: number, workerPath?: string) {
    this.workerCount = workerCount;
    const isCompiled = __filename.endsWith('.js');
    const defaultWorker = path.join(__dirname, `evaluation-worker${isCompiled ? '.js' : '.ts'}`);
    this.workerPath = workerPath ?? defaultWorker;
  }

  public init(): void {
    for (let i = 0; i < this.workerCount; i++) {
      this.spawnWorker();
    }
  }

  public async submit(batch: PackedEvaluationBatch): Promise<EvaluationResponseEnvelope> {
    if (this.shuttingDown) {
      throw new Error('EvaluationWorkerPool is shutting down');
    }

    const request: EvaluationRequestEnvelope = {
      requestId: createEvaluationRequestId('eval'),
      kind: 'evaluate',
      batch
    };

    return new Promise<EvaluationResponseEnvelope>((resolve, reject) => {
      this.pending.set(request.requestId, { resolve, reject });

      const worker = this.idleWorkers.shift();
      if (worker) {
        this.dispatch(worker, request);
      } else {
        this.pending.get(request.requestId)!.reject = (reason?: unknown) => {
          this.pending.delete(request.requestId);
          reject(reason);
        };
        this.queueRequest(request);
      }
    });
  }

  public async shutdown(): Promise<void> {
    this.shuttingDown = true;
    await Promise.all(this.workers.map(worker => worker.terminate()));
    this.workers.length = 0;
    this.idleWorkers.length = 0;
    this.pending.clear();
    this.workerToRequest.clear();
  }

  private queue: EvaluationRequestEnvelope[] = [];

  private queueRequest(request: EvaluationRequestEnvelope): void {
    this.queue.push(request);
  }

  private spawnWorker(): void {
    const worker = new Worker(this.workerPath, {
      execArgv: this.workerPath.endsWith('.js') ? [] : ['-r', 'ts-node/register']
    });

    worker.on('message', (message: EvaluationEnvelope) => {
      if (isEvaluationResponseEnvelope(message)) {
        const pending = this.pending.get(message.requestId);
        if (pending) {
          pending.resolve(message);
          this.pending.delete(message.requestId);
        }
      } else if (isEvaluationErrorEnvelope(message)) {
        const pending = this.pending.get(message.requestId);
        if (pending) {
          pending.reject(new Error(message.error.message));
          this.pending.delete(message.requestId);
        }
      }

      this.releaseWorker(worker);
    });

    worker.on('error', error => {
      const requestId = this.workerToRequest.get(worker);
      if (requestId) {
        const pending = this.pending.get(requestId);
        if (pending) pending.reject(error);
        this.pending.delete(requestId);
      }
      this.detachWorker(worker);
      if (!this.shuttingDown) this.spawnWorker();
    });

    worker.on('exit', code => {
      if (code !== 0 && !this.shuttingDown) {
        console.error(`[EvaluationWorkerPool] Worker exited with code ${code}`);
      }
      this.detachWorker(worker);
      if (!this.shuttingDown) this.spawnWorker();
    });

    this.workers.push(worker);
    this.idleWorkers.push(worker);
    this.flushQueue();
  }

  private dispatch(worker: Worker, request: EvaluationRequestEnvelope): void {
    this.workerToRequest.set(worker, request.requestId);
    worker.postMessage(request);
  }

  private releaseWorker(worker: Worker): void {
    this.workerToRequest.delete(worker);
    if (this.shuttingDown) return;
    const next = this.queue.shift();
    if (next) {
      this.dispatch(worker, next);
    } else {
      this.idleWorkers.push(worker);
    }
  }

  private detachWorker(worker: Worker): void {
    const workerIdx = this.workers.indexOf(worker);
    if (workerIdx !== -1) this.workers.splice(workerIdx, 1);

    const idleIdx = this.idleWorkers.indexOf(worker);
    if (idleIdx !== -1) this.idleWorkers.splice(idleIdx, 1);

    this.workerToRequest.delete(worker);
  }

  private flushQueue(): void {
    while (this.queue.length > 0 && this.idleWorkers.length > 0) {
      const worker = this.idleWorkers.shift();
      const request = this.queue.shift();
      if (!worker || !request) return;
      this.dispatch(worker, request);
    }
  }
}
