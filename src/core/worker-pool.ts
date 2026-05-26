import { Worker } from 'worker_threads';
import { EventEmitter } from 'events';

export interface Task {
  data: any;
  resolve: (value: any) => void;
  reject: (reason?: any) => void;
}

export class WorkerPool extends EventEmitter {
  private workers: Worker[] = [];
  private idleWorkers: Worker[] = [];
  private taskQueue: Task[] = [];
  private workerPath: string;
  private size: number;
  private isShuttingDown = false;

  constructor(workerPath: string, size: number) {
    super();
    this.workerPath = workerPath;
    this.size = size;
  }

  /**
   * Initialize the pool by spawning walkers
   */
  public init() {
    for (let i = 0; i < this.size; i++) {
      this.addNewWorker();
    }
    console.error(`[WorkerPool] Initialized with ${this.size} workers.`);
  }

  private addNewWorker() {
    const isCompiled = this.workerPath.endsWith('.js');
    const worker = new Worker(this.workerPath, {
      execArgv: isCompiled ? [] : ['-r', 'ts-node/register']
    });
    worker.unref();

    worker.on('message', (result) => {
      const task = (worker as any).currentTask as Task;
      if (task) {
        task.resolve(result);
        (worker as any).currentTask = null;
      }
      this.makeWorkerIdle(worker);
    });

    worker.on('error', (err) => {
      console.error('[WorkerPool] Worker error:', err);
      const task = (worker as any).currentTask as Task;
      if (task) {
        task.reject(err);
      }
      this.removeWorker(worker);
      if (!this.isShuttingDown) this.addNewWorker();
    });

    worker.on('exit', (code) => {
      if (code !== 0 && !this.isShuttingDown) {
        console.error(`[WorkerPool] Worker stopped with exit code ${code}`);
      }
      this.removeWorker(worker);
      if (!this.isShuttingDown) this.addNewWorker();
    });

    this.workers.push(worker);
    this.makeWorkerIdle(worker);
  }

  private makeWorkerIdle(worker: Worker) {
    if (this.isShuttingDown) return;
    
    const nextTask = this.taskQueue.shift();
    if (nextTask) {
      this.runTask(worker, nextTask);
    } else {
      this.idleWorkers.push(worker);
    }
  }

  private runTask(worker: Worker, task: Task) {
    (worker as any).currentTask = task;
    worker.postMessage(task.data);
  }

  private removeWorker(worker: Worker) {
    const idx = this.workers.indexOf(worker);
    if (idx !== -1) this.workers.splice(idx, 1);
    
    const idleIdx = this.idleWorkers.indexOf(worker);
    if (idleIdx !== -1) this.idleWorkers.splice(idleIdx, 1);
  }

  /**
   * Execute a task in the pool
   */
  public async execute(data: any): Promise<any> {
    if (this.isShuttingDown) {
      throw new Error('WorkerPool is shutting down');
    }

    return new Promise((resolve, reject) => {
      const task: Task = { data, resolve, reject };
      const worker = this.idleWorkers.shift();

      if (worker) {
        this.runTask(worker, task);
      } else {
        this.taskQueue.push(task);
      }
    });
  }

  /**
   * Shutdown the pool
   */
  public async shutdown() {
    this.isShuttingDown = true;
    const terminationPromises = this.workers.map(w => w.terminate());
    await Promise.all(terminationPromises);
    this.workers = [];
    this.idleWorkers = [];
    this.taskQueue = [];
    console.error('[WorkerPool] Shutdown complete.');
  }
}
