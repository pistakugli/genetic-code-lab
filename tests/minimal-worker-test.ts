import { Worker, isMainThread, parentPort, workerData } from 'worker_threads';
import path from 'path';

if (isMainThread) {
    console.log("[Launcher] Starting minimal worker test...");
    const worker = new Worker(__filename, {
        workerData: "Hello from Main!",
        execArgv: ['-r', 'ts-node/register'],
        stderr: true,
        stdout: true
    });
    
    worker.stdout.on('data', (d) => console.log(`[Worker-Stdout] ${d}`));
    worker.stderr.on('data', (d) => console.log(`[Worker-Stderr] ${d}`));
    
    worker.on('message', (msg) => console.log(`[Launcher] Received: ${msg}`));
    worker.on('error', (err) => console.error(`[Launcher] Worker Error:`, err));
    worker.on('exit', (code) => console.log(`[Launcher] Worker exited with code ${code}`));
    
    worker.postMessage("PING");
    
    setTimeout(() => worker.terminate(), 5000);
} else {
    console.log(`[Worker] Started. Data: ${workerData}`);
    parentPort?.on('message', (msg) => {
        console.log(`[Worker] Received: ${msg}`);
        parentPort?.postMessage("PONG");
    });
}
