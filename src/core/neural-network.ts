import { random } from '../utils/helpers';
/**
 * 🧠 Genetic Code Lab — Pure TypeScript Neural Network
 * Hand-coded MLP with backpropagation. Zero dependencies.
 * Designed for GP-Neural Stacking: GP evolves features, NN classifies.
 */

// ─── Matrix Utilities ────────────────────────────────────────────────────────

function matMul(A: number[][], B: number[][]): number[][] {
    const rows = A.length, cols = B[0].length, inner = B.length;
    const C: number[][] = Array.from({ length: rows }, () => new Array(cols).fill(0));
    for (let i = 0; i < rows; i++)
        for (let k = 0; k < inner; k++)
            for (let j = 0; j < cols; j++)
                C[i][j] += A[i][k] * B[k][j];
    return C;
}

function transpose(M: number[][]): number[][] {
    const rows = M.length, cols = M[0].length;
    const T: number[][] = Array.from({ length: cols }, () => new Array(rows));
    for (let i = 0; i < rows; i++)
        for (let j = 0; j < cols; j++)
            T[j][i] = M[i][j];
    return T;
}

// ─── Activation Functions ────────────────────────────────────────────────────

function tanh(x: number): number { return Math.tanh(x); }
function tanhDeriv(x: number): number { const t = Math.tanh(x); return 1 - t * t; }
function sigmoid(x: number): number { return 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, x)))); }

// ─── Xavier Initialization ───────────────────────────────────────────────────

function xavierInit(fanIn: number, fanOut: number): number[][] {
    const limit = Math.sqrt(6 / (fanIn + fanOut));
    return Array.from({ length: fanIn }, () =>
        Array.from({ length: fanOut }, () => (random() * 2 - 1) * limit)
    );
}

// ─── Dense Layer ─────────────────────────────────────────────────────────────

class DenseLayer {
    weights: number[][];
    biases: number[];
    activation: 'tanh' | 'sigmoid' | 'relu' | 'linear';
    
    // Cached for backprop
    private lastInput: number[][] = [];
    private lastPreActivation: number[][] = [];
    private lastOutput: number[][] = [];
    
    // Gradient accumulators
    private dW: number[][] = [];
    private dB: number[] = [];

    constructor(inputSize: number, outputSize: number, activation: 'tanh' | 'sigmoid' | 'relu' | 'linear' = 'tanh') {
        this.weights = xavierInit(inputSize, outputSize);
        this.biases = new Array(outputSize).fill(0);
        this.activation = activation;
        this.dW = Array.from({ length: inputSize }, () => new Array(outputSize).fill(0));
        this.dB = new Array(outputSize).fill(0);
    }

    forward(input: number[][]): number[][] {
        this.lastInput = input;
        const batchSize = input.length;
        const outSize = this.biases.length;
        
        // Z = X @ W + b
        const Z = matMul(input, this.weights);
        for (let i = 0; i < batchSize; i++)
            for (let j = 0; j < outSize; j++)
                Z[i][j] += this.biases[j];
        
        this.lastPreActivation = Z;
        
        // Apply activation
        const A: number[][] = Z.map(row => row.map(v => {
            switch (this.activation) {
                case 'tanh': return tanh(v);
                case 'sigmoid': return sigmoid(v);
                case 'relu': return Math.max(0, v);
                case 'linear': return v;
            }
        }));
        
        this.lastOutput = A;
        return A;
    }

    backward(dOutput: number[][]): number[][] {
        const batchSize = dOutput.length;
        const outSize = this.biases.length;
        const inSize = this.weights.length;
        
        // dZ = dOutput * activation'(preActivation)
        const dZ: number[][] = Array.from({ length: batchSize }, (_, i) =>
            Array.from({ length: outSize }, (_, j) => {
                const z = this.lastPreActivation[i][j];
                let deriv: number;
                switch (this.activation) {
                    case 'tanh': deriv = tanhDeriv(z); break;
                    case 'sigmoid': { const s = sigmoid(z); deriv = s * (1 - s); break; }
                    case 'relu': deriv = z > 0 ? 1 : 0; break;
                    case 'linear': deriv = 1; break;
                }
                return dOutput[i][j] * deriv;
            })
        );
        
        // dW = input^T @ dZ / batchSize
        const inputT = transpose(this.lastInput);
        const grad = matMul(inputT, dZ);
        for (let i = 0; i < inSize; i++)
            for (let j = 0; j < outSize; j++)
                this.dW[i][j] = grad[i][j] / batchSize;
        
        // dB = mean(dZ, axis=0)
        for (let j = 0; j < outSize; j++) {
            let sum = 0;
            for (let i = 0; i < batchSize; i++) sum += dZ[i][j];
            this.dB[j] = sum / batchSize;
        }
        
        // dInput = dZ @ W^T
        return matMul(dZ, transpose(this.weights));
    }

    update(lr: number) {
        for (let i = 0; i < this.weights.length; i++)
            for (let j = 0; j < this.weights[0].length; j++)
                this.weights[i][j] -= lr * this.dW[i][j];
        for (let j = 0; j < this.biases.length; j++)
            this.biases[j] -= lr * this.dB[j];
    }
}

// ─── MLP (Multi-Layer Perceptron) ────────────────────────────────────────────

export class MLP {
    private layers: DenseLayer[];

    constructor(layerSizes: number[], activations?: ('tanh' | 'sigmoid' | 'relu' | 'linear')[]) {
        this.layers = [];
        for (let i = 0; i < layerSizes.length - 1; i++) {
            const act = activations 
                ? activations[i] 
                : (i === layerSizes.length - 2 ? 'sigmoid' : 'tanh');
            this.layers.push(new DenseLayer(layerSizes[i], layerSizes[i + 1], act));
        }
    }

    predict(input: number[][]): number[][] {
        let x = input;
        for (const layer of this.layers) x = layer.forward(x);
        return x;
    }

    backward(dOutput: number[][]): void {
        let grad = dOutput;
        for (let l = this.layers.length - 1; l >= 0; l--) {
            grad = this.layers[l].backward(grad);
        }
    }

    updateWeights(lr: number): void {
        for (const layer of this.layers) {
            layer.update(lr);
        }
    }

    /**
     * Train with mini-batch SGD + binary cross-entropy loss
     */
    train(
        X: number[][],
        Y: number[],
        epochs: number = 500,
        lr: number = 0.01,
        batchSize: number = 32,
        verbose: boolean = false
    ): void {
        const n = X.length;
        
        for (let epoch = 0; epoch < epochs; epoch++) {
            // Shuffle
            const indices = Array.from({ length: n }, (_, i) => i);
            for (let i = n - 1; i > 0; i--) {
                const j = Math.floor(random() * (i + 1));
                [indices[i], indices[j]] = [indices[j], indices[i]];
            }
            
            let epochLoss = 0;
            let batches = 0;
            
            for (let start = 0; start < n; start += batchSize) {
                const end = Math.min(start + batchSize, n);
                const batchX = indices.slice(start, end).map(i => X[i]);
                const batchY = indices.slice(start, end).map(i => Y[i]);
                
                // Forward
                const output = this.predict(batchX);
                
                // Binary Cross-Entropy Loss + gradient
                const dOutput: number[][] = [];
                let batchLoss = 0;
                
                for (let i = 0; i < batchX.length; i++) {
                    const pred = Math.max(1e-7, Math.min(1 - 1e-7, output[i][0]));
                    const target = batchY[i];
                    batchLoss += -(target * Math.log(pred) + (1 - target) * Math.log(1 - pred));
                    // Gradient of BCE w.r.t. sigmoid output
                    dOutput.push([pred - target]);
                }
                
                epochLoss += batchLoss / batchX.length;
                batches++;
                
                // Backward
                let grad = dOutput;
                for (let l = this.layers.length - 1; l >= 0; l--) {
                    grad = this.layers[l].backward(grad);
                }
                
                // Update
                const currentLR = lr * (1 / (1 + 0.001 * epoch)); // LR decay
                for (const layer of this.layers) layer.update(currentLR);
            }
            
            if (verbose && epoch % 100 === 0) {
                console.error(`  [NN] Epoch ${epoch}: Loss = ${(epochLoss / batches).toFixed(4)}`);
            }
        }
    }

    /**
     * Predict class (0 or 1)
     */
    classify(input: number[][]): number[] {
        const probs = this.predict(input);
        return probs.map(p => p[0] > 0.5 ? 1 : 0);
    }

    /**
     * Accuracy on labeled data
     */
    accuracy(X: number[][], Y: number[]): number {
        const preds = this.classify(X);
        let correct = 0;
        for (let i = 0; i < Y.length; i++) if (preds[i] === Y[i]) correct++;
        return correct / Y.length;
    }
}
