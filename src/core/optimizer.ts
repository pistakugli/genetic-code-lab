import { PopulationBuffer, OpCode } from './pop-buffer';
import { BlockEvaluator } from './block-evaluator';
import { PostfixBridge } from './postfix-bridge';

/**
 * PHASE 42: HYBRID NUMERICAL OPTIMIZER
 * High-performance constant tuning for DOGP binary buffers.
 */
export class NumericalOptimizer {
    
    /**
     * Adaptive Gradient-Based Hillclimbing (AGBHC)
     * Phase 55: Beyond SOTA Numerical Surgery.
     * Uses numerical gradients and adaptive learning rates to pin singularities.
     */
    public static refineConstants(
        buffer: PopulationBuffer,
        indices: number[],
        testX: Float64Array,
        testY: Float64Array,
        inputSize: number,
        iterations: number = 15,
        fitnessMode: 'regression' | 'classification' = 'regression',
        classificationWeight: number = 0.8
    ): void {
        for (const genomeIdx of indices) {
            let bestFitness = buffer.fitness[genomeIdx];
            // Phase 55: Deep Refinement for high-fitness elite candidates
            const actualIterations = bestFitness > 0.4 ? iterations * 3 : iterations;
            if (bestFitness === -Infinity || bestFitness === 0) {
                BlockEvaluator.evaluateBlock(buffer, genomeIdx, 1, testX, testY, inputSize, fitnessMode, classificationWeight);
                bestFitness = buffer.fitness[genomeIdx];
            }

            const constOffset = genomeIdx * 10;
            const learningRates = new Float64Array(10);
            let lrInitialized = false;

            for (let iter = 0; iter < actualIterations; iter++) {
                // Phase 62.2: Dynamic Adaptive Precision & Schedule
                const EPS = bestFitness > 0.6 ? 1e-7 : 1e-4; 
                let lrBase = 0.05;
                if (bestFitness > 0.999) lrBase = 0.0001; 
                else if (bestFitness > 0.95) lrBase = 0.001;
                else if (bestFitness > 0.75) lrBase = 0.01;
                
                if (!lrInitialized) {
                    learningRates.fill(lrBase);
                    lrInitialized = true;
                }
                
                let anyImprovement = false;
                
                for (let cIdx = 0; cIdx < 10; cIdx++) {
                    const original = buffer.constBuffer[constOffset + cIdx];
                    
                    // 1. Numerical Gradient Calculation (Central Difference)
                    buffer.constBuffer[constOffset + cIdx] = original + EPS;
                    BlockEvaluator.evaluateBlock(buffer, genomeIdx, 1, testX, testY, inputSize, fitnessMode, classificationWeight);
                    const fPlus = buffer.fitness[genomeIdx];
                    
                    buffer.constBuffer[constOffset + cIdx] = original - EPS;
                    BlockEvaluator.evaluateBlock(buffer, genomeIdx, 1, testX, testY, inputSize, fitnessMode, classificationWeight);
                    const fMinus = buffer.fitness[genomeIdx];
                    
                    let gradient = (fPlus - fMinus) / (2 * EPS);
                    
                    // Phase 56.8: Unit-Gradient Normalization
                    // Prevents 'Hyper-Kicking' away from singularities
                    if (Math.abs(gradient) > 1.0) gradient = Math.sign(gradient);

                    if (Math.abs(gradient) < 1e-12) {
                        buffer.constBuffer[constOffset + cIdx] = original;
                        continue;
                    }

                    // 2. Adaptive Step
                    const direction = gradient > 0 ? 1 : -1;
                    const stepSize = learningRates[cIdx];
                    
                    buffer.constBuffer[constOffset + cIdx] = original + (direction * stepSize);
                    BlockEvaluator.evaluateBlock(buffer, genomeIdx, 1, testX, testY, inputSize, fitnessMode, classificationWeight);
                    const newFitness = buffer.fitness[genomeIdx];
                    
                    if (newFitness > bestFitness) {
                        bestFitness = newFitness;
                        learningRates[cIdx] *= 1.5; // Accelerate in correct direction
                        anyImprovement = true;
                    } else {
                        // Trial 2: Periodic Stochastic Jump (Phase 62 Forensic Schedule)
                        if (iter % 10 === 0) {
                            const jump = (Math.random() * 2 - 1) * 0.1;
                            const jumpCandidate = original + jump;
                            buffer.constBuffer[constOffset + cIdx] = jumpCandidate;
                            BlockEvaluator.evaluateBlock(buffer, genomeIdx, 1, testX, testY, inputSize, fitnessMode, classificationWeight);
                            const jumpFitness = buffer.fitness[genomeIdx];
                            
                            const accepted = jumpFitness > bestFitness;
                            // console.error(`[PROOF JUMP] iter=${iter} | old=${original.toFixed(6)} | new=${jumpCandidate.toFixed(6)} | delta=${jump.toFixed(6)} | fitPre=${bestFitness.toFixed(6)} | fitPost=${jumpFitness.toFixed(6)} | accepted=${accepted ? 'YES' : 'NO'}`);

                            if (accepted) {
                                bestFitness = jumpFitness;
                                anyImprovement = true;
                                continue;
                            }
                        }

                        // Trial 3: Try smaller step in case we overshot
                        buffer.constBuffer[constOffset + cIdx] = original + (direction * stepSize * 0.1);
                        BlockEvaluator.evaluateBlock(buffer, genomeIdx, 1, testX, testY, inputSize, fitnessMode, classificationWeight);
                        if (buffer.fitness[genomeIdx] > bestFitness) {
                            bestFitness = buffer.fitness[genomeIdx];
                            anyImprovement = true;
                        } else {
                            buffer.constBuffer[constOffset + cIdx] = original;
                            learningRates[cIdx] *= 0.5; // Back off
                        }
                    }
                }
                
                if (!anyImprovement && iter > 5) break; 
            }
        }
    }

    /**
     * High-Precision Individual Refinement (Phase 56)
     * Wraps a single Genome into a micro-buffer and precision-tunes constants.
     */
    public static refineIndividual(
        genome: any, 
        testX: Float64Array,
        testY: Float64Array,
        inputSize: number,
        iterations: number = 200, // Increased for breakthrough
        fitnessMode: 'regression' | 'classification' = 'regression',
        caseWeights?: Float64Array
    ): void {
        const numCases = testX.length / inputSize;
        const microBuf = new PopulationBuffer(1, genome.genes.length + 10, 10, numCases);
        
        if (caseWeights) {
            for (let i = 0; i < numCases; i++) microBuf.caseWeights[i] = caseWeights[i];
        }

        // Phase 62.1: Unified Execution Path - Use PostfixBridge to bake correctly
        // Extract variable names from genome to ensure correct mapping
        // IMPORTANT: The variableMap MUST match what the BlockEvaluator/PopulationBuffer expects (x0, x1...)
        const variableMap = [];
        for (let i = 0; i < inputSize; i++) variableMap.push(`x${i}`);
        
        const baked = PostfixBridge.bake(genome, variableMap);
        
        const packedInstructions = new Uint32Array(baked.opcodes.length);
        for (let i = 0; i < baked.opcodes.length; i++) {
            packedInstructions[i] = (baked.opcodes[i] << 24) | (baked.data[i] & 0xFFFFFF);
        }
        
        microBuf.writeGenome(0, packedInstructions, baked.data, baked.constants);
        
        // Phase 56.5: High-Precision Surgical Refinement
        // Using ultra-low LR to crawl into singularities
        this.refineConstants(microBuf, [0], testX, testY, inputSize, iterations, fitnessMode);
        
        // 4. Final Sync: Ensure fitness field is up-to-date after optimization
        BlockEvaluator.evaluateBlock(microBuf, 0, 1, testX, testY, inputSize, fitnessMode);
        // console.error(`[NumericalOptimizer] DEBUG: microBuf.fitness[0]=${microBuf.fitness[0]}`);

        // 5. Extract Constants back to Genome
        const refinedConstants = microBuf.getConstants(0);
        let cRead = 0;
        for (let i = 0; i < genome.genes.length; i++) {
            const op = genome.genes[i].operation.toLowerCase();
            if (op === 'constant' || op === 'const') {
                genome.genes[i].operands[0] = refinedConstants[cRead++];
            }
        }
        genome.fitness = microBuf.fitness[0];
    }
}

