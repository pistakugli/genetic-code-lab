import { PopulationBuffer, OpCode } from './pop-buffer';
import { BlockEvaluator } from './block-evaluator';
import { Genome } from './types';
import { PostfixBridge } from './postfix-bridge';

/**
 * Genetic Code Lab - Boolean Refiner
 * Performs discrete optimization on binary genomes using hill-climbing.
 */
export class BooleanRefiner {

    /**
     * Refines a candidate individual in a PopulationBuffer.
     */
    public static refine(
        buffer: PopulationBuffer,
        idx: number,
        inputs: Float64Array,
        expected: Float64Array,
        inputSize: number
    ): void {
        const offset = buffer.offsets[idx];
        const len = buffer.lengths[idx];
        const initialFitness = buffer.fitness[idx];
        const logicOps = [OpCode.AND, OpCode.OR, OpCode.NAND, OpCode.NOR];

        const fitnessMode = 'classification';
        let bestFitness = initialFitness;

        for (let i = 0; i < len; i++) {
            const originalInstruction = buffer.geneBuffer[offset + i];
            const { op, data } = PopulationBuffer.unpack(originalInstruction);

            if (op === OpCode.XN) {
                for (let v = 0; v < inputSize; v++) {
                    if (v === data) continue;
                    buffer.geneBuffer[offset + i] = PopulationBuffer.pack(OpCode.XN, v);
                    BlockEvaluator.evaluateBlock(buffer, idx, 1, inputs, expected, inputSize, fitnessMode);
                    if (buffer.fitness[idx] > bestFitness) {
                        bestFitness = buffer.fitness[idx];
                    } else {
                        buffer.geneBuffer[offset + i] = PopulationBuffer.pack(OpCode.XN, data);
                    }
                }
            } else if (logicOps.includes(op)) {
                for (const nextOp of logicOps) {
                    if (nextOp === op) continue;
                    buffer.geneBuffer[offset + i] = PopulationBuffer.pack(nextOp, data);
                    BlockEvaluator.evaluateBlock(buffer, idx, 1, inputs, expected, inputSize, fitnessMode);
                    if (buffer.fitness[idx] > bestFitness) {
                        bestFitness = buffer.fitness[idx];
                    } else {
                        buffer.geneBuffer[offset + i] = PopulationBuffer.pack(op, data);
                    }
                }
            }
        }
        buffer.fitness[idx] = bestFitness;
    }

    /**
     * Refines a Genome (AST) by converting it to postfix, optimizing, and unbaking.
     */
    public static refineIndividual(
        genome: Genome,
        inputs: Float64Array,
        expected: Float64Array,
        inputSize: number
    ): void {
        // 1. Bake to postfix
        const varMap = Array.from({ length: 32 }, (_, i) => `b${i}`);
        const baked = PostfixBridge.bake(genome, varMap);
        
        // 2. Setup micro-buffer for optimization
        const microBuf = new PopulationBuffer(1, baked.opcodes.length, 1, expected.length);
        const binaryOpcodes = baked.opcodes.map((op, i) => PopulationBuffer.pack(op, baked.data[i]));
        microBuf.writeGenome(0, binaryOpcodes, [], baked.constants);

        // 3. Initial evaluation
        BlockEvaluator.evaluateBlock(microBuf, 0, 1, inputs, expected, inputSize, 'classification');
        
        // 4. Refine in-buffer
        this.refine(microBuf, 0, inputs, expected, inputSize);

        // 5. Unbake back to genome
        const finalOpcodes: OpCode[] = [];
        const finalData: number[] = [];
        for (let i = 0; i < microBuf.lengths[0]; i++) {
            const { op, data } = PopulationBuffer.unpack(microBuf.geneBuffer[microBuf.offsets[0] + i]);
            finalOpcodes.push(op);
            finalData.push(data);
        }

        const optimized = PostfixBridge.unbake(finalOpcodes, finalData, Array.from(microBuf.getConstants(0)), varMap);
        genome.genes = optimized.genes;
        genome.fitness = microBuf.fitness[0];
    }
}
