import { PostfixBridge } from '../src/core/postfix-bridge';
import { PopulationBuffer, OpCode } from '../src/core/pop-buffer';
import { BlockEvaluator } from '../src/core/block-evaluator';
import { Gene, Genome } from '../src/core/types';
import { verifyF3 } from './canonical-f3-verifier';

/**
 * STEP 2: ENGINE-NATIVE IDEAL GENOME PROOF
 * Formula: (x^2 + 1) / (x - 2)
 */

const idealGenes: Gene[] = [
    { id: 'root', type: 'function', operation: 'div', operands: ['num', 'den'] },
    { id: 'num', type: 'function', operation: 'add', operands: ['x2', 'c1'] },
    { id: 'x2', type: 'function', operation: 'mul', operands: ['x', 'x'] },
    { id: 'x', type: 'terminal', operation: 'x', operands: [] }, // 'x' instead of 'variable' match variableMap
    { id: 'c1', type: 'terminal', operation: 'const', operands: [1.0] },
    { id: 'den', type: 'function', operation: 'sub', operands: ['x', 'c2'] },
    { id: 'c2', type: 'terminal', operation: 'const', operands: [2.0] },
];

const genome: Genome = {
    id: 'ideal',
    genes: idealGenes,
    fitness: 0,
    generation: 0,
    age: 0
};

// 1. Bake to Postfix
const variableMap = ['x'];
const baked = PostfixBridge.bake(genome, variableMap);

// 2. Load into PopulationBuffer
// Buffer constructor: maxGenomes, maxNodesPerGenome, maxConstantsPerGenome, maxTestCases
const testPoints = 100;
const buffer = new PopulationBuffer(1, 128, 10, testPoints);

// Manually pack instructions for PopulationBuffer.writeGenome
const packedInstructions = new Uint32Array(baked.opcodes.length);
for (let i = 0; i < baked.opcodes.length; i++) {
    packedInstructions[i] = (baked.opcodes[i] << 24) | (baked.data[i] & 0xFFFFFF);
}
// writeGenome(targetIdx, packedOpcodes, data_ignored, constants)
buffer.writeGenome(0, packedInstructions, baked.data, baked.constants);

// 3. Prepare Test Data
const inputs = new Float64Array(testPoints);
const expected = new Float64Array(testPoints);
const start = -5, end = 5, step = (end - start) / (testPoints - 1);
for (let i = 0; i < testPoints; i++) {
    const x = start + i * step;
    inputs[i] = x;
    const den = x - 2;
    // Ground Truth with Sentinel
    expected[i] = Math.abs(den) < 1e-12 ? 1e12 : (x * x + 1) / den;
}

// 4. Run Engine Evaluation
console.log("--- PROOF ENGINE-NATIVE (STEP 2) ---");
const fitnessResults = BlockEvaluator.evaluateBlock(buffer, 0, 1, inputs, expected, 1, 'regression');
const engineFitness = fitnessResults[0];

// 5. Forensic Vector Comparison
console.log("\nForensic Vector Check (First 5 points):");
const enginePreds = BlockEvaluator.getPredictions(buffer, 0, inputs, 1);
for (let i = 0; i < 5; i++) {
    console.log(`x=${inputs[i].toFixed(2)} | Target=${expected[i].toFixed(4)} | Engine=${enginePreds[i].toFixed(4)}`);
}

// 6. Canonical Validation (Cross-check)
const engineEvalFunc = (x: number): number => {
    const predVector = BlockEvaluator.getPredictions(buffer, 0, new Float64Array([x]), 1);
    return predVector[0];
};
const proof = verifyF3(engineEvalFunc);

console.log(`\nSTATUS: ${proof.honestFitness > 0.999 ? "PASS" : "FAIL"}`);
console.log(`Engine Fitness Output: ${engineFitness.toFixed(15)}`);
console.log(`Canonical Honest Fitness: ${proof.honestFitness.toFixed(15)}`);
console.log(`Metric Match: ${Math.abs(engineFitness - proof.honestFitness) < 1e-10 ? "YES" : "NO"}`);
console.log(`MAE: ${proof.mae.toExponential(4)}`);
