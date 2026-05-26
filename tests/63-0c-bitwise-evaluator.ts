import { BlockEvaluator } from '../src/core/block-evaluator';
import { PopulationBuffer, OpCode } from '../src/core/pop-buffer';

/**
 * 63.0 C: Bitwise Evaluator Proof
 * Manual construction of a XOR-2 circuit and verification via BlockEvaluator.
 */

function runBitwiseProof() {
    console.log("--- 63.0 C: BITWISE EVALUATOR PROOF ---");

    // 1. Setup XOR-2 Dataset (4 cases)
    const inputs = new Float64Array([
        0, 0,
        0, 1,
        1, 0,
        1, 1
    ]);
    const expected = new Float64Array([0, 1, 1, 0]);

    // 2. Setup Population Buffer (Size 1, 128 nodes, 10 constants, 4 test cases)
    const buffer = new PopulationBuffer(1, 128, 10, 4);
    
    // 3. Manual Genome: (a OR b) AND (a NAND b)
    // RPN: a, b, OR, a, b, NAND, AND
    const opcodes = new Uint32Array([
        (OpCode.XN << 24) | 0, // a
        (OpCode.XN << 24) | 1, // b
        (OpCode.OR << 24) | 0,
        (OpCode.XN << 24) | 0, // a
        (OpCode.XN << 24) | 1, // b
        (OpCode.NAND << 24) | 0,
        (OpCode.AND << 24) | 0
    ]);

    buffer.writeGenome(0, opcodes, [], new Float64Array(10));
    buffer.caseWeights.fill(1.0);

    // 4. Evaluate
    const results = BlockEvaluator.evaluateBlock(
        buffer, 0, 1, inputs, expected, 2, 'regression'
    );

    const fitness = buffer.fitness[0];
    console.log(`Evaluated Fitness: ${fitness.toFixed(4)}`);
    console.log("Case Errors:", Array.from(buffer.caseErrors.subarray(0, 4)));

    // For parity-2, 100% accuracy means error is 0.
    if (fitness !== 1.0) {
        console.error("FAIL: 100% accuracy not achieved on manual XOR.");
        process.exit(1);
    }

    console.log("\n63.0 C: STATUS: PASS");
}

runBitwiseProof();
