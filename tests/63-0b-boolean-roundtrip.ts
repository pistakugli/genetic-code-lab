import { PostfixBridge } from '../src/core/postfix-bridge';
import { OpCode } from '../src/core/pop-buffer';
import { Genome } from '../src/core/types';

/**
 * 63.0 B: Boolean Roundtrip Proof
 * Verifies structural identity (original opcodes === rebaked opcodes) for boolean logic.
 */

function runRoundtripProof() {
    console.log("--- 63.0 B: BOOLEAN ROUNDTRIP PROOF ---");

    // 1. Define Boolean AST (NAND(AND(b0, b1), b2))
    const booleanGenome: Genome = {
        id: 'bool-test',
        genes: [
            { id: 'g0', type: 'function', operation: 'nand', operands: ['g1', 'g2'] },
            { id: 'g1', type: 'function', operation: 'and', operands: ['b0', 'b1'] },
            { id: 'g2', type: 'terminal', operation: 'b2', operands: [] },
            { id: 'b0', type: 'terminal', operation: 'b0', operands: [] },
            { id: 'b1', type: 'terminal', operation: 'b1', operands: [] }
        ],
        fitness: 0,
        generation: 0,
        age: 0
    };

    const variableMap = ['b0', 'b1', 'b2'];

    // 2. Bake to Binary
    const baked = PostfixBridge.bake(booleanGenome, variableMap);
    console.log("Baked Opcodes:", baked.opcodes.join(','));
    console.log("Baked Data:   ", baked.data.join(','));

    // RPN for NAND(AND(b0, b1), b2) -> b0, b1, AND, b2, NAND
    const expectedOpcodes = [OpCode.XN, OpCode.XN, OpCode.AND, OpCode.XN, OpCode.NAND];
    const expectedData = [0, 1, 0, 2, 0]; // b0=0, b1=1, b2=2

    console.log("Expected Opcodes:", expectedOpcodes.join(','));
    
    if (baked.opcodes.join(',') !== expectedOpcodes.join(',')) {
        console.error("FAIL: Postfix bake mismatch.");
        process.exit(1);
    }

    // 3. Unbake back to AST
    const unbaked = PostfixBridge.unbake(baked.opcodes, baked.data, baked.constants, variableMap);
    
    // 4. Rebake and verify Exact Identity
    const rebaked = PostfixBridge.bake(unbaked, variableMap);
    console.log("Rebaked Opcodes:", rebaked.opcodes.join(','));

    if (baked.opcodes.join(',') !== rebaked.opcodes.join(',')) {
        console.error("FAIL: Structural Identity failure (Baked !== Rebaked).");
        process.exit(1);
    }

    console.log("\n63.0 B: STATUS: PASS");
}

runRoundtripProof();
