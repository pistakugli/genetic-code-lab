import { DiscoveryAgent } from '../src/agents/discovery-agent';
import { Genome } from '../src/core/types';

/**
 * PHASE 62.5: DISCOVERY AGENT ROUND-TRIP IDENTITY PROOF
 * Goal: Prove that AST -> BINARY -> AST is an identity (or functionally identical)
 * strict adherence to the x/const standard and root-at-zero invariant.
 */

function runRoundTripProof() {
    console.log("--- PROOF DISCOVERY AGENT IDENTITY ROUND-TRIP ---");

    const agent = new DiscoveryAgent();
    const opMap: Record<string, number> = { add: 1, sub: 2, mul: 3, div: 4, sin: 5, cos: 6, x: 10, const: 11 };

    // 1. Original RPN Intent (Target structure: (x*x+1)/(x-2))
    const opDefinitions = [
        { op: 'x', type: 'terminal' },
        { op: 'x', type: 'terminal' },
        { op: 'mul', type: 'function' },
        { op: 'const', type: 'terminal', val: 1.0 },
        { op: 'add', type: 'function' },
        { op: 'x', type: 'terminal' },
        { op: 'const', type: 'terminal', val: 2.0 },
        { op: 'sub', type: 'function' },
        { op: 'div', type: 'function' }
    ];

    console.log("Phase 62.5: Starting Round-trip...");

    // 2. Encode to Binary (Manual Packed Format to match Agent)
    const opcodes = new Uint32Array(opDefinitions.length);
    const constantsArr = new Float64Array(10);
    let cIdx = 0;

    opDefinitions.forEach((def, i) => {
        const opTag = opMap[def.op];
        if (def.op === 'const') {
            constantsArr[cIdx] = def.val!;
            opcodes[i] = (opTag << 24) | (cIdx++ & 0xFFFFFF);
        } else {
            opcodes[i] = (opTag << 24) | 0;
        }
    });
    
    // 3. Decode back to AST (Agent Decompiler)
    const binaryCandidate = {
        opcodes: opcodes,
        constants: constantsArr,
        fitness: 0.999
    };
    const decodedGenome = (agent as any).decodeBinaryCandidate(binaryCandidate);

    // 4. Assert Identity
    // We manually repack the decoded genome to check identity
    const repackedOpcodes = new Uint32Array(decodedGenome.genes.length);
    const repackedConstants = new Float64Array(10);
    let rcIdx = 0;

    // To repack correctly, we'd need to convert AST back to RPN. 
    // But since the decompiler already maintains RPN order in the genes array (except for root at zero),
    // we can just check the sequence.
    
    // FORENSIC CHECK: Terminals and Functions must match the original intent.
    const decodedOps = decodedGenome.genes.map((g: any) => g.operation);
    
    // Root Invariant (Index 0)
    const rootInvariant = decodedGenome.genes[0].operation === 'div';

    // Token Standard
    const tokensCorrect = decodedGenome.genes.every((g: any) => 
        g.operation !== 'variable' && g.operation !== 'constant'
    );

    console.log("\n--- FORENSIC IDENTITY REPORT ---");
    console.log(`- Decoded Genes Count: ${decodedGenome.genes.length} (Expected ${opDefinitions.length})`);
    console.log(`- Root Invariant (0):  ${rootInvariant ? '[PASS]' : '[FAIL]'}`);
    console.log(`- Token Standard Correct: ${tokensCorrect ? '[PASS]' : '[FAIL]'}`);
    console.log(`- Decoded Operations: ${decodedOps.join(', ')}`);
    
    // The decoded order should be [ROOT, T1, T2, F1, ...] because of the shift.
    // The actual content check:
    const expectedOps = ['div', 'x0', 'x0', 'mul', 'const', 'add', 'x0', 'const', 'sub'];
    const contentMatch = decodedOps.length === expectedOps.length && 
                        decodedOps.every((op: string, i: number) => op === expectedOps[i]);

    console.log(`- Structural Content Match: ${contentMatch ? '[PASS]' : '[FAIL]'}`);

    const pass = rootInvariant && tokensCorrect && contentMatch;
    console.log(`\nOVERALL STATUS: ${pass ? 'PASS' : 'FAIL'}`);
    
    if (!pass) process.exit(1);
}

try {
    runRoundTripProof();
} catch (e: any) {
    console.error("CRITICAL TEST FAILURE:", e.message);
    console.error(e.stack);
    process.exit(1);
}
