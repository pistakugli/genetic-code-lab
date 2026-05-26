import { EvolutionEngine } from '../src/core/evolution';
import { DiscoveryAgent } from '../src/agents/discovery-agent';
import { AsyncAgentBridge } from '../src/core/async-agent-bridge';
import { createBooleanParityProblem } from '../src/solvers/boolean-parity';
import { OpCode } from '../src/core/pop-buffer';

/**
 * PHASE 63.2: UNIFIED POLYGLOT PROOF
 * Goal: Prove autonomous domain switching between Continuous (F3) and Discrete (BP-7).
 */

async function runUnifiedProof() {
    console.log("--- 63.2: UNIFIED POLYGLOT PROOF ---");
    console.log("Stage 1: F3-Rational (Continuous)");
    console.log("Stage 2: Boolean-7 Parity (Discrete)\n");

    const agent = new DiscoveryAgent();
    const bridge = new AsyncAgentBridge(agent);

    // --- STAGE 1: F3-RATIONAL (CONTINUOUS) ---
    console.log(">> STARTING STAGE 1: F3-RATIONAL...");
    const testX = new Float64Array(1001);
    const testY = new Float64Array(1001);
    for (let i = 0; i <= 1000; i++) {
        const x = -5.0 + (i * 0.01);
        testX[i] = x;
        const den = x - 2.0;
        let y = (x * x + 1.0) / den;
        if (!isFinite(y) || Math.abs(den) < 1e-12) y = 1e12; 
        if (Math.abs(y) > 1e11) y = 1e12;
        testY[i] = y;
    }

    const f3Problem: any = {
        name: 'F3-Rational',
        functionSet: [
            { name: 'add', arity: 2, implementation: (a: any, b: any) => a + b, returnType: 'number' },
            { name: 'sub', arity: 2, implementation: (a: any, b: any) => a - b, returnType: 'number' },
            { name: 'mul', arity: 2, implementation: (a: any, b: any) => a * b, returnType: 'number' },
            { name: 'div', arity: 2, implementation: (a: any, b: any) => a / b, returnType: 'number' }
        ],
        terminalSet: [{ name: 'x', type: 'variable' }, { name: 'const', type: 'constant' }],
        testCases: Array.from(testX).map((x, i) => ({ input: [x], expectedOutput: testY[i] })),
        fitnessFunction: () => 0 
    };

    const engineF3 = new EvolutionEngine(f3Problem, {
        population: { 
            size: 100, 
            eliteCount: 2, 
            maxGenomeSize: 50,
            crossoverRate: 0.8,
            mutationRate: 0.1,
            maxGenomeDepth: 10,
            tournamentSize: 3
        },
        maxGenerations: 60
    });
    engineF3.setDiscoveryAgent(agent);
    // @ts-ignore
    engineF3.agentBridge = bridge;

    // Run Stage 1
    for (let gen = 0; gen <= 60; gen++) {
        await (engineF3 as any).step(gen, 0);
        const best = engineF3.getBestGenome();
        if (best && best.fitness >= 0.999999999999) break;
    }

    const f3Winner = engineF3.getBestGenome()!;
    const f3Profile = (engineF3 as any).domainProfile;

    // --- STAGE 1 ASSERTIONS ---
    console.log("\n[STAGE 1] ASSERTIONS:");
    const f3DomainPass = f3Profile.domainClass === 'continuous';
    const f3FitPass = f3Winner.fitness >= 0.999999999999;
    
    let logicInF3 = 0;
    const logicOps = [OpCode.AND, OpCode.OR, OpCode.NAND, OpCode.NOR];
    // @ts-ignore
    for (const opRaw of f3Winner.opcodes) {
        if (logicOps.includes(opRaw >> 24)) logicInF3++;
    }
    const f3ArithPass = logicInF3 === 0;

    console.log(`1. Domain Detection (continuous): ${f3DomainPass ? 'PASS ✅' : 'FAIL ❌'}`);
    console.log(`2. Target Fitness (1.0):         ${f3FitPass ? 'PASS ✅' : 'FAIL ❌'}`);
    console.log(`3. Arithmetic-Only Audit:        ${f3ArithPass ? 'PASS ✅' : 'FAIL ❌'}`);

    if (!f3DomainPass || !f3FitPass || !f3ArithPass) {
        console.error("\nSTAGE 1 FAILED. Terminating Unified Proof.");
        process.exit(1);
    }

    // --- STAGE 2: BOOLEAN-7 PARITY (DISCRETE) ---
    console.log("\n>> STARTING STAGE 2: BOOLEAN-7 PARITY...");
    const bp7Problem = createBooleanParityProblem(7);
    const engineBP7 = new EvolutionEngine(bp7Problem, {
        population: { 
            size: 500, 
            eliteCount: 20, 
            maxGenomeSize: 200,
            crossoverRate: 0.8,
            mutationRate: 0.1,
            maxGenomeDepth: 15,
            tournamentSize: 5
        },
        maxGenerations: 50
    });
    engineBP7.setDiscoveryAgent(agent);
    // @ts-ignore
    engineBP7.agentBridge = bridge;

    // Run Stage 2
    for (let gen = 0; gen <= 50; gen++) {
        await (engineBP7 as any).step(gen, 0);
        const best = engineBP7.getBestGenome();
        if (best && best.fitness >= 1.0) break;
    }

    const bp7Winner = engineBP7.getBestGenome()!;
    const bp7Profile = (engineBP7 as any).domainProfile;

    // --- STAGE 2 ASSERTIONS ---
    console.log("\n[STAGE 2] ASSERTIONS:");
    const bp7DomainPass = bp7Profile.domainClass === 'discrete';
    const bp7FitPass = bp7Winner.fitness >= 1.0;
    
    let arithInBP7 = 0;
    const arithOps = [OpCode.ADD, OpCode.SUB, OpCode.MUL, OpCode.DIV, OpCode.SIN, OpCode.COS];
    // @ts-ignore
    for (const opRaw of bp7Winner.opcodes) {
        if (arithOps.includes(opRaw >> 24)) arithInBP7++;
    }
    const bp7LogicPass = arithInBP7 === 0;

    console.log(`4. Domain Detection (discrete):   ${bp7DomainPass ? 'PASS ✅' : 'FAIL ❌'}`);
    console.log(`5. Target Fitness (1.0):         ${bp7FitPass ? 'PASS ✅' : 'FAIL ❌'}`);
    console.log(`6. Logic-Only Audit:             ${bp7LogicPass ? 'PASS ✅' : 'FAIL ❌'}`);

    if (!bp7DomainPass || !bp7FitPass || !bp7LogicPass) {
        console.error("\nSTAGE 2 FAILED. Terminating Unified Proof.");
        process.exit(1);
    }

    console.log("\n--- PHASE 63.2 UNIFIED POLYGLOT PROOF: ALL PASS ✅ ---");
    process.exit(0);
}

runUnifiedProof().catch(e => {
    console.error("CRITICAL UNIFIED PROOF FAILURE:", e.message);
    process.exit(1);
});
