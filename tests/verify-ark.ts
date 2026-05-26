import { EvolutionEngine } from '../src/core/evolution';
import { OpCode, PopulationBuffer } from '../src/core/pop-buffer';

async function verifyArk() {
    console.log("--- PHASE 34: HOLISTIC ARK VERIFIER ---");
    const size = 1000;
    const engine = new EvolutionEngine({
        functionSet: [{ name: 'add', arity: 2, implementation: (a: any, b: any) => a + b, returnType: 'number' }],
        terminalSet: [{ name: 'x', type: 'variable' }],
        fitnessFunction: () => 1.0, 
        testCases: [{ input: [1], expectedOutput: 2 }]
    }, {
        population: { 
            size, 
            eliteCount: 10,
            crossoverRate: 0.8,
            mutationRate: 0.2,
            maxGenomeDepth: 8,
            maxGenomeSize: 128,
            tournamentSize: 3,
            parsimonyCoefficient: 0.002
        },
        maxGenerations: 5
    });

    (engine as any).isBinaryMode = true;
    (engine as any).currentBuffer = new PopulationBuffer(size, 128, 10, 1);
    (engine as any).nextBuffer = new PopulationBuffer(size, 128, 10, 1);
    
    // Seed noise
    (engine as any).currentBuffer.randomize(1, [OpCode.ADD]);

    // Seed agentic champion
    const baked = { opcodes: [OpCode.CONST], data: [0], constants: [0.777] };
    (engine as any).currentBuffer.writeGenome(500, baked.opcodes, baked.data, baked.constants);
    (engine as any).currentBuffer.ancestorMap[500] = 1; 

    console.log(`[Test] Pre-Trigger Check: ancestorMap[500] = ${(engine as any).currentBuffer.ancestorMap[500]}`);
    engine.triggerBinaryCataclysm();

    // EXAMINE BOTH
    console.log("[Test] Auditing buffers Post-Cataclysm...");
    
    function checkBuffer(buf: PopulationBuffer, name: string) {
        let foundAt = -1;
        for (let i = 0; i < size; i++) {
            if (buf.ancestorMap[i] === 1) {
                const offset = buf.offsets[i];
                const op = buf.geneBuffer[offset] >>> 24;
                const cIdx = buf.geneBuffer[offset] & 0xFFFFFF;
                const val = buf.constBuffer[i * 10 + cIdx];
                if (op === OpCode.CONST && val === 0.777) {
                    foundAt = i;
                    break;
                }
            }
        }
        if (foundAt >= 0) console.log(`[PASS] Found Champion in ${name} at index ${foundAt}.`);
        else console.log(`[FAIL] No Champion in ${name}. (Agentic indices: ${buf.ancestorMap.filter(x => x === 1).length})`);
        return foundAt >= 0;
    }

    const currentSuccess = checkBuffer((engine as any).currentBuffer, "currentBuffer");
    const nextSuccess = checkBuffer((engine as any).nextBuffer, "nextBuffer");

    if (currentSuccess) {
        console.log("--- ARK INTEGRITY VERIFIED ---");
    } else {
        console.error("[FAIL] Ark did not preserve Agentic Champion in active buffer!");
        process.exit(1);
    }
}

verifyArk().catch(console.error);
