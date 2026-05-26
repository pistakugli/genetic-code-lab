import { DomainAnalyzer } from '../src/core/domain-analyzer';
import { DiscoveryAgent } from '../src/agents/discovery-agent';

async function runProof() {
    console.log("--- PHASE 64.2 CLASSIFICATION PROOF ---");

    const inputSize = 5;
    const numSamples = Math.pow(2, inputSize);
    const inputs = new Float64Array(numSamples * inputSize);
    const majorityOutputs = new Float64Array(numSamples);
    const parityOutputs = new Float64Array(numSamples);

    // Generate Truth Tables
    for (let i = 0; i < numSamples; i++) {
        let ones = 0;
        for (let j = 0; j < inputSize; j++) {
            const bit = (i >> j) & 1;
            inputs[i * inputSize + j] = bit;
            if (bit === 1) ones++;
        }
        majorityOutputs[i] = ones >= 3 ? 1 : 0;
        parityOutputs[i] = ones % 2 === 1 ? 1 : 0;
    }

    // 1. Test Majority Classification
    console.log("\n[TEST] Majority-5");
    const majorityProfile = DomainAnalyzer.analyze(inputs, majorityOutputs, inputSize);
    console.log(`Signals -> Sym: ${majorityProfile.signals.symmetryScore?.toFixed(4)}, Mono: ${majorityProfile.signals.monotonicityScore?.toFixed(4)}, Alt: ${majorityProfile.signals.alternationScore?.toFixed(4)}`);
    
    const agentM = new DiscoveryAgent();
    agentM.setEvaluationData(inputs, majorityOutputs, inputSize, 'classification');
    agentM.setDomainProfile(majorityProfile);
    await agentM.pushPopulationSnapshot([{ fitness: 0.1, genes: [{id: 'root', type: 'terminal', operation: 'b0', operands: []}] } as any], 20);

    // 2. Test Parity Classification
    console.log("\n[TEST] Parity-5");
    const parityProfile = DomainAnalyzer.analyze(inputs, parityOutputs, inputSize);
    console.log(`Signals -> Sym: ${parityProfile.signals.symmetryScore?.toFixed(4)}, Mono: ${parityProfile.signals.monotonicityScore?.toFixed(4)}, Alt: ${parityProfile.signals.alternationScore?.toFixed(4)}`);

    const agentP = new DiscoveryAgent();
    agentP.setEvaluationData(inputs, parityOutputs, inputSize, 'classification');
    agentP.setDomainProfile(parityProfile);
    await agentP.pushPopulationSnapshot([{ fitness: 0.1, genes: [{id: 'root', type: 'terminal', operation: 'b0', operands: []}] } as any], 20);

    console.log("\n--- PROOF COMPLETE ---");
}

runProof().catch(console.error);
