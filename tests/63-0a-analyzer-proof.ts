import { DomainAnalyzer } from '../src/core/domain-analyzer';

/**
 * 63.0 A: Domain Analyzer Proof
 * Verifies that the analyzer can distinguish between continuous and discrete datasets.
 */

function runAnalyzerProof() {
    console.log("--- 63.0 A: DOMAIN ANALYZER PROOF ---");

    // 1. Continuous Dataset (F3-Rational)
    const contInputs = new Float64Array(10);
    const contOutputs = new Float64Array(10);
    for (let i = 0; i < 10; i++) {
        contInputs[i] = i;
        contOutputs[i] = (i * i + 1) / (i - 2.5); // Random smooth rational
    }

    const contProfile = DomainAnalyzer.analyze(contInputs, contOutputs, 1);
    console.log(`Continuous Dataset: ${contProfile.domainClass} (Confidence: ${contProfile.confidence.toFixed(2)})`);
    console.log("Signals:", JSON.stringify(contProfile.signals));
    
    if (contProfile.domainClass !== 'continuous') {
        console.error("FAIL: F3-Rational misclassified as discrete.");
        process.exit(1);
    }

    // 2. Discrete Dataset (Boolean Parity-3)
    const discInputs = new Float64Array(8 * 3);
    const discOutputs = new Float64Array(8);
    for (let i = 0; i < 8; i++) {
        let ones = 0;
        for (let j = 0; j < 3; j++) {
            const bit = (i >> j) & 1;
            discInputs[i * 3 + j] = bit;
            if (bit === 1) ones++;
        }
        discOutputs[i] = ones % 2;
    }

    const discProfile = DomainAnalyzer.analyze(discInputs, discOutputs, 3);
    console.log(`\nDiscrete Dataset: ${discProfile.domainClass} (Confidence: ${discProfile.confidence.toFixed(2)})`);
    console.log("Signals:", JSON.stringify(discProfile.signals));

    if (discProfile.domainClass !== 'discrete') {
        console.error("FAIL: Boolean Parity misclassified as continuous.");
        process.exit(1);
    }

    console.log("\n63.0 A: STATUS: PASS");
}

runAnalyzerProof();
