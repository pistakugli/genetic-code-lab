import { DiscoveryAgent } from '../src/agents/discovery-agent';
import { DomainProfile } from '../src/core/domain-analyzer';
import { Genome } from '../src/core/types';

async function runSpamAudit() {
    console.log("--- STARTING SPAM AUDIT ---");
    
    const agent = new DiscoveryAgent();
    let logBuffer: string[] = [];
    const originalLog = console.log;
    const originalError = console.error;
    
    // Capture logs
    console.log = (...args: any[]) => {
        logBuffer.push(args.join(' '));
        originalLog(...args);
    };
    console.error = (...args: any[]) => {
        logBuffer.push(args.join(' '));
        originalError(...args);
    };

    const discreteProfile: DomainProfile = {
        domainClass: 'discrete',
        confidence: 1.0,
        signals: {
            cardinality: 2,
            range: [0, 1],
            isBinary: true,
            inputDensity: 1.0,
            entropy: 1.0,
            symmetryScore: 1.0,
            monotonicityScore: 1.0,
            alternationScore: 0.1
        },
        recommendations: {
            evaluator: 'bitwise',
            refiner: 'boolean',
            operatorSet: 'logic',
            inputWidth: 5
        }
    };

    const continuousProfile: DomainProfile = {
        domainClass: 'continuous',
        confidence: 1.0,
        signals: {
            cardinality: 1000,
            range: [-10, 10],
            isBinary: false,
            inputDensity: 1.0,
            entropy: 1.0
        },
        recommendations: {
            evaluator: 'standard',
            refiner: 'numerical',
            operatorSet: 'arithmetic',
            inputWidth: 1
        }
    };

    // --- TEST 1: Discrete Stagnation ---
    console.log("\n[AUDIT] Stage 1: Discrete Stagnation (Majority)");
    logBuffer = [];
    agent.setDomainProfile(discreteProfile);
    agent.setEvaluationData(new Float64Array(160), new Float64Array(32), 5, 'classification');

    for (let gen = 0; gen < 50; gen++) {
        await agent.pushPopulationSnapshot([{ fitness: 0.1, genes: [{ id: 'dummy', type: 'terminal', operation: 'b0', operands: [] }] }] as any, gen);
    }

    const discreteInjections = agent.pullAgentCandidates(50).filter(c => c.log.actionApplied === 'LOGIC_PIVOT');
    const discretePivotLogs = logBuffer.filter(l => l.includes('Strategy Pivot: MAJORITY'));
    const discreteRegimeLogs = logBuffer.filter(l => l.includes('Regime set to: DISCRETE'));

    console.log(`Discrete Injections: ${discreteInjections.length}`);
    console.log(`Discrete Pivot Logs: ${discretePivotLogs.length}`);
    console.log(`Discrete Regime Logs: ${discreteRegimeLogs.length}`);

    if (discreteInjections.length !== 1) throw new Error(`Discrete Spam Detected: ${discreteInjections.length} injections!`);
    if (discretePivotLogs.length !== 1) throw new Error(`Discrete Log Spam Detected: ${discretePivotLogs.length} pivots!`);
    if (discreteRegimeLogs.length !== 1) throw new Error(`Discrete Regime Log Spam! Count: ${discreteRegimeLogs.length}`);

    // --- TEST 2: Continuous Stagnation ---
    console.log("\n[AUDIT] Stage 2: Continuous Stagnation (Asymptotic)");
    logBuffer = [];
    agent.setDomainProfile(continuousProfile);
    agent.currentAction = 'STRUCTURAL_PIVOT'; 
    agent.setEvaluationData(new Float64Array(100), new Float64Array(100), 1, 'regression');

    // Simulate asymptote
    const testX = new Float64Array(100);
    const testY = new Float64Array(100);
    for(let i=0; i<100; i++) { testX[i] = i; testY[i] = i === 50 ? 1e12 : 1; }
    agent.setEvaluationData(testX, testY, 1, 'regression');
    agent.currentAction = 'STRUCTURAL_PIVOT'; 

    for (let gen = 51; gen < 101; gen++) {
        await agent.pushPopulationSnapshot([{ fitness: 0.1, genes: [{ id: 'root', type: 'terminal', operation: 'x', operands: [] }] }] as any, gen);
    }

    const continuousInjections = agent.pullAgentCandidates(101).filter(c => c.log.actionApplied === 'structural-pivot-injection');
    const continuousPivotLogs = logBuffer.filter(l => l.includes('Strategy Pivot: STRUCTURAL_PIVOT'));
    const continuousRegimeLogs = logBuffer.filter(l => l.includes('Regime set to: CONTINUOUS'));

    console.log(`Continuous Injections: ${continuousInjections.length}`);
    console.log(`Continuous Pivot Logs: ${continuousPivotLogs.length}`);
    console.log(`Continuous Regime Logs: ${continuousRegimeLogs.length}`);

    if (continuousInjections.length !== 1) throw new Error(`Continuous Spam Detected: ${continuousInjections.length} injections!`);
    if (continuousPivotLogs.length !== 1) throw new Error(`Continuous Log Spam Detected: ${continuousPivotLogs.length} pivots!`);
    if (continuousRegimeLogs.length !== 1) throw new Error(`Continuous Regime Log Spam! Count: ${continuousRegimeLogs.length}`);

    console.log("\n--- SPAM AUDIT: COMPLETE PASS ✅ ---");
    
    // Restore logs
    console.log = originalLog;
    console.error = originalError;
}

runSpamAudit().catch(e => {
    console.error(`\n--- SPAM AUDIT: FAILED ❌ ---\n${e.message}`);
    process.exit(1);
});
