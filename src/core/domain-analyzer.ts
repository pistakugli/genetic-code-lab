/**
 * Genetic Code Lab - Domain Analyzer
 * Classifies problems into Continuous, Discrete, or Mixed domains.
 */

export interface DomainProfile {
    domainClass: 'continuous' | 'discrete' | 'mixed';
    confidence: number;
    signals: {
        cardinality: number;
        range: [number, number];
        isBinary: boolean;
        inputDensity: number;
        entropy: number;
        // Phase 64: Logic-Specific Signals
        symmetryScore?: number;
        influenceMap?: number[];
        selectorMap?: number[];
        outputBalance?: number;
        weightProfile?: number[];
        monotonicityScore?: number;
        alternationScore?: number;
        logicThreshold?: number;
    };
    recommendations: {
        evaluator: 'standard' | 'bitwise';
        refiner: 'numerical' | 'boolean' | 'none';
        operatorSet: 'arithmetic' | 'logic' | 'mixed';
        inputWidth?: number;
    };
}

export class DomainAnalyzer {

    public static analyze(inputs: Float64Array, outputs: Float64Array, inputSize: number): DomainProfile {
        const numSamples = outputs.length;
        if (numSamples === 0) throw new Error("Analyzer: No samples provided.");

        const uniqueOutputs = new Set<number>();
        let min = Infinity;
        let max = -Infinity;
        let isBinary = true;
        let sumOutputs = 0;

        for (let i = 0; i < numSamples; i++) {
            const y = outputs[i];
            uniqueOutputs.add(y);
            if (y < min) min = y;
            if (y > max) max = y;
            if (y !== 0 && y !== 1) isBinary = false;
            sumOutputs += y;
        }

        const cardinality = uniqueOutputs.size;
        const outputBalance = sumOutputs / numSamples;

        const uniqueInputs = new Set<string>();
        for (let i = 0; i < numSamples; i++) {
            const pattern = inputs.subarray(i * inputSize, (i + 1) * inputSize).join(',');
            uniqueInputs.add(pattern);
        }
        const inputDensity = uniqueInputs.size / numSamples;

        const freqMap = new Map<number, number>();
        uniqueOutputs.forEach(val => freqMap.set(val, 0));
        for (let i = 0; i < numSamples; i++) {
            const y = outputs[i];
            freqMap.set(y, freqMap.get(y)! + 1);
        }
        let entropy = 0;
        for (const count of freqMap.values()) {
            const p = count / numSamples;
            entropy -= p * Math.log2(p);
        }

        let domainClass: 'continuous' | 'discrete' | 'mixed' = 'continuous';
        let confidence = 0.8;

        if (isBinary && cardinality <= 2) {
            domainClass = 'discrete';
            confidence = 0.95;
        } else if (cardinality < numSamples * 0.1 && cardinality < 50) {
            domainClass = 'discrete';
            confidence = 0.7;
        }

        // Logic Signals Calculation (if discrete)
        let symmetryScore: number | undefined;
        let influenceMap: number[] | undefined;
        let selectorMap: number[] | undefined;
        let weightProfile: number[] | undefined;
        let monotonicityScore: number | undefined;
        let alternationScore: number | undefined;
        let logicThreshold: number | undefined;

        if (domainClass === 'discrete' && isBinary) {
            // 1. Influence Map: Partial derivatives approximation
            influenceMap = new Array(inputSize).fill(0);
            for (let b = 0; b < inputSize; b++) {
                let flips = 0;
                const map = new Map<string, number>();
                for (let i = 0; i < numSamples; i++) {
                    const row = inputs.subarray(i * inputSize, (i + 1) * inputSize);
                    const key = Array.from(row).map((v, idx) => idx === b ? 'X' : v).join(',');
                    if (map.has(key)) {
                        if (map.get(key) !== outputs[i]) flips++;
                    } else {
                        map.set(key, outputs[i]);
                    }
                }
                influenceMap[b] = flips / (numSamples / 2 || 1);
            }

            // 2. Symmetry Score: Variance within Hamming weights
            const weightOutputs = new Map<number, number[]>();
            for (let i = 0; i < numSamples; i++) {
                const row = inputs.subarray(i * inputSize, (i + 1) * inputSize);
                const weight = row.reduce((sum, v) => sum + (v > 0.5 ? 1 : 0), 0);
                if (!weightOutputs.has(weight)) weightOutputs.set(weight, []);
                weightOutputs.get(weight)!.push(outputs[i]);
            }

            const sortedWeights = Array.from(weightOutputs.keys()).sort((a, b) => a - b);
            weightProfile = sortedWeights.map(w => {
                const vals = weightOutputs.get(w)!;
                return vals.reduce((s, v) => s + v, 0) / vals.length;
            });

            let totalWeightVariance = 0;
            for (const vals of weightOutputs.values()) {
                const mean = vals.reduce((s, v) => s + v, 0) / vals.length;
                const variance = vals.reduce((s, v) => s + Math.pow(v - mean, 2), 0) / (vals.length || 1);
                totalWeightVariance += variance;
            }
            symmetryScore = 1.0 - (totalWeightVariance / (weightOutputs.size || 1));

            // 3. Monotonicity & Alternation
            if (weightProfile.length >= 2) {
                let inc = 0;
                let alts = 0;
                for (let i = 1; i < weightProfile.length; i++) {
                    if (weightProfile[i] >= weightProfile[i - 1]) inc++;
                    alts += Math.abs(weightProfile[i] - weightProfile[i - 1]);
                }
                monotonicityScore = inc / (weightProfile.length - 1);
                alternationScore = alts / (weightProfile.length - 1);
            } else {
                monotonicityScore = 1.0;
                alternationScore = 0.0;
            }

            // 4. Selector Map: Detection of high-influence control bits (MUX detection)
            selectorMap = influenceMap.map(inf => inf > 0.4 ? 1 : 0);

            // 5. Logic Threshold Detection (Phase 65)
            if (monotonicityScore > 0.9 && weightProfile.length > 0) {
                for (let w = 0; w < weightProfile.length; w++) {
                    if (weightProfile[w] >= 0.5) {
                        // The sortedWeights array maps index to actual Hamming weight
                        const actualWeights = Array.from(weightOutputs.keys()).sort((a,b) => a-b);
                        logicThreshold = actualWeights[w];
                        break;
                    }
                }
            }
        }

        const recommendations: any = {};
        if (domainClass === 'discrete') {
            recommendations.evaluator = 'bitwise';
            recommendations.refiner = 'boolean';
            recommendations.operatorSet = 'logic';
        } else {
            recommendations.evaluator = 'standard';
            recommendations.operatorSet = 'arithmetic';
            recommendations.refiner = 'numerical';
        }
        recommendations.inputWidth = inputSize;

        return {
            domainClass,
            confidence,
            signals: {
                cardinality,
                range: [min, max],
                isBinary,
                inputDensity,
                entropy,
                outputBalance,
                symmetryScore,
                influenceMap,
                selectorMap,
                weightProfile,
                monotonicityScore,
                alternationScore,
                logicThreshold
            },
            recommendations: recommendations as any
        };
    }
}
