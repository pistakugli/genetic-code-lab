/**
 * CANONICAL F3 VERIFIER (Phase 62.1)
 * Rule: Single function, zero engine dependency, strict precision.
 */

export interface F3Proof {
    expression: string;
    predictions: number[];
    targets: number[];
    mae: number;
    honestFitness: number;
    nanCount: number;
    infCount: number;
    clampedCount: number;
}

export function verifyF3(evalFunc: (x: number) => number, numPoints: number = 100): F3Proof {
    const points = numPoints;
    const start = -5;
    const end = 5;
    const step = (end - start) / (points - 1);
    
    const predictions: number[] = [];
    const targets: number[] = [];
    let totalMAE = 0;
    let nanCount = 0;
    let infCount = 0;
    let clampedCount = 0;

    for (let i = 0; i < points; i++) {
        const x = start + i * step;
        
        // Ground Truth: f(x) = (x^2 + 1) / (x - 2)
        const den = x - 2;
        let target = (x * x + 1) / den;
        
        // Handle Singularity in Ground Truth
        if (!isFinite(target)) target = 1e12; // Standard Sentinel
        if (Math.abs(target) > 1e11) target = 1e12;

        let pred = evalFunc(x);
        
        if (isNaN(pred)) nanCount++;
        if (!isFinite(pred)) infCount++;
        
        // Measure Error
        let diff = Math.abs(pred - target);
        
        // Phase 56: Sentinel Alignment
        if (Math.abs(target) > 1e11) {
            diff = Math.abs(pred - 1e12);
        }

        predictions.push(pred);
        targets.push(target);
        totalMAE += diff;
    }

    const mae = totalMAE / points;
    const honestFitness = 1.0 / (1.0 + mae);

    return {
        expression: "", // To be filled by caller
        predictions,
        targets,
        mae,
        honestFitness,
        nanCount,
        infCount,
        clampedCount
    };
}

// Minimal Running Proof
if (require.main === module) {
    console.log("--- PROOF CANONICAL F3 RUN ---");
    // Ideal: (x^2+1)/(x-2)
    const ideal = (x: number) => {
        const den = x - 2;
        if (Math.abs(den) < 1e-12) return 1e12;
        return (x * x + 1) / den;
    };
    
    const proof = verifyF3(ideal);
    console.log(`MAE: ${proof.mae.toFixed(15)}`);
    console.log(`HonestFitness: ${proof.honestFitness.toFixed(15)}`);
    console.log(`Singularities: NaN=${proof.nanCount}, Inf=${proof.infCount}`);
}
