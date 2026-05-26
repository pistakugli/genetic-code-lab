import * as fs from 'fs';
import * as path from 'path';

const funcs: Record<string, (x: number[]) => number> = {
    'P1': (x: number[]) => x[0] / (1 + x[0] * x[0]),
    'P2': (x: number[]) => (x[0] * x[0] + 1) / (x[0] + 1),
    'P3': (x: number[]) => (2 * x[0] + 1) / (x[0] * x[0] + 1),
    'P4': (x: number[]) => (x[0] + x[1]) / (1 + x[0] * x[1]),
    'P5': (x: number[]) => (x[0] - x[1]) / (1 - x[0] * x[1]),
    'A3': (x: number[]) => (x[0]*x[1] + x[0] - x[1]) / (1 + x[0]*x[0] + x[1]*x[1])
};

function evaluateFormula(formulaStr: string, x0: number, x1: number = 0): number {
    if (!formulaStr || formulaStr.trim() === '') return NaN;
    try {
        let jsExpr = formulaStr
            .replace(/\[/g, '(')
            .replace(/\]/g, ')')
            .replace(/x0/g, `(${x0})`)
            .replace(/x1/g, `(${x1})`);
        return eval(jsExpr);
    } catch (e) {
        return NaN;
    }
}

function processLogFile(filePath: string, problemPrefix: string) {
    if (!fs.existsSync(filePath)) {
        console.log(`File not found: ${filePath}`);
        return;
    }
    const lines = fs.readFileSync(filePath, 'utf8').split('\n');
    const tableLines = lines.filter(l => l.includes(` | ${problemPrefix}`) && l.includes(' | '));

    const points1D = [[0], [1], [-0.5], [2.492], [-0.999], [0.767], [4.061]];
    const points2D = [[0, 0], [0.5, 0.5], [-0.5, 0.5], [0.370, 0.448], [0.052, -0.411]];

    for (const line of tableLines) {
        const parts = line.split('|').map(s => s.trim());
        if (parts.length < 5) continue;
        const prob = parts[1];
        
        const formulaMatch = line.match(/`([^`]+)`/);
        if (!formulaMatch) continue;
        const formula = formulaMatch[1];

        if (!funcs[prob]) continue;

        console.log(`\nChecking: ${prob} -> ${formula}`);
        const is2D = prob === 'P4' || prob === 'P5' || prob === 'A3';
        const points = is2D ? points2D : points1D;

        let allValid = true;
        console.log(' | Point | Target | Raw Candidate | Difference | Status |');
        for (const pt of points) {
            const target = funcs[prob](pt);
            const rawVal = evaluateFormula(formula, pt[0], pt[1] || 0);
            const diff = Math.abs(target - rawVal);
            
            const status = (isNaN(rawVal) || !isFinite(rawVal) || diff > 0.1) ? 'INVALID' : 'VALID';
            if (status === 'INVALID') allValid = false;
            
            console.log(` | [${pt.join(', ')}] | ${target.toFixed(6)} | ${rawVal.toFixed(6)} | ${diff.toFixed(6)} | ${status} |`);
        }
        console.log(` -> VERDICT: ${allValid ? 'POINTWISE PASSED' : 'POINTWISE FAILED'}`);
    }
}

console.log("=== FINAL POINTWISE RAW PROOF ===\n");
const packDir = path.join(process.cwd(), 'proof_pack');
processLogFile(path.join(packDir, '73_family.txt'), 'P');
processLogFile(path.join(packDir, '74_1_a3.txt'), 'A3');
