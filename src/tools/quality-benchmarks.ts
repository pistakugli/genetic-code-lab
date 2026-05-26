/**
 * Genetic Code Lab - SOTA Quality Benchmarks
 */
import { ProblemDefinition } from '../core/types';

export class SotaBenchmarks {
    
    static getF3(): ProblemDefinition<number[], number> {
        return {
            name: 'F3-Rational',
            description: 'Rational function approximation: (x^2 + 1) / (x - 2)',
            functionSet: [
                { name: 'add', arity: 2, implementation: (a:any, b:any)=>a+b, returnType: 'number' },
                { name: 'sub', arity: 2, implementation: (a:any, b:any)=>a-b, returnType: 'number' },
                { name: 'mul', arity: 2, implementation: (a:any, b:any)=>a*b, returnType: 'number' },
                { name: 'div', arity: 2, implementation: (a:any, b:any)=>a/(b === 0 ? 1 : b), returnType: 'number' }
            ],
            terminalSet: [{ name: 'x0', type: 'variable' }],
            testCases: Array.from({length: 50}, (_, i) => {
                let x = -1 + (i * 0.1); 
                if (Math.abs(x - 2.0) < 1e-7) x += 0.05;
                const weight = Math.abs(x - 2.0) < 0.25 ? 20.0 : 1.0;
                return { input: [x], expectedOutput: (x*x + 1) / (x - 2), weight };
            }),
            fitnessFunction: () => 0
        };
    }

    static getClassification(): ProblemDefinition<number[], number> {
        const rng = () => Math.random();
        return {
            name: 'Circle-Classification',
            description: 'Binary classification: Is (x,y) inside the unit circle?',
            functionSet: [
                { name: 'add', arity: 2, implementation: (a:any, b:any)=>a+b, returnType: 'number' },
                { name: 'sub', arity: 2, implementation: (a:any, b:any)=>a-b, returnType: 'number' },
                { name: 'mul', arity: 2, implementation: (a:any, b:any)=>a*b, returnType: 'number' },
                { name: 'div', arity: 2, implementation: (a:any, b:any)=>a/(b === 0 ? 1 : b), returnType: 'number' }
            ],
            terminalSet: [
                { name: 'x0', type: 'variable' },
                { name: 'x1', type: 'variable' }
            ],
            testCases: Array.from({length: 100}, () => {
                const x = rng() * 2 - 1;
                const y = rng() * 2 - 1;
                return { input: [x, y], expectedOutput: (x*x + y*y < 1.0) ? 1 : 0 };
            }),
            fitnessFunction: () => 0,
            fitnessMode: 'classification' as any
        };
    }

    static getNightmareKoza3(): ProblemDefinition<number[], number> {
        return {
            name: 'Nightmare-Koza3',
            description: 'x^6 - 2x^4 + x^2',
            functionSet: [
                { name: 'add', arity: 2, implementation: (a:any, b:any)=>a+b, returnType: 'number' },
                { name: 'sub', arity: 2, implementation: (a:any, b:any)=>a-b, returnType: 'number' },
                { name: 'mul', arity: 2, implementation: (a:any, b:any)=>a*b, returnType: 'number' },
                { name: 'div', arity: 2, implementation: (a:any, b:any)=>a/(b === 0 ? 1 : b), returnType: 'number' }
            ],
            terminalSet: [{ name: 'x0', type: 'variable' }],
            testCases: Array.from({length: 100}, (_, i) => {
                const x = -1.0 + (i * 0.02);
                const target = Math.pow(x, 6) - 2 * Math.pow(x, 4) + Math.pow(x, 2);
                return { input: [x], expectedOutput: target, weight: 1.0 };
            }),
            fitnessFunction: () => 0
        };
    }

    static getPagie1(): ProblemDefinition<number[], number> {
        const testCases = [];
        for (let x = -5; x <= 5; x += 0.4) {
            for (let y = -5; y <= 5; y += 0.4) {
                const x4 = Math.pow(x, -4);
                const y4 = Math.pow(y, -4);
                const target = (isFinite(x4) ? 1 / (1 + x4) : 0) + (isFinite(y4) ? 1 / (1 + y4) : 0);
                testCases.push({ input: [x, y], expectedOutput: target, weight: 1.0 });
            }
        }
        return {
            name: 'Pagie-1',
            description: 'Classic 2D benchmark',
            functionSet: [
                { name: 'add', arity: 2, implementation: (a:any, b:any)=>a+b, returnType: 'number' },
                { name: 'sub', arity: 2, implementation: (a:any, b:any)=>a-b, returnType: 'number' },
                { name: 'mul', arity: 2, implementation: (a:any, b:any)=>a*b, returnType: 'number' },
                { name: 'div', arity: 2, implementation: (a:any, b:any)=>a/(b === 0 ? 1 : b), returnType: 'number' }
            ],
            terminalSet: [{ name: 'x0', type: 'variable' }, { name: 'x1', type: 'variable' }],
            testCases,
            fitnessFunction: () => 0
        };
    }

    static getFeynman153(): ProblemDefinition<number[], number> {
        const testCases = [];
        for (let i = 0; i < 100; i++) {
            // Increase range to avoid constant-collapse near zero
            const u = 0.1 + (Math.random() * 0.85); 
            const v = 0.1 + (Math.random() * 0.85); 
            const target = (u + v) / (1 + u * v);
            testCases.push({ input: [u, v], expectedOutput: target, weight: 1.0 });
        }
        return {
            name: 'Feynman-I.15.3',
            description: 'Relativistic addition: (u+v)/(1+u*v)',
            functionSet: [
                { name: 'add', arity: 2, implementation: (a:any, b:any)=>a+b, returnType: 'number' },
                { name: 'sub', arity: 2, implementation: (a:any, b:any)=>a-b, returnType: 'number' },
                { name: 'mul', arity: 2, implementation: (a:any, b:any)=>a*b, returnType: 'number' },
                { name: 'div', arity: 2, implementation: (a:any, b:any)=>a/(b === 0 ? 1 : b), returnType: 'number' }
            ],
            terminalSet: [{ name: 'x0', type: 'variable' }, { name: 'x1', type: 'variable' }],
            testCases,
            fitnessFunction: () => 0
        };
    }
}
