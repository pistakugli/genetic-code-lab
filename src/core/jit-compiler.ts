import { Genome, Gene } from './types';

/**
 * 🚀 Genetic Code Lab - JIT Compiler
 * Compiles evolved genomes into native JavaScript functions for 10-100x speedup.
 */

export type JITFunction = (x: number[], c: number[]) => number;

export class JITCompiler {
  /**
   * Compiles a genome into a native JS function
   */
  public static compile(genome: Genome, featNames?: string[]): { 
    fn: JITFunction, 
    constants: number[], 
    constantGeneIds: string[] 
  } {
    if (genome.genes.length === 0) return { fn: () => 0, constants: [], constantGeneIds: [] };

    const geneMap = new Map<string, Gene>();
    genome.genes.forEach(g => geneMap.set(g.id, g));

    const memo = new Map<string, string>();
    const constants: number[] = [];
    const constantGeneIds: string[] = [];

    function buildCode(gene: Gene): string {
      if (memo.has(gene.id)) return memo.get(gene.id)!;

      let code = '';
      if (gene.type === 'terminal') {
        if (gene.operation === 'const') {
          const cIdx = constants.length;
          constants.push(gene.operands[0] as number);
          constantGeneIds.push(gene.id);
          code = `c[${cIdx}]`; 
        } else {
          let varIdx = 0;
          if (featNames) {
            varIdx = featNames.indexOf(gene.operation);
            if (varIdx === -1) varIdx = 0;
          } else {
            const match = gene.operation.match(/\d+/);
            varIdx = match ? parseInt(match[0]) : 0;
          }
          code = `x[${varIdx}]`;
        }
      } else {
        const op = gene.operation;
        const args = gene.operands.map(opId => {
          if (typeof opId === 'string') {
            const child = geneMap.get(opId);
            return child ? buildCode(child) : '0';
          }
          return '0';
        });

        switch (op) {
          case 'add': code = `(${args[0]} + ${args[1]})`; break;
          case 'sub': code = `(${args[0]} - ${args[1]})`; break;
          case 'mul': code = `(${args[0]} * ${args[1]})`; break;
          case 'div': code = `(Math.abs(${args[1]}) > 1e-9 ? ${args[0]} / ${args[1]} : 0)`; break;
          case 'abs': code = `Math.abs(${args[0]})`; break;
          case 'sin': code = `Math.sin(${args[0]})`; break;
          case 'cos': code = `Math.cos(${args[0]})`; break;
          case 'exp': code = `Math.exp(${args[0]})`; break;
          case 'log1p': code = `Math.log1p(Math.abs(${args[0]}))`; break;
          case 'neg': code = `(-${args[0]})`; break;
          case 'sqr': code = `(${args[0]} * ${args[0]})`; break;
          case 'cube': code = `(${args[0]} * ${args[0]} * ${args[0]})`; break;
          case 'tanh': code = `Math.tanh(${args[0]})`; break;
          case 'sqrt': code = `Math.sqrt(Math.abs(${args[0]}))`; break;
          default: code = '0';
        }
      }

      memo.set(gene.id, code);
      return code;
    }

    const rootCode = buildCode(genome.genes[0]);
    
    const finalCode = `
      try {
        const res = ${rootCode};
        return (typeof res === 'number' && Number.isFinite(res)) ? res : 0;
      } catch (e) {
        return 0;
      }
    `;

    try {
      return { 
        fn: new Function('x', 'c', finalCode) as JITFunction,
        constants,
        constantGeneIds
      };
    } catch (e) {
      console.error('JIT Compilation Error:', e);
      return { fn: () => 0, constants: [], constantGeneIds: [] };
    }
  }

  /**
   * Decompiles a genome into a mathematically readable string representation.
   */
  public static decompile(genome: Genome, featNames?: string[]): string {
    if (!genome || genome.genes.length === 0) return '0';

    const geneMap = new Map<string, Gene>();
    genome.genes.forEach(g => geneMap.set(g.id, g));

    function buildMath(geneId: string): string {
      const gene = geneMap.get(geneId);
      if (!gene) return '0';

      if (gene.type === 'terminal') {
        if (gene.operation === 'const') {
          return Number(gene.operands[0]).toFixed(3);
        } else {
           if (featNames && featNames.includes(gene.operation)) return gene.operation;
           const match = gene.operation.match(/\d+/);
           const idx = match ? parseInt(match[0]) : 0;
           return featNames ? (featNames[idx] || `x${idx}`) : `x${idx}`;
        }
      }

      const args = gene.operands.map(opId => typeof opId === 'string' ? buildMath(opId) : '0');

      switch (gene.operation) {
        case 'add': return `(${args[0]} + ${args[1]})`;
        case 'sub': return `(${args[0]} - ${args[1]})`;
        case 'mul': return `(${args[0]} * ${args[1]})`;
        case 'div': return `(${args[0]} / ${args[1]})`;
        case 'abs': return `|${args[0]}|`;
        case 'sin': return `sin(${args[0]})`;
        case 'cos': return `cos(${args[0]})`;
        case 'exp': return `exp(${args[0]})`;
        case 'log1p': return `log(1 + |${args[0]}|)`;
        case 'neg': return `(-${args[0]})`;
        case 'sqr': return `(${args[0]}^2)`;
        case 'cube': return `(${args[0]}^3)`;
        case 'tanh': return `tanh(${args[0]})`;
        case 'sqrt': return `sqrt(|${args[0]}|)`;
        default: return '0';
      }
    }

    try {
      return buildMath(genome.genes[0].id);
    } catch (e) {
      return 'Err';
    }
  }
}
