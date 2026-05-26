import { Genome, Gene } from './types';

/**
 * 🚀 Genetic Code Lab - WGSL Compiler (WebGPU Shading Language)
 * Converts evolved genomes into high-performance GPU compute kernels.
 * Part of the "Beyond SOTA" Scaling Phase.
 */
export class WGSLCompiler {
  /**
   * Compiles a genome into a WGSL expression string.
   * Expects input variables to be mapped to a uniform or storage buffer.
   */
  public static compile(genome: Genome, featNames?: string[]): { 
    wgsl: string, 
    constants: number[]
  } {
    if (genome.genes.length === 0) return { wgsl: '0.0', constants: [] };

    const geneMap = new Map<string, Gene>();
    genome.genes.forEach(g => geneMap.set(g.id, g));

    const memo = new Map<string, string>();
    const constants: number[] = [];

    function buildCode(gene: Gene): string {
      if (memo.has(gene.id)) return memo.get(gene.id)!;

      let code = '';
      if (gene.type === 'terminal') {
        if (gene.operation === 'const') {
          const cIdx = constants.length;
          constants.push(Number(gene.operands[0]));
          code = `constants[${cIdx}]`; 
        } else {
          let varIdx = 0;
          if (featNames) {
            varIdx = featNames.indexOf(gene.operation);
            if (varIdx === -1) varIdx = 0;
          } else {
            const match = gene.operation.match(/\d+/);
            varIdx = match ? parseInt(match[0]) : 0;
          }
          // Mapping to wgsl input buffer: inputs[index]
          code = `inputs[${varIdx}]`;
        }
      } else {
        const op = gene.operation;
        const args = gene.operands.map(opId => {
          if (typeof opId === 'string') {
            const child = geneMap.get(opId);
            return child ? buildCode(child) : '0.0';
          }
          return '0.0';
        });

        switch (op) {
          case 'add': code = `(${args[0]} + ${args[1]})`; break;
          case 'sub': code = `(${args[0]} - ${args[1]})`; break;
          case 'mul': code = `(${args[0]} * ${args[1]})`; break;
          case 'div': code = `select(${args[0]} / ${args[1]}, 0.0, abs(${args[1]}) < 1e-9)`; break;
          case 'abs': code = `abs(${args[0]})`; break;
          case 'sin': code = `sin(${args[0]})`; break;
          case 'cos': code = `cos(${args[0]})`; break;
          case 'exp': code = `exp(${args[0]})`; break;
          case 'log1p': code = `log(abs(${args[0]}) + 1.0)`; break;
          case 'neg': code = `(-${args[0]})`; break;
          case 'sqr': code = `(${args[0]} * ${args[0]})`; break;
          case 'cube': code = `(${args[0]} * ${args[0]} * ${args[0]})`; break;
          case 'tanh': code = `tanh(${args[0]})`; break;
          case 'sqrt': code = `sqrt(abs(${args[0]}))`; break;
          default: code = '0.0';
        }
      }

      memo.set(gene.id, code);
      return code;
    }

    // Root is the first gene in the list (genome root)
    const rootCode = buildCode(genome.genes[0]);
    
    return { 
      wgsl: rootCode,
      constants
    };
  }

  /**
   * Wraps the WGSL expression into a full compute shader kernel.
   * This kernel evaluates the fitness of ONE individual across ALL test cases.
   */
  public static generateFullKernel(expression: string, inputSize: number, constantSize: number): string {
    return `
      @group(0) @binding(0) var<storage, read> inputs: array<f32>;
      @group(0) @binding(1) var<storage, read> targets: array<f32>;
      @group(0) @binding(2) var<storage, read> constants: array<f32>;
      @group(0) @binding(3) var<storage, read_write> results: array<f32>;

      @compute @workgroup_size(64)
      fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
          let test_idx = global_id.x;
          let num_tests = arrayLength(&targets);

          if (test_idx >= num_tests) {
              return;
          }

          // In this architecture, inputs are flattened: [TC0_X0, TC0_X1, TC1_X0, TC1_X1...]
          // We need to point to the start of this test case's inputs
          let input_offset = test_idx * ${inputSize}u;
          
          // Helper to access inputs[i] inside the expression
          // Since the expression uses inputs[i], we need to replace it or map it.
          // For simplicity, we define a local array for the expression to use.
          var local_inputs: array<f32, ${inputSize}>;
          for (var i = 0u; i < ${inputSize}u; i = i + 1u) {
              local_inputs[i] = inputs[input_offset + i];
          }

          let prediction = ${expression.replace(/inputs\[(\d+)\]/g, 'local_inputs[$1]')};
          let target = targets[test_idx];
          
          // Standard MSE or simple absolute error
          let error = prediction - target;
          results[test_idx] = error * error;
      }
    `;
  }
}
