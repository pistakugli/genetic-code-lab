/**
 * Seed Template Builder
 * 
 * Constructs pre-structured genome skeletons for hard compositional problems.
 * These seeds provide structural scaffolding that the GP engine can't easily
 * discover through random initialization alone. The numerical optimizer then
 * fine-tunes the constants.
 * 
 * IMPORTANT: Seeds are NOT exact solutions. They are structural hints with
 * random constants that must be optimized.
 */

import { Genome, Gene } from './types';
import { generateId, random } from '../utils/helpers';

// Helper to create a terminal gene
function terminal(op: string, value?: number): Gene {
  return {
    id: generateId(),
    type: 'terminal',
    operation: op,
    operands: value !== undefined ? [value] : [],
    metadata: { terminalType: op === 'const' ? 'constant' : 'variable' }
  };
}

// Helper to create a function gene (operands filled later)
function func(op: string, arity: number): Gene {
  return {
    id: generateId(),
    type: 'function',
    operation: op,
    operands: [],
    metadata: { arity, returnType: 'number' }
  };
}

/**
 * Build a genome from a tree description.
 * Tree is described as nested arrays: ['mul', ['exp', ['mul', 'const', 'x']], ['cos', 'x']]
 */
type TreeNode = string | number | [string, ...TreeNode[]];

function buildTree(node: TreeNode): Gene[] {
  if (typeof node === 'number') {
    return [terminal('const', node)];
  }
  if (typeof node === 'string') {
    return [terminal(node)];
  }

  // It's a function call: [op, ...children]
  const [op, ...children] = node;
  const arity = children.length;
  const root = func(op, arity);
  const allGenes: Gene[] = [root];

  for (const child of children) {
    const childGenes = buildTree(child);
    root.operands.push(childGenes[0].id);
    allGenes.push(...childGenes);
  }

  return allGenes;
}

/**
 * Build a complete Genome from a tree description.
 */
export function buildSeedGenome(tree: TreeNode): Genome {
  const genes = buildTree(tree);
  return {
    id: generateId(),
    genes,
    fitness: -Infinity,
    generation: 0,
    age: 0
  };
}

/**
 * Generate a batch of seed genomes for transcendental compositional problems.
 * These are structural patterns that the GP engine struggles to assemble randomly.
 * Constants are randomized so each seed is unique but structurally correct.
 */
export function generateTranscendentalSeeds(count: number): Genome[] {
  const templates: TreeNode[] = [
    // Pattern 1: mul(exp(mul(const, x)), cos(x))  -- Damped oscillator skeleton
    ['mul', ['exp', ['mul', ['const', -0.5 + random() * 0.4], 'x']], ['cos', 'x']],
    
    // Pattern 2: mul(exp(mul(const, x)), cos(mul(const, x)))  -- Full damped oscillator
    ['mul', ['exp', ['mul', ['const', -0.3 + random() * 0.2], 'x']], ['cos', ['mul', ['const', 0.5 + random()], 'x']]],

    // Pattern 3: mul(exp(neg(x)), sin(x))  -- Decay * sin variant
    ['mul', ['exp', ['neg', ['mul', ['const', 0.05 + random() * 0.2], 'x']]], ['sin', 'x']],

    // Pattern 4: mul(exp(mul(const, x)), x)  -- Exponential growth/decay * linear
    ['mul', ['exp', ['mul', ['const', -0.2 + random() * 0.1], 'x']], 'x'],

    // Pattern 5: mul(cos(x), cos(x))  -- Trig product
    ['mul', ['cos', 'x'], ['cos', ['mul', ['const', 0.5 + random()], 'x']]],

    // Pattern 6: add(mul(exp(neg(x)), cos(x)), const)  -- damped osc + offset
    ['add', ['mul', ['exp', ['neg', ['mul', ['const', 0.05 + random() * 0.15], 'x']]], ['cos', 'x']], ['const', random() * 0.1]],

    // Pattern 7: mul(exp(mul(const, x)), sub(cos(x), sin(x)))  -- Complex oscillation
    ['mul', ['exp', ['mul', ['const', -0.2 + random() * 0.1], 'x']], ['sub', ['cos', 'x'], ['sin', 'x']]],

    // Pattern 8: exp(mul(const, x))  -- Pure decay
    ['exp', ['mul', ['const', -0.15 + random() * 0.1], 'x']],
  ];

  const seeds: Genome[] = [];
  for (let i = 0; i < count; i++) {
    // Pick a template, rebuild each time for fresh random constants
    const templateIdx = i % templates.length;
    // Re-generate the template to get fresh random constants
    const freshTemplates: TreeNode[] = [
      ['mul', ['exp', ['mul', ['const', -0.5 + random() * 0.4], 'x']], ['cos', 'x']],
      ['mul', ['exp', ['mul', ['const', -0.3 + random() * 0.2], 'x']], ['cos', ['mul', ['const', 0.5 + random()], 'x']]],
      ['mul', ['exp', ['neg', ['mul', ['const', 0.05 + random() * 0.2], 'x']]], ['sin', 'x']],
      ['mul', ['exp', ['mul', ['const', -0.2 + random() * 0.1], 'x']], 'x'],
      ['mul', ['cos', 'x'], ['cos', ['mul', ['const', 0.5 + random()], 'x']]],
      ['add', ['mul', ['exp', ['neg', ['mul', ['const', 0.05 + random() * 0.15], 'x']]], ['cos', 'x']], ['const', random() * 0.1]],
      ['mul', ['exp', ['mul', ['const', -0.2 + random() * 0.1], 'x']], ['sub', ['cos', 'x'], ['sin', 'x']]],
      ['exp', ['mul', ['const', -0.15 + random() * 0.1], 'x']],
    ];
    seeds.push(buildSeedGenome(freshTemplates[templateIdx]));
  }
  return seeds;
}
