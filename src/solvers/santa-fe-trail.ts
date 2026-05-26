import { FunctionDefinition, Genome, ProblemDefinition, TerminalDefinition } from '../core/types';

export type SantaFeTraceEvent = {
  step: number;
  x: number;
  y: number;
  foodCollected: number;
};

export type SantaFeSimulationResult = {
  foodCollected: number;
  totalFood: number;
  stepsUsed: number;
  fitness: number;
  trace: SantaFeTraceEvent[];
};

const SANTA_FE_TRAIL_ROWS = [
  'S###............................',
  '...#............................',
  '...#.....................###....',
  '...#....................#....#..',
  '...#....................#....#..',
  '...####.#####........##.........',
  '............#................#..',
  '............#.......#...........',
  '............#.......#........#..',
  '............#.......#...........',
  '....................#...........',
  '............#................#..',
  '............#...................',
  '............#.......#.....###...',
  '............#.......#..#........',
  '.................#..............',
  '................................',
  '............#...........#.......',
  '............#...#..........#....',
  '............#...#...............',
  '............#...#...............',
  '............#...#.........#.....',
  '............#..........#........',
  '............#...................',
  '...##..#####....#...............',
  '.#..............#...............',
  '.#..............#...............',
  '.#......#######.................',
  '.#.....#........................',
  '.......#........................',
  '..####..........................',
  '................................'
];

const GRID_SIZE = 32;
const MAX_STEPS = 600;

function normalizeOperation(operation: string): string {
  switch (operation) {
    case 'ifFoodAhead':
    case 'if-food-ahead':
    case 'if_food_ahead':
      return 'ifFoodAhead';
    case 'prog2':
    case 'progn2':
      return 'prog2';
    case 'prog3':
    case 'progn3':
      return 'prog3';
    case 'turnLeft':
    case 'left':
      return 'turnLeft';
    case 'turnRight':
    case 'right':
      return 'turnRight';
    default:
      return operation;
  }
}

function parseSantaFeTrail(): { grid: number[][]; startX: number; startY: number; totalFood: number } {
  if (SANTA_FE_TRAIL_ROWS.length !== GRID_SIZE) {
    throw new Error(`Santa Fe trail must have ${GRID_SIZE} rows, got ${SANTA_FE_TRAIL_ROWS.length}`);
  }

  let startX = -1;
  let startY = -1;
  let totalFood = 0;
  const grid = SANTA_FE_TRAIL_ROWS.map((row, y) => {
    if (row.length !== GRID_SIZE) {
      throw new Error(`Santa Fe row ${y} must have ${GRID_SIZE} columns, got ${row.length}`);
    }
    return row.split('').map((cell, x) => {
      if (cell === 'S') {
        if (startX !== -1) throw new Error('Santa Fe trail must contain exactly one start cell');
        startX = x;
        startY = y;
        return 0;
      }
      if (cell === '#') {
        totalFood++;
        return 1;
      }
      if (cell === '.') return 0;
      throw new Error(`Unsupported Santa Fe cell '${cell}' at ${x},${y}`);
    });
  });

  if (startX < 0 || startY < 0) throw new Error('Santa Fe trail is missing start cell S');
  if (totalFood !== 89) throw new Error(`Santa Fe trail must contain 89 food cells, got ${totalFood}`);
  return { grid, startX, startY, totalFood };
}

const CANONICAL_TRAIL = parseSantaFeTrail();

export function getSantaFeFoodCount(): number {
  return CANONICAL_TRAIL.totalFood;
}

export function simulateSantaFeTrail(genome: Genome, maxSteps: number = MAX_STEPS): SantaFeSimulationResult {
  const grid = CANONICAL_TRAIL.grid.map(row => [...row]);
  const nodeMap = new Map(genome.genes.map(gene => [gene.id, gene]));
  const root = genome.genes[0];
  if (!root) {
    return { foodCollected: 0, totalFood: CANONICAL_TRAIL.totalFood, stepsUsed: 0, fitness: 0, trace: [] };
  }

  let x = CANONICAL_TRAIL.startX;
  let y = CANONICAL_TRAIL.startY;
  let dx = 1;
  let dy = 0;
  let steps = 0;
  let foodCollected = 0;
  const trace: SantaFeTraceEvent[] = [];

  const executeNode = (nodeId: string): void => {
    if (steps >= maxSteps || foodCollected >= CANONICAL_TRAIL.totalFood) return;
    const node = nodeMap.get(nodeId);
    if (!node) return;

    const operation = normalizeOperation(node.operation);
    if (node.type === 'terminal') {
      if (operation === 'move') {
        steps++;
        x = (x + dx + GRID_SIZE) % GRID_SIZE;
        y = (y + dy + GRID_SIZE) % GRID_SIZE;
        if (grid[y][x] === 1) {
          foodCollected++;
          grid[y][x] = 0;
          trace.push({ step: steps, x, y, foodCollected });
        }
      } else if (operation === 'turnLeft') {
        steps++;
        const oldDx = dx;
        dx = dy;
        dy = -oldDx;
      } else if (operation === 'turnRight') {
        steps++;
        const oldDx = dx;
        dx = -dy;
        dy = oldDx;
      }
      return;
    }

    if (operation === 'ifFoodAhead') {
      const nx = (x + dx + GRID_SIZE) % GRID_SIZE;
      const ny = (y + dy + GRID_SIZE) % GRID_SIZE;
      const branch = grid[ny][nx] === 1 ? node.operands[0] : node.operands[1];
      if (typeof branch === 'string') executeNode(branch);
    } else if (operation === 'prog2') {
      for (const child of node.operands.slice(0, 2)) {
        if (typeof child === 'string') executeNode(child);
        if (steps >= maxSteps || foodCollected >= CANONICAL_TRAIL.totalFood) break;
      }
    } else if (operation === 'prog3') {
      for (const child of node.operands.slice(0, 3)) {
        if (typeof child === 'string') executeNode(child);
        if (steps >= maxSteps || foodCollected >= CANONICAL_TRAIL.totalFood) break;
      }
    }
  };

  while (steps < maxSteps && foodCollected < CANONICAL_TRAIL.totalFood) {
    executeNode(root.id);
  }

  return {
    foodCollected,
    totalFood: CANONICAL_TRAIL.totalFood,
    stepsUsed: steps,
    fitness: foodCollected / CANONICAL_TRAIL.totalFood,
    trace
  };
}

export function createSantaFeTrailProblem(): ProblemDefinition<Record<string, never>, number> {
  const functionSet: FunctionDefinition[] = [
    { name: 'ifFoodAhead', arity: 2, implementation: () => null, returnType: 'void' },
    { name: 'prog2', arity: 2, implementation: () => null, returnType: 'void' },
    { name: 'prog3', arity: 3, implementation: () => null, returnType: 'void' }
  ];

  const terminalSet: TerminalDefinition[] = [
    { name: 'move', type: 'void', description: 'Move forward' },
    { name: 'turnLeft', type: 'void', description: 'Turn left' },
    { name: 'turnRight', type: 'void', description: 'Turn right' }
  ];

  return {
    name: 'Santa Fe Trail',
    description: 'Classic 32x32 Santa Fe artificial ant problem with 89 food cells and 600-step budget',
    functionSet,
    terminalSet,
    testCases: [{ input: {}, expectedOutput: 1.0 }],
    fitnessFunction: (genome: Genome) => simulateSantaFeTrail(genome).fitness
  };
}
