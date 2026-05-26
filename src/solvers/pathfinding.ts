/**
 * Genetic Code Lab - Pathfinding Problem
 * Evolves path-finding algorithms for grid navigation
 */

import {
  ProblemDefinition,
  TestCase,
  FunctionDefinition,
  TerminalDefinition,
  Genome,
  Gene,
  FitnessFunction
} from '../core/types';
import { random, randomChoice } from '../utils/helpers';

// ============================================================================
// TYPES
// ============================================================================

interface Position {
  x: number;
  y: number;
}

type Grid = number[][]; // 0 = empty, 1 = obstacle

type Direction = 'up' | 'down' | 'left' | 'right' | 'toward';

// ============================================================================
// DIRECTION FUNCTIONS
// ============================================================================

const DIRECTIONS: Record<Direction, Position> = {
  up: { x: 0, y: -1 },
  down: { x: 0, y: 1 },
  left: { x: -1, y: 0 },
  right: { x: 1, y: 0 },
  toward: { x: 0, y: 0 }  // Placeholder, handled specially
};

// ============================================================================
// FUNCTION SET
// ============================================================================

export const PATHFINDING_FUNCTIONS: FunctionDefinition[] = [
  {
    name: 'moveUp',
    arity: 0,
    implementation: () => 'up',
    returnType: 'direction',
    description: 'Move up'
  },
  {
    name: 'moveDown',
    arity: 0,
    implementation: () => 'down',
    returnType: 'direction',
    description: 'Move down'
  },
  {
    name: 'moveLeft',
    arity: 0,
    implementation: () => 'left',
    returnType: 'direction',
    description: 'Move left'
  },
  {
    name: 'moveRight',
    arity: 0,
    implementation: () => 'right',
    returnType: 'direction',
    description: 'Move right'
  },
  {
    name: 'moveTowardGoal',
    arity: 0,
    implementation: () => 'toward',
    returnType: 'direction',
    description: 'Move toward goal (smart)'
  },
  {
    name: 'avoidObstacles',
    arity: 1,
    implementation: (dir) => dir,
    returnType: 'direction',
    description: 'Avoid obstacles if direction blocked'
  },
  {
    name: 'sequence',
    arity: 2,
    implementation: () => 'seq',
    returnType: 'sequence',
    description: 'Execute sequence of moves'
  },
  {
    name: 'repeat',
    arity: 1,
    implementation: () => 'repeat',
    returnType: 'sequence',
    description: 'Repeat move'
  }
];

// ============================================================================
// TERMINAL SET
// ============================================================================

export const PATHFINDING_TERMINALS: TerminalDefinition[] = [
  { name: 'up', type: 'constant', value: 'up', description: 'Direction: up' },
  { name: 'down', type: 'constant', value: 'down', description: 'Direction: down' },
  { name: 'left', type: 'constant', value: 'left', description: 'Direction: left' },
  { name: 'right', type: 'constant', value: 'right', description: 'Direction: right' }
];

// ============================================================================
// GENOME EXECUTION
// ============================================================================

/**
 * Extract moves from genome
 */
export function extractMoves(genome: Genome): Direction[] {
  const moves: Direction[] = [];

  for (const gene of genome.genes) {
    if (gene.type === 'terminal') {
      const dir = gene.operation;
      if (['up', 'down', 'left', 'right', 'toward'].includes(dir)) {
        moves.push(dir as Direction);
      }
    } else if (gene.type === 'function') {
      const op = gene.operation;
      if (['moveUp', 'moveDown', 'moveLeft', 'moveRight'].includes(op)) {
        const dirMap: Record<string, Direction> = {
          moveUp: 'up',
          moveDown: 'down',
          moveLeft: 'left',
          moveRight: 'right'
        };
        moves.push(dirMap[op]);
      } else if (op === 'moveTowardGoal') {
        moves.push('toward');
      }
    }
  }

  return moves;
}

/**
 * Execute pathfinding on grid
 */
export function executePathfinding(
  grid: Grid,
  start: Position,
  goal: Position,
  moves: Direction[],
  maxSteps: number = 100
): {
  path: Position[];
  reached: boolean;
  steps: number;
  distanceToGoal: number;
} {
  const path: Position[] = [{ ...start }];
  let current = { ...start };
  let steps = 0;

  for (const move of moves) {
    if (steps >= maxSteps) break;

    let nextPos: Position;

    if (move === 'toward') {
      // Smart move toward goal
      nextPos = getSmartMove(grid, current, goal);
    } else {
      const delta = DIRECTIONS[move];
      nextPos = {
        x: current.x + delta.x,
        y: current.y + delta.y
      };
    }

    // Check bounds and obstacles
    if (isValidMove(grid, nextPos)) {
      current = nextPos;
      path.push({ ...current });
      steps++;

      // Check if reached goal
      if (current.x === goal.x && current.y === goal.y) {
        return {
          path,
          reached: true,
          steps,
          distanceToGoal: 0
        };
      }
    }
  }

  // Calculate final distance to goal
  const distance = Math.abs(current.x - goal.x) + Math.abs(current.y - goal.y);

  return {
    path,
    reached: false,
    steps,
    distanceToGoal: distance
  };
}

/**
 * Get smart move toward goal (avoiding obstacles)
 */
function getSmartMove(grid: Grid, current: Position, goal: Position): Position {
  const dx = goal.x - current.x;
  const dy = goal.y - current.y;

  // Try preferred direction first
  const preferredDir = Math.abs(dx) >= Math.abs(dy)
    ? (dx > 0 ? 'right' : 'left')
    : (dy > 0 ? 'down' : 'up');

  const delta = DIRECTIONS[preferredDir];
  const nextPos = {
    x: current.x + delta.x,
    y: current.y + delta.y
  };

  if (isValidMove(grid, nextPos)) {
    return nextPos;
  }

  // Try other directions
  const otherDirs: Direction[] = ['up', 'down', 'left', 'right'].filter(d => d !== preferredDir) as Direction[];

  for (const dir of otherDirs) {
    const d = DIRECTIONS[dir];
    const pos = { x: current.x + d.x, y: current.y + d.y };
    if (isValidMove(grid, pos)) {
      return pos;
    }
  }

  // Stuck - stay in place
  return current;
}

/**
 * Check if move is valid
 */
function isValidMove(grid: Grid, pos: Position): boolean {
  if (pos.y < 0 || pos.y >= grid.length) return false;
  if (pos.x < 0 || pos.x >= grid[0].length) return false;
  return grid[pos.y][pos.x] === 0;
}

// ============================================================================
// FITNESS FUNCTION
// ============================================================================

export interface PathfindingTestCase extends TestCase<{ grid: Grid; start: Position; goal: Position }, boolean> {
  input: {
    grid: Grid;
    start: Position;
    goal: Position;
  };
  expectedOutput: boolean;
}

export const pathfindingFitnessFunction: FitnessFunction<
  { grid: Grid; start: Position; goal: Position },
  boolean
> = (genome: Genome, testCases: PathfindingTestCase[]): number => {
  const moves = extractMoves(genome);

  if (moves.length === 0) return 0;

  let totalScore = 0;

  for (const tc of testCases) {
    const { grid, start, goal } = tc.input;
    const result = executePathfinding(grid, start, goal, moves);

    let score = 0;

    if (result.reached) {
      // Bonus for reaching goal
      score += 0.7;

      // Bonus for efficiency (fewer steps)
      const optimalSteps = Math.abs(goal.x - start.x) + Math.abs(goal.y - start.y);
      const efficiency = optimalSteps / Math.max(1, result.steps);
      score += Math.min(0.3, efficiency * 0.3);
    } else {
      // Partial credit for getting closer
      const initialDistance = Math.abs(goal.x - start.x) + Math.abs(goal.y - start.y);
      const improvement = (initialDistance - result.distanceToGoal) / Math.max(1, initialDistance);
      score += Math.max(0, improvement * 0.5);
    }

    totalScore += score;
  }

  return totalScore / testCases.length;
};

// ============================================================================
// PROBLEM FACTORY
// ============================================================================

/**
 * Generate a random grid
 */
function generateRandomGrid(
  width: number,
  height: number,
  obstacleProbability: number = 0.2
): Grid {
  const grid: Grid = [];

  for (let y = 0; y < height; y++) {
    const row: number[] = [];
    for (let x = 0; x < width; x++) {
      row.push(Math.random() < obstacleProbability ? 1 : 0);
    }
    grid.push(row);
  }

  return grid;
}

/**
 * Find random empty position
 */
function findEmptyPosition(grid: Grid): Position {
  const emptyPositions: Position[] = [];

  for (let y = 0; y < grid.length; y++) {
    for (let x = 0; x < grid[0].length; x++) {
      if (grid[y][x] === 0) {
        emptyPositions.push({ x, y });
      }
    }
  }

  return randomChoice(emptyPositions) || { x: 0, y: 0 };
}

/**
 * Ensure path exists between start and goal (using BFS)
 */
function ensurePathExists(
  grid: Grid,
  start: Position,
  goal: Position
): Grid {
  const newGrid = grid.map(row => [...row]);

  // BFS to check if path exists
  const visited = new Set<string>();
  const queue: Position[] = [start];
  visited.add(`${start.x},${start.y}`);

  while (queue.length > 0) {
    const current = queue.shift()!;

    if (current.x === goal.x && current.y === goal.y) {
      return newGrid; // Path exists
    }

    for (const dir of Object.values(DIRECTIONS)) {
      const next = { x: current.x + dir.x, y: current.y + dir.y };
      const key = `${next.x},${next.y}`;

      if (!visited.has(key) && isValidMove(newGrid, next)) {
        visited.add(key);
        queue.push(next);
      }
    }
  }

  // No path - clear some obstacles
  newGrid[goal.y][goal.x] = 0;

  // Create a simple corridor
  const midX = Math.floor((start.x + goal.x) / 2);
  const midY = Math.floor((start.y + goal.y) / 2);

  for (let x = Math.min(start.x, midX); x <= Math.max(start.x, midX); x++) {
    newGrid[start.y][x] = 0;
  }
  for (let y = Math.min(start.y, goal.y); y <= Math.max(start.y, goal.y); y++) {
    newGrid[y][midX] = 0;
  }
  for (let x = Math.min(midX, goal.x); x <= Math.max(midX, goal.x); x++) {
    newGrid[goal.y][x] = 0;
  }

  return newGrid;
}

/**
 * Create a pathfinding problem
 */
export function createPathfindingProblem(
  gridWidth: number = 10,
  gridHeight: number = 10,
  testCaseCount: number = 10
): ProblemDefinition<{ grid: Grid; start: Position; goal: Position }, boolean> {
  const testCases: PathfindingTestCase[] = [];

  for (let i = 0; i < testCaseCount; i++) {
    let grid = generateRandomGrid(gridWidth, gridHeight, 0.2);
    const start = findEmptyPosition(grid);
    let goal = findEmptyPosition(grid);

    // Ensure start and goal are different
    while (goal.x === start.x && goal.y === start.y) {
      goal = findEmptyPosition(grid);
    }

    // Ensure path exists
    grid = ensurePathExists(grid, start, goal);
    grid[start.y][start.x] = 0;
    grid[goal.y][goal.x] = 0;

    testCases.push({
      input: { grid, start, goal },
      expectedOutput: true,
      weight: 1
    });
  }

  return {
    name: 'Pathfinding',
    description: `Evolve a pathfinding algorithm for ${gridWidth}x${gridHeight} grids`,
    functionSet: PATHFINDING_FUNCTIONS,
    terminalSet: PATHFINDING_TERMINALS,
    testCases,
    fitnessFunction: pathfindingFitnessFunction,
    config: {
      population: {
        size: 150,
        eliteCount: 3,
        crossoverRate: 0.75,
        mutationRate: 0.25,
        maxGenomeDepth: 10,
        maxGenomeSize: 50,
        tournamentSize: 4
      },
      maxGenerations: 150,
      targetFitness: 0.9,
      stagnationLimit: 25
    }
  };
}

/**
 * Visualize path on grid
 */
export function visualizePath(
  grid: Grid,
  path: Position[],
  start: Position,
  goal: Position
): string {
  const lines: string[] = [];

  for (let y = 0; y < grid.length; y++) {
    let line = '';
    for (let x = 0; x < grid[0].length; x++) {
      if (x === start.x && y === start.y) {
        line += 'S ';
      } else if (x === goal.x && y === goal.y) {
        line += 'G ';
      } else if (grid[y][x] === 1) {
        line += '█ ';
      } else if (path.some(p => p.x === x && p.y === y)) {
        line += '· ';
      } else {
        line += '. ';
      }
    }
    lines.push(line);
  }

  return lines.join('\n');
}
