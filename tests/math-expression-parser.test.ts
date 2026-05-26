import { compileMathExpression, evaluateMathExpression } from '../src/utils/math-expression-parser';

describe('math expression parser', () => {
  test('evaluates arithmetic with precedence and parentheses', () => {
    expect(evaluateMathExpression('x*x + 2*x + 1', { x: 3 })).toBe(16);
    expect(evaluateMathExpression('(x + 1) * (x + 1)', { x: 3 })).toBe(16);
    expect(evaluateMathExpression('2 + 3 * 4 ^ 2', {})).toBe(50);
  });

  test('supports unary minus, constants, and one-argument functions', () => {
    expect(evaluateMathExpression('-x + abs(-2)', { x: 5 })).toBe(-3);
    expect(evaluateMathExpression('sqrt(9) + cos(0)', {})).toBe(4);
    expect(evaluateMathExpression('sin(pi / 2)', {})).toBeCloseTo(1, 10);
  });

  test('compiles once and evaluates against different scopes', () => {
    const fn = compileMathExpression('x0 ^ 2 + 1');
    expect(fn({ x0: 2 })).toBe(5);
    expect(fn({ x0: 4 })).toBe(17);
  });

  test('rejects unknown identifiers instead of executing code', () => {
    expect(() => evaluateMathExpression('process.exit(1)', { x: 1 })).toThrow(/Unknown identifier|Unsupported character|Invalid expression|Invalid number/);
  });
});
