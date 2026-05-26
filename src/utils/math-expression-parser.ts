export type MathScope = Record<string, number>;

type Token =
  | { type: 'number'; value: number }
  | { type: 'identifier'; value: string }
  | { type: 'operator'; value: OperatorValue }
  | { type: 'leftParen' }
  | { type: 'rightParen' }
  | { type: 'comma' };

type RpnToken = Extract<Token, { type: 'number' | 'identifier' | 'operator' }>;
type OperatorValue = '+' | '-' | '*' | '/' | '^' | 'u-';

const FUNCTIONS: Record<string, (value: number) => number> = {
  sin: Math.sin,
  cos: Math.cos,
  tan: Math.tan,
  abs: Math.abs,
  sqrt: value => value < 0 ? Number.NaN : Math.sqrt(value),
  log: value => value <= 0 ? Number.NaN : Math.log(value),
  exp: Math.exp
};

const CONSTANTS: Record<string, number> = {
  pi: Math.PI,
  e: Math.E
};

const PRECEDENCE: Record<string, number> = {
  '+': 1,
  '-': 1,
  '*': 2,
  '/': 2,
  '^': 3,
  'u-': 4
};

const RIGHT_ASSOCIATIVE = new Set(['^', 'u-']);

function isDigit(char: string): boolean {
  return char >= '0' && char <= '9';
}

function isAlpha(char: string): boolean {
  return /[A-Za-z_]/.test(char);
}

function tokenize(expression: string): Token[] {
  const tokens: Token[] = [];
  let i = 0;
  let mayBeUnary = true;

  while (i < expression.length) {
    const char = expression[i];
    if (/\s/.test(char)) {
      i++;
      continue;
    }

    if (isDigit(char) || char === '.') {
      const start = i;
      i++;
      while (i < expression.length && (isDigit(expression[i]) || expression[i] === '.')) i++;
      const raw = expression.slice(start, i);
      const value = Number(raw);
      if (!Number.isFinite(value)) throw new Error(`Invalid number: ${raw}`);
      tokens.push({ type: 'number', value });
      mayBeUnary = false;
      continue;
    }

    if (isAlpha(char)) {
      const start = i;
      i++;
      while (i < expression.length && /[A-Za-z0-9_]/.test(expression[i])) i++;
      tokens.push({ type: 'identifier', value: expression.slice(start, i).toLowerCase() });
      mayBeUnary = false;
      continue;
    }

    if (char === '(') {
      tokens.push({ type: 'leftParen' });
      mayBeUnary = true;
      i++;
      continue;
    }

    if (char === ')') {
      tokens.push({ type: 'rightParen' });
      mayBeUnary = false;
      i++;
      continue;
    }

    if (char === ',') {
      tokens.push({ type: 'comma' });
      mayBeUnary = true;
      i++;
      continue;
    }

    if ('+-*/^'.includes(char)) {
      tokens.push({ type: 'operator', value: (char === '-' && mayBeUnary ? 'u-' : char) as OperatorValue });
      mayBeUnary = true;
      i++;
      continue;
    }

    throw new Error(`Unsupported character: ${char}`);
  }

  return tokens;
}

function toRpn(tokens: Token[]): RpnToken[] {
  const output: RpnToken[] = [];
  const operators: Token[] = [];

  for (let i = 0; i < tokens.length; i++) {
    const token = tokens[i];
    const next = tokens[i + 1];

    if (token.type === 'number') {
      output.push(token);
      continue;
    }

    if (token.type === 'identifier') {
      if (next?.type === 'leftParen') operators.push(token);
      else output.push(token);
      continue;
    }

    if (token.type === 'comma') {
      throw new Error('Functions with multiple arguments are not supported');
    }

    if (token.type === 'operator') {
      while (operators.length > 0) {
        const top = operators[operators.length - 1];
        if (top.type !== 'operator') break;
        const topPrecedence = PRECEDENCE[top.value];
        const tokenPrecedence = PRECEDENCE[token.value];
        const shouldPop = RIGHT_ASSOCIATIVE.has(token.value)
          ? tokenPrecedence < topPrecedence
          : tokenPrecedence <= topPrecedence;
        if (!shouldPop) break;
        output.push(operators.pop() as RpnToken);
      }
      operators.push(token);
      continue;
    }

    if (token.type === 'leftParen') {
      operators.push(token);
      continue;
    }

    if (token.type === 'rightParen') {
      while (operators.length > 0 && operators[operators.length - 1].type !== 'leftParen') {
        output.push(operators.pop() as RpnToken);
      }
      if (operators.length === 0) throw new Error('Mismatched parentheses');
      operators.pop();
      if (operators[operators.length - 1]?.type === 'identifier') {
        output.push(operators.pop() as RpnToken);
      }
    }
  }

  while (operators.length > 0) {
    const token = operators.pop() as Token;
    if (token.type === 'leftParen' || token.type === 'rightParen') throw new Error('Mismatched parentheses');
    output.push(token as RpnToken);
  }

  return output;
}

function evaluateRpn(tokens: RpnToken[], scope: MathScope): number {
  const stack: number[] = [];

  for (const token of tokens) {
    if (token.type === 'number') {
      stack.push(token.value);
      continue;
    }

    if (token.type === 'identifier') {
      if (token.value in FUNCTIONS) {
        const arg = stack.pop();
        if (arg === undefined) throw new Error(`Function ${token.value} missing argument`);
        stack.push(FUNCTIONS[token.value](arg));
      } else if (token.value in scope) {
        stack.push(scope[token.value]);
      } else if (token.value in CONSTANTS) {
        stack.push(CONSTANTS[token.value]);
      } else {
        throw new Error(`Unknown identifier: ${token.value}`);
      }
      continue;
    }

    if (token.value === 'u-') {
      const arg = stack.pop();
      if (arg === undefined) throw new Error('Unary minus missing argument');
      stack.push(-arg);
      continue;
    }

    const right = stack.pop();
    const left = stack.pop();
    if (left === undefined || right === undefined) throw new Error(`Operator ${token.value} missing operand`);

    switch (token.value) {
      case '+': stack.push(left + right); break;
      case '-': stack.push(left - right); break;
      case '*': stack.push(left * right); break;
      case '/': stack.push(right === 0 ? Number.NaN : left / right); break;
      case '^': stack.push(Math.pow(left, right)); break;
    }
  }

  if (stack.length !== 1) throw new Error('Invalid expression');
  return stack[0];
}

export function compileMathExpression(expression: string): (scope: MathScope) => number {
  const rpn = toRpn(tokenize(expression));
  return (scope: MathScope) => evaluateRpn(rpn, scope);
}

export function evaluateMathExpression(expression: string, scope: MathScope): number {
  return compileMathExpression(expression)(scope);
}
