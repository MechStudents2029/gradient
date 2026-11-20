import { Value } from "./value.js";

/** Unary functions an expression may call, mapped to the matching `Value` op. */
export const FUNCTIONS = ["tanh", "exp", "log", "relu", "sigmoid"] as const;
export type FunctionName = (typeof FUNCTIONS)[number];

export class ParseError extends Error {
  constructor(message: string, readonly pos: number) {
    super(`${message} (at ${pos})`);
    this.name = "ParseError";
  }
}

type Token =
  | { kind: "num"; value: number; pos: number }
  | { kind: "id"; name: string; pos: number }
  | { kind: "sym"; sym: string; pos: number }
  | { kind: "end"; pos: number };

function tokenize(src: string): Token[] {
  const tokens: Token[] = [];
  let i = 0;
  while (i < src.length) {
    const c = src[i]!;
    if (/\s/.test(c)) {
      i++;
    } else if (/[0-9.]/.test(c)) {
      const m = /^(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?/.exec(src.slice(i));
      if (!m) throw new ParseError(`bad number`, i);
      tokens.push({ kind: "num", value: Number(m[0]), pos: i });
      i += m[0].length;
    } else if (/[A-Za-z_]/.test(c)) {
      const m = /^[A-Za-z_][A-Za-z0-9_]*/.exec(src.slice(i))!;
      tokens.push({ kind: "id", name: m[0], pos: i });
      i += m[0].length;
    } else if ("+-*/^()".includes(c)) {
      tokens.push({ kind: "sym", sym: c, pos: i });
      i++;
    } else {
      throw new ParseError(`unexpected character '${c}'`, i);
    }
  }
  tokens.push({ kind: "end", pos: src.length });
  return tokens;
}

/**
 * Parse an arithmetic expression into a graph of `Value`s.
 *
 * Grammar (usual precedence, `^` binds tightest and is right-associative):
 *
 *     expr  := term (('+' | '-') term)*
 *     term  := unary (('*' | '/') unary)*
 *     unary := '-' unary | power
 *     power := atom ('^' ['-'] number)?
 *     atom  := number | name | fn '(' expr ')' | '(' expr ')'
 *
 * Each variable name becomes a single leaf `Value` (labelled with its name), so
 * `x*x` reuses one node and its gradient accumulates from both uses. Exponents
 * must be constants because `Value.pow` takes a plain number.
 */
export function parseExpr(src: string, vars: Readonly<Record<string, number>> = {}): Value {
  const tokens = tokenize(src);
  const leaves = new Map<string, Value>();
  let k = 0;

  const peek = (): Token => tokens[k]!;
  const isSym = (s: string): boolean => {
    const t = peek();
    return t.kind === "sym" && t.sym === s;
  };
  const expect = (s: string): void => {
    if (!isSym(s)) throw new ParseError(`expected '${s}'`, peek().pos);
    k++;
  };

  const expr = (): Value => {
    let v = term();
    while (isSym("+") || isSym("-")) {
      const op = (tokens[k++] as { sym: string }).sym;
      const rhs = term();
      v = op === "+" ? v.add(rhs) : v.sub(rhs);
    }
    return v;
  };

  const term = (): Value => {
    let v = unary();
    while (isSym("*") || isSym("/")) {
      const op = (tokens[k++] as { sym: string }).sym;
      const rhs = unary();
      v = op === "*" ? v.mul(rhs) : v.div(rhs);
    }
    return v;
  };

  const unary = (): Value => {
    if (isSym("-")) {
      k++;
      return unary().neg();
    }
    return power();
  };

  const power = (): Value => {
    const base = atom();
    if (!isSym("^")) return base;
    k++;
    let sign = 1;
    if (isSym("-")) {
      sign = -1;
      k++;
    }
    const t = peek();
    if (t.kind !== "num") throw new ParseError("exponent must be a number", t.pos);
    k++;
    return base.pow(sign * t.value);
  };

  const atom = (): Value => {
    const t = peek();
    if (t.kind === "num") {
      k++;
      return new Value(t.value, [], "", String(t.value));
    }
    if (t.kind === "id") {
      k++;
      if (isSym("(")) {
        if (!(FUNCTIONS as readonly string[]).includes(t.name)) {
          throw new ParseError(`unknown function '${t.name}'`, t.pos);
        }
        k++;
        const arg = expr();
        expect(")");
        return arg[t.name as FunctionName]();
      }
      if (!(t.name in vars)) throw new ParseError(`no value for variable '${t.name}'`, t.pos);
      let leaf = leaves.get(t.name);
      if (!leaf) {
        leaf = new Value(vars[t.name]!, [], "", t.name);
        leaves.set(t.name, leaf);
      }
      return leaf;
    }
    if (isSym("(")) {
      k++;
      const v = expr();
      expect(")");
      return v;
    }
    throw new ParseError(t.kind === "end" ? "unexpected end of input" : "unexpected token", t.pos);
  };

  const result = expr();
  if (peek().kind !== "end") throw new ParseError("unexpected trailing input", peek().pos);
  result.label ||= "out";
  return result;
}

/** Parse variable bindings like `a = 2, b = -3; c=0.5` into a record. */
export function parseVars(src: string): Record<string, number> {
  const out: Record<string, number> = {};
  for (const part of src.split(/[,;\n]/)) {
    const s = part.trim();
    if (!s) continue;
    const m = /^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(\S+)$/.exec(s);
    const value = m ? Number(m[2]) : NaN;
    if (!m || !Number.isFinite(value)) throw new Error(`bad binding '${s}' (use name = number)`);
    out[m[1]!] = value;
  }
  return out;
}
