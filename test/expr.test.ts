import { describe, expect, it } from "vitest";
import { ParseError, Value, numericGrad, parseExpr, parseVars, toGraph } from "../src/index.js";

describe("parseExpr", () => {
  it("respects precedence and associativity", () => {
    expect(parseExpr("1 + 2 * 3").data).toBe(7);
    expect(parseExpr("(1 + 2) * 3").data).toBe(9);
    expect(parseExpr("8 - 3 - 2").data).toBe(3);
    expect(parseExpr("12 / 3 / 2").data).toBe(2);
    expect(parseExpr("-2^2").data).toBe(-4);
    expect(parseExpr("2^-1").data).toBe(0.5);
  });

  it("evaluates functions and variables", () => {
    const v = parseExpr("tanh(a*b + c)", { a: 2, b: -3, c: 10 });
    expect(v.data).toBeCloseTo(Math.tanh(4));
    expect(v.op).toBe("tanh");
    expect(parseExpr("sigmoid(0) + relu(-1) + exp(0) + log(1)").data).toBeCloseTo(1.5);
  });

  it("shares one leaf per variable so gradients accumulate", () => {
    const y = parseExpr("x*x + x", { x: 3 });
    y.backward();
    const leaves = toGraph(y).nodes.filter((n) => n.label === "x");
    expect(leaves).toHaveLength(1);
    expect(leaves[0]!.grad).toBe(7);
  });

  it("produces gradients that match finite differences", () => {
    const src = "sigmoid(w1*x1 + w2*x2 + b) / (1 + a^2)";
    const names = ["w1", "x1", "w2", "x2", "b", "a"];
    const xs = [0.4, -1.1, 2.0, 0.3, -0.2, 0.7];
    const bind = (vals: number[]) => Object.fromEntries(names.map((n, i) => [n, vals[i]!]));
    const out = parseExpr(src, bind(xs));
    out.backward();
    const grads = Object.fromEntries(toGraph(out).nodes.map((n) => [n.label, n.grad]));
    const numeric = numericGrad((vals) => parseExpr(src, bind(vals)).data, xs);
    names.forEach((n, i) => expect(grads[n]).toBeCloseTo(numeric[i]!, 5));
  });

  it("labels the output node", () => {
    expect(parseExpr("a + 1", { a: 1 }).label).toBe("out");
  });

  it("reports useful errors", () => {
    expect(() => parseExpr("a + ", { a: 1 })).toThrow(ParseError);
    expect(() => parseExpr("q * 2")).toThrow(/no value for variable 'q'/);
    expect(() => parseExpr("sqrt(1)")).toThrow(/unknown function 'sqrt'/);
    expect(() => parseExpr("2 ^ x", { x: 1 })).toThrow(/exponent must be a number/);
    expect(() => parseExpr("(1 + 2")).toThrow(/expected '\)'/);
    expect(() => parseExpr("1 $ 2")).toThrow(/unexpected character/);
    expect(() => parseExpr("1 2")).toThrow(/trailing/);
  });

  it("returns real Value nodes", () => {
    expect(parseExpr("1")).toBeInstanceOf(Value);
  });
});

describe("parseVars", () => {
  it("reads name = number bindings", () => {
    expect(parseVars("a = 2, b=-3; c = 1e-1\n")).toEqual({ a: 2, b: -3, c: 0.1 });
    expect(parseVars("  ")).toEqual({});
  });

  it("rejects malformed bindings", () => {
    expect(() => parseVars("a = two")).toThrow(/bad binding/);
    expect(() => parseVars("= 3")).toThrow(/bad binding/);
  });
});
