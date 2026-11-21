import { describe, expect, it } from "vitest";
import {
  derivativesAt,
  graphJet,
  parseExpr,
  taylorCoefficients,
  taylorPolynomial,
  toGraph,
} from "../src/index.js";

const factorial = (k: number): number => (k <= 1 ? 1 : k * factorial(k - 1));

function expectClose(actual: number[], expected: number[], digits = 12): void {
  expect(actual).toHaveLength(expected.length);
  expected.forEach((e, k) => expect(actual[k]).toBeCloseTo(e, digits));
}

describe("taylorCoefficients: known series", () => {
  it("exp(x) at 0 is Σ xᵏ/k!", () => {
    expectClose(taylorCoefficients("exp(x)", 0, 10), Array.from({ length: 11 }, (_, k) => 1 / factorial(k)));
  });

  it("exp(x) at a is eᵃ Σ (x − a)ᵏ/k!", () => {
    const a = 0.7;
    expectClose(taylorCoefficients("exp(x)", a, 8), Array.from({ length: 9 }, (_, k) => Math.exp(a) / factorial(k)));
  });

  it("sin and cos at 0 alternate odd and even terms", () => {
    expectClose(taylorCoefficients("sin(x)", 0, 7), [0, 1, 0, -1 / 6, 0, 1 / 120, 0, -1 / 5040]);
    expectClose(taylorCoefficients("cos(x)", 0, 6), [1, 0, -1 / 2, 0, 1 / 24, 0, -1 / 720]);
  });

  it("the k-th derivative of sin at a is sin(a + kπ/2)", () => {
    const a = 1.3;
    expectClose(derivativesAt("sin(x)", a, 9), Array.from({ length: 10 }, (_, k) => Math.sin(a + (k * Math.PI) / 2)), 10);
  });

  it("log(1 + x) at 0 is x − x²/2 + x³/3 − ...", () => {
    expectClose(taylorCoefficients("log(1 + x)", 0, 8), [0, 1, -1 / 2, 1 / 3, -1 / 4, 1 / 5, -1 / 6, 1 / 7, -1 / 8]);
  });

  it("sigmoid and tanh at 0", () => {
    expectClose(taylorCoefficients("sigmoid(x)", 0, 7), [1 / 2, 1 / 4, 0, -1 / 48, 0, 1 / 480, 0, -17 / 80640]);
    expectClose(taylorCoefficients("tanh(x)", 0, 7), [0, 1, 0, -1 / 3, 0, 2 / 15, 0, -17 / 315]);
  });

  it("negative and fractional powers follow the binomial series", () => {
    expectClose(taylorCoefficients("1 / (1 - x)", 0, 6), [1, 1, 1, 1, 1, 1, 1]);
    // (1 + x)^½ = 1 + x/2 − x²/8 + x³/16 − 5x⁴/128
    expectClose(taylorCoefficients("(1 + x)^0.5", 0, 4), [1, 1 / 2, -1 / 8, 1 / 16, -5 / 128]);
  });

  it("composes: exp(sin x) = 1 + x + x²/2 − x⁴/8 − x⁵/15 + ...", () => {
    expectClose(taylorCoefficients("exp(sin(x))", 0, 5), [1, 1, 1 / 2, 0, -1 / 8, -1 / 15]);
  });
});

describe("taylorCoefficients: polynomials", () => {
  it("re-expands a cubic around a exactly and stops at its degree", () => {
    // 3x³ − 2x + 5 around a = 2: f = 25, f' = 9x² − 2 = 34, f''/2 = 9x = 18, f'''/6 = 3
    const c = taylorCoefficients("3*x^3 - 2*x + 5", 2, 6);
    expect(c).toEqual([25, 34, 18, 3, 0, 0, 0]);
    for (const x of [-3, -0.5, 0, 1.7, 4]) {
      expect(taylorPolynomial(c, 2, x)).toBeCloseTo(3 * x ** 3 - 2 * x + 5, 10);
    }
  });

  it("handles integer powers of a base that is 0 at a", () => {
    expect(taylorCoefficients("x^2", 0, 3)).toEqual([0, 0, 1, 0]);
    expect(taylorCoefficients("(x - 1)^4", 1, 5)).toEqual([0, 0, 0, 0, 1, 0]);
  });

  it("rejects series that do not exist", () => {
    expect(() => taylorCoefficients("x^-1", 0, 3)).toThrow(RangeError);
    expect(() => taylorCoefficients("x^0.5", 0, 3)).toThrow(RangeError);
    expect(() => taylorCoefficients("x", 0, -1)).toThrow(RangeError);
  });

  it("treats other names as constants and a constant expression as flat", () => {
    expect(taylorCoefficients("w*x^2 + b", 1, 3, { vars: { w: 3, b: -1 } })).toEqual([2, 6, 3, 0]);
    expect(taylorCoefficients("t^2", 2, 2, { variable: "t" })).toEqual([4, 4, 1]);
    expect(taylorCoefficients("exp(1)", 5, 2)).toEqual([Math.E, 0, 0]);
  });
});

describe("agreement with the reverse-mode engine", () => {
  const src = "sigmoid(2*x - 1) * cos(x) + log(1 + x^2) / (x + 3)";

  it("c₀ is the forward value and c₁ is backward()'s gradient", () => {
    for (const a of [-1.2, 0, 0.4, 2.5]) {
      const out = parseExpr(src, { x: a });
      out.backward();
      const x = toGraph(out).nodes.find((n) => n.label === "x")!;
      const [c0, c1] = taylorCoefficients(src, a, 1);
      expect(c0).toBeCloseTo(out.data, 12);
      expect(c1).toBeCloseTo(x.grad, 12);
    }
  });

  it("f'' matches a finite difference of the reverse-mode gradient", () => {
    const grad = (a: number): number => {
      const out = parseExpr(src, { x: a });
      out.backward();
      return toGraph(out).nodes.find((n) => n.label === "x")!.grad;
    };
    const a = 0.4;
    const h = 1e-5;
    const second = (grad(a + h) - grad(a - h)) / (2 * h);
    expect(derivativesAt(src, a, 2)[2]).toBeCloseTo(second, 5);
  });

  it("graphJet works on a graph built by hand", () => {
    const out = parseExpr("x*x*x", { x: 2 });
    const x = out.topo().find((v) => v.label === "x")!;
    expect(graphJet(out, x, 4)).toEqual([8, 12, 6, 1, 0]);
  });
});

describe("approximation error", () => {
  it("shrinks as the order grows near a", () => {
    for (const [src, f, a] of [
      ["exp(x)", Math.exp, 0.3],
      ["sin(x)", Math.sin, 1.0],
      ["log(1 + x)", (x: number) => Math.log(1 + x), 0],
      ["sigmoid(x)", (x: number) => 1 / (1 + Math.exp(-x)), 0.5],
    ] as const) {
      const c = taylorCoefficients(src, a, 8);
      const x = a + 0.4;
      const errors = [1, 2, 3, 4, 5, 6].map((n) => Math.abs(taylorPolynomial(c, a, x, n) - f(x)));
      for (let i = 1; i < errors.length; i++) expect(errors[i]!).toBeLessThan(errors[i - 1]!);
      expect(errors.at(-1)!).toBeLessThan(1e-3);
    }
  });

  it("respects the Lagrange remainder bound for exp", () => {
    // |eˣ − Pₙ(x)| ≤ e^max(a, x) · |x − a|ⁿ⁺¹ / (n + 1)!
    const a = 0;
    const c = taylorCoefficients("exp(x)", a, 12);
    for (const x of [-1, 0.5, 1.5]) {
      for (let n = 0; n <= 10; n++) {
        const bound = (Math.exp(Math.max(a, x)) * Math.abs(x - a) ** (n + 1)) / factorial(n + 1);
        expect(Math.abs(Math.exp(x) - taylorPolynomial(c, a, x, n))).toBeLessThanOrEqual(bound * (1 + 1e-9) + 1e-15);
      }
    }
  });

  it("error near a scales like |x − a|ⁿ⁺¹", () => {
    const c = taylorCoefficients("cos(x)", 0.8, 3);
    const err = (h: number): number => Math.abs(taylorPolynomial(c, 0.8, 0.8 + h) - Math.cos(0.8 + h));
    // Halving h should cut a 4th-order error by about 2⁴ = 16.
    const ratio = err(0.02) / err(0.01);
    expect(ratio).toBeGreaterThan(14);
    expect(ratio).toBeLessThan(18);
  });
});
