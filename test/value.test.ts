import { describe, expect, it } from "vitest";
import { Value, numericGrad } from "../src/index.js";

type Fn = (v: Value[]) => Value;

function check(fn: Fn, xs: number[]): void {
  const vars = xs.map((x) => new Value(x));
  fn(vars).backward();
  const numeric = numericGrad((ys) => fn(ys.map((y) => new Value(y))).data, xs);
  vars.forEach((v, i) => expect(v.grad).toBeCloseTo(numeric[i]!, 5));
}

describe("Value forward pass", () => {
  it("computes arithmetic", () => {
    const a = new Value(2);
    const b = new Value(-3);
    expect(a.mul(b).add(10).data).toBe(4);
    expect(a.div(4).data).toBe(0.5);
    expect(a.sub(b).data).toBe(5);
  });
});

describe("Value backward pass", () => {
  it("gives the textbook product-rule gradients", () => {
    const a = new Value(2);
    const b = new Value(-3);
    a.mul(b).backward();
    expect(a.grad).toBe(-3);
    expect(b.grad).toBe(2);
  });

  it("accumulates gradients when a node is reused", () => {
    const x = new Value(3);
    x.mul(x).add(x).backward(); // d/dx (x^2 + x) = 2x + 1
    expect(x.grad).toBe(7);
  });

  it("resets gradients between backward calls", () => {
    const x = new Value(3);
    const y = x.mul(x);
    y.backward();
    y.backward();
    expect(x.grad).toBe(6);
  });

  it("zeroes the ReLU gradient for negative inputs", () => {
    const x = new Value(-1);
    x.relu().backward();
    expect(x.grad).toBe(0);
  });

  const cases: [string, Fn, number[]][] = [
    ["polynomial", ([x]) => x!.pow(3).sub(x!.mul(2)).add(1), [1.7]],
    ["quotient", ([x, y]) => x!.div(y!), [1.3, -0.6]],
    ["exp/log", ([x]) => x!.exp().add(1).log(), [0.4]],
    ["tanh neuron", ([x, w, b]) => x!.mul(w!).add(b!).tanh(), [0.5, -1.2, 0.3]],
    ["relu chain", ([x, y]) => x!.mul(y!).relu().pow(2), [1.5, 0.8]],
    [
      "deep mix",
      ([a, b, c]) => a!.mul(b!).add(c!.pow(2)).tanh().mul(a!.exp()).sub(b!.div(c!)),
      [0.3, -0.7, 1.1],
    ],
  ];

  it.each(cases)("matches finite differences: %s", (_, fn, xs) => check(fn, xs));
});

describe("topo", () => {
  it("orders parents before children", () => {
    const a = new Value(1);
    const b = a.add(2);
    const c = b.mul(a);
    const order = c.topo();
    expect(order.indexOf(a)).toBeLessThan(order.indexOf(b));
    expect(order.indexOf(b)).toBeLessThan(order.indexOf(c));
    expect(order.at(-1)).toBe(c);
  });
});
