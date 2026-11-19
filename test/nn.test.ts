import { describe, expect, it } from "vitest";
import { MLP, Value, mse, numericGrad, sgdStep } from "../src/index.js";

describe("sigmoid", () => {
  it("matches the finite-difference gradient", () => {
    const x = new Value(0.7);
    const y = x.sigmoid().mul(x);
    y.backward();
    const [g] = numericGrad(([a]) => a! / (1 + Math.exp(-a!)), [0.7]);
    expect(x.grad).toBeCloseTo(g!, 6);
  });
});

describe("MLP", () => {
  it("has the right number of parameters", () => {
    // 2→4: 4·(2+1) = 12, 4→1: 1·(4+1) = 5
    expect(new MLP([2, 4, 1]).parameters()).toHaveLength(17);
  });

  it("is deterministic for a fixed seed", () => {
    const a = new MLP([2, 3, 1], { seed: 7 }).forward([0.5, -1])[0]!.data;
    const b = new MLP([2, 3, 1], { seed: 7 }).forward([0.5, -1])[0]!.data;
    expect(a).toBe(b);
  });

  it("backprop through the whole net matches finite differences", () => {
    const net = new MLP([2, 3, 1], { seed: 3 });
    const params = net.parameters();
    const loss = () => mse([net.forward([0.3, -0.8])[0]!], [0.5]);
    loss().backward();
    const analytic = params.map((p) => p.grad);
    const numeric = numericGrad(
      (xs) => {
        const saved = params.map((p) => p.data);
        xs.forEach((x, i) => (params[i]!.data = x));
        const l = loss().data;
        saved.forEach((x, i) => (params[i]!.data = x));
        return l;
      },
      params.map((p) => p.data),
    );
    analytic.forEach((g, i) => expect(g).toBeCloseTo(numeric[i]!, 5));
  });

  it("learns XOR with gradient descent", () => {
    const xs = [[0, 0], [0, 1], [1, 0], [1, 1]];
    const ys = [0, 1, 1, 0];
    const net = new MLP([2, 4, 1], { hidden: "tanh", output: "sigmoid", seed: 1 });
    let first = 0;
    let last = 0;
    for (let step = 0; step < 2000; step++) {
      const loss = mse(xs.map((x) => net.forward(x)[0]!), ys);
      if (step === 0) first = loss.data;
      last = loss.data;
      loss.backward();
      sgdStep(net.parameters(), 1.0);
    }
    expect(last).toBeLessThan(first / 10);
    xs.forEach((x, i) => expect(Math.round(net.forward(x)[0]!.data)).toBe(ys[i]));
  });
});
