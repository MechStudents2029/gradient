import { describe, expect, it } from "vitest";
import { MLP, Trainer, accuracy, decisionGrid, makeDataset, mse, predict, trainEpoch } from "../src/index.js";
import type { DatasetName } from "../src/index.js";

describe("predict", () => {
  it("matches the Value forward pass for every activation", () => {
    for (const [hidden, output] of [["tanh", "sigmoid"], ["relu", "linear"], ["sigmoid", "tanh"]] as const) {
      const net = new MLP([2, 5, 3, 2], { hidden, output, seed: 11 });
      for (const x of [[0.3, -0.7], [1.2, 0.4], [-1, -1]]) {
        const fast = predict(net, x);
        const slow = net.forward(x).map((v) => v.data);
        fast.forEach((y, i) => expect(y).toBeCloseTo(slow[i]!, 12));
      }
    }
  });
});

describe("trainEpoch", () => {
  it("returns the pre-update MSE and moves every weight against its gradient", () => {
    const data = makeDataset("xor", { n: 20, seed: 1 });
    const net = new MLP([2, 4, 1], { hidden: "tanh", output: "sigmoid", seed: 2 });
    const before = mse(data.xs.map((x) => net.forward(x)[0]!), data.ys);
    before.backward();
    const grads = net.parameters().map((p) => p.grad);
    const old = net.parameters().map((p) => p.data);
    const lr = 0.3;
    const loss = trainEpoch(net, data, lr);
    expect(loss).toBeCloseTo(before.data, 12);
    net.parameters().forEach((p, i) => expect(p.data).toBeCloseTo(old[i]! - lr * grads[i]!, 12));
  });
});

describe("decisionGrid", () => {
  it("is res² long, row 0 at the top, and agrees with predict at cell centres", () => {
    const net = new MLP([2, 3, 1], { output: "sigmoid", seed: 4 });
    const res = 6;
    const g = decisionGrid(net, { res, lo: -1, hi: 1 });
    expect(g).toHaveLength(36);
    const step = 2 / res;
    // top-left cell centre is (lo + step/2, hi - step/2)
    expect(g[0]).toBeCloseTo(predict(net, [-1 + step / 2, 1 - step / 2])[0]!, 12);
    // bottom-right
    expect(g[35]).toBeCloseTo(predict(net, [1 - step / 2, -1 + step / 2])[0]!, 12);
    for (const v of g) expect(v).toBeGreaterThan(0), expect(v).toBeLessThan(1);
  });
});

describe("Trainer", () => {
  it("is reproducible for a fixed seed", () => {
    const data = makeDataset("moons", { n: 40, seed: 1 });
    const a = new Trainer(data, { seed: 3, hidden: 4 });
    const b = new Trainer(data, { seed: 3, hidden: 4 });
    a.step(5);
    b.step(5);
    expect(a.losses).toEqual(b.losses);
    expect(a.epoch).toBe(5);
  });

  const targets: [DatasetName, number][] = [["xor", 0.95], ["circles", 0.95], ["moons", 0.9]];
  for (const [name, minAcc] of targets) {
    it(`separates ${name} (accuracy ≥ ${minAcc})`, () => {
      const data = makeDataset(name, { n: 80, seed: 7 });
      const t = new Trainer(data, { hidden: 8, lr: 1.0, seed: 1 });
      const first = t.step();
      t.step(399);
      expect(t.losses.at(-1)!).toBeLessThan(first / 3);
      expect(t.accuracy()).toBeGreaterThanOrEqual(minAcc);
      expect(accuracy(t.net, data)).toBe(t.accuracy());
    });
  }
});
