import { describe, expect, it } from "vitest";
import { circlesDataset, gaussian, makeDataset, moonsDataset, mulberry32, xorDataset } from "../src/index.js";
import type { DatasetName } from "../src/index.js";

const names: DatasetName[] = ["xor", "moons", "circles"];

describe("gaussian", () => {
  it("has mean ≈ 0 and variance ≈ 1", () => {
    const rand = mulberry32(5);
    const xs = Array.from({ length: 20000 }, () => gaussian(rand));
    const mean = xs.reduce((a, b) => a + b, 0) / xs.length;
    const variance = xs.reduce((a, b) => a + (b - mean) ** 2, 0) / xs.length;
    expect(Math.abs(mean)).toBeLessThan(0.03);
    expect(Math.abs(variance - 1)).toBeLessThan(0.05);
  });
});

describe("datasets", () => {
  for (const name of names) {
    it(`${name}: right size, binary labels, roughly balanced, inside the plot`, () => {
      const { xs, ys } = makeDataset(name, { n: 200, seed: 3 });
      expect(xs).toHaveLength(200);
      expect(ys).toHaveLength(200);
      expect(ys.every((y) => y === 0 || y === 1)).toBe(true);
      const ones = ys.filter((y) => y === 1).length;
      expect(ones).toBeGreaterThan(60);
      expect(ones).toBeLessThan(140);
      for (const [x, y] of xs) {
        expect(Math.abs(x)).toBeLessThan(1.5);
        expect(Math.abs(y)).toBeLessThan(1.5);
      }
    });

    it(`${name}: deterministic per seed, different across seeds`, () => {
      expect(makeDataset(name, { seed: 9 })).toEqual(makeDataset(name, { seed: 9 }));
      expect(makeDataset(name, { seed: 9 }).xs).not.toEqual(makeDataset(name, { seed: 10 }).xs);
    });
  }

  it("xor labels follow the sign rule when there is no noise", () => {
    const { xs, ys } = xorDataset({ n: 300, noise: 0, seed: 2 });
    xs.forEach(([x, y], i) => expect(ys[i]).toBe(x * y < 0 ? 1 : 0));
  });

  it("circles: inner class has radius 0.5, outer class radius 1 (noise-free)", () => {
    const { xs, ys } = circlesDataset({ n: 50, noise: 0, seed: 4 });
    xs.forEach(([x, y], i) => expect(Math.hypot(x, y)).toBeCloseTo(ys[i] === 1 ? 0.5 : 1, 10));
  });

  it("moons: noise-free points lie on their half circles", () => {
    const { xs, ys } = moonsDataset({ n: 60, noise: 0, seed: 6 });
    xs.forEach(([px, py], i) => {
      // Undo the recentre/scale, then check distance to the moon's centre.
      const x = px / 0.8 + 0.5;
      const y = py / 0.8 + 0.25;
      const [cx, cy] = ys[i] === 0 ? [0, 0] : [1, 0.5];
      expect(Math.hypot(x - cx, y - cy)).toBeCloseTo(1, 10);
      if (ys[i] === 0) expect(y).toBeGreaterThanOrEqual(-1e-12);
      else expect(y).toBeLessThanOrEqual(0.5 + 1e-12);
    });
  });

  it("rejects unknown names", () => {
    expect(() => makeDataset("spiral" as DatasetName)).toThrow(/unknown dataset/);
  });
});
