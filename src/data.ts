import { mulberry32 } from "./nn.js";

/** A labelled 2D toy dataset: points in roughly [-1.5, 1.5]², labels 0 or 1. */
export interface Dataset {
  xs: [number, number][];
  ys: number[];
}

export type DatasetName = "xor" | "moons" | "circles";

export interface DatasetOptions {
  /** Number of points (split evenly between the two classes). */
  n?: number;
  /** Standard deviation of the Gaussian noise added to each coordinate. */
  noise?: number;
  seed?: number;
}

/** Standard normal sample via Box–Muller, driven by a uniform PRNG. */
export function gaussian(rand: () => number): number {
  const u = 1 - rand(); // (0, 1] so log is finite
  const v = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/**
 * XOR: points uniform in [-1, 1]², label 1 when x and y have opposite signs.
 * Points too close to an axis are resampled so the classes stay separable.
 */
export function xorDataset({ n = 100, noise = 0.05, seed = 1 }: DatasetOptions = {}): Dataset {
  const rand = mulberry32(seed);
  const xs: [number, number][] = [];
  const ys: number[] = [];
  while (xs.length < n) {
    const x = rand() * 2 - 1;
    const y = rand() * 2 - 1;
    if (Math.abs(x) < 0.1 || Math.abs(y) < 0.1) continue;
    xs.push([x + noise * gaussian(rand), y + noise * gaussian(rand)]);
    ys.push(x * y < 0 ? 1 : 0);
  }
  return { xs, ys };
}

/**
 * Two interleaving half circles (scikit-learn's make_moons), recentred and
 * scaled to sit around the origin. Class 0 is the upper moon.
 */
export function moonsDataset({ n = 100, noise = 0.1, seed = 1 }: DatasetOptions = {}): Dataset {
  const rand = mulberry32(seed);
  const xs: [number, number][] = [];
  const ys: number[] = [];
  for (let i = 0; i < n; i++) {
    const label = i % 2;
    const t = Math.PI * rand();
    const [x, y] = label === 0 ? [Math.cos(t), Math.sin(t)] : [1 - Math.cos(t), 0.5 - Math.sin(t)];
    // Raw moons span x ∈ [-1, 2], y ∈ [-0.5, 1]; centre on (0.5, 0.25).
    xs.push([(x - 0.5 + noise * gaussian(rand)) * 0.8, (y - 0.25 + noise * gaussian(rand)) * 0.8]);
    ys.push(label);
  }
  return { xs, ys };
}

/** Concentric circles: class 1 is the inner disc (radius 0.5), class 0 the outer ring (radius 1). */
export function circlesDataset({ n = 100, noise = 0.08, seed = 1 }: DatasetOptions = {}): Dataset {
  const rand = mulberry32(seed);
  const xs: [number, number][] = [];
  const ys: number[] = [];
  for (let i = 0; i < n; i++) {
    const label = i % 2;
    const r = label === 1 ? 0.5 : 1;
    const t = 2 * Math.PI * rand();
    xs.push([r * Math.cos(t) + noise * gaussian(rand), r * Math.sin(t) + noise * gaussian(rand)]);
    ys.push(label);
  }
  return { xs, ys };
}

export const DATASETS: Record<DatasetName, (opts?: DatasetOptions) => Dataset> = {
  xor: xorDataset,
  moons: moonsDataset,
  circles: circlesDataset,
};

/** Build a dataset by name. */
export function makeDataset(name: DatasetName, opts: DatasetOptions = {}): Dataset {
  const make = DATASETS[name];
  if (!make) throw new Error(`unknown dataset: ${String(name)}`);
  return make(opts);
}
