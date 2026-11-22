import { MLP, mse, sgdStep } from "./nn.js";
import type { Activation } from "./nn.js";
import type { Dataset } from "./data.js";

function act(x: number, a: Activation): number {
  switch (a) {
    case "tanh":
      return Math.tanh(x);
    case "relu":
      return x > 0 ? x : 0;
    case "sigmoid":
      return 1 / (1 + Math.exp(-x));
    case "linear":
      return x;
  }
}

/**
 * Forward pass on plain numbers, reading the current weights out of the
 * network's `Value`s. Builds no graph, so it is cheap enough to evaluate the
 * net on a whole heatmap grid every frame.
 */
export function predict(net: MLP, x: readonly number[]): number[] {
  let h = x.slice();
  for (const layer of net.layers) {
    h = layer.neurons.map((n) => {
      let s = n.b.data;
      n.w.forEach((wi, i) => (s += wi.data * h[i]!));
      return act(s, n.act);
    });
  }
  return h;
}

/** Fraction of points whose output, thresholded at 0.5, matches the label. */
export function accuracy(net: MLP, data: Dataset): number {
  let correct = 0;
  data.xs.forEach((x, i) => {
    if ((predict(net, x)[0]! >= 0.5 ? 1 : 0) === data.ys[i]) correct++;
  });
  return data.xs.length ? correct / data.xs.length : 0;
}

/**
 * One full-batch gradient-descent epoch: forward every point, MSE loss,
 * backward, θ ← θ − η∇θL. Returns the loss before the update.
 */
export function trainEpoch(net: MLP, data: Dataset, lr: number): number {
  const loss = mse(data.xs.map((x) => net.forward(x)[0]!), data.ys);
  loss.backward();
  sgdStep(net.parameters(), lr);
  return loss.data;
}

export interface GridOptions {
  /** Cells per side. */
  res?: number;
  /** Square domain [lo, hi]² to sample. */
  lo?: number;
  hi?: number;
}

/**
 * Net output on a res × res grid of cell centres, row-major with row 0 at the
 * top (largest y), ready to paint as a decision-boundary heatmap.
 */
export function decisionGrid(net: MLP, { res = 40, lo = -1.5, hi = 1.5 }: GridOptions = {}): Float64Array {
  const out = new Float64Array(res * res);
  const step = (hi - lo) / res;
  for (let r = 0; r < res; r++) {
    const y = hi - (r + 0.5) * step;
    for (let c = 0; c < res; c++) {
      const x = lo + (c + 0.5) * step;
      out[r * res + c] = predict(net, [x, y])[0]!;
    }
  }
  return out;
}

export interface TrainerOptions {
  hidden?: number;
  lr?: number;
  seed?: number;
}

/**
 * Owns a 2-h-h-1 tanh/sigmoid network and a dataset, and records the loss
 * history. The web page calls `step()` once per animation frame.
 */
export class Trainer {
  net: MLP;
  lr: number;
  epoch = 0;
  readonly losses: number[] = [];

  constructor(readonly data: Dataset, readonly opts: TrainerOptions = {}) {
    this.lr = opts.lr ?? 0.5;
    this.net = Trainer.build(opts);
  }

  static build({ hidden = 8, seed = 42 }: TrainerOptions): MLP {
    return new MLP([2, hidden, hidden, 1], { hidden: "tanh", output: "sigmoid", seed });
  }

  /** Run `epochs` full-batch epochs and return the last loss. */
  step(epochs = 1): number {
    let loss = NaN;
    for (let i = 0; i < epochs; i++) {
      loss = trainEpoch(this.net, this.data, this.lr);
      this.losses.push(loss);
      this.epoch++;
    }
    return loss;
  }

  accuracy(): number {
    return accuracy(this.net, this.data);
  }
}
