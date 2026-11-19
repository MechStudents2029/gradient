import { Value } from "./value.js";

/** Deterministic PRNG (mulberry32) so training runs are reproducible. */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export type Activation = "tanh" | "relu" | "sigmoid" | "linear";

function activate(v: Value, act: Activation): Value {
  switch (act) {
    case "tanh":
      return v.tanh();
    case "relu":
      return v.relu();
    case "sigmoid":
      return v.sigmoid();
    case "linear":
      return v;
  }
}

/** One neuron: act(w · x + b). */
export class Neuron {
  readonly w: Value[];
  readonly b: Value;
  constructor(nin: number, readonly act: Activation, rand: () => number) {
    // Uniform in [-1, 1) scaled by 1/sqrt(nin) keeps pre-activations O(1).
    const scale = 1 / Math.sqrt(nin);
    this.w = Array.from({ length: nin }, () => new Value((rand() * 2 - 1) * scale));
    this.b = new Value(0);
  }
  forward(x: readonly Value[]): Value {
    let sum = this.b;
    this.w.forEach((wi, i) => {
      sum = sum.add(wi.mul(x[i]!));
    });
    return activate(sum, this.act);
  }
  parameters(): Value[] {
    return [...this.w, this.b];
  }
}

export class Layer {
  readonly neurons: Neuron[];
  constructor(nin: number, nout: number, act: Activation, rand: () => number) {
    this.neurons = Array.from({ length: nout }, () => new Neuron(nin, act, rand));
  }
  forward(x: readonly Value[]): Value[] {
    return this.neurons.map((n) => n.forward(x));
  }
  parameters(): Value[] {
    return this.neurons.flatMap((n) => n.parameters());
  }
}

/** Multi-layer perceptron: hidden layers use `hidden`, the last layer uses `output`. */
export class MLP {
  readonly layers: Layer[];
  constructor(
    sizes: readonly number[],
    { hidden = "tanh", output = "linear", seed = 42 }: { hidden?: Activation; output?: Activation; seed?: number } = {},
  ) {
    const rand = mulberry32(seed);
    this.layers = sizes.slice(1).map((nout, i) =>
      new Layer(sizes[i]!, nout, i === sizes.length - 2 ? output : hidden, rand),
    );
  }
  forward(x: readonly (Value | number)[]): Value[] {
    let h = x.map(Value.from);
    for (const layer of this.layers) h = layer.forward(h);
    return h;
  }
  parameters(): Value[] {
    return this.layers.flatMap((l) => l.parameters());
  }
}

/** Mean squared error: (1/n) Σ (pred − target)². */
export function mse(preds: readonly Value[], targets: readonly number[]): Value {
  let total = new Value(0);
  preds.forEach((p, i) => {
    total = total.add(p.sub(targets[i]!).pow(2));
  });
  return total.div(preds.length);
}

/** Plain gradient descent: θ ← θ − η · ∂L/∂θ. */
export function sgdStep(params: readonly Value[], lr: number): void {
  for (const p of params) p.data -= lr * p.grad;
}
