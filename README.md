# gradient

[![CI](https://github.com/MechStudents2029/gradient/actions/workflows/ci.yml/badge.svg)](https://github.com/MechStudents2029/gradient/actions/workflows/ci.yml)

A from-scratch reverse-mode automatic differentiation engine in TypeScript.

The plan: a tiny scalar autodiff core, then a web UI that draws the computation
graph, walks through derivatives step by step, shows Taylor-series
approximations, and trains a small neural net live in the browser.

Work in progress — see the roadmap below.

## Quick start

```ts
import { Value } from "./src/index.js";

const x = new Value(0.5, [], "", "x");
const w = new Value(-1.2, [], "", "w");
const y = x.mul(w).add(0.3).tanh();
y.backward();
console.log(y.data, x.grad, w.grad); // dy/dx = w(1 - y²), dy/dw = x(1 - y²)
```

Train a 2-4-1 network on XOR:

```ts
import { MLP, mse, sgdStep } from "./src/index.js";

const xs = [[0, 0], [0, 1], [1, 0], [1, 1]];
const ys = [0, 1, 1, 0];
const net = new MLP([2, 4, 1], { hidden: "tanh", output: "sigmoid", seed: 1 });
for (let step = 0; step < 2000; step++) {
  const loss = mse(xs.map((x) => net.forward(x)[0]!), ys);
  loss.backward();
  sgdStep(net.parameters(), 1.0);
}
```

```sh
npm install
npm test
npm run dev   # graph visualizer
```

## Graph visualizer

Live demo: **https://gradient-two-amber.vercel.app**

The `web/` page draws any expression as a computation graph. Pick a preset
(tanh neuron, sigmoid neuron, a reused variable, squared error, softplus) or
type your own, e.g. `tanh(x1*w1 + x2*w2 + b)` with `x1 = 2, w1 = -3, ...`.
Every node shows its forward value; press **Step backward** (or →) to apply
one node's local chain-rule step at a time and watch gradients flow back to
the inputs.

```sh
npm run dev      # local dev server
npm run build    # static site in dist/
```

The same data is available from code:

```ts
import { parseExpr, toGraph, layoutGraph, backwardTrace } from "./src/index.js";

const out = parseExpr("tanh(a*b + c)", { a: 2, b: -3, c: 6.5 });
const graph = toGraph(out);          // nodes (topo order) + edges
const layout = layoutGraph(graph);   // x/y for each node id
const steps = backwardTrace(out);    // gradients after each backward step
```

## Roadmap

- [x] Day 1: scalar `Value` with reverse-mode `backward()` and gradient checks
- [x] Day 2: `sigmoid`, a tiny MLP, and SGD that learns XOR
- [ ] Day 3: computation-graph visualizer (web UI)
- [ ] Day 4: step-by-step derivative walkthrough + Taylor series
- [ ] Day 5: live training demo, deployed on Vercel

## The math

Each `Value` stores its number, its parents, and a local derivative rule.
`backward()` sorts the graph topologically, seeds `dL/dL = 1`, and walks it in
reverse, applying the chain rule at every edge:

```
dL/dparent += dL/dchild · dchild/dparent
```

Gradients use `+=` because a node used twice (like `x` in `x·x + x`) receives a
contribution from every path to the output.

| op | local derivative |
| --- | --- |
| `a + b` | 1, 1 |
| `a · b` | b, a |
| `xᵏ` | k·xᵏ⁻¹ |
| `eˣ` | eˣ |
| `ln x` | 1/x |
| `tanh x` | 1 − tanh² x |
| `relu x` | 1 if x > 0 else 0 |
| `σ(x)` | σ(x)(1 − σ(x)) |

## Training a network

A neuron computes `act(w · x + b)`; a layer is a list of neurons; an `MLP`
stacks layers. Nothing new is needed for backprop: the network is just a big
graph of `Value`s, so `loss.backward()` fills in `∂L/∂θ` for every weight.

The loss is mean squared error and the update is plain gradient descent:

```
L = (1/n) Σ (ŷᵢ − yᵢ)²        ∂L/∂ŷᵢ = (2/n)(ŷᵢ − yᵢ)
θ ← θ − η · ∂L/∂θ
```

Weights start uniform in `[−1, 1) / √nin` from a seeded PRNG, so every run is
reproducible.

## Gradient checking

Every op is tested against the central finite difference

```
f'(x) ≈ (f(x + h) − f(x − h)) / 2h      (error O(h²))
```

via `numericGrad` in `src/gradcheck.ts`, including a deep mixed expression of
`tanh`, `exp`, powers, and division.
