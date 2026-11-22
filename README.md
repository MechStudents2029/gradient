# gradient

[![CI](https://github.com/MechStudents2029/gradient/actions/workflows/ci.yml/badge.svg)](https://github.com/MechStudents2029/gradient/actions/workflows/ci.yml)

A from-scratch reverse-mode automatic differentiation engine in TypeScript.

The plan: a tiny scalar autodiff core, then a web UI that draws the computation
graph, walks through derivatives step by step, shows Taylor-series
approximations, and trains a small neural net live in the browser.

The graph visualizer, the Taylor-series view, and the live training page are
up at https://gradient-two-amber.vercel.app.

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

## Taylor series

Live demo: **https://gradient-two-amber.vercel.app/taylor.html**

`backward()` gives first derivatives. For f″, f‴, ... at a point, gradient
pushes a whole truncated power series through the same graph instead
(Taylor-mode autodiff). Substitute x = a + t; then every node is a polynomial
in t up to degree n:

```
u(a + t) = u₀ + u₁t + u₂t² + ... + uₙtⁿ + O(tⁿ⁺¹)      uₖ = u⁽ᵏ⁾(a) / k!
```

The input is `[a, 1, 0, ...]`, constants are `[c, 0, 0, ...]`, and each op
maps coefficient arrays to coefficient arrays exactly. Products are the Cauchy
product `(uv)ₖ = Σⱼ uⱼvₖ₋ⱼ`; functions come from their ODE. For y = eᵘ,
y′ = y·u′ gives

```
k·yₖ = Σⱼ₌₁..ₖ j·uⱼ·yₖ₋ⱼ
```

so the coefficients fill in one at a time. The result is the Taylor polynomial

```
Pₙ(x) = Σₖ₌₀..ₙ f⁽ᵏ⁾(a)/k! · (x − a)ᵏ
```

with no finite differences and no symbolic algebra.

```ts
import { taylorCoefficients, derivativesAt, taylorPolynomial } from "./src/index.js";

taylorCoefficients("sin(x)", 0, 5);  // [0, 1, 0, -1/6, 0, 1/120]
derivativesAt("exp(sin(x))", 0, 3);  // [1, 1, 1, 0]
const c = taylorCoefficients("log(1 + x)", 0, 6);
taylorPolynomial(c, 0, 0.2);         // ≈ log(1.2)
```

## Live training

Live demo: **https://gradient-two-amber.vercel.app/train.html**

Pick a 2D toy dataset (XOR, two moons, or concentric circles) and a seed, then
press **Start**. A 2 → h → h → 1 network (tanh hidden layers, sigmoid output)
trains with full-batch gradient descent on MSE, and two plots redraw every
frame:

- **Decision boundary**: the net's output over [−1.5, 1.5]², blue ≈ 0 and
  orange ≈ 1, with the 0.5 contour drawn in white and the training points on top.
- **Loss curve**: MSE per epoch on a log scale.

The learning-rate slider (η from 0.01 to about 4, log scale) applies live, so you
can push η too high mid-run and watch the loss blow up. Changing the dataset,
seed, or hidden size rebuilds the network from the same seed, so every run is
reproducible. **Step** runs exactly one epoch.

The same pieces are plain library code:

```ts
import { Trainer, makeDataset, decisionGrid } from "./src/index.js";

const data = makeDataset("moons", { n: 120, seed: 1 });
const t = new Trainer(data, { hidden: 8, lr: 1.0, seed: 1 });
t.step(400);                      // 400 full-batch epochs
t.losses.at(-1);                  // final MSE
t.accuracy();                     // fraction classified correctly at 0.5
decisionGrid(t.net, { res: 50 }); // 50×50 outputs for a heatmap
```

## Roadmap

- [x] Day 1: scalar `Value` with reverse-mode `backward()` and gradient checks
- [x] Day 2: `sigmoid`, a tiny MLP, and SGD that learns XOR
- [x] Day 3: computation-graph visualizer (web UI), with an early deploy on Vercel
- [x] Day 4: Taylor series via Taylor-mode autodiff, with a plot of f and P₁..P_N (the step-by-step backward walkthrough shipped with Day 3)
- [ ] Day 5: live training demo in the browser

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
| `sin x` | cos x |
| `cos x` | −sin x |

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
