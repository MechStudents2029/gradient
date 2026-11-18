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

```sh
npm install
npm test
```

## Roadmap

- [x] Day 1: scalar `Value` with reverse-mode `backward()` and gradient checks
- [ ] Day 2: more ops and a tiny MLP
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
