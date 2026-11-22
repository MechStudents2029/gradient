# Changelog

## Day 5 — 2026-11-22

- Seeded 2D toy datasets in `src/data.ts`: `xorDataset`, `moonsDataset`, `circlesDataset`, `makeDataset`, and a Box–Muller `gaussian`.
- Training helpers in `src/train.ts`: `trainEpoch` (full-batch gradient descent on MSE), `predict` (a graph-free forward pass on plain numbers), `accuracy`, `decisionGrid`, and a `Trainer` that keeps the loss history.
- Web page `train.html`: train an MLP live with a decision-boundary heatmap and a log-scale loss curve; controls for dataset, seed, learning rate, hidden size, and speed, plus start/pause/step/reset.
- Nav links between all three demo pages.
- Tests for dataset shape, determinism, and geometry; `predict` against the `Value` forward pass; `trainEpoch` against a hand-rolled SGD step; and the trainer separating all three datasets. 75 tests.

## Day 4 — 2026-11-21

- Taylor-mode autodiff in `src/taylor.ts`: push truncated power series (jets) through the `Value` graph to get exact f⁽ᵏ⁾(a) for any order in one forward sweep.
- `taylorCoefficients`, `derivativesAt`, `taylorPolynomial`, `graphJet`, and per-op jet recurrences (`+`, `*`, `pow`, `exp`, `log`, `sin`, `cos`, `tanh`, `sigmoid`, `relu`).
- New `sin` and `cos` ops on `Value` and in the expression parser.
- Web view `taylor.html`: f(x) against P₁..P_N around a draggable expansion point, with the coefficient formula and a derivative table.
- Tests against known series, the reverse-mode gradient, and the Lagrange remainder. 57 tests.

## Day 3 — 2026-11-20

- `toGraph`, `layoutGraph`, and `backwardTrace` turn a `Value` graph into plain data: topologically ordered nodes with depth, edges with argument slots, a layered layout, and per-step gradient snapshots.
- `Value.backwardSteps()` runs the reverse sweep one node at a time; `backward()` is built on it.
- `parseExpr` / `parseVars`: a small expression parser so graphs can be typed as text.
- Web visualizer in `web/` (Vite + vanilla TS): SVG graph with values and gradients, stepping through the backward pass.
- CI also builds the demo. 38 tests.

## Day 2 — 2026-11-19

- `sigmoid` op with derivative σ(1 − σ).
- `Neuron`, `Layer`, and `MLP` built on `Value`, with tanh/relu/sigmoid/linear activations.
- `mse` loss, `sgdStep`, and a seeded `mulberry32` PRNG for reproducible init.
- Tests: whole-network backprop vs finite differences, and an MLP that learns XOR (17 tests).

## Day 1 — 2026-11-18

- Scalar `Value` node with `add`, `mul`, `pow`, `sub`, `div`, `exp`, `log`, `tanh`, `relu`.
- Reverse-mode `backward()` over a topological sort, with gradient accumulation for reused nodes.
- `numericGrad` central-difference checker and 12 Vitest cases.
- GitHub Actions CI (typecheck + tests).
