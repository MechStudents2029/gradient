# Changelog

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
