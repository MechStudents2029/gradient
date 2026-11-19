# Changelog

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
