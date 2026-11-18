# Changelog

## Day 1 — 2026-11-18

- Scalar `Value` node with `add`, `mul`, `pow`, `sub`, `div`, `exp`, `log`, `tanh`, `relu`.
- Reverse-mode `backward()` over a topological sort, with gradient accumulation for reused nodes.
- `numericGrad` central-difference checker and 12 Vitest cases.
- GitHub Actions CI (typecheck + tests).
