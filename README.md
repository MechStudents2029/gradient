# gradient

A from-scratch reverse-mode automatic differentiation engine in TypeScript.

The plan: a tiny scalar autodiff core, then a web UI that draws the computation
graph, walks through derivatives step by step, shows Taylor-series
approximations, and trains a small neural net live in the browser.

Work in progress — see the roadmap below.

## Roadmap

- [ ] Day 1: scalar `Value` with reverse-mode `backward()` and gradient checks
- [ ] Day 2: more ops and a tiny MLP
- [ ] Day 3: computation-graph visualizer (web UI)
- [ ] Day 4: step-by-step derivative walkthrough + Taylor series
- [ ] Day 5: live training demo, deployed on Vercel
