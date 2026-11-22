# The toy datasets

All three generators take `{ n, noise, seed }` and draw from the same seeded
`mulberry32` PRNG as the weight init, so a seed pins down the whole run. Points
sit inside [−1.5, 1.5]², the square the training page draws.

## Gaussian noise

Noise is Box–Muller: for u, v uniform on (0, 1],

```
z = √(−2 ln u) · cos(2πv)
```

is a standard normal sample. Each coordinate gets `noise · z` added. A test
checks the sample mean ≈ 0 and variance ≈ 1 over 20,000 draws.

## XOR

Points uniform in [−1, 1]², labelled 1 when x·y < 0 (opposite quadrants).
Points within 0.1 of an axis are resampled so the two classes are separated
by a margin before noise is added. No line separates them; one hidden layer
of tanh units can (see [why-xor.md](why-xor.md)).

## Two moons

Same shape as scikit-learn's `make_moons`. With t uniform on [0, π]:

```
class 0:  (cos t, sin t)              upper half circle around (0, 0)
class 1:  (1 − cos t, 0.5 − sin t)    lower half circle around (1, 0.5)
```

The raw moons span x ∈ [−1, 2], y ∈ [−0.5, 1], so they are shifted by
(−0.5, −0.25) and scaled by 0.8 to sit around the origin.

## Circles

Class 1 lies on a circle of radius 0.5, class 0 on radius 1, at uniform
angles. A linear model is useless here; the hidden layer has to learn
something like r² = x² + y² and threshold it.

Classes alternate point by point in moons and circles, so they are exactly
balanced; XOR is balanced in expectation.
