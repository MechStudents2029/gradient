# Training math and the decision boundary

The training page fits a network f(x; θ) with a sigmoid output, so
f ∈ (0, 1) can be read as "how class-1 is this point". Labels are 0 or 1.

## One epoch

Every epoch is one full-batch gradient-descent step:

```
L(θ) = (1/n) Σᵢ (f(xᵢ; θ) − yᵢ)²
θ ← θ − η · ∇θ L
```

`trainEpoch` builds the whole loss as a single `Value` graph (n forward
passes feeding one `mse` node), calls `loss.backward()` once, and applies
`sgdStep`. `backward()` zeroes every reachable gradient before the sweep, so
nothing leaks between epochs and no `zeroGrad` call is needed.

The sigmoid output makes the MSE gradient pick up a factor σ(1 − σ):

```
∂L/∂z = (2/n)(f − y) · f(1 − f)        (z = pre-activation of the output)
```

so confidently wrong points (f near 0 or 1) learn slowly. That is why a large
η (around 1) works well here, and it is the usual argument for cross-entropy,
which cancels that factor.

## Why a separate `predict`

Drawing the heatmap evaluates the net at res² points every frame (2,500 at
res = 50). Building a `Value` graph for each would allocate thousands of nodes
just to throw them away. `predict` reads the current weights out of the
network's `Value`s and runs the same arithmetic on plain numbers. A test pins
it to `net.forward(x)` to 12 digits for every activation.

## Drawing the boundary

`decisionGrid` samples f at the centre of each cell of a res × res grid over
[lo, hi]², row 0 at the top so it maps straight onto canvas pixels. The page
colours each cell on a blue → grey → orange ramp, scales the small image up
with smoothing, and marks the 0.5 contour by drawing a cell edge wherever two
neighbouring cells fall on opposite sides of 0.5. The decision boundary is
exactly that level set {x : f(x) = 0.5}.
