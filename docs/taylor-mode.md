# Taylor-mode recurrences

`src/taylor.ts` represents every node as a truncated series in t, where
x = a + t:

```
u(a + t) = Σₖ₌₀..ₙ uₖ tᵏ + O(tⁿ⁺¹),     uₖ = u⁽ᵏ⁾(a) / k!
```

Each op needs a rule that produces y₀..yₙ from its inputs' coefficients.

## Sums and products

Coefficients of a sum add. A product is the Cauchy product, truncated:

```
(u·v)ₖ = Σⱼ₌₀..ₖ uⱼ vₖ₋ⱼ
```

## Functions from their ODE

For y = g(u) with a simple derivative, differentiate in t and match the
coefficient of tᵏ⁻¹ on both sides. Using `(y′)ₖ₋₁ = k·yₖ`:

| op | ODE | recurrence (k ≥ 1) |
| --- | --- | --- |
| eᵘ | y′ = y u′ | k yₖ = Σⱼ₌₁..ₖ j uⱼ yₖ₋ⱼ |
| ln u | u y′ = u′ | yₖ = (uₖ − (1/k) Σⱼ₌₁..ₖ₋₁ j yⱼ uₖ₋ⱼ) / u₀ |
| uᵖ | u y′ = p y u′ | yₖ = 1/(k u₀) · Σⱼ₌₁..ₖ (p j − (k − j)) uⱼ yₖ₋ⱼ |
| sin u, cos u | s′ = c u′, c′ = −s u′ | k sₖ = Σ j uⱼ cₖ₋ⱼ,  k cₖ = −Σ j uⱼ sₖ₋ⱼ |
| tanh u | y′ = (1 − y²) u′ | k yₖ = Σ j uⱼ wₖ₋ⱼ,  w = 1 − y² |
| σ(u) | y′ = (y − y²) u′ | k yₖ = Σ j uⱼ wₖ₋ⱼ,  w = y − y² |

The right-hand sides only use y₀..yₖ₋₁ (and w₀..wₖ₋₁, which only need
y₀..yₖ₋₁), so each coefficient is computed from ones already known. sin and
cos are computed together because each one's recurrence needs the other.

## Edge cases

- `uᵖ` with a non-negative integer p uses repeated squaring with the Cauchy
  product. That is exact, so polynomials re-expand exactly, and it works when
  u₀ = 0 (x² at 0). Any other p at u₀ = 0 throws: x⁻¹ and √x have no Taylor
  series at 0.
- `ln u` at u₀ ≤ 0 gives NaN/−∞, matching `Value.log`.
- `relu` is exact away from 0. At 0 it has no series; like `backward()`, it
  uses slope 0 there.

## Why not run backward twice?

Nested reverse mode needs the backward pass itself to build a graph of
`Value`s. Here `backwardStep` closures write plain numbers into `grad`, so a
second pass has nothing to differentiate. Taylor mode reuses the forward graph
as-is, costs O(n²) per node for order n, and gives all orders at once, which
is what a Taylor plot needs. The tests cross-check it against the engine:
c₁ equals the reverse-mode gradient, and 2·c₂ matches a finite difference of
that gradient.
