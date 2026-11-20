# The chain rule, one edge at a time

Take `out = tanh(a·b + c)` and name the pieces `p = a·b`, `s = p + c`,
`out = tanh(s)`. The graph has an edge for every place one value feeds another,
and each edge has a *local derivative*: how much the child moves when that one
parent moves.

| edge | local derivative |
| --- | --- |
| `s → out` | `1 − tanh²(s)` |
| `p → s`, `c → s` | `1`, `1` |
| `a → p`, `b → p` | `b`, `a` |

The chain rule says the derivative of the output with respect to any node is
the product of local derivatives along the path to the output:

```
∂out/∂a = ∂out/∂s · ∂s/∂p · ∂p/∂a = (1 − tanh²(s)) · 1 · b
```

Reverse mode computes these products from the output end. Each node keeps a
running `grad = ∂out/∂node`; when a node's step runs, it multiplies its own
`grad` by each edge's local derivative and adds the result to the parent:

```
parent.grad += child.grad · ∂child/∂parent
```

So the work per edge is one multiply and one add, and the whole backward pass
costs about the same as the forward pass, no matter how many inputs there are.

## When paths merge

If a value feeds more than one child, it has more than one path to the output,
and the multivariable chain rule sums them:

```
∂out/∂x = Σ over children k of  ∂out/∂k · ∂k/∂x
```

That sum is why gradients use `+=`. For `x·x + x` at `x = 3`, the two edges
into the product each contribute `1 · 3`, the edge into the sum contributes
`1 · 1`, and `∂out/∂x = 3 + 3 + 1 = 7`, which is `2x + 1`. The "reused
variable" preset in the visualizer shows this step by step.
