# How the graph is laid out and walked

## Export

`toGraph(root)` calls `root.topo()`, a depth-first search that pushes a node
only after all of its parents. That list is the node order, and a node's id is
just its position in it (`n0`, `n1`, ...). Edges go from parent to child and
carry a `slot` (0 or 1) saying which argument they feed, so `x * x` is one
node with two edges into the product.

## Layout

Each node gets a depth: leaves are 0, and any other node is
`1 + max(depth of its parents)`, the longest path from an input. Because the
nodes are already in topological order, one pass computes every depth.

`layoutGraph` puts depth `d` in column `d`, so every edge points left to right.
Inside a column nodes keep their topological order and the column is centred
against the tallest one. This is the simplest form of a layered
(Sugiyama-style) drawing; it skips crossing reduction, which is fine for the
small expressions the demo is for.

## Backward order

`backward()` walks the same topological list in reverse. That order guarantees
a node's gradient is final before it is passed on: every child of a node comes
after it in the list, so in reverse every child has already pushed its share
into the node.

`backwardTrace` records that walk. Step 0 is the seed (`∂out/∂out = 1`, every
other gradient 0). Each later step is one non-leaf node applying its local
rule, followed by a snapshot of every gradient. Leaves have no rule, so they
never get a step of their own; they just fill up as their children run. The
last snapshot equals the result of `backward()`, which the tests check.
