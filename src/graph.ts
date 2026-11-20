import type { Op, Value } from "./value.js";

export interface GraphNode {
  /** Stable id: "n" + position in the topological order. */
  id: string;
  label: string;
  op: Op;
  data: number;
  grad: number;
  /** Longest path from any leaf: leaves are 0, the output is the deepest. */
  depth: number;
}

export interface GraphEdge {
  /** Parent (input) node id. */
  from: string;
  /** Child (output) node id. */
  to: string;
  /** Which argument of `to` this edge feeds (0 or 1). */
  slot: number;
}

export interface Graph {
  /** Topologically ordered: every node appears after all of its parents. */
  nodes: GraphNode[];
  edges: GraphEdge[];
  /** Id of the node the graph was exported from. */
  root: string;
}

/** Snapshot the computation graph that ends at `root` as plain data. */
export function toGraph(root: Value): Graph {
  const order = root.topo();
  const ids = new Map<Value, string>();
  order.forEach((v, i) => ids.set(v, `n${i}`));

  const depth = new Map<Value, number>();
  for (const v of order) {
    depth.set(v, v.parents.length === 0 ? 0 : 1 + Math.max(...v.parents.map((p) => depth.get(p)!)));
  }

  const nodes = order.map((v) => ({
    id: ids.get(v)!,
    label: v.label,
    op: v.op,
    data: v.data,
    grad: v.grad,
    depth: depth.get(v)!,
  }));
  const edges = order.flatMap((v) =>
    v.parents.map((p, slot) => ({ from: ids.get(p)!, to: ids.get(v)!, slot })),
  );
  return { nodes, edges, root: ids.get(root)! };
}

export interface Point {
  x: number;
  y: number;
}

export interface Layout {
  positions: Record<string, Point>;
  width: number;
  height: number;
}

export interface LayoutOptions {
  /** Horizontal distance between depth columns. */
  colGap?: number;
  /** Vertical distance between nodes in one column. */
  rowGap?: number;
  /** Padding around the drawing. */
  margin?: number;
}

/**
 * Layered left-to-right layout: column = depth, so every edge points to the
 * right. Within a column, nodes keep their topological order and the column is
 * centred vertically. Returns node centres plus the overall drawing size.
 */
export function layoutGraph(graph: Graph, opts: LayoutOptions = {}): Layout {
  const { colGap = 180, rowGap = 90, margin = 60 } = opts;
  const columns = new Map<number, GraphNode[]>();
  for (const n of graph.nodes) {
    const col = columns.get(n.depth) ?? [];
    col.push(n);
    columns.set(n.depth, col);
  }
  const maxRows = Math.max(0, ...[...columns.values()].map((c) => c.length));
  const maxDepth = Math.max(0, ...graph.nodes.map((n) => n.depth));
  const innerHeight = Math.max(0, maxRows - 1) * rowGap;

  const positions: Record<string, Point> = {};
  for (const [d, col] of columns) {
    const offset = (innerHeight - (col.length - 1) * rowGap) / 2;
    col.forEach((n, i) => {
      positions[n.id] = { x: margin + d * colGap, y: margin + offset + i * rowGap };
    });
  }
  return {
    positions,
    width: graph.nodes.length ? 2 * margin + maxDepth * colGap : 0,
    height: graph.nodes.length ? 2 * margin + innerHeight : 0,
  };
}

export interface TraceStep {
  /** Node whose local chain-rule step was just applied (null for the seed). */
  active: string | null;
  /** Gradient of every node (by id) after this step. */
  grads: Record<string, number>;
}

/**
 * Run `root.backward()` one node at a time and record every node's gradient
 * after each step. Step 0 is the seed dL/dL = 1; then one step per non-leaf
 * node in reverse topological order. Leaves have no local rule, so they are
 * skipped. Leaves `root`'s graph with its final gradients.
 */
export function backwardTrace(root: Value): TraceStep[] {
  const order = root.topo();
  const ids = new Map<Value, string>();
  order.forEach((v, i) => ids.set(v, `n${i}`));
  const snapshot = (): Record<string, number> =>
    Object.fromEntries(order.map((v) => [ids.get(v)!, v.grad]));

  const seed = Object.fromEntries(order.map((v) => [ids.get(v)!, v === root ? 1 : 0]));
  const steps: TraceStep[] = [{ active: null, grads: seed }];
  for (const v of root.backwardSteps()) {
    if (v.parents.length > 0) steps.push({ active: ids.get(v)!, grads: snapshot() });
  }
  return steps;
}
