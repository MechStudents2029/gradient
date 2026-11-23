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

export interface RouteOptions {
  /** Width of a drawn node box, centred on its layout position. */
  nodeWidth: number;
  /** Height of a drawn node box, centred on its layout position. */
  nodeHeight: number;
  /** Extra clearance kept between an edge and any box it passes. */
  padding?: number;
  /**
   * Vertical spread between the two input ports of an op node: slot 0 lands
   * `spread / 2` above centre and slot 1 the same below. Leaves have one port.
   */
  portSpread?: number;
}

export interface EdgeRoute {
  edge: GraphEdge;
  /**
   * Polyline from the parent's right side to the child's left side. Adjacent
   * column edges are just [start, end]. An edge that skips columns gets two
   * extra points per skipped column (entering and leaving it) at a height
   * where no box sits, so it runs through a gap instead of over a node.
   */
  points: Point[];
}

/**
 * Route every edge of a layered layout. Edges between neighbouring columns go
 * straight from port to port. An edge that jumps over column d checks where
 * the straight line would cross that column; if it would clip a box there, it
 * is moved to the nearest free gap (between two nodes, or above / below the
 * column) and runs horizontally through it.
 */
export function routeEdges(graph: Graph, layout: Layout, opts: RouteOptions): EdgeRoute[] {
  const { nodeWidth: w, nodeHeight: h, padding = 6, portSpread = 0 } = opts;
  const byId = new Map(graph.nodes.map((n) => [n.id, n]));
  const columns = new Map<number, number[]>();
  for (const n of graph.nodes) {
    const ys = columns.get(n.depth) ?? [];
    ys.push(layout.positions[n.id]!.y);
    columns.set(n.depth, ys);
  }
  for (const ys of columns.values()) ys.sort((a, b) => a - b);

  // Candidate heights in a column where an edge can pass without touching a box.
  const gaps = (ys: number[]): number[] => {
    const out = [ys[0]! - h / 2 - padding - h / 4];
    for (let i = 1; i < ys.length; i++) out.push((ys[i - 1]! + ys[i]!) / 2);
    out.push(ys.at(-1)! + h / 2 + padding + h / 4);
    return out;
  };
  const clear = (ys: number[], y: number): boolean => ys.every((cy) => Math.abs(y - cy) > h / 2 + padding);

  return graph.edges.map((edge) => {
    const from = byId.get(edge.from)!;
    const to = byId.get(edge.to)!;
    const p = layout.positions[edge.from]!;
    const c = layout.positions[edge.to]!;
    const start = { x: p.x + w / 2, y: p.y };
    const endY = c.y + (to.op === "" ? 0 : (edge.slot - 0.5) * portSpread);
    const end = { x: c.x - w / 2, y: endY };
    const points: Point[] = [start];
    let prev = start;
    for (let d = from.depth + 1; d < to.depth; d++) {
      const ys = columns.get(d);
      if (!ys) continue;
      const cx = layout.positions[graph.nodes.find((n) => n.depth === d)!.id]!.x;
      // Where a straight line from the last point to the end would cross this column.
      const t = (cx - prev.x) / (end.x - prev.x);
      const straight = prev.y + t * (end.y - prev.y);
      const y = clear(ys, straight)
        ? straight
        : gaps(ys).reduce((best, g) => (Math.abs(g - straight) < Math.abs(best - straight) ? g : best));
      points.push({ x: cx - w / 2 - padding, y }, { x: cx + w / 2 + padding, y });
      prev = points.at(-1)!;
    }
    points.push(end);
    return { edge, points };
  });
}

/**
 * SVG path data for a routed edge: a horizontal-tangent cubic between each
 * pair of points, so straight runs through gaps stay flat and the bends
 * between columns are smooth S-curves.
 */
export function routePath(points: Point[]): string {
  if (points.length === 0) return "";
  let d = `M ${points[0]!.x} ${points[0]!.y}`;
  for (let i = 1; i < points.length; i++) {
    const a = points[i - 1]!;
    const b = points[i]!;
    if (a.y === b.y) {
      d += ` L ${b.x} ${b.y}`;
    } else {
      const mx = (a.x + b.x) / 2;
      d += ` C ${mx} ${a.y}, ${mx} ${b.y}, ${b.x} ${b.y}`;
    }
  }
  return d;
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
