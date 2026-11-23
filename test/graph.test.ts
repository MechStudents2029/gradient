import { describe, expect, it } from "vitest";
import { Value, backwardTrace, layoutGraph, parseExpr, routeEdges, routePath, toGraph } from "../src/index.js";
import type { EdgeRoute, Graph, Layout } from "../src/index.js";

function neuron() {
  const a = new Value(2, [], "", "a");
  const b = new Value(-3, [], "", "b");
  const c = new Value(10, [], "", "c");
  const out = a.mul(b).add(c).tanh();
  out.label = "out";
  return { a, b, c, out };
}

describe("toGraph", () => {
  it("exports every node and edge with forward values", () => {
    const { out } = neuron();
    const g = toGraph(out);
    expect(g.nodes).toHaveLength(6); // a, b, c, a*b, +c, tanh
    expect(g.edges).toHaveLength(5);
    const byLabel = Object.fromEntries(g.nodes.map((n) => [n.label, n]));
    expect(byLabel.a!.data).toBe(2);
    expect(byLabel.out!.op).toBe("tanh");
    expect(byLabel.out!.data).toBeCloseTo(Math.tanh(4));
    expect(g.root).toBe(byLabel.out!.id);
  });

  it("orders nodes so every parent comes before its child", () => {
    const { out } = neuron();
    const g = toGraph(out);
    const index = new Map(g.nodes.map((n, i) => [n.id, i]));
    for (const e of g.edges) expect(index.get(e.from)!).toBeLessThan(index.get(e.to)!);
    expect(g.nodes.at(-1)!.id).toBe(g.root);
  });

  it("gives leaves depth 0 and children the longest-path depth", () => {
    const { out } = neuron();
    const depths = Object.fromEntries(toGraph(out).nodes.map((n) => [n.label || n.op, n.depth]));
    expect(depths).toEqual({ a: 0, b: 0, c: 0, "*": 1, "+": 2, out: 3 });
  });

  it("keeps one node and two edges for a value used twice", () => {
    const x = new Value(3, [], "", "x");
    const g = toGraph(x.mul(x));
    expect(g.nodes).toHaveLength(2);
    expect(g.edges.map((e) => e.slot)).toEqual([0, 1]);
    expect(new Set(g.edges.map((e) => e.from)).size).toBe(1);
  });

  it("captures gradients after backward()", () => {
    const { a, b, out } = neuron();
    out.backward();
    const byLabel = Object.fromEntries(toGraph(out).nodes.map((n) => [n.label, n]));
    expect(byLabel.out!.grad).toBe(1);
    expect(byLabel.a!.grad).toBe(a.grad);
    expect(byLabel.b!.grad).toBe(b.grad);
  });
});

describe("layoutGraph", () => {
  it("puts each depth in its own column so edges point right", () => {
    const g = toGraph(neuron().out);
    const { positions } = layoutGraph(g);
    for (const e of g.edges) expect(positions[e.from]!.x).toBeLessThan(positions[e.to]!.x);
  });

  it("never stacks two nodes on the same spot and fits them in the canvas", () => {
    const g = toGraph(neuron().out);
    const { positions, width, height } = layoutGraph(g, { colGap: 100, rowGap: 50, margin: 20 });
    const spots = new Set(Object.values(positions).map((p) => `${p.x},${p.y}`));
    expect(spots.size).toBe(g.nodes.length);
    for (const p of Object.values(positions)) {
      expect(p.x).toBeGreaterThanOrEqual(20);
      expect(p.x).toBeLessThanOrEqual(width - 20);
      expect(p.y).toBeGreaterThanOrEqual(20);
      expect(p.y).toBeLessThanOrEqual(height - 20);
    }
    expect(width).toBe(2 * 20 + 3 * 100);
    expect(height).toBe(2 * 20 + 2 * 50); // three leaves in the widest column
  });

  it("handles a single leaf", () => {
    const { positions, width, height } = layoutGraph(toGraph(new Value(1)), { margin: 10 });
    expect(Object.values(positions)).toEqual([{ x: 10, y: 10 }]);
    expect([width, height]).toEqual([20, 20]);
  });
});

describe("backwardTrace", () => {
  it("starts from the seed and ends at the full backward() gradients", () => {
    const { out } = neuron();
    const steps = backwardTrace(out);
    const g = toGraph(out); // grads are now final
    expect(steps[0]!.active).toBeNull();
    expect(steps[0]!.grads[g.root]).toBe(1);
    expect(Object.values(steps[0]!.grads).filter((x) => x !== 0)).toEqual([1]);
    const last = steps.at(-1)!.grads;
    for (const n of g.nodes) expect(last[n.id]).toBe(n.grad);
  });

  it("takes one step per non-leaf node, from the output back to the inputs", () => {
    const { out } = neuron();
    const steps = backwardTrace(out);
    const ops = Object.fromEntries(toGraph(out).nodes.map((n) => [n.id, n.op]));
    expect(steps.slice(1).map((s) => ops[s.active!])).toEqual(["tanh", "+", "*"]);
  });

  it("only fills a node's parents when that node's step runs", () => {
    const { out } = neuron();
    const steps = backwardTrace(out);
    const g = toGraph(out);
    const id = (label: string) => g.nodes.find((n) => n.label === label)!.id;
    // After the tanh and + steps, c is known but a and b still wait on the * step.
    expect(steps[2]!.grads[id("c")]).not.toBe(0);
    expect(steps[2]!.grads[id("a")]).toBe(0);
    expect(steps[3]!.grads[id("a")]).toBeCloseTo(-3 * (1 - Math.tanh(4) ** 2));
  });

  it("matches backward() even when re-run on a graph with stale gradients", () => {
    const x = new Value(1.5, [], "", "x");
    const y = x.mul(x).add(x);
    y.backward();
    y.backward();
    expect(x.grad).toBeCloseTo(2 * 1.5 + 1);
    const steps = backwardTrace(y);
    expect(steps.at(-1)!.grads[toGraph(y).nodes.find((n) => n.label === "x")!.id]).toBeCloseTo(4);
  });
});

describe("routeEdges", () => {
  const W = 132;
  const H = 58;
  const opts = { nodeWidth: W, nodeHeight: H, portSpread: 18 };
  const setup = (expr: string, vars: Record<string, number>) => {
    const g = toGraph(parseExpr(expr, vars));
    const layout = layoutGraph(g, { colGap: W + 70, rowGap: H + 30, margin: W / 2 + 16 });
    return { g, layout, routes: routeEdges(g, layout, opts) };
  };

  /** Every segment that overlaps a box horizontally must pass above or below it. */
  function expectNoOverlap(g: Graph, layout: Layout, routes: EdgeRoute[]) {
    for (const { edge, points } of routes) {
      for (const n of g.nodes) {
        if (n.id === edge.from || n.id === edge.to) continue;
        const { x, y } = layout.positions[n.id]!;
        for (let i = 1; i < points.length; i++) {
          const a = points[i - 1]!;
          const b = points[i]!;
          if (Math.max(a.x, b.x) <= x - W / 2 || Math.min(a.x, b.x) >= x + W / 2) continue;
          expect(a.y).toBe(b.y); // only flat runs cross a column
          expect(Math.abs(a.y - y)).toBeGreaterThan(H / 2);
        }
      }
    }
  }

  it("draws neighbouring-column edges straight from port to port", () => {
    const { g, layout, routes } = setup("tanh(a*b + c)", { a: 2, b: -3, c: 6.5 });
    const mul = g.nodes.find((n) => n.op === "*")!;
    const add = g.nodes.find((n) => n.op === "+")!;
    const r = routes.find((r) => r.edge.from === mul.id && r.edge.to === add.id)!;
    expect(r.points).toEqual([
      { x: layout.positions[mul.id]!.x + W / 2, y: layout.positions[mul.id]!.y },
      { x: layout.positions[add.id]!.x - W / 2, y: layout.positions[add.id]!.y - 9 },
    ]);
  });

  it("routes a skip-column edge around the node it would have crossed", () => {
    // x feeds both x*x (column 1) and the final + (column 2); all three sit on
    // one row, so the straight x -> + edge would run right through x*x.
    const { g, layout, routes } = setup("x*x + x", { x: 3 });
    const x = g.nodes.find((n) => n.label === "x")!;
    const add = g.nodes.find((n) => n.op === "+")!;
    const mul = g.nodes.find((n) => n.op === "*")!;
    expect(layout.positions[x.id]!.y).toBe(layout.positions[mul.id]!.y);
    const skip = routes.find((r) => r.edge.from === x.id && r.edge.to === add.id)!;
    expect(skip.points).toHaveLength(4);
    expectNoOverlap(g, layout, routes);
  });

  it("keeps every edge off every box in a deeper graph", () => {
    const { g, layout, routes } = setup("(y - (w*x + b))^2 + w*w + log(1 + exp(x))", { w: 0.5, x: 2, b: 0.1, y: 3 });
    expect(routes).toHaveLength(g.edges.length);
    expect(routes.some((r) => r.points.length > 2)).toBe(true);
    expectNoOverlap(g, layout, routes);
  });

  it("turns points into SVG path data with flat runs and S-curves", () => {
    expect(routePath([])).toBe("");
    expect(
      routePath([
        { x: 0, y: 0 },
        { x: 10, y: 20 },
        { x: 30, y: 20 },
      ]),
    ).toBe("M 0 0 C 5 0, 5 20, 10 20 L 30 20");
  });
});
