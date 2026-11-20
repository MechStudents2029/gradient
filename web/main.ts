import {
  backwardTrace,
  layoutGraph,
  parseExpr,
  parseVars,
  toGraph,
  type Graph,
  type GraphNode,
  type TraceStep,
} from "../src/index.js";

interface Preset {
  name: string;
  expr: string;
  vars: string;
}

const PRESETS: Preset[] = [
  { name: "tanh neuron", expr: "tanh(x1*w1 + x2*w2 + b)", vars: "x1 = 2, x2 = 0, w1 = -3, w2 = 1, b = 6.8814" },
  { name: "sigmoid neuron", expr: "sigmoid(w*x + b)", vars: "w = 1.5, x = 0.8, b = -0.5" },
  { name: "tanh(a·b + c)", expr: "tanh(a*b + c)", vars: "a = 2, b = -3, c = 6.5" },
  { name: "reused variable", expr: "x*x + x", vars: "x = 3" },
  { name: "squared error", expr: "(y - (w*x + b))^2", vars: "w = 0.5, x = 2, b = 0.1, y = 3" },
  { name: "softplus", expr: "log(1 + exp(x))", vars: "x = 0.7" },
];

const NODE_W = 132;
const NODE_H = 58;
const SVG_NS = "http://www.w3.org/2000/svg";

const $ = <T extends HTMLElement>(id: string): T => document.getElementById(id) as T;
const presetEl = $<HTMLSelectElement>("preset");
const exprEl = $<HTMLInputElement>("expr");
const varsEl = $<HTMLInputElement>("vars");
const statusEl = $<HTMLParagraphElement>("status");
const progressEl = $<HTMLSpanElement>("progress");
const svg = document.getElementById("graph") as unknown as SVGSVGElement;
const buttons = {
  reset: $<HTMLButtonElement>("reset"),
  back: $<HTMLButtonElement>("back"),
  step: $<HTMLButtonElement>("step"),
  run: $<HTMLButtonElement>("run"),
};

let graph: Graph | null = null;
let trace: TraceStep[] = [];
/** -1 = forward pass only (no gradients yet), 0 = seed, k = after k local steps. */
let cursor = -1;

const fmt = (x: number): string => {
  if (Number.isInteger(x)) return String(x);
  const a = Math.abs(x);
  return a !== 0 && (a < 1e-3 || a >= 1e5) ? x.toExponential(2) : x.toFixed(4);
};

const OP_SYMBOL: Record<string, string> = { "+": "+", "*": "×", pow: "pow" };

function title(n: GraphNode): string {
  const op = OP_SYMBOL[n.op] ?? n.op;
  if (n.op === "") return n.label || fmt(n.data);
  return n.label ? `${n.label} = ${op}` : op;
}

/** Plain-language local derivative for the node whose step just ran. */
function localRule(n: GraphNode, parents: GraphNode[]): string {
  const [a, b] = parents;
  switch (n.op) {
    case "+":
      return "∂out/∂a = 1, ∂out/∂b = 1 (addition copies the gradient)";
    case "*":
      return `∂out/∂a = b = ${fmt(b!.data)}, ∂out/∂b = a = ${fmt(a!.data)}`;
    case "pow":
      return `∂out/∂x = k·x^(k−1)`;
    case "exp":
      return `∂out/∂x = eˣ = ${fmt(n.data)}`;
    case "log":
      return `∂out/∂x = 1/x = ${fmt(1 / a!.data)}`;
    case "tanh":
      return `∂out/∂x = 1 − out² = ${fmt(1 - n.data * n.data)}`;
    case "relu":
      return `∂out/∂x = ${a!.data > 0 ? 1 : 0} (x ${a!.data > 0 ? ">" : "≤"} 0)`;
    case "sigmoid":
      return `∂out/∂x = σ(1 − σ) = ${fmt(n.data * (1 - n.data))}`;
    default:
      return "";
  }
}

function el<K extends keyof SVGElementTagNameMap>(tag: K, attrs: Record<string, string | number>): SVGElementTagNameMap[K] {
  const e = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, String(v));
  return e;
}

function render(): void {
  svg.replaceChildren();
  if (!graph) return;
  const g = graph;
  const layout = layoutGraph(g, { colGap: NODE_W + 70, rowGap: NODE_H + 30, margin: NODE_W / 2 + 16 });
  svg.setAttribute("width", String(layout.width));
  svg.setAttribute("height", String(layout.height));

  const step = cursor >= 0 ? trace[cursor] : undefined;
  const active = step?.active ?? null;
  const done = new Set(trace.slice(1, cursor + 1).map((s) => s.active!));
  const byId = new Map(g.nodes.map((n) => [n.id, n]));

  for (const e of g.edges) {
    const p = layout.positions[e.from]!;
    const c = layout.positions[e.to]!;
    const x1 = p.x + NODE_W / 2;
    const x2 = c.x - NODE_W / 2;
    // Two inputs into one node land at slightly different heights.
    const y2 = c.y + (byId.get(e.to)!.op === "" ? 0 : (e.slot - 0.5) * 18);
    const mx = (x1 + x2) / 2;
    svg.append(
      el("path", {
        d: `M ${x1} ${p.y} C ${mx} ${p.y}, ${mx} ${y2}, ${x2} ${y2}`,
        class: e.to === active ? "edge hot" : "edge",
      }),
    );
  }

  for (const n of g.nodes) {
    const { x, y } = layout.positions[n.id]!;
    const classes = ["node", n.op === "" ? "leaf" : "op"];
    if (done.has(n.id) || (n.op === "" && step && step.grads[n.id] !== 0)) classes.push("done");
    if (n.id === active) classes.push("active");
    const grp = el("g", { class: classes.join(" "), transform: `translate(${x - NODE_W / 2} ${y - NODE_H / 2})` });
    grp.append(el("rect", { width: NODE_W, height: NODE_H, rx: 8 }));
    const lines: [string, string][] = [
      [title(n), "title"],
      [`data ${fmt(n.data)}`, "data"],
    ];
    const grad = step?.grads[n.id];
    lines.push([grad === undefined ? "grad –" : `grad ${fmt(grad)}`, grad ? "grad" : "grad zero"]);
    lines.forEach(([text, cls], i) => {
      const t = el("text", { x: 10, y: 18 + i * 16, class: cls });
      t.textContent = text;
      grp.append(t);
    });
    const tip = el("title", {});
    tip.textContent = `${n.id}: ${title(n)}\ndepth ${n.depth}`;
    grp.append(tip);
    svg.append(grp);
  }

  const last = trace.length - 1;
  progressEl.textContent = cursor < 0 ? "forward pass" : `step ${cursor} / ${last}`;
  buttons.back.disabled = cursor < 0;
  buttons.step.disabled = cursor >= last;
  buttons.run.disabled = cursor >= last;
  buttons.reset.disabled = cursor < 0;

  statusEl.classList.remove("error");
  const root = byId.get(g.root)!;
  if (cursor < 0) {
    statusEl.textContent = `Forward pass: out = ${fmt(root.data)}. ${g.nodes.length} nodes, ${g.edges.length} edges. Press "Step backward" to start.`;
  } else if (!active) {
    statusEl.textContent = `Seed: ∂out/∂out = 1. Every other gradient starts at 0.`;
  } else {
    const n = byId.get(active)!;
    const parents = g.edges.filter((e) => e.to === active).sort((a, b) => a.slot - b.slot).map((e) => byId.get(e.from)!);
    const into = [...new Set(parents)].map((p) => `${title(p)} → ${fmt(step!.grads[p.id]!)}`).join(", ");
    statusEl.textContent =
      `${title(n)}: grad ${fmt(step!.grads[n.id]!)} flows to its inputs. ${localRule(n, parents)}. ` +
      `Inputs now: ${into}.` + (cursor === last ? " Backward pass complete." : "");
  }
}

function build(): void {
  try {
    const out = parseExpr(exprEl.value, parseVars(varsEl.value));
    graph = toGraph(out); // forward values only, before any backward pass
    trace = backwardTrace(out);
    cursor = -1;
    render();
  } catch (err) {
    graph = null;
    trace = [];
    svg.replaceChildren();
    statusEl.textContent = err instanceof Error ? err.message : String(err);
    statusEl.classList.add("error");
    progressEl.textContent = "";
    for (const b of Object.values(buttons)) b.disabled = true;
  }
}

function loadPreset(i: number): void {
  const p = PRESETS[i]!;
  exprEl.value = p.expr;
  varsEl.value = p.vars;
  build();
}

PRESETS.forEach((p, i) => presetEl.append(new Option(p.name, String(i))));
presetEl.addEventListener("change", () => loadPreset(Number(presetEl.value)));
$<HTMLFormElement>("controls").addEventListener("submit", (e) => {
  e.preventDefault();
  build();
});
buttons.reset.addEventListener("click", () => {
  cursor = -1;
  render();
});
buttons.back.addEventListener("click", () => {
  cursor = Math.max(-1, cursor - 1);
  render();
});
buttons.step.addEventListener("click", () => {
  cursor = Math.min(trace.length - 1, cursor + 1);
  render();
});
buttons.run.addEventListener("click", () => {
  cursor = trace.length - 1;
  render();
});
document.addEventListener("keydown", (e) => {
  if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement) return;
  if (e.key === "ArrowRight") buttons.step.click();
  if (e.key === "ArrowLeft") buttons.back.click();
});

loadPreset(0);
