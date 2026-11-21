import { parseExpr, taylorCoefficients, taylorPolynomial } from "../src/index.js";

interface Preset {
  name: string;
  expr: string;
  range: [number, number];
  a: number;
}

const PRESETS: Preset[] = [
  { name: "sin x", expr: "sin(x)", range: [-7, 7], a: 0 },
  { name: "eˣ", expr: "exp(x)", range: [-4, 3], a: 0 },
  { name: "cos x", expr: "cos(x)", range: [-7, 7], a: 1 },
  { name: "ln(1 + x)", expr: "log(1 + x)", range: [-0.95, 3], a: 0 },
  { name: "sigmoid", expr: "sigmoid(x)", range: [-8, 8], a: 0 },
  { name: "tanh", expr: "tanh(x)", range: [-4, 4], a: 0 },
  { name: "1 / (1 + x²)", expr: "1 / (1 + x^2)", range: [-3, 3], a: 0 },
  { name: "exp(sin x)", expr: "exp(sin(x))", range: [-5, 5], a: 0.5 },
  { name: "cubic (exact at N ≥ 3)", expr: "x^3 - 2*x + 1", range: [-3, 3], a: 1 },
];

const W = 900;
const H = 440;
const PAD = 36;
const SAMPLES = 480;
const SVG_NS = "http://www.w3.org/2000/svg";

const $ = <T extends HTMLElement>(id: string): T => document.getElementById(id) as T;
const presetEl = $<HTMLSelectElement>("preset");
const exprEl = $<HTMLInputElement>("expr");
const rangeEl = $<HTMLInputElement>("range");
const aEl = $<HTMLInputElement>("a");
const nEl = $<HTMLInputElement>("n");
const aOut = $<HTMLOutputElement>("a-out");
const nOut = $<HTMLOutputElement>("n-out");
const statusEl = $<HTMLParagraphElement>("status");
const polyEl = $<HTMLParagraphElement>("poly");
const tbody = document.querySelector("#coeffs tbody") as HTMLTableSectionElement;
const svg = document.getElementById("plot") as unknown as SVGSVGElement;

let src = "";
let xmin = -1;
let xmax = 1;
/** Samples of f on the current range; NaN where f is undefined. */
let fx: number[] = [];

const fmt = (x: number): string => {
  if (!Number.isFinite(x)) return String(x);
  if (Number.isInteger(x)) return String(x);
  const m = Math.abs(x);
  return m !== 0 && (m < 1e-3 || m >= 1e5) ? x.toExponential(3) : String(Number(x.toPrecision(5)));
};

const color = (k: number, n: number): string => `hsl(${210 - (170 * (k - 1)) / Math.max(1, n - 1)} 85% 62%)`;

function el<K extends keyof SVGElementTagNameMap>(tag: K, attrs: Record<string, string | number>): SVGElementTagNameMap[K] {
  const e = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, String(v));
  return e;
}

const xs = (): number[] => Array.from({ length: SAMPLES + 1 }, (_, i) => xmin + ((xmax - xmin) * i) / SAMPLES);

/** Polyline path that breaks at non-finite or far-off-screen values. */
function pathFor(ys: number[], sx: (x: number) => number, sy: (y: number) => number, ylo: number, yhi: number): string {
  const span = yhi - ylo;
  let d = "";
  let pen = false;
  xs().forEach((x, i) => {
    const y = ys[i]!;
    if (!Number.isFinite(y) || y < ylo - 4 * span || y > yhi + 4 * span) {
      pen = false;
      return;
    }
    d += `${pen ? "L" : "M"}${sx(x).toFixed(1)} ${sy(y).toFixed(1)}`;
    pen = true;
  });
  return d;
}

function sample(): void {
  fx = xs().map((x) => {
    const y = parseExpr(src, { x }).data;
    return Number.isFinite(y) ? y : NaN;
  });
}

function render(): void {
  const a = Number(aEl.value);
  const n = Number(nEl.value);
  aOut.textContent = fmt(a);
  nOut.textContent = String(n);
  svg.replaceChildren();
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  statusEl.classList.remove("error");

  let coeffs: number[];
  try {
    coeffs = taylorCoefficients(src, a, n);
  } catch (err) {
    coeffs = [];
    statusEl.textContent = err instanceof Error ? err.message : String(err);
    statusEl.classList.add("error");
  }

  // y-range from f itself, padded, so wild polynomials get clipped instead of squashing the plot.
  const finite = fx.filter(Number.isFinite);
  let ylo = finite.length ? Math.min(...finite) : -1;
  let yhi = finite.length ? Math.max(...finite) : 1;
  if (yhi - ylo < 1e-9) {
    ylo -= 1;
    yhi += 1;
  }
  const padY = 0.25 * (yhi - ylo);
  ylo -= padY;
  yhi += padY;
  const sx = (x: number): number => PAD + ((x - xmin) / (xmax - xmin)) * (W - 2 * PAD);
  const sy = (y: number): number => H - PAD - ((y - ylo) / (yhi - ylo)) * (H - 2 * PAD);

  const clip = el("clipPath", { id: "area" });
  clip.append(el("rect", { x: PAD, y: PAD, width: W - 2 * PAD, height: H - 2 * PAD }));
  svg.append(clip);
  if (ylo < 0 && yhi > 0) svg.append(el("line", { x1: PAD, x2: W - PAD, y1: sy(0), y2: sy(0), class: "axis" }));
  if (xmin < 0 && xmax > 0) svg.append(el("line", { x1: sx(0), x2: sx(0), y1: PAD, y2: H - PAD, class: "axis" }));
  for (const [x, anchor] of [[xmin, "start"], [xmax, "end"]] as const) {
    const t = el("text", { x: sx(x), y: H - PAD + 18, class: "tick", "text-anchor": anchor });
    t.textContent = fmt(x);
    svg.append(t);
  }
  for (const y of [ylo, yhi]) {
    const t = el("text", { x: PAD - 4, y: sy(y) + 4, class: "tick", "text-anchor": "end" });
    t.textContent = fmt(Number(y.toPrecision(3)));
    svg.append(t);
  }

  const g = el("g", { "clip-path": "url(#area)" });
  svg.append(g);
  for (let k = 1; k <= Math.min(n, coeffs.length - 1); k++) {
    const ys = xs().map((x) => taylorPolynomial(coeffs, a, x, k));
    g.append(el("path", { d: pathFor(ys, sx, sy, ylo, yhi), class: "taylor", stroke: color(k, n), "stroke-width": k === n ? 2.6 : 1.3 }));
  }
  g.append(el("path", { d: pathFor(fx, sx, sy, ylo, yhi), class: "f" }));
  if (coeffs.length && Number.isFinite(coeffs[0]!)) {
    g.append(el("line", { x1: sx(a), x2: sx(a), y1: PAD, y2: H - PAD, class: "at" }));
    g.append(el("circle", { cx: sx(a), cy: sy(coeffs[0]!), r: 5, class: "point" }));
  }

  // Legend.
  svg.append(el("rect", { x: PAD + 2, y: PAD + 1, width: 46 * (n + 1) + 4, height: 20, rx: 4, class: "legend-bg" }));
  const legend = [["f(x)", "var(--text)"], ...Array.from({ length: n }, (_, i) => [`P${i + 1}`, color(i + 1, n)])];
  legend.forEach(([label, c], i) => {
    const t = el("text", { x: PAD + 8 + i * 46, y: PAD + 14, class: "legend", fill: c! });
    t.textContent = label!;
    svg.append(t);
  });

  tbody.replaceChildren();
  let fact = 1;
  coeffs.forEach((c, k) => {
    if (k > 0) fact *= k;
    const row = document.createElement("tr");
    for (const text of [String(k), fmt(c * fact), fmt(c)]) {
      const td = document.createElement("td");
      td.textContent = text;
      row.append(td);
    }
    tbody.append(row);
  });
  polyEl.textContent = coeffs.length ? `P${n}(x) = ${polyString(coeffs, a)}` : "";

  if (coeffs.length) {
    const x = a + 0.1 * (xmax - xmin);
    const fxv = parseExpr(src, { x }).data;
    statusEl.textContent =
      `f(${fmt(a)}) = ${fmt(coeffs[0]!)}, f′(${fmt(a)}) = ${fmt(coeffs[1] ?? 0)}. ` +
      `At x = ${fmt(x)}: |f − P${n}| = ${fmt(Math.abs(fxv - taylorPolynomial(coeffs, a, x)))}.`;
  }
}

function polyString(coeffs: number[], a: number): string {
  const base = a === 0 ? "x" : `(x ${a > 0 ? "−" : "+"} ${fmt(Math.abs(a))})`;
  const sup = (k: number): string => String(k).replace(/\d/g, (d) => "⁰¹²³⁴⁵⁶⁷⁸⁹"[Number(d)]!);
  const terms: string[] = [];
  coeffs.forEach((c, k) => {
    if (Math.abs(c) < 1e-12) return;
    const mag = fmt(Math.abs(c));
    const body = k === 0 ? mag : `${mag === "1" ? "" : mag}${base}${k > 1 ? sup(k) : ""}`;
    terms.push(terms.length === 0 ? (c < 0 ? `−${body}` : body) : `${c < 0 ? "−" : "+"} ${body}`);
  });
  return terms.join(" ") || "0";
}

function build(): void {
  try {
    const m = /^\s*(-?[\d.]+)\s*(?:\.\.|,|to)\s*(-?[\d.]+)\s*$/.exec(rangeEl.value);
    const lo = m ? Number(m[1]) : NaN;
    const hi = m ? Number(m[2]) : NaN;
    if (!(lo < hi)) throw new Error("x range must look like -3..3");
    src = exprEl.value;
    parseExpr(src, { x: (lo + hi) / 2 }); // surface parse errors before sampling
    xmin = lo;
    xmax = hi;
    aEl.min = String(lo);
    aEl.max = String(hi);
    if (Number(aEl.value) < lo || Number(aEl.value) > hi) aEl.value = String((lo + hi) / 2);
    sample();
    render();
  } catch (err) {
    svg.replaceChildren();
    tbody.replaceChildren();
    polyEl.textContent = "";
    statusEl.textContent = err instanceof Error ? err.message : String(err);
    statusEl.classList.add("error");
  }
}

function loadPreset(i: number): void {
  const p = PRESETS[i]!;
  exprEl.value = p.expr;
  rangeEl.value = `${p.range[0]}..${p.range[1]}`;
  aEl.min = String(p.range[0]);
  aEl.max = String(p.range[1]);
  aEl.value = String(p.a);
  build();
}

PRESETS.forEach((p, i) => presetEl.append(new Option(p.name, String(i))));
presetEl.addEventListener("change", () => loadPreset(Number(presetEl.value)));
$<HTMLFormElement>("controls").addEventListener("submit", (e) => {
  e.preventDefault();
  build();
});
aEl.addEventListener("input", render);
nEl.addEventListener("input", render);

loadPreset(0);
