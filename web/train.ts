import { Trainer, decisionGrid, makeDataset } from "../src/index.js";
import type { Dataset, DatasetName } from "../src/index.js";

const LO = -1.5;
const HI = 1.5;
const RES = 50;
const MAX_EPOCHS = 5000;

const $ = <T extends HTMLElement>(id: string): T => document.getElementById(id) as T;
const datasetEl = $<HTMLSelectElement>("dataset");
const seedEl = $<HTMLInputElement>("seed");
const lrEl = $<HTMLInputElement>("lr");
const hiddenEl = $<HTMLInputElement>("hidden");
const speedEl = $<HTMLInputElement>("speed");
const startBtn = $<HTMLButtonElement>("start");
const stepBtn = $<HTMLButtonElement>("step");
const resetBtn = $<HTMLButtonElement>("reset");
const statusEl = $<HTMLParagraphElement>("status");
const boundary = $<HTMLCanvasElement>("boundary");
const lossCanvas = $<HTMLCanvasElement>("loss");
const bctx = boundary.getContext("2d")!;
const lctx = lossCanvas.getContext("2d")!;

// Low-res offscreen buffer for the heatmap, scaled up with smoothing.
const heat = document.createElement("canvas");
heat.width = RES;
heat.height = RES;
const hctx = heat.getContext("2d")!;
const heatImg = hctx.createImageData(RES, RES);

const css = getComputedStyle(document.documentElement);
const MUTED = css.getPropertyValue("--muted").trim() || "#8b93a7";
const BORDER = css.getPropertyValue("--border").trim() || "#39404f";
const ACCENT = css.getPropertyValue("--accent").trim() || "#f5a524";

const lr = (): number => 10 ** Number(lrEl.value);
const fmt = (x: number, p = 4): string => (Number.isFinite(x) ? String(Number(x.toPrecision(p))) : String(x));

let data: Dataset;
let trainer: Trainer;
let running = false;
let raf = 0;

function rebuild(): void {
  const seed = Math.max(0, Math.floor(Number(seedEl.value) || 0));
  data = makeDataset(datasetEl.value as DatasetName, { n: 120, seed });
  trainer = new Trainer(data, { hidden: Number(hiddenEl.value), lr: lr(), seed });
  draw();
}

// Blend blue (0) → panel grey (0.5) → orange (1).
const C0 = [56, 132, 255];
const CM = [23, 26, 33];
const C1 = [245, 165, 36];
function colour(p: number): [number, number, number] {
  const t = Math.min(1, Math.max(0, p));
  const [a, b, s] = t < 0.5 ? [C0, CM, t / 0.5] : [CM, C1, (t - 0.5) / 0.5];
  return [0, 1, 2].map((i) => a[i]! + (b[i]! - a[i]!) * s) as [number, number, number];
}

function drawBoundary(): void {
  const grid = decisionGrid(trainer.net, { res: RES, lo: LO, hi: HI });
  for (let i = 0; i < grid.length; i++) {
    const [r, g, b] = colour(grid[i]!);
    heatImg.data[4 * i] = r;
    heatImg.data[4 * i + 1] = g;
    heatImg.data[4 * i + 2] = b;
    heatImg.data[4 * i + 3] = 255;
  }
  hctx.putImageData(heatImg, 0, 0);
  const W = boundary.width;
  const H = boundary.height;
  bctx.imageSmoothingEnabled = true;
  bctx.drawImage(heat, 0, 0, W, H);

  // 0.5 contour: mark cells whose right or lower neighbour is on the other side.
  const cell = W / RES;
  bctx.strokeStyle = "rgba(255,255,255,0.85)";
  bctx.lineWidth = 1.5;
  bctx.beginPath();
  for (let r = 0; r < RES; r++) {
    for (let c = 0; c < RES; c++) {
      const v = grid[r * RES + c]! >= 0.5;
      if (c + 1 < RES && v !== grid[r * RES + c + 1]! >= 0.5) {
        bctx.moveTo((c + 1) * cell, r * cell);
        bctx.lineTo((c + 1) * cell, (r + 1) * cell);
      }
      if (r + 1 < RES && v !== grid[(r + 1) * RES + c]! >= 0.5) {
        bctx.moveTo(c * cell, (r + 1) * cell);
        bctx.lineTo((c + 1) * cell, (r + 1) * cell);
      }
    }
  }
  bctx.stroke();

  const sx = (x: number): number => ((x - LO) / (HI - LO)) * W;
  const sy = (y: number): number => H - ((y - LO) / (HI - LO)) * H;
  data.xs.forEach(([x, y], i) => {
    bctx.beginPath();
    bctx.arc(sx(x), sy(y), 4.5, 0, 2 * Math.PI);
    bctx.fillStyle = data.ys[i] === 1 ? "#ffcf70" : "#7fb2ff";
    bctx.fill();
    bctx.lineWidth = 1.2;
    bctx.strokeStyle = "#0f1115";
    bctx.stroke();
  });
}

function drawLoss(): void {
  const W = lossCanvas.width;
  const H = lossCanvas.height;
  const P = 40;
  lctx.clearRect(0, 0, W, H);
  const losses = trainer.losses;
  const n = Math.max(losses.length, 50);
  const logs = losses.map((l) => Math.log10(Math.max(l, 1e-6)));
  const top = logs.length ? Math.ceil(Math.max(...logs)) : 0;
  const bottom = logs.length ? Math.min(Math.floor(Math.min(...logs)), top - 1) : -2;
  const sx = (i: number): number => P + (i / (n - 1)) * (W - 2 * P);
  const sy = (v: number): number => H - P - ((v - bottom) / (top - bottom)) * (H - 2 * P);

  lctx.font = "11px ui-monospace, Menlo, monospace";
  lctx.fillStyle = MUTED;
  lctx.strokeStyle = BORDER;
  lctx.lineWidth = 1;
  lctx.textAlign = "right";
  for (let d = bottom; d <= top; d++) {
    lctx.beginPath();
    lctx.moveTo(P, sy(d));
    lctx.lineTo(W - P, sy(d));
    lctx.stroke();
    lctx.fillText(`1e${d}`, P - 4, sy(d) + 4);
  }
  lctx.textAlign = "center";
  lctx.fillText("0", sx(0), H - P + 16);
  lctx.fillText(String(n - 1), sx(n - 1), H - P + 16);
  lctx.fillText("epoch", W / 2, H - P + 16);

  if (!logs.length) return;
  lctx.strokeStyle = ACCENT;
  lctx.lineWidth = 2;
  lctx.beginPath();
  logs.forEach((v, i) => (i ? lctx.lineTo(sx(i), sy(v)) : lctx.moveTo(sx(i), sy(v))));
  lctx.stroke();
}

function draw(): void {
  drawBoundary();
  drawLoss();
  const last = trainer.losses.at(-1);
  statusEl.textContent =
    `epoch ${trainer.epoch} · loss ${last === undefined ? "—" : fmt(last)} · ` +
    `accuracy ${(100 * trainer.accuracy()).toFixed(1)}% · ${trainer.net.parameters().length} parameters`;
  startBtn.textContent = running ? "Pause" : "Start";
}

function frame(): void {
  if (!running) return;
  trainer.lr = lr();
  trainer.step(Number(speedEl.value));
  if (trainer.epoch >= MAX_EPOCHS || !Number.isFinite(trainer.losses.at(-1)!)) running = false;
  draw();
  if (running) raf = requestAnimationFrame(frame);
}

function setRunning(on: boolean): void {
  running = on;
  cancelAnimationFrame(raf);
  if (on) raf = requestAnimationFrame(frame);
  draw();
}

function syncOutputs(): void {
  $<HTMLOutputElement>("lr-out").textContent = fmt(lr(), 3);
  $<HTMLOutputElement>("hidden-out").textContent = hiddenEl.value;
  $<HTMLOutputElement>("speed-out").textContent = speedEl.value;
}

startBtn.addEventListener("click", () => setRunning(!running));
stepBtn.addEventListener("click", () => {
  setRunning(false);
  trainer.lr = lr();
  trainer.step();
  draw();
});
resetBtn.addEventListener("click", () => {
  setRunning(false);
  rebuild();
});
for (const el of [datasetEl, seedEl, hiddenEl]) {
  el.addEventListener("change", () => {
    setRunning(false);
    syncOutputs();
    rebuild();
  });
}
for (const el of [lrEl, hiddenEl, speedEl]) el.addEventListener("input", syncOutputs);

syncOutputs();
rebuild();
