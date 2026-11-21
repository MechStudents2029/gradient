import { parseExpr } from "./expr.js";
import type { Value } from "./value.js";

/**
 * Taylor-mode differentiation ("jets").
 *
 * `backward()` gives one number per node: a first derivative. To get f'', f''',
 * ... at a point we push a whole truncated Taylor series through the same graph
 * instead of a single number. Substitute x = a + t; every node u then becomes
 *
 *     u(a + t) = u₀ + u₁ t + u₂ t² + ... + uₙ tⁿ + O(tⁿ⁺¹),   uₖ = u⁽ᵏ⁾(a) / k!
 *
 * The input x is the jet [a, 1, 0, ..., 0] and constants are [c, 0, ..., 0].
 * Each op has an exact recurrence on these coefficient arrays (below), so one
 * forward sweep in topological order yields every coefficient of f up to order
 * n with no finite differences and no symbolic algebra. Derivatives are then
 * f⁽ᵏ⁾(a) = k! · fₖ. The cost is O(n²) per node.
 *
 * Most recurrences come from writing the op as an ODE in t. For y = eᵘ,
 * y' = y u', and matching coefficients of tᵏ⁻¹ gives
 *
 *     k yₖ = Σⱼ₌₁..ₖ j uⱼ yₖ₋ⱼ
 *
 * which only uses y₀..yₖ₋₁, so the coefficients come out one at a time.
 */
export type Jet = number[];

/** u + v, coefficient-wise. */
export function jetAdd(u: Jet, v: Jet): Jet {
  return u.map((x, k) => x + v[k]!);
}

/** u · v: the Cauchy product, truncated to the same order. */
export function jetMul(u: Jet, v: Jet): Jet {
  return u.map((_, k) => {
    let s = 0;
    for (let j = 0; j <= k; j++) s += u[j]! * v[k - j]!;
    return s;
  });
}

/** eᵘ: y' = y u'  ⇒  k yₖ = Σ j uⱼ yₖ₋ⱼ. */
export function jetExp(u: Jet): Jet {
  const y = [Math.exp(u[0]!)];
  for (let k = 1; k < u.length; k++) {
    let s = 0;
    for (let j = 1; j <= k; j++) s += j * u[j]! * y[k - j]!;
    y.push(s / k);
  }
  return y;
}

/** ln u: u y' = u'  ⇒  yₖ = (uₖ − (1/k) Σⱼ₌₁..ₖ₋₁ j yⱼ uₖ₋ⱼ) / u₀. */
export function jetLog(u: Jet): Jet {
  const u0 = u[0]!;
  const y = [Math.log(u0)];
  for (let k = 1; k < u.length; k++) {
    let s = 0;
    for (let j = 1; j < k; j++) s += j * y[j]! * u[k - j]!;
    y.push((u[k]! - s / k) / u0);
  }
  return y;
}

/**
 * uᵖ for a constant p. A non-negative integer p uses repeated squaring, which is
 * exact (so polynomials come out exactly) and fine when u₀ = 0. Otherwise
 * u y' = p y u' gives
 *
 *     yₖ = 1/(k u₀) · Σⱼ₌₁..ₖ (p j − (k − j)) uⱼ yₖ₋ⱼ
 *
 * which needs u₀ ≠ 0: uᵖ is not analytic where its base is 0.
 */
export function jetPow(u: Jet, p: number): Jet {
  if (Number.isInteger(p) && p >= 0) {
    let result: Jet = u.map((_, k) => (k === 0 ? 1 : 0));
    let base = u;
    for (let e = p; e > 0; e = Math.floor(e / 2)) {
      if (e % 2 === 1) result = jetMul(result, base);
      if (e > 1) base = jetMul(base, base);
    }
    return result;
  }
  const u0 = u[0]!;
  if (u0 === 0) throw new RangeError(`x^${p} has no Taylor series where its base is 0`);
  const y = [u0 ** p];
  for (let k = 1; k < u.length; k++) {
    let s = 0;
    for (let j = 1; j <= k; j++) s += (p * j - (k - j)) * u[j]! * y[k - j]!;
    y.push(s / (k * u0));
  }
  return y;
}

/** sin u and cos u together: s' = c u', c' = −s u'. */
export function jetSinCos(u: Jet): [Jet, Jet] {
  const s = [Math.sin(u[0]!)];
  const c = [Math.cos(u[0]!)];
  for (let k = 1; k < u.length; k++) {
    let ss = 0;
    let cc = 0;
    for (let j = 1; j <= k; j++) {
      ss += j * u[j]! * c[k - j]!;
      cc -= j * u[j]! * s[k - j]!;
    }
    s.push(ss / k);
    c.push(cc / k);
  }
  return [s, c];
}

/**
 * y = g(u) where g' = h(g) for a polynomial h, e.g. tanh' = 1 − tanh² and
 * σ' = σ − σ². Builds w = h(y) alongside y: k yₖ = Σ j uⱼ wₖ₋ⱼ.
 */
function jetRiccati(u: Jet, y0: number, h: (y: Jet, k: number) => number): Jet {
  const y = [y0];
  const w: number[] = [h(y, 0)];
  for (let k = 1; k < u.length; k++) {
    let s = 0;
    for (let j = 1; j <= k; j++) s += j * u[j]! * w[k - j]!;
    y.push(s / k);
    w.push(h(y, k));
  }
  return y;
}

/** Coefficient k of y², using y₀..yₖ. */
const square = (y: Jet, k: number): number => {
  let s = 0;
  for (let i = 0; i <= k; i++) s += y[i]! * y[k - i]!;
  return s;
};

/** tanh u: y' = (1 − y²) u'. */
export function jetTanh(u: Jet): Jet {
  return jetRiccati(u, Math.tanh(u[0]!), (y, k) => (k === 0 ? 1 : 0) - square(y, k));
}

/** σ(u): y' = (y − y²) u'. */
export function jetSigmoid(u: Jet): Jet {
  return jetRiccati(u, 1 / (1 + Math.exp(-u[0]!)), (y, k) => y[k]! - square(y, k));
}

/**
 * relu u. Away from 0 it is either u or 0, so the series is exact. At u₀ = 0 it
 * has no Taylor series; we follow the engine's convention (slope 0 at 0) and
 * return the zero jet.
 */
export function jetRelu(u: Jet): Jet {
  return u[0]! > 0 ? u.slice() : u.map(() => 0);
}

/**
 * Push a degree-`order` jet through an existing `Value` graph, seeding
 * `variable` as x = variable.data + t. Returns the coefficients of `root`.
 */
export function graphJet(root: Value, variable: Value | null, order: number): Jet {
  if (!Number.isInteger(order) || order < 0) throw new RangeError(`order must be a non-negative integer, got ${order}`);
  const jets = new Map<Value, Jet>();
  const constant = (c: number): Jet => Array.from({ length: order + 1 }, (_, k) => (k === 0 ? c : 0));
  for (const v of root.topo()) {
    const [a, b] = v.parents.map((p) => jets.get(p)!);
    let jet: Jet;
    switch (v.op) {
      case "": {
        jet = constant(v.data);
        if (v === variable && order >= 1) jet[1] = 1;
        break;
      }
      case "+":
        jet = jetAdd(a!, b!);
        break;
      case "*":
        jet = jetMul(a!, b!);
        break;
      case "pow":
        jet = jetPow(a!, exponentOf(v));
        break;
      case "exp":
        jet = jetExp(a!);
        break;
      case "log":
        jet = jetLog(a!);
        break;
      case "sin":
        jet = jetSinCos(a!)[0];
        break;
      case "cos":
        jet = jetSinCos(a!)[1];
        break;
      case "tanh":
        jet = jetTanh(a!);
        break;
      case "sigmoid":
        jet = jetSigmoid(a!);
        break;
      case "relu":
        jet = jetRelu(a!);
        break;
    }
    jets.set(v, jet);
  }
  return jets.get(root)!;
}

/** `pow` nodes store their exponent so the jet sweep can reuse it. */
function exponentOf(v: Value): number {
  if (v.exponent === undefined) throw new Error("pow node is missing its exponent");
  return v.exponent;
}

export interface TaylorOptions {
  /** Name of the expansion variable (default `x`). */
  variable?: string;
  /** Values for any other names in the expression; they are held constant. */
  vars?: Readonly<Record<string, number>>;
}

/**
 * Taylor coefficients of a parsed 1-D expression around x = a:
 *
 *     f(x) ≈ Σₖ₌₀..ₙ cₖ (x − a)ᵏ,   cₖ = f⁽ᵏ⁾(a) / k!
 */
export function taylorCoefficients(src: string, a: number, order: number, opts: TaylorOptions = {}): number[] {
  const name = opts.variable ?? "x";
  const root = parseExpr(src, { ...opts.vars, [name]: a });
  // Number literals are labelled with their digits, so only the variable's own leaf has this label.
  const leaf = root.topo().find((v) => v.op === "" && v.label === name) ?? null;
  return graphJet(root, leaf, order);
}

/** f(a), f'(a), ..., f⁽ⁿ⁾(a): the Taylor coefficients scaled by k!. */
export function derivativesAt(src: string, a: number, order: number, opts: TaylorOptions = {}): number[] {
  let fact = 1;
  return taylorCoefficients(src, a, order, opts).map((c, k) => {
    if (k > 0) fact *= k;
    return c * fact;
  });
}

/** Evaluate Σ cₖ (x − a)ᵏ by Horner's rule, using the first `order + 1` coefficients. */
export function taylorPolynomial(coeffs: readonly number[], a: number, x: number, order = coeffs.length - 1): number {
  const t = x - a;
  let s = 0;
  for (let k = Math.min(order, coeffs.length - 1); k >= 0; k--) s = s * t + coeffs[k]!;
  return s;
}
