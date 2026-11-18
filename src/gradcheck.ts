/**
 * Central finite difference: f'(x) ≈ (f(x+h) - f(x-h)) / 2h, error O(h^2).
 * Used to check the analytic gradients from `backward()`.
 */
export function numericGrad(f: (xs: number[]) => number, xs: number[], h = 1e-5): number[] {
  return xs.map((_, i) => {
    const plus = xs.slice();
    const minus = xs.slice();
    plus[i]! += h;
    minus[i]! -= h;
    return (f(plus) - f(minus)) / (2 * h);
  });
}
