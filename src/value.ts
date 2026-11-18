/**
 * A scalar node in a computation graph.
 *
 * Every arithmetic op returns a new Value that remembers its parents and a
 * local `backwardStep` closure. Calling `backward()` on the output walks the
 * graph in reverse topological order and applies the chain rule:
 *
 *     dL/dparent += dL/dchild * dchild/dparent
 */
export type Op = "" | "+" | "*" | "pow" | "exp" | "log" | "tanh" | "relu";

export type Scalar = Value | number;

export class Value {
  data: number;
  grad = 0;
  readonly parents: readonly Value[];
  readonly op: Op;
  label: string;
  private backwardStep: () => void = () => {};

  constructor(data: number, parents: readonly Value[] = [], op: Op = "", label = "") {
    this.data = data;
    this.parents = parents;
    this.op = op;
    this.label = label;
  }

  static from(x: Scalar): Value {
    return x instanceof Value ? x : new Value(x);
  }

  add(other: Scalar): Value {
    const o = Value.from(other);
    const out = new Value(this.data + o.data, [this, o], "+");
    out.backwardStep = () => {
      this.grad += out.grad;
      o.grad += out.grad;
    };
    return out;
  }

  mul(other: Scalar): Value {
    const o = Value.from(other);
    const out = new Value(this.data * o.data, [this, o], "*");
    out.backwardStep = () => {
      this.grad += o.data * out.grad;
      o.grad += this.data * out.grad;
    };
    return out;
  }

  /** x^k for a constant exponent k: d/dx = k * x^(k-1). */
  pow(k: number): Value {
    const out = new Value(this.data ** k, [this], "pow");
    out.backwardStep = () => {
      this.grad += k * this.data ** (k - 1) * out.grad;
    };
    return out;
  }

  neg(): Value {
    return this.mul(-1);
  }

  sub(other: Scalar): Value {
    return this.add(Value.from(other).neg());
  }

  div(other: Scalar): Value {
    return this.mul(Value.from(other).pow(-1));
  }

  /** d/dx e^x = e^x. */
  exp(): Value {
    const e = Math.exp(this.data);
    const out = new Value(e, [this], "exp");
    out.backwardStep = () => {
      this.grad += e * out.grad;
    };
    return out;
  }

  /** d/dx ln x = 1/x. */
  log(): Value {
    const out = new Value(Math.log(this.data), [this], "log");
    out.backwardStep = () => {
      this.grad += (1 / this.data) * out.grad;
    };
    return out;
  }

  /** d/dx tanh x = 1 - tanh^2 x. */
  tanh(): Value {
    const t = Math.tanh(this.data);
    const out = new Value(t, [this], "tanh");
    out.backwardStep = () => {
      this.grad += (1 - t * t) * out.grad;
    };
    return out;
  }

  /** ReLU: derivative is 1 for x > 0, else 0. */
  relu(): Value {
    const out = new Value(Math.max(0, this.data), [this], "relu");
    out.backwardStep = () => {
      this.grad += (this.data > 0 ? 1 : 0) * out.grad;
    };
    return out;
  }

  /** Nodes reachable from this one, parents before children. */
  topo(): Value[] {
    const order: Value[] = [];
    const seen = new Set<Value>();
    const visit = (v: Value): void => {
      if (seen.has(v)) return;
      seen.add(v);
      for (const p of v.parents) visit(p);
      order.push(v);
    };
    visit(this);
    return order;
  }

  /** Reverse-mode sweep: seed dL/dL = 1, then apply the chain rule backwards. */
  backward(): void {
    const order = this.topo();
    for (const v of order) v.grad = 0;
    this.grad = 1;
    for (let i = order.length - 1; i >= 0; i--) order[i]!.backwardStep();
  }
}
