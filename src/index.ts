export { Value } from "./value.js";
export type { Op, Scalar } from "./value.js";
export { numericGrad } from "./gradcheck.js";
export { MLP, Layer, Neuron, mse, sgdStep, mulberry32 } from "./nn.js";
export type { Activation } from "./nn.js";
export { toGraph, layoutGraph, backwardTrace } from "./graph.js";
export type { Graph, GraphNode, GraphEdge, Layout, LayoutOptions, Point, TraceStep } from "./graph.js";
export { parseExpr, parseVars, ParseError, FUNCTIONS } from "./expr.js";
export type { FunctionName } from "./expr.js";
export {
  taylorCoefficients,
  derivativesAt,
  taylorPolynomial,
  graphJet,
  jetAdd,
  jetMul,
  jetExp,
  jetLog,
  jetPow,
  jetSinCos,
  jetTanh,
  jetSigmoid,
  jetRelu,
} from "./taylor.js";
export type { Jet, TaylorOptions } from "./taylor.js";
export { gaussian, xorDataset, moonsDataset, circlesDataset, makeDataset, DATASETS } from "./data.js";
export type { Dataset, DatasetName, DatasetOptions } from "./data.js";
export { predict, accuracy, trainEpoch, decisionGrid, Trainer } from "./train.js";
export type { GridOptions, TrainerOptions } from "./train.js";
