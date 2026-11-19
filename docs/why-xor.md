# Why XOR?

XOR is the smallest dataset a linear model cannot fit. The points (0,1) and
(1,0) are class 1, (0,0) and (1,1) are class 0, and no single line separates
them. A single neuron `σ(w·x + b)` draws exactly one line, so it gets stuck.

Adding a hidden layer of `tanh` units lets the network bend the input space:
each hidden unit draws its own line, and the output neuron combines them. With
4 hidden units, plain gradient descent (η = 1, 2000 steps) drives the MSE down
by more than 10× and classifies all four points correctly. That run is pinned
down as a test in `test/nn.test.ts`.

It is also a good end-to-end check of the engine: if any local derivative were
wrong, training would stall or diverge.
