# Neuron and Value – A Simple Visual Guide

This guide explains how a Neuron is built from Value objects, how a forward pass creates a computation graph, how gradients flow backward, and how Layers and MLPs extend the same idea.

## 1) Value and Neuron (what they are)

- Value
  - Holds a number: `data`
  - Holds a gradient: `grad`
  - Remembers how it was created: `parents` and `op`
  - Knows how to send gradients to its parents: `backwardFn`

- Neuron
  - Has Value parameters: a list of weights and a bias
  - Takes Value inputs
  - Computes: `z = w·x + b` and (optionally) an activation like `tanh(z)`
  - Returns a Value output

Think: Value = “worker that does math and tracks gradients”. Neuron = “small machine made of many Value workers”.

Visual (Neuron made of Value objects):
```
NEURON
  weights: [ Value(w0), Value(w1), Value(w2), ... ]
  bias:      Value(b)

All of these are Value objects:
  - have data and grad
  - remember parents and op
  - take part in backprop
```

## 2) Forward pass (what happens)

When you call `neuron.forward([x1, x2, ...])`:
- The neuron multiplies each input Value by its weight Value (new Value nodes)
- Adds them all up, plus the bias (more Value nodes)
- Applies an activation if configured (another Value node)
- Returns the final Value (the output)

This builds a directed acyclic graph (DAG) of Value nodes. Every intermediate result is a Value too.

Small example (3 inputs):
```
v1 = w0 * x1
v2 = w1 * x2
v3 = w2 * x3
sum1 = bias + v1
sum2 = sum1 + v2
output = sum2 + v3         # (then maybe output = tanh(output))
```

Visual (small computation graph):
```
w0 --\            w1 --\            w2 --\
      (*)--> v1         (*)--> v2         (*)--> v3
x1 --/              x2 --/              x3 --/

bias ----> (+) -> sum1
sum1 ----> (+) -> sum2
v2   ----^
v3   ------------------> (+) -> output
```

## 3) Backward pass (how gradients flow)

When you call `output.backward()`:
1. Set `output.grad = 1.0` (d(output)/d(output))
2. Visit nodes in reverse order (topological order)
3. Each node uses its local derivative to send gradient to its parents
   - Example: if `c = a * b`, then
     - `a.grad += b.data * c.grad`
     - `b.grad += a.data * c.grad`

For the simple 3‑input neuron, you get intuitive gradients like:
- `w0.grad = x1`, `w1.grad = x2`, `w2.grad = x3`, `bias.grad = 1`
- Inputs also get gradients based on the weights
```
output.grad = 1
  ↳ (+) passes 1 to both parents
  ↳ (+) passes 1 to both parents
  ↳ for (*) nodes: send (other.data * grad) to each parent
```

## 4) Update step (how learning happens)

After `backward()`:
```
for each parameter p (a Value):
    p.data = p.data - learning_rate * p.grad
```
This changes the number inside the same Value object (no new objects; references stay the same).

## 5) Layer (many neurons in parallel)

A Layer is “neurons in parallel”:
- Same input Values go into every neuron
- Each neuron has its own weights/bias (their own Value parameters)
- The layer returns a list of output Values (one per neuron)

Visual:
```
inputs (Values)
   │
   ├─> Neuron 1  → y1 (Value)
   ├─> Neuron 2  → y2 (Value)
   ├─> ...
   └─> Neuron M  → yM (Value)
```

Parameter count for a layer with `nin` inputs and `nout` neurons: `nout * (nin + 1)` (weights + bias).

## 6) MLP (layers in sequence)

An MLP chains layers:
- Outputs from Layer 1 (list of Values) become inputs to Layer 2
- This repeats until the final output layer
- Everything is still Value nodes linked in one big graph

Visual:
```
inputs → Layer 1 → Layer 2 → ... → Output Layer → outputs (Values)
```

In code, use:
- `forward(inputs)` when the final layer has 1 output (returns a single Value)
- `forwardAll(inputs)` when the final layer has multiple outputs (returns a list of Values)

## 7) Training loop (the 5 steps)

1. Forward pass: compute predictions (Values)
2. Build a scalar loss Value (e.g., hinge loss or MSE); add L2 if needed
3. Backward pass: `loss.backward()` fills `grad` for all upstream Values
4. Update: for each parameter Value, subtract `learning_rate * grad`
5. Zero gradients before the next step if you reuse the same graph

Tip for hinge loss (SVM max‑margin): labels should be in `{−1, +1}`.

## 8) Key takeaways

- Neuron, Layer, and MLP are all built from Value nodes
- Forward pass creates the computation graph; backward pass uses the chain rule across that graph
- Parameters are just Value objects you update
- The same mental model scales from a single neuron to an MLP

## 9) See it in action

- Run the demos to print numbers and save PNG graphs:
  - `./gradlew run` (computation-graph walkthrough)
  - `java -cp build/classes/java/main com.vaibhavkhare.ml.autograd.demo.NeuronValueExample`
  - `java -cp build/classes/java/main com.vaibhavkhare.ml.autograd.demo.MicrogradReplicaDemo`
  - `java -cp build/classes/java/main com.vaibhavkhare.ml.autograd.demo.AdvancedClassifierDemo`

With the pictures and these steps, you’ll “see” how backprop works. Once it clicks for a single neuron, the rest (layers and MLPs) are just more of the same.


