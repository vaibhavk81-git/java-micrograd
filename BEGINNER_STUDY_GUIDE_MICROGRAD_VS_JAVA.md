## Micrograd to Java: A Beginner’s Study Guide to Autograd and Neural Nets

This guide helps a new ML student learn core neural network concepts by comparing Andrej Karpathy’s Python micrograd with this Java implementation. You’ll see how the same ideas map across languages and how to read, run, and extend the code.

- Audience: Newcomers to ML and code who want to understand autograd and simple neural nets
- Prerequisites: Basic programming comfort; high-school calculus intuition (derivatives as “slopes”)
- Reference: micrograd repository [link]

### What you’ll learn
- What automatic differentiation (autograd) is, and why we use it
- How a scalar `Value` builds a dynamic computation graph and supports backpropagation
- How neurons, layers, and an MLP are constructed from `Value`s
- How training works: forward pass, loss, backward pass, gradient descent
- Where Python and Java differ and how to translate ideas between them

---

## Part 1. Autograd in one picture

When you compute with numbers that “remember” how they were created, you implicitly build a Directed Acyclic Graph (DAG). Each node stores:
- data: the forward value
- grad: the gradient of the final output w.r.t. this node
- parents: which nodes created this node
- backwardFn: how to push gradient to parents

Backward pass sets the final output’s grad to 1 and walks the DAG in reverse topological order, calling each node’s `backwardFn` to apply the chain rule and accumulate grads.

Concept diagram:
```
 x ----\            ----> loss
        \          /
         (+) --> z
        /          \
 y ----/            ----> other nodes ...
```

Key idea: gradients accumulate. If a node feeds multiple downstream paths, its `grad` is the sum from those paths.

---

## Part 2. The `Value` class: Python vs Java

In micrograd (Python), `Value` overloads operators like `+`, `*`, `**`, and tracks a graph on every operation. In Java, we call methods instead of using operator overloading, but the ideas are identical.

### Mapping at a glance
- Python `a + b` → Java `a.add(b)` or `a.add(3.0)`
- Python `a * b` → Java `a.mul(b)`
- Python `a ** 2` → Java `a.pow(2)`
- Python `a / b` → Java `a.div(b)`
- Python `-a` → Java `a.neg()`
- Python `relu()` → Java `relu()`; Python often also uses `tanh()` via `Value.tanh()` in examples; Java supports `tanh()`, `sigmoid()`, `exp()`, `log()`

What both store per node:
- data (`double` in Java)
- grad (`double`)
- parents (references to input nodes)
- op (string label, optional)
- backwardFn (captures the local derivative logic)

Why topological sort?
- We need to visit nodes in reverse creation order to ensure parents’ gradients are computed after their children. Java’s `backward()` builds a topo order and then runs `backwardFn` in reverse order.

---

## Part 3. From `Value` to Neuron, Layer, MLP

### Neuron
- A neuron computes a weighted sum plus bias, then an activation.
- Math: z = w·x + b; out = activation(z)
- In Python micrograd: `nn.Neuron.__call__` loops over inputs and weights, then applies `tanh` or `relu` depending on a flag.
- In Java: `Neuron.forward(inputs)` does the same with explicit `Value` ops. Activation is selected via `ActivationType` (TANH, RELU, or LINEAR).

Why activations?
- Without them, a stack of linear layers collapses to a single linear function. Non-linearity lets nets approximate complex functions.

### Layer
- A `Layer` holds multiple `Neuron`s that all receive the same inputs in parallel and produce a list of outputs.
- In both Python and Java, a layer is “neurons in parallel.”

### MLP (Multi-Layer Perceptron)
- An MLP chains layers: the output of one layer becomes the input to the next.
- Hidden layers use non-linear activations, the output layer is often linear for regression; classification adds a non-linearity in the loss (e.g., sigmoid + BCE).

---

## Part 4. Training loop (the three steps)

1) Forward pass: compute predictions from inputs
2) Loss: measure how wrong the predictions are
3) Backward pass: call `loss.backward()` to fill all grads
4) Update: for each parameter p, `p.data -= learning_rate * p.grad`
5) Zero grads: before the next iteration, zero gradients (they accumulate)

Python micrograd demos do this explicitly. Java demos do the same with `Module.parameters()` and `Module.zeroGrad()` for consistency and convenience.

Common beginner questions:
- Why do we zero grads? Because gradients accumulate by addition (chain rule across multiple paths and/or batches). Zeroing prevents “leaking” old gradients into the next step.
- Where does the chain rule happen? Inside each `backwardFn`: local derivative × upstream gradient; then add into each parent’s `grad`.

Label encoding tip:
- The SVM max‑margin (hinge) loss expects labels in {−1, +1}. If you train with hinge loss, ensure your dataset encodes classes as ±1 to get correct gradients.

---

## Part 5. Side-by-side: how concepts translate

Concept to code mapping:
- Dynamic graph: Every new `Value` created by an op points back to parents and stores a local `backwardFn`.
- Reverse-mode autodiff: Start from final scalar, set grad=1, walk parents backward.
- Parameters: Neuron weights and bias are `Value`s. `parameters()` collects them for optimization.
- Activations: `tanh`, `relu` (Java also provides `sigmoid`, `exp`, `log`).
- Architecture: Neuron → Layer (list of neurons) → MLP (list of layers).

Language differences to be aware of:
- Operator overloading: Python supports it; Java uses methods (`add`, `mul`, `pow`, etc.).
- Enums and interfaces: Java adds `ActivationType` enum and a `Module` interface with `parameters()` and `zeroGrad()` defaults, offering a clean, unified API.
- Seed control: The Java API offers RNG-based constructors for reproducibility in demos/tests.

---

## Part 6. A hands-on mini exercise (both versions)

Goal: Fit a tiny network so its output matches a target for a single input vector. This mirrors the demos and builds intuition.

Python (micrograd-like pseudocode):
```python
from micrograd.engine import Value
from micrograd import nn

model = nn.MLP(3, [4, 4, 1])
target = 0.7

for step in range(20):
    # forward
    pred = model([Value(1.0), Value(2.0), Value(3.0)])
    loss = (pred - Value(target))**2
    
    # backward
    model.zero_grad()
    loss.backward()
    
    # update
    for p in model.parameters():
        p.data -= 0.05 * p.grad
```

Java (this repository):
```java
MLP mlp = new MLP(java.util.Arrays.asList(3, 4, 4, 1));
double target = 0.7;
double lr = 0.05;

for (int step = 0; step < 20; step++) {
    var inputs = java.util.Arrays.asList(new Value(1.0, "x1"), new Value(2.0, "x2"), new Value(3.0, "x3"));
    Value pred = mlp.forward(inputs);
    Value loss = pred.sub(new Value(target, "t")).pow(2);

    mlp.zeroGrad();      // clear accumulated grads
    loss.backward();     // fill grads via chain rule
    for (Value p : mlp.parameters()) {
        p.setData(p.getData() - lr * p.getGrad());
    }
}
```

What to notice:
- The structure is identical: forward → loss → backward → update → zero.
- The only difference is syntax, not concepts.

---

## Part 7. Visualizing the graph

micrograd uses a small helper (often in a notebook) to render the graph with Graphviz. This repo includes a `GraphVisualizer` that saves PNGs. Run the Java demo to see clear, labeled graphs of both data and gradients. Seeing the graph helps build intuition about how gradients flow. Note: images render even without native Graphviz (via a JS fallback); installing Graphviz is optional but faster.

---

## Part 8. Putting it all together: step-by-step mental model

- Forward pass builds a graph of `Value`s.
- Loss is a single scalar `Value` you care about.
- `loss.backward()` walks the graph in reverse, filling `grad` in each upstream node.
- Parameters are just special `Value`s you update with gradient descent.
- Repeating this loop trains the model to reduce loss.

If you keep this simple picture in mind, you can reason through any shape of network.

---

## Part 9. Where the Java code extends micrograd

- `ActivationType` enum replaces boolean flags, making code more readable.
- Extra math/activations: `sigmoid()`, `exp()`, `log()` in addition to `tanh()` and `relu()`.
- `Module` interface provides `parameters()` and `zeroGrad()` for all components.
- RNG-friendly constructors enable reproducible demos/tests.
- Clearer separation of `forward` vs `forwardAll` in `MLP` for single-/multi-output cases.

These are pragmatic Java-isms that make the library pleasant to use while keeping the spirit of micrograd.

---

## Part 10. How to run and learn (Java)

1) Build
```bash
./gradlew build
```

2) Run the educational demo with graphs and explanations
```bash
./gradlew run
```
This prints a walkthrough and writes PNGs for graph visualization (see `example1_simple.png`, etc.).

3) Explore more demos (see README or DEMOS_AND_VISUALIZATION.md)
```bash
# Neuron/Value walkthrough with graph images
java -cp build/classes/java/main com.vaibhavkhare.ml.autograd.demo.NeuronValueExample

# Micrograd-style classifier (hinge loss + L2)
java -cp build/classes/java/main com.vaibhavkhare.ml.autograd.demo.MicrogradReplicaDemo

# Enhanced classifier (BCE option, clipping, circles)
java -cp build/classes/java/main com.vaibhavkhare.ml.autograd.demo.AdvancedClassifierDemo
```

Learning checklist:
- Watch printed values and gradients change through training.
- Open the generated graph PNGs and trace parents/children.
- Skim `Value.add`, `mul`, `pow`, `relu`, `tanh` and their `backwardFn`s to cement chain rule intuition.
- Step up from a single neuron → a layer → an MLP.

---

## Quick reference: concept → class

- Scalar with autograd: `Value`
- Neuron building block: `Neuron`
- Parallel neurons: `Layer`
- Stack of layers (network): `MLP`
- Parameter collection and grad reset: `Module`
- Activation configuration: `ActivationType`
- Graph visualization: `GraphVisualizer`

---

## FAQ for beginners

**Do I have to understand all the math first?** No. Start with the code and pictures. The chain rule pieces inside each `backwardFn` are small and readable.

**Why not compute gradients by hand?** For complex graphs, it’s tedious and error-prone. Autograd automates the chain rule and scales to large networks.

**Why does loss need to be a single scalar?** Reverse-mode autodiff (backprop) computes gradients of one scalar output w.r.t. many parameters efficiently, which is exactly what training needs.

**Is this like PyTorch?** Conceptually, yes—micrograd is a tiny educational ancestor. PyTorch layers/ops are far more feature-rich and optimized, but the mental model is very similar.

---

## Further reading and references

- Andrej Karpathy’s micrograd repository (Python): see README and demos [link]
- This repo’s `README.md` for commands and demos
- `COMPARISON_WITH_MICROGRAD.md` and `FEATURE_COMPARISON.txt` for deeper comparisons

---

Happy learning! Once you’re comfortable here, you’ll find frameworks like PyTorch feel much less mysterious.

[
link
]: https://github.com/karpathy/micrograd/tree/master


