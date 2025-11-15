# java-micrograd

A concise, readable Java implementation of Andrej Karpathy's micrograd ideas: scalar autograd + tiny neural nets (Neuron → Layer → MLP) built directly on a dynamic computation graph.

Inspired by [Andrej Karpathy's micrograd](https://github.com/karpathy/micrograd) - Vibe coded my way to understand NN fundamentals.

- Core: `Value` nodes build a graph at runtime and backprop through it
- NN building blocks: `Neuron`, `Layer`, `MLP`
- Demos: exact micrograd replica and an enhanced classifier
- Visualizations: computation graphs and decision boundaries

**Project goal**: Learn-by-building. Show the full path from scalars to a working MLP with clear code.

## Quick Start

Prerequisites
- JDK 21+

Build and Test
```bash
./gradlew build
./gradlew test
```

Run Demos
```bash
# Basic computation graph demo (Application)
./gradlew run

# Exact micrograd-style classifier (console output + optional boundary image)
java -cp build/classes/java/main com.vaibhavkhare.ml.autograd.demo.MicrogradReplicaDemo

# Enhanced classifier with extras (BCE, gradient clipping, circles dataset)
java -cp build/classes/java/main com.vaibhavkhare.ml.autograd.demo.AdvancedClassifierDemo

# Neuron/Value walkthrough with computation graph images
java -cp build/classes/java/main com.vaibhavkhare.ml.autograd.demo.NeuronValueExample
```

Outputs
- Computation graphs (PNG) if Graphviz is available (e.g., `neuron_forward_pass.png`)
- Decision boundary images from demos (see the Demos & Visualization guide)

## Features
- Autograd: add, sub, mul, div, pow, neg; tanh/relu/sigmoid/exp/log ops
- Backprop: reverse topological traversal with gradient accumulation
- Neural nets: dense `Neuron`, `Layer`, `MLP` with configurable activations
- Visualization: computation graph rendering via Graphviz

## Project Layout
```
src/main/java/com/vaibhavkhare/ml/autograd/
  ActivationType.java  GraphVisualizer.java  Layer.java  MLP.java
  Module.java          Neuron.java           Value.java
  demo/
    Application.java             # Graph basics + neuron demo
    MicrogradReplicaDemo.java    # Exact micrograd-style classifier
    AdvancedClassifierDemo.java  # Enhanced classifier (extras)
    NeuronValueExample.java      # Step-by-step Value/Neuron walkthrough + images
```

## Visualization (Optional)
- Images render even without native Graphviz via a JS fallback (you may see console warnings).
- Optional: Install Graphviz for faster/native rendering and fewer warnings.
  - macOS: `brew install graphviz`

## Attribution
- [Andrej Karpathy's micrograd](https://github.com/karpathy/micrograd)

## My thoughts
- [From Micrograd to Java: Why I Spent a Weekend Reimplementing Neural Nets from Scratch](https://medium.com/@vaibhavk81/from-micrograd-to-java-why-i-spent-a-weekend-reimplementing-neural-nets-from-scratch-b084a8e0d4b0)

