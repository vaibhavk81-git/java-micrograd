# Demos & Visualization Guide

Concise guide to run demos and generate images for this project.

## Prerequisites
- JDK 21+

Optional (for faster/native image rendering)
- Graphviz: macOS `brew install graphviz`
  - Without Graphviz, images still render via a JS fallback (you may see console warnings).

## 1) Basic Demo (Application)
Shows computation graph basics and a simple neuron.
```bash
./gradlew run
```
Outputs:
- `example1_simple.png`
- `example2_complex.png`
- `example3_neuron.png`

## 2) Micrograd Replica (Exact Style)
Matches the spirit of micrograd’s demo (moons dataset, SVM margin loss) with stabilized training settings for Java RNGs.
```bash
./gradlew build
java -cp build/classes/java/main com.vaibhavkhare.ml.autograd.demo.MicrogradReplicaDemo
```
Console:
- Training progress, accuracy, configuration notes
Images:
- Decision boundary PNG (e.g., `micrograd_replica_decision_boundary.png`) — native Graphviz speeds up rendering but is not required
Settings used here:
- Architecture: [2, 16, 16, 1]
- Loss: SVM max-margin + L2 (alpha=1e-4)
- Optimizer: SGD, learning rate = 0.1, steps = 200
- Stability: gentle gradient clipping (±10)
These tweaks make accuracy reliably high (often 85–100%) across Java RNG sequences.

## 3) Advanced Classifier (Enhanced)
Adds BCE loss, gradient clipping, and a circles dataset section.
```bash
./gradlew build
java -cp build/classes/java/main com.vaibhavkhare.ml.autograd.demo.AdvancedClassifierDemo
```
Console:
- Per-epoch loss and accuracy
- Summary of configuration and results
Images:
- `moons_decision_boundary.png`
- `circles_decision_boundary.png`

## 4) Neuron & Value Walkthrough (With Images)
Step-by-step creation, forward, backward, parameter update, activations, and shared-input gradients.
```bash
./gradlew build
java -cp build/classes/java/main com.vaibhavkhare.ml.autograd.demo.NeuronValueExample
```
Images:
- `neuron_forward_pass.png` (forward graph)
- `neuron_backward_pass.png` (graph after backward; gradients visible)
- `neuron_linear_activation.png` vs `neuron_tanh_activation.png` (activation node comparison)
- `neuron_shared_inputs.png` (gradient accumulation with shared inputs)

## Tips
- If images don’t appear: install Graphviz and rerun, or check for fallback warnings in the console.
- Images are saved in the project root by default.
- Re-run commands after code changes: `./gradlew build` to ensure classes are fresh.

## Troubleshooting
- “dot command not found”: either install Graphviz (`brew install graphviz`) or rely on the JS fallback (images may still render, with warnings).
- GraalVM warnings in console: safe to ignore; they don’t affect results.

## Attribution
Inspired by Andrej Karpathy’s micrograd (`https://github.com/karpathy/micrograd`). This is a Java-first educational implementation.


