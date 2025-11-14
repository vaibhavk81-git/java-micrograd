package com.vaibhavkhare.ml.autograd;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Layer represents a collection of neurons that process the same inputs in parallel.
 * 
 * <p>A layer is a fundamental component in neural networks. It consists of multiple
 * neurons that all receive the same inputs but have different weights, allowing each
 * neuron to learn different features from the data.
 * 
 * <p>This class implements the {@link Module} interface, making it a composite component
 * that contains multiple {@link Neuron} modules.
 * 
 * <h2>Mathematical Operation</h2>
 * <p>For a layer with n neurons and m inputs:
 * <pre>
 *   Input: [x₁, x₂, ..., xₘ]
 *   
 *   Each neuron computes:
 *   y₁ = activation(w₁₁x₁ + w₁₂x₂ + ... + w₁ₘxₘ + b₁)
 *   y₂ = activation(w₂₁x₁ + w₂₂x₂ + ... + w₂ₘxₘ + b₂)
 *   ...
 *   yₙ = activation(wₙ₁x₁ + wₙ₂x₂ + ... + wₙₘxₘ + bₙ)
 *   
 *   Output: [y₁, y₂, ..., yₙ]
 * </pre>
 * 
 * <h2>Why Do We Need Layers?</h2>
 * <p>Layers enable neural networks to learn hierarchical representations:
 * <ol>
 *   <li><b>Parallel Processing</b>: Multiple neurons learn different features simultaneously</li>
 *   <li><b>Feature Extraction</b>: Each neuron specializes in detecting specific patterns</li>
 *   <li><b>Dimensionality Transformation</b>: Maps from input dimension to output dimension</li>
 *   <li><b>Representation Learning</b>: Automatically learns useful representations of data</li>
 * </ol>
 * 
 * <h2>Example: Image Edge Detection</h2>
 * <p>In an image processing network:
 * <ul>
 *   <li>Neuron 1 might learn to detect horizontal edges</li>
 *   <li>Neuron 2 might learn to detect vertical edges</li>
 *   <li>Neuron 3 might learn to detect diagonal edges</li>
 *   <li>All process the same image pixels but extract different features</li>
 * </ul>
 * 
 * <h2>Layer Architecture</h2>
 * <pre>
 *           Input Layer            Hidden Layer           Output Layer
 *          (3 features)            (4 neurons)            (2 neurons)
 *         
 *             x₁ ────┬─────────────→ n₁ ────┐
 *                    │              n₂ ────┼──────→ n₁ ──→ y₁
 *             x₂ ────┼─────────────→ n₃ ────┤
 *                    │              n₄ ────┘       n₂ ──→ y₂
 *             x₃ ────┘
 *         
 *         Each neuron receives ALL inputs
 *         Each neuron has its own weights
 *         Each neuron produces one output
 * </pre>
 * 
 * <h2>Example Usage</h2>
 * <pre>
 * // Create a layer: 3 inputs → 4 outputs
 * Layer layer = new Layer(3, 4);
 * 
 * // Prepare inputs
 * List&lt;Value&gt; inputs = Arrays.asList(
 *     new Value(1.0, "x1"),
 *     new Value(2.0, "x2"),
 *     new Value(3.0, "x3")
 * );
 * 
 * // Forward pass - all 4 neurons process inputs
 * List&lt;Value&gt; outputs = layer.forward(inputs);
 * // outputs.size() == 4 (one per neuron)
 * 
 * // Backward pass
 * Value loss = computeLoss(outputs);
 * loss.backward();
 * 
 * // Update all neurons' parameters
 * for (Value param : layer.parameters()) {
 *     param.setData(param.getData() - 0.01 * param.getGrad());
 * }
 * 
 * // Zero gradients for next iteration
 * layer.zeroGrad();
 * </pre>
 * 
 * @author Vaibhav Khare
 * @see Module
 * @see Neuron
 * @see MLP
 */
public class Layer implements Module {

    private final List<Neuron> neurons;
    private final int inputSize;
    private final int outputSize;

    /**
     * Creates a layer with default tanh activation for all neurons.
     * 
     * <p><b>Why Multiple Neurons?</b>
     * Having multiple neurons allows the layer to:
     * <ul>
     *   <li>Learn multiple features from the same input</li>
     *   <li>Capture different aspects of the data</li>
     *   <li>Provide richer representations for subsequent layers</li>
     * </ul>
     * 
     * <p><b>Dimensionality Transformation:</b>
     * This layer transforms data from nin-dimensional space to nout-dimensional space.
     * For example, Layer(784, 128) would be used in MNIST to reduce 784 pixels to 128 features.
     *
     * @param nin  number of input features (each neuron receives nin inputs)
     * @param nout number of output features (number of neurons in this layer)
     * @throws IllegalArgumentException if nin or nout is less than 1
     */
    public Layer(int nin, int nout) {
        this(nin, nout, ActivationType.TANH);
    }

    /**
     * Creates a layer with specified activation function for all neurons.
     * 
     * <p><b>When to Use Different Activations:</b>
     * <ul>
     *   <li><b>TANH</b>: Hidden layers, when you need zero-centered outputs</li>
     *   <li><b>RELU</b>: Hidden layers, generally faster training, prevents vanishing gradients</li>
     *   <li><b>LINEAR</b>: Output layer for regression tasks</li>
     * </ul>
     *
     * @param nin            number of input features
     * @param nout           number of output features
     * @param activationType type of activation function
     * @throws IllegalArgumentException if nin or nout is less than 1 or activation type is null
     */
    public Layer(int nin, int nout, ActivationType activationType) {
        if (nin < 1) {
            throw new IllegalArgumentException("Number of inputs must be at least 1, got: " + nin);
        }
        if (nout < 1) {
            throw new IllegalArgumentException("Number of outputs must be at least 1, got: " + nout);
        }
        if (activationType == null) {
            throw new IllegalArgumentException("Activation type cannot be null");
        }

        this.inputSize = nin;
        this.outputSize = nout;
        this.neurons = new ArrayList<>(nout);

        // Create nout neurons, each with nin inputs
        // Each neuron will have different random weights (breaking symmetry)
        for (int i = 0; i < nout; i++) {
            this.neurons.add(new Neuron(nin, activationType));
        }
    }

    /**
     * Creates a layer with custom seed for reproducible initialization.
     * Useful for testing and debugging.
     * 
     * <p><b>Reproducibility in Neural Networks:</b>
     * Setting a seed ensures:
     * <ul>
     *   <li>Same weight initialization every time</li>
     *   <li>Reproducible training results (for same data and hyperparameters)</li>
     *   <li>Easier debugging and testing</li>
     * </ul>
     *
     * @param nin  number of input features
     * @param nout number of output features
     * @param seed random seed for reproducible initialization
     */
    // Seed-based constructors have been removed in favor of Random-based APIs.

    /**
     * Creates a layer using a provided Random instance for all neuron initializations.
     * This allows threading a single RNG through all layers so the entire model
     * is initialized from a single random sequence.
     *
     * @param nin            number of input features
     * @param nout           number of output features
     * @param activationType activation function for this layer
     * @param rng            random generator used for initialization (must not be null)
     */
    public Layer(int nin, int nout, ActivationType activationType, Random rng) {
        if (nin < 1) {
            throw new IllegalArgumentException("Number of inputs must be at least 1, got: " + nin);
        }
        if (nout < 1) {
            throw new IllegalArgumentException("Number of outputs must be at least 1, got: " + nout);
        }
        if (activationType == null) {
            throw new IllegalArgumentException("Activation type cannot be null");
        }
        if (rng == null) {
            throw new IllegalArgumentException("Random generator cannot be null");
        }

        this.inputSize = nin;
        this.outputSize = nout;
        this.neurons = new ArrayList<>(nout);

        for (int i = 0; i < nout; i++) {
            this.neurons.add(new Neuron(nin, activationType, rng));
        }
    }

    /**
     * Convenience constructor using TANH activation and provided RNG.
     */
    public Layer(int nin, int nout, Random rng) {
        this(nin, nout, ActivationType.TANH, rng);
    }

    /**
     * Forward pass: compute outputs for all neurons in this layer.
     * 
     * <p><b>Parallel Processing:</b>
     * All neurons in this layer process the SAME inputs simultaneously (conceptually).
     * This is different from sequential processing in an MLP where layers process
     * one after another.
     * 
     * <p><b>Step-by-Step Process:</b>
     * <ol>
     *   <li>Each neuron receives the complete input vector</li>
     *   <li>Each neuron computes its weighted sum + bias + activation</li>
     *   <li>All neuron outputs are collected into an output list</li>
     *   <li>The output list becomes input for the next layer (if in an MLP)</li>
     * </ol>
     * 
     * <p><b>Example with 2 neurons, 3 inputs:</b>
     * <pre>
     * inputs = [x₁=1.0, x₂=2.0, x₃=3.0]
     * 
     * Neuron 1: w₁₁=0.5, w₁₂=0.3, w₁₃=-0.2, b₁=0.1
     *   y₁ = tanh(0.5*1.0 + 0.3*2.0 + (-0.2)*3.0 + 0.1) = tanh(0.5) ≈ 0.46
     * 
     * Neuron 2: w₂₁=-0.1, w₂₂=0.7, w₂₃=0.4, b₂=-0.3
     *   y₂ = tanh(-0.1*1.0 + 0.7*2.0 + 0.4*3.0 + (-0.3)) = tanh(2.1) ≈ 0.97
     * 
     * outputs = [y₁=0.46, y₂=0.97]
     * </pre>
     * 
     * <p><b>Computational Graph:</b>
     * Each neuron creates its own subgraph. All subgraphs share the same input Values
     * but have independent weight Values.
     *
     * @param inputs list of input Values (size must match nin)
     * @return list of output Values (size will be nout)
     * @throws IllegalArgumentException if input size doesn't match nin
     */
    public List<Value> forward(List<Value> inputs) {
        if (inputs.size() != inputSize) {
            throw new IllegalArgumentException(
                String.format("Expected %d inputs, got %d", inputSize, inputs.size()));
        }

        // Each neuron independently processes the same inputs
        // Using simple loop for better performance in forward pass
        List<Value> outputs = new ArrayList<>(outputSize);
        for (Neuron neuron : neurons) {
            outputs.add(neuron.forward(inputs));
        }
        return outputs;
    }

    /**
     * Returns all learnable parameters from all neurons in this layer.
     * 
     * <p><b>Parameter Organization:</b>
     * Parameters are organized by neuron:
     * <pre>
     * [neuron1_weight1, neuron1_weight2, ..., neuron1_bias,
     *  neuron2_weight1, neuron2_weight2, ..., neuron2_bias,
     *  ...]
     * </pre>
     * 
     * <p><b>Why This Matters:</b>
     * During training, we need to:
     * <ol>
     *   <li>Access all parameters to compute total number of parameters</li>
     *   <li>Update all parameters during gradient descent</li>
     *   <li>Zero all gradients before next backward pass</li>
     *   <li>Save/load all parameters for model persistence</li>
     * </ol>
     * 
     * <p><b>Example Usage in Training:</b>
     * <pre>
     * // Update all parameters in this layer
     * for (Value param : layer.parameters()) {
     *     param.setData(param.getData() - learningRate * param.getGrad());
     * }
     * </pre>
     *
     * @return unmodifiable list containing all parameters from all neurons
     */
    @Override
    public List<Value> parameters() {
        // Collect parameters from all neurons
        List<Value> allParams = new ArrayList<>();
        for (Neuron neuron : neurons) {
            allParams.addAll(neuron.parameters());
        }
        return Collections.unmodifiableList(allParams);
    }

    /**
     * Returns the number of input features this layer expects.
     *
     * @return number of inputs
     */
    public int getInputSize() {
        return inputSize;
    }

    /**
     * Returns the number of output features this layer produces.
     * This is also the number of neurons in the layer.
     *
     * @return number of outputs (number of neurons)
     */
    public int getOutputSize() {
        return outputSize;
    }

    /**
     * Returns all neurons in this layer.
     *
     * @return unmodifiable list of neurons
     */
    public List<Neuron> getNeurons() {
        return Collections.unmodifiableList(neurons);
    }

    @Override
    public String toString() {
        String activation = neurons.isEmpty() ? "none" : 
            neurons.get(0).getActivationType().getDisplayName();
        return String.format("Layer(nin=%d, nout=%d, activation=%s)", 
            inputSize, outputSize, activation);
    }
}

