package com.vaibhavkhare.ml.autograd;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Neuron represents a single artificial neuron in a neural network.
 * 
 * <p>A neuron is the fundamental building block of neural networks. It performs a weighted
 * sum of its inputs, adds a bias term, and then applies an activation function.
 * 
 * <p>This class implements the {@link Module} interface, making it a leaf component
 * in the neural network hierarchy with its own learnable parameters.
 * 
 * <h2>Mathematical Operation</h2>
 * <p>For inputs x₁, x₂, ..., xₙ, a neuron computes:
 * <pre>
 *   z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b    (weighted sum + bias)
 *   output = activation(z)               (apply activation function)
 * </pre>
 * 
 * <p>Where:
 * <ul>
 *   <li>w₁, w₂, ..., wₙ are the <b>weights</b> (learnable parameters)</li>
 *   <li>b is the <b>bias</b> (learnable parameter)</li>
 *   <li>activation is a non-linear function (tanh, ReLU, etc.)</li>
 * </ul>
 * 
 * <h2>Why Do We Need Neurons?</h2>
 * <p>Neurons enable neural networks to learn complex, non-linear patterns:
 * <ol>
 *   <li><b>Weighted Sum</b>: Combines inputs with learned importance (weights)</li>
 *   <li><b>Bias</b>: Allows the neuron to shift its activation threshold</li>
 *   <li><b>Activation</b>: Introduces non-linearity, enabling complex function approximation</li>
 * </ol>
 * 
 * <h2>Training Process</h2>
 * <p>During training:
 * <ol>
 *   <li><b>Forward pass</b>: Compute output from inputs</li>
 *   <li><b>Backward pass</b>: Compute gradients via backpropagation</li>
 *   <li><b>Update</b>: Adjust weights and bias using gradients (gradient descent)</li>
 * </ol>
 * 
 * <h2>Example Usage</h2>
 * <pre>
 * // Create a neuron with 3 inputs
 * Neuron neuron = new Neuron(3);
 * 
 * // Prepare inputs
 * List&lt;Value&gt; inputs = Arrays.asList(
 *     new Value(1.0, "x1"),
 *     new Value(2.0, "x2"),
 *     new Value(3.0, "x3")
 * );
 * 
 * // Forward pass
 * Value output = neuron.forward(inputs);
 * 
 * // Backward pass (compute gradients)
 * output.backward();
 * 
 * // Access parameters for optimization
 * List&lt;Value&gt; params = neuron.parameters();
 * for (Value param : params) {
 *     // Update: param.data -= learning_rate * param.grad
 *     param.setData(param.getData() - 0.01 * param.getGrad());
 * }
 * 
 * // Zero gradients for next iteration
 * neuron.zeroGrad();
 * </pre>
 * 
 * @author Vaibhav Khare
 * @see Module
 * @see Layer
 * @see MLP
 */
public class Neuron implements Module {

    private final List<Value> weights;
    private final Value bias;
    private final ActivationType activationType;

    /**
     * Creates a neuron with random initial weights and bias.
     * 
     * <p><b>Weight Initialization:</b>
     * Weights are initialized randomly in the range [-1, 1]. This random initialization
     * is crucial for neural networks because:
     * <ul>
     *   <li>If all weights start at the same value, all neurons learn the same features (symmetry problem)</li>
     *   <li>Random initialization breaks symmetry, allowing neurons to learn different features</li>
     *   <li>The range [-1, 1] prevents gradients from exploding or vanishing initially</li>
     * </ul>
     * 
     * <p><b>Default Activation:</b>
     * Uses tanh activation by default. tanh(x) squashes values to [-1, 1] and is zero-centered,
     * which often helps with training stability.
     *
     * @param nin number of input features (number of weights to create)
     * @throws IllegalArgumentException if nin is less than 1
     */
    public Neuron(int nin) {
        this(nin, ActivationType.TANH);
    }

    /**
     * Creates a neuron with specified activation function.
     * 
     * <p><b>Activation Functions:</b>
     * <ul>
     *   <li><b>TANH</b>: Outputs in range [-1, 1], zero-centered, smooth gradient</li>
     *   <li><b>RELU</b>: Outputs in range [0, ∞), fast computation, can "die" if always negative</li>
     *   <li><b>LINEAR</b>: Linear neuron (no activation), used in output layers for regression</li>
     * </ul>
     *
     * @param nin            number of input features
     * @param activationType type of activation function
     * @throws IllegalArgumentException if nin is less than 1 or activation type is null
     */
    public Neuron(int nin, ActivationType activationType) {
        if (nin < 1) {
            throw new IllegalArgumentException("Number of inputs must be at least 1, got: " + nin);
        }
        
        if (activationType == null) {
            throw new IllegalArgumentException("Activation type cannot be null");
        }

        this.activationType = activationType;
        this.weights = new ArrayList<>(nin);
        
        // Initialize weights randomly in range [-1, 1]
        Random random = new Random();
        for (int i = 0; i < nin; i++) {
            double weight = (random.nextDouble() * 2.0) - 1.0; // Random in [-1, 1]
            this.weights.add(new Value(weight, "w" + i));
        }
        
        // Initialize bias randomly in range [-1, 1]
        double biasValue = (random.nextDouble() * 2.0) - 1.0;
        this.bias = new Value(biasValue, "b");
    }

    /**
     * Creates a neuron with custom seed for reproducible random initialization.
     * Useful for testing and debugging.
     *
     * @param nin  number of input features
     * @param seed random seed for reproducible initialization
     */
    // Seed-based constructors have been removed in favor of Random-based APIs.

    /**
     * Creates a neuron with a provided Random instance for reproducible initialization
     * across an entire network. This allows threading a single RNG through all layers
     * so that all parameters are sampled from one continuous random sequence.
     *
     * @param nin            number of input features
     * @param activationType type of activation function
     * @param rng            random generator used for initialization (must not be null)
     */
    public Neuron(int nin, ActivationType activationType, Random rng) {
        if (nin < 1) {
            throw new IllegalArgumentException("Number of inputs must be at least 1, got: " + nin);
        }
        if (activationType == null) {
            throw new IllegalArgumentException("Activation type cannot be null");
        }
        if (rng == null) {
            throw new IllegalArgumentException("Random generator cannot be null");
        }

        this.activationType = activationType;
        this.weights = new ArrayList<>(nin);

        for (int i = 0; i < nin; i++) {
            double weight = (rng.nextDouble() * 2.0) - 1.0;
            this.weights.add(new Value(weight, "w" + i));
        }

        double biasValue = (rng.nextDouble() * 2.0) - 1.0;
        this.bias = new Value(biasValue, "b");
    }

    /**
     * Convenience constructor using TANH activation with provided RNG.
     */
    public Neuron(int nin, Random rng) {
        this(nin, ActivationType.TANH, rng);
    }

    /**
     * Forward pass: compute the neuron's output for given inputs.
     * 
     * <p><b>Step-by-Step Process:</b>
     * <ol>
     *   <li><b>Weighted Sum</b>: Multiply each input by its weight: w₁x₁ + w₂x₂ + ... + wₙxₙ</li>
     *   <li><b>Add Bias</b>: Add the bias term: sum + b</li>
     *   <li><b>Activation</b>: Apply non-linear activation function (if enabled)</li>
     * </ol>
     * 
     * <p><b>Why Each Step Matters:</b>
     * <ul>
     *   <li><b>Weighted Sum</b>: Learns which inputs are important (high weight = high importance)</li>
     *   <li><b>Bias</b>: Allows the neuron to activate even when inputs are zero</li>
     *   <li><b>Activation</b>: Enables learning non-linear patterns (essential for deep learning)</li>
     * </ul>
     * 
     * <p><b>Example:</b>
     * If inputs = [1.0, 2.0], weights = [0.5, -0.3], bias = 0.8:
     * <pre>
     *   weighted_sum = 0.5*1.0 + (-0.3)*2.0 = 0.5 - 0.6 = -0.1
     *   with_bias = -0.1 + 0.8 = 0.7
     *   output = tanh(0.7) ≈ 0.604
     * </pre>
     *
     * @param inputs list of input Values (must match the number of weights)
     * @return the neuron's output as a Value (with gradient tracking enabled)
     * @throws IllegalArgumentException if input size doesn't match number of weights
     */
    public Value forward(List<Value> inputs) {
        if (inputs.size() != weights.size()) {
            throw new IllegalArgumentException(
                String.format("Expected %d inputs, got %d", weights.size(), inputs.size()));
        }

        // Step 1: Compute weighted sum: w₁x₁ + w₂x₂ + ... + wₙxₙ
        // Start with the bias value
        Value activation = bias;
        
        // Add each weighted input
        for (int i = 0; i < weights.size(); i++) {
            // w[i] * x[i]
            Value weightedInput = weights.get(i).mul(inputs.get(i));
            // Accumulate: sum = sum + w[i] * x[i]
            activation = activation.add(weightedInput);
        }
        
        // Step 2: Apply activation function (if not LINEAR)
        // This introduces non-linearity, which is essential for learning complex patterns
        if (activationType != ActivationType.LINEAR) {
            activation = applyActivation(activation);
        }
        
        return activation;
    }

    /**
     * Applies the configured activation function to the input value.
     * 
     * <p><b>Activation Functions Explained:</b>
     * <ul>
     *   <li><b>TANH</b>: 
     *       <ul>
     *         <li>Range: [-1, 1]</li>
     *         <li>Zero-centered (helps with gradient flow)</li>
     *         <li>Smooth, differentiable everywhere</li>
     *         <li>Can saturate (gradient → 0 for large |x|)</li>
     *       </ul>
     *   </li>
     *   <li><b>RELU</b>:
     *       <ul>
     *         <li>Range: [0, ∞)</li>
     *         <li>Fast to compute</li>
     *         <li>Doesn't saturate for positive values</li>
     *         <li>Can "die" if always negative (gradient = 0)</li>
     *       </ul>
     *   </li>
     *   <li><b>LINEAR</b>:
     *       <ul>
     *         <li>Range: (-∞, ∞)</li>
     *         <li>No transformation</li>
     *         <li>Used for regression output layers</li>
     *       </ul>
     *   </li>
     * </ul>
     *
     * @param value the pre-activation value
     * @return the activated value
     */
    private Value applyActivation(Value value) {
        return switch (activationType) {
            case TANH -> value.tanh();
            case RELU -> value.relu();
            case LINEAR -> value; // No activation (linear)
        };
    }

    /**
     * Returns all learnable parameters of this neuron (weights and bias).
     * 
     * <p><b>Why We Need This:</b>
     * During training, we need to:
     * <ol>
     *   <li>Access gradients: {@code param.getGrad()}</li>
     *   <li>Update parameters: {@code param.setData(param.getData() - learningRate * param.getGrad())}</li>
     *   <li>Zero gradients before next iteration: {@code param.setGrad(0.0)}</li>
     * </ol>
     * 
     * <p>This method collects all parameters in one list for easy iteration during optimization.
     * 
     * <p><b>Training Example:</b>
     * <pre>
     * // Forward and backward pass
     * Value output = neuron.forward(inputs);
     * output.backward();
     * 
     * // Gradient descent update
     * double learningRate = 0.01;
     * for (Value param : neuron.parameters()) {
     *     double newValue = param.getData() - learningRate * param.getGrad();
     *     param.setData(newValue);
     * }
     * </pre>
     *
     * @return unmodifiable list containing all weights followed by the bias
     */
    @Override
    public List<Value> parameters() {
        List<Value> params = new ArrayList<>(weights.size() + 1);
        params.addAll(weights);
        params.add(bias);
        return Collections.unmodifiableList(params);
    }

    /**
     * Returns the number of input features this neuron expects.
     *
     * @return number of inputs (number of weights)
     */
    public int getInputSize() {
        return weights.size();
    }

    /**
     * Returns the weights of this neuron.
     *
     * @return unmodifiable list of weight Values
     */
    public List<Value> getWeights() {
        return Collections.unmodifiableList(weights);
    }

    /**
     * Returns the bias of this neuron.
     *
     * @return bias Value
     */
    public Value getBias() {
        return bias;
    }

    /**
     * Returns the type of activation function used.
     *
     * @return activation type
     */
    public ActivationType getActivationType() {
        return activationType;
    }

    @Override
    public String toString() {
        return String.format("Neuron(nin=%d, activation=%s)", 
            weights.size(), 
            activationType.getDisplayName());
    }
}

