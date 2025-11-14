package com.vaibhavkhare.ml.autograd;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * MLP (Multi-Layer Perceptron) represents a complete feedforward neural network.
 * 
 * <p>An MLP is a stack of layers where the output of one layer becomes the input
 * to the next layer. This sequential processing allows the network to learn
 * hierarchical representations of increasing complexity.
 * 
 * <p>This class implements the {@link Module} interface, making it a composite component
 * that contains multiple {@link Layer} modules, which in turn contain {@link Neuron} modules.
 * 
 * <h2>Architecture</h2>
 * <pre>
 *   Input → Hidden Layer 1 → Hidden Layer 2 → ... → Output Layer → Output
 *   
 *   Example: MLP([3, 4, 4, 1])
 *   
 *   Input(3)  Hidden1(4)  Hidden2(4)  Output(1)
 *      x₁ ──┐              
 *      x₂ ──┼──→ [4 neurons] ──→ [4 neurons] ──→ [1 neuron] ──→ y
 *      x₃ ──┘
 * </pre>
 * 
 * <h2>Why Multi-Layer?</h2>
 * <p>Multiple layers enable the network to learn complex, non-linear functions:
 * <ol>
 *   <li><b>Layer 1</b>: Learns simple features (edges, colors)</li>
 *   <li><b>Layer 2</b>: Combines simple features into complex features (shapes, textures)</li>
 *   <li><b>Layer 3</b>: Combines complex features into abstract concepts (objects, patterns)</li>
 *   <li><b>Output</b>: Makes final prediction based on learned features</li>
 * </ol>
 * 
 * <h2>Universal Approximation Theorem</h2>
 * <p>A neural network with:
 * <ul>
 *   <li>At least one hidden layer</li>
 *   <li>Non-linear activation functions</li>
 *   <li>Sufficient neurons</li>
 * </ul>
 * can approximate any continuous function to arbitrary accuracy!
 * 
 * <h2>Real-World Example: Digit Recognition</h2>
 * <pre>
 * MLP([784, 128, 64, 10]) for MNIST:
 * 
 * Input Layer:    784 neurons (28x28 pixel image)
 *                  ↓
 * Hidden Layer 1: 128 neurons (learn edges, strokes)
 *                  ↓
 * Hidden Layer 2: 64 neurons (learn digit parts)
 *                  ↓
 * Output Layer:   10 neurons (one per digit 0-9)
 * </pre>
 * 
 * <h2>Information Flow</h2>
 * <pre>
 * Forward Pass:
 *   Input → Layer1 → Layer2 → ... → LayerN → Output
 *   (Data flows forward through the network)
 * 
 * Backward Pass:
 *   Output ← Layer1 ← Layer2 ← ... ← LayerN ← Loss
 *   (Gradients flow backward through the network)
 * </pre>
 * 
 * <h2>Example Usage</h2>
 * <pre>
 * // Create a network: 3 inputs → 4 hidden → 4 hidden → 1 output
 * MLP mlp = new MLP(Arrays.asList(3, 4, 4, 1));
 * 
 * // Forward pass
 * List&lt;Value&gt; inputs = Arrays.asList(
 *     new Value(1.0, "x1"),
 *     new Value(2.0, "x2"),
 *     new Value(3.0, "x3")
 * );
 * Value output = mlp.forward(inputs);
 * 
 * // Compute loss
 * Value target = new Value(0.5, "target");
 * Value loss = output.sub(target).pow(2);
 * 
 * // Backward pass
 * loss.backward();
 * 
 * // Update all parameters
 * double learningRate = 0.01;
 * for (Value param : mlp.parameters()) {
 *     param.setData(param.getData() - learningRate * param.getGrad());
 * }
 * 
 * // Zero gradients for next iteration
 * mlp.zeroGrad();
 * </pre>
 * 
 * @author Vaibhav Khare
 * @see Module
 * @see Layer
 * @see Neuron
 */
public class MLP implements Module {

    private final List<Layer> layers;
    private final List<Integer> layerSizes;

    /**
     * Creates an MLP with the specified layer sizes and default tanh activation.
     * 
     * <p><b>Layer Sizes Explanation:</b>
     * The list defines the architecture of the network:
     * <ul>
     *   <li>First element: input dimension</li>
     *   <li>Middle elements: hidden layer dimensions</li>
     *   <li>Last element: output dimension</li>
     * </ul>
     * 
     * <p><b>Example Architectures:</b>
     * <pre>
     * [2, 1]       - Simple perceptron: 2 inputs → 1 output
     * [3, 4, 1]    - One hidden layer: 3 → 4 → 1
     * [784, 128, 10] - MNIST classifier: 784 → 128 → 10
     * [10, 5, 5, 2]  - Two hidden layers: 10 → 5 → 5 → 2
     * </pre>
     * 
     * <p><b>Choosing Architecture:</b>
     * <ul>
     *   <li><b>Input size</b>: Determined by your data (e.g., number of features)</li>
     *   <li><b>Hidden layers</b>: More layers = more complex patterns, but harder to train</li>
     *   <li><b>Hidden size</b>: More neurons = more capacity, but slower and risk overfitting</li>
     *   <li><b>Output size</b>: Determined by your task (1 for regression, n for n-class classification)</li>
     * </ul>
     * 
     * <p><b>Rule of Thumb:</b>
     * Start simple (1-2 hidden layers), then add complexity if needed.
     *
     * @param layerSizes list of layer sizes [input_size, hidden1_size, ..., output_size]
     * @throws IllegalArgumentException if layerSizes has less than 2 elements
     */
    public MLP(List<Integer> layerSizes) {
        this(layerSizes, ActivationType.TANH);
    }

    /**
     * Creates an MLP with custom activation function for hidden layers.
     * 
     * <p><b>Activation Strategy:</b>
     * <ul>
     *   <li>All hidden layers use the specified activation (TANH or RELU)</li>
     *   <li>Output layer uses linear activation (no activation function)</li>
     *   <li>This is standard for regression tasks</li>
     * </ul>
     * 
     * <p><b>For Classification:</b>
     * You would typically add softmax or sigmoid after the output layer
     * during loss computation, not as part of the network itself.
     * 
     * <p><b>Activation Choice Guide:</b>
     * <ul>
     *   <li><b>RELU</b>: Default choice for most problems, fast, prevents vanishing gradients</li>
     *   <li><b>TANH</b>: When you need zero-centered outputs, good for RNNs</li>
     * </ul>
     *
     * @param layerSizes      list of layer sizes
     * @param activationType  type of activation for hidden layers
     * @throws IllegalArgumentException if layerSizes has less than 2 elements or activation type is null
     */
    public MLP(List<Integer> layerSizes, ActivationType activationType) {
        if (layerSizes == null || layerSizes.size() < 2) {
            throw new IllegalArgumentException(
                "MLP requires at least 2 layer sizes (input and output), got: " + 
                (layerSizes == null ? "null" : layerSizes.size()));
        }

        if (activationType == null) {
            throw new IllegalArgumentException("Activation type cannot be null");
        }

        // Validate all sizes are positive
        for (int i = 0; i < layerSizes.size(); i++) {
            if (layerSizes.get(i) < 1) {
                throw new IllegalArgumentException(
                    String.format("Layer size at index %d must be positive, got: %d", 
                        i, layerSizes.get(i)));
            }
        }

        this.layerSizes = new ArrayList<>(layerSizes);
        this.layers = new ArrayList<>();

        // Create layers: each layer connects consecutive sizes
        // Example: [3, 4, 2] creates layers: (3→4) and (4→2)
        for (int i = 0; i < layerSizes.size() - 1; i++) {
            int nin = layerSizes.get(i);
            int nout = layerSizes.get(i + 1);
            
            // Last layer (output layer) typically has no activation for regression
            // Hidden layers use the specified activation
            boolean isOutputLayer = (i == layerSizes.size() - 2);
            ActivationType layerActivationType = isOutputLayer ? ActivationType.LINEAR : activationType;
            
            layers.add(new Layer(nin, nout, layerActivationType));
        }
    }

    /**
     * Creates an MLP with custom seed for reproducible initialization.
     * 
     * <p><b>Reproducibility Benefits:</b>
     * <ul>
     *   <li>Debugging: Same initialization helps isolate bugs</li>
     *   <li>Testing: Ensures consistent test results</li>
     *   <li>Comparison: Fair comparison of different hyperparameters</li>
     *   <li>Research: Reproducible experiments for papers</li>
     * </ul>
     *
     * @param layerSizes list of layer sizes
     * @param seed       random seed for reproducible initialization
     */
    // Seed-based constructors have been removed in favor of Random-based APIs.

    /**
     * Creates an MLP using a provided Random instance for all parameter initializations.
     * This allows threading a single RNG through the entire network so that all
     * parameters are sampled from one continuous random sequence.
     *
     * @param layerSizes     list of layer sizes
     * @param activationType activation for hidden layers (output is linear)
     * @param rng            random generator used for initialization (must not be null)
     */
    public MLP(List<Integer> layerSizes, ActivationType activationType, Random rng) {
        if (layerSizes == null || layerSizes.size() < 2) {
            throw new IllegalArgumentException(
                "MLP requires at least 2 layer sizes (input and output), got: " + 
                (layerSizes == null ? "null" : layerSizes.size()));
        }
        if (activationType == null) {
            throw new IllegalArgumentException("Activation type cannot be null");
        }
        if (rng == null) {
            throw new IllegalArgumentException("Random generator cannot be null");
        }

        for (int i = 0; i < layerSizes.size(); i++) {
            if (layerSizes.get(i) < 1) {
                throw new IllegalArgumentException(
                    String.format("Layer size at index %d must be positive, got: %d", 
                        i, layerSizes.get(i)));
            }
        }

        this.layerSizes = new ArrayList<>(layerSizes);
        this.layers = new ArrayList<>();

        for (int i = 0; i < layerSizes.size() - 1; i++) {
            int nin = layerSizes.get(i);
            int nout = layerSizes.get(i + 1);

            boolean isOutputLayer = (i == layerSizes.size() - 2);
            ActivationType layerActivationType = isOutputLayer ? ActivationType.LINEAR : activationType;

            layers.add(new Layer(nin, nout, layerActivationType, rng));
        }
    }

    /**
     * Convenience constructor using TANH activation with provided RNG.
     */
    public MLP(List<Integer> layerSizes, Random rng) {
        this(layerSizes, ActivationType.TANH, rng);
    }

    /**
     * Forward pass: propagate input through all layers sequentially.
     * Returns a single output Value.
     * 
     * <p><b>⚠️ Important:</b> This method requires the network to have exactly 1 output neuron.
     * For multi-output networks (e.g., classification), use {@link #forwardAll(List)} instead.
     * 
     * <p><b>Sequential Processing:</b>
     * Data flows through the network one layer at a time:
     * <pre>
     * input → layer1 → intermediate1 → layer2 → intermediate2 → ... → output
     * </pre>
     * 
     * <p><b>Typical Use Case:</b>
     * Regression tasks with single output, e.g., predicting house prices, temperatures, etc.
     * 
     * <p><b>Example:</b>
     * <pre>
     * MLP mlp = new MLP(Arrays.asList(3, 4, 1));  // 3 inputs, 4 hidden, 1 output
     * List&lt;Value&gt; inputs = Arrays.asList(new Value(1.0), new Value(2.0), new Value(3.0));
     * Value prediction = mlp.forward(inputs);  // Single prediction value
     * </pre>
     * 
     * <p><b>Computational Graph:</b>
     * Each layer creates its own subgraphs, which are all connected:
     * <pre>
     * Input Values → Layer1 (creates Values) → Layer2 (creates Values) → Output
     *                    ↓                           ↓
     *              All tracked in               All tracked in
     *           computational graph          computational graph
     * </pre>
     * 
     * <p><b>Gradient Flow:</b>
     * During backward pass, gradients flow back through all layers automatically
     * because all Value objects are connected in the computational graph.
     *
     * @param inputs list of input Values (size must match first layer's input size)
     * @return single output Value
     * @throws IllegalArgumentException if input size doesn't match network's input size
     * @throws IllegalStateException if network has multiple outputs (use forwardAll instead)
     */
    public Value forward(List<Value> inputs) {
        List<Value> outputs = forwardAll(inputs);
        
        if (outputs.size() != 1) {
            throw new IllegalStateException(
                String.format("forward() expects single output, but network produces %d outputs. " +
                             "Use forwardAll() for multi-output networks.", outputs.size()));
        }
        
        return outputs.get(0);
    }

    /**
     * Forward pass: propagate input through all layers and return all outputs.
     * 
     * <p><b>Use This For:</b>
     * <ul>
     *   <li>Classification tasks (multiple class scores)</li>
     *   <li>Multi-output regression (predicting multiple values)</li>
     *   <li>Any network with output layer size > 1</li>
     * </ul>
     * 
     * <p><b>Sequential Processing:</b>
     * Data flows through the network one layer at a time:
     * <pre>
     * input → layer1 → intermediate1 → layer2 → intermediate2 → ... → outputs
     * </pre>
     * 
     * <p><b>Example for Classification:</b>
     * <pre>
     * MLP mlp = new MLP(Arrays.asList(784, 128, 10));  // MNIST: 784 → 128 → 10 classes
     * List&lt;Value&gt; inputs = loadMNISTImage();
     * List&lt;Value&gt; logits = mlp.forwardAll(inputs);  // 10 class scores
     * 
     * // Find predicted class
     * int predictedClass = 0;
     * double maxScore = logits.get(0).getData();
     * for (int i = 1; i < logits.size(); i++) {
     *     if (logits.get(i).getData() > maxScore) {
     *         maxScore = logits.get(i).getData();
     *         predictedClass = i;
     *     }
     * }
     * </pre>
     * 
     * <p><b>Example for Multi-Output Regression:</b>
     * <pre>
     * MLP mlp = new MLP(Arrays.asList(3, 4, 2));  // Predict 2 values from 3 inputs
     * List&lt;Value&gt; inputs = Arrays.asList(new Value(1.0), new Value(2.0), new Value(3.0));
     * List&lt;Value&gt; predictions = mlp.forwardAll(inputs);  // [y1, y2]
     * 
     * Value y1 = predictions.get(0);
     * Value y2 = predictions.get(1);
     * </pre>
     *
     * @param inputs list of input Values (size must match first layer's input size)
     * @return list of output Values (size matches output layer size)
     * @throws IllegalArgumentException if input size doesn't match network's input size
     */
    public List<Value> forwardAll(List<Value> inputs) {
        if (inputs.size() != layerSizes.get(0)) {
            throw new IllegalArgumentException(
                String.format("Expected %d inputs, got %d", layerSizes.get(0), inputs.size()));
        }

        // Start with the input
        List<Value> current = inputs;
        
        // Pass through each layer sequentially
        for (Layer layer : layers) {
            current = layer.forward(current);
        }
        
        return current;
    }

    /**
     * Returns all learnable parameters from all layers.
     * 
     * <p><b>Parameter Count:</b>
     * For MLP([3, 4, 2]):
     * <ul>
     *   <li>Layer1: 3 inputs × 4 neurons = 12 weights, 4 biases = 16 params</li>
     *   <li>Layer2: 4 inputs × 2 neurons = 8 weights, 2 biases = 10 params</li>
     *   <li>Total: 26 parameters</li>
     * </ul>
     * 
     * <p><b>General Formula:</b>
     * For layer connecting n inputs to m outputs:
     * Parameters = (n × m) + m = m × (n + 1)
     * 
     * <p><b>Why This Matters:</b>
     * <ul>
     *   <li>More parameters = more capacity to learn</li>
     *   <li>But also: more data needed, slower training, risk of overfitting</li>
     *   <li>Model size = number of parameters × bytes per parameter</li>
     * </ul>
     * 
     * <p><b>Training Usage:</b>
     * <pre>
     * // Gradient descent on entire network
     * for (Value param : mlp.parameters()) {
     *     param.setData(param.getData() - learningRate * param.getGrad());
     * }
     * </pre>
     *
     * @return unmodifiable list containing all parameters from all layers
     */
    @Override
    public List<Value> parameters() {
        List<Value> allParams = new ArrayList<>();
        for (Layer layer : layers) {
            allParams.addAll(layer.parameters());
        }
        return Collections.unmodifiableList(allParams);
    }

    /**
     * Returns the architecture of this MLP.
     *
     * @return list of layer sizes [input_size, hidden1_size, ..., output_size]
     */
    public List<Integer> getLayerSizes() {
        return Collections.unmodifiableList(layerSizes);
    }

    /**
     * Returns all layers in this MLP.
     *
     * @return unmodifiable list of layers
     */
    public List<Layer> getLayers() {
        return Collections.unmodifiableList(layers);
    }

    /**
     * Returns the number of layers in this MLP.
     * This is the number of weight transformations, not including the input.
     *
     * @return number of layers
     */
    public int getNumLayers() {
        return layers.size();
    }

    /**
     * Returns the input dimension of this MLP.
     *
     * @return input dimension
     */
    public int getInputSize() {
        return layerSizes.get(0);
    }

    /**
     * Returns the output dimension of this MLP.
     *
     * @return output dimension
     */
    public int getOutputSize() {
        return layerSizes.get(layerSizes.size() - 1);
    }

    @Override
    public String toString() {
        return String.format("MLP(layers=%s, total_params=%d)", 
            layerSizes, parameters().size());
    }
}

