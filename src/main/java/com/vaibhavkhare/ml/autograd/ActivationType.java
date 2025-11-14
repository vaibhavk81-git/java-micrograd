package com.vaibhavkhare.ml.autograd;

/**
 * Activation functions supported by neurons in the neural network.
 * 
 * <p>Activation functions introduce non-linearity into neural networks, enabling them
 * to learn complex patterns. Without non-linear activations, a multi-layer network
 * would be equivalent to a single-layer network.
 * 
 * <h2>Choosing an Activation Function</h2>
 * <ul>
 *   <li><b>{@link #RELU}</b>: Default choice for hidden layers
 *       <ul>
 *         <li>Fast computation</li>
 *         <li>Prevents vanishing gradients</li>
 *         <li>Can suffer from "dying ReLU" problem</li>
 *       </ul>
 *   </li>
 *   <li><b>{@link #TANH}</b>: Good for hidden layers when zero-centered outputs are desired
 *       <ul>
 *         <li>Zero-centered (helps with gradient flow)</li>
 *         <li>Smooth, differentiable everywhere</li>
 *         <li>Can saturate (gradient → 0 for large |x|)</li>
 *       </ul>
 *   </li>
 *   <li><b>{@link #LINEAR}</b>: For output layers in regression tasks
 *       <ul>
 *         <li>No transformation, direct pass-through</li>
 *         <li>Allows unbounded outputs</li>
 *         <li>Used when predicting continuous values</li>
 *       </ul>
 *   </li>
 * </ul>
 * 
 * <h2>Example Usage</h2>
 * <pre>
 * // Create neuron with ReLU activation
 * Neuron neuron = new Neuron(3, ActivationType.RELU);
 * 
 * // Create layer with tanh activation
 * Layer layer = new Layer(10, 5, ActivationType.TANH);
 * 
 * // Create MLP with ReLU in hidden layers
 * MLP mlp = new MLP(Arrays.asList(784, 128, 10), ActivationType.RELU);
 * </pre>
 * 
 * @author Vaibhav Khare
 */
public enum ActivationType {
    
    /**
     * Hyperbolic Tangent activation function.
     * 
     * <p><b>Formula:</b> tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
     * 
     * <p><b>Range:</b> (-1, 1)
     * 
     * <p><b>Properties:</b>
     * <ul>
     *   <li>Zero-centered: outputs have mean close to 0</li>
     *   <li>Smooth gradient: derivative is continuous</li>
     *   <li>Saturates: gradient approaches 0 for large |x|</li>
     * </ul>
     * 
     * <p><b>Derivative:</b> tanh'(x) = 1 - tanh²(x)
     * 
     * <p><b>When to Use:</b>
     * <ul>
     *   <li>Hidden layers in smaller networks</li>
     *   <li>Recurrent neural networks (RNNs)</li>
     *   <li>When you need zero-centered activations</li>
     * </ul>
     */
    TANH("tanh"),
    
    /**
     * Rectified Linear Unit activation function.
     * 
     * <p><b>Formula:</b> ReLU(x) = max(0, x)
     * 
     * <p><b>Range:</b> [0, ∞)
     * 
     * <p><b>Properties:</b>
     * <ul>
     *   <li>Computationally efficient: simple max operation</li>
     *   <li>Non-saturating for positive values: helps prevent vanishing gradients</li>
     *   <li>Sparse activation: only ~50% of neurons activated</li>
     *   <li>Can "die": neurons can get stuck outputting 0</li>
     * </ul>
     * 
     * <p><b>Derivative:</b> ReLU'(x) = 1 if x > 0, else 0
     * 
     * <p><b>When to Use:</b>
     * <ul>
     *   <li><b>Default choice</b> for most hidden layers</li>
     *   <li>Deep neural networks</li>
     *   <li>Convolutional neural networks (CNNs)</li>
     * </ul>
     * 
     * <p><b>Variants:</b> Leaky ReLU, Parametric ReLU, ELU (not yet implemented)
     */
    RELU("relu"),
    
    /**
     * Linear (identity) activation function.
     * 
     * <p><b>Formula:</b> f(x) = x
     * 
     * <p><b>Range:</b> (-∞, ∞)
     * 
     * <p><b>Properties:</b>
     * <ul>
     *   <li>No transformation: output equals input</li>
     *   <li>Unbounded: can output any real number</li>
     *   <li>No non-linearity: doesn't add learning capacity</li>
     * </ul>
     * 
     * <p><b>Derivative:</b> f'(x) = 1
     * 
     * <p><b>When to Use:</b>
     * <ul>
     *   <li><b>Output layer</b> for regression tasks</li>
     *   <li>When predicting continuous values (prices, temperatures, etc.)</li>
     *   <li>Never use in hidden layers (loses non-linearity)</li>
     * </ul>
     * 
     * <p><b>Example:</b> Predicting house prices, stock values, etc.
     */
    LINEAR("linear");
    
    private final String displayName;
    
    /**
     * Creates an activation type with the specified display name.
     *
     * @param displayName the display name for this activation type
     */
    ActivationType(String displayName) {
        this.displayName = displayName;
    }
    
    /**
     * Returns the display name of this activation type.
     *
     * @return the display name (e.g., "tanh", "relu", "linear")
     */
    public String getDisplayName() {
        return displayName;
    }
    
    /**
     * Parses an activation type from a string (case-insensitive).
     * Provided for backwards compatibility and configuration file parsing.
     * 
     * <p><b>Supported Values:</b>
     * <ul>
     *   <li>"tanh" → {@link #TANH}</li>
     *   <li>"relu" → {@link #RELU}</li>
     *   <li>"linear" or "none" → {@link #LINEAR}</li>
     * </ul>
     * 
     * <p><b>Example:</b>
     * <pre>
     * ActivationType type = ActivationType.fromString("relu");  // Returns RELU
     * </pre>
     *
     * @param name the string representation of the activation type
     * @return the corresponding ActivationType enum value
     * @throws IllegalArgumentException if the name is not recognized
     */
    public static ActivationType fromString(String name) {
        if (name == null || name.trim().isEmpty()) {
            throw new IllegalArgumentException("Activation type name cannot be null or empty");
        }
        
        return switch (name.toLowerCase().trim()) {
            case "tanh" -> TANH;
            case "relu" -> RELU;
            case "linear", "none" -> LINEAR;
            default -> throw new IllegalArgumentException(
                String.format("Unknown activation type: '%s'. Supported values: tanh, relu, linear, none", name));
        };
    }
    
    @Override
    public String toString() {
        return displayName;
    }
}

