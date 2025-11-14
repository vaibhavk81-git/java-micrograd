package com.vaibhavkhare.ml.autograd;

import java.util.Collections;
import java.util.List;

/**
 * Module is the base interface for all neural network components.
 * 
 * <p>This interface is inspired by PyTorch's nn.Module and provides a common
 * contract for all components that have learnable parameters. It follows the
 * same design pattern as Andrej Karpathy's micrograd.
 * 
 * <h2>Purpose</h2>
 * <p>The Module interface serves several key purposes:
 * <ul>
 *   <li><b>Unified Interface</b>: All neural network components implement the same interface</li>
 *   <li><b>Parameter Management</b>: Standardized way to access and manage parameters</li>
 *   <li><b>Gradient Control</b>: Convenient methods for gradient manipulation</li>
 *   <li><b>Composability</b>: Enables building complex networks from simple components</li>
 * </ul>
 * 
 * <h2>Design Pattern</h2>
 * <p>This follows the <b>Composite Pattern</b>:
 * <pre>
 *              Module (interface)
 *                    ↑
 *      ┌─────────────┼─────────────┐
 *      │             │             │
 *   Neuron        Layer          MLP
 *   (leaf)      (composite)   (composite)
 *      │             │             │
 *      │         contains       contains
 *      │         Neurons        Layers
 *      │             │             │
 *   Individual    Multiple      Multiple
 *   parameters    neurons       layers
 * </pre>
 * 
 * <h2>Hierarchy</h2>
 * <ul>
 *   <li><b>Neuron</b>: Leaf component - has individual parameters (weights, bias)</li>
 *   <li><b>Layer</b>: Composite - contains multiple Neurons</li>
 *   <li><b>MLP</b>: Composite - contains multiple Layers</li>
 * </ul>
 * 
 * <h2>Example Usage</h2>
 * <pre>
 * // All components implement Module
 * Module neuron = new Neuron(3);
 * Module layer = new Layer(3, 4);
 * Module mlp = new MLP(Arrays.asList(3, 4, 1));
 * 
 * // Unified parameter access
 * List&lt;Value&gt; neuronParams = neuron.parameters();  // 4 params (3 weights + 1 bias)
 * List&lt;Value&gt; layerParams = layer.parameters();    // 16 params (4 neurons × 4 params)
 * List&lt;Value&gt; mlpParams = mlp.parameters();        // All parameters in network
 * 
 * // Training loop with Module interface
 * Module model = createModel();  // Can be any Module
 * 
 * for (int epoch = 0; epoch &lt; 100; epoch++) {
 *     // Forward pass
 *     Value loss = computeLoss(model, data);
 *     
 *     // Backward pass
 *     loss.backward();
 *     
 *     // Update parameters (works for any Module!)
 *     for (Value param : model.parameters()) {
 *         param.setData(param.getData() - learningRate * param.getGrad());
 *     }
 *     
 *     // Zero gradients for next iteration
 *     model.zeroGrad();
 * }
 * </pre>
 * 
 * <h2>Benefits of Module Interface</h2>
 * <ol>
 *   <li><b>Polymorphism</b>: Write code that works with any Module type</li>
 *   <li><b>Composability</b>: Easily build complex models from simple components</li>
 *   <li><b>Code Reuse</b>: Training loops work with any Module</li>
 *   <li><b>Extensibility</b>: Easy to add new Module types</li>
 *   <li><b>Testability</b>: Mock Modules for unit testing</li>
 * </ol>
 * 
 * <h2>Comparison with Micrograd</h2>
 * <pre>
 * Micrograd (Python):
 *   class Module:
 *       def zero_grad(self):
 *           for p in self.parameters():
 *               p.grad = 0
 *       
 *       def parameters(self):
 *           return []
 * 
 * This implementation (Java):
 *   interface Module {
 *       List&lt;Value&gt; parameters();
 *       
 *       default void zeroGrad() {
 *           for (Value p : parameters()) {
 *               p.setGrad(0.0);
 *           }
 *       }
 *   }
 * </pre>
 * 
 * <p><b>Note:</b> Java's interface with default methods provides the same
 * functionality as Python's base class, but with compile-time type checking
 * and better performance.
 * 
 * @author Vaibhav Khare
 * @see Neuron
 * @see Layer
 * @see MLP
 */
public interface Module {
    
    /**
     * Returns all learnable parameters in this module and its sub-modules.
     * 
     * <p><b>Implementation Requirements:</b>
     * <ul>
     *   <li>Must return all parameters that need gradients</li>
     *   <li>Should return an immutable list to prevent modification</li>
     *   <li>For composite modules, should collect parameters from all sub-modules</li>
     *   <li>Order should be consistent across calls</li>
     * </ul>
     * 
     * <p><b>Examples:</b>
     * <pre>
     * // Neuron: Returns its own weights and bias
     * Neuron neuron = new Neuron(3);
     * List&lt;Value&gt; params = neuron.parameters();  // [w0, w1, w2, b] = 4 params
     * 
     * // Layer: Returns parameters from all neurons
     * Layer layer = new Layer(3, 4);
     * List&lt;Value&gt; params = layer.parameters();  // 4 neurons × 4 params = 16 params
     * 
     * // MLP: Returns parameters from all layers
     * MLP mlp = new MLP(Arrays.asList(3, 4, 1));
     * List&lt;Value&gt; params = mlp.parameters();  // All parameters in network
     * </pre>
     * 
     * <p><b>Usage in Training:</b>
     * <pre>
     * // Access parameters for gradient descent
     * for (Value param : module.parameters()) {
     *     double newValue = param.getData() - learningRate * param.getGrad();
     *     param.setData(newValue);
     * }
     * </pre>
     *
     * @return an immutable list of all Value parameters in this module
     */
    List<Value> parameters();
    
    /**
     * Zeros out the gradients of all parameters in this module.
     * 
     * <p><b>Why This is Needed:</b>
     * In PyTorch and other frameworks, gradients accumulate by default (using +=).
     * This is essential for:
     * <ul>
     *   <li>Handling multiple paths in computation graph (chain rule)</li>
     *   <li>Batch gradient accumulation</li>
     *   <li>Gradient accumulation across mini-batches</li>
     * </ul>
     * 
     * <p>However, this means gradients must be zeroed before each new backward pass
     * to avoid accumulating gradients from previous iterations.
     * 
     * <p><b>When to Call:</b>
     * <ul>
     *   <li><b>Before backward pass</b>: Zero gradients before computing new ones</li>
     *   <li><b>After parameter update</b>: Clean up for next iteration</li>
     *   <li><b>Between mini-batches</b>: If not accumulating gradients</li>
     * </ul>
     * 
     * <p><b>Training Loop Pattern:</b>
     * <pre>
     * for (int epoch = 0; epoch &lt; epochs; epoch++) {
     *     for (Batch batch : dataLoader) {
     *         // 1. Zero gradients from previous iteration
     *         model.zeroGrad();
     *         
     *         // 2. Forward pass
     *         Value loss = computeLoss(model, batch);
     *         
     *         // 3. Backward pass (accumulates gradients)
     *         loss.backward();
     *         
     *         // 4. Update parameters
     *         optimizer.step();
     *     }
     * }
     * </pre>
     * 
     * <p><b>Alternative Pattern (Zero After Update):</b>
     * <pre>
     * for (int epoch = 0; epoch &lt; epochs; epoch++) {
     *     // Forward pass
     *     Value loss = computeLoss(model, data);
     *     
     *     // Backward pass
     *     loss.backward();
     *     
     *     // Update parameters
     *     for (Value param : model.parameters()) {
     *         param.setData(param.getData() - learningRate * param.getGrad());
     *     }
     *     
     *     // Zero gradients for next iteration
     *     model.zeroGrad();
     * }
     * </pre>
     * 
     * <p><b>Implementation:</b>
     * This is a default method that calls {@link #parameters()} and zeros each
     * parameter's gradient. Subclasses can override for performance optimizations.
     * 
     * <p><b>Example:</b>
     * <pre>
     * MLP model = new MLP(Arrays.asList(3, 4, 1));
     * 
     * // Gradients are initially zero
     * System.out.println(model.parameters().get(0).getGrad());  // 0.0
     * 
     * // After backward pass, gradients are accumulated
     * Value loss = computeLoss(model, data);
     * loss.backward();
     * System.out.println(model.parameters().get(0).getGrad());  // Some value
     * 
     * // Zero gradients for next iteration
     * model.zeroGrad();
     * System.out.println(model.parameters().get(0).getGrad());  // 0.0 again
     * </pre>
     * 
     * <p><b>Note:</b> This is equivalent to PyTorch's {@code model.zero_grad()}
     * and TensorFlow's gradient tape management.
     */
    default void zeroGrad() {
        for (Value param : parameters()) {
            param.setGrad(0.0);
        }
    }
    
    /**
     * Returns the number of learnable parameters in this module.
     * 
     * <p><b>Use Cases:</b>
     * <ul>
     *   <li>Model size estimation</li>
     *   <li>Memory usage calculation</li>
     *   <li>Debugging and validation</li>
     *   <li>Comparing model architectures</li>
     * </ul>
     * 
     * <p><b>Example:</b>
     * <pre>
     * MLP model = new MLP(Arrays.asList(784, 128, 10));
     * System.out.println("Model has " + model.numParameters() + " parameters");
     * // Output: Model has 101770 parameters
     * // Breakdown: (784×128 + 128) + (128×10 + 10) = 100480 + 1290 = 101770
     * </pre>
     * 
     * <p><b>Implementation:</b>
     * This is a default method that simply returns the size of the parameters list.
     *
     * @return the number of parameters in this module
     */
    default int numParameters() {
        return parameters().size();
    }
    
    /**
     * Returns an empty list of parameters.
     * 
     * <p>This is a convenience method for modules that don't have any parameters,
     * such as activation functions or pooling layers (if you implement them).
     * 
     * <p><b>Example Use Case:</b>
     * <pre>
     * public class ReLULayer implements Module {
     *     &#64;Override
     *     public List&lt;Value&gt; parameters() {
     *         return Module.emptyParameters();  // No learnable parameters
     *     }
     *     
     *     public Value forward(Value input) {
     *         return input.relu();
     *     }
     * }
     * </pre>
     *
     * @return an empty immutable list
     */
    static List<Value> emptyParameters() {
        return Collections.emptyList();
    }
}

