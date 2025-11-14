package com.vaibhavkhare.ml.autograd;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Value represents a node in a computational graph with automatic differentiation capabilities.
 * 
 * <p>This class implements the core autograd engine, similar to PyTorch's or TensorFlow's
 * automatic differentiation. Each Value stores:
 * <ul>
 *   <li>The actual data (forward pass result)</li>
 *   <li>The gradient (computed during backward pass)</li>
 *   <li>References to parent nodes in the computation graph</li>
 *   <li>The operation that created this Value</li>
 *   <li>A function to compute gradients during backpropagation</li>
 * </ul>
 * 
 * <p><b>⚠️ THREAD SAFETY WARNING:</b>
 * This class is <b>NOT thread-safe</b>. Each thread should use its own Value instances.
 * Sharing Value instances across threads during forward/backward passes will result in
 * race conditions and incorrect gradients due to:
 * <ul>
 *   <li>Non-atomic updates to {@code data} and {@code grad} fields</li>
 *   <li>Gradient accumulation (+=) not being thread-safe</li>
 *   <li>Shared mutable state in the computation graph</li>
 * </ul>
 * 
 * <p>For multi-threaded training:
 * <ul>
 *   <li><b>Option 1:</b> Create separate model instances per thread (recommended)</li>
 *   <li><b>Option 2:</b> Use external synchronization around forward/backward passes</li>
 *   <li><b>Option 3:</b> Process batches sequentially in a single thread</li>
 * </ul>
 * 
 * <p>Example usage:
 * <pre>
 * Value x = new Value(2.0, "x");
 * Value y = new Value(3.0, "y");
 * Value z = x.mul(y).add(x);
 * z.backward();  // Compute gradients
 * System.out.println(x.getGrad());  // Prints gradient dz/dx
 * </pre>
 * 
 * @author Vaibhav Khare
 */
public class Value {

    private double data;
    private double grad;
    private final List<Value> parents;
    private final String op;
    private String label;
    private Runnable backwardFn;

    /**
     * Creates a leaf Value node (typically an input or parameter).
     *
     * @param data  the numeric value
     * @param label optional label for visualization (can be empty string)
     */
    public Value(double data, String label) {
        this.data = data;
        this.grad = 0.0;
        this.parents = new ArrayList<>();
        this.op = "";
        this.label = label;
        this.backwardFn = () -> {};
    }

    /**
     * Creates a Value node resulting from an operation.
     *
     * @param data    the computed value
     * @param parents the input Values that produced this result
     * @param op      the operation name (e.g., "+", "*", "tanh")
     * @param label   optional label for visualization (can be empty string)
     */
    public Value(double data, List<Value> parents, String op, String label) {
        this.data = data;
        this.grad = 0.0;
        this.parents = new ArrayList<>(parents); // Defensive copy
        this.op = op;
        this.label = label;
        this.backwardFn = () -> {};
    }

    public double getData() {
        return data;
    }

    public void setData(double data) {
        this.data = data;
    }
    
    public double getGrad() {
        return grad;
    }

    public void setGrad(double grad) {
        this.grad = grad;
    }

    /**
     * Returns an unmodifiable view of the parent nodes.
     *
     * @return the list of parent Values (immutable)
     */
    public List<Value> getParents() {
        return Collections.unmodifiableList(parents);
    }

    public String getOp() {
        return op;
    }

    public String getLabel() {
        return label;
    }

    public void setLabel(String label) {
        this.label = label;
    }

    // Arithmetic operations
    // We will do addition, multiplication, and negation as basic arithmetic operations.

    /**
     * Why we use "+=" instead of "=":
     * - A Value can be used in multiple operations (e.g., a = x + y; b = x * z).
     * - When backpropagating, gradients from all paths where x is used must be accumulated.
     * - This is the multivariate chain rule: if x affects the output through multiple paths,
     *   the total gradient is the sum of gradients from each path.
     * - Using "+=" ensures we accumulate gradients from all downstream operations.
     * 
     */ 

    /**
     * Addition in Micrograd
     * 
     * Suppose you have a and b, and you compute c = a + b:
     * 
     * Forward pass: Just like normal math, c gets the value a + b.
     * 
     * Backward pass (gradient):
     * - Derivative of c with respect to a is 1 (because if you increase a by 1, c goes up by 1).
     * - Same for b: derivative is 1.
     * 
     * So, during backpropagation, the upstream gradient (coming from after c) is passed 
     * directly to both a and b, unchanged.
     */
    public Value add(Value other) {
        Value a = this;
        Value b = other;
        // Forward pass c.data = a.data + b.data
        Value out = new Value(a.data + b.data, new ArrayList<>(Arrays.asList(a, b)), "+", "");
        // Backward pass a.grad += out.grad, b.grad += out.grad
        // Because of the chain rule, the gradient of the output is propagated back to the input.
        out.backwardFn = () -> {
            a.grad += out.grad;
            b.grad += out.grad;
        };
        return out;
    }

    /**
     * Multiplication in Micrograd
     * 
     * Suppose you have a and b, and you compute c = a * b:
     * 
     * Forward pass: c gets the value a * b.
     * 
     * Backward pass (gradient):
     * - Derivative of c with respect to a is b (because ∂(a*b)/∂a = b).
     *   In other words: if a increases by a small amount, c increases by that amount times b.
     * - Same for b: derivative is a.
     * 
     * So, during backpropagation, the upstream gradient (coming from after c) is passed to both a and b, multiplied by their current values.
     */
    public Value mul(Value other) {
        Value a = this;
        Value b = other;
        // Forward pass c.data = a.data * b.data
        Value out = new Value(a.data * b.data, new ArrayList<>(Arrays.asList(a, b)), "*", "");
        // Backward pass: 
        // - a.grad += b.data * out.grad (derivative of c w.r.t. a is b)
        // - b.grad += a.data * out.grad (derivative of c w.r.t. b is a)
        out.backwardFn = () -> {
            a.grad += b.data * out.grad;
            b.grad += a.data * out.grad;
        };
        return out;
    }

    /**
     * Negation in Micrograd
     * 
     * Suppose you have a, and you compute b = -a:
     * 
     * Forward pass: b gets the value -a.
     * 
     * Backward pass (gradient):
     * - Derivative of b with respect to a is -1 (because ∂(-a)/∂a = -1).
     * 
     * So, during backpropagation, the upstream gradient is negated before being passed to a.
     */
    public Value neg() {
        Value a = this;
        // Forward pass: out.data = -a.data
        Value out = new Value(-a.data, new ArrayList<>(Arrays.asList(a)), "neg", "");
        // Backward pass: a.grad += -1 * out.grad
        out.backwardFn = () -> a.grad += -out.grad;
        return out;
    }

    /**
     * Subtraction in Micrograd
     * 
     * Suppose you have a and b, and you compute c = a - b:
     * 
     * Forward pass: c gets the value a - b.
     * 
     * Backward pass (gradient):
     * - Derivative of c with respect to a is 1 (because ∂(a-b)/∂a = 1).
     * - Derivative of c with respect to b is -1 (because ∂(a-b)/∂b = -1).
     * 
     * Implementation: c = a + (-b), so we reuse add and neg operations.
     */
    public Value sub(Value other) {
        return this.add(other.neg());
    }

    /**
     * Power in Micrograd
     * 
     * Suppose you have a, and you compute b = a^n (where n is a constant):
     * 
     * Forward pass: b gets the value a^n.
     * 
     * Backward pass (gradient):
     * - Derivative of b with respect to a is n * a^(n-1) (power rule).
     * 
     * So, during backpropagation, the upstream gradient is multiplied by n * a^(n-1).
     */
    public Value pow(double exponent) {
        Value a = this;
        // Forward pass: out.data = a.data^exponent
        Value out = new Value(Math.pow(a.data, exponent), new ArrayList<>(Arrays.asList(a)), "pow", "");
        // Backward pass: a.grad += exponent * a.data^(exponent-1) * out.grad
        out.backwardFn = () -> a.grad += exponent * Math.pow(a.data, exponent - 1) * out.grad;
        return out;
    }

    /**
     * Division in Micrograd
     * 
     * Suppose you have a and b, and you compute c = a / b:
     * 
     * Forward pass: c gets the value a / b.
     * 
     * Backward pass (gradient):
     * - Derivative of c with respect to a is 1/b (because ∂(a/b)/∂a = 1/b).
     * - Derivative of c with respect to b is -a/b^2 (because ∂(a/b)/∂b = -a/b^2).
     * 
     * Implementation: c = a * b^(-1), so we reuse mul and pow operations.
     */
    public Value div(Value other) {
        return this.mul(other.pow(-1));
    }

    // Scalar operations (for convenience)
    
    /**
     * Add a scalar to this Value (this + scalar)
     */
    public Value add(double scalar) {
        return this.add(new Value(scalar, ""));
    }

    /**
     * Multiply this Value by a scalar (this * scalar)
     */
    public Value mul(double scalar) {
        return this.mul(new Value(scalar, ""));
    }

    /**
     * Subtract a scalar from this Value (this - scalar)
     */
    public Value sub(double scalar) {
        return this.sub(new Value(scalar, ""));
    }

    /**
     * Divide this Value by a scalar (this / scalar)
     */
    public Value div(double scalar) {
        return this.div(new Value(scalar, ""));
    }

    // Right-hand operations (for when Value is on the right side)

    /**
     * Right-hand subtraction: scalar - this
     * Used when a scalar is on the left side of subtraction.
     */
    public Value rsub(double scalar) {
        return new Value(scalar, "").sub(this);
    }

    /**
     * Right-hand division: scalar / this
     * Used when a scalar is on the left side of division.
     */
    public Value rdiv(double scalar) {
        return new Value(scalar, "").div(this);
    }

    // Activation Functions

    /**
     * Hyperbolic Tangent (tanh) Activation Function
     * 
     * tanh squashes values to the range (-1, 1), making it useful for neural networks.
     * 
     * Forward pass: out = tanh(a)
     * 
     * Backward pass (gradient):
     * - Derivative of tanh(a) with respect to a is (1 - tanh²(a))
     * - This can be computed as (1 - out²) where out = tanh(a)
     * 
     * The tanh function is commonly used as an activation function in neural networks
     * because it's zero-centered and has a smooth gradient.
     */
    public Value tanh() {
        Value a = this;
        // Forward pass: out.data = tanh(a.data)
        double t = Math.tanh(a.data);
        Value out = new Value(t, new ArrayList<>(Arrays.asList(a)), "tanh", "");
        // Backward pass: a.grad += (1 - tanh²(a)) * out.grad
        // Since t = tanh(a), we use (1 - t²)
        out.backwardFn = () -> a.grad += (1 - t * t) * out.grad;
        return out;
    }

    /**
     * Exponential Function (exp)
     * 
     * Computes e^x where e is Euler's number (~2.718).
     * 
     * Forward pass: out = e^a
     * 
     * Backward pass (gradient):
     * - Derivative of e^a with respect to a is e^a
     * - This means the derivative equals the function value itself
     * 
     * The exp function is used in softmax, sigmoid, and many other operations.
     */
    public Value exp() {
        Value a = this;
        // Forward pass: out.data = e^(a.data)
        double x = Math.exp(a.data);
        Value out = new Value(x, new ArrayList<>(Arrays.asList(a)), "exp", "");
        // Backward pass: a.grad += e^a * out.grad
        // Since x = e^a, we use x directly
        out.backwardFn = () -> a.grad += x * out.grad;
        return out;
    }

    /**
     * Rectified Linear Unit (ReLU) Activation Function
     * 
     * ReLU is one of the most popular activation functions in modern neural networks.
     * It outputs the input if positive, otherwise outputs zero: max(0, x)
     * 
     * Forward pass: out = max(0, a)
     * 
     * Backward pass (gradient):
     * - Derivative is 1 if a > 0, otherwise 0
     * - This creates a "gate" that passes gradients through only for positive values
     * 
     * ReLU is popular because:
     * - It's computationally efficient
     * - It helps mitigate the vanishing gradient problem
     * - It introduces non-linearity while being simple
     */
    public Value relu() {
        Value a = this;
        // Forward pass: out.data = max(0, a.data)
        Value out = new Value(a.data < 0 ? 0 : a.data, new ArrayList<>(Arrays.asList(a)), "ReLU", "");
        // Backward pass: a.grad += (out.data > 0 ? 1 : 0) * out.grad
        out.backwardFn = () -> a.grad += (out.data > 0 ? 1 : 0) * out.grad;
        return out;
    }

    /**
     * Applies the sigmoid activation function.
     * 
     * <p>Sigmoid: σ(x) = 1 / (1 + e^(-x))
     * 
     * <p>The sigmoid function:
     * - Maps input to range (0, 1)
     * - Used for binary classification
     * - S-shaped curve
     * - Derivative: σ'(x) = σ(x) * (1 - σ(x))
     * 
     * <p>Example:
     * <pre>
     * Value x = new Value(0.0);
     * Value y = x.sigmoid();  // y.data = 0.5
     * 
     * Value x2 = new Value(2.0);
     * Value y2 = x2.sigmoid();  // y2.data ≈ 0.88
     * </pre>
     * 
     * Forward pass: out = 1 / (1 + exp(-a))
     * 
     * Backward pass (gradient):
     * - Derivative: σ'(x) = σ(x) * (1 - σ(x))
     * - This gives gradients in range (0, 0.25]
     * 
     * Sigmoid is popular for:
     * - Binary classification (output layer)
     * - Gating mechanisms in LSTMs
     * - When you need probabilities (0-1 range)
     */
    public Value sigmoid() {
        Value a = this;
        // Forward pass: out.data = 1 / (1 + exp(-a.data))
        double sigmoid = 1.0 / (1.0 + Math.exp(-a.data));
        Value out = new Value(sigmoid, new ArrayList<>(Arrays.asList(a)), "sigmoid", "");
        // Backward pass: a.grad += sigmoid * (1 - sigmoid) * out.grad
        out.backwardFn = () -> a.grad += sigmoid * (1.0 - sigmoid) * out.grad;
        return out;
    }

    /**
     * Applies the natural logarithm function.
     * 
     * <p>Natural logarithm: ln(x)
     * 
     * <p>The log function:
     * - Maps (0, ∞) to (-∞, ∞)
     * - Inverse of exp()
     * - Derivative: d(ln(x))/dx = 1/x
     * 
     * <p>Example:
     * <pre>
     * Value x = new Value(1.0);
     * Value y = x.log();  // y.data = 0.0
     * 
     * Value x2 = new Value(Math.E);
     * Value y2 = x2.log();  // y2.data = 1.0
     * </pre>
     * 
     * Forward pass: out = ln(a)
     * 
     * Backward pass (gradient):
     * - Derivative: d(ln(x))/dx = 1/x
     * - Note: only defined for x > 0
     * 
     * Log is commonly used for:
     * - Cross-entropy loss
     * - Log-likelihood
     * - Numerical stability in multiplications
     * 
     * @throws ArithmeticException if called on non-positive value
     */
    public Value log() {
        Value a = this;
        if (a.data <= 0) {
            throw new ArithmeticException("Cannot compute log of non-positive number: " + a.data);
        }
        // Forward pass: out.data = ln(a.data)
        Value out = new Value(Math.log(a.data), new ArrayList<>(Arrays.asList(a)), "log", "");
        // Backward pass: a.grad += (1 / a.data) * out.grad
        out.backwardFn = () -> a.grad += (1.0 / a.data) * out.grad;
        return out;
    }

    // Backpropagation
    
    /**
     * Performs backpropagation to compute gradients for all nodes in the computation graph.
     * 
     * <p>This method:
     * <ol>
     *   <li>Builds a topological ordering of all nodes in the computation graph</li>
     *   <li>Initializes all gradients to zero</li>
     *   <li>Sets this node's gradient to 1.0 (since d(output)/d(output) = 1)</li>
     *   <li>Traverses the graph in reverse topological order, calling each node's backward function</li>
     * </ol>
     * 
     * <p>After calling this method, all ancestor nodes will have their gradients populated.
     */
    public void backward() {
        List<Value> topo = new ArrayList<>();
        Set<Value> visited = new HashSet<>();
        buildTopo(this, topo, visited);

        // Zero out all gradients
        for (Value v : topo) {
            v.grad = 0.0;
        }
        
        // The gradient of output with respect to itself is 1
        this.grad = 1.0;

        // Backpropagate through the graph in reverse topological order
        for (int i = topo.size() - 1; i >= 0; i--) {
            topo.get(i).backwardFn.run();
        }
    }

    /**
     * Builds a topological ordering of the computation graph.
     *
     * @param v       the current node
     * @param topo    the list to store the topological ordering
     * @param visited set of already visited nodes
     */
    private static void buildTopo(Value v, List<Value> topo, Set<Value> visited) {
        if (visited.contains(v)) {
            return;
        }
        visited.add(v);
        for (Value p : v.parents) {
            buildTopo(p, topo, visited);
        }
        topo.add(v);
    }

    // Utilities
    
    /**
     * Zeros out gradients for all Values in the provided list.
     * Useful when reusing Values across multiple backward passes.
     *
     * @param values the list of Values to zero out (can be null)
     */
    public static void zeroGrad(List<Value> values) {
        if (values == null) {
            return;
        }
        for (Value v : values) {
            if (v != null) {
                v.grad = 0.0;
            }
        }
    }

    @Override
    public String toString() {
        return "Value{" +
                "data=" + data +
                ", grad=" + grad +
                ", op='" + op + '\'' +
                (label != null && !label.isEmpty() ? ", label='" + label + '\'' : "") +
                '}';
    }

}
