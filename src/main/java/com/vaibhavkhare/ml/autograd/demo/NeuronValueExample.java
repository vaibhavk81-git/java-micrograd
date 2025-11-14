package com.vaibhavkhare.ml.autograd.demo;

import com.vaibhavkhare.ml.autograd.ActivationType;
import com.vaibhavkhare.ml.autograd.GraphVisualizer;
import com.vaibhavkhare.ml.autograd.Neuron;
import com.vaibhavkhare.ml.autograd.Value;

import java.util.Arrays;
import java.util.List;

/**
 * Demonstrates the relationship between Neuron and Value classes.
 * Run this to see concrete examples of how they work together.
 * 
 * <p>This example includes visualizations of computation graphs:
 * <ul>
 *   <li>Example 2: Forward pass graph (neuron_forward_pass.png)</li>
 *   <li>Example 3: Backward pass graph with gradients (neuron_backward_pass.png)</li>
 *   <li>Example 5: Linear vs Tanh activation graphs (neuron_linear_activation.png, neuron_tanh_activation.png)</li>
 *   <li>Example 6: Shared inputs with gradient accumulation (neuron_shared_inputs.png)</li>
 * </ul>
 * 
 * <p>Run with: {@code ./gradlew run --args="com.vaibhavkhare.ml.autograd.demo.NeuronValueExample"}
 */
@SuppressWarnings("java:S106") // Allow System.out for demonstration
public class NeuronValueExample {

    // Constants for repeated strings
    private static final String BOX_TOP = "╔═══════════════════════════════════════════════════════════╗";
    private static final String BOX_BOTTOM = "╚═══════════════════════════════════════════════════════════╝";
    private static final String GRAPH_SKIP_MSG = "%n  (Graph visualization skipped: %s)%n";
    private static final String X1_GRAD_FORMAT = "  x1.grad = %.4f%n";
    private static final String X2_GRAD_FORMAT = "  x2.grad = %.4f%n";

    public static void main(String[] args) {
        example1CreatingNeuron();
        example2ForwardPassTrace();
        example3BackwardPassTrace();
        example4ParameterUpdate();
        example5WithActivation();
        example6SharedInputs();
    }

    /**
     * Example 1: What's inside a Neuron?
     */
    private static void example1CreatingNeuron() {
        System.out.println(BOX_TOP);
        System.out.println("║  Example 1: Creating a Neuron - What Value Objects Exist? ║");
        System.out.println(BOX_BOTTOM);
        System.out.println();

        Neuron neuron = new Neuron(2, ActivationType.LINEAR, new java.util.Random(42L));

        System.out.println("Created: Neuron(2 inputs, no activation)");
        System.out.println("\nNeuron contains these Value objects:");
        System.out.println("  Weights:");
        for (int i = 0; i < neuron.getWeights().size(); i++) {
            Value w = neuron.getWeights().get(i);
            System.out.printf("    w%d = %s%n", i, w);
        }
        System.out.println("  Bias:");
        System.out.printf("    b = %s%n", neuron.getBias());

        System.out.println("\nTotal Value objects in neuron: " + neuron.parameters().size());
        System.out.println("All are Value objects with data, grad, and computational graph tracking!\n");
    }

    /**
     * Example 2: Forward pass - creating new Value objects
     */
    private static void example2ForwardPassTrace() {
        System.out.println(BOX_TOP);
        System.out.println("║  Example 2: Forward Pass - New Value Objects Created      ║");
        System.out.println(BOX_BOTTOM);
        System.out.println();

        Neuron neuron = new Neuron(2, ActivationType.LINEAR, new java.util.Random(42L));

        System.out.println("Neuron weights:");
        System.out.printf("  w0 = %.4f%n", neuron.getWeights().get(0).getData());
        System.out.printf("  w1 = %.4f%n", neuron.getWeights().get(1).getData());
        System.out.printf("  bias = %.4f%n", neuron.getBias().getData());

        System.out.println("\nCreating input Values:");
        Value x1 = new Value(3.0, "x1");
        Value x2 = new Value(4.0, "x2");
        System.out.printf("  x1 = %.4f%n", x1.getData());
        System.out.printf("  x2 = %.4f%n", x2.getData());

        System.out.println("\nForward pass computation:");
        List<Value> inputs = Arrays.asList(x1, x2);
        Value output = neuron.forward(inputs);

        System.out.println("  Step 1: w0 * x1 = " + neuron.getWeights().get(0).getData() + " * " + x1.getData() + " = " + (neuron.getWeights().get(0).getData() * x1.getData()));
        System.out.println("  Step 2: w1 * x2 = " + neuron.getWeights().get(1).getData() + " * " + x2.getData() + " = " + (neuron.getWeights().get(1).getData() * x2.getData()));
        System.out.println("  Step 3: bias + (w0*x1) + (w1*x2)");
        System.out.printf("  Result: output = %.4f%n", output.getData());

        System.out.println("\nInspecting the output Value:");
        System.out.println("  Output is a Value object: " + output);
        System.out.println("  Output has " + output.getParents().size() + " parents in the graph");
        System.out.println("  Output operation: '" + output.getOp() + "'");

        System.out.println("\nComputational graph was built automatically!");
        System.out.println("  Each operation (mul, add) created new Value objects");
        System.out.println("  These Value objects remember their parents");
        
        // Visualize the computation graph
        try {
            GraphVisualizer.visualize(output, "neuron_forward_pass");
            System.out.printf("%n✓ Computation graph saved to: neuron_forward_pass.png%n");
            System.out.println("  Open the image to see the full graph structure!");
            System.out.println();
        } catch (Exception e) {
            System.out.printf(GRAPH_SKIP_MSG, e.getMessage());
        }
    }

    /**
     * Example 3: Backward pass - gradients flow through Value objects
     */
    private static void example3BackwardPassTrace() {
        System.out.println(BOX_TOP);
        System.out.println("║  Example 3: Backward Pass - Gradients Flow Through Values ║");
        System.out.println(BOX_BOTTOM);
        System.out.println();

        Neuron neuron = new Neuron(2, ActivationType.LINEAR, new java.util.Random(42L));
        Value x1 = new Value(3.0, "x1");
        Value x2 = new Value(4.0, "x2");

        System.out.println("Before forward pass - all gradients are 0:");
        System.out.printf("  w0.grad = %.4f%n", neuron.getWeights().get(0).getGrad());
        System.out.printf("  w1.grad = %.4f%n", neuron.getWeights().get(1).getGrad());
        System.out.printf("  bias.grad = %.4f%n", neuron.getBias().getGrad());
        System.out.printf(X1_GRAD_FORMAT, x1.getGrad());
        System.out.printf(X2_GRAD_FORMAT, x2.getGrad());

        System.out.println("\nForward pass:");
        List<Value> inputs = Arrays.asList(x1, x2);
        Value output = neuron.forward(inputs);
        System.out.printf("  output = %.4f%n", output.getData());

        System.out.println("\nCalling output.backward() - this is Value's method!");
        output.backward();

        System.out.println("\nAfter backward pass - gradients computed:");
        System.out.printf("  w0.grad = %.4f  (∂output/∂w0 = x1 = %.4f)%n", 
            neuron.getWeights().get(0).getGrad(), x1.getData());
        System.out.printf("  w1.grad = %.4f  (∂output/∂w1 = x2 = %.4f)%n", 
            neuron.getWeights().get(1).getGrad(), x2.getData());
        System.out.printf("  bias.grad = %.4f  (∂output/∂bias = 1.0)%n", 
            neuron.getBias().getGrad());
        System.out.printf("  x1.grad = %.4f  (∂output/∂x1 = w0 = %.4f)%n", 
            x1.getGrad(), neuron.getWeights().get(0).getData());
        System.out.printf("  x2.grad = %.4f  (∂output/∂x2 = w1 = %.4f)%n", 
            x2.getGrad(), neuron.getWeights().get(1).getData());

        System.out.println("\nKey insight: Gradients flowed through Value objects!");
        System.out.println("  Neuron didn't compute gradients - Value's backward() did!");
        
        // Visualize the computation graph with gradients
        try {
            GraphVisualizer.visualize(output, "neuron_backward_pass");
            System.out.printf("%n✓ Computation graph with gradients saved to: neuron_backward_pass.png%n");
            System.out.println("  Open the image to see gradients at each node!");
            System.out.println();
        } catch (Exception e) {
            System.out.printf(GRAPH_SKIP_MSG, e.getMessage());
        }
    }

    /**
     * Example 4: Parameter update - modifying Value objects
     */
    private static void example4ParameterUpdate() {
        System.out.println(BOX_TOP);
        System.out.println("║  Example 4: Parameter Update - Modifying Value Objects    ║");
        System.out.println(BOX_BOTTOM);
        System.out.println();

        Neuron neuron = new Neuron(2, ActivationType.LINEAR, new java.util.Random(42L));

        System.out.println("Initial parameters (these are Value objects):");
        for (int i = 0; i < neuron.getWeights().size(); i++) {
            System.out.printf("  w%d.data = %.4f, w%d.grad = %.4f%n", 
                i, neuron.getWeights().get(i).getData(),
                i, neuron.getWeights().get(i).getGrad());
        }
        System.out.printf("  bias.data = %.4f, bias.grad = %.4f%n", 
            neuron.getBias().getData(), neuron.getBias().getGrad());

        // Do forward and backward
        Value x1 = new Value(3.0, "x1");
        Value x2 = new Value(4.0, "x2");
        Value output = neuron.forward(Arrays.asList(x1, x2));
        output.backward();

        System.out.println("\nAfter forward + backward (gradients computed):");
        for (int i = 0; i < neuron.getWeights().size(); i++) {
            System.out.printf("  w%d.grad = %.4f%n", i, neuron.getWeights().get(i).getGrad());
        }
        System.out.printf("  bias.grad = %.4f%n", neuron.getBias().getGrad());

        // Update parameters
        System.out.println("\nApplying gradient descent (learning_rate = 0.1):");
        System.out.println("  new_value = old_value - learning_rate * gradient");
        double learningRate = 0.1;
        for (Value param : neuron.parameters()) {
            double oldValue = param.getData();
            double gradient = param.getGrad();
            double newValue = oldValue - learningRate * gradient;
            param.setData(newValue);  // Modifying the Value object!
            System.out.printf("  %.4f - 0.1 * %.4f = %.4f%n", oldValue, gradient, newValue);
        }

        System.out.println("\nAfter update (same Value objects, different data):");
        for (int i = 0; i < neuron.getWeights().size(); i++) {
            System.out.printf("  w%d.data = %.4f (Value object modified)%n", 
                i, neuron.getWeights().get(i).getData());
        }
        System.out.printf("  bias.data = %.4f (Value object modified)%n", 
            neuron.getBias().getData());

        System.out.println("\nWe modified the SAME Value objects that neuron holds!\n");
    }

    /**
     * Example 5: Neuron with activation - extra Value created
     */
    private static void example5WithActivation() {
        System.out.println(BOX_TOP);
        System.out.println("║  Example 5: Activation Function - Extra Value Created     ║");
        System.out.println(BOX_BOTTOM);
        System.out.println();

        // Linear neuron
        Neuron linearNeuron = new Neuron(2, ActivationType.LINEAR, new java.util.Random(42L));
        Value x1 = new Value(1.0, "x1");
        Value x2 = new Value(2.0, "x2");
        Value linearOutput = linearNeuron.forward(Arrays.asList(x1, x2));

        System.out.println("Linear neuron (no activation):");
        System.out.printf("  output.data = %.4f%n", linearOutput.getData());
        System.out.println("  output.op = '" + linearOutput.getOp() + "'");

        // Tanh neuron (same weights via seed)
        Neuron tanhNeuron = new Neuron(2, ActivationType.TANH, new java.util.Random(42L));
        Value x3 = new Value(1.0, "x3");
        Value x4 = new Value(2.0, "x4");
        Value tanhOutput = tanhNeuron.forward(Arrays.asList(x3, x4));

        System.out.println("\nTanh neuron (with activation):");
        System.out.printf("  pre-activation = %.4f (same as linear)%n", linearOutput.getData());
        System.out.printf("  output.data = %.4f (after tanh)%n", tanhOutput.getData());
        System.out.println("  output.op = '" + tanhOutput.getOp() + "'");

        System.out.println("\nThe activation function is a Value operation!");
        System.out.println("  tanh() creates a new Value with op='tanh'");
        System.out.println("  This Value tracks the pre-activation as its parent");
        
        // Visualize both graphs to compare
        try {
            GraphVisualizer.visualize(linearOutput, "neuron_linear_activation");
            GraphVisualizer.visualize(tanhOutput, "neuron_tanh_activation");
            System.out.printf("%n✓ Computation graphs saved:%n");
            System.out.println("  - neuron_linear_activation.png (no activation)");
            System.out.println("  - neuron_tanh_activation.png (with tanh activation)");
            System.out.println("  Compare the graphs to see the extra tanh node!");
            System.out.println();
        } catch (Exception e) {
            System.out.printf(GRAPH_SKIP_MSG, e.getMessage());
        }
    }

    /**
     * Example 6: Shared inputs - gradient accumulation
     */
    private static void example6SharedInputs() {
        System.out.println(BOX_TOP);
        System.out.println("║  Example 6: Shared Inputs - Gradient Accumulation         ║");
        System.out.println(BOX_BOTTOM);
        System.out.println();

        Neuron neuron1 = new Neuron(2, ActivationType.LINEAR, new java.util.Random(42L));
        Neuron neuron2 = new Neuron(2, ActivationType.LINEAR, new java.util.Random(123L));

        // SAME input Value objects used by both neurons
        Value x1 = new Value(1.0, "x1");
        Value x2 = new Value(2.0, "x2");
        List<Value> inputs = Arrays.asList(x1, x2);

        System.out.println("Created 2 neurons with same inputs");
        System.out.println("  x1 and x2 are Value objects shared by both neurons");

        Value out1 = neuron1.forward(inputs);
        Value out2 = neuron2.forward(inputs);

        System.out.printf("%nNeuron 1 output: %.4f%n", out1.getData());
        System.out.printf("Neuron 2 output: %.4f%n", out2.getData());

        // Combine outputs
        Value combined = out1.add(out2);
        System.out.printf("Combined output: %.4f%n", combined.getData());

        System.out.println("\nBefore backward - input gradients:");
        System.out.printf(X1_GRAD_FORMAT, x1.getGrad());
        System.out.printf(X2_GRAD_FORMAT, x2.getGrad());

        combined.backward();

        System.out.println("\nAfter backward - input gradients:");
        System.out.printf(X1_GRAD_FORMAT, x1.getGrad());
        System.out.printf(X2_GRAD_FORMAT, x2.getGrad());

        System.out.println("\nGradients ACCUMULATED from both paths!");
        System.out.println("  x1.grad = contribution from neuron1 + contribution from neuron2");
        System.out.println("  This is why Value uses '+=' for gradients!");
        System.out.println("\nThe same Value objects received gradients from multiple sources");
        
        // Visualize the combined computation graph showing gradient accumulation
        try {
            GraphVisualizer.visualize(combined, "neuron_shared_inputs");
            System.out.printf("%n✓ Combined computation graph saved to: neuron_shared_inputs.png%n");
            System.out.println("  Open the image to see how gradients accumulate at shared inputs!");
            System.out.println();
        } catch (Exception e) {
            System.out.printf(GRAPH_SKIP_MSG, e.getMessage());
        }
    }
}

