package com.vaibhavkhare.ml.autograd.demo;

import com.vaibhavkhare.ml.autograd.ActivationType;
import com.vaibhavkhare.ml.autograd.GraphVisualizer;
import com.vaibhavkhare.ml.autograd.Layer;
import com.vaibhavkhare.ml.autograd.MLP;
import com.vaibhavkhare.ml.autograd.Neuron;
import com.vaibhavkhare.ml.autograd.Value;

/**
 * Demonstration application that walks through the core ideas of this project.
 * 
 * What it covers (quick tour):
 * 1) Computation graph basics with scalar Values:
 *    - Example 1: f(x,y) = (x + y) * (x - y) with backward() and gradients
 *    - Example 2: g(a,b) = (a^2 + b^2) / (a + b) with labels for clarity
 *    - Example 3: Neuron-like computation with tanh activation
 *    Each example also saves a PNG graph using GraphVisualizer.
 * 
 * 2) From scalars to neural nets:
 *    - Neuron: forward pass, simple training loop, and activation comparison
 *    - Layer: multiple neurons processing the same inputs; brief training
 *    - MLP: a small feedforward net; forward pass and short training demo
 * 
 * Design goals:
 * - Keep output self-explanatory with System.out (educational first)
 * - Show both numeric results and saved computation graphs
 * - Provide a single entry point to understand Value → Neuron → Layer → MLP
 */
@SuppressWarnings({"java:S106", "java:S1192"}) // Allow System.out and string literals for demo purposes
public class Application {
    
    private static final String RESULT_PREFIX = "Result: ";
    private static final String GRADIENT_PREFIX = "Gradient ";
    
    private Application() {
        // Private constructor to hide implicit public one
    }
    
    public static void main(String[] args) {
        System.out.println("=== Micrograd-style Computation Graph Visualization ===\n");
        
        // Example 1: Simple expression similar to micrograd demo
        // f(x, y) = (x + y) * (x - y)
        System.out.println("Example 1: f(x, y) = (x + y) * (x - y)");
        Value x = new Value(3.0, "x");
        Value y = new Value(2.0, "y");
        
        Value sum = x.add(y);
        sum.setLabel("x+y");
        
        Value diff = x.sub(y);
        diff.setLabel("x-y");
        
        Value result = sum.mul(diff);
        result.setLabel("f");
        
        // Run backward pass
        result.backward();
        
        System.out.println(RESULT_PREFIX + result.getData());
        System.out.println(GRADIENT_PREFIX + "df/dx: " + x.getGrad());
        System.out.println(GRADIENT_PREFIX + "df/dy: " + y.getGrad());
        
        // Visualize
        GraphVisualizer.visualize(result, "example1_simple");
        
        // Example 2: More complex expression with power
        // g(a, b) = (a^2 + b^2) / (a + b)
        System.out.println("\nExample 2: g(a, b) = (a^2 + b^2) / (a + b)");
        Value a = new Value(4.0, "a");
        Value b = new Value(3.0, "b");
        
        Value a2 = a.pow(2);
        a2.setLabel("a²");
        
        Value b2 = b.pow(2);
        b2.setLabel("b²");
        
        Value numerator = a2.add(b2);
        numerator.setLabel("a²+b²");
        
        Value denominator = a.add(b);
        denominator.setLabel("a+b");
        
        Value result2 = numerator.div(denominator);
        result2.setLabel("g");
        
        // Run backward pass
        result2.backward();
        
        System.out.println(RESULT_PREFIX + result2.getData());
        System.out.println(GRADIENT_PREFIX + "dg/da: " + a.getGrad());
        System.out.println(GRADIENT_PREFIX + "dg/db: " + b.getGrad());
        
        // Visualize
        GraphVisualizer.visualize(result2, "example2_complex");
        
        // Example 3: Neuron-like computation with tanh activation
        // out = tanh(w1*x1 + w2*x2 + b)
        System.out.println("\nExample 3: Neuron computation with tanh activation");
        Value w1 = new Value(0.5, "w1");
        Value x1 = new Value(2.0, "x1");
        Value w2 = new Value(-0.3, "w2");
        Value x2 = new Value(1.5, "x2");
        Value bias = new Value(0.8, "b");
        
        Value wx1 = w1.mul(x1);
        wx1.setLabel("w1*x1");
        
        Value wx2 = w2.mul(x2);
        wx2.setLabel("w2*x2");
        
        Value sum1 = wx1.add(wx2);
        sum1.setLabel("Σ");
        
        Value preActivation = sum1.add(bias);
        preActivation.setLabel("z");
        
        Value neuronOut = preActivation.tanh();
        neuronOut.setLabel("out");
        
        // Run backward pass
        neuronOut.backward();
        
        System.out.println(RESULT_PREFIX + neuronOut.getData());
        System.out.println(GRADIENT_PREFIX + "dout/dw1: " + w1.getGrad());
        System.out.println(GRADIENT_PREFIX + "dout/dx1: " + x1.getGrad());
        System.out.println(GRADIENT_PREFIX + "dout/db: " + bias.getGrad());
        
        // Visualize
        GraphVisualizer.visualize(neuronOut, "example3_neuron");
        
        System.out.println("\n✓ All visualizations saved!");
        System.out.println("  - example1_simple.png");
        System.out.println("  - example2_complex.png");
        System.out.println("  - example3_neuron.png");
        
        // Example 4: Using the Neuron class
        demonstrateNeuron();
        
        // Example 5: Using the Layer class
        demonstrateLayer();
        
        // Example 6: Using the MLP class (complete neural network)
        demonstrateMLP();
    }
    
    /**
     * Demonstrates the Neuron class with detailed explanations.
     */
    private static void demonstrateNeuron() {
        System.out.println("\n╔════════════════════════════════════════════════════════╗");
        System.out.println("║        NEURON CLASS DEMONSTRATION                      ║");
        System.out.println("╚════════════════════════════════════════════════════════╝");
        
        // Part 1: Creating and using a single neuron
        System.out.println("\n--- Part 1: Single Neuron Forward Pass ---");
        System.out.println("Creating a neuron with 3 inputs (like x, y, z coordinates)");
        
        Neuron neuron = new Neuron(3, new java.util.Random(42L)); // Use RNG for reproducible results
        
        System.out.println("Initial neuron: " + neuron);
        System.out.println("\nWeights:");
        for (int i = 0; i < neuron.getWeights().size(); i++) {
            System.out.printf("  w%d = %.4f%n", i, neuron.getWeights().get(i).getData());
        }
        System.out.printf("Bias: b = %.4f%n", neuron.getBias().getData());
        
        // Create some inputs
        java.util.List<Value> inputs = java.util.Arrays.asList(
            new Value(1.5, "x"),
            new Value(2.0, "y"),
            new Value(-0.5, "z")
        );
        
        System.out.println("\nInputs:");
        System.out.println("  x = 1.5");
        System.out.println("  y = 2.0");
        System.out.println("  z = -0.5");
        
        // Forward pass
        Value output = neuron.forward(inputs);
        
        System.out.println("\nComputation:");
        System.out.println("  weighted_sum = w0*x + w1*y + w2*z + b");
        System.out.printf("  output = tanh(weighted_sum) = %.4f%n", output.getData());
        
        // Part 2: Training a neuron (gradient descent)
        System.out.println("\n--- Part 2: Training a Neuron with Gradient Descent ---");
        System.out.println("Goal: Adjust weights to make output closer to target value");
        
        double targetValue = 0.8;
        System.out.printf("Target output: %.4f%n", targetValue);
        System.out.printf("Current output: %.4f%n", output.getData());
        System.out.printf("Initial error: %.4f%n", Math.abs(targetValue - output.getData()));
        
        // Training loop
        System.out.println("\nTraining for 20 iterations...");
        int iterations = 20;
        double learningRate = 0.1;
        
        for (int iter = 0; iter < iterations; iter++) {
            // Forward pass (recreate inputs each time since backward modifies them)
            java.util.List<Value> trainInputs = java.util.Arrays.asList(
                new Value(1.5, "x"),
                new Value(2.0, "y"),
                new Value(-0.5, "z")
            );
            
            Value prediction = neuron.forward(trainInputs);
            
            // Compute loss: (prediction - target)²
            Value target = new Value(targetValue, "target");
            Value diff = prediction.sub(target);
            Value loss = diff.pow(2);
            
            // Backward pass
            loss.backward();
            
            // Gradient descent update: param = param - learning_rate * gradient
            for (Value param : neuron.parameters()) {
                double updatedValue = param.getData() - learningRate * param.getGrad();
                param.setData(updatedValue);
            }
            
            // Print progress
            if (iter % 5 == 0 || iter == iterations - 1) {
                System.out.printf("  Iteration %2d: output=%.4f, loss=%.6f%n", 
                    iter, prediction.getData(), loss.getData());
            }
        }
        
        // Final result
        System.out.println("\n--- Training Complete ---");
        System.out.println("\nUpdated weights:");
        for (int i = 0; i < neuron.getWeights().size(); i++) {
            System.out.printf("  w%d = %.4f%n", i, neuron.getWeights().get(i).getData());
        }
        System.out.printf("Bias: b = %.4f%n", neuron.getBias().getData());
        
        // Final forward pass
        java.util.List<Value> finalInputs = java.util.Arrays.asList(
            new Value(1.5, "x"),
            new Value(2.0, "y"),
            new Value(-0.5, "z")
        );
        Value finalOutput = neuron.forward(finalInputs);
        
        System.out.printf("%nFinal output: %.4f%n", finalOutput.getData());
        System.out.printf("Target value: %.4f%n", targetValue);
        System.out.printf("Final error: %.6f%n", Math.abs(targetValue - finalOutput.getData()));
        System.out.printf("%n✓ The neuron learned to produce output closer to the target!%n");
        
        // Part 3: Different activation functions
        System.out.println("\n--- Part 3: Different Activation Functions ---");
        
        Neuron tanhNeuron = new Neuron(2, ActivationType.TANH, new java.util.Random(123L));
        Neuron reluNeuron = new Neuron(2, ActivationType.RELU, new java.util.Random(123L));
        Neuron linearNeuron = new Neuron(2, ActivationType.LINEAR, new java.util.Random(123L));
        
        java.util.List<Value> testInputs = java.util.Arrays.asList(
            new Value(0.5, "x"),
            new Value(-0.3, "y")
        );
        
        System.out.println("Same inputs to different neurons:");
        System.out.println("  x = 0.5, y = -0.3");
        System.out.println("\nOutputs:");
        System.out.printf("  tanh neuron:   %.4f (range: [-1, 1])%n", 
            tanhNeuron.forward(java.util.Arrays.asList(
                new Value(0.5, "x"), new Value(-0.3, "y")
            )).getData());
        System.out.printf("  ReLU neuron:   %.4f (range: [0, ∞))%n", 
            reluNeuron.forward(java.util.Arrays.asList(
                new Value(0.5, "x"), new Value(-0.3, "y")
            )).getData());
        System.out.printf("  Linear neuron: %.4f (range: (-∞, ∞))%n", 
            linearNeuron.forward(testInputs).getData());
        
        System.out.println("\n✓ Neuron demonstration complete!");
        System.out.println("\nKey Takeaways:");
        System.out.println("  • Neurons combine inputs with learned weights");
        System.out.println("  • Gradient descent adjusts weights to minimize error");
        System.out.println("  • Activation functions introduce non-linearity");
        System.out.println("  • Different activations suit different problems");
    }
    
    /**
     * Demonstrates the Layer class with detailed explanations.
     */
    private static void demonstrateLayer() {
        System.out.println("\n╔════════════════════════════════════════════════════════╗");
        System.out.println("║           LAYER CLASS DEMONSTRATION                    ║");
        System.out.println("╚════════════════════════════════════════════════════════╝");
        
        // Part 1: Creating a layer
        System.out.println("\n--- Part 1: Creating a Layer ---");
        System.out.println("A layer contains multiple neurons that process the same inputs");
        
        Layer layer = new Layer(3, 4, new java.util.Random(42L)); // 3 inputs → 4 outputs
        
        System.out.println("Created: " + layer);
        System.out.println("  Input size: 3");
        System.out.println("  Output size: 4 (4 neurons)");
        System.out.println("  Parameters: " + layer.parameters().size());
        System.out.println("  (Each neuron has 3 weights + 1 bias = 4 params)");
        System.out.println("  (Total: 4 neurons × 4 params = 16 parameters)");
        
        // Part 2: Forward pass through layer
        System.out.println("\n--- Part 2: Layer Forward Pass ---");
        java.util.List<Value> inputs = java.util.Arrays.asList(
            new Value(1.0, "x1"),
            new Value(2.0, "x2"),
            new Value(3.0, "x3")
        );
        
        System.out.println("Inputs: [1.0, 2.0, 3.0]");
        java.util.List<Value> outputs = layer.forward(inputs);
        
        System.out.println("Outputs (" + outputs.size() + " values, one per neuron):");
        for (int i = 0; i < outputs.size(); i++) {
            System.out.printf("  Neuron %d → %.4f%n", i+1, outputs.get(i).getData());
        }
        
        System.out.println("\nEach neuron learns different features from the same inputs!");
        
        // Part 3: Training a layer
        System.out.println("\n--- Part 3: Training a Layer ---");
        System.out.println("Goal: Make all outputs closer to target values");
        
        java.util.List<Double> targets = java.util.Arrays.asList(0.5, 0.8, -0.3, 0.2);
        
        System.out.println("Current outputs: " + java.util.stream.IntStream.range(0, 4)
            .mapToObj(i -> String.format("%.3f", outputs.get(i).getData()))
            .collect(java.util.stream.Collectors.joining(", ")));
        System.out.println("Target outputs:  " + java.util.stream.IntStream.range(0, 4)
            .mapToObj(i -> String.format("%.3f", targets.get(i)))
            .collect(java.util.stream.Collectors.joining(", ")));
        
        // Compute loss
        Value loss = new Value(0.0, "zero");
        for (int i = 0; i < outputs.size(); i++) {
            Value target = new Value(targets.get(i), "target" + i);
            Value diff = outputs.get(i).sub(target);
            loss = loss.add(diff.pow(2));
        }
        
        System.out.printf("Initial loss (MSE): %.6f%n", loss.getData());
        
        // Training loop
        int iterations = 10;
        double learningRate = 0.1;
        
        for (int iter = 0; iter < iterations; iter++) {
            // Forward pass
            java.util.List<Value> trainInputs = java.util.Arrays.asList(
                new Value(1.0, "x1"),
                new Value(2.0, "x2"),
                new Value(3.0, "x3")
            );
            java.util.List<Value> predictions = layer.forward(trainInputs);
            
            // Compute loss
            Value currentLoss = new Value(0.0, "zero");
            for (int i = 0; i < predictions.size(); i++) {
                Value target = new Value(targets.get(i), "target" + i);
                Value diff = predictions.get(i).sub(target);
                currentLoss = currentLoss.add(diff.pow(2));
            }
            
            // Backward pass
            currentLoss.backward();
            
            // Update parameters
            for (Value param : layer.parameters()) {
                param.setData(param.getData() - learningRate * param.getGrad());
            }
            
            if (iter % 3 == 0 || iter == iterations - 1) {
                System.out.printf("  Iteration %2d: loss=%.6f%n", iter, currentLoss.getData());
            }
        }
        
        System.out.println("\n✓ Layer learned to produce outputs closer to targets!");
    }
    
    /**
     * Demonstrates the MLP class with detailed explanations.
     */
    private static void demonstrateMLP() {
        System.out.println("\n╔════════════════════════════════════════════════════════╗");
        System.out.println("║           MLP (NEURAL NETWORK) DEMONSTRATION           ║");
        System.out.println("╚════════════════════════════════════════════════════════╝");
        
        // Part 1: Creating an MLP
        System.out.println("\n--- Part 1: Creating a Neural Network ---");
        System.out.println("MLP = Multi-Layer Perceptron (feedforward neural network)");
        
        MLP mlp = new MLP(java.util.Arrays.asList(3, 4, 4, 1), new java.util.Random(42L));
        
        System.out.println("\nCreated: " + mlp);
        System.out.println("\nArchitecture:");
        System.out.println("  Input Layer:    3 features");
        System.out.println("  Hidden Layer 1: 4 neurons");
        System.out.println("  Hidden Layer 2: 4 neurons");
        System.out.println("  Output Layer:   1 neuron");
        System.out.println("\nTotal parameters: " + mlp.parameters().size());
        System.out.println("  Layer 1: 3→4 = 4×(3+1) = 16 params");
        System.out.println("  Layer 2: 4→4 = 4×(4+1) = 20 params");
        System.out.println("  Layer 3: 4→1 = 1×(4+1) = 5 params");
        System.out.println("  Total: 16 + 20 + 5 = 41 parameters");
        
        // Part 2: Forward pass
        System.out.println("\n--- Part 2: Forward Pass Through Network ---");
        java.util.List<Value> inputs = java.util.Arrays.asList(
            new Value(1.0, "x1"),
            new Value(2.0, "x2"),
            new Value(3.0, "x3")
        );
        
        System.out.println("Input: [1.0, 2.0, 3.0]");
        System.out.println("\nData flows through network:");
        System.out.println("  Input (3) → Layer1 → Hidden1 (4) → Layer2 → Hidden2 (4) → Layer3 → Output (1)");
        
        Value output = mlp.forward(inputs);
        System.out.printf("%nOutput: %.6f%n", output.getData());
        
        // Part 3: Training the network
        System.out.println("\n--- Part 3: Training the Neural Network ---");
        System.out.println("Task: Learn to approximate a simple function");
        System.out.println("Goal: For inputs [1, 2, 3], output should be 0.7");
        
        double targetValue = 0.7;
        int iterations = 20;
        double learningRate = 0.05;
        
        System.out.printf("%nInitial output: %.6f%n", output.getData());
        System.out.printf("Target output:  %.6f%n", targetValue);
        System.out.printf("Initial error:  %.6f%n%n", Math.abs(output.getData() - targetValue));
        
        System.out.println("Training...");
        
        for (int iter = 0; iter < iterations; iter++) {
            // Forward pass
            java.util.List<Value> trainInputs = java.util.Arrays.asList(
                new Value(1.0, "x1"),
                new Value(2.0, "x2"),
                new Value(3.0, "x3")
            );
            Value prediction = mlp.forward(trainInputs);
            
            // Compute loss: (prediction - target)²
            Value target = new Value(targetValue, "target");
            Value diff = prediction.sub(target);
            Value loss = diff.pow(2);
            
            // Backward pass
            loss.backward();
            
            // Gradient descent
            for (Value param : mlp.parameters()) {
                param.setData(param.getData() - learningRate * param.getGrad());
            }
            
            if (iter % 5 == 0 || iter == iterations - 1) {
                System.out.printf("  Iteration %2d: output=%.6f, loss=%.8f%n", 
                    iter, prediction.getData(), loss.getData());
            }
        }
        
        // Final evaluation
        System.out.println("\n--- Training Complete ---");
        java.util.List<Value> finalInputs = java.util.Arrays.asList(
            new Value(1.0, "x1"),
            new Value(2.0, "x2"),
            new Value(3.0, "x3")
        );
        Value finalOutput = mlp.forward(finalInputs);
        
        System.out.printf("%nFinal output:   %.6f%n", finalOutput.getData());
        System.out.printf("Target output:  %.6f%n", targetValue);
        System.out.printf("Final error:    %.8f%n", Math.abs(finalOutput.getData() - targetValue));
        
        System.out.println("\n✓ Neural network learned to approximate the function!");
        
        // Part 4: Network insights
        System.out.println("\n--- Part 4: What the Network Learned ---");
        System.out.println("The network discovered a function through training:");
        System.out.println("  • Layer 1 extracted initial features from inputs");
        System.out.println("  • Layer 2 combined those features into higher-level patterns");
        System.out.println("  • Layer 3 made final prediction based on learned patterns");
        System.out.println("\nThis is how neural networks learn complex relationships!");
        System.out.println("With more data and layers, they can learn very complex functions.");
        
        System.out.println("\n═══════════════════════════════════════════════════════");
        System.out.println("            ✓ ALL DEMONSTRATIONS COMPLETE!");
        System.out.println("═══════════════════════════════════════════════════════");
        System.out.println("\nYou now have a complete autograd engine with:");
        System.out.println("  ✓ Value - Automatic differentiation");
        System.out.println("  ✓ Neuron - Single computational unit");
        System.out.println("  ✓ Layer - Collection of neurons");
        System.out.println("  ✓ MLP - Complete neural network");
        System.out.println("\nReady to build and train your own neural networks! 🚀");
    }
}

