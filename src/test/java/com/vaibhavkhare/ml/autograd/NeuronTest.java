package com.vaibhavkhare.ml.autograd;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Nested;

import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive test suite for the Neuron class.
 * Tests neuron construction, forward pass, gradient computation, and parameter management.
 */
class NeuronTest {

    private static final double EPSILON = 1e-10;

    @Nested
    @DisplayName("Neuron Construction")
    class ConstructionTests {

        @Test
        @DisplayName("Create neuron with default activation (tanh)")
        void testDefaultConstruction() {
            Neuron neuron = new Neuron(3);
            
            assertEquals(3, neuron.getInputSize(), "Should have 3 inputs");
            assertEquals(4, neuron.parameters().size(), "Should have 3 weights + 1 bias = 4 parameters");
            assertTrue(neuron.getActivationType() != ActivationType.LINEAR, "Should use activation by default");
            assertEquals(ActivationType.TANH, neuron.getActivationType(), "Default activation should be tanh");
        }

        @Test
        @DisplayName("Create neuron with custom activation")
        void testCustomActivation() {
            Neuron neuronTanh = new Neuron(2, ActivationType.TANH);
            Neuron neuronRelu = new Neuron(2, ActivationType.RELU);
            Neuron neuronNone = new Neuron(2, ActivationType.LINEAR);
            
            assertEquals(ActivationType.TANH, neuronTanh.getActivationType());
            assertEquals(ActivationType.RELU, neuronRelu.getActivationType());
            assertEquals(ActivationType.LINEAR, neuronNone.getActivationType());
        }

        @Test
        @DisplayName("Create neuron with seed for reproducibility")
        void testSeededConstruction() {
            Neuron neuron1 = new Neuron(3, new Random(42L));
            Neuron neuron2 = new Neuron(3, new Random(42L));
            
            // Same seed should produce same weights
            for (int i = 0; i < neuron1.getWeights().size(); i++) {
                assertEquals(neuron1.getWeights().get(i).getData(), 
                           neuron2.getWeights().get(i).getData(), 
                           EPSILON, 
                           "Same seed should produce same weights");
            }
            
            assertEquals(neuron1.getBias().getData(), 
                       neuron2.getBias().getData(), 
                       EPSILON, 
                       "Same seed should produce same bias");
        }

        @Test
        @DisplayName("Weights initialized in range [-1, 1]")
        void testWeightInitializationRange() {
            Neuron neuron = new Neuron(10, new Random(42L));
            
            for (Value weight : neuron.getWeights()) {
                assertTrue(weight.getData() >= -1.0 && weight.getData() <= 1.0,
                    "Weight should be in range [-1, 1], got: " + weight.getData());
            }
            
            double biasValue = neuron.getBias().getData();
            assertTrue(biasValue >= -1.0 && biasValue <= 1.0,
                "Bias should be in range [-1, 1], got: " + biasValue);
        }

        @Test
        @DisplayName("Invalid input size throws exception")
        void testInvalidInputSize() {
            assertThrows(IllegalArgumentException.class, () -> new Neuron(0),
                "Should throw exception for 0 inputs");
            assertThrows(IllegalArgumentException.class, () -> new Neuron(-1),
                "Should throw exception for negative inputs");
        }

        @Test
        @DisplayName("Invalid activation type throws exception")
        void testInvalidActivationType() {
            assertThrows(IllegalArgumentException.class, 
                () -> new Neuron(2, ActivationType.fromString("invalid")),
                "Should throw exception for invalid activation type");
        }

        @Test
        @DisplayName("Single input neuron")
        void testSingleInputNeuron() {
            Neuron neuron = new Neuron(1);
            assertEquals(1, neuron.getInputSize());
            assertEquals(2, neuron.parameters().size(), "Should have 1 weight + 1 bias");
        }
    }

    @Nested
    @DisplayName("Forward Pass")
    class ForwardPassTests {

        @Test
        @DisplayName("Forward pass with tanh activation")
        void testForwardPassTanh() {
            // Create neuron with known weights for predictable behavior
            Neuron neuron = new Neuron(2, new Random(42L));
            
            List<Value> inputs = Arrays.asList(
                new Value(1.0, "x1"),
                new Value(2.0, "x2")
            );
            
            Value output = neuron.forward(inputs);
            
            assertNotNull(output, "Output should not be null");
            // Output should be in tanh range [-1, 1]
            assertTrue(output.getData() >= -1.0 && output.getData() <= 1.0,
                "Tanh output should be in range [-1, 1]");
        }

        @Test
        @DisplayName("Forward pass with ReLU activation")
        void testForwardPassRelu() {
            Neuron neuron = new Neuron(2, ActivationType.RELU, new Random(42L));
            
            List<Value> inputs = Arrays.asList(
                new Value(1.0, "x1"),
                new Value(2.0, "x2")
            );
            
            Value output = neuron.forward(inputs);
            
            assertNotNull(output, "Output should not be null");
            // ReLU output should be non-negative
            assertTrue(output.getData() >= 0.0, "ReLU output should be non-negative");
        }

        @Test
        @DisplayName("Forward pass without activation (linear)")
        void testForwardPassLinear() {
            Neuron neuron = new Neuron(2, ActivationType.LINEAR, new Random(42L));
            
            List<Value> inputs = Arrays.asList(
                new Value(1.0, "x1"),
                new Value(2.0, "x2")
            );
            
            Value output = neuron.forward(inputs);
            
            assertNotNull(output, "Output should not be null");
            // Linear output can be any value (no activation constraint)
        }

        @Test
        @DisplayName("Forward pass computes correct weighted sum")
        void testWeightedSumComputation() {
            // Create neuron with seed 0 and manually verify computation
            Neuron neuron = new Neuron(2, ActivationType.LINEAR, new Random(123L));
            
            List<Value> inputs = Arrays.asList(
                new Value(3.0, "x1"),
                new Value(4.0, "x2")
            );
            
            Value output = neuron.forward(inputs);
            
            // Manually compute expected output
            double expected = neuron.getWeights().get(0).getData() * 3.0 
                           + neuron.getWeights().get(1).getData() * 4.0 
                           + neuron.getBias().getData();
            
            assertEquals(expected, output.getData(), EPSILON,
                "Output should match manual weighted sum calculation");
        }

        @Test
        @DisplayName("Forward pass with zero inputs")
        void testForwardPassZeroInputs() {
            Neuron neuron = new Neuron(3, ActivationType.LINEAR, new Random(42L));
            
            List<Value> inputs = Arrays.asList(
                new Value(0.0, "x1"),
                new Value(0.0, "x2"),
                new Value(0.0, "x3")
            );
            
            Value output = neuron.forward(inputs);
            
            // With zero inputs, output should be just the bias
            assertEquals(neuron.getBias().getData(), output.getData(), EPSILON,
                "With zero inputs, output should equal bias");
        }

        @Test
        @DisplayName("Forward pass with negative inputs")
        void testForwardPassNegativeInputs() {
            Neuron neuron = new Neuron(2, new Random(42L));
            
            List<Value> inputs = Arrays.asList(
                new Value(-1.0, "x1"),
                new Value(-2.0, "x2")
            );
            
            Value output = neuron.forward(inputs);
            
            assertNotNull(output, "Should handle negative inputs");
        }

        @Test
        @DisplayName("Wrong number of inputs throws exception")
        void testWrongInputSize() {
            Neuron neuron = new Neuron(3);
            
            List<Value> tooFewInputs = Arrays.asList(
                new Value(1.0, "x1"),
                new Value(2.0, "x2")
            );
            
            assertThrows(IllegalArgumentException.class, 
                () -> neuron.forward(tooFewInputs),
                "Should throw exception when input size doesn't match");
            
            List<Value> tooManyInputs = Arrays.asList(
                new Value(1.0, "x1"),
                new Value(2.0, "x2"),
                new Value(3.0, "x3"),
                new Value(4.0, "x4")
            );
            
            assertThrows(IllegalArgumentException.class, 
                () -> neuron.forward(tooManyInputs),
                "Should throw exception when input size doesn't match");
        }

        @Test
        @DisplayName("Multiple forward passes with same inputs produce same output")
        void testDeterministicForwardPass() {
            Neuron neuron = new Neuron(2, new Random(42L));
            
            List<Value> inputs = Arrays.asList(
                new Value(1.5, "x1"),
                new Value(2.5, "x2")
            );
            
            Value output1 = neuron.forward(inputs);
            
            // Create new input Values with same data
            List<Value> inputs2 = Arrays.asList(
                new Value(1.5, "x1"),
                new Value(2.5, "x2")
            );
            
            Value output2 = neuron.forward(inputs2);
            
            assertEquals(output1.getData(), output2.getData(), EPSILON,
                "Same inputs should produce same output");
        }
    }

    @Nested
    @DisplayName("Backpropagation")
    class BackpropagationTests {

        @Test
        @DisplayName("Gradients flow through neuron")
        void testGradientFlow() {
            Neuron neuron = new Neuron(2, ActivationType.LINEAR, new Random(42L));
            
            List<Value> inputs = Arrays.asList(
                new Value(1.0, "x1"),
                new Value(2.0, "x2")
            );
            
            Value output = neuron.forward(inputs);
            output.backward();
            
            // All parameters should have gradients
            for (Value weight : neuron.getWeights()) {
                assertNotEquals(0.0, weight.getGrad(), 
                    "Weight should have gradient after backward pass (unless input was 0)");
            }
            
            // Bias gradient should always be non-zero (usually 1.0 or activation derivative)
            assertNotEquals(0.0, neuron.getBias().getGrad(), 
                "Bias should have gradient after backward pass");
            
            // Inputs should have gradients
            for (Value input : inputs) {
                assertNotEquals(0.0, input.getGrad(),
                    "Input should have gradient after backward pass (unless weight was 0)");
            }
        }

        @Test
        @DisplayName("Gradient computation for linear neuron")
        void testLinearNeuronGradients() {
            Neuron neuron = new Neuron(2, ActivationType.LINEAR, new Random(42L));
            
            List<Value> inputs = Arrays.asList(
                new Value(3.0, "x1"),
                new Value(4.0, "x2")
            );
            
            Value output = neuron.forward(inputs);
            output.backward();
            
            // For linear neuron: output = w1*x1 + w2*x2 + b
            // doutput/dw1 = x1, doutput/dw2 = x2, doutput/db = 1
            assertEquals(3.0, neuron.getWeights().get(0).getGrad(), EPSILON,
                "Weight gradient should equal input value for linear neuron");
            assertEquals(4.0, neuron.getWeights().get(1).getGrad(), EPSILON,
                "Weight gradient should equal input value for linear neuron");
            assertEquals(1.0, neuron.getBias().getGrad(), EPSILON,
                "Bias gradient should be 1.0 for linear neuron");
        }

        @Test
        @DisplayName("Input gradients match weights for linear neuron")
        void testInputGradients() {
            Neuron neuron = new Neuron(2, ActivationType.LINEAR, new Random(42L));
            
            List<Value> inputs = Arrays.asList(
                new Value(3.0, "x1"),
                new Value(4.0, "x2")
            );
            
            Value output = neuron.forward(inputs);
            output.backward();
            
            // For linear neuron: doutput/dx1 = w1, doutput/dx2 = w2
            assertEquals(neuron.getWeights().get(0).getData(), inputs.get(0).getGrad(), EPSILON,
                "Input gradient should equal weight for linear neuron");
            assertEquals(neuron.getWeights().get(1).getData(), inputs.get(1).getGrad(), EPSILON,
                "Input gradient should equal weight for linear neuron");
        }

        @Test
        @DisplayName("Activation function affects gradients")
        void testActivationGradients() {
            Neuron linearNeuron = new Neuron(2, ActivationType.LINEAR, new Random(42L));
            Neuron tanhNeuron = new Neuron(2, ActivationType.TANH, new Random(42L));
            
            List<Value> inputs1 = Arrays.asList(
                new Value(1.0, "x1"),
                new Value(2.0, "x2")
            );
            
            List<Value> inputs2 = Arrays.asList(
                new Value(1.0, "x1"),
                new Value(2.0, "x2")
            );
            
            Value output1 = linearNeuron.forward(inputs1);
            Value output2 = tanhNeuron.forward(inputs2);
            
            output1.backward();
            output2.backward();
            
            // Tanh activation should scale gradients differently than linear
            // (they should be different due to tanh's derivative)
            assertNotEquals(inputs1.get(0).getGrad(), inputs2.get(0).getGrad(),
                "Activation function should affect gradient computation");
        }
    }

    @Nested
    @DisplayName("Parameter Management")
    class ParameterManagementTests {

        @Test
        @DisplayName("parameters() returns all weights and bias")
        void testParametersMethod() {
            Neuron neuron = new Neuron(3);
            
            List<Value> params = neuron.parameters();
            
            assertEquals(4, params.size(), "Should have 3 weights + 1 bias");
            
            // First 3 should be weights
            for (int i = 0; i < 3; i++) {
                assertEquals(neuron.getWeights().get(i), params.get(i),
                    "First 3 parameters should be weights");
            }
            
            // Last should be bias
            assertEquals(neuron.getBias(), params.get(3), "Last parameter should be bias");
        }

        @Test
        @DisplayName("parameters() returns unmodifiable list")
        void testParametersUnmodifiable() {
            Neuron neuron = new Neuron(2);
            List<Value> params = neuron.parameters();
            
            assertThrows(UnsupportedOperationException.class, 
                () -> params.add(new Value(1.0, "extra")),
                "parameters() should return unmodifiable list");
        }

        @Test
        @DisplayName("getWeights() returns unmodifiable list")
        void testWeightsUnmodifiable() {
            Neuron neuron = new Neuron(2);
            List<Value> weights = neuron.getWeights();
            
            assertThrows(UnsupportedOperationException.class, 
                () -> weights.add(new Value(1.0, "extra")),
                "getWeights() should return unmodifiable list");
        }

        @Test
        @DisplayName("Parameter update simulation")
        void testParameterUpdate() {
            Neuron neuron = new Neuron(2, ActivationType.LINEAR, new Random(42L));
            
            // Store initial values
            double initialWeight0 = neuron.getWeights().get(0).getData();
            double initialWeight1 = neuron.getWeights().get(1).getData();
            double initialBias = neuron.getBias().getData();
            
            // Forward and backward pass
            List<Value> inputs = Arrays.asList(
                new Value(1.0, "x1"),
                new Value(2.0, "x2")
            );
            Value output = neuron.forward(inputs);
            output.backward();
            
            // Simulate gradient descent update
            double learningRate = 0.1;
            for (Value param : neuron.parameters()) {
                double newValue = param.getData() - learningRate * param.getGrad();
                param.setData(newValue);
            }
            
            // Parameters should have changed
            assertNotEquals(initialWeight0, neuron.getWeights().get(0).getData(),
                "Weight should change after gradient descent");
            assertNotEquals(initialWeight1, neuron.getWeights().get(1).getData(),
                "Weight should change after gradient descent");
            assertNotEquals(initialBias, neuron.getBias().getData(),
                "Bias should change after gradient descent");
        }
    }

    @Nested
    @DisplayName("Utility Methods")
    class UtilityMethodsTests {

        @Test
        @DisplayName("toString method")
        void testToString() {
            Neuron neuron = new Neuron(3);
            String str = neuron.toString();
            
            assertTrue(str.contains("Neuron"), "toString should contain 'Neuron'");
            assertTrue(str.contains("nin=3"), "toString should show input size");
            assertTrue(str.contains("activation"), "toString should mention activation");
        }

        @Test
        @DisplayName("getInputSize method")
        void testGetInputSize() {
            Neuron neuron1 = new Neuron(1);
            Neuron neuron5 = new Neuron(5);
            Neuron neuron10 = new Neuron(10);
            
            assertEquals(1, neuron1.getInputSize());
            assertEquals(5, neuron5.getInputSize());
            assertEquals(10, neuron10.getInputSize());
        }

        @Test
        @DisplayName("usesActivation method")
        void testUsesActivation() {
            Neuron withActivation = new Neuron(2, ActivationType.TANH);
            Neuron withoutActivation = new Neuron(2, ActivationType.LINEAR);
            
            assertTrue(withActivation.getActivationType() != ActivationType.LINEAR);
            assertFalse(withoutActivation.getActivationType() != ActivationType.LINEAR);
        }

        @Test
        @DisplayName("getActivationType method")
        void testGetActivationType() {
            Neuron tanhNeuron = new Neuron(2, ActivationType.TANH);
            Neuron reluNeuron = new Neuron(2, ActivationType.RELU);
            
            assertEquals(ActivationType.TANH, tanhNeuron.getActivationType());
            assertEquals(ActivationType.RELU, reluNeuron.getActivationType());
        }
    }
}

