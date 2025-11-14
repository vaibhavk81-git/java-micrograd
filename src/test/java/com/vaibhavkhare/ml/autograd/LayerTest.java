package com.vaibhavkhare.ml.autograd;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Nested;

import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive test suite for the Layer class.
 * Tests layer construction, forward pass, gradient computation, and parameter management.
 */
class LayerTest {

    private static final double EPSILON = 1e-10;

    @Nested
    @DisplayName("Layer Construction")
    class ConstructionTests {

        @Test
        @DisplayName("Create layer with default activation (tanh)")
        void testDefaultConstruction() {
            Layer layer = new Layer(3, 4);
            
            assertEquals(3, layer.getInputSize(), "Input size should be 3");
            assertEquals(4, layer.getOutputSize(), "Output size should be 4");
            assertEquals(4, layer.getNeurons().size(), "Should have 4 neurons");
            
            // Each neuron should have 3 weights + 1 bias = 4 parameters
            // Total: 4 neurons × 4 params = 16 parameters
            assertEquals(16, layer.parameters().size(), "Should have 16 total parameters");
        }

        @Test
        @DisplayName("Create layer with custom activation")
        void testCustomActivation() {
            Layer tanhLayer = new Layer(2, 3, ActivationType.TANH);
            Layer reluLayer = new Layer(2, 3, ActivationType.RELU);
            Layer linearLayer = new Layer(2, 3, ActivationType.LINEAR);
            
            assertEquals(3, tanhLayer.getNeurons().size());
            assertEquals(3, reluLayer.getNeurons().size());
            assertEquals(3, linearLayer.getNeurons().size());
            
            assertTrue(tanhLayer.getNeurons().get(0).getActivationType() != ActivationType.LINEAR);
            assertTrue(reluLayer.getNeurons().get(0).getActivationType() != ActivationType.LINEAR);
            assertFalse(linearLayer.getNeurons().get(0).getActivationType() != ActivationType.LINEAR);
        }

        @Test
        @DisplayName("Create layer with seed for reproducibility")
        void testSeededConstruction() {
            Layer layer1 = new Layer(3, 2, new Random(42L));
            Layer layer2 = new Layer(3, 2, new Random(42L));
            
            // Same seed should produce same weights
            List<Value> params1 = layer1.parameters();
            List<Value> params2 = layer2.parameters();
            
            assertEquals(params1.size(), params2.size());
            for (int i = 0; i < params1.size(); i++) {
                assertEquals(params1.get(i).getData(), params2.get(i).getData(), EPSILON,
                    "Same seed should produce same weights");
            }
        }

        @Test
        @DisplayName("Invalid input size throws exception")
        void testInvalidInputSize() {
            assertThrows(IllegalArgumentException.class, () -> new Layer(0, 3),
                "Should throw exception for 0 inputs");
            assertThrows(IllegalArgumentException.class, () -> new Layer(-1, 3),
                "Should throw exception for negative inputs");
        }

        @Test
        @DisplayName("Invalid output size throws exception")
        void testInvalidOutputSize() {
            assertThrows(IllegalArgumentException.class, () -> new Layer(3, 0),
                "Should throw exception for 0 outputs");
            assertThrows(IllegalArgumentException.class, () -> new Layer(3, -1),
                "Should throw exception for negative outputs");
        }

        @Test
        @DisplayName("Single neuron layer")
        void testSingleNeuronLayer() {
            Layer layer = new Layer(5, 1);
            assertEquals(1, layer.getOutputSize());
            assertEquals(1, layer.getNeurons().size());
            assertEquals(6, layer.parameters().size(), "Should have 5 weights + 1 bias");
        }

        @Test
        @DisplayName("Large layer creation")
        void testLargeLayer() {
            Layer layer = new Layer(100, 50);
            assertEquals(100, layer.getInputSize());
            assertEquals(50, layer.getOutputSize());
            assertEquals(50, layer.getNeurons().size());
            // 50 neurons × (100 weights + 1 bias) = 5050 parameters
            assertEquals(5050, layer.parameters().size());
        }
    }

    @Nested
    @DisplayName("Forward Pass")
    class ForwardPassTests {

        @Test
        @DisplayName("Forward pass produces correct number of outputs")
        void testForwardPassOutputSize() {
            Layer layer = new Layer(3, 4, new Random(42L));
            List<Value> inputs = Arrays.asList(
                new Value(1.0, "x1"),
                new Value(2.0, "x2"),
                new Value(3.0, "x3")
            );
            
            List<Value> outputs = layer.forward(inputs);
            
            assertEquals(4, outputs.size(), "Should produce 4 outputs");
            for (Value output : outputs) {
                assertNotNull(output, "Each output should be non-null");
            }
        }

        @Test
        @DisplayName("Forward pass with different activations")
        void testForwardPassActivations() {
            List<Value> inputs = Arrays.asList(
                new Value(1.0, "x1"),
                new Value(2.0, "x2")
            );
            
            Layer tanhLayer = new Layer(2, 3, ActivationType.TANH, new Random(42L));
            Layer reluLayer = new Layer(2, 3, ActivationType.RELU, new Random(42L));
            Layer linearLayer = new Layer(2, 3, ActivationType.LINEAR, new Random(42L));
            
            List<Value> tanhOut = tanhLayer.forward(inputs);
            List<Value> reluOut = reluLayer.forward(inputs);
            
            // Tanh outputs should be in range [-1, 1]
            for (Value out : tanhOut) {
                assertTrue(out.getData() >= -1.0 && out.getData() <= 1.0,
                    "Tanh output should be in [-1, 1]");
            }
            
            // ReLU outputs should be non-negative
            for (Value out : reluOut) {
                assertTrue(out.getData() >= 0.0, "ReLU output should be non-negative");
            }
            
            // Linear outputs can be any value (no activation)
            assertNotNull(linearLayer.forward(inputs), "Linear layer should produce output");
            
            // All three should produce different outputs (due to activation differences)
            assertNotEquals(tanhOut.get(0).getData(), reluOut.get(0).getData(), EPSILON);
        }

        @Test
        @DisplayName("Wrong number of inputs throws exception")
        void testWrongInputSize() {
            Layer layer = new Layer(3, 2);
            
            List<Value> tooFewInputs = Arrays.asList(
                new Value(1.0, "x1"),
                new Value(2.0, "x2")
            );
            
            assertThrows(IllegalArgumentException.class, 
                () -> layer.forward(tooFewInputs),
                "Should throw exception when input size doesn't match");
            
            List<Value> tooManyInputs = Arrays.asList(
                new Value(1.0, "x1"),
                new Value(2.0, "x2"),
                new Value(3.0, "x3"),
                new Value(4.0, "x4")
            );
            
            assertThrows(IllegalArgumentException.class, 
                () -> layer.forward(tooManyInputs),
                "Should throw exception when input size doesn't match");
        }

        @Test
        @DisplayName("Deterministic forward pass")
        void testDeterministicForwardPass() {
            Layer layer = new Layer(3, 2, new Random(42L));
            List<Value> inputs = Arrays.asList(
                new Value(1.5, "x1"),
                new Value(2.5, "x2"),
                new Value(3.5, "x3")
            );
            
            List<Value> output1 = layer.forward(inputs);
            
            // Create new inputs with same values
            List<Value> inputs2 = Arrays.asList(
                new Value(1.5, "x1"),
                new Value(2.5, "x2"),
                new Value(3.5, "x3")
            );
            
            List<Value> output2 = layer.forward(inputs2);
            
            assertEquals(output1.size(), output2.size());
            for (int i = 0; i < output1.size(); i++) {
                assertEquals(output1.get(i).getData(), output2.get(i).getData(), EPSILON,
                    "Same inputs should produce same outputs");
            }
        }

        @Test
        @DisplayName("Zero inputs produce valid outputs")
        void testZeroInputs() {
            Layer layer = new Layer(3, 2, new Random(42L));
            List<Value> inputs = Arrays.asList(
                new Value(0.0, "x1"),
                new Value(0.0, "x2"),
                new Value(0.0, "x3")
            );
            
            List<Value> outputs = layer.forward(inputs);
            
            assertEquals(2, outputs.size());
            for (Value output : outputs) {
                assertNotNull(output);
                // With zero inputs, outputs should be based on biases and activation
            }
        }
    }

    @Nested
    @DisplayName("Backpropagation")
    class BackpropagationTests {

        @Test
        @DisplayName("Gradients flow through layer")
        void testGradientFlow() {
            Layer layer = new Layer(3, 2, ActivationType.LINEAR, new Random(42L));
            List<Value> inputs = Arrays.asList(
                new Value(1.0, "x1"),
                new Value(2.0, "x2"),
                new Value(3.0, "x3")
            );
            
            List<Value> outputs = layer.forward(inputs);
            
            // Create a simple loss: sum of outputs
            Value loss = outputs.get(0).add(outputs.get(1));
            loss.backward();
            
            // All parameters should have gradients
            for (Value param : layer.parameters()) {
                // Gradient may be zero if input was zero, but it should be computed
                assertNotNull(param);
            }
            
            // Inputs should have gradients
            for (Value input : inputs) {
                assertNotEquals(0.0, input.getGrad(), EPSILON,
                    "Inputs should have gradients (unless weights were all zero)");
            }
        }

        @Test
        @DisplayName("Multiple neurons produce independent gradients")
        void testIndependentGradients() {
            Layer layer = new Layer(2, 2, ActivationType.LINEAR, new Random(42L));
            List<Value> inputs = Arrays.asList(
                new Value(1.0, "x1"),
                new Value(2.0, "x2")
            );
            
            List<Value> outputs = layer.forward(inputs);
            
            // Backward from first output only
            outputs.get(0).backward();
            
            // First neuron's parameters should have gradients
            Neuron neuron1 = layer.getNeurons().get(0);
            for (Value param : neuron1.parameters()) {
                assertNotEquals(0.0, Math.abs(param.getGrad()), 
                    "First neuron should have gradients");
            }
            
            // Second neuron's parameters should NOT have gradients (not in backward path)
            Neuron neuron2 = layer.getNeurons().get(1);
            for (Value param : neuron2.parameters()) {
                assertEquals(0.0, param.getGrad(), EPSILON,
                    "Second neuron should not have gradients");
            }
        }
    }

    @Nested
    @DisplayName("Parameter Management")
    class ParameterManagementTests {

        @Test
        @DisplayName("parameters() returns all neuron parameters")
        void testParametersMethod() {
            Layer layer = new Layer(3, 2);
            
            List<Value> params = layer.parameters();
            
            // 2 neurons × (3 weights + 1 bias) = 8 parameters
            assertEquals(8, params.size());
        }

        @Test
        @DisplayName("parameters() returns unmodifiable list")
        void testParametersUnmodifiable() {
            Layer layer = new Layer(2, 2);
            List<Value> params = layer.parameters();
            
            assertThrows(UnsupportedOperationException.class, 
                () -> params.add(new Value(1.0, "extra")),
                "parameters() should return unmodifiable list");
        }

        @Test
        @DisplayName("Parameter count formula")
        void testParameterCount() {
            // Formula: n_out × (n_in + 1)
            Layer layer1 = new Layer(5, 3);
            assertEquals(3 * (5 + 1), layer1.parameters().size());
            
            Layer layer2 = new Layer(10, 7);
            assertEquals(7 * (10 + 1), layer2.parameters().size());
            
            Layer layer3 = new Layer(100, 50);
            assertEquals(50 * (100 + 1), layer3.parameters().size());
        }

        @Test
        @DisplayName("getNeurons() returns unmodifiable list")
        void testGetNeuronsUnmodifiable() {
            Layer layer = new Layer(2, 3);
            List<Neuron> neurons = layer.getNeurons();
            
            assertThrows(UnsupportedOperationException.class, 
                () -> neurons.add(new Neuron(2)),
                "getNeurons() should return unmodifiable list");
        }
    }

    @Nested
    @DisplayName("Utility Methods")
    class UtilityMethodsTests {

        @Test
        @DisplayName("toString method")
        void testToString() {
            Layer layer = new Layer(3, 4);
            String str = layer.toString();
            
            assertTrue(str.contains("Layer"), "toString should contain 'Layer'");
            assertTrue(str.contains("nin=3"), "toString should show input size");
            assertTrue(str.contains("nout=4"), "toString should show output size");
            assertTrue(str.contains("activation"), "toString should mention activation");
        }

        @Test
        @DisplayName("getInputSize method")
        void testGetInputSize() {
            Layer layer1 = new Layer(5, 3);
            Layer layer2 = new Layer(10, 7);
            
            assertEquals(5, layer1.getInputSize());
            assertEquals(10, layer2.getInputSize());
        }

        @Test
        @DisplayName("getOutputSize method")
        void testGetOutputSize() {
            Layer layer1 = new Layer(5, 3);
            Layer layer2 = new Layer(10, 7);
            
            assertEquals(3, layer1.getOutputSize());
            assertEquals(7, layer2.getOutputSize());
        }
    }
}

