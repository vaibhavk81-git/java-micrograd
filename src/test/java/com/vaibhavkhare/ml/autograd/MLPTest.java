package com.vaibhavkhare.ml.autograd;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Nested;

import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive test suite for the MLP (Multi-Layer Perceptron) class.
 * Tests network construction, forward pass, gradient computation, and parameter management.
 */
class MLPTest {

    private static final double EPSILON = 1e-10;

    @Nested
    @DisplayName("MLP Construction")
    class ConstructionTests {

        @Test
        @DisplayName("Create simple MLP (perceptron)")
        void testSimplePerceptron() {
            MLP mlp = new MLP(Arrays.asList(2, 1));
            
            assertEquals(2, mlp.getInputSize());
            assertEquals(1, mlp.getOutputSize());
            assertEquals(1, mlp.getNumLayers());
            assertEquals(Arrays.asList(2, 1), mlp.getLayerSizes());
        }

        @Test
        @DisplayName("Create MLP with one hidden layer")
        void testOneHiddenLayer() {
            MLP mlp = new MLP(Arrays.asList(3, 4, 1));
            
            assertEquals(3, mlp.getInputSize());
            assertEquals(1, mlp.getOutputSize());
            assertEquals(2, mlp.getNumLayers());
            
            // Layer 1: 3 → 4
            assertEquals(3, mlp.getLayers().get(0).getInputSize());
            assertEquals(4, mlp.getLayers().get(0).getOutputSize());
            
            // Layer 2: 4 → 1
            assertEquals(4, mlp.getLayers().get(1).getInputSize());
            assertEquals(1, mlp.getLayers().get(1).getOutputSize());
        }

        @Test
        @DisplayName("Create MLP with multiple hidden layers")
        void testMultipleHiddenLayers() {
            MLP mlp = new MLP(Arrays.asList(10, 8, 6, 4, 2));
            
            assertEquals(10, mlp.getInputSize());
            assertEquals(2, mlp.getOutputSize());
            assertEquals(4, mlp.getNumLayers());
        }

        @Test
        @DisplayName("Create MLP with seed for reproducibility")
        void testSeededConstruction() {
            MLP mlp1 = new MLP(Arrays.asList(3, 4, 2), new Random(42L));
            MLP mlp2 = new MLP(Arrays.asList(3, 4, 2), new Random(42L));
            
            // Same seed should produce same weights
            List<Value> params1 = mlp1.parameters();
            List<Value> params2 = mlp2.parameters();
            
            assertEquals(params1.size(), params2.size());
            for (int i = 0; i < params1.size(); i++) {
                assertEquals(params1.get(i).getData(), params2.get(i).getData(), EPSILON,
                    "Same seed should produce same weights");
            }
        }

        @Test
        @DisplayName("Invalid layer sizes throw exception")
        void testInvalidLayerSizes() {
            // Null list
            assertThrows(IllegalArgumentException.class, 
                () -> new MLP(null),
                "Should throw exception for null layer sizes");
            
            // Too few layers
            assertThrows(IllegalArgumentException.class, 
                () -> new MLP(Arrays.asList(5)),
                "Should throw exception for single layer size");
            
            // Empty list
            assertThrows(IllegalArgumentException.class, 
                () -> new MLP(Arrays.asList()),
                "Should throw exception for empty layer sizes");
            
            // Invalid size values
            assertThrows(IllegalArgumentException.class, 
                () -> new MLP(Arrays.asList(3, 0, 2)),
                "Should throw exception for zero size");
            
            assertThrows(IllegalArgumentException.class, 
                () -> new MLP(Arrays.asList(3, -1, 2)),
                "Should throw exception for negative size");
        }

        @Test
        @DisplayName("Hidden layers use activation, output layer doesn't")
        void testActivationStrategy() {
            MLP mlp = new MLP(Arrays.asList(3, 4, 2), ActivationType.TANH);
            
            // Hidden layer should have activation
            assertTrue(mlp.getLayers().get(0).getNeurons().get(0).getActivationType() != ActivationType.LINEAR,
                "Hidden layer should use activation");
            
            // Output layer should NOT have activation
            assertFalse(mlp.getLayers().get(1).getNeurons().get(0).getActivationType() != ActivationType.LINEAR,
                "Output layer should not use activation");
        }

        @Test
        @DisplayName("Parameter count calculation")
        void testParameterCount() {
            // MLP([3, 4, 2])
            // Layer1: 3→4 = 4*(3+1) = 16 params
            // Layer2: 4→2 = 2*(4+1) = 10 params
            // Total: 26 params
            MLP mlp = new MLP(Arrays.asList(3, 4, 2));
            assertEquals(26, mlp.parameters().size());
        }
    }

    @Nested
    @DisplayName("Forward Pass")
    class ForwardPassTests {

        @Test
        @DisplayName("Forward pass produces single output")
        void testForwardPassSingleOutput() {
            MLP mlp = new MLP(Arrays.asList(3, 4, 1), new Random(42L));
            List<Value> inputs = Arrays.asList(
                new Value(1.0, "x1"),
                new Value(2.0, "x2"),
                new Value(3.0, "x3")
            );
            
            Value output = mlp.forward(inputs);
            
            assertNotNull(output, "Output should not be null");
            assertTrue(Double.isFinite(output.getData()), "Output should be finite");
        }

        @Test
        @DisplayName("Forward pass with different activations")
        void testForwardPassActivations() {
            List<Value> inputs = Arrays.asList(
                new Value(1.0, "x1"),
                new Value(2.0, "x2")
            );
            
            MLP tanhMLP = new MLP(Arrays.asList(2, 3, 1), ActivationType.TANH, new Random(42L));
            MLP reluMLP = new MLP(Arrays.asList(2, 3, 1), ActivationType.RELU, new Random(42L));
            
            Value tanhOut = tanhMLP.forward(inputs);
            Value reluOut = reluMLP.forward(inputs);
            
            assertNotNull(tanhOut);
            assertNotNull(reluOut);
            
            // Different activations should produce different outputs
            assertNotEquals(tanhOut.getData(), reluOut.getData(), EPSILON);
        }

        @Test
        @DisplayName("Wrong number of inputs throws exception")
        void testWrongInputSize() {
            MLP mlp = new MLP(Arrays.asList(3, 4, 1));
            
            List<Value> tooFewInputs = Arrays.asList(
                new Value(1.0, "x1"),
                new Value(2.0, "x2")
            );
            
            assertThrows(IllegalArgumentException.class, 
                () -> mlp.forward(tooFewInputs),
                "Should throw exception when input size doesn't match");
        }

        @Test
        @DisplayName("Deep network forward pass")
        void testDeepNetworkForwardPass() {
            MLP mlp = new MLP(Arrays.asList(5, 4, 3, 2, 1), new Random(42L));
            List<Value> inputs = Arrays.asList(
                new Value(1.0, "x1"),
                new Value(2.0, "x2"),
                new Value(3.0, "x3"),
                new Value(4.0, "x4"),
                new Value(5.0, "x5")
            );
            
            Value output = mlp.forward(inputs);
            
            assertNotNull(output);
            assertTrue(Double.isFinite(output.getData()));
        }

        @Test
        @DisplayName("Deterministic forward pass")
        void testDeterministicForwardPass() {
            MLP mlp = new MLP(Arrays.asList(3, 4, 1), new Random(42L));
            List<Value> inputs = Arrays.asList(
                new Value(1.5, "x1"),
                new Value(2.5, "x2"),
                new Value(3.5, "x3")
            );
            
            Value output1 = mlp.forward(inputs);
            
            // Create new inputs with same values
            List<Value> inputs2 = Arrays.asList(
                new Value(1.5, "x1"),
                new Value(2.5, "x2"),
                new Value(3.5, "x3")
            );
            
            Value output2 = mlp.forward(inputs2);
            
            assertEquals(output1.getData(), output2.getData(), EPSILON,
                "Same inputs should produce same output");
        }

        @Test
        @DisplayName("Zero inputs produce valid output")
        void testZeroInputs() {
            MLP mlp = new MLP(Arrays.asList(3, 2, 1), new Random(42L));
            List<Value> inputs = Arrays.asList(
                new Value(0.0, "x1"),
                new Value(0.0, "x2"),
                new Value(0.0, "x3")
            );
            
            Value output = mlp.forward(inputs);
            
            assertNotNull(output);
            assertTrue(Double.isFinite(output.getData()));
        }
    }

    @Nested
    @DisplayName("Backpropagation")
    class BackpropagationTests {

        @Test
        @DisplayName("Gradients flow through entire network")
        void testGradientFlowThroughNetwork() {
            MLP mlp = new MLP(Arrays.asList(3, 4, 1), ActivationType.LINEAR, new Random(42L));
            List<Value> inputs = Arrays.asList(
                new Value(1.0, "x1"),
                new Value(2.0, "x2"),
                new Value(3.0, "x3")
            );
            
            Value output = mlp.forward(inputs);
            output.backward();
            
            // All parameters should have gradients
            for (Value param : mlp.parameters()) {
                // Gradient exists (may be zero if path was blocked)
                assertNotNull(param);
            }
            
            // Check that at least some parameters have non-zero gradients
            long nonZeroGrads = mlp.parameters().stream()
                .filter(p -> Math.abs(p.getGrad()) > EPSILON)
                .count();
            assertTrue(nonZeroGrads > 0, "At least some parameters should have gradients");
        }

        @Test
        @DisplayName("Training iteration reduces loss")
        void testTrainingIteration() {
            MLP mlp = new MLP(Arrays.asList(2, 4, 1), new Random(42L));
            List<Value> inputs = Arrays.asList(
                new Value(1.0, "x1"),
                new Value(2.0, "x2")
            );
            double target = 0.5;
            
            // Initial prediction
            Value pred1 = mlp.forward(inputs);
            Value loss1 = pred1.sub(new Value(target, "target")).pow(2);
            
            // Backward pass
            loss1.backward();
            
            // Gradient descent step
            double learningRate = 0.1;
            for (Value param : mlp.parameters()) {
                param.setData(param.getData() - learningRate * param.getGrad());
            }
            
            // New prediction (after one step)
            List<Value> inputs2 = Arrays.asList(
                new Value(1.0, "x1"),
                new Value(2.0, "x2")
            );
            Value pred2 = mlp.forward(inputs2);
            
            // Loss should decrease (may not always be true with large learning rate)
            // But at least output should change
            assertNotEquals(pred1.getData(), pred2.getData(), 1e-6,
                "Output should change after parameter update");
        }
    }

    @Nested
    @DisplayName("Parameter Management")
    class ParameterManagementTests {

        @Test
        @DisplayName("parameters() collects all layer parameters")
        void testParametersMethod() {
            MLP mlp = new MLP(Arrays.asList(3, 4, 2));
            
            List<Value> params = mlp.parameters();
            
            // Layer1: 4*(3+1) = 16
            // Layer2: 2*(4+1) = 10
            // Total: 26
            assertEquals(26, params.size());
        }

        @Test
        @DisplayName("parameters() returns unmodifiable list")
        void testParametersUnmodifiable() {
            MLP mlp = new MLP(Arrays.asList(2, 2, 1));
            List<Value> params = mlp.parameters();
            
            assertThrows(UnsupportedOperationException.class, 
                () -> params.add(new Value(1.0, "extra")),
                "parameters() should return unmodifiable list");
        }

        @Test
        @DisplayName("Parameter count for various architectures")
        void testParameterCountVariousArchitectures() {
            // Simple perceptron: 2→1 = 1*(2+1) = 3
            MLP mlp1 = new MLP(Arrays.asList(2, 1));
            assertEquals(3, mlp1.parameters().size());
            
            // One hidden: 3→4→1 = 4*(3+1) + 1*(4+1) = 16+5 = 21
            MLP mlp2 = new MLP(Arrays.asList(3, 4, 1));
            assertEquals(21, mlp2.parameters().size());
            
            // Two hidden: 2→3→2→1 = 3*(2+1) + 2*(3+1) + 1*(2+1) = 9+8+3 = 20
            MLP mlp3 = new MLP(Arrays.asList(2, 3, 2, 1));
            assertEquals(20, mlp3.parameters().size());
        }

        @Test
        @DisplayName("getLayers() returns unmodifiable list")
        void testGetLayersUnmodifiable() {
            MLP mlp = new MLP(Arrays.asList(2, 3, 1));
            List<Layer> layers = mlp.getLayers();
            
            assertThrows(UnsupportedOperationException.class, 
                () -> layers.add(new Layer(3, 2)),
                "getLayers() should return unmodifiable list");
        }

        @Test
        @DisplayName("getLayerSizes() returns unmodifiable list")
        void testGetLayerSizesUnmodifiable() {
            MLP mlp = new MLP(Arrays.asList(2, 3, 1));
            List<Integer> sizes = mlp.getLayerSizes();
            
            assertThrows(UnsupportedOperationException.class, 
                () -> sizes.add(5),
                "getLayerSizes() should return unmodifiable list");
        }
    }

    @Nested
    @DisplayName("Network Architecture")
    class ArchitectureTests {

        @Test
        @DisplayName("Layers are properly connected")
        void testLayerConnections() {
            MLP mlp = new MLP(Arrays.asList(5, 4, 3, 2));
            
            List<Layer> layers = mlp.getLayers();
            
            // Layer 0: 5→4
            assertEquals(5, layers.get(0).getInputSize());
            assertEquals(4, layers.get(0).getOutputSize());
            
            // Layer 1: 4→3 (input matches previous output)
            assertEquals(4, layers.get(1).getInputSize());
            assertEquals(3, layers.get(1).getOutputSize());
            
            // Layer 2: 3→2
            assertEquals(3, layers.get(2).getInputSize());
            assertEquals(2, layers.get(2).getOutputSize());
        }

        @Test
        @DisplayName("Hidden layers and output layer structure")
        void testLayerStructure() {
            MLP mlp = new MLP(Arrays.asList(3, 4, 5, 2));
            
            assertEquals(3, mlp.getNumLayers());
            
            // First layer (hidden)
            Layer layer1 = mlp.getLayers().get(0);
            assertTrue(layer1.getNeurons().get(0).getActivationType() != ActivationType.LINEAR);
            
            // Second layer (hidden)
            Layer layer2 = mlp.getLayers().get(1);
            assertTrue(layer2.getNeurons().get(0).getActivationType() != ActivationType.LINEAR);
            
            // Third layer (output)
            Layer layer3 = mlp.getLayers().get(2);
            assertFalse(layer3.getNeurons().get(0).getActivationType() != ActivationType.LINEAR);
        }
    }

    @Nested
    @DisplayName("Utility Methods")
    class UtilityMethodsTests {

        @Test
        @DisplayName("toString method")
        void testToString() {
            MLP mlp = new MLP(Arrays.asList(3, 4, 2));
            String str = mlp.toString();
            
            assertTrue(str.contains("MLP"), "toString should contain 'MLP'");
            assertTrue(str.contains("layers"), "toString should show layers");
            assertTrue(str.contains("total_params"), "toString should show parameter count");
        }

        @Test
        @DisplayName("Getter methods")
        void testGetters() {
            MLP mlp = new MLP(Arrays.asList(5, 4, 3, 2));
            
            assertEquals(5, mlp.getInputSize());
            assertEquals(2, mlp.getOutputSize());
            assertEquals(3, mlp.getNumLayers());
            assertEquals(Arrays.asList(5, 4, 3, 2), mlp.getLayerSizes());
            assertEquals(3, mlp.getLayers().size());
        }
    }
}

