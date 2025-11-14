package com.vaibhavkhare.ml.autograd;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test class to verify all activation functions work correctly
 * with proper forward and backward passes.
 */
class ActivationTest {

    private static final double EPSILON = 1e-10;

    @Test
    @DisplayName("Test tanh activation function")
    void testTanh() {
        Value x = new Value(0.5, "x");
        Value y = x.tanh();
        y.setLabel("tanh(x)");
        y.backward();

        double expectedOutput = Math.tanh(0.5);
        double expectedGradient = 1 - Math.pow(Math.tanh(0.5), 2);

        assertEquals(expectedOutput, y.getData(), EPSILON, "tanh(0.5) output");
        assertEquals(expectedGradient, x.getGrad(), EPSILON, "tanh gradient");
    }

    @Test
    @DisplayName("Test exp activation function")
    void testExp() {
        Value x = new Value(2.0, "x");
        Value y = x.exp();
        y.setLabel("exp(x)");
        y.backward();

        double expectedOutput = Math.exp(2.0);
        double expectedGradient = Math.exp(2.0);

        assertEquals(expectedOutput, y.getData(), EPSILON, "exp(2.0) output");
        assertEquals(expectedGradient, x.getGrad(), EPSILON, "exp gradient");
    }

    @Test
    @DisplayName("Test ReLU activation with positive input")
    void testReluPositive() {
        Value x = new Value(3.0, "x");
        Value y = x.relu();
        y.setLabel("ReLU(x)");
        y.backward();

        assertEquals(3.0, y.getData(), EPSILON, "ReLU(3.0) output");
        assertEquals(1.0, x.getGrad(), EPSILON, "ReLU gradient for positive input");
    }

    @Test
    @DisplayName("Test ReLU activation with negative input")
    void testReluNegative() {
        Value x = new Value(-2.0, "x");
        Value y = x.relu();
        y.setLabel("ReLU(x)");
        y.backward();

        assertEquals(0.0, y.getData(), EPSILON, "ReLU(-2.0) output");
        assertEquals(0.0, x.getGrad(), EPSILON, "ReLU gradient for negative input");
    }

    @Test
    @DisplayName("Test ReLU activation at zero")
    void testReluZero() {
        Value x = new Value(0.0, "x");
        Value y = x.relu();
        y.backward();

        assertEquals(0.0, y.getData(), EPSILON, "ReLU(0.0) output");
        assertEquals(0.0, x.getGrad(), EPSILON, "ReLU gradient at zero");
    }

    @Test
    @DisplayName("Test complex expression with multiple activations")
    void testComplexExpression() {
        // Test: tanh(x) + exp(y) + ReLU(z)
        Value x = new Value(1.0, "x");
        Value y = new Value(0.5, "y");
        Value z = new Value(-1.0, "z");

        Value t1 = x.tanh();
        t1.setLabel("tanh(x)");

        Value t2 = y.exp();
        t2.setLabel("exp(y)");

        Value t3 = z.relu();
        t3.setLabel("ReLU(z)");

        Value result = t1.add(t2).add(t3);
        result.setLabel("result");
        result.backward();

        // Verify outputs
        assertEquals(Math.tanh(1.0), t1.getData(), EPSILON, "tanh(1.0)");
        assertEquals(Math.exp(0.5), t2.getData(), EPSILON, "exp(0.5)");
        assertEquals(0.0, t3.getData(), EPSILON, "ReLU(-1.0)");

        double expectedResult = Math.tanh(1.0) + Math.exp(0.5);
        assertEquals(expectedResult, result.getData(), EPSILON, "Complex expression result");

        // Verify gradients
        double expectedGradX = 1 - Math.pow(Math.tanh(1.0), 2);
        assertEquals(expectedGradX, x.getGrad(), EPSILON, "Gradient dx");
        assertEquals(Math.exp(0.5), y.getGrad(), EPSILON, "Gradient dy");
        assertEquals(0.0, z.getGrad(), EPSILON, "Gradient dz (blocked by ReLU)");
    }

    @Test
    @DisplayName("Test tanh with zero input")
    void testTanhZero() {
        Value x = new Value(0.0, "x");
        Value y = x.tanh();
        y.backward();

        assertEquals(0.0, y.getData(), EPSILON, "tanh(0) output");
        assertEquals(1.0, x.getGrad(), EPSILON, "tanh gradient at zero");
    }

    @Test
    @DisplayName("Test tanh with large positive value")
    void testTanhLarge() {
        Value x = new Value(10.0, "x");
        Value y = x.tanh();
        y.backward();

        assertTrue(y.getData() > 0.999, "tanh(10) should be close to 1");
        assertTrue(x.getGrad() < 0.001, "tanh gradient should be near 0 for large values");
    }

    @Test
    @DisplayName("Test exp with zero")
    void testExpZero() {
        Value x = new Value(0.0, "x");
        Value y = x.exp();
        y.backward();

        assertEquals(1.0, y.getData(), EPSILON, "exp(0) should be 1");
        assertEquals(1.0, x.getGrad(), EPSILON, "exp gradient at zero");
    }

    @Test
    @DisplayName("Test activation functions are differentiable")
    void testDifferentiability() {
        // Test that multiple backward passes work correctly
        Value x = new Value(2.0, "x");
        Value y = x.tanh();
        
        // First backward pass
        y.backward();
        double grad1 = x.getGrad();
        
        // Reset and do second backward pass
        x.setGrad(0.0);
        y.backward();
        double grad2 = x.getGrad();
        
        assertEquals(grad1, grad2, EPSILON, "Gradients should be consistent across multiple backward passes");
    }
}
