package com.vaibhavkhare.ml.autograd;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Nested;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive test suite for the Value class autograd engine.
 * Tests arithmetic operations, gradient computation, and backpropagation.
 */
class ValueTest {

    private static final double EPSILON = 1e-10;
    private static final double LOOSE_EPSILON = 1e-6; // For operations with more numerical error

    @Nested
    @DisplayName("Basic Arithmetic Operations")
    class ArithmeticOperationsTests {

        @Test
        @DisplayName("Addition: forward pass")
        void testAdditionForward() {
            Value a = new Value(2.0, "a");
            Value b = new Value(3.0, "b");
            Value c = a.add(b);

            assertEquals(5.0, c.getData(), EPSILON, "2.0 + 3.0 should equal 5.0");
        }

        @Test
        @DisplayName("Addition: backward pass")
        void testAdditionBackward() {
            Value a = new Value(2.0, "a");
            Value b = new Value(3.0, "b");
            Value c = a.add(b);
            c.backward();

            assertEquals(1.0, a.getGrad(), EPSILON, "∂c/∂a should be 1.0");
            assertEquals(1.0, b.getGrad(), EPSILON, "∂c/∂b should be 1.0");
        }

        @Test
        @DisplayName("Multiplication: forward pass")
        void testMultiplicationForward() {
            Value a = new Value(2.0, "a");
            Value b = new Value(3.0, "b");
            Value c = a.mul(b);

            assertEquals(6.0, c.getData(), EPSILON, "2.0 * 3.0 should equal 6.0");
        }

        @Test
        @DisplayName("Multiplication: backward pass")
        void testMultiplicationBackward() {
            Value a = new Value(2.0, "a");
            Value b = new Value(3.0, "b");
            Value c = a.mul(b);
            c.backward();

            assertEquals(3.0, a.getGrad(), EPSILON, "∂c/∂a should be b = 3.0");
            assertEquals(2.0, b.getGrad(), EPSILON, "∂c/∂b should be a = 2.0");
        }

        @Test
        @DisplayName("Subtraction: forward pass")
        void testSubtractionForward() {
            Value a = new Value(5.0, "a");
            Value b = new Value(3.0, "b");
            Value c = a.sub(b);

            assertEquals(2.0, c.getData(), EPSILON, "5.0 - 3.0 should equal 2.0");
        }

        @Test
        @DisplayName("Subtraction: backward pass")
        void testSubtractionBackward() {
            Value a = new Value(5.0, "a");
            Value b = new Value(3.0, "b");
            Value c = a.sub(b);
            c.backward();

            assertEquals(1.0, a.getGrad(), EPSILON, "∂c/∂a should be 1.0");
            assertEquals(-1.0, b.getGrad(), EPSILON, "∂c/∂b should be -1.0");
        }

        @Test
        @DisplayName("Division: forward pass")
        void testDivisionForward() {
            Value a = new Value(6.0, "a");
            Value b = new Value(3.0, "b");
            Value c = a.div(b);

            assertEquals(2.0, c.getData(), EPSILON, "6.0 / 3.0 should equal 2.0");
        }

        @Test
        @DisplayName("Division: backward pass")
        void testDivisionBackward() {
            Value a = new Value(6.0, "a");
            Value b = new Value(3.0, "b");
            Value c = a.div(b);
            c.backward();

            assertEquals(1.0/3.0, a.getGrad(), EPSILON, "∂c/∂a should be 1/b = 1/3");
            // ∂(a/b)/∂b = -a/b² = -6/9 = -2/3
            assertEquals(-2.0/3.0, b.getGrad(), EPSILON, "∂c/∂b should be -a/b² = -2/3");
        }

        @Test
        @DisplayName("Power: forward pass")
        void testPowerForward() {
            Value a = new Value(3.0, "a");
            Value b = a.pow(2);

            assertEquals(9.0, b.getData(), EPSILON, "3.0^2 should equal 9.0");
        }

        @Test
        @DisplayName("Power: backward pass")
        void testPowerBackward() {
            Value a = new Value(3.0, "a");
            Value b = a.pow(2);
            b.backward();

            // ∂(a²)/∂a = 2*a = 2*3 = 6
            assertEquals(6.0, a.getGrad(), EPSILON, "∂(a²)/∂a should be 2*a = 6.0");
        }

        @Test
        @DisplayName("Power with fractional exponent")
        void testPowerFractional() {
            Value a = new Value(4.0, "a");
            Value b = a.pow(0.5); // Square root
            b.backward();

            assertEquals(2.0, b.getData(), EPSILON, "4.0^0.5 should equal 2.0");
            // ∂(a^0.5)/∂a = 0.5 * a^(-0.5) = 0.5 / 2 = 0.25
            assertEquals(0.25, a.getGrad(), EPSILON, "∂(√a)/∂a should be 0.25");
        }

        @Test
        @DisplayName("Negation: forward pass")
        void testNegationForward() {
            Value a = new Value(5.0, "a");
            Value b = a.neg();

            assertEquals(-5.0, b.getData(), EPSILON, "-5.0 should equal -5.0");
        }

        @Test
        @DisplayName("Negation: backward pass")
        void testNegationBackward() {
            Value a = new Value(5.0, "a");
            Value b = a.neg();
            b.backward();

            assertEquals(-1.0, a.getGrad(), EPSILON, "∂(-a)/∂a should be -1.0");
        }
    }

    @Nested
    @DisplayName("Scalar Operations")
    class ScalarOperationsTests {

        @Test
        @DisplayName("Add scalar")
        void testAddScalar() {
            Value a = new Value(2.0, "a");
            Value b = a.add(3.0);
            b.backward();

            assertEquals(5.0, b.getData(), EPSILON);
            assertEquals(1.0, a.getGrad(), EPSILON);
        }

        @Test
        @DisplayName("Multiply by scalar")
        void testMultiplyScalar() {
            Value a = new Value(2.0, "a");
            Value b = a.mul(3.0);
            b.backward();

            assertEquals(6.0, b.getData(), EPSILON);
            assertEquals(3.0, a.getGrad(), EPSILON);
        }

        @Test
        @DisplayName("Subtract scalar")
        void testSubtractScalar() {
            Value a = new Value(5.0, "a");
            Value b = a.sub(2.0);
            b.backward();

            assertEquals(3.0, b.getData(), EPSILON);
            assertEquals(1.0, a.getGrad(), EPSILON);
        }

        @Test
        @DisplayName("Divide by scalar")
        void testDivideScalar() {
            Value a = new Value(6.0, "a");
            Value b = a.div(2.0);
            b.backward();

            assertEquals(3.0, b.getData(), EPSILON);
            assertEquals(0.5, a.getGrad(), EPSILON);
        }
    }

    @Nested
    @DisplayName("Right-Hand Operations")
    class RightHandOperationsTests {

        @Test
        @DisplayName("Right-hand subtraction: scalar - value")
        void testRightSubtraction() {
            Value a = new Value(3.0, "a");
            Value b = a.rsub(5.0); // 5.0 - 3.0
            b.backward();

            assertEquals(2.0, b.getData(), EPSILON, "5.0 - 3.0 should equal 2.0");
            assertEquals(-1.0, a.getGrad(), EPSILON, "∂(c-a)/∂a should be -1.0");
        }

        @Test
        @DisplayName("Right-hand division: scalar / value")
        void testRightDivision() {
            Value a = new Value(2.0, "a");
            Value b = a.rdiv(6.0); // 6.0 / 2.0
            b.backward();

            assertEquals(3.0, b.getData(), EPSILON, "6.0 / 2.0 should equal 3.0");
            // ∂(c/a)/∂a = -c/a² = -6/4 = -1.5
            assertEquals(-1.5, a.getGrad(), EPSILON, "∂(c/a)/∂a should be -c/a²");
        }
    }

    @Nested
    @DisplayName("Gradient Accumulation")
    class GradientAccumulationTests {

        @Test
        @DisplayName("Gradient accumulation through multiple paths")
        void testMultiplePathsAccumulation() {
            // Create a computation where x is used twice: f(x) = x + x
            Value x = new Value(3.0, "x");
            Value y = x.add(x);
            y.backward();

            // Gradient should be 2.0 (accumulated from both paths)
            assertEquals(2.0, x.getGrad(), EPSILON, "Gradient should accumulate from both uses");
        }

        @Test
        @DisplayName("Complex gradient accumulation: (x + x) * x")
        void testComplexAccumulation() {
            Value x = new Value(2.0, "x");
            Value a = x.add(x); // a = 2x
            Value b = a.mul(x);  // b = 2x * x = 2x²
            b.backward();

            // db/dx = 4x = 8.0
            assertEquals(8.0, x.getGrad(), EPSILON);
        }

        @Test
        @DisplayName("Diamond pattern gradient accumulation")
        void testDiamondPattern() {
            // Pattern: x → a → c
            //          x → b → c
            // where c = a + b, a = x * 2, b = x * 3
            Value x = new Value(1.0, "x");
            Value a = x.mul(2.0);
            Value b = x.mul(3.0);
            Value c = a.add(b);
            c.backward();

            // dc/dx through a: dc/da * da/dx = 1 * 2 = 2
            // dc/dx through b: dc/db * db/dx = 1 * 3 = 3
            // Total: 2 + 3 = 5
            assertEquals(5.0, x.getGrad(), EPSILON);
        }
    }

    @Nested
    @DisplayName("Complex Expressions")
    class ComplexExpressionTests {

        @Test
        @DisplayName("Micrograd-style expression: (x*y) + (x*y)")
        void testMicrogradExample() {
            Value x = new Value(2.0, "x");
            Value y = new Value(3.0, "y");
            Value z = x.mul(y).add(x.mul(y));
            z.backward();

            // z = xy + xy = 2xy
            assertEquals(12.0, z.getData(), EPSILON);
            // dz/dx = 2y = 6
            assertEquals(6.0, x.getGrad(), EPSILON);
            // dz/dy = 2x = 4
            assertEquals(4.0, y.getGrad(), EPSILON);
        }

        @Test
        @DisplayName("Complex polynomial: (x² + y²) * (x - y)")
        void testComplexPolynomial() {
            Value x = new Value(3.0, "x");
            Value y = new Value(2.0, "y");
            
            Value x2 = x.pow(2);    // x² = 9
            Value y2 = y.pow(2);    // y² = 4
            Value sum = x2.add(y2); // x² + y² = 13
            Value diff = x.sub(y);  // x - y = 1
            Value result = sum.mul(diff); // (x² + y²)(x - y) = 13
            
            result.backward();
            
            assertEquals(13.0, result.getData(), EPSILON);
            
            // Numerical gradient check
            double h = 1e-7;
            
            // For x:
            double f1 = ((3.0 + h) * (3.0 + h) + 4.0) * ((3.0 + h) - 2.0);
            double f2 = (9.0 + 4.0) * (3.0 - 2.0);
            double numericalGradX = (f1 - f2) / h;
            assertEquals(numericalGradX, x.getGrad(), LOOSE_EPSILON);
            
            // For y:
            f1 = (9.0 + (2.0 + h) * (2.0 + h)) * (3.0 - (2.0 + h));
            f2 = 13.0;
            double numericalGradY = (f1 - f2) / h;
            assertEquals(numericalGradY, y.getGrad(), LOOSE_EPSILON);
        }

        @Test
        @DisplayName("Deep computation chain")
        void testDeepChain() {
            Value x = new Value(2.0, "x");
            Value y = x.mul(2.0).add(1.0).mul(3.0).sub(2.0); // ((x*2 + 1)*3 - 2)
            y.backward();

            // y = (2x + 1)*3 - 2 = 6x + 3 - 2 = 6x + 1
            // dy/dx = 6
            assertEquals(13.0, y.getData(), EPSILON, "((2*2+1)*3-2) = 13");
            assertEquals(6.0, x.getGrad(), EPSILON, "Gradient through chain should be 6.0");
        }
    }

    @Nested
    @DisplayName("Backpropagation Correctness")
    class BackpropagationTests {

        @Test
        @DisplayName("Multiple backward passes reset gradients")
        void testMultipleBackwardPasses() {
            Value x = new Value(3.0, "x");
            Value y = x.mul(2.0);
            
            // First backward pass
            y.backward();
            assertEquals(2.0, x.getGrad(), EPSILON);
            
            // Second backward pass should reset gradients
            y.backward();
            assertEquals(2.0, x.getGrad(), EPSILON, "Gradient should be reset and recomputed");
        }

        @Test
        @DisplayName("Output gradient initialized to 1.0")
        void testOutputGradient() {
            Value x = new Value(5.0, "x");
            x.backward();
            
            assertEquals(1.0, x.getGrad(), EPSILON, "Output node gradient should be 1.0");
        }

        @Test
        @DisplayName("Sanity check: known gradients")
        void testSanityCheck() {
            // f(a,b) = a*b + a*a + b*b
            Value a = new Value(3.0, "a");
            Value b = new Value(4.0, "b");
            
            Value ab = a.mul(b);      // 12
            Value a2 = a.mul(a);      // 9
            Value b2 = b.mul(b);      // 16
            Value sum1 = ab.add(a2);  // 21
            Value sum2 = sum1.add(b2); // 37
            
            sum2.backward();
            
            assertEquals(37.0, sum2.getData(), EPSILON);
            // df/da = b + 2a = 4 + 6 = 10
            assertEquals(10.0, a.getGrad(), EPSILON);
            // df/db = a + 2b = 3 + 8 = 11
            assertEquals(11.0, b.getGrad(), EPSILON);
        }
    }

    @Nested
    @DisplayName("Edge Cases and Special Values")
    class EdgeCaseTests {

        @Test
        @DisplayName("Operations with zero")
        void testZeroOperations() {
            Value x = new Value(0.0, "x");
            Value y = new Value(5.0, "y");
            
            Value sum = x.add(y);
            Value product = x.mul(y);
            
            sum.backward();
            assertEquals(1.0, x.getGrad(), EPSILON);
            
            product.backward();
            assertEquals(5.0, x.getGrad(), EPSILON);
        }

        @Test
        @DisplayName("Multiplication by zero")
        void testMultiplicationByZero() {
            Value x = new Value(5.0, "x");
            Value zero = new Value(0.0, "zero");
            Value product = x.mul(zero);
            product.backward();
            
            assertEquals(0.0, product.getData(), EPSILON);
            assertEquals(0.0, x.getGrad(), EPSILON, "Gradient through zero multiplication is zero");
        }

        @Test
        @DisplayName("Negative numbers")
        void testNegativeNumbers() {
            Value x = new Value(-3.0, "x");
            Value y = new Value(-2.0, "y");
            Value product = x.mul(y);
            product.backward();
            
            assertEquals(6.0, product.getData(), EPSILON, "-3 * -2 = 6");
            assertEquals(-2.0, x.getGrad(), EPSILON);
            assertEquals(-3.0, y.getGrad(), EPSILON);
        }

        @Test
        @DisplayName("Power with zero exponent")
        void testPowerZero() {
            Value x = new Value(5.0, "x");
            Value y = x.pow(0);
            y.backward();
            
            assertEquals(1.0, y.getData(), EPSILON, "x^0 = 1");
            assertEquals(0.0, x.getGrad(), EPSILON, "d(x^0)/dx = 0");
        }

        @Test
        @DisplayName("Power with negative exponent")
        void testPowerNegative() {
            Value x = new Value(2.0, "x");
            Value y = x.pow(-2); // 1/x²
            y.backward();
            
            assertEquals(0.25, y.getData(), EPSILON, "2^(-2) = 0.25");
            // d(x^-2)/dx = -2*x^-3 = -2/8 = -0.25
            assertEquals(-0.25, x.getGrad(), EPSILON);
        }

        @Test
        @DisplayName("Large numbers")
        void testLargeNumbers() {
            Value x = new Value(1e6, "x");
            Value y = new Value(1e6, "y");
            Value sum = x.add(y);
            sum.backward();
            
            assertEquals(2e6, sum.getData(), EPSILON);
            assertEquals(1.0, x.getGrad(), EPSILON);
            assertEquals(1.0, y.getGrad(), EPSILON);
        }

        @Test
        @DisplayName("Small numbers")
        void testSmallNumbers() {
            Value x = new Value(1e-6, "x");
            Value y = new Value(1e-6, "y");
            Value sum = x.add(y);
            sum.backward();
            
            assertEquals(2e-6, sum.getData(), EPSILON);
            assertEquals(1.0, x.getGrad(), EPSILON);
            assertEquals(1.0, y.getGrad(), EPSILON);
        }
    }

    @Nested
    @DisplayName("Graph Structure")
    class GraphStructureTests {

        @Test
        @DisplayName("Parent tracking")
        void testParentTracking() {
            Value a = new Value(2.0, "a");
            Value b = new Value(3.0, "b");
            Value c = a.add(b);
            
            assertEquals(2, c.getParents().size(), "Addition should have 2 parents");
            assertTrue(c.getParents().contains(a), "Should contain first parent");
            assertTrue(c.getParents().contains(b), "Should contain second parent");
        }

        @Test
        @DisplayName("Operation tracking")
        void testOperationTracking() {
            Value a = new Value(2.0, "a");
            Value b = new Value(3.0, "b");
            
            assertEquals("", a.getOp(), "Leaf nodes should have empty op");
            
            Value c = a.add(b);
            assertEquals("+", c.getOp(), "Addition operation should be tracked");
            
            Value d = a.mul(b);
            assertEquals("*", d.getOp(), "Multiplication operation should be tracked");
            
            Value e = a.pow(2);
            assertEquals("pow", e.getOp(), "Power operation should be tracked");
        }

        @Test
        @DisplayName("Label tracking")
        void testLabelTracking() {
            Value a = new Value(2.0, "myLabel");
            assertEquals("myLabel", a.getLabel());
            
            a.setLabel("newLabel");
            assertEquals("newLabel", a.getLabel());
        }
    }

    @Nested
    @DisplayName("Utility Methods")
    class UtilityMethodsTests {

        @Test
        @DisplayName("zeroGrad with valid list")
        void testZeroGrad() {
            Value a = new Value(1.0, "a");
            Value b = new Value(2.0, "b");
            
            // Set some gradients
            a.setGrad(5.0);
            b.setGrad(10.0);
            
            // Zero them out
            Value.zeroGrad(java.util.Arrays.asList(a, b));
            
            assertEquals(0.0, a.getGrad(), EPSILON);
            assertEquals(0.0, b.getGrad(), EPSILON);
        }

        @Test
        @DisplayName("zeroGrad with null list")
        void testZeroGradNull() {
            // Should not throw exception
            assertDoesNotThrow(() -> Value.zeroGrad(null));
        }

        @Test
        @DisplayName("toString method")
        void testToString() {
            Value a = new Value(2.5, "a");
            a.setGrad(1.5);
            
            String str = a.toString();
            assertTrue(str.contains("data=2.5"), "Should contain data value");
            assertTrue(str.contains("grad=1.5"), "Should contain gradient value");
            assertTrue(str.contains("label='a'"), "Should contain label");
        }

        @Test
        @DisplayName("Data getter and setter")
        void testDataGetterSetter() {
            Value a = new Value(1.0, "a");
            assertEquals(1.0, a.getData(), EPSILON);
            
            a.setData(5.0);
            assertEquals(5.0, a.getData(), EPSILON);
        }

        @Test
        @DisplayName("Grad getter and setter")
        void testGradGetterSetter() {
            Value a = new Value(1.0, "a");
            assertEquals(0.0, a.getGrad(), EPSILON, "Initial gradient should be 0");
            
            a.setGrad(3.5);
            assertEquals(3.5, a.getGrad(), EPSILON);
        }
    }
}

