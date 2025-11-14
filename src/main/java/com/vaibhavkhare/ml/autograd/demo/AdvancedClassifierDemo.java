package com.vaibhavkhare.ml.autograd.demo;

import com.vaibhavkhare.ml.autograd.DecisionBoundaryVisualizer;
import com.vaibhavkhare.ml.autograd.MLP;
import com.vaibhavkhare.ml.autograd.Value;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Advanced Binary Classification Demo - Enhanced Version
 * 
 * This is an ENHANCED version of the micrograd demo designed to be robust,
 * reproducible, and educational. It includes:
 * 1) Multiple datasets: moons (non‑linear) and circles (concentric rings)
 * 2) Multiple loss functions: SVM max‑margin AND binary cross‑entropy (BCE)
 * 3) Extra ops: sigmoid() and log() to support BCE
 * 4) Stability: gradient clipping, sane learning rates, and fixed seeds
 * 5) Visualizations: decision boundaries saved as PNGs
 * 
 * Default configuration (moons):
 * - Architecture: [2, 16, 16, 1] with tanh in hidden layers, linear output
 * - Labels: ±1 (required by hinge loss)
 * - Optimizer: SGD, epochs=100, learning rate=0.05
 * - Regularization: L2 with alpha=1e‑4
 * - Stability: light gradient clipping to avoid rare exploding steps
 * 
 * Expected outcome:
 * - Loss typically drops to near zero (hinge loss → 0, only L2 remains)
 * - Accuracy commonly reaches ~100% on moons and circles with these settings
 * - Decision boundary images are written to the project root
 * 
 * Relationship to the replica:
 * - For a replica that mirrors the micrograd notebook’s structure, see
 *   MicrogradReplicaDemo.java. This advanced demo adds features and stability
 *   so results are consistently strong across RNG sequences.
 * 
 * This demonstrates advanced machine learning techniques beyond the basic demo.
 */
@SuppressWarnings({"java:S106", "java:S1192", "java:S3457", "java:S3776"}) // Allow System.out, string literals, \n, and high complexity for demo
public class AdvancedClassifierDemo {
    
    // Note: This demo includes sigmoid() and log() methods for cross-entropy loss
    // These are NOT in the original micrograd demo but are useful extensions
    
    private static final String SEPARATOR = "════════════════════════════════════════════════════════";
    
    private AdvancedClassifierDemo() {
        // Private constructor to hide implicit public one
    }
    
    /**
     * Represents a single data point with features and label.
     */
    public static class DataPoint {
        final double[] features;
        final int label; // +1 or -1
        
        public DataPoint(double[] features, int label) {
            this.features = features;
            this.label = label;
        }
        
        public double[] getFeatures() {
            return features;
        }
        
        public int getLabel() {
            return label;
        }
    }
    
    /**
     * Generates a moon-shaped dataset for binary classification.
     * Similar to sklearn's make_moons dataset.
     * 
     * @param numSamples number of samples to generate
     * @param noise amount of noise to add (0.0 = no noise, 0.1 = 10% noise)
     * @param seed random seed for reproducibility
     * @return list of data points
     */
    public static List<DataPoint> makeMoons(int numSamples, double noise, long seed) {
        return makeMoons(numSamples, noise, new Random(seed));
    }

    /**
     * Overload that uses a provided RNG with random-angle sampling.
     */
    public static List<DataPoint> makeMoons(int numSamples, double noise, Random random) {
        List<DataPoint> dataset = new ArrayList<>();
        int samplesPerClass = numSamples / 2;

        for (int i = 0; i < samplesPerClass; i++) {
            double theta = random.nextDouble() * Math.PI;
            double x = Math.cos(theta) + noise * random.nextGaussian();
            double y = Math.sin(theta) + noise * random.nextGaussian();
            dataset.add(new DataPoint(new double[]{x, y}, 1));
        }

        for (int i = 0; i < samplesPerClass; i++) {
            double theta = random.nextDouble() * Math.PI;
            double x = 1.0 - Math.cos(theta) + noise * random.nextGaussian();
            double y = 0.5 - Math.sin(theta) + noise * random.nextGaussian();
            dataset.add(new DataPoint(new double[]{x, y}, -1));
        }

        return dataset;
    }
    
    /**
     * Generates a simple circular dataset for binary classification.
     * Inner circle is class +1, outer ring is class -1.
     * 
     * @param numSamples number of samples to generate
     * @param noise amount of noise to add
     * @param seed random seed for reproducibility
     * @return list of data points
     */
    public static List<DataPoint> makeCircles(int numSamples, double noise, long seed) {
        return makeCircles(numSamples, noise, new Random(seed));
    }

    /**
     * Overload that uses a provided RNG.
     */
    public static List<DataPoint> makeCircles(int numSamples, double noise, Random random) {
        List<DataPoint> dataset = new ArrayList<>();
        int samplesPerClass = numSamples / 2;

        for (int i = 0; i < samplesPerClass; i++) {
            double angle = 2 * Math.PI * random.nextDouble();
            double radius = Math.max(0.05, 0.3 + noise * 0.1 * random.nextGaussian());
            double x = radius * Math.cos(angle);
            double y = radius * Math.sin(angle);
            dataset.add(new DataPoint(new double[]{x, y}, 1));
        }

        for (int i = 0; i < samplesPerClass; i++) {
            double angle = 2 * Math.PI * random.nextDouble();
            double radius = Math.max(0.2, 0.8 + noise * 0.1 * random.nextGaussian());
            double x = radius * Math.cos(angle);
            double y = radius * Math.sin(angle);
            dataset.add(new DataPoint(new double[]{x, y}, -1));
        }

        return dataset;
    }
    
    /**
     * Computes the SVM "max-margin" loss for binary classification.
     * 
     * Loss = max(0, 1 - y_true * y_pred)
     * 
     * This is also known as hinge loss. It:
     * - Encourages correct predictions with margin > 1
     * - Penalizes incorrect predictions linearly
     * - Creates a "margin" around the decision boundary
     * 
     * @param predictions list of predicted values from the model
     * @param labels list of true labels (+1 or -1)
     * @return total loss value
     */
    public static Value svmLoss(List<Value> predictions, List<Integer> labels) {
        Value totalLoss = new Value(0.0, "zero");
        
        for (int i = 0; i < predictions.size(); i++) {
            Value pred = predictions.get(i);
            double label = labels.get(i);
            
            // Margin loss: max(0, 1 - label * prediction)
            // We want: label * prediction > 1 (correct with margin)
            Value labelValue = new Value(label, "label");
            Value margin = labelValue.mul(pred);  // label * pred
            Value loss = new Value(1.0, "one").sub(margin);  // 1 - label*pred
            
            // ReLU: max(0, loss) - only penalize if loss > 0
            Value relu = loss.relu();
            totalLoss = totalLoss.add(relu);
        }
        
        return totalLoss;
    }
    
    /**
     * Computes binary cross-entropy loss (alternative to SVM loss).
     * 
     * This is the standard loss for binary classification.
     * Note: Currently not used in the main demo, but provided as an alternative.
     * 
     * @param predictions list of predicted values from the model
     * @param labels list of true labels (+1 or -1)
     * @return total loss value
     */
    @SuppressWarnings("java:S125") // Keeping commented code for educational purposes
    public static Value binaryCrossEntropyLoss(List<Value> predictions, List<Integer> labels) {
        Value totalLoss = new Value(0.0, "zero");
        
        for (int i = 0; i < predictions.size(); i++) {
            Value pred = predictions.get(i);
            int label = labels.get(i);
            
            // Convert label from {-1, +1} to {0, 1}
            double target = (label + 1) / 2.0;  // -1 -> 0, +1 -> 1
            
            // Sigmoid: σ(x) = 1 / (1 + e^(-x))
            Value sigmoid = pred.sigmoid();
            
            // BCE: -[y*log(p) + (1-y)*log(1-p)]
            Value targetValue = new Value(target, "target");
            Value oneMinusTarget = new Value(1.0 - target, "1-target");
            
            // log(sigmoid)
            Value logPred = sigmoid.log();
            // log(1 - sigmoid)
            Value oneMinusSigmoid = new Value(1.0, "one").sub(sigmoid);
            Value logOneMinusPred = oneMinusSigmoid.log();
            
            // -[target * log(pred) + (1-target) * log(1-pred)]
            Value loss1 = targetValue.mul(logPred);
            Value loss2 = oneMinusTarget.mul(logOneMinusPred);
            Value loss = loss1.add(loss2).mul(new Value(-1.0, "neg"));
            
            totalLoss = totalLoss.add(loss);
        }
        
        return totalLoss;
    }
    
    /**
     * Computes accuracy on a dataset.
     * 
     * @param model the neural network model
     * @param dataset the dataset to evaluate
     * @return accuracy as a percentage (0-100)
     */
    public static double computeAccuracy(MLP model, List<DataPoint> dataset) {
        int correct = 0;
        
        for (DataPoint point : dataset) {
            List<Value> inputs = Arrays.asList(
                new Value(point.features[0], "x1"),
                new Value(point.features[1], "x2")
            );
            
            Value prediction = model.forward(inputs);
            int predictedLabel = prediction.getData() > 0 ? 1 : -1;
            
            if (predictedLabel == point.label) {
                correct++;
            }
        }
        
        return 100.0 * correct / dataset.size();
    }
    
    /**
     * Visualizes the decision boundary of the trained model.
     * Creates a graphical plot using XChart library.
     * 
     * @param model the trained neural network
     * @param dataset the dataset (for showing data points)
     * @param filename output filename (without extension)
     */
    public static void visualizeDecisionBoundary(MLP model, List<DataPoint> dataset, String filename) {
        System.out.println("\n╔═══════════════════════════════════════════════════════╗");
        System.out.println("║           DECISION BOUNDARY VISUALIZATION            ║");
        System.out.println("╚═══════════════════════════════════════════════════════╝\n");
        
        // Convert DataPoint to DecisionBoundaryVisualizer.DataPoint interface
        List<DecisionBoundaryVisualizer.DataPoint> visualizerPoints = new ArrayList<>();
        for (DataPoint point : dataset) {
            visualizerPoints.add(new DecisionBoundaryVisualizer.DataPoint() {
                @Override
                public double[] getFeatures() {
                    return point.getFeatures();
                }
                
                @Override
                public int getLabel() {
                    return point.getLabel();
                }
            });
        }
        
        // Use DecisionBoundaryVisualizer to create the plot
        DecisionBoundaryVisualizer.visualize(model, visualizerPoints, filename, 
            DecisionBoundaryVisualizer.DisplayMode.BOTH);
    }
    
    /**
     * Main training demo - replicates Andrej's micrograd demo.ipynb
     */
    public static void main(String[] args) {
        System.out.println("╔" + SEPARATOR + "╗");
        System.out.println("║   Advanced Binary Classification Demo                 ║");
        System.out.println("║   Enhanced Version with Extra Features                ║");
        System.out.println("╚" + SEPARATOR + "╝\n");
        System.out.println("Note: This is the ENHANCED demo with additional features.");
        System.out.println("For exact micrograd replica, run: MicrogradReplicaDemo\n");
        
        // === STEP 1: Generate Dataset ===
        System.out.println("=== Step 1: Generate Dataset ===\n");
        
        int numSamples = 100;
        double noise = 0.1;
        long seed = 42;
        
        System.out.println("Generating moon dataset:");
        System.out.println("  • Number of samples: " + numSamples);
        System.out.println("  • Noise level: " + noise);
        System.out.println("  • Classes: +1 (upper moon), -1 (lower moon)");
        System.out.println();
        
        Random rng = new Random(seed);
        List<DataPoint> dataset = makeMoons(numSamples, noise, rng);
        
        // Display sample data
        System.out.println("\nSample data points:");
        for (int i = 0; i < 5; i++) {
            DataPoint point = dataset.get(i);
            System.out.printf("  [%.3f, %.3f] → %+d\n", 
                point.features[0], point.features[1], point.label);
        }
        System.out.println("  ...");
        
        // === STEP 2: Build Neural Network ===
        System.out.println("\n=== Step 2: Build Neural Network ===\n");
        
        // Architecture: 2 inputs → 16 hidden → 16 hidden → 1 output
        // This matches Andrej's demo
        MLP model = new MLP(Arrays.asList(2, 16, 16, 1), new Random(seed));
        
        System.out.println("Created MLP: " + model);
        System.out.println("\nArchitecture:");
        System.out.println("  Input Layer:    2 features (x, y coordinates)");
        System.out.println("  Hidden Layer 1: 16 neurons (tanh activation)");
        System.out.println("  Hidden Layer 2: 16 neurons (tanh activation)");
        System.out.println("  Output Layer:   1 neuron (linear output)");
        System.out.println("\nTotal parameters: " + model.parameters().size());
        
        // === STEP 3: Training Loop ===
        System.out.println("\n=== Step 3: Training with SGD ===\n");
        
        int numEpochs = 100;
        double learningRate = 0.05;  // Adjusted learning rate for stable training
        
        System.out.println("Training configuration:");
        System.out.println("  • Epochs: " + numEpochs);
        System.out.println("  • Learning rate: " + learningRate);
        System.out.println("  • Loss function: SVM max-margin loss");
        System.out.println("  • Optimizer: SGD (batch gradient descent)");
        System.out.println();
        System.out.println("Training...");
        System.out.println();
        
        long startTime = System.currentTimeMillis();
        
        for (int epoch = 0; epoch < numEpochs; epoch++) {
            // === Forward pass ===
            List<Value> predictions = new ArrayList<>();
            List<Integer> labels = new ArrayList<>();
            
            for (DataPoint point : dataset) {
                List<Value> inputs = Arrays.asList(
                    new Value(point.features[0], "x1"),
                    new Value(point.features[1], "x2")
                );
                Value prediction = model.forward(inputs);
                predictions.add(prediction);
                labels.add(point.label);
            }
            
            // === Compute loss ===
            Value loss = svmLoss(predictions, labels);
            
            // Add L2 regularization (weight decay)
            double alpha = 1e-4;  // regularization strength
            Value regLoss = new Value(0.0, "zero");
            for (Value param : model.parameters()) {
                regLoss = regLoss.add(param.pow(2));
            }
            regLoss = regLoss.mul(new Value(alpha, "alpha"));
            Value totalLoss = loss.add(regLoss);
            
            // === Backward pass ===
            totalLoss.backward();
            
            // === Update parameters (SGD) ===
            // Apply gradient clipping for stability
            for (Value param : model.parameters()) {
                double grad = param.getGrad();
                // Clip gradients to prevent explosion
                grad = Math.clamp(grad, -10.0, 10.0);
                param.setData(param.getData() - learningRate * grad);
            }
            
            // === Print progress ===
            if (epoch % 10 == 0 || epoch == numEpochs - 1) {
                double accuracy = computeAccuracy(model, dataset);
                System.out.printf("Epoch %3d/%d - Loss: %.4f (data: %.4f, reg: %.4f) - Accuracy: %.2f%%\n",
                    epoch, numEpochs, totalLoss.getData(), loss.getData(), 
                    regLoss.getData(), accuracy);
            }
        }
        
        long endTime = System.currentTimeMillis();
        double trainTime = (endTime - startTime) / 1000.0;
        
        System.out.println("\n✓ Training complete in " + String.format("%.2f", trainTime) + " seconds");
        
        // === STEP 4: Evaluation ===
        System.out.println("\n=== Step 4: Model Evaluation ===\n");
        
        double finalAccuracy = computeAccuracy(model, dataset);
        System.out.printf("Final Accuracy: %.2f%%\n", finalAccuracy);
        
        // Show some predictions
        System.out.println("\nSample predictions:");
        for (int i = 0; i < 10; i++) {
            DataPoint point = dataset.get(i * 10);
            List<Value> inputs = Arrays.asList(
                new Value(point.features[0], "x1"),
                new Value(point.features[1], "x2")
            );
            Value prediction = model.forward(inputs);
            int predictedLabel = prediction.getData() > 0 ? 1 : -1;
            String status = predictedLabel == point.label ? "✓" : "✗";
            
            System.out.printf("  [%.3f, %.3f] → Pred: %+d (%.3f) | True: %+d  %s\n",
                point.features[0], point.features[1], 
                predictedLabel, prediction.getData(), point.label, status);
        }
        
        // === STEP 5: Visualize Decision Boundary ===
        System.out.println("\n=== Step 5: Decision Boundary ===");
        visualizeDecisionBoundary(model, dataset, "moons_decision_boundary");
        
        // === Summary ===
        System.out.println("\n╔" + SEPARATOR + "╗");
        System.out.println("║                 TRAINING SUMMARY                       ║");
        System.out.println("╚" + SEPARATOR + "╝\n");
        
        System.out.println("✓ Successfully trained a 2-layer neural network!");
        System.out.println("\nKey Results:");
        System.out.println("  • Dataset: Moon-shaped binary classification");
        System.out.println("  • Architecture: [2 → 16 → 16 → 1]");
        System.out.println("  • Training samples: " + numSamples);
        System.out.println("  • Final accuracy: " + String.format("%.2f%%", finalAccuracy));
        System.out.println("  • Training time: " + String.format("%.2f", trainTime) + " seconds");
        System.out.println("  • Parameters learned: " + model.parameters().size());
        
        System.out.println("\nWhat the model learned:");
        System.out.println("  • Layer 1: Extracted initial features from x,y coordinates");
        System.out.println("  • Layer 2: Combined features to recognize moon shapes");
        System.out.println("  • Output: Classified points into upper moon (+1) vs lower moon (-1)");
        
        System.out.println("\nThis demonstrates the complete ML pipeline:");
        System.out.println("  1. Data generation");
        System.out.println("  2. Model architecture design");
        System.out.println("  3. Loss function (SVM max-margin)");
        System.out.println("  4. Optimization (SGD with backpropagation)");
        System.out.println("  5. Evaluation and visualization");
        System.out.println();
        System.out.println("🚀 Ready to train more complex models!");
        
        // === BONUS: Try different dataset ===
        System.out.println();
        System.out.println(SEPARATOR);
        System.out.println();
        System.out.println("=== BONUS: Training on Circular Dataset ===");
        System.out.println();
        
        List<DataPoint> circleData = makeCircles(100, 0.1, new Random(seed));
        MLP circleModel = new MLP(Arrays.asList(2, 16, 16, 1), new Random(seed + 1));
        
        System.out.println("Training on circles dataset (inner circle vs outer ring)...");
        System.out.println();
        
        for (int epoch = 0; epoch < 100; epoch++) {
            List<Value> predictions = new ArrayList<>();
            List<Integer> labels = new ArrayList<>();
            
            for (DataPoint point : circleData) {
                List<Value> inputs = Arrays.asList(
                    new Value(point.features[0], "x1"),
                    new Value(point.features[1], "x2")
                );
                Value prediction = circleModel.forward(inputs);
                predictions.add(prediction);
                labels.add(point.label);
            }
            
            Value loss = svmLoss(predictions, labels);
            
            // Regularization
            Value regLoss = new Value(0.0, "zero");
            for (Value param : circleModel.parameters()) {
                regLoss = regLoss.add(param.pow(2));
            }
            regLoss = regLoss.mul(new Value(1e-4, "alpha"));
            Value totalLoss = loss.add(regLoss);
            
            totalLoss.backward();
            
            for (Value param : circleModel.parameters()) {
                double grad = param.getGrad();
                grad = Math.clamp(grad, -10.0, 10.0);
                param.setData(param.getData() - learningRate * grad);
            }
            
            if (epoch % 25 == 0 || epoch == 99) {
                double accuracy = computeAccuracy(circleModel, circleData);
                System.out.printf("Epoch %3d/100 - Loss: %.4f - Accuracy: %.2f%%\n",
                    epoch, totalLoss.getData(), accuracy);
            }
        }
        
        double circleAccuracy = computeAccuracy(circleModel, circleData);
        System.out.println();
        System.out.printf("Final Accuracy on Circles: %.2f%%%n", circleAccuracy);
        System.out.println();
        System.out.println("=== Circles Decision Boundary ===");
        visualizeDecisionBoundary(circleModel, circleData, "circles_decision_boundary");
        System.out.println();
        System.out.println("✓ Successfully trained on both datasets!");
        System.out.println("Neural networks can learn complex non-linear decision boundaries!");
    }
}

