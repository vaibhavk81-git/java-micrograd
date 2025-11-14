package com.vaibhavkhare.ml.autograd.demo;

import com.vaibhavkhare.ml.autograd.DecisionBoundaryVisualizer;
import com.vaibhavkhare.ml.autograd.MLP;
import com.vaibhavkhare.ml.autograd.Value;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Micrograd Replica Demo - Closest Possible Match
 * 
 * This demo replicates the original micrograd notebook as closely as possible:
 * - Uses make_moons dataset
 * - MLP architecture: [2, 16, 16, 1]
 * - SVM max-margin loss
 * - Learning rate: 0.1 (stability tweak vs original LR=1.0)
 * - L2 regularization: 1e-4
 * - Simple SGD training
 * - Gentle gradient clipping (±10) for stability across RNGs
 * - Decision boundary visualization
 * 
 * NOTES
 * - This mirrors micrograd's functionality but uses small stability tweaks:
 *   lower LR and light gradient clipping to avoid rare exploding steps with hinge loss.
 *   This makes results consistent across Java RNG sequences while preserving behavior.
 * 
 * NOTE ON RESULTS:
 * Due to RNG and numeric differences, exact numbers will vary vs. Python.
 * This demo uses stabilized settings (LR=0.1, 200 steps, gentle grad clipping)
 * so it typically converges to high accuracy (≈85–100%) while mirroring the
 * original micrograd approach (hinge loss + L2 on [2,16,16,1]).
 * For a feature-rich/optimized version (extra losses, clipping, circles),
 * see AdvancedClassifierDemo.
 * 
 * Reference: https://github.com/karpathy/micrograd/blob/master/demo.ipynb
 */
@SuppressWarnings({"java:S106", "java:S1192", "java:S3457"}) // Allow System.out, string literals, \n for demo
public class MicrogradReplicaDemo {
    
    private static final String SEPARATOR = "════════════════════════════════════════════════════════";
    
    private MicrogradReplicaDemo() {
        // Private constructor to hide implicit public one
    }
    
    /**
     * Data point with 2D features and binary label.
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
     * Make moons dataset - exactly like sklearn.datasets.make_moons
     * Generates two interleaving half circles.
     * 
     * @param numSamples number of samples to generate
     * @param noise standard deviation of Gaussian noise
     * @param seed random seed for reproducibility
     * @return list of data points
     */
    public static List<DataPoint> makeMoons(int numSamples, double noise, long seed) {
        return makeMoons(numSamples, noise, new Random(seed));
    }

    /**
     * Overload that uses a provided RNG (closer to sklearn's random sampling).
     */
    public static List<DataPoint> makeMoons(int numSamples, double noise, Random random) {
        List<DataPoint> dataset = new ArrayList<>();
        int samplesPerClass = numSamples / 2;

        // First moon (class +1): random theta in [0, pi]
        for (int i = 0; i < samplesPerClass; i++) {
            double theta = random.nextDouble() * Math.PI;
            double x = Math.cos(theta) + noise * random.nextGaussian();
            double y = Math.sin(theta) + noise * random.nextGaussian();
            dataset.add(new DataPoint(new double[]{x, y}, 1));
        }

        // Second moon (class -1): shifted/scaled
        for (int i = 0; i < samplesPerClass; i++) {
            double theta = random.nextDouble() * Math.PI;
            double x = 1.0 - Math.cos(theta) + noise * random.nextGaussian();
            double y = 0.5 - Math.sin(theta) + noise * random.nextGaussian();
            dataset.add(new DataPoint(new double[]{x, y}, -1));
        }

        return dataset;
    }
    
    /**
     * SVM max-margin loss - exactly as in micrograd.
     * Loss = max(0, 1 - y_true * y_pred)
     * 
     * This is hinge loss - encourages correct predictions with margin > 1.
     * 
     * @param predictions model predictions
     * @param labels true labels (+1 or -1)
     * @return total loss
     */
    public static Value svmLoss(List<Value> predictions, List<Integer> labels) {
        Value totalLoss = new Value(0.0, "zero");
        
        for (int i = 0; i < predictions.size(); i++) {
            Value pred = predictions.get(i);
            double label = labels.get(i);
            
            // Compute: max(0, 1 - label * prediction)
            Value labelValue = new Value(label, "label");
            Value margin = labelValue.mul(pred);  // label * pred
            Value loss = new Value(1.0, "one").sub(margin);  // 1 - label*pred
            
            // Apply ReLU to get max(0, loss)
            Value relu = loss.relu();
            totalLoss = totalLoss.add(relu);
        }
        
        return totalLoss;
    }
    
    /**
     * Compute accuracy on dataset.
     * 
     * @param model the MLP model
     * @param dataset the dataset
     * @return accuracy percentage (0-100)
     */
    public static double accuracy(MLP model, List<DataPoint> dataset) {
        int correct = 0;
        
        for (DataPoint point : dataset) {
            List<Value> inputs = Arrays.asList(
                new Value(point.features[0], "x"),
                new Value(point.features[1], "y")
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
     * Visualize decision boundary using graphical plot.
     * Replicates the matplotlib visualization from the original demo.
     * 
     * @param model the trained model
     * @param dataset the dataset
     * @param filename output filename (without extension)
     */
    public static void visualizeDecisionBoundary(MLP model, List<DataPoint> dataset, String filename) {
        System.out.println("\nDecision Boundary:");
        System.out.println("Generating graphical visualization...\n");
        
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
     * Main method - exact replica of micrograd demo.ipynb
     */
    public static void main(String[] args) {
        System.out.println("╔" + SEPARATOR + "╗");
        System.out.println("║      Micrograd Demo - Exact Replica                    ║");
        System.out.println("║      https://github.com/karpathy/micrograd             ║");
        System.out.println("╚" + SEPARATOR + "╝\n");
        
        // Set random seed (micrograd uses np.random.seed(1337) and random.seed(1337))
        long seed = 1337;
        
        // Generate moon dataset (use numpy-like RNG)
        System.out.println("Generating make_moons dataset...");
        int numSamples = 100;
        double noise = 0.1;
        Random rngNumpy = new Random(seed);
        List<DataPoint> dataset = makeMoons(numSamples, noise, rngNumpy);
        System.out.println("Dataset: " + numSamples + " samples, 2 features, 2 classes\n");
        
        // Create MLP model: [2, 16, 16, 1]
        // This is the exact architecture from the micrograd demo
        System.out.println("Creating MLP model...");
        // Use a separate RNG for model init (closer to Python's random)
        Random rngPython = new Random(seed);
        MLP model = new MLP(Arrays.asList(2, 16, 16, 1), rngPython);
        System.out.println(model);
        System.out.println("Total parameters: " + model.parameters().size() + "\n");
        
        // Training loop - matching micrograd's approach
        System.out.println("Training...\n");
        int epochs = 200;            // Increased steps for stability
        double learningRate = 0.1;   // More stable across RNG differences
        double alpha = 1e-4;  // L2 regularization strength
        
        for (int epoch = 0; epoch < epochs; epoch++) {
            // Forward pass - batch prediction
            List<Value> predictions = new ArrayList<>();
            List<Integer> labels = new ArrayList<>();
            
            for (DataPoint point : dataset) {
                List<Value> inputs = Arrays.asList(
                    new Value(point.features[0], "x"),
                    new Value(point.features[1], "y")
                );
                Value prediction = model.forward(inputs);
                predictions.add(prediction);
                labels.add(point.label);
            }
            
            // Compute loss (SVM loss + L2 regularization)
            Value dataLoss = svmLoss(predictions, labels);
            
            // L2 regularization (apply to weights only, exclude biases)
            Value regLoss = new Value(0.0, "zero");
            for (Value param : model.parameters()) {
                String label = param.getLabel();
                if (label != null && label.startsWith("w")) {
                    regLoss = regLoss.add(param.pow(2));
                }
            }
            regLoss = regLoss.mul(new Value(alpha, "alpha"));
            
            Value totalLoss = dataLoss.add(regLoss);
            
            // Backward pass
            totalLoss.backward();
            
            // Update parameters (SGD) with gentle gradient clipping for stability
            for (Value param : model.parameters()) {
                double g = param.getGrad();
                // clip gradients to avoid occasional exploding steps with hinge loss
                if (g > 10.0) g = 10.0;
                if (g < -10.0) g = -10.0;
                param.setData(param.getData() - learningRate * g);
            }
            
            // Print progress
            if (epoch % 10 == 0 || epoch == epochs - 1) {
                double acc = accuracy(model, dataset);
                System.out.printf("step %d loss %.6f, accuracy %.1f%%\n",
                    epoch, totalLoss.getData(), acc);
            }
        }
        
        System.out.println("\nTraining complete!");
        
        // Final accuracy
        double finalAcc = accuracy(model, dataset);
        System.out.printf("\nFinal accuracy: %.2f%%\n", finalAcc);
        
        // Visualize decision boundary
        visualizeDecisionBoundary(model, dataset, "micrograd_replica_decision_boundary");
        
        System.out.println("╔" + SEPARATOR + "╗");
        System.out.println("║              Demo Complete!                            ║");
        System.out.println("╚" + SEPARATOR + "╝");
        System.out.println("\nThis demo mirrors micrograd's demo.ipynb approach:");
        System.out.println("  ✓ make_moons dataset");
        System.out.println("  ✓ MLP architecture [2, 16, 16, 1]");
        System.out.println("  ✓ SVM max-margin loss");
        System.out.println("  ✓ L2 regularization (alpha=1e-4)");
        System.out.println("  ✓ SGD optimizer (LR=0.1, 200 steps, gentle grad clipping)");
        System.out.println("  ✓ Decision boundary visualization");
        System.out.println("\nNote: Final accuracy may differ from Python version due to");
        System.out.println("different RNG implementations (Java vs Python/NumPy).");
        System.out.println("\nOriginal micrograd output: step 99 loss ~0.011, accuracy ~100%");
        System.out.println("Java replica (stabilized settings): typically 85-100% accuracy");
        System.out.println("\nFor guaranteed 100% accuracy with optimized training,");
        System.out.println("run: AdvancedClassifierDemo");
    }
}

