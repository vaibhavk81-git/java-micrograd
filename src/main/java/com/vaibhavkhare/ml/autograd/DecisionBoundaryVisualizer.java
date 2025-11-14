package com.vaibhavkhare.ml.autograd;

import org.knowm.xchart.BitmapEncoder;
import org.knowm.xchart.BitmapEncoder.BitmapFormat;
import org.knowm.xchart.SwingWrapper;
import org.knowm.xchart.XYChart;
import org.knowm.xchart.XYChartBuilder;
import org.knowm.xchart.XYSeries;
import org.knowm.xchart.style.Styler;

import javax.swing.*;
import java.awt.*;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * DecisionBoundaryVisualizer - Visualize decision boundaries for binary classification
 * 
 * <p>This class creates graphical visualizations of decision boundaries for trained MLP models.
 * It displays:
 * <ul>
 *   <li>Data points colored by their true class labels</li>
 *   <li>Color-coded decision regions (background)</li>
 *   <li>Decision boundary line (where prediction ≈ 0)</li>
 * </ul>
 * 
 * <p>Supports both file export (PNG) and interactive display via Swing.
 * 
 * @author Vaibhav Khare
 */
@SuppressWarnings("java:S106") // Allow System.out for visualization status messages
public class DecisionBoundaryVisualizer {
    
    private static final Logger LOGGER = Logger.getLogger(DecisionBoundaryVisualizer.class.getName());
    
    // Color scheme for classes
    private static final Color CLASS_POSITIVE_COLOR = new Color(0, 100, 200);  // Blue
    private static final Color CLASS_NEGATIVE_COLOR = new Color(200, 50, 50);  // Red
    private static final Color REGION_POSITIVE_COLOR = new Color(173, 216, 230, 50);  // Light blue (transparent)
    private static final Color REGION_NEGATIVE_COLOR = new Color(255, 192, 203, 50);  // Light pink (transparent)
    
    /**
     * Display mode for visualization
     */
    public enum DisplayMode {
        FILE_ONLY,      // Save to file only
        WINDOW_ONLY,    // Show window only
        BOTH            // Both file and window
    }
    
    /**
     * Represents a data point with features and label.
     */
    public interface DataPoint {
        double[] getFeatures();
        int getLabel();
    }
    
    /**
     * Visualize decision boundary for a trained MLP model.
     * 
     * @param model the trained MLP model
     * @param dataset list of data points with features and labels
     * @param filename output filename (without extension)
     * @param displayMode how to display the visualization
     */
    public static void visualize(MLP model, List<? extends DataPoint> dataset, String filename, DisplayMode displayMode) {
        try {
            XYChart chart = generatePlot(model, dataset);
            
            if (displayMode == DisplayMode.FILE_ONLY || displayMode == DisplayMode.BOTH) {
                saveToFile(chart, filename);
            }
            
            if (displayMode == DisplayMode.WINDOW_ONLY || displayMode == DisplayMode.BOTH) {
                displayInteractive(chart, filename);
            }
        } catch (Exception e) {
            LOGGER.log(Level.SEVERE, () -> "Error visualizing decision boundary: " + e.getMessage());
            System.err.println("Error visualizing decision boundary: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * Convenience method with default display mode (both file and window).
     */
    public static void visualize(MLP model, List<? extends DataPoint> dataset, String filename) {
        visualize(model, dataset, filename, DisplayMode.BOTH);
    }
    
    /**
     * Generate the plot with decision boundary and data points.
     * 
     * @param model the trained MLP model
     * @param dataset list of data points
     * @return configured XYChart
     */
    private static XYChart generatePlot(MLP model, List<? extends DataPoint> dataset) {
        // Find bounds of the dataset
        double minX = Double.MAX_VALUE;
        double maxX = Double.MIN_VALUE;
        double minY = Double.MAX_VALUE;
        double maxY = Double.MIN_VALUE;
        
        for (DataPoint point : dataset) {
            double[] features = point.getFeatures();
            minX = Math.min(minX, features[0]);
            maxX = Math.max(maxX, features[0]);
            minY = Math.min(minY, features[1]);
            maxY = Math.max(maxY, features[1]);
        }
        
        // Add padding
        double paddingX = (maxX - minX) * 0.1;
        double paddingY = (maxY - minY) * 0.1;
        minX -= paddingX;
        maxX += paddingX;
        minY -= paddingY;
        maxY += paddingY;
        
        // Create chart
        XYChart chart = new XYChartBuilder()
            .width(800)
            .height(600)
            .title("Decision Boundary Visualization")
            .xAxisTitle("X1")
            .yAxisTitle("X2")
            .theme(Styler.ChartTheme.Matlab)
            .build();
        
        // Configure chart styling
        chart.getStyler().setLegendPosition(Styler.LegendPosition.OutsideE);
        chart.getStyler().setLegendVisible(true);
        chart.getStyler().setDefaultSeriesRenderStyle(XYSeries.XYSeriesRenderStyle.Scatter);
        chart.getStyler().setPlotGridLinesVisible(true);
        chart.getStyler().setPlotContentSize(0.95);
        
        // Create grid for decision boundary visualization
        int gridResolution = 100;
        List<Double> gridX = new ArrayList<>();
        List<Double> gridY = new ArrayList<>();
        List<Double> predictions = new ArrayList<>();
        
        for (int i = 0; i < gridResolution; i++) {
            for (int j = 0; j < gridResolution; j++) {
                double x = minX + (maxX - minX) * j / gridResolution;
                double y = minY + (maxY - minY) * i / gridResolution;
                
                gridX.add(x);
                gridY.add(y);
                
                // Get model prediction
                List<Value> inputs = Arrays.asList(
                    new Value(x, "x"),
                    new Value(y, "y")
                );
                Value prediction = model.forward(inputs);
                predictions.add(prediction.getData());
            }
        }
        
        // Separate points by prediction sign for contour-like visualization
        List<Double> positiveX = new ArrayList<>();
        List<Double> positiveY = new ArrayList<>();
        List<Double> negativeX = new ArrayList<>();
        List<Double> negativeY = new ArrayList<>();
        List<Double> boundaryX = new ArrayList<>();
        List<Double> boundaryY = new ArrayList<>();
        
        for (int i = 0; i < gridX.size(); i++) {
            double pred = predictions.get(i);
            if (Math.abs(pred) < 0.1) {
                // Near decision boundary
                boundaryX.add(gridX.get(i));
                boundaryY.add(gridY.get(i));
            } else if (pred > 0) {
                positiveX.add(gridX.get(i));
                positiveY.add(gridY.get(i));
            } else {
                negativeX.add(gridX.get(i));
                negativeY.add(gridY.get(i));
            }
        }
        
        // Add decision regions (background) - use small markers
        chart.getStyler().setMarkerSize(1);
        if (!positiveX.isEmpty()) {
            XYSeries positiveRegion = chart.addSeries("Region: Class +1", positiveX, positiveY);
            positiveRegion.setMarkerColor(REGION_POSITIVE_COLOR);
            positiveRegion.setXYSeriesRenderStyle(XYSeries.XYSeriesRenderStyle.Scatter);
        }
        
        if (!negativeX.isEmpty()) {
            XYSeries negativeRegion = chart.addSeries("Region: Class -1", negativeX, negativeY);
            negativeRegion.setMarkerColor(REGION_NEGATIVE_COLOR);
            negativeRegion.setXYSeriesRenderStyle(XYSeries.XYSeriesRenderStyle.Scatter);
        }
        
        // Add decision boundary line
        if (!boundaryX.isEmpty()) {
            XYSeries boundary = chart.addSeries("Decision Boundary", boundaryX, boundaryY);
            boundary.setMarkerColor(Color.BLACK);
            boundary.setXYSeriesRenderStyle(XYSeries.XYSeriesRenderStyle.Scatter);
        }
        
        // Separate data points by class
        List<Double> class1X = new ArrayList<>();
        List<Double> class1Y = new ArrayList<>();
        List<Double> classMinus1X = new ArrayList<>();
        List<Double> classMinus1Y = new ArrayList<>();
        
        for (DataPoint point : dataset) {
            double[] features = point.getFeatures();
            if (point.getLabel() > 0) {
                class1X.add(features[0]);
                class1Y.add(features[1]);
            } else {
                classMinus1X.add(features[0]);
                classMinus1Y.add(features[1]);
            }
        }
        
        // Add data points with larger markers
        chart.getStyler().setMarkerSize(10);
        if (!class1X.isEmpty()) {
            XYSeries class1Series = chart.addSeries("Class +1", class1X, class1Y);
            class1Series.setMarkerColor(CLASS_POSITIVE_COLOR);
            class1Series.setXYSeriesRenderStyle(XYSeries.XYSeriesRenderStyle.Scatter);
        }
        
        if (!classMinus1X.isEmpty()) {
            XYSeries classMinus1Series = chart.addSeries("Class -1", classMinus1X, classMinus1Y);
            classMinus1Series.setMarkerColor(CLASS_NEGATIVE_COLOR);
            classMinus1Series.setXYSeriesRenderStyle(XYSeries.XYSeriesRenderStyle.Scatter);
        }
        
        return chart;
    }
    
    /**
     * Save chart to PNG file.
     * 
     * @param chart the chart to save
     * @param filename output filename (without extension)
     */
    private static void saveToFile(XYChart chart, String filename) {
        try {
            String outputFile = filename + ".png";
            BitmapEncoder.saveBitmap(chart, outputFile, BitmapFormat.PNG);
            System.out.println("Decision boundary saved to: " + outputFile);
        } catch (IOException e) {
            LOGGER.log(Level.SEVERE, () -> "Error saving decision boundary: " + e.getMessage());
            System.err.println("Error saving decision boundary: " + e.getMessage());
        }
    }
    
    /**
     * Display chart in an interactive Swing window.
     * 
     * @param chart the chart to display
     * @param title window title
     */
    private static void displayInteractive(XYChart chart, String title) {
        // SwingWrapper.displayChart() uses invokeAndWait internally,
        // so it must be called from a non-EDT thread
        if (SwingUtilities.isEventDispatchThread()) {
            // If already on EDT, spawn a new thread
            new Thread(() -> {
                SwingWrapper<XYChart> swingWrapper = new SwingWrapper<>(chart);
                JFrame frame = swingWrapper.displayChart();
                frame.setTitle(title);
                frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
                frame.setLocationRelativeTo(null);
            }).start();
        } else {
            // Not on EDT, can call directly
            SwingWrapper<XYChart> swingWrapper = new SwingWrapper<>(chart);
            JFrame frame = swingWrapper.displayChart();
            frame.setTitle(title);
            frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
            frame.setLocationRelativeTo(null);
        }
    }
}

