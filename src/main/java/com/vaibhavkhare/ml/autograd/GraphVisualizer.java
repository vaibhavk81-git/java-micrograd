package com.vaibhavkhare.ml.autograd;

import guru.nidi.graphviz.attribute.*;
import guru.nidi.graphviz.engine.*;
import guru.nidi.graphviz.model.*;

import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.logging.Level;
import java.util.logging.Logger;

import static guru.nidi.graphviz.model.Factory.*;

/**
 * GraphVisualizer - Visualize computation graphs similar to micrograd's visualization
 * 
 * <p>This class creates a visual representation of the computation graph built by Value objects.
 * Similar to micrograd's draw_dot() function in Python, it shows:
 * <ul>
 *   <li>Value nodes with their data and gradients</li>
 *   <li>Operation nodes (like +, *, tanh, etc.)</li>
 *   <li>Edges showing the flow of computation</li>
 * </ul>
 * 
 * <p>The visualization uses explicit color values to avoid SVG rendering warnings.
 * 
 * @author Vaibhav Khare
 */
@SuppressWarnings("java:S106") // Allow System.out for visualization status messages
public class GraphVisualizer {
    
    private static final Logger LOGGER = Logger.getLogger(GraphVisualizer.class.getName());
    
    // Explicit color definitions to avoid SVG rendering issues
    private static final String VALUE_NODE_COLOR = "#ADD8E6"; // Light blue
    private static final String OP_NODE_COLOR = "#D3D3D3";    // Light gray
    
    private final Set<Value> visitedValues = new HashSet<>();
    private final Map<String, Node> nodeMap = new HashMap<>();
    
    static {
        // Suppress noisy SVG rendering warnings from the graphviz library
        Logger.getLogger("com.kitfox.svg").setLevel(Level.SEVERE);
        Logger.getLogger("guru.nidi.graphviz.engine").setLevel(Level.WARNING);
    }
    
    /**
     * Visualize the computation graph rooted at the given Value
     * 
     * @param root The root Value node (typically the output/loss)
     * @param filename The output filename (without extension)
     * @param format The output format (PNG, SVG, PDF, etc.)
     */
    public static void visualize(Value root, String filename, Format format) {
        GraphVisualizer visualizer = new GraphVisualizer();
        MutableGraph g = visualizer.buildGraph(root);
        visualizer.render(g, filename, format);
    }
    
    /**
     * Convenience method to visualize as PNG
     */
    public static void visualize(Value root, String filename) {
        visualize(root, filename, Format.PNG);
    }
    
    /**
     * Build the graph structure by traversing the computation graph
     */
    private MutableGraph buildGraph(Value root) {
        MutableGraph g = mutGraph("ComputationGraph")
            .setDirected(true)
            .graphAttrs()
                .add(Rank.dir(Rank.RankDir.LEFT_TO_RIGHT));  // Left to right layout
        
        buildGraphRecursive(root, g);
        return g;
    }
    
    /**
     * Recursively build the graph
     */
    private void buildGraphRecursive(Value v, MutableGraph g) {
        if (visitedValues.contains(v)) {
            return;
        }
        visitedValues.add(v);
        
        // Create a unique ID for this value node
        String valueId = "value_" + System.identityHashCode(v);
        
        // Create label showing data and gradient
        String label = String.format("{ data %.4f | grad %.4f }", v.getData(), v.getGrad());
        if (v.getLabel() != null && !v.getLabel().isEmpty()) {
            label = v.getLabel() + " | " + label;
        }
        
        // Create the value node (rectangular box)
        // Using explicit hex color to avoid SVG rendering warnings
        Node valueNode = node(valueId)
            .with(Label.of(label))
            .with(Shape.RECORD)
            .with(Style.FILLED)
            .with(Color.rgb(VALUE_NODE_COLOR).fill());
        
        nodeMap.put(valueId, valueNode);
        
        // If this value has an operation, create an operation node
        if (v.getOp() != null && !v.getOp().isEmpty()) {
            String opId = "op_" + System.identityHashCode(v);
            
            // Create operation node (small circle)
            // Using explicit hex color to avoid SVG rendering warnings
            Node opNode = node(opId)
                .with(Label.of(v.getOp()))
                .with(Shape.CIRCLE)
                .with(Style.FILLED)
                .with(Color.rgb(OP_NODE_COLOR).fill())
                .with("fontsize", "12")
                .with("width", "0.5")
                .with("height", "0.5");
            
            nodeMap.put(opId, opNode);
            
            // Add edge from operation to value
            g.add(opNode.link(valueNode));
            
            // Process parent values and connect them to the operation
            for (Value parent : v.getParents()) {
                buildGraphRecursive(parent, g);  // Recursively build parent nodes
                
                String parentId = "value_" + System.identityHashCode(parent);
                Node parentNode = nodeMap.get(parentId);
                
                // Edge from parent value to operation
                g.add(parentNode.link(opNode));
            }
        } else {
            // Add the leaf node
            g.add(valueNode);
            
            // Process children even if no operation (for leaf nodes)
            for (Value parent : v.getParents()) {
                buildGraphRecursive(parent, g);
            }
        }
    }
    
    /**
     * Render the graph to a file with optimized settings.
     * 
     * @param g        the graph to render
     * @param filename the output filename (without extension)
     * @param format   the output format (PNG, SVG, etc.)
     */
    private void render(MutableGraph g, String filename, Format format) {
        try {
            String outputFile = filename + "." + format.name().toLowerCase();
            
            // Render to file with optimized engine settings
            Graphviz.fromGraph(g)
                .width(1200)
                .engine(Engine.DOT)  // Explicitly use DOT engine for better compatibility
                .render(format)
                .toFile(new File(outputFile));
            
            System.out.println("Graph saved to: " + outputFile);
            
        } catch (IOException e) {
            LOGGER.log(Level.SEVERE, () -> "Error rendering graph: " + e.getMessage());
            System.err.println("Error rendering graph: " + e.getMessage());
        }
    }
}

