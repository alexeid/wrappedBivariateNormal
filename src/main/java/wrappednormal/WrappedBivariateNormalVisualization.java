package wrappednormal;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.geom.*;
import java.util.*;
import java.util.List;

/**
 * Interactive visualization of Wrapped Bivariate Normal distributions and their product.
 * Shows individual components, product distribution, credible regions, and samples.
 */
public class WrappedBivariateNormalVisualization extends JFrame {

    private VisualizationPanel vizPanel;
    private JLabel statsLabel;
    private WrappedBivariateNormal wbn1, wbn2, wbn3;
    private ProductWrappedBivariateNormal product;

    public WrappedBivariateNormalVisualization() {
        super("Wrapped Bivariate Normal Product Distribution");
        System.out.println("Creating visualization window...");

        // Initialize distributions
        initializeDistributions();

        // Setup UI
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setLayout(new BorderLayout());

        // Create visualization panel
        System.out.println("Creating visualization panel...");
        vizPanel = new VisualizationPanel();
        add(vizPanel, BorderLayout.CENTER);

        // Create control panel
        System.out.println("Creating control panel...");
        JPanel controlPanel = createControlPanel();
        add(controlPanel, BorderLayout.SOUTH);

        // Create stats panel
        System.out.println("Creating stats panel...");
        JPanel statsPanel = createStatsPanel();
        add(statsPanel, BorderLayout.NORTH);

        // Pack to preferred sizes
        pack();

        // Set size and center
        setSize(1200, 700);
        setLocationRelativeTo(null);
        System.out.println("Window setup complete. Size: " + getWidth() + "x" + getHeight());
    }

    private void initializeDistributions() {
        System.out.println("Initializing distributions...");

        // Component 1: Tight constraint
        wbn1 = new WrappedBivariateNormal(Math.PI/2, Math.PI, 0.2, 0.2, 0.1);
        System.out.println("  Component 1: " + wbn1);

        // Component 2: Loose constraint
        wbn2 = new WrappedBivariateNormal(Math.PI, Math.PI/2, 0.6, 0.6, 0.2);
        System.out.println("  Component 2: " + wbn2);

        // Component 3: Medium constraint
        wbn3 = new WrappedBivariateNormal(3*Math.PI/4, 3*Math.PI/4, 0.4, 0.4, 0.0);
        System.out.println("  Component 3: " + wbn3);

        // Product distribution
        product = new ProductWrappedBivariateNormal(wbn1, wbn2, wbn3);
        System.out.println("  Product: " + product);
    }

    private JPanel createControlPanel() {
        JPanel panel = new JPanel();
        panel.setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));

        JButton regenerateBtn = new JButton("Generate New Samples");
        regenerateBtn.addActionListener(e -> {
            vizPanel.generateSamples();
            vizPanel.repaint();
        });

        JCheckBox showSamplesBox = new JCheckBox("Show Samples", true);
        showSamplesBox.addActionListener(e -> {
            vizPanel.setShowSamples(showSamplesBox.isSelected());
            vizPanel.repaint();
        });

        JCheckBox showContoursBox = new JCheckBox("Show Contours", true);
        showContoursBox.addActionListener(e -> {
            vizPanel.setShowContours(showContoursBox.isSelected());
            vizPanel.repaint();
        });

        JCheckBox showGridBox = new JCheckBox("Show Grid", true);
        showGridBox.addActionListener(e -> {
            vizPanel.setShowGrid(showGridBox.isSelected());
            vizPanel.repaint();
        });

        panel.add(regenerateBtn);
        panel.add(Box.createHorizontalStrut(20));
        panel.add(showSamplesBox);
        panel.add(showContoursBox);
        panel.add(showGridBox);

        return panel;
    }

    private JPanel createStatsPanel() {
        JPanel panel = new JPanel();
        panel.setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));
        panel.setLayout(new BoxLayout(panel, BoxLayout.Y_AXIS));

        // Title
        JLabel titleLabel = new JLabel("Effective Weights (based on variance):");
        titleLabel.setFont(new Font("Arial", Font.BOLD, 14));
        panel.add(titleLabel);

        // Calculate and display weights
        double[] weights = product.getEffectiveWeights();
        statsLabel = new JLabel(String.format(
                "<html>Component 1 (tight, σ=0.2): <b>%.1f%%</b> | " +
                        "Component 2 (loose, σ=0.6): <b>%.1f%%</b> | " +
                        "Component 3 (medium, σ=0.4): <b>%.1f%%</b></html>",
                weights[0] * 100, weights[1] * 100, weights[2] * 100
        ));
        panel.add(statsLabel);

        return panel;
    }

    /**
     * Main visualization panel that renders the distributions.
     */
    private class VisualizationPanel extends JPanel {
        private static final int MARGIN = 50;
        private static final int SAMPLE_COUNT = 200;

        private boolean showSamples = true;
        private boolean showContours = true;
        private boolean showGrid = true;

        private double[][] samples1, samples2, samples3, samplesProduct;
        private double threshold50_1, threshold95_1;
        private double threshold50_2, threshold95_2;
        private double threshold50_3, threshold95_3;
        private double threshold50_p, threshold95_p;

        public VisualizationPanel() {
            setBackground(Color.WHITE);
            setPreferredSize(new Dimension(1000, 600));
            System.out.println("VisualizationPanel created, generating initial samples...");
            generateSamples();
            System.out.println("Initial samples generated.");
        }

        public void generateSamples() {
            System.out.println("Generating " + SAMPLE_COUNT + " samples for each distribution...");
            long startTime, endTime;

            // Generate samples
            startTime = System.currentTimeMillis();
            samples1 = wbn1.sample(SAMPLE_COUNT);
            endTime = System.currentTimeMillis();
            System.out.println("  Component 1 samples: " + samples1.length + " (" + (endTime - startTime) + " ms)");

            startTime = System.currentTimeMillis();
            samples2 = wbn2.sample(SAMPLE_COUNT);
            endTime = System.currentTimeMillis();
            System.out.println("  Component 2 samples: " + samples2.length + " (" + (endTime - startTime) + " ms)");

            startTime = System.currentTimeMillis();
            samples3 = wbn3.sample(SAMPLE_COUNT);
            endTime = System.currentTimeMillis();
            System.out.println("  Component 3 samples: " + samples3.length + " (" + (endTime - startTime) + " ms)");

            System.out.println("  Generating product samples...");
            startTime = System.currentTimeMillis();
            samplesProduct = product.sample(SAMPLE_COUNT);
            endTime = System.currentTimeMillis();
            System.out.println("  Product samples: " + samplesProduct.length + " (" + (endTime - startTime) + " ms)");

            // Calculate thresholds
            System.out.println("Calculating credible region thresholds...");

            startTime = System.currentTimeMillis();
            threshold50_1 = findCredibleThreshold(wbn1, samples1, 0.5);
            threshold95_1 = findCredibleThreshold(wbn1, samples1, 0.95);
            endTime = System.currentTimeMillis();
            System.out.println("  Component 1 thresholds: 50%=" + String.format("%.6f", threshold50_1) +
                    ", 95%=" + String.format("%.6f", threshold95_1) + " (" + (endTime - startTime) + " ms)");

            startTime = System.currentTimeMillis();
            threshold50_2 = findCredibleThreshold(wbn2, samples2, 0.5);
            threshold95_2 = findCredibleThreshold(wbn2, samples2, 0.95);
            endTime = System.currentTimeMillis();
            System.out.println("  Component 2 thresholds: 50%=" + String.format("%.6f", threshold50_2) +
                    ", 95%=" + String.format("%.6f", threshold95_2) + " (" + (endTime - startTime) + " ms)");

            startTime = System.currentTimeMillis();
            threshold50_3 = findCredibleThreshold(wbn3, samples3, 0.5);
            threshold95_3 = findCredibleThreshold(wbn3, samples3, 0.95);
            endTime = System.currentTimeMillis();
            System.out.println("  Component 3 thresholds: 50%=" + String.format("%.6f", threshold50_3) +
                    ", 95%=" + String.format("%.6f", threshold95_3) + " (" + (endTime - startTime) + " ms)");

            // For product, use normalized PDF
            System.out.println("Calculating product distribution thresholds...");
            startTime = System.currentTimeMillis();

            // First, estimate normalization constant
            System.out.println("  Estimating normalization constant...");
            long normStart = System.currentTimeMillis();
            double Z = product.estimateNormalizationConstant(100);
            long normEnd = System.currentTimeMillis();
            System.out.println("  Normalization constant: " + String.format("%.6f", Z) + " (" + (normEnd - normStart) + " ms)");

            // Then calculate PDFs
            System.out.println("  Calculating PDFs for product samples...");
            long pdfStart = System.currentTimeMillis();
            double[] productPdfs = new double[samplesProduct.length];
            for (int i = 0; i < samplesProduct.length; i++) {
                productPdfs[i] = product.pdf(samplesProduct[i][0], samplesProduct[i][1], 100);
                if (i % 50 == 0) {
                    System.out.println("    Processed " + i + "/" + samplesProduct.length + " samples");
                }
            }
            long pdfEnd = System.currentTimeMillis();
            System.out.println("  PDF calculations complete (" + (pdfEnd - pdfStart) + " ms)");

            Arrays.sort(productPdfs);
            threshold50_p = productPdfs[productPdfs.length - (int)(0.5 * productPdfs.length) - 1];
            threshold95_p = productPdfs[productPdfs.length - (int)(0.95 * productPdfs.length) - 1];
            endTime = System.currentTimeMillis();
            System.out.println("  Product thresholds: 50%=" + String.format("%.6f", threshold50_p) +
                    ", 95%=" + String.format("%.6f", threshold95_p) +
                    " (total: " + (endTime - startTime) + " ms)");

            System.out.println("Sample generation complete!");
        }

        private double findCredibleThreshold(WrappedBivariateNormal dist, double[][] samples, double level) {
            double[] pdfs = new double[samples.length];
            for (int i = 0; i < samples.length; i++) {
                pdfs[i] = dist.pdf(samples[i][0], samples[i][1]);
            }
            Arrays.sort(pdfs);
            int idx = pdfs.length - (int)(level * pdfs.length) - 1;
            return pdfs[Math.max(0, idx)];
        }

        @Override
        protected void paintComponent(Graphics g) {
            super.paintComponent(g);

            Graphics2D g2 = (Graphics2D) g;
            g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

            int width = getWidth();
            int height = getHeight();

            // Skip if panel is too small
            if (width < 100 || height < 100) {
                System.out.println("Panel too small to draw: " + width + "x" + height);
                return;
            }

            System.out.println("Painting visualization panel (" + width + "x" + height + ")...");

            int halfWidth = width / 2;

            // Draw left panel (individual components)
            drawPanel(g2, 0, 0, halfWidth, height, "Individual Components", true);

            // Draw right panel (product distribution)
            drawPanel(g2, halfWidth, 0, halfWidth, height, "Product Distribution", false);

            // Draw legend
            drawLegend(g2, width - 250, height - 200);
        }

        private void drawPanel(Graphics2D g2, int x, int y, int width, int height,
                               String title, boolean isComponents) {
            // Fill background first to see if method is called
            g2.setColor(isComponents ? new Color(255, 240, 240) : new Color(240, 240, 255));
            g2.fillRect(x, y, width, height);

            // Draw title
            g2.setFont(new Font("Arial", Font.BOLD, 16));
            g2.setColor(Color.BLACK);
            FontMetrics fm = g2.getFontMetrics();
            g2.drawString(title, x + (width - fm.stringWidth(title)) / 2, y + 30);

            // Calculate plot area
            int plotX = x + MARGIN;
            int plotY = y + MARGIN + 20;
            int plotWidth = width - 2 * MARGIN;
            int plotHeight = height - 2 * MARGIN - 40;

            // Draw background
            g2.setColor(new Color(250, 250, 250));
            g2.fillRect(plotX, plotY, plotWidth, plotHeight);

            // Draw grid
            if (showGrid) {
                drawGrid(g2, plotX, plotY, plotWidth, plotHeight);
            }

            // Draw axes
            drawAxes(g2, plotX, plotY, plotWidth, plotHeight);

            // Set clipping region
            Shape oldClip = g2.getClip();
            g2.setClip(plotX, plotY, plotWidth, plotHeight);

            if (isComponents) {
                // Draw individual components
                drawDistribution(g2, plotX, plotY, plotWidth, plotHeight,
                        wbn1, samples1, threshold50_1, threshold95_1,
                        new Color(255, 0, 0, 77), Color.RED);

                drawDistribution(g2, plotX, plotY, plotWidth, plotHeight,
                        wbn2, samples2, threshold50_2, threshold95_2,
                        new Color(0, 255, 0, 77), Color.GREEN);

                drawDistribution(g2, plotX, plotY, plotWidth, plotHeight,
                        wbn3, samples3, threshold50_3, threshold95_3,
                        new Color(0, 0, 255, 77), Color.BLUE);
            } else {
                // Draw product distribution
                drawProductDistribution(g2, plotX, plotY, plotWidth, plotHeight);
            }

            // Restore clipping
            g2.setClip(oldClip);
        }

        private void drawGrid(Graphics2D g2, int x, int y, int width, int height) {
            g2.setColor(new Color(230, 230, 230));
            g2.setStroke(new BasicStroke(1));

            // Vertical lines
            for (int i = 0; i <= 4; i++) {
                int xPos = x + (i * width) / 4;
                g2.drawLine(xPos, y, xPos, y + height);
            }

            // Horizontal lines
            for (int i = 0; i <= 4; i++) {
                int yPos = y + (i * height) / 4;
                g2.drawLine(x, yPos, x + width, yPos);
            }
        }

        private void drawAxes(Graphics2D g2, int x, int y, int width, int height) {
            g2.setColor(Color.BLACK);
            g2.setStroke(new BasicStroke(2));

            // X-axis
            g2.drawLine(x, y + height, x + width, y + height);

            // Y-axis
            g2.drawLine(x, y, x, y + height);

            // Labels
            g2.setFont(new Font("Arial", Font.PLAIN, 12));
            String[] labels = {"0", "π/2", "π", "3π/2", "2π"};

            // X-axis labels
            for (int i = 0; i < 5; i++) {
                int xPos = x + (i * width) / 4;
                int yPos = y + height + 20;

                FontMetrics fm = g2.getFontMetrics();
                g2.drawString(labels[i], xPos - fm.stringWidth(labels[i])/2, yPos);

                // Tick marks
                g2.drawLine(xPos, y + height, xPos, y + height + 5);
            }

            // Y-axis labels
            for (int i = 0; i < 5; i++) {
                int xPos = x - 10;
                int yPos = y + height - (i * height) / 4;

                FontMetrics fm = g2.getFontMetrics();
                g2.drawString(labels[i], xPos - fm.stringWidth(labels[i]), yPos + 5);

                // Tick marks
                g2.drawLine(x - 5, yPos, x, yPos);
            }

            // Axis labels
            g2.setFont(new Font("Arial", Font.BOLD, 14));
            g2.drawString("θ₁", x + width/2 - 10, y + height + 40);

            // Rotate for Y-axis label
            AffineTransform oldTransform = g2.getTransform();
            g2.rotate(-Math.PI/2, x - 40, y + height/2);
            g2.drawString("θ₂", x - 40, y + height/2);
            g2.setTransform(oldTransform);
        }

        private void drawDistribution(Graphics2D g2, int x, int y, int width, int height,
                                      WrappedBivariateNormal dist, double[][] samples,
                                      double threshold50, double threshold95,
                                      Color fillColor, Color lineColor) {

            // Draw contours
            if (showContours) {
                // 95% contour (dashed)
                g2.setColor(lineColor);
                g2.setStroke(new BasicStroke(1, BasicStroke.CAP_BUTT,
                        BasicStroke.JOIN_MITER, 10.0f, new float[]{5.0f}, 0.0f));
                drawContour(g2, x, y, width, height, dist, threshold95);

                // 50% contour (solid)
                g2.setStroke(new BasicStroke(2));
                drawContour(g2, x, y, width, height, dist, threshold50);
            }

            // Draw samples
            if (showSamples) {
                g2.setColor(fillColor);
                for (double[] sample : samples) {
                    int px = x + (int)(sample[0] * width / (2 * Math.PI));
                    int py = y + height - (int)(sample[1] * height / (2 * Math.PI));
                    g2.fillOval(px - 2, py - 2, 4, 4);
                }
            }
        }

        private void drawProductDistribution(Graphics2D g2, int x, int y, int width, int height) {
            // Draw heatmap
            int resolution = 50;
            double step = 2 * Math.PI / resolution;

            for (int i = 0; i < resolution; i++) {
                for (int j = 0; j < resolution; j++) {
                    double theta1 = i * step;
                    double theta2 = j * step;
                    double pdf = product.pdf(theta1, theta2, 100);

                    // Map PDF to color intensity
                    float intensity = (float)Math.min(1.0, pdf * 20);
                    Color color = new Color(0.5f, 0, 0.5f, intensity * 0.5f);

                    int px = x + (int)(theta1 * width / (2 * Math.PI));
                    int py = y + height - (int)(theta2 * height / (2 * Math.PI));
                    int cellWidth = Math.max(1, width / resolution);
                    int cellHeight = Math.max(1, height / resolution);

                    g2.setColor(color);
                    g2.fillRect(px, py - cellHeight, cellWidth, cellHeight);
                }
            }

            // Draw contours
            if (showContours) {
                Color purple = new Color(128, 0, 128);

                // 95% contour (dashed)
                g2.setColor(purple);
                g2.setStroke(new BasicStroke(1, BasicStroke.CAP_BUTT,
                        BasicStroke.JOIN_MITER, 10.0f, new float[]{5.0f}, 0.0f));
                drawProductContour(g2, x, y, width, height, threshold95_p);

                // 50% contour (solid)
                g2.setStroke(new BasicStroke(2));
                drawProductContour(g2, x, y, width, height, threshold50_p);
            }

            // Draw samples
            if (showSamples) {
                g2.setColor(Color.BLACK);
                for (double[] sample : samplesProduct) {
                    int px = x + (int)(sample[0] * width / (2 * Math.PI));
                    int py = y + height - (int)(sample[1] * height / (2 * Math.PI));
                    g2.fillOval(px - 3, py - 3, 6, 6);
                }
            }
        }

        private void drawContour(Graphics2D g2, int x, int y, int width, int height,
                                 WrappedBivariateNormal dist, double threshold) {
            List<Point2D> contourPoints = new ArrayList<>();
            int resolution = 100;
            double step = 2 * Math.PI / resolution;

            for (int i = 0; i < resolution; i++) {
                for (int j = 0; j < resolution; j++) {
                    double theta1 = i * step;
                    double theta2 = j * step;
                    double pdf = dist.pdf(theta1, theta2);

                    if (Math.abs(pdf - threshold) < threshold * 0.1) {
                        int px = x + (int)(theta1 * width / (2 * Math.PI));
                        int py = y + height - (int)(theta2 * height / (2 * Math.PI));
                        contourPoints.add(new Point2D.Double(px, py));
                    }
                }
            }

            // Draw contour points
            for (Point2D p : contourPoints) {
                g2.fillOval((int)p.getX() - 1, (int)p.getY() - 1, 2, 2);
            }
        }

        private void drawProductContour(Graphics2D g2, int x, int y, int width, int height,
                                        double threshold) {
            List<Point2D> contourPoints = new ArrayList<>();
            int resolution = 80;
            double step = 2 * Math.PI / resolution;

            for (int i = 0; i < resolution; i++) {
                for (int j = 0; j < resolution; j++) {
                    double theta1 = i * step;
                    double theta2 = j * step;
                    double pdf = product.pdf(theta1, theta2, 100);

                    if (Math.abs(pdf - threshold) < threshold * 0.2) {
                        int px = x + (int)(theta1 * width / (2 * Math.PI));
                        int py = y + height - (int)(theta2 * height / (2 * Math.PI));
                        contourPoints.add(new Point2D.Double(px, py));
                    }
                }
            }

            // Draw contour points
            for (Point2D p : contourPoints) {
                g2.fillOval((int)p.getX() - 1, (int)p.getY() - 1, 2, 2);
            }
        }

        private void drawLegend(Graphics2D g2, int x, int y) {
            g2.setColor(new Color(240, 240, 240));
            g2.fillRoundRect(x - 10, y - 10, 240, 180, 10, 10);
            g2.setColor(Color.BLACK);
            g2.drawRoundRect(x - 10, y - 10, 240, 180, 10, 10);

            g2.setFont(new Font("Arial", Font.BOLD, 14));
            g2.drawString("Legend", x, y + 5);

            g2.setFont(new Font("Arial", Font.PLAIN, 12));
            int yOffset = 25;

            // Component 1
            g2.setColor(new Color(255, 0, 0, 77));
            g2.fillRect(x, y + yOffset, 20, 15);
            g2.setColor(Color.BLACK);
            g2.drawString("Component 1 (σ=0.2)", x + 25, y + yOffset + 12);

            // Component 2
            yOffset += 20;
            g2.setColor(new Color(0, 255, 0, 77));
            g2.fillRect(x, y + yOffset, 20, 15);
            g2.setColor(Color.BLACK);
            g2.drawString("Component 2 (σ=0.6)", x + 25, y + yOffset + 12);

            // Component 3
            yOffset += 20;
            g2.setColor(new Color(0, 0, 255, 77));
            g2.fillRect(x, y + yOffset, 20, 15);
            g2.setColor(Color.BLACK);
            g2.drawString("Component 3 (σ=0.4)", x + 25, y + yOffset + 12);

            // Product
            yOffset += 20;
            g2.setColor(new Color(128, 0, 128, 128));
            g2.fillRect(x, y + yOffset, 20, 15);
            g2.setColor(Color.BLACK);
            g2.drawString("Product Distribution", x + 25, y + yOffset + 12);

            // Contour lines
            yOffset += 25;
            g2.setStroke(new BasicStroke(2));
            g2.drawLine(x, y + yOffset, x + 20, y + yOffset);
            g2.drawString("50% credible region", x + 25, y + yOffset + 3);

            yOffset += 15;
            g2.setStroke(new BasicStroke(1, BasicStroke.CAP_BUTT,
                    BasicStroke.JOIN_MITER, 10.0f, new float[]{5.0f}, 0.0f));
            g2.drawLine(x, y + yOffset, x + 20, y + yOffset);
            g2.setStroke(new BasicStroke(1));
            g2.drawString("95% credible region", x + 25, y + yOffset + 3);
        }

        public void setShowSamples(boolean show) { showSamples = show; }
        public void setShowContours(boolean show) { showContours = show; }
        public void setShowGrid(boolean show) { showGrid = show; }
    }

    public static void main(String[] args) {
        System.out.println("Starting Wrapped Bivariate Normal Visualization...");

        SwingUtilities.invokeLater(() -> {
            try {
                System.out.println("Creating visualization frame on EDT...");
                WrappedBivariateNormalVisualization viz = new WrappedBivariateNormalVisualization();
                System.out.println("Setting frame visible...");
                viz.setVisible(true);
                System.out.println("Visualization window should now be visible.");
            } catch (Exception e) {
                System.err.println("Error creating visualization: " + e.getMessage());
                e.printStackTrace();
            }
        });

        System.out.println("Main method complete. Visualization running in EDT.");
    }
}