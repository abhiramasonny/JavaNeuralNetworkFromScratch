package src;

import java.awt.image.BufferedImage;
import javax.swing.*;
import java.awt.*;
import java.util.ArrayList;
import java.util.List;
import static src.MNIST.width;
import static src.MNIST.height;

public class MetricsVisualizer extends JFrame {
    public static JFrame displayImg(double[] image, int prediction) {

        int scale = 10; // Scale factor to make each pixel bigger
        BufferedImage bufferedImage = new BufferedImage(width * scale, height * scale, BufferedImage.TYPE_BYTE_GRAY);
    
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int pixelValue = (int) (image[y * width + x] * 255);
                for (int dy = 0; dy < scale; dy++) {
                    for (int dx = 0; dx < scale; dx++) {
                        bufferedImage.setRGB(x * scale + dx, y * scale + dy, new Color(pixelValue, pixelValue, pixelValue).getRGB());
                    }
                }
            }
        }
    
        JFrame frame = new JFrame();
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setSize(300, 300);
        frame.setLayout(new BorderLayout());
    
        JLabel imageLabel = new JLabel(new ImageIcon(bufferedImage));
        frame.add(imageLabel, BorderLayout.CENTER);
    
        JLabel predictionLabel = new JLabel("Prediction: " + prediction);
        frame.add(predictionLabel, BorderLayout.SOUTH);
    
        frame.setVisible(true);
        return frame;
    }

    public static JFrame displayImgCIFAR(double[] image, String prediction, int width, int height) {

        int scale = 10;
        BufferedImage bufferedImage = new BufferedImage(width * scale, height * scale, BufferedImage.TYPE_INT_RGB);

        int channels = 3;

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                // Get the pixel value for each channel and normalize it
                int r = (int) (image[(y * width + x) * channels] * 255);
                int g = (int) (image[(y * width + x) * channels + 1] * 255);
                int b = (int) (image[(y * width + x) * channels + 2] * 255);

                // Ensure the values are within the 0-255 range
                r = Math.max(0, Math.min(255, r));
                g = Math.max(0, Math.min(255, g));
                b = Math.max(0, Math.min(255, b));

                // Set the pixel value with RGB colors
                for (int dy = 0; dy < scale; dy++) {
                    for (int dx = 0; dx < scale; dx++) {
                        bufferedImage.setRGB(x * scale + dx, y * scale + dy, new Color(r, g, b).getRGB());
                    }
                }
            }
        }

        JFrame frame = new JFrame();
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setSize(300, 300);
        frame.setLayout(new BorderLayout());

        JLabel imageLabel = new JLabel(new ImageIcon(bufferedImage));
        frame.add(imageLabel, BorderLayout.CENTER);

        JLabel predictionLabel = new JLabel("Prediction: " + prediction);
        frame.add(predictionLabel, BorderLayout.SOUTH);

        frame.setVisible(true);
        return frame;
    }

    private List<Double> losses = new ArrayList<>();
    private List<Double> accuracies = new ArrayList<>();
    private int epochs;

    public MetricsVisualizer(int epochs) {
        this.epochs = epochs;
        setTitle("Training Metrics");
        setSize(800, 600);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    }

    public void logMetrics(double loss, double accuracy) {
        losses.add(loss);
        accuracies.add(accuracy);
        repaint();
    }

    @Override
    public void paint(Graphics g) {
        super.paint(g);
        Graphics2D g2d = (Graphics2D) g;

        // Draw axes
        g2d.drawLine(50, 50, 50, 550); // Y-axis
        g2d.drawLine(50, 550, 750, 550); // X-axis

        // Draw labels
        g2d.drawString("Epochs", 375, 580);
        g2d.drawString("Loss / Accuracy", 10, 300);

        if (!losses.isEmpty()) {
            g2d.setColor(Color.RED);
            drawCurve(g2d, losses, 50, 550, "Loss");
        }

        if (!accuracies.isEmpty()) {
            g2d.setColor(Color.BLUE);
            drawCurve(g2d, accuracies, 50, 550, "Accuracy");
        }
    }

    private void drawCurve(Graphics2D g2d, List<Double> data, int xStart, int yStart, String label) {
        if (data.size() < 2) {
            return;
        }

        int xStep = (700 / epochs);
        double maxValue = data.stream().mapToDouble(Double::doubleValue).max().orElse(1.0);

        for (int i = 1; i < data.size(); i++) {
            int x1 = xStart + (i - 1) * xStep;
            int y1 = (int) (yStart - (data.get(i - 1) / maxValue) * 500);
            int x2 = xStart + i * xStep;
            int y2 = (int) (yStart - (data.get(i) / maxValue) * 500);
            g2d.drawLine(x1, y1, x2, y2);
        }

        // Label the last data point
        int lastX = xStart + (data.size() - 1) * xStep;
        int lastY = (int) (yStart - (data.get(data.size() - 1) / maxValue) * 500);
        g2d.drawString(label, lastX, lastY);
    }


    public static void displayConfusionMatrix(int[][] confusionMatrix, String[] labels) {
        JFrame frame = new JFrame("Confusion Matrix");
        frame.setSize(800, 800);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        JPanel panel = new JPanel() {
            @Override
            protected void paintComponent(Graphics g) {
                super.paintComponent(g);
                Graphics2D g2d = (Graphics2D) g;

                int cellSize = 50;
                for (int i = 0; i < confusionMatrix.length; i++) {
                    for (int j = 0; j < confusionMatrix[0].length; j++) {
                        int value = confusionMatrix[i][j];
                        g2d.setColor(Color.WHITE);
                        g2d.fillRect(j * cellSize, i * cellSize, cellSize, cellSize);
                        g2d.setColor(Color.BLACK);
                        g2d.drawRect(j * cellSize, i * cellSize, cellSize, cellSize);
                        g2d.drawString(String.valueOf(value), j * cellSize + cellSize / 4, i * cellSize + cellSize / 2);
                    }
                }

                if (labels != null) {
                    for (int i = 0; i < labels.length; i++) {
                        g2d.drawString(labels[i], i * cellSize + cellSize / 2, confusionMatrix.length * cellSize + 20);
                        g2d.drawString(labels[i], confusionMatrix.length * cellSize + 20, i * cellSize + cellSize / 2);
                    }
                }
            }
        };
        frame.add(panel);
        frame.setVisible(true);
    }
}
