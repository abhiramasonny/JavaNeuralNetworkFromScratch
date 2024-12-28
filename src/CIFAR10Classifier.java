package src;

import static src.Utilities.*;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import javax.swing.JFrame;
import src.NeuralNetwork.ActivationFunction;

public class CIFAR10Classifier {

    public static final int WIDTH = 32;  // CIFAR-10 image dimensions
    public static final int HEIGHT = 32;
    public static final int CHANNELS = 3;  // RGB channels
    public static final int INPUT_SIZE = WIDTH * HEIGHT * CHANNELS;

    public static void main(String[] args) {
        try {
            String dataDir = "data/cifar-10-batches-bin";
            int numTrainingExamples = 50000;
            int outputSize = 10;  // 10 classes in CIFAR-10
            int epochs = 5;
            int batchSize = 1024;
            double splitRatio = 0.8;
            double learningRate = 0.001;

            int numTrain = (int) (numTrainingExamples * splitRatio);
            int numTest = numTrainingExamples - numTrain;
            int[] hiddenSizes = {512, 256, 128};

            ActivationFunction[] activationFunctions = {
                ActivationFunction.RELU,       // Conv Layer 1
                ActivationFunction.RELU,       // Conv Layer 2
                ActivationFunction.RELU,       // Conv Layer 3
                ActivationFunction.RELU,       // Fully Connected Layer 1
                ActivationFunction.RELU,       // Fully Connected Layer 2
                ActivationFunction.SOFTMAX     // Output Layer
            };            

            System.out.println("Loading CIFAR-10 data...");
            Map<String, double[][]> data = loadCIFAR10(dataDir, numTrainingExamples, WIDTH, HEIGHT, CHANNELS);
            double[][] X = data.get("X");
            double[][] Y = data.get("Y");
            double[][] XTrain = new double[X.length][numTrain];
            double[][] YTrain = new double[Y.length][numTrain];
            double[][] XTest = new double[X.length][numTest];
            double[][] YTest = new double[Y.length][numTest];
            
            int[] indices = getRandomPermutation(numTrainingExamples);
            splitData(X, Y, XTrain, YTrain, XTest, YTest, indices, numTrain);

            NeuralNetwork nn = new NeuralNetwork(INPUT_SIZE, hiddenSizes, outputSize, learningRate, epochs, batchSize, activationFunctions);
            System.out.println("Starting training...");
            nn.train(XTrain, YTrain);
            System.out.println("Training completed.");

            evaluateModel(nn, XTrain, YTrain, XTest, YTest, outputSize);

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static void splitData(double[][] X, double[][] Y, double[][] XTrain, double[][] YTrain, double[][] XTest, double[][] YTest, int[] indices, int numTrain) {
        for (int i = 0; i < indices.length; i++) {
            double[][] targetX = i < numTrain ? XTrain : XTest;
            double[][] targetY = i < numTrain ? YTrain : YTest;
            int targetIndex = i < numTrain ? i : i - numTrain;

            for (int j = 0; j < X.length; j++) {
                targetX[j][targetIndex] = X[j][indices[i]];
            }
            for (int j = 0; j < Y.length; j++) {
                targetY[j][targetIndex] = Y[j][indices[i]];
            }
        }
    }

    private static void evaluateModel(NeuralNetwork nn, double[][] XTrain, double[][] YTrain, double[][] XTest, double[][] YTest, int outputSize) {
        int[] trainPredictions = nn.predict(XTrain);
        int[] trainLabels = convertOneHotToLabels(YTrain);
        double trainAccuracy = computeAccuracy(trainPredictions, trainLabels);
        System.out.println("Training accuracy: " + trainAccuracy);

        int[] testPredictions = nn.predict(XTest);
        int[] testLabels = convertOneHotToLabels(YTest);
        double testAccuracy = computeAccuracy(testPredictions, testLabels);
        System.out.println("Test accuracy: " + testAccuracy);

        int[][] confusionMatrix = computeConfusionMatrix(testPredictions, testLabels, outputSize);
        String[] options = {"Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"};
        MetricsVisualizer.displayConfusionMatrix(confusionMatrix, options);

        displaySamplePredictions(XTest, testPredictions, options);
    }

    private static void displaySamplePredictions(double[][] XTest, int[] testPredictions, String[] options) {
        for (int i = 0; i < Math.min(10, testPredictions.length); i++) {
            double[] image = Arrays.copyOfRange(XTest[i], 0, INPUT_SIZE);
            JFrame frame = MetricsVisualizer.displayImgCIFAR(image, options[testPredictions[i]], WIDTH, HEIGHT);

            try {
                Thread.sleep(1000);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            } finally {
                frame.dispose();
            }
        }
    }

    public static Map<String, double[][]> loadCIFAR10(String dataDir, int numData, int width, int height, int channels) throws IOException {
        int imageSize = width * height * channels;
        int labelSize = 10; // CIFAR-10 has 10 classes
        double[][] X = new double[imageSize][numData];
        double[][] Y = new double[labelSize][numData];

        File folder = new File(dataDir);
        File[] batchFiles = folder.listFiles((dir, name) -> name.startsWith("data_batch") && name.endsWith(".bin"));

        if (batchFiles == null || batchFiles.length == 0) {
            throw new IllegalArgumentException("No CIFAR-10 data batch files found in directory: " + dataDir);
        }

        int currentIndex = 0;
        for (File batchFile : batchFiles) {
            if (currentIndex >= numData) break;

            System.out.println("Loading batch: " + batchFile.getName());
            try (InputStream inputStream = new FileInputStream(batchFile)) {
                byte[] buffer = new byte[imageSize + 1];
                while (currentIndex < numData && inputStream.read(buffer) == buffer.length) {
                    int label = buffer[0] & 0xFF;
                    Y[label][currentIndex] = 1.0;

                    for (int i = 0; i < imageSize; i++) {
                        X[i][currentIndex] = (buffer[i + 1] & 0xFF) / 255.0;
                    }
                    currentIndex++;
                }
            }
        }

        if (currentIndex < numData) {
            throw new IllegalStateException("Only " + currentIndex + " examples were loaded, but " + numData + " were requested.");
        }

        System.out.println("Loaded " + currentIndex + " examples.");
        Map<String, double[][]> data = new HashMap<>();
        data.put("X", X);
        data.put("Y", Y);

        return data;
    }
}
