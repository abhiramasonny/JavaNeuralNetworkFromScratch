package src;

import static src.Utilities.*;
import java.io.IOException;
import java.util.Map;
import javax.swing.JFrame;
import src.NeuralNetwork.ActivationFunction;

public class Main {
    public static void main(String[] args) {
        try {
            String trainImagesPath = "data/train-images.idx3-ubyte";
            String trainLabelsPath = "data/train-labels.idx1-ubyte";
            int numTrainingExamples = 12000, inputSize = 28 * 28, outputSize = 10, epochs = 10, batchSize = 64;
            double splitRatio = 0.8, learningRate = 0.001;
            int numTrain = (int) (numTrainingExamples * splitRatio), numTest = numTrainingExamples - numTrain;
            int[] hiddenSizes = {256, 128, 64};
            ActivationFunction[] funcs = {
                ActivationFunction.LEAKY_RELU,
                ActivationFunction.LEAKY_RELU,
                ActivationFunction.LEAKY_RELU,
                ActivationFunction.SOFTMAX // Output layer activation
            };

            Map<String, double[][]> data = loadMNIST(trainImagesPath, trainLabelsPath, numTrainingExamples);
            double[][] X = data.get("X"), Y = data.get("Y");
            double[][] X_train = new double[X.length][numTrain], Y_train = new double[Y.length][numTrain];
            double[][] X_test = new double[X.length][numTest], Y_test = new double[Y.length][numTest];
            int[] indices = getRandomPermutation(numTrainingExamples);

            for (int i = 0; i < numTrainingExamples; i++) {
                double[][] X_target = i < numTrain ? X_train : X_test, Y_target = i < numTrain ? Y_train : Y_test;
                int targetIndex = i < numTrain ? i : i - numTrain;
                for (int j = 0; j < X.length; j++) X_target[j][targetIndex] = X[j][indices[i]];
                for (int j = 0; j < Y.length; j++) Y_target[j][targetIndex] = Y[j][indices[i]];
            }

            NeuralNetwork nn = new NeuralNetwork(inputSize, hiddenSizes, outputSize, learningRate, epochs, batchSize, funcs);
            System.out.println("Starting training...");
            nn.train(X_train, Y_train);
            System.out.println("Training completed.");

            int[] trainPredictions = nn.predict(X_train);
            int[] trainLabels = convertOneHotToLabels(Y_train);
            double trainAccuracy = computeAccuracy(trainPredictions, trainLabels);
            System.out.println("Training accuracy: " + trainAccuracy);

            int[] testPredictions = nn.predict(X_test);
            int[] testLabels = convertOneHotToLabels(Y_test);
            double testAccuracy = computeAccuracy(testPredictions, testLabels);
            System.out.println("Test accuracy: " + testAccuracy);
            int[][] confusionMatrix = computeConfusionMatrix(testPredictions, testLabels, outputSize);
            MetricsVisualizer.displayConfusionMatrix(confusionMatrix, new String[]{"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"});

            // Optionally display some images with predictions
            for (int i = 0; i <= 10 && i < testPredictions.length; i++) {
                double[] image = new double[inputSize];
                for (int j = 0; j < inputSize; j++) {
                    image[j] = X_test[j][i];
                }
                JFrame displayed = MetricsVisualizer.displayImg(image, testPredictions[i]);
                try {
                    Thread.sleep(1000);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                } finally {
                    displayed.dispose();
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
