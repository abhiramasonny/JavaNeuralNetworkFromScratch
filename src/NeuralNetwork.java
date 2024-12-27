package src;

import java.io.*;
import java.util.*;
import javax.swing.*;
import static src.Utilities.*;

public class NeuralNetwork {

    private int inputSize;
    private int[] hiddenSizes;
    private int outputSize;
    private double learningRate;
    private int epochs;
    private int batchSize;

    private ActivationFunction activationFunction;

    private Map<String, double[][]> parameters = new HashMap<>();
    private Map<String, double[][]> adamCache = new HashMap<>();

    // Hyperparameters for ADAM optimization
    public static double beta1 = 0.9;
    public static double beta2 = 0.999;
    public static double epsilon = 1e-8;

    public enum ActivationFunction {SIGMOID, RELU, TANH}

    public NeuralNetwork(int inputSize, int[] hiddenSizes, int outputSize, double learningRate, int epochs, int batchSize, ActivationFunction activationFunction) {
        this.inputSize = inputSize;
        this.hiddenSizes = hiddenSizes;
        this.outputSize = outputSize;
        this.learningRate = learningRate;
        this.epochs = epochs;
        this.batchSize = batchSize;
        this.activationFunction = activationFunction;
        init();
    }
    private void init() {
        @SuppressWarnings("unused")
        int layers = hiddenSizes.length + 1;
        int[] layerSizes = new int[hiddenSizes.length + 2];
        layerSizes[0] = inputSize;
        System.arraycopy(hiddenSizes, 0, layerSizes, 1, hiddenSizes.length);
        layerSizes[layerSizes.length - 1] = outputSize;

        Random rand = new Random();

        for (int l = 1; l < layerSizes.length; l++) {
            String W = "W" + l;
            String b = "b" + l;

            double[][] weights = new double[layerSizes[l]][layerSizes[l - 1]];
            double[][] biases = new double[layerSizes[l]][1];

            // He initialization for ReLU or Xavier for sigmoid/tanh
            double factor;
            if (activationFunction == ActivationFunction.RELU) {
                factor = Math.sqrt(2.0 / layerSizes[l - 1]);
            } else {
                factor = Math.sqrt(1.0 / layerSizes[l - 1]);
            }

            for (int i = 0; i < weights.length; i++) {
                for (int j = 0; j < weights[0].length; j++) {
                    weights[i][j] = rand.nextGaussian() * factor;
                }
            }

            parameters.put(W, weights);
            parameters.put(b, biases);

            adamCache.put("m" + W, new double[weights.length][weights[0].length]);
            adamCache.put("v" + W, new double[weights.length][weights[0].length]);
            adamCache.put("m" + b, new double[biases.length][biases[0].length]);
            adamCache.put("v" + b, new double[biases.length][biases[0].length]);
        }
    }

    public Map<String, double[][]> forwardProp(double[][] X) {
        Map<String, double[][]> cache = new HashMap<>();
        double[][] A_prev = X;
        cache.put("A0", A_prev);

        int L = hiddenSizes.length + 1;

        for (int l = 1; l <= L; l++) {
            String W = "W" + l;
            String b = "b" + l;

            double[][] Wl = parameters.get(W);
            double[][] bl = parameters.get(b);

            double[][] Z = addVectors(multiplyMatrices(Wl, A_prev), bl);
            cache.put("Z" + l, Z);

            double[][] A;
            if (l == L) {
                A = softmax(Z);
            } else {
                A = activation(Z);
            }
            cache.put("A" + l, A);
            A_prev = A;
        }

        return cache;
    }

    private double[][] activation(double[][] Z) {
        switch (activationFunction) {
            case RELU:
                return relu(Z);
            case SIGMOID:
                return sigmoid(Z);
            case TANH:
                return tanh(Z);
            default:
                return sigmoid(Z);
        }
    }

    private double[][] activationDerivative(double[][] Z) {
        switch (activationFunction) {
            case RELU:
                return reluDerivative(Z);
            case SIGMOID:
                return sigmoidDerivative(Z);
            case TANH:
                return tanhDerivative(Z);
            default:
                return sigmoidDerivative(Z);
        }
    }

    private double[][] softmax(double[][] Z) {
        int m = Z.length;
        int n = Z[0].length;
        double[][] A = new double[m][n];

        for (int j = 0; j < n; j++) {
            double max = Double.NEGATIVE_INFINITY;
            for (int i = 0; i < m; i++) {
                if (Z[i][j] > max) {
                    max = Z[i][j];
                }
            }

            double sumExp = 0.0;
            for (int i = 0; i < m; i++) {
                A[i][j] = Math.exp(Z[i][j] - max);
                sumExp += A[i][j];
            }

            for (int i = 0; i < m; i++) {
                A[i][j] /= sumExp;
            }
        }

        return A;
    }

    public Map<String, double[][]> backProp(double[][] X, double[][] Y, Map<String, double[][]> cache) {
        Map<String, double[][]> gradients = new HashMap<>();
        int m = X[0].length;
        int L = hiddenSizes.length + 1;

        double[][] dA_prev = null;

        double[][] A_L = cache.get("A" + L);
        double[][] dZ_L = subtractMatrices(A_L, Y);
        gradients.put("dZ" + L, dZ_L);
        double[][] A_prev_L = cache.get("A" + (L - 1));
        double[][] dW_L = scalarMultiply(multiplyMatrices(dZ_L, transpose(A_prev_L)), 1.0 / m);
        double[][] db_L = scalarMultiply(sumColumns(dZ_L), 1.0 / m);
        gradients.put("dW" + L, dW_L);
        gradients.put("db" + L, db_L);

        dA_prev = multiplyMatrices(transpose(parameters.get("W" + L)), dZ_L);

        for (int l = L - 1; l >= 1; l--) {
            @SuppressWarnings("unused")
            String Wl_plus1 = "W" + (l + 1);
            double[][] Zl = cache.get("Z" + l);
            double[][] dZl = multiplyElementWise(dA_prev, activationDerivative(Zl));
            gradients.put("dZ" + l, dZl);

            double[][] A_prev_l = cache.get("A" + (l - 1));
            double[][] dWl = scalarMultiply(multiplyMatrices(dZl, transpose(A_prev_l)), 1.0 / m);
            double[][] dbl = scalarMultiply(sumColumns(dZl), 1.0 / m);

            gradients.put("dW" + l, dWl);
            gradients.put("db" + l, dbl);

            if (l > 1) {
                dA_prev = multiplyMatrices(transpose(parameters.get("W" + l)), dZl);
            }
        }

        return gradients;
    }

    public void updateParameters(Map<String, double[][]> gradients, int t) {
        for (String key : parameters.keySet()) {
            double[][] theta = parameters.get(key);
            double[][] dtheta = gradients.get("d" + key);
            double[][] m = adamCache.get("m" + key);
            double[][] v = adamCache.get("v" + key);

            m = addMatrices(scalarMultiply(m, beta1), scalarMultiply(dtheta, 1 - beta1));
            v = addMatrices(scalarMultiply(v, beta2), scalarMultiply(squareMatrix(dtheta), 1 - beta2));

            double[][] m_hat = scalarDivide(m, (1 - Math.pow(beta1, t)));
            double[][] v_hat = scalarDivide(v, (1 - Math.pow(beta2, t)));

            double[][] update = scalarMultiply(elementWiseDivide(m_hat, addScalar(sqrtMatrix(v_hat), epsilon)), learningRate);
            theta = subtractMatrices(theta, update);

            parameters.put(key, theta);
            adamCache.put("m" + key, m);
            adamCache.put("v" + key, v);
        }
    }

    public void train(double[][] X, double[][] Y) {
        int m = X[0].length;
        int t = 0;
    
        MetricsVisualizer visualizer = new MetricsVisualizer(epochs);
        visualizer.setVisible(true);
    
        for (int epoch = 0; epoch < epochs; epoch++) {
            int[] permutation = getRandomPermutation(m);
            double[][] X_shuffled = shuffleColumns(X, permutation);
            double[][] Y_shuffled = shuffleColumns(Y, permutation);
    
            double totalLoss = 0.0;
    
            for (int i = 0; i < m; i += batchSize) {
                t++;
                int end = Math.min(i + batchSize, m);
                double[][] X_batch = getBatch(X_shuffled, i, end);
                double[][] Y_batch = getBatch(Y_shuffled, i, end);
    
                Map<String, double[][]> cache = forwardProp(X_batch);
                double cost = computeCost(cache.get("A" + (hiddenSizes.length + 1)), Y_batch);
                Map<String, double[][]> gradients = backProp(X_batch, Y_batch, cache);
                updateParameters(gradients, t);
    
                totalLoss += cost;
            }
    
            double epochLoss = totalLoss / (m / batchSize);
            int[] predictions = predict(X);
            int[] labels = convertOneHotToLabels(Y);
            double accuracy = computeAccuracy(predictions, labels);
    
            System.out.println("Epoch " + (epoch + 1) + ": Loss = " + epochLoss + ", Accuracy = " + accuracy);
            visualizer.logMetrics(epochLoss, accuracy);
        }
    }

    public int[] predict(double[][] X) {
        Map<String, double[][]> cache = forwardProp(X);
        double[][] A_final = cache.get("A" + (hiddenSizes.length + 1));
        int m = A_final[0].length;
        int[] predictions = new int[m];

        for (int j = 0; j < m; j++) {
            double max = Double.NEGATIVE_INFINITY;
            int maxIndex = -1;
            for (int i = 0; i < A_final.length; i++) {
                if (A_final[i][j] > max) {
                    max = A_final[i][j];
                    maxIndex = i;
                }
            }
            predictions[j] = maxIndex;
        }
        return predictions;
    }

    public static void main(String[] args) {
        try {
            String trainImagesPath = "data/train-images.idx3-ubyte";
            String trainLabelsPath = "data/train-labels.idx1-ubyte";
            int numTrainingExamples = 12000, inputSize = 28 * 28, outputSize = 10, epochs = 3, batchSize = 32;
            double splitRatio = 0.8, learningRate = 0.001;
            int numTrain = (int) (numTrainingExamples * splitRatio), numTest = numTrainingExamples - numTrain;
            int[] hiddenSizes = {128, 64, 64, 32};
            ActivationFunction func = ActivationFunction.RELU;
    
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
    
            NeuralNetwork nn = new NeuralNetwork(inputSize, hiddenSizes, outputSize, learningRate, epochs, batchSize, func);
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
    
            for (int i = 0; i <= 100 && i < trainPredictions.length; i++) {
                double[] image = new double[inputSize];
                for (int j = 0; j < inputSize; j++) {
                    image[j] = X_train[j][i];
                }
                JFrame displayed = MetricsVisualizer.displayImg(image, trainPredictions[i]);
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