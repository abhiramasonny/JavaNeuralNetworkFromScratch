import java.io.*;
import java.util.*;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Paths;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;

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
    private double beta1 = 0.9;
    private double beta2 = 0.999;
    private double epsilon = 1e-8;

    public enum ActivationFunction {SIGMOID, RELU, TANH}

    public NeuralNetwork(int inputSize, int[] hiddenSizes, int outputSize, double learningRate, int epochs, int batchSize, ActivationFunction activationFunction) {
        this.inputSize = inputSize;
        this.hiddenSizes = hiddenSizes;
        this.outputSize = outputSize;
        this.learningRate = learningRate;
        this.epochs = epochs;
        this.batchSize = batchSize;
        this.activationFunction = activationFunction;

        initializeParameters();
    }

    private void initializeParameters() {
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

    public Map<String, double[][]> forwardPropagation(double[][] X) {
        Map<String, double[][]> cache = new HashMap<>();
        double[][] A_prev = X;
        cache.put("A0", A_prev);

        int L = hiddenSizes.length + 1;

        for (int l = 1; l <= L; l++) {
            String W = "W" + l;
            String b = "b" + l;

            double[][] Wl = parameters.get(W);
            double[][] bl = parameters.get(b);

            double[][] Z = MatrixUtilities.addVectors(MatrixUtilities.multiplyMatrices(Wl, A_prev), bl);
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

    private double[][] sigmoid(double[][] Z) {
        int m = Z.length;
        int n = Z[0].length;
        double[][] A = new double[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double val = 1.0 / (1.0 + Math.exp(-Z[i][j]));
                A[i][j] = val;
            }
        }
        return A;
    }

    private double[][] sigmoidDerivative(double[][] Z) {
        double[][] S = sigmoid(Z);
        return MatrixUtilities.multiplyElementWise(S, MatrixUtilities.subtractScalar(S, 1.0));
    }

    private double[][] relu(double[][] Z) {
        int m = Z.length;
        int n = Z[0].length;
        double[][] A = new double[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                A[i][j] = Math.max(0, Z[i][j]);
            }
        }
        return A;
    }

    private double[][] reluDerivative(double[][] Z) {
        int m = Z.length;
        int n = Z[0].length;
        double[][] dZ = new double[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                dZ[i][j] = Z[i][j] > 0 ? 1.0 : 0.0;
            }
        }
        return dZ;
    }

    private double[][] tanh(double[][] Z) {
        int m = Z.length;
        int n = Z[0].length;
        double[][] A = new double[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                A[i][j] = Math.tanh(Z[i][j]);
            }
        }
        return A;
    }

    private double[][] tanhDerivative(double[][] Z) {
        double[][] A = tanh(Z);
        return MatrixUtilities.subtractScalar(MatrixUtilities.multiplyElementWise(A, A), 1.0);
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

    public Map<String, double[][]> backwardPropagation(double[][] X, double[][] Y, Map<String, double[][]> cache) {
        Map<String, double[][]> gradients = new HashMap<>();
        int m = X[0].length;
        int L = hiddenSizes.length + 1;

        double[][] dA_prev = null;

        double[][] A_L = cache.get("A" + L);
        double[][] dZ_L = MatrixUtilities.subtractMatrices(A_L, Y);
        gradients.put("dZ" + L, dZ_L);
        double[][] A_prev_L = cache.get("A" + (L - 1));
        double[][] dW_L = MatrixUtilities.scalarMultiply(MatrixUtilities.multiplyMatrices(dZ_L, MatrixUtilities.transpose(A_prev_L)), 1.0 / m);
        double[][] db_L = MatrixUtilities.scalarMultiply(MatrixUtilities.sumColumns(dZ_L), 1.0 / m);
        gradients.put("dW" + L, dW_L);
        gradients.put("db" + L, db_L);

        dA_prev = MatrixUtilities.multiplyMatrices(MatrixUtilities.transpose(parameters.get("W" + L)), dZ_L);

        for (int l = L - 1; l >= 1; l--) {
            @SuppressWarnings("unused")
            String Wl_plus1 = "W" + (l + 1);
            double[][] Zl = cache.get("Z" + l);
            double[][] dZl = MatrixUtilities.multiplyElementWise(dA_prev, activationDerivative(Zl));
            gradients.put("dZ" + l, dZl);

            double[][] A_prev_l = cache.get("A" + (l - 1));
            double[][] dWl = MatrixUtilities.scalarMultiply(MatrixUtilities.multiplyMatrices(dZl, MatrixUtilities.transpose(A_prev_l)), 1.0 / m);
            double[][] dbl = MatrixUtilities.scalarMultiply(MatrixUtilities.sumColumns(dZl), 1.0 / m);

            gradients.put("dW" + l, dWl);
            gradients.put("db" + l, dbl);

            if (l > 1) {
                dA_prev = MatrixUtilities.multiplyMatrices(MatrixUtilities.transpose(parameters.get("W" + l)), dZl);
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

            m = MatrixUtilities.addMatrices(MatrixUtilities.scalarMultiply(m, beta1), MatrixUtilities.scalarMultiply(dtheta, 1 - beta1));
            v = MatrixUtilities.addMatrices(MatrixUtilities.scalarMultiply(v, beta2), MatrixUtilities.scalarMultiply(MatrixUtilities.squareMatrix(dtheta), 1 - beta2));

            double[][] m_hat = MatrixUtilities.scalarDivide(m, (1 - Math.pow(beta1, t)));
            double[][] v_hat = MatrixUtilities.scalarDivide(v, (1 - Math.pow(beta2, t)));

            double[][] update = MatrixUtilities.scalarMultiply(MatrixUtilities.elementWiseDivide(m_hat, MatrixUtilities.addScalar(MatrixUtilities.sqrtMatrix(v_hat), epsilon)), learningRate);
            theta = MatrixUtilities.subtractMatrices(theta, update);

            parameters.put(key, theta);
            adamCache.put("m" + key, m);
            adamCache.put("v" + key, v);
        }
    }

    public double computeCost(double[][] Y_hat, double[][] Y) {
        int m = Y[0].length;
        double cost = 0.0;

        for (int i = 0; i < Y.length; i++) {
            for (int j = 0; j < m; j++) {
                cost -= Y[i][j] * Math.log(Y_hat[i][j] + epsilon);
            }
        }
        cost /= m;
        return cost;
    }

    public void train(double[][] X, double[][] Y) {
        int m = X[0].length;
        int t = 0;

        for (int epoch = 0; epoch < epochs; epoch++) {
            int[] permutation = MatrixUtilities.getRandomPermutation(m);
            double[][] X_shuffled = MatrixUtilities.shuffleColumns(X, permutation);
            double[][] Y_shuffled = MatrixUtilities.shuffleColumns(Y, permutation);

            for (int i = 0; i < m; i += batchSize) {
                t++;

                int end = Math.min(i + batchSize, m);
                double[][] X_batch = MatrixUtilities.getBatch(X_shuffled, i, end);
                double[][] Y_batch = MatrixUtilities.getBatch(Y_shuffled, i, end);

                Map<String, double[][]> cache = forwardPropagation(X_batch);
                double cost = computeCost(cache.get("A" + (hiddenSizes.length + 1)), Y_batch);
                Map<String, double[][]> gradients = backwardPropagation(X_batch, Y_batch, cache);

                updateParameters(gradients, t);
                if (i%100==0) {
                    System.out.println("cost: "+ cost);
                }
            }
            System.out.println("Epoch " + (epoch + 1) + "/" + epochs + " completed.");
        }
    }
    public void displayImageAndPrediction(double[] image, int prediction) {
        int width = 28;
        int height = 28;
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
    }

    public int[] predict(double[][] X) {
        Map<String, double[][]> cache = forwardPropagation(X);
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

    public double computeAccuracy(int[] predictions, int[] labels) {
        int correct = 0;
        for (int i = 0; i < predictions.length; i++) {
            if (predictions[i] == labels[i]) {
                correct++;
            }
        }
        return ((double) correct) / predictions.length;
    }

    public static Map<String, double[][]> loadMNIST(String imagesPath, String labelsPath, int numData) throws IOException {
        byte[] imageBytes = Files.readAllBytes(Paths.get(imagesPath));
        ByteBuffer imageBuffer = ByteBuffer.wrap(imageBytes);
        byte[] labelBytes = Files.readAllBytes(Paths.get(labelsPath));
        ByteBuffer labelBuffer = ByteBuffer.wrap(labelBytes);

        imageBuffer.getInt(); // Magic number
        int numImages = imageBuffer.getInt();
        int numRows = imageBuffer.getInt();
        int numCols = imageBuffer.getInt();

        labelBuffer.getInt(); // Magic number
        @SuppressWarnings("unused")
        int numLabels = labelBuffer.getInt();

        int totalData = Math.min(numData, numImages);
        double[][] X = new double[numRows * numCols][totalData];
        double[][] Y = new double[10][totalData];

        for (int i = 0; i < totalData; i++) {
            for (int j = 0; j < numRows * numCols; j++) {
                int pixel = imageBuffer.get() & 0xFF;
                X[j][i] = pixel / 255.0;
            }
            int label = labelBuffer.get() & 0xFF;
            for (int k = 0; k < 10; k++) {
                Y[k][i] = (k == label) ? 1.0 : 0.0;
            }
        }

        Map<String, double[][]> data = new HashMap<>();
        data.put("X", X);
        data.put("Y", Y);

        return data;
    }

    public static void main(String[] args) {
        try {
            String trainImagesPath = "data/train-images.idx3-ubyte";
            String trainLabelsPath = "data/train-labels.idx1-ubyte";
    
            int numTrainingExamples = 6000;
    
            Map<String, double[][]> data = loadMNIST(trainImagesPath, trainLabelsPath, numTrainingExamples);
            double[][] X = data.get("X");
            double[][] Y = data.get("Y");
            
            double splitRatio = 0.8;
            int numTrain = (int) (numTrainingExamples * splitRatio);
            int numTest = numTrainingExamples - numTrain;
    
            double[][] X_train = new double[X.length][numTrain];
            double[][] Y_train = new double[Y.length][numTrain];
            double[][] X_test = new double[X.length][numTest];
            double[][] Y_test = new double[Y.length][numTest];
    
            int[] indices = MatrixUtilities.getRandomPermutation(numTrainingExamples);
    
            for (int i = 0; i < numTrain; i++) {
                for (int j = 0; j < X.length; j++) {
                    X_train[j][i] = X[j][indices[i]];
                }
                for (int j = 0; j < Y.length; j++) {
                    Y_train[j][i] = Y[j][indices[i]];
                }
            }
    
            for (int i = numTrain; i < numTrainingExamples; i++) {
                for (int j = 0; j < X.length; j++) {
                    X_test[j][i - numTrain] = X[j][indices[i]];
                }
                for (int j = 0; j < Y.length; j++) {
                    Y_test[j][i - numTrain] = Y[j][indices[i]];
                }
            }
    
            System.out.println("Loaded and split the data.");
    
            // Network parameters
            int inputSize = 28 * 28;
            int[] hiddenSizes = {128, 64, 64, 32};
            int outputSize = 10;
            double learningRate = 0.001;
            int epochs = 3;
            int batchSize = 64;
    
            NeuralNetwork nn = new NeuralNetwork(inputSize, hiddenSizes, outputSize, learningRate, epochs, batchSize, ActivationFunction.RELU);
            System.out.println("Starting training...");
            nn.train(X_train, Y_train);
            System.out.println("Training completed.");
    
            int[] trainPredictions = nn.predict(X_train);
            int[] trainLabels = MatrixUtilities.convertOneHotToLabels(Y_train);
            double trainAccuracy = nn.computeAccuracy(trainPredictions, trainLabels);
            System.out.println("Training accuracy: " + trainAccuracy);
    
            int[] testPredictions = nn.predict(X_test);
            int[] testLabels = MatrixUtilities.convertOneHotToLabels(Y_test);
            double testAccuracy = nn.computeAccuracy(testPredictions, testLabels);
            System.out.println("Test accuracy: " + testAccuracy);

            for (int i = 0; i <= 100 && i < trainPredictions.length; i++) {
                double[] image = new double[inputSize];
                for (int j = 0; j < inputSize; j++) {
                    image[j] = X_train[j][i];
                }
                int firstPrediction = trainPredictions[i];
                nn.displayImageAndPrediction(image, firstPrediction);
                try {
                    Thread.sleep(1000);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }

        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}