package src;

import static src.Utilities.*;



import java.util.HashMap;
import java.util.Map;
import java.util.Random;


public class NeuralNetwork {

    public static final double BETA1 = 0.9;            // Exponential decay rate for the first moment estimates
    public static final double BETA2 = 0.999;          // Exponential decay rate for the second moment estimates
    public static final double EPSILON = 1e-8;         // Small constant to prevent division by zero
    public static final double LEAKY_RELU_ALPHA = 0.01; // Alpha value for Leaky ReLU
    public static final double DEFAULT_ALPHA = 0.1;     // Default alpha for PReLU and ELU
    public static final int EARLY_STOPPING_PATIENCE = 5; // Patience for early stopping

    public final int inputSize;
    public final int[] hiddenSizes;
    public final int outputSize;
    public final double learningRate;
    public final int epochs;
    public final int batchSize;

    public final ActivationFunction[] activationFunctions;

    public final Map<String, double[][]> parameters = new HashMap<>();
    public final Map<String, double[][]> adamCache = new HashMap<>();
    public final Map<Integer, Double> alphaParameters = new HashMap<>();

    public enum ActivationFunction {
        SIGMOID, RELU, TANH, LEAKY_RELU, ELU, PRELU, SOFTMAX
    }

    public NeuralNetwork(int inputSize, int[] hiddenSizes, int outputSize,
                         double learningRate, int epochs, int batchSize,
                         ActivationFunction[] activationFunctions) {
        this.inputSize = inputSize;
        this.hiddenSizes = hiddenSizes.clone();
        this.outputSize = outputSize;
        this.learningRate = learningRate;
        this.epochs = epochs;
        this.batchSize = batchSize;
        this.activationFunctions = activationFunctions.clone();
        init();
    }

    private void init() {
        int layers = hiddenSizes.length + 1;
        int[] layerSizes = new int[layers + 1];
        layerSizes[0] = inputSize;
        System.arraycopy(hiddenSizes, 0, layerSizes, 1, hiddenSizes.length);
        layerSizes[layerSizes.length - 1] = outputSize;

        Random rand = new Random();

        for (int l = 1; l < layerSizes.length; l++) {
            String W = "W" + l;
            String b = "b" + l;

            int currentLayerSize = layerSizes[l];
            int previousLayerSize = layerSizes[l - 1];

            double[][] weights = new double[currentLayerSize][previousLayerSize];
            double[][] biases = new double[currentLayerSize][1];

            // He initialization for ReLU/variants or Xavier for sigmoid/tanh
            double factor;
            ActivationFunction af = activationFunctions[Math.min(l - 1, activationFunctions.length - 1)];
            if (af == ActivationFunction.RELU || af == ActivationFunction.LEAKY_RELU
                    || af == ActivationFunction.PRELU || af == ActivationFunction.ELU) {
                factor = Math.sqrt(2.0 / previousLayerSize);
            } else {
                factor = Math.sqrt(1.0 / previousLayerSize);
            }

            for (int i = 0; i < currentLayerSize; i++) {
                for (int j = 0; j < previousLayerSize; j++) {
                    weights[i][j] = rand.nextGaussian() * factor;
                }
            }

            parameters.put(W, weights);
            parameters.put(b, biases);

            adamCache.put("m" + W, new double[currentLayerSize][previousLayerSize]);
            adamCache.put("v" + W, new double[currentLayerSize][previousLayerSize]);
            adamCache.put("m" + b, new double[currentLayerSize][1]);
            adamCache.put("v" + b, new double[currentLayerSize][1]);
        }

        // Initialize alpha parameters for PReLU and ELU
        for (int l = 1; l <= layers; l++) {
            ActivationFunction af = activationFunctions[Math.min(l - 1, activationFunctions.length - 1)];
            if (af == ActivationFunction.PRELU || af == ActivationFunction.ELU) {
                alphaParameters.put(l, DEFAULT_ALPHA);
            }
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

            ActivationFunction af = activationFunctions[Math.min(l - 1, activationFunctions.length - 1)];
            double[][] A;
            if (af == ActivationFunction.SOFTMAX) {
                A = softmax(Z);
            } else {
                A = activation(Z, af, l); // Pass layer index l
            }
            cache.put("A" + l, A);
            A_prev = A;
        }

        return cache;
    }

    private double[][] activation(double[][] Z, ActivationFunction af, int layer) {
        switch (af) {
            case RELU: return relu(Z);
            case SIGMOID: return sigmoid(Z);
            case TANH: return tanh(Z);
            case LEAKY_RELU: return leakyRelu(Z);
            case ELU: return elu(Z, alphaParameters.getOrDefault(layer, DEFAULT_ALPHA));
            case PRELU: return prelu(Z, alphaParameters.getOrDefault(layer, DEFAULT_ALPHA));
            default: return sigmoid(Z);
        }
    }

    private double[][] activationDerivative(double[][] Z, ActivationFunction af, int layer) {
        switch (af) {
            case RELU: return reluDerivative(Z);
            case SIGMOID: return sigmoidDerivative(Z);
            case TANH: return tanhDerivative(Z);
            case LEAKY_RELU: return leakyReluDerivative(Z);
            case ELU: return eluDerivative(Z, alphaParameters.getOrDefault(layer, DEFAULT_ALPHA));
            case PRELU: return preluDerivative(Z, alphaParameters.getOrDefault(layer, DEFAULT_ALPHA));
            default: return sigmoidDerivative(Z);
        }
    }

    public Map<String, double[][]> backProp(double[][] X, double[][] Y, Map<String, double[][]> cache) {
        Map<String, double[][]> gradients = new HashMap<>();
        int m = X[0].length;
        int L = hiddenSizes.length + 1;

        double[][] dA_prev;

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
            double[][] Zl = cache.get("Z" + l);
            ActivationFunction af = activationFunctions[Math.min(l - 1, activationFunctions.length - 1)];
            double[][] dZl = multiplyElementWise(dA_prev, activationDerivative(Zl, af, l));
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
            if (dtheta == null) continue;

            double[][] m = adamCache.get("m" + key);
            double[][] v = adamCache.get("v" + key);

            m = addMatrices(scalarMultiply(m, BETA1), scalarMultiply(dtheta, 1 - BETA1));
            v = addMatrices(scalarMultiply(v, BETA2), scalarMultiply(squareMatrix(dtheta), 1 - BETA2));

            double[][] m_hat = scalarDivide(m, (1 - Math.pow(BETA1, t)));
            double[][] v_hat = scalarDivide(v, (1 - Math.pow(BETA2, t)));

            double[][] update = scalarMultiply(
                    elementWiseDivide(m_hat, addScalar(sqrtMatrix(v_hat), EPSILON)),
                    learningRate);
            theta = subtractMatrices(theta, update);

            parameters.put(key, theta);
            adamCache.put("m" + key, m);
            adamCache.put("v" + key, v);
        }

        for (Map.Entry<Integer, Double> entry : alphaParameters.entrySet()) {
            int layer = entry.getKey();
            double alpha = entry.getValue();
            alphaParameters.put(layer, alpha);
        }
    }

    public void train(double[][] X, double[][] Y) {
        int m = X[0].length;
        int t = 0;

        MetricsVisualizer visualizer = new MetricsVisualizer(epochs);
        visualizer.setVisible(true);

        // Early stopping stuff
        double bestLoss = Double.MAX_VALUE;
        int wait = 0;

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

            System.out.println("Epoch " + (epoch + 1) +
                    ": Loss = " + epochLoss +
                    ", Accuracy = " + accuracy);
            visualizer.logMetrics(epochLoss, accuracy);

            if (epochLoss < bestLoss) {
                bestLoss = epochLoss;
                wait = 0;
            } else {
                wait++;
                if (wait >= EARLY_STOPPING_PATIENCE) {
                    System.out.println("Early stopping at epoch " + (epoch + 1));
                    break;
                }
            }
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
}