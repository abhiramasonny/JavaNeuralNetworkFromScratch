package src;
import static src.NeuralNetwork.epsilon;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

public class Utilities {
    private static final Random rand = new Random();

    //fast log
    final static int significandBits = 52;
    final static int tableBits = 10;
    final static int fractionBits = significandBits - tableBits;
    final static double log2 = 0.69314718056;
    final static double [] table = new double [(1 << tableBits) + 1];
    static {
        for (int bits = 0;bits <= 1 << tableBits;bits++)
            table [bits] = Math.log (1 + bits / (double) (1 << tableBits));
    }

    public static double fastLog (double x) {
        long bits = Double.doubleToLongBits (x);
        if ((bits & (1L << 63)) != 0)
            throw new IllegalArgumentException ("logarithm of negative number");
        int exponent = (int) (bits >> 52);
        int index = (int) (bits >> fractionBits) & ((1 << tableBits) - 1);
        double fraction = (bits & ((1 << fractionBits) - 1)) / (double) (1 << fractionBits);
        return (exponent - 1023) * log2 + fraction * table [index + 1] + (1 - fraction) * table [index];
    }
    public static double computeCost(double[][] Y_hat, double[][] Y) {
        int m = Y[0].length;
        double cost = 0.0;

        for (int i = 0; i < Y.length; i++) {
            double[] Y_row = Y[i];
            double[] Y_hat_row = Y_hat[i];
            for (int j = 0; j < m; j++) {
                cost -= Y_row[j] * fastLog(Y_hat_row[j] + epsilon);
            }
        }
        cost /= m;
        return cost;
    }

    public static double computeAccuracy(int[] predictions, int[] labels) {
        int correct = 0;
        for (int i = 0; i < predictions.length; i++) {
            if (predictions[i] == labels[i]) {
                correct++;
            }
        }
        return ((double) correct) / predictions.length;
    }

    public static int[][] computeConfusionMatrix(int[] predictions, int[] labels, int numClasses) {
        int[][] matrix = new int[numClasses][numClasses];
        for (int i = 0; i < predictions.length; i++) {
            matrix[labels[i]][predictions[i]]++;
        }
        return matrix;
    }
    
    public static double[][] multiplyMatrices(double[][] A, double[][] B) {
        int rows = A.length;
        int sharedDim = A[0].length;
        int cols = B[0].length;
        double[][] C = new double[rows][cols];

        // Transpose B to improve cache performance
        double[][] B_T = new double[cols][sharedDim];
        for (int i = 0; i < sharedDim; i++) {
            for (int j = 0; j < cols; j++) {
                B_T[j][i] = B[i][j];
            }
        }

        for (int i = 0; i < rows; i++) {
            double[] A_row = A[i];
            double[] C_row = C[i];
            for (int j = 0; j < cols; j++) {
                double[] B_T_row = B_T[j];
                double sum = 0.0;
                for (int k = 0; k < sharedDim; k++) {
                    sum += A_row[k] * B_T_row[k];
                }
                C_row[j] = sum;
            }
        }
        return C;
    }

    public static double[][] transpose(double[][] A) {
        int m = A.length;
        int n = A[0].length;
        double[][] T = new double[n][m];

        for (int i = 0; i < m; i++) {
            double[] A_row = A[i];
            for (int j = 0; j < n; j++) {
                T[j][i] = A_row[j];
            }
        }
        return T;
    }

    public static double[][] addVectors(double[][] A, double[][] b) {
        int m = A.length;
        int n = A[0].length;
        double[][] C = new double[m][n];

        for (int i = 0; i < m; i++) {
            double bias = b[i][0];
            double[] A_row = A[i];
            double[] C_row = C[i];
            for (int j = 0; j < n; j++) {
                C_row[j] = A_row[j] + bias;
            }
        }
        return C;
    }

    public static double[][] subtractMatrices(double[][] A, double[][] B) {
        int m = A.length;
        int n = A[0].length;
        double[][] C = new double[m][n];

        for (int i = 0; i < m; i++) {
            double[] A_row = A[i], B_row = B[i], C_row = C[i];
            for (int j = 0; j < n; j++) {
                C_row[j] = A_row[j] - B_row[j];
            }
        }
        return C;
    }

    public static double[][] addMatrices(double[][] A, double[][] B) {
        int m = A.length;
        int n = A[0].length;
        double[][] C = new double[m][n];

        for (int i = 0; i < m; i++) {
            double[] A_row = A[i], B_row = B[i], C_row = C[i];
            for (int j = 0; j < n; j++) {
                C_row[j] = A_row[j] + B_row[j];
            }
        }
        return C;
    }

    public static double[][] multiplyElementWise(double[][] A, double[][] B) {
        int m = A.length;
        int n = A[0].length;
        double[][] C = new double[m][n];

        for (int i = 0; i < m; i++) {
            double[] A_row = A[i], B_row = B[i], C_row = C[i];
            for (int j = 0; j < n; j++) {
                C_row[j] = A_row[j] * B_row[j];
            }
        }
        return C;
    }

    public static double[][] elementWiseDivide(double[][] A, double[][] B) {
        int m = A.length;
        int n = A[0].length;
        double[][] C = new double[m][n];

        for (int i = 0; i < m; i++) {
            double[] A_row = A[i], B_row = B[i], C_row = C[i];
            for (int j = 0; j < n; j++) {
                C_row[j] = A_row[j] / B_row[j];
            }
        }
        return C;
    }

    public static double[][] scalarMultiply(double[][] A, double scalar) {
        int m = A.length;
        int n = A[0].length;
        double[][] C = new double[m][n];
    
        for (int i = 0; i < m; i++) {
            double[] A_row = A[i];
            double[] C_row = C[i];
            for (int j = 0; j < n; j++) {
                C_row[j] = A_row[j] * scalar;
            }
        }
        return C;
    }

    public static double[][] scalarDivide(double[][] A, double scalar) {
        int m = A.length;
        int n = A[0].length;
        double[][] C = new double[m][n];
        double invScalar = 1.0 / scalar; // Precompute reciprocal
        for (int i = 0; i < m; i++) {
            double[] A_row = A[i];
            double[] C_row = C[i];
            for (int j = 0; j < n; j++) {
                C_row[j] = A_row[j] * invScalar;
            }
        }
        return C;
    }

    public static double[][] addScalar(double[][] A, double scalar) {
        int m = A.length;
        int n = A[0].length;
        double[][] C = new double[m][n];
    
        for (int i = 0; i < m; i++) {
            double[] A_row = A[i];
            double[] C_row = C[i];
            for (int j = 0; j < n; j++) {
                C_row[j] = A_row[j] + scalar;
            }
        }
        return C;
    }

    public static double[][] subtractScalar(double[][] A, double scalar) {
        int m = A.length;
        int n = A[0].length;
        double[][] C = new double[m][n];
    
        for (int i = 0; i < m; i++) {
            double[] A_row = A[i];
            double[] C_row = C[i];
            for (int j = 0; j < n; j++) {
                C_row[j] = A_row[j] - scalar;
            }
        }
        return C;
    }

    public static double[][] sqrtMatrix(double[][] A) {
        int m = A.length;
        int n = A[0].length;
        double[][] C = new double[m][n];
    
        for (int i = 0; i < m; i++) {
            double[] A_row = A[i];
            double[] C_row = C[i];
            for (int j = 0; j < n; j++) {
                C_row[j] = Math.sqrt(A_row[j]);
            }
        }
        return C;
    }

    public static double[][] squareMatrix(double[][] A) {
        int m = A.length;
        int n = A[0].length;
        double[][] C = new double[m][n];
        for (int i = 0; i < m; i++) {
            double[] A_row = A[i];
            double[] C_row = C[i];
            for (int j = 0; j < n; j++) {
                double val = A_row[j];
                C_row[j] = val * val;
            }
        }
        return C;
    }

    // Sum over columns (for biases)
    public static double[][] sumColumns(double[][] A) {
        int m = A.length;
        int n = A[0].length;
        double[][] sum = new double[m][1];

        for (int i = 0; i < m; i++) {
            double[] A_row = A[i];
            double s = 0.0;
            for (int j = 0; j < n; j++) {
                s += A_row[j];
            }
            sum[i][0] = s;
        }
        return sum;
    }

    public static int[] getRandomPermutation(int m) {
        int[] permutation = new int[m];
        for (int i = 0; i < m; i++) {
            permutation[i] = i;
        }
        for (int i = m - 1; i > 0; i--) {
            int index = rand.nextInt(i + 1);
            int temp = permutation[index];
            permutation[index] = permutation[i];
            permutation[i] = temp;
        }
        return permutation;
    }

    public static double[][] shuffleColumns(double[][] A, int[] permutation) {
        int m = A.length;
        int n = A[0].length;
        double[][] shuffled = new double[m][n];
        for (int i = 0; i < n; i++) {
            int permutedIndex = permutation[i];
            for (int j = 0; j < m; j++) {
                shuffled[j][i] = A[j][permutedIndex];
            }
        }
        return shuffled;
    }

    public static double[][] getBatch(double[][] A, int start, int end) {
        int m = A.length;
        int n = end - start;
        double[][] batch = new double[m][n];
        for (int i = 0; i < m; i++) {
            System.arraycopy(A[i], start, batch[i], 0, n);
        }
        return batch;
    }

    public static int[] convertOneHotToLabels(double[][] Y) {
        int m = Y.length;
        int n = Y[0].length;
        int[] labels = new int[n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                if (Y[j][i] == 1.0) {
                    labels[i] = j;
                    break;
                }
            }
        }
        return labels;
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
        labelBuffer.getInt(); // Number of labels, should be equal to numImages

        int totalData = Math.min(numData, numImages);
        int imageSize = numRows * numCols;
        double[][] X = new double[imageSize][totalData];
        double[][] Y = new double[10][totalData];

        for (int i = 0; i < totalData; i++) {
            for (int j = 0; j < imageSize; j++) {
                int pixel = imageBuffer.get() & 0xFF;
                X[j][i] = pixel / 255.0;
            }
            int label = labelBuffer.get() & 0xFF;
            Y[label][i] = 1.0;
        }

        Map<String, double[][]> data = new HashMap<>();
        data.put("X", X);
        data.put("Y", Y);

        return data;
    }
}