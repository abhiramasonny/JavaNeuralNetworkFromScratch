package src;

import static src.NeuralNetwork.EPSILON;
import static src.NeuralNetwork.LEAKY_RELU_ALPHA;

import java.util.Random;
import java.util.stream.IntStream;

public class Utilities {
    private static final Random RANDOM = new Random();

    public static double computeCost(double[][] Y_hat, double[][] Y) {
        int m = Y[0].length;
        return IntStream.range(0, Y.length).parallel().mapToDouble(i -> {
            double cost = 0.0;
            double[] Y_row = Y[i];
            double[] Y_hat_row = Y_hat[i];
            for (int j = 0; j < m; j++) {
                cost -= Y_row[j] * Math.log(Y_hat_row[j] + EPSILON);
            }
            return cost;
        }).sum() / m;
    }
    
    public static double[][] sigmoidDerivative(double[][] Z) {
        int m = Z.length;
        int n = Z[0].length;
        double[][] dZ = new double[m][n];
        IntStream.range(0, m).parallel().forEach(i -> {
            for (int j = 0; j < n; j++) {
                double s = 1.0 / (1.0 + Math.exp(-Z[i][j]));
                dZ[i][j] = s * (1.0 - s);
            }
        });
        return dZ;
    }
    
    public static double[][] sigmoid(double[][] Z) {
        int m = Z.length;
        int n = Z[0].length;
        double[][] A = new double[m][n];
        IntStream.range(0, m).parallel().forEach(i -> {
            for (int j = 0; j < n; j++) {
                A[i][j] = 1.0 / (1.0 + Math.exp(-Z[i][j]));
            }
        });
        return A;
    }
    // ReLU activation
    public static double[][] relu(double[][] Z) {
        int m = Z.length;
        int n = Z[0].length;
        double[][] A = new double[m][n];
        IntStream.range(0, m).parallel().forEach(i -> {
            for (int j = 0; j < n; j++) {
                A[i][j] = Math.max(0, Z[i][j]);
            }
        });
        return A;
    }

    // ReLU derivative
    public static double[][] reluDerivative(double[][] Z) {
        int m = Z.length;
        int n = Z[0].length;
        double[][] dZ = new double[m][n];
        IntStream.range(0, m).parallel().forEach(i -> {
            for (int j = 0; j < n; j++) {
                dZ[i][j] = Z[i][j] > 0 ? 1.0 : 0.0;
            }
        });
        return dZ;
    }

    
    public static double[][] tanh(double[][] Z) {
        int m = Z.length;
        int n = Z[0].length;
        double[][] A = new double[m][n];
        for (int i = 0; i < m; i++) {
            double[] Z_row = Z[i];
            double[] A_row = A[i];
            for (int j = 0; j < n; j++) {
                A_row[j] = 1-(2*(1/(1+Math.exp(Z_row[j]*2))));
            }
        }
        return A;
    }
    
    public static double[][] tanhDerivative(double[][] Z) {
        int m = Z.length;
        int n = Z[0].length;
        double[][] dZ = new double[m][n];
        for (int i = 0; i < m; i++) {
            double[] Z_row = Z[i];
            double[] dZ_row = dZ[i];
            for (int j = 0; j < n; j++) {
                double t = 1-(2*(1/(1+Math.exp(Z_row[j]*2))));
                dZ_row[j] = 1.0 - t * t;
            }
        }
        return dZ;
    }

    // Leaky ReLU activation
    public static double[][] leakyRelu(double[][] Z) {
        int m = Z.length;
        int n = Z[0].length;
        double[][] A = new double[m][n];
        IntStream.range(0, m).parallel().forEach(i -> {
            double[] Z_row = Z[i];
            double[] A_row = A[i];
            for (int j = 0; j < n; j++) {
                A_row[j] = Z_row[j] > 0 ? Z_row[j] : LEAKY_RELU_ALPHA * Z_row[j];
            }
        });
        return A;
    }

     // Leaky ReLU derivative
     public static double[][] leakyReluDerivative(double[][] Z) {
        int m = Z.length;
        int n = Z[0].length;
        double[][] dZ = new double[m][n];
        IntStream.range(0, m).parallel().forEach(i -> {
            double[] Z_row = Z[i];
            double[] dZ_row = dZ[i];
            for (int j = 0; j < n; j++) {
                dZ_row[j] = Z_row[j] > 0 ? 1.0 : LEAKY_RELU_ALPHA;
            }
        });
        return dZ;
    }

    // PReLU activation
    public static double[][] prelu(double[][] Z, double alpha) {
        int m = Z.length;
        int n = Z[0].length;
        double[][] A = new double[m][n];
        IntStream.range(0, m).parallel().forEach(i -> {
            double[] Z_row = Z[i];
            double[] A_row = A[i];
            for (int j = 0; j < n; j++) {
                A_row[j] = Z_row[j] > 0 ? Z_row[j] : alpha * Z_row[j];
            }
        });
        return A;
    }

    // PReLU derivative
    public static double[][] preluDerivative(double[][] Z, double alpha) {
        int m = Z.length;
        int n = Z[0].length;
        double[][] dZ = new double[m][n];
        IntStream.range(0, m).parallel().forEach(i -> {
            double[] Z_row = Z[i];
            double[] dZ_row = dZ[i];
            for (int j = 0; j < n; j++) {
                dZ_row[j] = Z_row[j] > 0 ? 1.0 : alpha;
            }
        });
        return dZ;
    }

    // ELU activation
    public static double[][] elu(double[][] Z, double alpha) {
        int m = Z.length;
        int n = Z[0].length;
        double[][] A = new double[m][n];
        IntStream.range(0, m).parallel().forEach(i -> {
            double[] Z_row = Z[i];
            double[] A_row = A[i];
            for (int j = 0; j < n; j++) {
                A_row[j] = Z_row[j] >= 0 ? Z_row[j] : alpha * (Math.exp(Z_row[j]) - 1);
            }
        });
        return A;
    }

    // ELU derivative
    public static double[][] eluDerivative(double[][] Z, double alpha) {
        int m = Z.length;
        int n = Z[0].length;
        double[][] dZ = new double[m][n];
        IntStream.range(0, m).parallel().forEach(i -> {
            double[] Z_row = Z[i];
            double[] dZ_row = dZ[i];
            for (int j = 0; j < n; j++) {
                dZ_row[j] = Z_row[j] >= 0 ? 1.0 : alpha * Math.exp(Z_row[j]);
            }
        });
        return dZ;
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
    public static double[][] softmax(double[][] Z) {
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

    public static double[][] multiplyMatrices(double[][] A, double[][] B) {
        int rowsA = A.length;
        int colsA = A[0].length;
        int colsB = B[0].length;

        double[][] C = new double[rowsA][colsB];
        IntStream.range(0, rowsA).parallel().forEach(i -> {
            for (int j = 0; j < colsB; j++) {
                double sum = 0.0;
                for (int k = 0; k < colsA; k++) {
                    sum += A[i][k] * B[k][j];
                }
                C[i][j] = sum;
            }
        });
        return C;
    }

    // Transpose matrix
    public static double[][] transposeMatrix(double[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        double[][] transposed = new double[cols][rows];

        IntStream.range(0, cols).parallel().forEach(i -> {
            for (int j = 0; j < rows; j++) {
                transposed[i][j] = matrix[j][i];
            }
        });

        return transposed;
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

    public static double[][] addMatrices(double[][] A, double[][] B) {
        return elementWiseOperation(A, B, Double::sum);
    }

    public static double[][] subtractMatrices(double[][] A, double[][] B) {
        return elementWiseOperation(A, B, (a, b) -> a - b);
    }

    public static double[][] multiplyElementWise(double[][] A, double[][] B) {
        return elementWiseOperation(A, B, (a, b) -> a * b);
    }
    public static double[][] elementWiseOperation(double[][] A, double[][] B, java.util.function.DoubleBinaryOperator func) {
        int m = A.length;
        int n = A[0].length;
        double[][] C = new double[m][n];
        IntStream.range(0, m).parallel().forEach(i -> {
            for (int j = 0; j < n; j++) {
                C[i][j] = func.applyAsDouble(A[i][j], B[i][j]);
            }
        });
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

    // Scalar addition
    public static double[][] addScalar(double[][] A, double scalar) {
        int m = A.length;
        int n = A[0].length;
        double[][] result = new double[m][n];

        IntStream.range(0, m).parallel().forEach(i -> {
            for (int j = 0; j < n; j++) {
                result[i][j] = A[i][j] + scalar;
            }
        });

        return result;
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
            int index = RANDOM.nextInt(i + 1);
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
}