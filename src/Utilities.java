package src;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

public class Utilities {
    public static int[][] computeConfusionMatrix(int[] predictions, int[] labels, int numClasses) {
        int[][] matrix = new int[numClasses][numClasses];
        for (int i = 0; i < predictions.length; i++) {
            matrix[labels[i]][predictions[i]]++;
        }
        return matrix;
    }
    
    public static double[][] multiplyMatrices(double[][] A, double[][] B) {
        int rows = A.length;
        int cols = B[0].length;
        int sharedDim = A[0].length;
        double[][] C = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double sum = 0.0;
                for (int k = 0; k < sharedDim; k++) {
                    sum += A[i][k] * B[k][j];
                }
                C[i][j] = sum;
            }
        }

        return C;
    }

    public static double[][] transpose(double[][] A) {
        int m = A.length;
        int n = A[0].length;
        double[][] T = new double[n][m];

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                T[j][i] = A[i][j];
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
            for (int j = 0; j < n; j++) {
                C[i][j] = A[i][j] + bias;
            }
        }

        return C;
    }

    public static double[][] subtractMatrices(double[][] A, double[][] B) {
        int m = A.length;
        int n = A[0].length;
        double[][] C = new double[m][n];

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                C[i][j] = A[i][j] - B[i][j];
            }
        }

        return C;
    }

    public static double[][] addMatrices(double[][] A, double[][] B) {
        int m = A.length;
        int n = A[0].length;
        double[][] C = new double[m][n];

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                C[i][j] = A[i][j] + B[i][j];
            }
        }

        return C;
    }

    public static double[][] multiplyElementWise(double[][] A, double[][] B) {
        int m = A.length;
        int n = A[0].length;
        double[][] C = new double[m][n];

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                C[i][j] = A[i][j] * B[i][j];
            }
        }

        return C;
    }

    public static double[][] elementWiseDivide(double[][] A, double[][] B) {
        int m = A.length;
        int n = A[0].length;
        double[][] C = new double[m][n];

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                C[i][j] = A[i][j] / B[i][j];
            }
        }

        return C;
    }

    public static double[][] scalarMultiply(double[][] A, double scalar) {
        int m = A.length;
        int n = A[0].length;
        double[][] C = new double[m][n];

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                C[i][j] = A[i][j] * scalar;
            }
        }

        return C;
    }

    public static double[][] scalarDivide(double[][] A, double scalar) {
        int m = A.length;
        int n = A[0].length;
        double[][] C = new double[m][n];

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                C[i][j] = A[i][j] / scalar;
            }
        }

        return C;
    }

    public static double[][] addScalar(double[][] A, double scalar) {
        int m = A.length;
        int n = A[0].length;
        double[][] C = new double[m][n];

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                C[i][j] = A[i][j] + scalar;
            }
        }

        return C;
    }

    public static double[][] subtractScalar(double[][] A, double scalar) {
        int m = A.length;
        int n = A[0].length;
        double[][] C = new double[m][n];

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                C[i][j] = A[i][j] - scalar;
            }
        }

        return C;
    }

    public static double[][] sqrtMatrix(double[][] A) {
        int m = A.length;
        int n = A[0].length;
        double[][] C = new double[m][n];

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                C[i][j] = Math.sqrt(A[i][j]);
            }
        }

        return C;
    }

    public static double[][] squareMatrix(double[][] A) {
        return multiplyElementWise(A, A);
    }

    // Sum over columns (for biases)
    public static double[][] sumColumns(double[][] A) {
        int m = A.length;
        int n = A[0].length;
        double[][] sum = new double[m][1];

        for (int i = 0; i < m; i++) {
            double s = 0.0;
            for (int j = 0; j < n; j++) {
                s += A[i][j];
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
        Random rand = new Random();
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
        int[] labels = new int[Y[0].length];
        for (int i = 0; i < Y[0].length; i++) {
            for (int j = 0; j < Y.length; j++) {
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
}
