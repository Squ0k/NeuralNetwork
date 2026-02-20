package Source;

public class NNClassifier extends Model {
    private final int classes;

    /**
     * Creates a classification model
     *
     * @param modelShape The shape of the model, excludes the input and output layers
     * @param inputSize  The size of the input layer
     * @param classes The amount of classes
     */
    public NNClassifier(int[] modelShape, int inputSize, int classes) {
        super(modelShape, inputSize, Activation.SIGMOID);
        this.classes = classes;
    }

    public Loss getLoss() {
        return Loss.CROSS_ENTROPY_LOSS;
    }

    public int getOutputSize() {
        return classes;
    }

    public double interpret(double[] raw) {
        double highestValue = 0;
        int result = 0;
        for (int i=0;i<raw.length;i++) {
            if (highestValue < raw[i]) {
                highestValue = raw[i];
                result = i;
            }
        }
        return result;
    }

    public double[] interpret(double raw) {
        double[] result = new double[classes];
        for (int i=0;i<classes;i++) {
            if (i == raw) {
                result[i] = 1;
            } else {
                result[i] = 0;
            }
        }
        return result;
    }
}
