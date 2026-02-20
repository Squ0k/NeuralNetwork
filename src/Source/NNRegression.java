package Source;

public class NNRegression extends Model {
    /**
     * Creates a regression model
     *
     * @param modelShape The shape of the model, excludes the input and output layers
     * @param inputSize  The size of the input layer
     */
    public NNRegression(int[] modelShape, int inputSize) {
        super(modelShape, inputSize, Activation.RELU);
    }

    public Loss getLoss() {
        return Loss.SUM_SQUARED_ERROR;
    }

    public int getOutputSize() {
        return 1;
    }

    public double interpret(double[] raw) {
        return raw[0];
    }

    public double[] interpret(double raw) {
        return new double[] {raw};
    }
}