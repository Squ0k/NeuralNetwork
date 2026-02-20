package Source;

import java.util.Arrays;

abstract class Model {

    private final int inputSize;
    private final int[] modelShape;

    private final Neuron[][] model;

    private double[][] dpCalc;
    private double[][] dpDiff;

    /**
     * Creates a new layered model with the desired type
     *
     * @param modelShape The shape of the model, excludes the input and output layers
     * @param inputSize The size of the input layer
     * @param outputActivation Activation function of the output neuron(s)
     */
    public Model(int[] modelShape, int inputSize, Activation outputActivation) {
        this.inputSize = inputSize;
        this.modelShape = modelShape;
        model = new Neuron[modelShape.length+2][getMaxSize()];
        for (int i=1;i<modelShape.length+1;i++) {
            for (int j=0;j<getLayerSize(i);j++) {
                model[i][j] = new Neuron(this, i, j, Activation.RELU);
            }
        }
        for (int j=0;j<getOutputSize();j++) {
            model[modelShape.length+1][j] = new Neuron(this, modelShape.length+1, j, outputActivation);
        }
    }

    public double[] predictRaw(double[] input) {
        if (input.length != getInputSize()) {
            throw new IllegalArgumentException("Mismatched Array Lengths!");
        }
        dpCalc = new double[modelShape.length+2][getMaxSize()];
        System.arraycopy(input, 0, dpCalc[0], 0, input.length);
        for (int i=1;i<modelShape.length+2;i++) {
            for (int j=0;j<getLayerSize(i);j++) {
                dpCalc[i][j] = model[i][j].calculate(
                        Arrays.copyOf(dpCalc[i-1], getLayerSize(i-1))
                );
            }
        }
        return Arrays.copyOf(dpCalc[modelShape.length+1], getOutputSize());
    }

    public double predict(double[] input) {
        if (input.length != getInputSize()) {
            throw new IllegalArgumentException("Mismatched Array Lengths!");
        }
        return interpret(predictRaw(input));
    }

    public void train(double[] input, double target, double learningRate) {
        if (input.length != getInputSize()) {
            throw new IllegalArgumentException("Mismatched Array Lengths!");
        }
        predictRaw(input); // fill dpCalc
        dpDiff = new double[modelShape.length+2][getMaxSize()];
        for (int w_i=1;w_i<modelShape.length+2;w_i++) {
            for (int w_j=0;w_j<getLayerSize(w_i);w_j++) {
                // Calculate differentials for weights
                for (int w_k=0;w_k<getLayerSize(w_i-1);w_k++) {
                    for (int j=0;j<getInputSize();j++) {
                        dpDiff[0][j] = 0;
                    }
                    for (int i=1;i<modelShape.length+2;i++) {
                        for (int j=0;j<getLayerSize(i);j++) {
                            dpDiff[i][j] = model[i][j].differentiateWeight(
                                    Arrays.copyOf(dpCalc[i-1], getLayerSize(i-1)),
                                    Arrays.copyOf(dpDiff[i-1], getLayerSize(i-1)),
                                    w_i, w_j, w_k
                            );
                        }
                    }
                    double[] y = interpret(target);
                    double gradient = 0;
                    for (int m=0;m<getOutputSize();m++) { // parse output neurons
                        gradient += getLoss().partialDifferentiate(
                                dpCalc[modelShape.length + 1][m], y[m]
                        ) * dpDiff[modelShape.length + 1][m];
                    }
                    model[w_i][w_j].updateWeight(w_k, -gradient * learningRate);
                }
                // Calculate differentials for bias
                for (int j=0;j<getInputSize();j++) {
                    dpDiff[0][j] = 0;
                }
                for (int i=1;i<modelShape.length+2;i++) {
                    for (int j=0;j<getLayerSize(i);j++) {
                        dpDiff[i][j] = model[i][j].differentiateBias(
                                Arrays.copyOf(dpCalc[i-1], getLayerSize(i-1)),
                                Arrays.copyOf(dpDiff[i-1], getLayerSize(i-1)),
                                w_i, w_j
                        );
                    }
                }
                double[] y = interpret(target);
                double gradient = 0;
                for (int m=0;m<getOutputSize();m++) { // parse output neurons
                    gradient += getLoss().partialDifferentiate(
                            dpCalc[modelShape.length + 1][m], y[m]
                    ) * dpDiff[modelShape.length + 1][m];
                }
                model[w_i][w_j].updateBias(-gradient * learningRate);
            }
        }
    }

    public int getInputSize() {
        return inputSize;
    }

    public int[] getModelShape() {
        return modelShape;
    }

    public int getMaxSize() {
        int maxSize = getInputSize();
        for (int size:modelShape) {
            maxSize = Math.max(size, maxSize);
        }
        return Math.max(maxSize, getOutputSize());
    }

    public int getLayerSize(int layer) {
        if (layer == 0) {
            return getInputSize();
        } else if (layer < modelShape.length + 1) {
            return modelShape[layer-1];
        } else if (layer == modelShape.length + 1) {
            return getOutputSize();
        } else {
            return 0;
        }
    }

    /** Loss function for this model. */
    public abstract Loss getLoss();

    /** Output dimension for this model. */
    public abstract int getOutputSize();

    /** Convert raw network output into a scalar. */
    public abstract double interpret(double[] raw);

    /** Convert target scalar into training format. */
    public abstract double[] interpret(double raw);
}
