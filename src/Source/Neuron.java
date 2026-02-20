package Source;

import java.util.Arrays;

class Neuron {

    private final Model model;
    private final int i, j;

    private double[] weights;
    private double bias;
    private final Activation activation;

    /**
     * @param model The parent model
     * @param i The layer number (layer -1 is the input layer)
     * @param j The ID within its layer (0-indexed)
     * @param activation The activation function
     */
    public Neuron(Model model, int i, int j, Activation activation) {
        this.i = i; this.j = j;
        this.model = model;
        weights = new double[model.getLayerSize(i-1)];
        for (int t=0;t<model.getLayerSize(i-1);t++) {
            weights[t] = Math.random();
        }
        bias = Math.random();
        this.activation = activation;
    }

    public double calculate(double[] prevLayer) {
        if (model.getLayerSize(i-1) != prevLayer.length) {
            throw new IllegalArgumentException("Mismatched Array Lengths!");
        }
        double result = 0;
        for (int k=0;k<prevLayer.length;k++) {
            result += weights[k] * prevLayer[k];
        }
        result += bias;
        return activation.apply(result);
    }

    public double differentiateWeight(double[] prevCalc, double[] prevDiff, int w_i, int w_j, int w_k) {
        if (model.getLayerSize(i-1) != prevCalc.length) {
            throw new IllegalArgumentException("Mismatched Array Lengths!");
        }
        if (model.getLayerSize(i-1) != prevDiff.length) {
            throw new IllegalArgumentException("Mismatched Array Lengths!");
        }
        double factor1 = 0;
        for (int n=0;n<prevCalc.length;n++) { // n parses k
            factor1 += weights[n] * prevCalc[n];
        }
        factor1 += bias;
        factor1 = activation.differentiate(factor1);
        double factor2 = 0;
        for (int n=0;n<prevDiff.length;n++) {
            if (i == w_i && j == w_j && n == w_k) {
                factor2 += prevCalc[n];
            } else {
                factor2 += weights[n] * prevDiff[n];
            }
        }
        return factor1 * factor2;
    }

    public double differentiateBias(double[] prevCalc, double[] prevDiff, int b_i, int b_j) {
        if (model.getLayerSize(i-1) != prevCalc.length) {
            throw new IllegalArgumentException("Mismatched Array Lengths!");
        }
        if (model.getLayerSize(i-1) != prevDiff.length) {
            throw new IllegalArgumentException("Mismatched Array Lengths!");
        }
        double factor1 = 0;
        for (int n=0;n<prevCalc.length;n++) { // n parses k
            factor1 += weights[n] * prevCalc[n];
        }
        factor1 += bias;
        factor1 = activation.differentiate(factor1);
        double factor2 = 0;
        for (int n=0;n<prevDiff.length;n++) {
            factor2 += weights[n] * prevDiff[n];
        }
        if (i == b_i && j == b_j) {
            factor2 += 1;
        }
        return factor1 * factor2;
    }

    public void updateWeight(int k, double change) {
        weights[k] += change;
    }

    public void updateBias(double change) {
        bias += change;
    }
}
