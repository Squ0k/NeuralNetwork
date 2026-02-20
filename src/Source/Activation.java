package Source;

import java.util.function.DoubleUnaryOperator;

class Activation {

    private final DoubleUnaryOperator activationFunction;
    private final DoubleUnaryOperator activationDerivative;

    public Activation(DoubleUnaryOperator function, DoubleUnaryOperator derivative) {
        activationFunction = function;
        activationDerivative = derivative;
    }

    public static final Activation IDENTITY = new Activation(
            x -> x,
            x -> 1
    );

    public static final Activation SIGMOID = new Activation(
            x -> 1 / (1+Math.pow(Math.E, -x)),
            x -> Math.pow(Math.E, -x) / Math.pow(1 + Math.pow(Math.E, -x), 2)
    );

    public static final Activation RELU = new Activation(
            x -> Math.max(0, x),
            x -> (x <= 0) ? 0 : 1
    );

    public double apply(double input) {
        return activationFunction.applyAsDouble(input);
    }

    public double differentiate(double input) {
        return activationDerivative.applyAsDouble(input);
    }
}
