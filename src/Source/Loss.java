package Source;

import java.util.function.DoubleBinaryOperator;

public class Loss {

    private final DoubleBinaryOperator lossFunction;
    private final DoubleBinaryOperator lossDerivative;

    /**
     * @param partialDerivative The derivative of the function, but Excluding the derivative of the model
     */
    public Loss(DoubleBinaryOperator function, DoubleBinaryOperator partialDerivative) {
        lossFunction = function;
        lossDerivative = partialDerivative;
    }

    public static final Loss SUM_SQUARED_ERROR = new Loss(
            (x, y) -> Math.pow(x - y, 2),
            (x, y) -> 2 * (x - y)
    );

    public static final Loss CROSS_ENTROPY_LOSS = new Loss(
            (x, y) -> - y * Math.log(x) - (1-y) * Math.log(1-x),
            (x, y) -> - y / x + (1-y) / (1-x)
    );

    public double calculate(double x, double y) {
        return lossFunction.applyAsDouble(x, y);
    }

    /**
     * @return The derivative of the function, but Excluding the derivative of the model
     */
    public double partialDifferentiate(double x, double y) {
        return lossDerivative.applyAsDouble(x, y);
    }
}
