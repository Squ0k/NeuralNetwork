import Source.*;

public class Main {
    public static void main(String[] args) {
        NNRegression model = new NNRegression(new int[]{1}, 2);
        for (int i=0;i<10000;i++) {
            double a = Math.random() * 10;
            double b = Math.random() * 10;

            model.train(new double[]{a, b}, a+b, 0.001);
        }
        for (int i=0;i<10;i++) {
            double a = (int) (Math.random() * 10);
            double b = (int) (Math.random() * 10);

            double c = model.predict(new double[]{a, b});
            System.out.println(a + " + " + b + " = " + c);
        }
    }
}
