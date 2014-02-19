package scalark.perceptron

trait ActivationFunction {
  def apply(x: Double): Double
  def derivative(x: Double): Double
}

object Logistic extends ActivationFunction {
  def apply(x: Double) = 1.0 / (1.0 + math.exp(-x))
  def derivative(x: Double) = x*(1-x)
}