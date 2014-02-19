package scalark.perceptron

/**
 * Compute loss on the output NN layer
 */
trait LossFunction {
  def loss(outputLayer: Layer): Double
  def gradient(outputLayer: Layer, outputTarget: Layer): LayerWithErrors
}