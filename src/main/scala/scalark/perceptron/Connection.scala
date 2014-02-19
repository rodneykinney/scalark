package scalark.perceptron

/**
 * Represents connection between two NN layers
 */
trait Connection {
  def forwardPropagate(inputLayer: Layer): Layer
  def backwardPropagate(inputLayer: Layer, outputLayer: LayerWithErrors): LayerWithErrors
  def generateUpdate(inputLayer: Layer, outputLayer: LayerWithErrors): ConnectionUpdate
}

class DenseConnection(val inputLayerSize: Int, val outputLayerSize: Int,
  val weights: IndexedSeq[Double],
  val activationFunction: ActivationFunction) extends Connection {
  def forwardPropagate(inputLayer: Layer) = {
    val outputValues = new Array[Double](outputLayerSize)
    for (
      (i, value) <- inputLayer.activeNodes;
      j <- (0 until outputLayerSize)
    ) {
      outputValues(j) += value * weights(i * inputLayerSize + j)
    }
    for (j <- (0 until outputLayerSize)) {
      outputValues(j) = activationFunction(outputValues(j))
    }
    new DenseLayer(outputValues)
  }
  def backwardPropagate(inputLayer: Layer, outputLayer: LayerWithErrors) = {
    val inputErrors = new Array[Double](inputLayerSize)
    for (
      (j, value, err) <- outputLayer.activeNodes;
      i <- (0 until inputLayerSize)
    ) {
      inputErrors(i) += err * activationFunction.derivative(value) * weights(i * inputLayerSize + j)
    }
    val inputValues = new Array[Double](inputLayerSize)
    for ((i, v) <- inputLayer.activeNodes) {
      inputValues(i) = v
    }
    new DenseLayerWithErrors(inputValues.zip(inputErrors))
  }
  def generateUpdate(inputLayer: Layer, outputLayer: LayerWithErrors) = {
    val deltas = new Array[Double](inputLayerSize * outputLayerSize)
    for (
      (i, value) <- inputLayer.activeNodes;
      (j, outValue, outErr) <- outputLayer.activeNodes
    ) {
      deltas(i * inputLayerSize + j) = value * outErr
    }
    new DenseConnectionUpdate(deltas)
  }
}

