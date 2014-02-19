/*
Copyright 2014 Rodney Kinney

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
package scalark.perceptron

/**
 * Represents connection between two NN layers
 * This abstraction allows for different network topologies via different implementing classes
 */
trait Connection {
  /**
   * Compute downstream layer based on upstream inputs
   */
  def forwardPropagate(inputLayer: Layer): Layer
  /**
   * Compute errors on upstream layer based on errors in the downstream layer
   */
  def backwardPropagate(inputLayer: Layer, outputLayer: LayerWithErrors): LayerWithErrors
  /**
   * Compute an incremental delta to this connection based on the values and errors in the upstream/downstream layers
   */
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
      outputValues(j) += value * weights(j * inputLayerSize + i)
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
      inputErrors(i) += err * activationFunction.derivative(value) * weights(j * inputLayerSize + i)
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
      deltas(j * inputLayerSize + i) = value * outErr
    }
    new DenseConnectionUpdate(deltas)
  }
}

