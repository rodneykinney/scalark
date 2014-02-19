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
 *  An immutable neural network, represented only by its connections.  
 *  Node values within a layer are computable by the network but not part of its state
 */
class Network(val connections: IndexedSeq[Connection]) {
  /**
   *  Perform batch gradient descent on the set of inputs
   *  Each input is a tuple consisting of the inputs and expected outputs
   */ 
  def trainStep(inputLayersAndLabels: Iterable[(Layer, Layer)], learningRate: Double, loss: LossFunction) = {
    // Accumulate set of batch updates to network weights
    val connectionUpdates = inputLayersAndLabels.foldLeft(connections.map(c => new NullConnectionUpdate().asInstanceOf[ConnectionUpdate])) {
      case (cumulativeUpdates, (input, target)) => {

        // Compute activated nodes in each layer
        val layers = connections.foldLeft(List(input)) {
          case (layerList, c) => (layerList :+ c.forwardPropagate(layerList.last))
        }

        // Back-propagate errors to each layer
        val outputLayerLoss = loss.gradient(layers.last, target)
        val errors = (connections.zip(layers).drop(1).foldRight(List(outputLayerLoss)) {
          case ((c, l), errorList) => errorList :+ c.backwardPropagate(l, errorList.last)
        }).reverse

        // Compute weight updates for each connection
        val updates = for (((connection, inputLayer), outputErrors) <- connections.zip(layers).zip(errors)) yield {
          connection.generateUpdate(inputLayer, outputErrors)
        }
        // Sum all the weight updates into a single cumulative update
        cumulativeUpdates.zip(updates).map { case (c, u) => c.merge(u) }
      }
    }
    // Apply weight updates and return new network
    new Network(connections.zip(connectionUpdates.map(_.scale(learningRate))).map { case (c, u) => u(c) })
  }
  /**
   * Apply network to compute output from the given inputs
   */
  def eval(inputLayer: Layer) = {
    connections.foldLeft(inputLayer) {
      case (l, c) => c.forwardPropagate(l)
    }
  }
}