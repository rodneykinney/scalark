package scalark.perceptron

class Network(connections: IndexedSeq[Connection]) {
  def train(inputLayersAndLabels: Iterable[(Layer, Layer)], loss: LossFunction) = {
    inputLayersAndLabels.foldLeft(connections.map(c => new NullConnectionUpdate).asInstanceOf[IndexedSeq[ConnectionUpdate]]) {
      case (cumulativeUpdates, (input, target)) => {
        val layers = (connections.foldLeft(List(input)) {
          case (layerList, c) => (layerList :+ c.forwardPropagate(layerList.last))
        }).toIndexedSeq
        val errors = (connections.zipWithIndex.drop(1).foldRight(List(loss.gradient(layers.last, target))) {
          case ((c, i), errorList) => errorList :+ c.backwardPropagate(layers(i - 1), errorList.last)
        }).reverse
        val updates = for (((connection, inputLayer), outputErrors) <- connections.zip(layers).zip(errors)) yield {
          connection.generateUpdate(inputLayer, outputErrors)
        }
        cumulativeUpdates.zip(updates).map { case (c, u) => c.merge(u) }
      }
    }
    //  }(cumulativeUpdate: IndexedSeq[ConnectionUpdate], incrementalUpdate: IndexedSeq[ConnectionUpdate]) => cumulativeUpdates.zip(incrementalUpdates).map { case (c, i) => c.merge(i) }
  }
  def eval(inputLayer: Layer) = {
    var l = inputLayer
    for (c <- connections) {
      l = c.forwardPropagate(l)
    }
    l
  }
}