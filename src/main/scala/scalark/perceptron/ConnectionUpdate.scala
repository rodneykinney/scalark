package scalark.perceptron

/**
 * Encapsulates a set of connection-weight updates
 */
trait ConnectionUpdate {
  def apply(input: Connection): Connection
  def scale(learningRate: Double): ConnectionUpdate
  def merge(other: ConnectionUpdate): ConnectionUpdate
}

class NullConnectionUpdate extends ConnectionUpdate {
  def apply(input:Connection) = input
  def scale(learningRate: Double) = this
  def merge(other: ConnectionUpdate) = other
}

class DenseConnectionUpdate(val deltas: IndexedSeq[Double]) extends ConnectionUpdate {
  def apply(input: Connection) = input match {
    case c: DenseConnection =>
      new DenseConnection(c.inputLayerSize,
        c.outputLayerSize,
        c.weights.zip(deltas).map { case (w, d) => w + d },
        c.activationFunction)
    case _ => sys.error("Cannot apply " + this + " to " + input)
  }
  def merge(other: ConnectionUpdate) = other match {
    case d: DenseConnectionUpdate => new DenseConnectionUpdate(deltas.zip(d.deltas).map { case (a, b) => a + b })
    case _ => sys.error("Cannot merge " + this + " with " + other)
  }
  def scale(learningRate: Double) = new DenseConnectionUpdate(deltas.map(_ * learningRate))
}