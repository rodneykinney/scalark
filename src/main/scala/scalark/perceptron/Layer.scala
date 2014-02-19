package scalark.perceptron

/**
 * A NN layer
 */
trait Layer {
  def activeNodes: Iterable[(Int, Double)]
}

class SparseLayer(val activeNodes: Iterable[(Int, Double)]) extends Layer

class DenseLayer(values: IndexedSeq[Double]) extends Layer {
  def activeNodes = values.zipWithIndex.map { case (v, i) => (i, v) }
  def value(i: Int) = values(i)
}

trait LayerWithErrors {
  def activeNodes: Iterable[(Int, Double, Double)]
}

class SparseLayerWithErrors(val activeNodes: Iterable[(Int, Double, Double)])

class DenseLayerWithErrors(valuesAndErrors: IndexedSeq[(Double, Double)]) extends LayerWithErrors {
  def activeNodes = valuesAndErrors.zipWithIndex.map { case ((v1, v2), i) => (i, v1, v2) }
}
