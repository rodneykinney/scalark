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
 * A set of nodes with values, with sparse representation
 */
/**
 */
trait Layer {
  def activeNodes: Iterable[(Int, Double)]
}

class SparseLayer(val activeNodes: Iterable[(Int, Double)]) extends Layer

class DenseLayer(values: IndexedSeq[Double]) extends Layer {
  def activeNodes = values.zipWithIndex.map { case (v, i) => (i, v) }
  def value(i: Int) = values(i)
}

/**
 * A set of nodes with values, plus values for the errors
 */
trait LayerWithErrors {
  def activeNodes: Iterable[(Int, Double, Double)]
}

class SparseLayerWithErrors(val activeNodes: Iterable[(Int, Double, Double)]) extends LayerWithErrors

class DenseLayerWithErrors(valuesAndErrors: IndexedSeq[(Double, Double)]) extends LayerWithErrors {
  def activeNodes = valuesAndErrors.zipWithIndex.map { case ((v1, v2), i) => (i, v1, v2) }
}
