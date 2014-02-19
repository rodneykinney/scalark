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
 * Encapsulates a set of connection-weight updates
 * Each concrete Connection class should have a corresponding ConnectionUpdate implementation, 
 * which will share its internal representation of the connection weights
 */
trait ConnectionUpdate {
  /**
   * Apply delta to the connection to obtain a new connection with updated weights
   */
  def apply(input: Connection): Connection
  /**
   * Apply an overall scale factor to this update
   */
  def scale(learningRate: Double): ConnectionUpdate
  /**
   * Combine this update with another (i.e. add the deltas from each)
   */
  def merge(other: ConnectionUpdate): ConnectionUpdate
}

/**
 * Dummy update that does nothing
 */
class NullConnectionUpdate extends ConnectionUpdate {
  def apply(input:Connection) = input
  def scale(learningRate: Double) = this
  def merge(other: ConnectionUpdate) = other
}

/**
 * Updates weights in a DenseConnection
 */
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