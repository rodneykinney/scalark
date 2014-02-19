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
 * Compute loss on the output NN layer
 */
trait LossFunction {
  /**
   * Scalar value of the loss, for global error calculation
   */
  def apply(outputLayer: Layer, outputTarget: Layer): Double
  /**
   * Given predictions (outputLayer) and labels (outputTarget), computes the 
   * gradient of the loss function with respect to each of the nodes in the layer
   * This is the error that it back-propagated through the network
   */
  def gradient(outputLayer: Layer, outputTarget: Layer): LayerWithErrors
}

/**
 * Multi-class loss function using soft-max and cross entropy
 * Score for each label is exp(nodeValue), normalized so that sum is 1
 * No-label gets a default score of 1
 */
object SoftMaxCrossEntropy extends LossFunction {
  private def getTrueClass(l: Layer) = l.activeNodes.find(_._2 > 0).map(_._1).toSet
  private def normalize(l: Layer) = 1 + l.activeNodes.map(n => math.exp(n._2)).sum
  def apply(outputLayer: Layer, outputTarget: Layer) = {
    val trueClass = getTrueClass(outputTarget)
    val normalization = normalize(outputLayer)
    val trueClassScore = outputLayer.activeNodes.find(n => trueClass(n._1)).map(n => math.exp(n._2)) getOrElse (0.0)
    math.log(trueClassScore / normalization)
  }
  def gradient(outputLayer: Layer, outputTarget: Layer) = {
    val trueClass = getTrueClass(outputTarget)
    val normalization = normalize(outputLayer)
    val valuesWithErrors = for ((i, v) <- outputLayer.activeNodes) yield {
      val s = math.exp(v) / normalization
      val err = if (trueClass(i)) (1 - s) else -s
      (i, v, err)
    }
    new SparseLayerWithErrors(valuesWithErrors)
  }
}