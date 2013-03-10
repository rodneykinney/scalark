/*
Copyright 2013 Rodney Kinney

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
package scalark.decisionTreeTraining

/**
 * An Orthogonal cost function is one in which the derivative with respect to f_i depends only on f_i
 */
abstract class OrthogonalCostFunction[LabelType] extends CostFunction[LabelType] {

  /** Value of the cost function for a single point */
  def cost(x: LabelInstance[LabelType], f: Double): Double

  /** Single-variable derivative */
  def derivative(x: LabelInstance[LabelType], f: Double): Double

  /** Single-variable second derivative */
  def secondDerivative(x: LabelInstance[LabelType], f: Double): Double

  def gradient(labels: Seq[LabelInstance[LabelType]], modelEval: Function[Int, Double]) = {
    for (l <- labels) yield {
      val modelScore = modelEval(l.rowId)
      derivative(l, modelScore)
    }
  }

  /** Single Newton-Raphson step to find minimum */
  def optimalDelta(regions: Seq[Seq[LabelInstance[LabelType]]], modelEval: Function[Int, Double]) = {
    for (nodes <- regions) yield {
      val scores = nodes.map(n => modelEval(n.rowId))
      -nodes.zip(scores).map(t => derivative(t._1, t._2)).sum / nodes.zip(scores).map(t => secondDerivative(t._1, t._2)).sum
    }
  }

  def totalCost(labels: Seq[LabelInstance[LabelType]], modelEval: Function[Int, Double]) = labels.map(l => cost(l, modelEval(l.rowId))).sum
}