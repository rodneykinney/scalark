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
import scala.collection._
/**
 * An Orthogonal cost function is one in which the derivative with respect to f_i depends only on f_i
 */
abstract class OrthogonalCostFunction[L, T <: Observation with Label[L]] extends CostFunction[L, T] {

  /** Value of the cost function for a single point */
  def cost[T1 <: T with Score](x: T1): Double

  /** Single-variable derivative */
  def derivative[T1 <: T with Score](x: T1): Double

  /** Single-variable second derivative */
  def secondDerivative[T1 <: T with Score](x: T1): Double

  def gradient[T1 <: T with Score](data: Seq[T1]) = {
    for (l <- data) yield {
      derivative(l)
    }
  }
  /*
  def gradient(data: Seq[T], rowIdToModelScore: Int => Double) = {
    for (l <- data) yield {
      val modelScore = rowIdToModelScore(l.rowId)
      derivative(l, modelScore)
    }
  }*/

  /** Single Newton-Raphson step to find minimum */
  /*def optimalDelta(regions: Seq[Seq[T]], modelEval: Function[Int, Double]) = {
    for (nodes <- regions) yield {
      val scores = nodes.map(n => modelEval(n.rowId))
      -nodes.zip(scores).map(t => derivative(t._1, t._2)).sum / nodes.zip(scores).map(t => secondDerivative(t._1, t._2)).sum
    }
  }*/
  def optimalDelta[T1 <: T with Score with Region](data: Seq[T1]) = {
    val regions = data.groupBy(row => row.regionId)
    (for ((regionId, regionData) <- regions) yield {
      val delta = -regionData.map(derivative(_)).sum / regionData.map(secondDerivative(_)).sum
      (regionId, delta)
    }).toMap
  }
  /*  def optimalDelta(data: Seq[T], rowIdToRegionId: Int => Int, rowIdToModelScore: Int => Double) = {
    val regions = data.groupBy(row => rowIdToRegionId(row.rowId))
    (for ((regionId, regionData) <- regions) yield {
      val scores = regionData.map(n => rowIdToModelScore(n.rowId))
      val delta = -regionData.zip(scores).map(t => derivative(t._1, t._2)).sum / regionData.zip(scores).map(t => secondDerivative(t._1, t._2)).sum
      (regionId, delta)
    }).toMap
  }*/

  def totalCost[T1 <: T with Score](labels: Seq[T1]) = labels.map(cost(_)).sum
}