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
abstract class OrthogonalCostFunction[L, T <: Label[L] with Weight] extends CostFunction[L, T] {

  /** Value of the cost function for a single point */
  def cost[T1 <: T with Score](x: T1): Double

  /** Single-variable derivative */
  def derivative[T1 <: T with Score](x: T1): Double

  /** Single-variable second derivative */
  def secondDerivative[T1 <: T with Score](x: T1): Double

  def gradient[T1 <: T with Score](data: Seq[T1]) = {
    for (l <- data) yield {
      l.weight * derivative(l)
    }
  }

  /** Single Newton-Raphson step to find minimum */
  def optimalDelta[T1 <: T with Score with Region](data: Seq[T1]) = {
    val regions = data.groupBy(row => row.regionId)
    (for ((regionId, regionData) <- regions) yield {
      val num = -regionData.map(l => l.weight * derivative(l)).sum
      val denom = math.max(regionData.map(l => l.weight * secondDerivative(l)).sum, 1.0e-6)
      val delta = num / denom
      (regionId, delta)
    }).toMap
  }

  def totalCost[T1 <: T with Score](labels: Seq[T1]) = labels.map(l => l.weight * cost(l)).sum
}