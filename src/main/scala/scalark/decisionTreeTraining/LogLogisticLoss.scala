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
 * Log-Logistic loss function: log(1 + exp(-yf))
 */
class LogLogisticLoss extends OrthogonalCostFunction[Boolean, Label[Boolean] with Weight] {

  def cost[T <: Label[Boolean] with Score](x: T) = math.log(1 + math.exp(if (x.label) -x.score else x.score))

  def derivative[T <: Label[Boolean] with Weight with Score](x: T) = {
    if (x.label) -1.0 / (1.0 + math.exp(x.score)) else 1.0 / (1.0 + math.exp(-x.score))
  }

  def secondDerivative[T <: Label[Boolean] with Weight with Score](x: T) = {
    val sigma = 1.0 / (1.0 + math.exp(x.score))
    sigma * (1 - sigma)
  }

  def optimalConstant[T <: Label[Boolean] with Weight](labels: Seq[T]) = {
    val (nTotal, nPositive) = ((0.0, 0.0) /: labels) { (t, l) => if (l.label) (t._1 + l.weight, t._2 + 1) else (t._1 + l.weight, t._2) }
    if (nPositive == 0 || nPositive == nTotal)
      math.log((nPositive + 1) / (nTotal + 1))
    else
      math.log(nPositive / (nTotal - nPositive))
  }
}
