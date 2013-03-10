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
class LogLogisticLoss extends OrthogonalCostFunction[Boolean] {

  def cost(x: LabelInstance[Boolean], f: Double) = math.log(1 + math.exp(if (x.label) -f else f))
  
  def optimalConstant(labels: Seq[LabelInstance[Boolean]]) = {
    var nTotal = labels.map(_.weight).sum
    var nPositive = labels.filter(_.label).map(_.weight).sum
    if (nPositive == 0 || nPositive == nTotal) { nPositive += 1; nTotal += 2 }
    math.log(nPositive / (nTotal - nPositive))
  }

  def derivative(x: LabelInstance[Boolean], f: Double) = if (x.label) -1.0 / (1.0 + math.exp(f)) else 1.0 / (1.0 + math.exp(-f))

  def secondDerivative(x: LabelInstance[Boolean], f: Double) = {
    val sigma = 1.0 / (1.0 + math.exp(f))
    sigma * (1 - sigma)
  }

}