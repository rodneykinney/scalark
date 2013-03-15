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
class LogLogisticLoss extends OrthogonalCostFunction[Boolean, Observation with Label[Boolean]] {

  def cost(x: Observation with Label[Boolean], f: Double) = math.log(1 + math.exp(if (x.label) -f else f))
  
  def optimalConstant(labels: Seq[Observation with Label[Boolean]]) = {
    val (nTotal,nPositive) = ((0.0,0.0) /: labels) {(t,l) => if (l.label) (t._1+1, t._2+1) else (t._1+1,t._2)}
    if (nPositive == 0 || nPositive == nTotal) 
      math.log((nPositive+1) / (nTotal+1))
    else 
      math.log(nPositive / (nTotal - nPositive))
  }

  def derivative(x: Observation with Label[Boolean], f: Double) = if (x.label) -1.0 / (1.0 + math.exp(f)) else 1.0 / (1.0 + math.exp(-f))

  def secondDerivative(x: Observation with Label[Boolean], f: Double) = {
    val sigma = 1.0 / (1.0 + math.exp(f))
    sigma * (1 - sigma)
  }

}
