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

trait CostFunction[LabelType] {
  /**
   * Constant value that minimizes the cost for this set of instances (e.g. mean value)
   */
  def optimalConstant(labels: Seq[LabelInstance[LabelType]]): Double
  /**
   * Gradient of the cost function
   */
  def gradient(labels: Seq[LabelInstance[LabelType]], modelEval: Function[Int, Double]): Seq[Double]
  /**
   * Total cost for this set of instances
   */
  def totalCost(ids: Seq[LabelInstance[LabelType]], modelEval: Function[Int, Double]): Double
  /**
   * Finds the constant value that should be added to the model score to minimize the cost for each set of instances
   */
  def optimalDelta(regions:Seq[Seq[LabelInstance[LabelType]]], modelEval:Function[Int,Double]):Seq[Double]
}