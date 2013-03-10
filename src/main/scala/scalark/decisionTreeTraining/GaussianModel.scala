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

/** Gaussian model that operates on a subset of features */
case class GaussianModel(val means: IndexedSeq[Double], val variance: IndexedSeq[IndexedSeq[Double]], val featureIndices: IndexedSeq[Int], val range: Int) extends Model {
  private val scale = 1.0 / math.pow(range, 2)
  def eval(features: Seq[Int]) = {
    var sum = 0.0
    for (
      i <- (0 until means.length);
      j <- (0 until means.length)
    ) sum += (features(featureIndices(i)) - means(i) * range) *
      variance(i)(j) * scale *
      (features(j) - means(j) * range)
    math.exp(-sum * sum)
  }
}

/** 
 * Bayes optimal classifier for boolean classification
 * Returned score is probability that label = true
 */
case class BayesOptimalBinaryModel(val positiveModel:Model, val negativeModel:Model) extends Model {
  def eval(features:Seq[Int]) = {
    val positiveScore = positiveModel.eval(features)
    val negativeScore = negativeModel.eval(features)
    positiveScore / (positiveScore + negativeScore)
  }
}