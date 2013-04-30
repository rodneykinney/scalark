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
    for (i <- (0 until variance.size)) {
      val projection = variance(i) zip (0 until means.length) map { case (v, j) => v * (features(featureIndices(j)) - means(j) * range) }
      sum += projection.map(x => x * x * scale).sum
    }
    math.exp(-sum * sum)
  }
}

