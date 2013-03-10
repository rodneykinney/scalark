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

import scala.util.Random
import scala.math._
import scala.collection._

class DataSynthesizer(nDim: Int, minFeatureValue: Int, maxFeatureValue: Int, seed: Int = 117) {
  private val range = maxFeatureValue - minFeatureValue
  private val scale = 1.0 / pow(range, 2)
  private val rand = new Random(seed)

  def regression(nRows: Int, nModes: Int, noise: Double) = {
    val model = gaussianMixtureModel(nModes)
    val rows = for (id <- (0 until nRows)) yield {
      val features = for (d <- (0 until nDim)) yield minFeatureValue + rand.nextInt(range)
      val value = model(features) * (1.0 - .5 * noise + noise * rand.nextDouble)
      new LabeledFeatureRow[Double](id, features, 1.0, value)
    }
    rows
  }

  def binaryClassification(nRows: Int, nModesPerClass: Int) = {
    val positiveModel = gaussianMixtureModel(nModesPerClass)
    val negativeModel = gaussianMixtureModel(nModesPerClass)
    val rows = for (id <- (0 until nRows)) yield {
      val features = for (d <- (0 until nDim)) yield minFeatureValue + rand.nextInt(range)
      val p1 = positiveModel(features)
      val p2 = negativeModel(features)
      val value = (p1 + p2) * rand.nextDouble()
      new LabeledFeatureRow[Boolean](id, features, 1.0, value < p1)
    }
    rows
  }

  def gaussianMixtureModel(nModes: Int) = {
    val modes = for (i <- (0 until nModes)) yield {
      val featureSubset = randomFeatureIndices
      val mean = (0 until featureSubset.size) map (i => rand.nextDouble)
      val variance = randomVariance(featureSubset.size)
      gaussianModel(mean, variance, featureSubset)
    }
    features:Seq[Int] => modes.map(_(features)).sum
  }

  def randomFeatureIndices = {
    val nFeatures = math.min(nDim, (1.5 - 2 * math.log(rand.nextDouble)).toInt)
    val indices = (0 until nDim).toArray
    for (i <- (0 until nFeatures)) {
      val n = rand.nextInt(nDim - i) + i
      val swap = indices(n)
      indices(n) = indices(i)
      indices(i) = swap
    }
    Vector.empty[Int] ++ indices.take(nFeatures)
  }

  def randomVariance(dims: Int) = {
    var N = Vector.empty[IndexedSeq[Double]]
    for (i <- (0 until dims)) {
      var col:IndexedSeq[Double] = (0 until dims) map (i => rand.nextDouble)
      for (j <- (0 until i)) {
        val projection = (N(j).zip(col)).map(t => t._1 * t._2).sum
        col = N(j) zip col map (t => t._2 - t._1 * projection)
      }
      val norm = math.sqrt(col map (x => x * x) sum)
      col = col map (_ / norm)
      N = N :+ col
    }
    val diags = (0 until dims) map (i => rand.nextDouble) map (x => x * x)
    N = N zip diags map (t => t._1.map(x => x * t._2))
    N
  }

  def gaussianModel(mean: IndexedSeq[Double], variance: IndexedSeq[IndexedSeq[Double]], featureIndices: IndexedSeq[Int]) = {
    var f = (features: Seq[Int]) => {
      var sum = 0.0
      for (
        i <- (0 until mean.length);
        j <- (0 until mean.length)
      ) sum += (features(featureIndices(i)) - mean(i) * range) *
        variance(i)(j) * scale *
        (features(j) - mean(j) * range)
      math.exp(-sum * sum)
    }
    f
  }

}