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

/**
 * Generates Gaussian-mixture distributions, and labeled data drawn from those distributions
 */
class DataSynthesizer(nDim: Int, minFeatureValue: Int, maxFeatureValue: Int, seed: Int = 117) {
  private val range = maxFeatureValue - minFeatureValue
  private val scale = 1.0 / pow(range, 2)
  private val rand = new Random(seed)

  def regression(nRows: Int, nModes: Int, noise: Double) = {
    val model = gaussianMixtureModel(nModes)
    val rows = for (id <- (0 until nRows)) yield {
      val features = for (d <- (0 until nDim)) yield minFeatureValue + rand.nextInt(range)
      val value = model.eval(features) * (1.0 - .5 * noise + noise * rand.nextDouble)
      LabeledRow(features = features, label = value)
    }
    rows
  }

  def generateData[L](nRows: Int, labelGenerator: (IndexedSeq[Int], Random) => L) = {
    for (id <- (0 until nRows)) yield {
      val features = for (d <- (0 until nDim)) yield minFeatureValue + rand.nextInt(range)
      LabeledRow(features = features, label = labelGenerator(features, rand))
    }
  }

  def binaryClassification(nRows: Int, nModesPerClass: Int, spikiness: Double = 2.5) = {
    generateData(nRows, new GenerativeModel(List(gaussianMixtureModel(nModesPerClass, spikiness), gaussianMixtureModel(nModesPerClass, spikiness)), _ > 0).assignLabel _)
  }

  def ranking(nQueries: Int, minResultsPerQuery: Int, maxResultsPerQuery: Int, nClasses: Int, nModesPerClass: Int, spikiness: Double = 2.5) = {
    val generator = new GenerativeModel((0 until nClasses).map(n => gaussianMixtureModel(nModesPerClass, spikiness)).toSeq, i => i)
    var rowId = 0
    (0 until nQueries) flatMap (queryId => {
      val rows = (for (i <- (0 until (minResultsPerQuery + rand.nextInt(maxResultsPerQuery - minResultsPerQuery)))) yield {
        val features = for (d <- (0 until nDim)) yield minFeatureValue + rand.nextInt(range)
        (features, generator.assignLabel(features, rand))
      })
      val queryRows = rows.sortBy(_._2).zipWithIndex.map(t => t match {
        case ((f, l), i) => LabeledQueryRow(queryId = queryId, label = l, features = f)
      })
      rowId += queryRows.size
      queryRows
    })
  }

  def gaussianMixtureModel(nModes: Int, spikiness: Double = 1.0) = {
    val modes = for (i <- (0 until nModes)) yield {
      val featureSubset = randomFeatureIndices
      val mean = (0 until featureSubset.size) map (i => rand.nextDouble)
      val variance = randomVariance(featureSubset.size, spikiness)
      new GaussianModel(mean, variance, featureSubset, range)
    }
    new AdditiveModel(modes)
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

  def randomVariance(dims: Int, spikiness: Double = 1.0) = {
    var N = Vector.empty[IndexedSeq[Double]]
    for (i <- (0 until dims)) {
      var col: IndexedSeq[Double] = (0 until dims) map (i => rand.nextDouble)
      for (j <- (0 until i)) {
        val projection = (N(j).zip(col)).map { case (m, v) => m * v }.sum
        col = N(j) zip col map { case (m, v) => v - m * projection }
      }
      val norm = math.sqrt(col map (x => x * x) sum)
      col = col map (_ / norm)
      N = N :+ col
    }
    val weight = (0 until dims) map (i => rand.nextDouble * spikiness)
    N = N zip weight map { case (v, c) => v.map(x => x * c) }
    N
  }
}