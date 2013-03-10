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
      val value = model.eval(features) * (1.0 - .5 * noise + noise * rand.nextDouble)
      new LabeledFeatureRow[Double](id, features, 1.0, value)
    }
    rows
  }

  def binaryClassificationDataAndOptimalModel(nRows: Int, nModesPerClass: Int) = {
    val model = new BayesOptimalBinaryModel(gaussianMixtureModel(nModesPerClass), gaussianMixtureModel(nModesPerClass))
    val rows = for (id <- (0 until nRows)) yield {
      val features = for (d <- (0 until nDim)) yield minFeatureValue + rand.nextInt(range)
      val trueProbability = model.eval(features)
      new LabeledFeatureRow[Boolean](id, features, 1.0, rand.nextDouble < trueProbability)
    }
    (rows,model)
  }

  def binaryClassification(nRows: Int, nModesPerClass: Int) = binaryClassificationDataAndOptimalModel(nRows, nModesPerClass)._1
  
  def gaussianMixtureModel(nModes: Int) = {
    val modes = for (i <- (0 until nModes)) yield {
      val featureSubset = randomFeatureIndices
      val mean = (0 until featureSubset.size) map (i => rand.nextDouble)
      val variance = randomVariance(featureSubset.size)
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


}