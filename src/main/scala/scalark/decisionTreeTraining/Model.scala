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

import scala.util._

trait Model {
  def eval(features: Seq[Int]): Double
}

/** A model that is the sum of other models */
case class AdditiveModel(val models: Seq[Model]) extends Model {
  def eval(features: Seq[Int]) = models.map(_.eval(features)).sum
}

/**
 * Assigns labels randomly, given a set of models
 * Each model produces the relative probability of that class
 */
case class GenerativeModel[L](val models: Seq[Model], labelConvert: Int => L) {
  def assignLabel(features: Seq[Int], rand: Random) = {
    val rawWeights = models map (_.eval(features))
    assert(rawWeights.forall(_ >= 0))
    val totalWeight = rawWeights.sum
    val weights = rawWeights map (_ / totalWeight)
    var score = rand.nextDouble()
    val index = (0 /: weights) { (i, w) => { score -= w; if (score < 0) i else i + 1 } }
    labelConvert(index)
  }
}

/**
 * Assigns labels by maximum likelihood,
 * given a set of models producing the relative probability of that class
 */
case class MaximumLikelihoodModel[L](val models: Seq[Model], labelConvert: Int => L) {
  def assignLabel(features: Seq[Int]) = {
    val weights = models map (_.eval(features))
    labelConvert(weights.zipWithIndex.maxBy(_._1)._2)
  }
}
