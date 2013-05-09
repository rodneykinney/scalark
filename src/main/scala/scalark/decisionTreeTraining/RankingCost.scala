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

import scala.collection._
import breeze.optimize._
import breeze.linalg._
/**
 * Cost function for ranking.  Cost is a sum over pairs of documents with unequal labels.  Each term is log-logistic function of the difference between the two scores
 * Important:  This implementation assumes that input data is in a canonical order, sorted first by query-id and then by label
 */
class RankingCost(maxIterations: Int = 2, memory: Int = 2) extends CostFunction[Int, Query with Label[Int] with Weight] {

  def optimalConstant[T1 <: Query with Label[Int]](labels: Seq[T1]) = {
    0.0
  }

  def gradient[T1 <: Query with Label[Int] with Weight with Score](docs: Seq[T1]) = {
    val gradients = new mutable.ArraySeq[Double](docs.length)
    val docsWithId = docs.zipWithIndex.map { case (row, rowId) => ObservationLabelQueryScore(rowId = rowId, label = row.label, queryId = row.queryId, weight = row.weight, score = row.score) }
    for ((better, worse) <- documentPairs(docsWithId)) {
      val delta = 1.0 / (1 + math.exp(worse.score - better.score))
      gradients(worse.rowId) -= delta
      gradients(better.rowId) += delta
    }
    gradients
  }

  def totalCost[T1 <: Query with Label[Int] with Weight with Score](queries: Seq[T1]) = {
    (for ((better, worse) <- documentPairs(queries)) yield {
      math.log(1 + math.exp(worse.score - better.score))
    }).sum
  }

  def optimalDelta[T1 <: Query with Label[Int] with Weight with Score with Region](data: Seq[T1]) = {
    val regionCount = data.map(_.regionId).max + 1
    val diff = new DiffFunction[DenseVector[Double]] {
      def calculate(regionValues: DenseVector[Double]) = {
        val dataAtDelta = for (row <- data) yield {
          new Query with Label[Int] with Weight with Score {
            def queryId = row.queryId
            def weight = row.weight
            def weight_=(value:Double) = ()
            def label = row.label
            def score = row.score + regionValues(row.regionId)
            def score_=(value:Double) = ()
          }
        }
        val cost = totalCost(dataAtDelta)
        val gradient = regionGradient(data, regionValues)
        (cost, gradient)
      }
    }
    val lbfgs = new LBFGS[DenseVector[Double]](maxIter = maxIterations, m = memory)
    val delta = lbfgs.minimize(diff, DenseVector.zeros[Double](regionCount))
    i: Int => delta(i)
  }

  private def regionGradient[T1 <: Query with Label[Int] with Score with Region](data: Seq[T1], regionValues: DenseVector[Double]) = {
    val regionGradients = DenseVector.zeros[Double](regionValues.size)
    for ((better, worse) <- documentPairs(data)) {
      val betterRegion = better.regionId
      val worseRegion = worse.regionId
      if (betterRegion != worseRegion) {
        val grad = 1.0 / (1 + math.exp(
          better.score + regionValues(betterRegion)
            - worse.score - regionValues(worseRegion)))
        regionGradients(betterRegion) -= grad
        regionGradients(worseRegion) += grad
      }
    }
    regionGradients
  }

  /** Iterate over all pairs of documents within each query.  Yield a tuple (better,worse) for each pair of documents with a different label */
  private def documentPairs[T <: Query with Label[Int]](docs: Seq[T]) = {
    for (
      query <- groupBySorted(docs, splitQueries);
      val labelGroups = groupBySorted(query, splitLabels).toIndexedSeq;
      label <- 0 until labelGroups.length;
      betterLabel <- label + 1 until labelGroups.length;
      worse <- labelGroups(label);
      better <- labelGroups(betterLabel)
    ) yield (better, worse)
  }

  private val splitQueries = (l: Seq[Query with Label[Int]]) => l.indexWhere(_.queryId != l.head.queryId)
  private val splitLabels = (l: Seq[Query with Label[Int]]) => l.indexWhere(_.label != l.head.label)

  private def groupBySorted[T](elems: Seq[T], splitFunction: Seq[T] => Int): List[Seq[T]] = {
    val splitIndex = splitFunction(elems)
    val (thisGroup, nextGroup) = elems.splitAt(splitIndex)
    thisGroup match {
      case Nil => List(nextGroup)
      case _ => thisGroup :: groupBySorted(nextGroup, splitFunction)
    }
  }
}