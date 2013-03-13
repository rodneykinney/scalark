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
/**
 * Cost function for ranking.  Cost is a sum over pairs of documents with unequal labels.  Each term is log-logistic function of the different of the two scores
 * Important:  This implementation assumes that input data is in a canonical order, sorted first by query-id and then by label
 */
class RankingCost extends CostFunction[Int, QueryDocPair[Int]] {

  def optimalConstant(labels: Seq[QueryDocPair[Int]]) = {
    0.0
  }

  def gradient(l: Seq[QueryDocPair[Int]], modelEval: Int => Double) = {
    val queries = groupBySorted(l, splitQueries)
    queryGradients(queries, modelEval)
  }

  def queryGradients(queries: List[Seq[QueryDocPair[Int]]], modelEval: Int => Double): Seq[Double] = {
    queries match {
      case head :: Nil => gradientByQuery(head, modelEval)
      case head :: tail => gradientByQuery(head, modelEval) ++ queryGradients(tail, modelEval)
      case _ => error("Error computing ranking cost gradient")
    }
  }

  def gradientByQuery(docs: Seq[QueryDocPair[Int]], modelEval: Int => Double) = {
    val gradients = new mutable.ArraySeq[Double](docs.length)
    val rowIdToIndex = docs.map(_.rowId).zipWithIndex.toMap
    for ((better, worse) <- documentPairs(docs)) {
      val delta = 1.0 / (1 + math.exp(modelEval(worse.rowId) - modelEval(better.rowId)))
      gradients(rowIdToIndex(worse.rowId)) -= delta
      gradients(rowIdToIndex(better.rowId)) += delta
    }
    gradients
  }

  def totalCost(queries: Seq[QueryDocPair[Int]], modelEval: Int => Double) = {
    documentPairs(queries).map(pair => math.log(1 + math.exp(modelEval(pair._2.rowId) - modelEval(pair._1.rowId)))).sum
  }

  def optimalDelta(regions: Seq[Seq[QueryDocPair[Int]]], modelEval: Function[Int, Double]) = {
    Seq(0.0)
  }
  def optimalDelta(data: Seq[QueryDocPair[Int]], rowIdToRegionId: Int => Int, rowIdToModelScore: Int => Double) = {
    x: Int => 0.0
  }
  /** Iterate over all pairs of documents with in a query.  Yield a tuple (better,worse) for each pair of documents with a different label */
  private def documentPairs(docs: Seq[QueryDocPair[Int]]) = {
    for (
      query <- groupBySorted(docs, splitQueries);
      val labelGroups = groupBySorted(query, splitLabels).toIndexedSeq;
      label <- 0 until labelGroups.length;
      betterLabel <- label + 1 until labelGroups.length;
      worse <- labelGroups(label);
      better <- labelGroups(betterLabel)
    ) yield (better, worse)
  }

  private val splitQueries = (l: Seq[QueryDocPair[Int]]) => l.indexWhere(_.queryId != l.head.queryId)
  private val splitLabels = (l: Seq[QueryDocPair[Int]]) => l.indexWhere(_.label != l.head.label)

  private def groupBySorted[T](elems: Seq[T], splitFunction: Seq[T] => Int): List[Seq[T]] = {
    val splitIndex = splitFunction(elems)
    val (thisGroup, nextGroup) = elems.splitAt(splitIndex)
    thisGroup match {
      case Nil => List(nextGroup)
      case _ => thisGroup :: groupBySorted(nextGroup, splitFunction)
    }
  }
}