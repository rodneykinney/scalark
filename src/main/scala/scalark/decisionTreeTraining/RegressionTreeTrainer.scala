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

import scala.collection.mutable.PriorityQueue
import scala.collection.immutable._

/**
 * Trains a regression decision tree
 */
class RegressionTreeTrainer(
  val config: DecisionTreeTrainConfig,
  val columns: Seq[FeatureColumn[Double]],
  val rowCount: Int) {
  val partition = new TreePartition(rowCount)
  private val splitter = new RegressionSplitFinder(config)

  private var _model: DecisionTreeModel = {
    val countSum = ((0.0, 0.0) /: columns.head.all(partition.root)) { (t, fi) => (t._1 + fi.weight, t._2 + fi.weight * fi.label) }
    val mean = countSum._2 / countSum._1
    new DecisionTreeModel(Vector(new DecisionTreeLeaf(0, mean)))
  }

  private var candidates: PriorityQueue[SplitCandidate] = _

  def train = {
    while (_model.leafCount < config.leafCount && nextIteration() != null) {}
    model
  }

  def model = _model

  def nextIteration(): SplitCandidate = {
    if (candidates == null) {
      candidates = new PriorityQueue[SplitCandidate]()(Ordering[Double].on[SplitCandidate](_.gain)) ++ columns.map(splitter.findSplitCandidate(_, partition.root)).filter(_ != null)
    }
    if (candidates.size > 0) {
      val bestCandidate = candidates.dequeue

      val (keep, drop) = candidates.partition(_.regionId != bestCandidate.regionId)
      candidates = keep

      val splitNode = partition(bestCandidate.regionId)
      val column = columns(bestCandidate.columnId)
      val leftIds = column.range(splitNode, 0, splitNode.size).filter(_.featureValue <= bestCandidate.threshold).map(_.rowId).toSet
      val (left, right) = partition.split(splitNode, leftIds.size)
      columns.foreach(_.repartition(splitNode, left, right, leftIds))
      candidates = candidates ++ columns.map(splitter.findSplitCandidate(_, left)).filter(_ != null)
      candidates = candidates ++ columns.map(splitter.findSplitCandidate(_, right)).filter(_ != null)

      _model = _model.merge(new DecisionTreeModel(
        Vector(new DecisionTreeSplit(splitNode.regionId, left.regionId, right.regionId, bestCandidate),
          new DecisionTreeLeaf(left.regionId, bestCandidate.leftValue),
          new DecisionTreeLeaf(right.regionId, bestCandidate.rightValue))))

      bestCandidate
    } else
      null
  }
}