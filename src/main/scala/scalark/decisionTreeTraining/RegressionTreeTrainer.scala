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
import scala.collection.parallel.immutable.ParSeq

/**
 * Trains a regression decision tree
 */
class RegressionTreeTrainer[T <: Observation with Weight with Feature with Label[Double]](
  val config: DecisionTreeTrainConfig,
  val columnOps: ColumnOperations,
  val rowCount: Int) {
  val partition = new TreePartition(rowCount)
  val splitFinder = new RegressionSplitFinder(config.minLeafSize)
  
  private var _model: DecisionTreeModel = {
    //TODO:  Cleanup
//    val (total, sum) = ((0.0, 0.0) /: columns.first.all(partition.root)) { (t, fi) => (t._1 + 1, t._2 + fi.label) }
//    val mean = sum / total
//    new DecisionTreeModel(Vector(new DecisionTreeLeaf(0, mean)))
    new DecisionTreeModel(Vector(new DecisionTreeLeaf(0, columnOps.weightedAverage(partition.root))))
  }

  private var candidates: PriorityQueue[SplitCandidate] = _

  def train = {
    while (_model.leafCount < config.leafCount && nextIteration() != null) {}
    model
  }

  def model = _model

  def nextIteration(): SplitCandidate = {
    if (candidates == null) {
      val initialCandidates = columnOps.getSplitCandidates(partition.root, splitFinder)
      //TODO: Cleanup
//      val split = new Splitter[T](splitter, partition.root)
//      val initialCandidates = columns.map(split(_)).collect { case Some(c) => c }.collect
      candidates = new PriorityQueue[SplitCandidate]()(Ordering[Double].on[SplitCandidate](_.gain)) ++ initialCandidates
    }
    if (candidates.size > 0) {
      val bestCandidate = candidates.dequeue

      val (keep, drop) = candidates.partition(_.regionId != bestCandidate.regionId)
      candidates = keep

      val regionToSplit = partition(bestCandidate.regionId)
      //TODO: cleanup
//      val leftIds = columns.filter(_.columnId == bestCandidate.columnId).map(column =>
//        column.range(regionToSplit, 0, regionToSplit.size).filter(_.featureValue <= bestCandidate.threshold).map(_.rowId).toSet).collect.first
      val leftIds = columnOps.selectIdsByFeature(bestCandidate.columnId, (feature:Int) => feature <= bestCandidate.threshold, regionToSplit)
      val (leftChildRegion, rightChildRegion) = partition.split(regionToSplit, leftIds.size)
//      columns.foreach(_.repartition(regionToSplit, leftChildRegion, rightChildRegion, leftIds))
      columnOps.repartitionAll(regionToSplit, leftChildRegion, rightChildRegion, leftIds)
      //TODO: Remove
//      println("In between")
//      val splitLeft = new Splitter[T](splitter, leftChildRegion)
//      val splitRight = new Splitter[T](splitter, rightChildRegion)
//
//      candidates = candidates ++ columns.map(splitLeft(_)).collect { case Some(c) => c }.collect
//      candidates = candidates ++ columns.map(splitRight(_)).collect { case Some(c) => c }.collect
      candidates = candidates ++ columnOps.getSplitCandidates(leftChildRegion, splitFinder)
      candidates = candidates ++ columnOps.getSplitCandidates(rightChildRegion, splitFinder)

      _model = _model.merge(new DecisionTreeModel(
        Vector(new DecisionTreeSplit(regionToSplit.regionId, leftChildRegion.regionId, rightChildRegion.regionId, bestCandidate),
          new DecisionTreeLeaf(leftChildRegion.regionId, bestCandidate.leftValue),
          new DecisionTreeLeaf(rightChildRegion.regionId, bestCandidate.rightValue))))

      bestCandidate
    } else
      null
  }
}

@serializable
class Splitter[T <: Observation with Weight with Feature with Label[Double]](val splitter: RegressionSplitFinder, val region: TreeRegion) {
  def apply(col: FeatureColumn[Double, T]) = splitter.findSplitCandidate(col, region)
}
