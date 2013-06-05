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
import spark.RDD
import spark.SparkContext

/**
 * Defines operations on data columns needed for training a decision tree
 */
trait ColumnOperations {
  /** For each column, return the best threshold for splitting training instances within the specified region */
  def getSplitCandidates(region: TreeRegion, splitFinder: RegressionSplitFinder): Seq[SplitCandidate]

  /** Return the row ids of all training instances within the specified region that satisfy the filter */
  def selectIdsByFeature(columnId: Int, featureFilter: Int => Boolean, region: TreeRegion): Set[Int]

  /** Repartition the data in all columns, splitting the parent region into two child regions */
  def repartitionAll(parentRegion: TreeRegion, leftChildRegion: TreeRegion, rightChildRegion: TreeRegion, leftIds: Int => Boolean): Unit

  /** Return the weighted average label for instances within the specified region */
  def weightedAverage(region: TreeRegion): Double
}

/** Factory interface for ColumnOperations */
trait ColumnOperationsFactory {
  def columnCount: Int
  def apply(columnFilter: Int => Boolean, weights: Int => Double, values: Int => Double): ColumnOperations
}

/** Single-threaded training on a single machine */
class SerialColumnOperations[T <: Observation with Weight with Feature with Label[Double]](val columns: IndexedSeq[FeatureColumn[Double, T]]) extends ColumnOperations {
  val size = columns.size
  def getSplitCandidates(region: TreeRegion, splitFinder: RegressionSplitFinder) = {
    columns.map(col => splitFinder.findSplitCandidate(col, region)).collect { case Some(c) => c }.seq
  }

  def selectIdsByFeature(columnId: Int, featureFilter: Int => Boolean, region: TreeRegion) = {
    columns.find(_.columnId == columnId).get.
      all(region).
      filter(row => featureFilter(row.featureValue)).
      map(_.rowId).toSet
  }

  def repartitionAll(parentRegion: TreeRegion, leftChildRegion: TreeRegion, rightChildRegion: TreeRegion, leftIds: Int => Boolean) = {
    for (col <- columns) {
      col.repartition(parentRegion, leftChildRegion, rightChildRegion, leftIds)
    }
  }

  def weightedAverage(region: TreeRegion) = {
    val (total, sum) = ((0.0, 0.0) /: columns.head.all(region)) { case ((w, s), fi) => (w + fi.weight, s + fi.label) }
    sum / total
  }

}

class SerialColumnOperationsFactory(columns: IndexedSeq[immutable.Seq[Observation with Feature with MutableLabel[Double] with MutableWeight]]) extends ColumnOperationsFactory {
  val columnCount = columns.size
  def apply(columnFilter: Int => Boolean, weights: Int => Double, gradients: Int => Double) = {
    val sampledColumns = columns.zipWithIndex.filter { case (c, columnId) => columnFilter(columnId) } map (_._1)
    val residualData = for ((c, columnId) <- sampledColumns.zipWithIndex) yield {
      for (row <- c) {
        row.weight = weights(row.rowId)
        row.label = -gradients(row.rowId)
      }
      val regressionInstances = mutable.ArraySeq.empty[Observation with Weight with Feature with Label[Double]] ++ c
      new FeatureColumn[Double, Observation with Weight with Feature with Label[Double]](regressionInstances, columnId)
    }
    new SerialColumnOperations(residualData)
  }
}

/** Multi-threaded training on a single machine */
class ParallelColumnOperations[T <: Observation with Weight with Feature with Label[Double]](val columns: parallel.ParSeq[FeatureColumn[Double, T]]) extends ColumnOperations {
  val size = columns.size
  def getSplitCandidates(region: TreeRegion, splitFinder: RegressionSplitFinder) = {
    columns.map(col => splitFinder.findSplitCandidate(col, region)).collect { case Some(c) => c }.seq
  }

  def selectIdsByFeature(columnId: Int, featureFilter: Int => Boolean, region: TreeRegion) = {
    columns.find(_.columnId == columnId).get.
      all(region).
      filter(row => featureFilter(row.featureValue)).
      map(_.rowId).toSet
  }

  def repartitionAll(parentRegion: TreeRegion, leftChildRegion: TreeRegion, rightChildRegion: TreeRegion, leftIds: Int => Boolean) = {
    for (col <- columns) {
      col.repartition(parentRegion, leftChildRegion, rightChildRegion, leftIds)
    }
  }

  def weightedAverage(region: TreeRegion) = {
    val (total, sum) = ((0.0, 0.0) /: columns.seq.head.all(region)) { case ((w, s), fi) => (w + fi.weight, s + fi.label) }
    sum / total
  }

}

class ParallelColumnOperationsFactory(columns: parallel.ParSeq[immutable.Seq[Observation with Feature with MutableLabel[Double] with MutableWeight]]) extends ColumnOperationsFactory {
  val columnCount = columns.size
  def apply(columnFilter: Int => Boolean, weights: Int => Double, gradients: Int => Double) = {
    val sampledColumns = columns.zipWithIndex.filter { case (c, columnId) => columnFilter(columnId) }
    val residualData = for ((c, columnId) <- sampledColumns) yield {
      for (row <- c) {
        row.weight = weights(row.rowId)
        row.label = -gradients(row.rowId)
      }
      val regressionInstances = mutable.ArraySeq.empty[Observation with Weight with Feature with Label[Double]] ++ c
      new FeatureColumn[Double, Observation with Weight with Feature with Label[Double]](regressionInstances, columnId)
    }
    new ParallelColumnOperations(residualData)
  }
}

/** Distributed training via Spark */
class DistributedColumnOperations[T <: Observation with Weight with Feature with Label[Double]](var columns: RDD[FeatureColumn[Double, T]]) extends ColumnOperations {
  def getSplitCandidates(region: TreeRegion, splitFinder: RegressionSplitFinder) = {
    val sf = splitFinder
    val counts = columns.map(col => col.size).collect
    columns.map(col => sf.findSplitCandidate(col, region)).collect { case Some(c) => c }.collect
  }
  def selectIdsByFeature(columnId: Int, featureFilter: Int => Boolean, region: TreeRegion) = {
    columns.filter(_.columnId == columnId).
      map(column => column.
        all(region).
        filter(row => featureFilter(row.featureValue)).
        map(_.rowId).toSet).collect.head
  }
  def repartitionAll(parentRegion: TreeRegion, leftChildRegion: TreeRegion, rightChildRegion: TreeRegion, leftIds: Int => Boolean) = {
    columns = (for (col <- columns) yield {
      col.repartition(parentRegion, leftChildRegion, rightChildRegion, leftIds)
    }).cache
  }
  def weightedAverage(region: TreeRegion) = {
    val (total, sum) = ((0.0, 0.0) /: columns.first.all(region)) { case ((w, s), fi) => (w + fi.weight, s + fi.label) }
    sum / total
  }
}

class DistributedColumnOperationsFactory(val columnCount: Int, val columns: RDD[Tuple2[Int,immutable.Seq[Observation with Feature with MutableLabel[Double] with MutableWeight]]]) extends ColumnOperationsFactory {
  def apply(columnFilter: Int => Boolean, weights: Int => Double, gradients: Int => Double) = {
    val residualData = columns filter(t => columnFilter(t._1)) map { t => 
      val (columnId, c) = t
      for (row <- c) {
        row.weight = weights(row.rowId)
        row.label = -gradients(row.rowId)
      }
      val regressionInstances = mutable.ArraySeq.empty[Observation with Weight with Feature with Label[Double]] ++ c
      new FeatureColumn[Double, Observation with Weight with Feature with Label[Double]](regressionInstances, columnId)
    }
    new DistributedColumnOperations(residualData)
  }

}