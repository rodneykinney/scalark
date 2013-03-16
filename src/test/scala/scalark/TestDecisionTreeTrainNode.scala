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
package scalark

import scalark.decisionTreeTraining._
import org.scalatest._
import scala.collection.immutable._

class TestDecisionTreeTrainNode extends FunSuite with BeforeAndAfter {
  var partition: TreePartition = _
  var column: FeatureColumn[Double] = _
  var rows: Seq[Observation with RowOfFeatures with Label[Double]] = _

  before {
    init
  }

  def init = {
    val size = 10
    val features = Array(3, 4, 2, 2, 2, 1, 6, 8, 20, 5)
    rows = (0 until size).map(i => ObservationRowLabel(rowId = i, features = Vector(features(i)), label = features(i).toDouble))
    partition = new TreePartition(size)
    column = rows.toSortedColumns.head
  }

  test("Node Range") {
    val firstThree = column.range(partition.root, 0, 3)
    assert(firstThree.size === 3)
    assert(firstThree.last.featureValue === 2)
  }

  test("Node Batch 1") {
    val batch1 = column.batch(partition.root, 0, 1)
    assert(batch1.size === 1)
    assert(batch1.last.featureValue === 1)
  }

  test("Node Batch 2") {
    val batch2 = column.batch(partition.root, 0, 2)
    assert(batch2.size === 4)
    assert(batch2.last.featureValue === 2)
  }

  test("Node Batch 3") {
    val batch2 = column.batch(partition.root, 9, 2)
    assert(batch2.size === 1)
    assert(batch2.last.featureValue === 20)
  }

  test("Node Split") {
    val ids = new scala.collection.immutable.HashSet() ++ Array(0, 3, 6, 7)
    val root = partition.root
    val (leftChild, rightChild) = partition.split(root, 4)
    column.repartition(root, leftChild, rightChild, ids)
    assert(leftChild.size === 4)
    assert(rightChild.size === 6)
    assert(column.range(leftChild, leftChild.size - 1, leftChild.size).last.featureValue === 8)
    assert(column.range(rightChild, 0, 1).head.featureValue === 1)
  }

  test("RegressionSplitFinder") {
    init
    val splitter = new RegressionSplitFinder(new DecisionTreeTrainConfig(minLeafSize = 1, leafCount = 100))
    var parent = partition.root

    // Best split is [1,2,2,2,3,4,5,6,8] [20]
    var split = splitter.findSplitCandidate(column, parent)
    assert(split.threshold === 8)
    var leftIds = column.range(parent, 0, parent.size).filter(_.featureValue <= split.threshold).map(_.rowId).toSet
    var (left, right) = partition.split(parent, leftIds.size)
    column.repartition(parent, left, right, leftIds)

    // Best split of left side is [1,2,2,2,3,4] [5,6,8]
    split = splitter.findSplitCandidate(column, left)
    assert(split.threshold === 4)
    leftIds = column.range(left, 0, left.size).filter(_.featureValue <= split.threshold).map(_.rowId).toSet
    parent = left
    val t = partition.split(left, leftIds.size); left = t._1; right = t._2
    column.repartition(parent, left, right, leftIds)

    // Best split is [5,6] [8]
    split = splitter.findSplitCandidate(column, right)
    assert(split.threshold === 6)
  }

  test("RegressionTreeTrainer") {
    init

    var trainer = new RegressionTreeTrainer(new DecisionTreeTrainConfig(minLeafSize = 1, leafCount = 2), Vector(column), rows.size)

    val model1 = trainer.model
    val loss1 = rows.map(r => math.pow(r.label - model1.eval(r.features), 2)).sum

    val split = trainer.nextIteration()
    val model2 = trainer.model
    val loss2 = rows.map(r => math.pow(r.label - model2.eval(r.features), 2)).sum
    assert(math.abs(loss2 + split.gain - loss1) < 1.0e-5)

  }
}