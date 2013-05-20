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
import org.junit.runner._
import org.scalatest.junit._
import scala.collection._
import org.scalatest.matchers.ShouldMatchers
import spark.SparkContext

@RunWith(classOf[JUnitRunner])
class TestDecisionTreeTrainNode extends FunSuite with BeforeAndAfter with ShouldMatchers with SparkTestUtils {
  val features = Array(3, 4, 2, 2, 2, 1, 6, 8, 20, 5)
  val partition: TreePartition = new TreePartition(features.size)
  var column: FeatureColumn[Double, Observation with Weight with Feature with Label[Double]] = _
  var rows: Seq[LabeledRow[Double]] = _

  before {
    init
  }

  def init = {
    rows = features.map(f => LabeledRow(features = Vector(f), label = f.toDouble))
    val col = rows.toSortedColumnData.head
    col.foreach(row => row.label = features(row.rowId))
    column = new FeatureColumn(col, 0)
  }

  test("Node Range") {
    val firstThree = column.range(partition.root, 0, 3)
    firstThree.size should be(3)
    firstThree.last.featureValue should be(2)
  }

  test("Node Batch 1") {
    val batch1 = column.batch(partition.root, 0, 1)
    batch1.size should be(1)
    batch1.last.featureValue should be(1)
  }

  test("Node Batch 2") {
    val batch2 = column.batch(partition.root, 0, 2)
    batch2.size should be(4)
    batch2.last.featureValue should be(2)
  }

  test("Node Batch 3") {
    val batch2 = column.batch(partition.root, 9, 2)
    batch2.size should be(1)
    batch2.last.featureValue should be(20)
  }

  test("Node Split") {
    val ids = Set(0, 3, 6, 7)
    val root = partition.root
    val (leftChild, rightChild) = partition.split(root, 4)
    column.repartition(root, leftChild, rightChild, ids)
    leftChild.size should be(4)
    rightChild.size should be(6)
    column.range(leftChild, leftChild.size - 1, leftChild.size).last.featureValue should be(8)
    column.range(rightChild, 0, 1).head.featureValue should be(1)
  }

  test("RegressionSplitFinder") {
    init
    val splitter = new RegressionSplitFinder(1)
    var parent = partition.root

    // Best split is [1,2,2,2,3,4,5,6,8] [20]
    var split = splitter.findSplitCandidate(column, parent).get
    split.threshold should be(8)
    var leftIds = column.range(parent, 0, parent.size).filter(_.featureValue <= split.threshold).map(_.rowId).toSet
    var (left, right) = partition.split(parent, leftIds.size)
    column.repartition(parent, left, right, leftIds)

    // Best split of left side is [1,2,2,2,3,4] [5,6,8]
    split = splitter.findSplitCandidate(column, left).get
    split.threshold should be(4)
    leftIds = column.range(left, 0, left.size).filter(_.featureValue <= split.threshold).map(_.rowId).toSet
    parent = left
    val t = partition.split(left, leftIds.size); left = t._1; right = t._2
    column.repartition(parent, left, right, leftIds)

    // Best split is [5,6] [8]
    split = splitter.findSplitCandidate(column, right).get
    split.threshold should be(6)
  }

  sparkTest("RegressionTreeTrainer") {
    init

    val columns = sc.parallelize(Vector(column))
    val columnOps = new DistributedColumnOperations(columns, sc)
    //TODO:  Remove
    println("Initialized columns")
    var trainer = new RegressionTreeTrainer(new DecisionTreeTrainConfig(minLeafSize = 1, leafCount = 2), columnOps, rows.size)

    val model1 = trainer.model
    val loss1 = rows.map(r => math.pow(r.label - model1.eval(r.features), 2)).sum

    val split = trainer.nextIteration()
    val model2 = trainer.model
    val loss2 = rows.map(r => math.pow(r.label - model2.eval(r.features), 2)).sum
    loss2 + split.gain should be(loss1 plusOrMinus 1.0e-5)
  }

}