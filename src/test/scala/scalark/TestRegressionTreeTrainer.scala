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
import scala.util._
import org.junit.runner._
import org.scalatest.junit._

@RunWith(classOf[JUnitRunner])
class TestRegressionTreeTrainer extends FunSuite {

  test("train/test") {
    val db = new DataSynthesizer(2, 0, 1000)
    val allRows = db.regression(1100, 1, 0.1)
    val (train, test) = allRows.splitAt(1000)
    val columns = train.toSortedColumnData.toFeatureColumns((rowId:Int) => 1.0, (rowId:Int) => allRows(rowId).label)
    val config = new DecisionTreeTrainConfig(minLeafSize = 1, leafCount = 500)
    val trainer = new RegressionTreeTrainer(config, columns.par, train.size)
    val trees = Vector(new Tuple2(null, trainer.model)) ++ (for (i <- (1 until config.leafCount)) yield { val s = trainer.nextIteration(); (s, trainer.model) })
    val trainError = trees.map(t => train.map(r => math.pow(r.label - t._2.eval(r.features), 2)).sum)
    val testError = trees.map(t => test.map(r => math.pow(r.label - t._2.eval(r.features), 2)).sum)
    assert(trainError.min === trainError.last)
    assert(testError.min < testError.last)
  }

  test("2-d single gaussian") {
    var db = new DataSynthesizer(2, 0, 1000)
    val rows = db.regression(10000, 1, 0.1)
    val config = new DecisionTreeTrainConfig(minLeafSize = 1, leafCount = 10)

    testTrainer(config, rows)
  }

  test("2-d linear") {
    var rows = Vector(LabeledRow(features = Array(0, 0), label = 0.))
    rows = rows :+ LabeledRow(features = Array(0, 1), label = 1.)
    rows = rows :+ LabeledRow(features = Array(1, 0), label = 2.)
    rows = rows :+ LabeledRow(features = Array(1, 1), label = 3.)

    val config = new DecisionTreeTrainConfig(minLeafSize = 1, leafCount = 4)
    testTrainer(config, rows)
  }

  test("100-d Gaussian") {
    val db = new DataSynthesizer(100, 0, 1000)
    val rows = db.regression(1000, 2, 0.1)
    val config = new DecisionTreeTrainConfig(minLeafSize = 5, leafCount = 50)

    testTrainer(config, rows)
  }

  def testTrainer(config: DecisionTreeTrainConfig, rows: IndexedSeq[LabeledRow[Double]]) = {
    val columns = rows.toSortedColumnData.toFeatureColumns((rowId:Int) => 1.0, (rowId:Int) => rows(rowId).label )
    val trainer = new RegressionTreeTrainer(config, columns.par, rows.size)
    val trees = Vector(new Tuple2(null, trainer.model)) ++ (for (i <- (1 until config.leafCount)) yield { val s = trainer.nextIteration(); (s, trainer.model) })
    val losses = trees.map(t => rows.map(r => math.pow(r.label - t._2.eval(r.features), 2)).sum)
    for (iter <- (1 until config.leafCount)) {
      var delta = losses(iter - 1) - losses(iter)
      assert(delta >= 0)
      if (trees(iter)._1 != null)
        assert(math.abs(delta - trees(iter)._1.gain) < 1.0e-5)
    }
  }
}