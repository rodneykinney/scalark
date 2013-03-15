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

class TestRegressionTreeTrainer extends FunSuite {

  test("train/test") {
    val db = new DataSynthesizer(2, 0, 1000)
    val allRows = db.regression(1100, 1, 0.1)
    val (train, test) = allRows.partition(r => r.rowId < 1000)
    val columns = train.toSortedColumns
    val config = new DecisionTreeTrainConfig(minLeafSize = 1, leafCount = 500)
    val trainer = new RegressionTreeTrainer(config, columns, train.size)
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
    var rows = Vector(Row(0, Array(0, 0), 0.))
    rows = rows :+ Row(1, Array(0, 1), 1.)
    rows = rows :+ Row(2, Array(1, 0), 2.)
    rows = rows :+ Row(3, Array(1, 1), 3.)

    val config = new DecisionTreeTrainConfig(minLeafSize = 1, leafCount = 4)
    testTrainer(config, rows)
  }

  test("100-d Gaussian") {
    val db = new DataSynthesizer(100, 0, 1000)
    val rows = db.regression(1000, 2, 0.1)
    val config = new DecisionTreeTrainConfig(minLeafSize = 5, leafCount = 50)

    testTrainer(config, rows)
  }

  def testTrainer(config: DecisionTreeTrainConfig, rows: IndexedSeq[Observation with RowOfFeatures with Label[Double]]) = {
    val columns = rows.toSortedColumns
    val trainer = new RegressionTreeTrainer(config, columns, rows.size)
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