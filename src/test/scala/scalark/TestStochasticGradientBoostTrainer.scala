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

@RunWith(classOf[JUnitRunner])
class TestStochasticGradientBoostTrainer extends FunSuite with Matchers {
  test("SGB - toy 1d") {
    val rows = Vector(
      LabeledRow(true, Vector(0)),
      LabeledRow(true, Vector(1)),
      LabeledRow(false, Vector(1)),
      LabeledRow(false, Vector(1)),
      LabeledRow(true, Vector(2)))
    val config = new StochasticGradientBoostTrainConfig(iterationCount = 5, leafCount = 3, learningRate = 1.0, minLeafSize = 1)
    val cost = new LogLogisticLoss()
    val trainer = new StochasticGradientBoostTrainer(config, cost, rows.map(_.asTrainable), rows.toSortedColumnData)
    val tol = 1.0e-8

    var models = Vector.empty[Model]
    trainer.train(models = models :+ trainer.model)
    // Mean-value model is log(3/2)
    rows.foreach(r => models(0).eval(r.features) should be (math.log(1.5) +- tol))

    // After one iteration, tree splits the range into three nodes
    models(1).eval(rows(0).features) should be (math.log(1.5) + 5.0 / 3 +- tol)
    models(1).eval(rows(4).features) should be (math.log(1.5) + 5.0 / 3 +- tol)
    models(1).eval(rows(1).features) should be (math.log(1.5) - 10.0 / 9 +- tol)
    models(1).eval(rows(2).features) should be (math.log(1.5) - 10.0 / 9 +- tol)
    models(1).eval(rows(3).features) should be (math.log(1.5) - 10.0 / 9 +- tol)

    // Middle range should converge to log(0.5)
    var delta = math.abs(models(1).eval(rows(1).features) - math.log(0.5))
    for (i <- (2 to 4)) {
      val newDelta = math.abs(models(i).eval(rows(1).features) - math.log(0.5))
      newDelta should be < (delta)
      delta = newDelta
    }
  }

  test("SGB - toy 2d") {
    val rows = Vector(
      LabeledRow(true, Vector(0, 0)),
      LabeledRow(false, Vector(0, 1)),
      LabeledRow(true, Vector(1, 0)),
      LabeledRow(false, Vector(1, 1)))
    val config = new StochasticGradientBoostTrainConfig(iterationCount = 10, leafCount = 4, learningRate = 1.0, minLeafSize = 1)
    val cost = new LogLogisticLoss()
    val trainingRows = rows.map(_.asTrainable)
    val trainer = new StochasticGradientBoostTrainer(config, cost, trainingRows, rows.toSortedColumnData)
    var models = Vector.empty[Model]
    trainer.train(models = models :+ trainer.model)
    val errorCount = models map (m => rows.count(r => m.eval(r.features) > 0 ^ r.label))
    val costs = for (m <- models) yield {
      for ((row, tRow) <- rows.zip(trainingRows)) { tRow.score = m.eval(row.features) }
      cost.totalCost(trainingRows)
    }
    // Losses should decrease monotonically on the training data
    for (i <- (1 until costs.length)) {
      costs(i) should be < (costs(i - 1))
    }
  }

  test("SGB - 2d") {
    val rows = new DataSynthesizer(nDim = 2, minFeatureValue = 0, maxFeatureValue = 1000).binaryClassification(10000, 2)
    val config = new StochasticGradientBoostTrainConfig(iterationCount = 10, leafCount = 6, learningRate = 1.0, minLeafSize = 10)
    val cost = new LogLogisticLoss()
    val trainingRows = rows.map(_.asTrainable)
    val trainer = new StochasticGradientBoostTrainer(config, cost, trainingRows, rows.toSortedColumnData)
    var models = Vector.empty[Model]
    trainer.train(models = models :+ trainer.model)
    val errorCount = models map (m => rows.count(r => m.eval(r.features) > 0 ^ r.label))
    val costs = for (m <- models) yield {
      for ((row, tRow) <- rows.zip(trainingRows)) { tRow.score = m.eval(row.features) }
      cost.totalCost(trainingRows)
    }
    // Losses should decrease monotonically on the training data
    for (i <- (1 until costs.length)) {
      costs(i) should be < (costs(i - 1))
    }
  }
}