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

class TestStochasticGradientBoostTrainer extends FunSuite {
  test("SGB - toy 1d") {
    val rows = Vector(
      ObservationRowLabel(0, Vector(0), true),
      ObservationRowLabel(1, Vector(1), true),
      ObservationRowLabel(2, Vector(1), false),
      ObservationRowLabel(3, Vector(1), false),
      ObservationRowLabel(4, Vector(2), true))
    val config = new StochasticGradientBoostTrainConfig(iterationCount = 10, leafCount = 3, learningRate = 1.0, minLeafSize = 1)
    val cost = new LogLogisticLoss()
    val trainer = new StochasticGradientBoostTrainer(config, cost, rows.map(r => ObservationLabel(r.rowId, r.label)), rows.toSortedColumns)
    val tol = 1.0e-8

    // Mean-value model is log(3/2)
    trainer.nextIteration()
    rows.foreach(r => assertWithin(trainer.model.eval(r.features), math.log(1.5), tol))

    // After one iteration, tree splits the range into three nodes
    trainer.nextIteration()
    assertWithin(trainer.model.eval(rows(0).features), math.log(1.5) + 5. / 3, tol)
    assertWithin(trainer.model.eval(rows(4).features), math.log(1.5) + 5. / 3, tol)
    assertWithin(trainer.model.eval(rows(1).features), math.log(1.5) - 10. / 9, tol)
    assertWithin(trainer.model.eval(rows(2).features), math.log(1.5) - 10. / 9, tol)
    assertWithin(trainer.model.eval(rows(3).features), math.log(1.5) - 10. / 9, tol)

    // Middle range should converge to log(0.5)
    var delta = math.abs(trainer.model.eval(rows(1).features) - math.log(0.5))
    for (i <- (1 to 3)) {
      trainer.nextIteration()
      val newDelta = math.abs(trainer.model.eval(rows(1).features) - math.log(0.5))
      assert(newDelta < delta)
      delta = newDelta
    }
  }

  test("SGB - toy 2d") {
    val rows = Vector(
      ObservationRowLabelScore(0, Vector(0, 0), true, 0.0),
      ObservationRowLabelScore(1, Vector(0, 1), false, 0.0),
      ObservationRowLabelScore(2, Vector(1, 0), true, 0.0),
      ObservationRowLabelScore(3, Vector(1, 1), false, 0.0))
    val config = new StochasticGradientBoostTrainConfig(iterationCount = 10, leafCount = 4, learningRate = 1.0, minLeafSize = 1)
    val cost = new LogLogisticLoss()
    val trainer = new StochasticGradientBoostTrainer(config, cost, rows.map(r => ObservationLabel(r.rowId, r.label)), rows.toSortedColumns)
    val models = (0 until config.iterationCount) map (i => { trainer.nextIteration(); trainer.model })
    val errorCount = models map (m => rows.count(r => m.eval(r.features) > 0 ^ r.label))
    val losses = for (m <- models) yield {
      for (row <- rows) { row.score = m.eval(row.features) }
      cost.totalCost(rows)
    }
    // Losses should decrease monotonically on the training data
    for (i <- (1 until losses.length)) {
      assert(losses(i) < losses(i - 1))
    }
  }

  test("SGB - 2d") {
    val rows = new DataSynthesizer(nDim = 2, minFeatureValue = 0, maxFeatureValue = 1000).binaryClassification(10000, 2) map (_.withScore(0.0))
    val config = new StochasticGradientBoostTrainConfig(iterationCount = 10, leafCount = 6, learningRate = 1.0, minLeafSize = 10)
    val cost = new LogLogisticLoss()
    val trainer = new StochasticGradientBoostTrainer(config, cost, rows.map(r => ObservationLabel(rowId = r.rowId, label = r.label)), rows.toSortedColumns)
    val models = (0 until config.iterationCount) map (i => { trainer.nextIteration(); trainer.model })
    val errorCount = models map (m => rows.count(r => m.eval(r.features) > 0 ^ r.label))
    val losses = for (m <- models) yield {
      for (row <- rows) { row.score = m.eval(row.features) }
      cost.totalCost(rows)
    }
    // Losses should decrease monotonically on the training data
    for (i <- (1 until losses.length)) {
      assert(losses(i) < losses(i - 1))
    }
  }

  private def assertWithin(value: Double, expected: Double, tolerance: Double) = assert(math.abs(value - expected) < tolerance)
}