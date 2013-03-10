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
import Extensions._
import org.scalatest._

class TestStochasticGradientBoostTrainer extends FunSuite {
  test("SGB - toy 1d") {
    val rows = Vector(
      new LabeledFeatureRow[Boolean](0, Vector(0), 1.0, true),
      new LabeledFeatureRow[Boolean](1, Vector(1), 1.0, true),
      new LabeledFeatureRow[Boolean](2, Vector(1), 1.0, false),
      new LabeledFeatureRow[Boolean](3, Vector(1), 1.0, false),
      new LabeledFeatureRow[Boolean](4, Vector(2), 1.0, true))
    val config = new StochasticGradientBoostTrainConfig(iterationCount = 10, leafCount = 3, learningRate = 1.0, minLeafSize = 1)
    val cost = new LogLogisticLoss()
    val trainer = new StochasticGradientBoostTrainer(config, cost, rows.toSortedColumns)
    val tol = 1.0e-8

    // Mean-value model is log(3/2)
    trainer.nextIteration()
    rows.foreach(r => assertWithin(trainer.model.eval(r), math.log(1.5), tol))

    // After one iteration, tree splits the range into three nodes
    trainer.nextIteration()
    assertWithin(trainer.model.eval(rows(0)), math.log(1.5) + 5. / 3, tol)
    assertWithin(trainer.model.eval(rows(4)), math.log(1.5) + 5. / 3, tol)
    assertWithin(trainer.model.eval(rows(1)), math.log(1.5) - 10. / 9, tol)
    assertWithin(trainer.model.eval(rows(2)), math.log(1.5) - 10. / 9, tol)
    assertWithin(trainer.model.eval(rows(3)), math.log(1.5) - 10. / 9, tol)

    // Middle range should converge to log(0.5)
    var delta = math.abs(trainer.model.eval(rows(1)) - math.log(0.5))
    for (i <- (1 to 3)) {
      trainer.nextIteration()
      val newDelta = math.abs(trainer.model.eval(rows(1)) - math.log(0.5))
      assert(newDelta < delta)
      delta = newDelta
    }
  }

  test("SGB - toy 2d") {
    val rows = Vector(
      new LabeledFeatureRow[Boolean](0, Vector(0, 0), 1.0, true),
      new LabeledFeatureRow[Boolean](1, Vector(0, 1), 1.0, false),
      new LabeledFeatureRow[Boolean](2, Vector(1, 0), 1.0, true),
      new LabeledFeatureRow[Boolean](3, Vector(1, 1), 1.0, false))
    val config = new StochasticGradientBoostTrainConfig(iterationCount = 10, leafCount = 4, learningRate = 1.0, minLeafSize = 1)
    val cost = new LogLogisticLoss()
    val trainer = new StochasticGradientBoostTrainer(config, cost, rows.toSortedColumns)
    val models = (0 until config.iterationCount) map (i => { trainer.nextIteration(); trainer.model })
    val errorCount = models map (m => rows.count(r => m.eval(r) > 0 ^ r.label))
    val losses = models map (m => cost.totalCost(rows, id => m.eval(rows(id))))
    // Losses should decrease monotonically on the training data
    for (i <- (1 until losses.length)) {
      assert(losses(i) < losses(i - 1))
    }
  }

  test("SGB - 2d") {
    val rows = new DataSynthesizer(nDim = 2, minFeatureValue = 0, maxFeatureValue = 1000).binaryClassification(10000, 2)
    val config = new StochasticGradientBoostTrainConfig(iterationCount = 10, leafCount = 6, learningRate = 1.0, minLeafSize = 10)
    val cost = new LogLogisticLoss()
    val trainer = new StochasticGradientBoostTrainer(config, cost, rows.toSortedColumns)
    val models = (0 until config.iterationCount) map (i => { trainer.nextIteration(); trainer.model })
    val errorCount = models map (m => rows.count(r => m.eval(r) > 0 ^ r.label))
    val losses = models map (m => cost.totalCost(rows, id => m.eval(rows(id))))
    // Losses should decrease monotonically on the training data
    for (i <- (1 until losses.length)) {
      assert(losses(i) < losses(i - 1))
    }
  }

  private def assertWithin(value: Double, expected: Double, tolerance: Double) = assert(math.abs(value - expected) < tolerance)
}