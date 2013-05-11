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
import scala.collection._
import org.junit.runner._
import org.scalatest.junit._

@RunWith(classOf[JUnitRunner])
class TestRanking extends FunSuite with BeforeAndAfter {
  before {
    breeze.util.logging.ConfiguredLogging.configuration = breeze.config.Configuration.fromMap(immutable.Map(
      "log.level" -> "warn"))
  }

  test("TotalCost") {
    val c = new RankingCost()

    val rows = List(LabeledQueryRow(label = 0, queryId = 0, features=Vector.empty[Int]).forTraining,
      LabeledQueryRow(label = 0, queryId = 0, features=Vector.empty[Int]).forTraining,
      LabeledQueryRow(label = 1, queryId = 0, features=Vector.empty[Int]).forTraining,
      LabeledQueryRow(label = 0, queryId = 1, features=Vector.empty[Int]).forTraining,
      LabeledQueryRow(label = 1, queryId = 1, features=Vector.empty[Int]).forTraining)

    for (row <- rows) row.score = row.label
    assert(math.abs(c.totalCost(rows) - 3 * math.log(1 + math.exp(-1))) < 1.0e-9)
    for (row <- rows) row.score = 0
    assert(math.abs(c.totalCost(rows) - 3 * math.log(2)) < 1.0e-9)
    assert(c.gradient(rows) === Seq(-.5, -.5, 1, -.5, .5))
  }

  test("Optimal Delta") {
    val rows = List(new { val rowId: Int = 0; var weight: Double = 1.0; var label: Int = 0; val queryId: Int = 0; var score = 0.; var regionId = 0 } with Observation with Label[Int] with Query with Score with Region with Weight,
      new { val rowId: Int = 1; var weight: Double = 1.0; var label: Int = 0; val queryId: Int = 0; var score = 0.; var regionId = 1 } with Observation with Label[Int] with Query with Score with Region with Weight,
      new { val rowId: Int = 2; var weight: Double = 1.0; var label: Int = 1; val queryId: Int = 0; var score = 0.; var regionId = 0 } with Observation with Label[Int] with Query with Score with Region with Weight,
      new { val rowId: Int = 3; var weight: Double = 1.0; var label: Int = 1; val queryId: Int = 0; var score = 0.; var regionId = 1 } with Observation with Label[Int] with Query with Score with Region with Weight,
      new { val rowId: Int = 4; var weight: Double = 1.0; var label: Int = 1; val queryId: Int = 0; var score = 0.; var regionId = 1 } with Observation with Label[Int] with Query with Score with Region with Weight)

    val costs = Range(1, 6).map(new RankingCost(_))
    val x = costs.head.totalCost(rows)
    assert(math.abs(costs.head.totalCost(rows) - 6 * math.log(2)) < 1.0e-9)
    val scores = for (cost <- costs) yield { val d = cost.optimalDelta(rows); d(1) - d(0) }
    // With more accurate LBFGS, difference in scores should converge to log(2)
    val target = math.log(2)
    for ((score, previousScore) <- scores.drop(1).zip(scores)) {
      assert(math.abs(target - score) < math.abs(target - previousScore))
    }
  }
  test("Toy 1d") {
    val rows = IndexedSeq(LabeledQueryRow(queryId = 0, label = 0, features = Vector(0)),
      LabeledQueryRow(queryId = 0, label = 0, features = Vector(1)),
      LabeledQueryRow(queryId = 0, label = 1, features = Vector(0)),
      LabeledQueryRow(queryId = 0, label = 1, features = Vector(1)),
      LabeledQueryRow(queryId = 0, label = 1, features = Vector(1)))
    val columns = rows.toSortedColumnData
    val config = new StochasticGradientBoostTrainConfig(iterationCount = 5, leafCount = 2, learningRate = 1.0, minLeafSize = 1)
    val cost = new RankingCost()
    val trainingRows = rows.map(_.forTraining)
    val trainer = new StochasticGradientBoostTrainer(config, cost, trainingRows, columns)
    var models = Vector.empty[Model]
    trainer.train(models = models :+ trainer.model)
    // Cost should decrease monotonically
    val costs = for (m <- models) yield {
      for ((row, tRow) <- rows.zip(trainingRows)) tRow.score = m.eval(row.features)
      cost.totalCost(trainingRows)
    }
    for ((nextCost, cost) <- costs.drop(1).zip(costs)) {
      assert(nextCost <= cost)
    }
    // Score difference should converge to log(2)
    val target = math.log(2)
    val convergence = for (m <- models) yield {
      m.eval(rows(1).features) - m.eval(rows(0).features)
    }
    for ((nextDiff, diff) <- convergence.drop(1).zip(convergence)) {
      assert(math.abs(nextDiff - target) <= math.abs(diff - target))
    }
  }

  test("3d") {
    val synthesizer = new DataSynthesizer(3, 0, 1000)
    val rows = synthesizer.ranking(nQueries = 100, minResultsPerQuery = 2, maxResultsPerQuery = 20, nClasses = 4, nModesPerClass = 2)
    val columns = rows.toSortedColumnData
    val config = new StochasticGradientBoostTrainConfig(iterationCount = 20, leafCount = 4, learningRate = 1.0, minLeafSize = 1)
    val cost = new RankingCost()
    val trainingRows = rows.map(_.forTraining).toIndexedSeq
    val trainer = new StochasticGradientBoostTrainer(config, cost, trainingRows, columns)
    var models = Vector.empty[Model]
    trainer.train(models = models :+ trainer.model)
    // Cost should decrease monotonically
    val costs = for (m <- models) yield {
      for ((row,tRow) <- rows.zip(trainingRows)) tRow.score = m.eval(row.features)
      cost.totalCost(trainingRows)
    }
    for ((nextCost, cost) <- costs.drop(1).zip(costs)) {
      assert(nextCost <= cost)
    }
  }
}
