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

class StochasticGradientBoostTrainer[L, T <: Label[L] with Weight](config: StochasticGradientBoostTrainConfig,
  cost: CostFunction[L, T],
  data: IndexedSeq[T with Score with Region],
  cols: immutable.Seq[immutable.Seq[Observation with Feature with Label[Double] with Weight]]) //(implicit scoreDecorator: T with Label[L] => DecorateWithScoreAndRegion[T with Label[L]]) 
  {

  //TODO:  Validation
  //require(labelData.validate)

  private var trees = Vector.empty[Model]
  private val rand = new util.Random(config.randomSeed)
  private val columns = cols.par

  def model = new AdditiveModel(trees)

  def train(iterationCallback: => Any) = {
    while (trees.size < config.iterationCount) {
      nextIteration()
      iterationCallback
    }
    model
  }

  def trainError = cost.totalCost(data)

  private def nextIteration() = {
    if (trees.size == 0) {
      // Initialize with constant value
      val mean = cost.optimalConstant(data)
      data foreach (_.score = mean)
      trees = trees :+ new DecisionTreeModel(Vector(new DecisionTreeLeaf(0, mean)))
    } else {
      val currentModel = model

      // Build training data to fit regression tree to gradient of the cost function
      val sampleSeed = rand.nextInt
      // Sample columns
      val columnSampler = sampler(columns.size, config.featureSampleRate, rand)
      val sampledColumns = columns.zipWithIndex.filter { case (c, columnId) => columnSampler(columnId) } map (_._1)
      // Sample rows
      val weights = 
      if (config.rowSampleRate == 1.0) {
        Array.fill(data.size)(1.0)
      }
      else {
        val w = new Array[Double](data.size)
        if (config.sampleRowsWithReplacement) {
          for (i <- 0 until (data.size * config.rowSampleRate + 0.5).toInt) w(rand.nextInt(data.size)) += 1
        } else {
          for (i <- 0 until data.size if rand.nextDouble < config.rowSampleRate) w(i) = 1
        }
        w
      }
      // Compute gradient
      val gradients = cost.gradient(data)
      // Compute residuals.  Regression tree will be fit to this data
      val residualData = for ((c, columnId) <- sampledColumns.zipWithIndex) yield {
        for (row <- c) {
          row.weight = weights(row.rowId)
          row.label = -gradients(row.rowId)
        }
        val regressionInstances = mutable.ArraySeq.empty[Observation with Weight with Feature with Label[Double]] ++ c
        new FeatureColumn[Double, Observation with Weight with Feature with Label[Double]](regressionInstances, columnId)
      }

      // Regression fit to the gradient
      val regressionTrainer = new RegressionTreeTrainer(config.treeConfig, residualData, data.size)
      val tree = regressionTrainer.train

      // Optimize constant values at leaves of the tree
      val (leaves, splits) = tree.nodes.partition(_.isLeaf)
      for (
        leaf <- leaves;
        row <- residualData.head.all(regressionTrainer.partition(leaf.regionId))
      ) {
        data(row.rowId).regionId = leaf.regionId
      }
      val regionIdToDelta = cost.optimalDelta(data)
      for (row <- data) {
        row.score += regionIdToDelta(row.regionId) * config.learningRate
      }
      val replacedLeaves = for (l <- leaves) yield {
        val delta = regionIdToDelta(l.regionId)
        require(!delta.isNaN, "Leaf with NaN value")
        new DecisionTreeLeaf(regionId = l.regionId, value = delta * config.learningRate)
      }
      val deltaModel = new DecisionTreeModel(splits ++ replacedLeaves)

      // Add tree to ensemble
      trees = trees :+ deltaModel
    }
  }
}
