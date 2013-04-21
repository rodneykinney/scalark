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

class StochasticGradientBoostTrainer[L, T <: Observation](
  config: StochasticGradientBoostTrainConfig,
  cost: CostFunction[L, T with Label[L]],
  labelData: Seq[T with Label[L]],
  columns: immutable.Seq[FeatureColumn[L, T with Label[L] with Feature]])(implicit scoreDecorator: T with Label[L] => DecorateWithScoreAndRegion[T with Label[L]]) {

  require(labelData.validate)

  private var trees = Vector.empty[Model]
  private val rootRegion = new TreeRegion(0, 0, labelData.size)
  private val rand = new util.Random(config.randomSeed)
  private val data = for (row <- labelData) yield row.withScoreAndRegion(score = 0.0, regionId = -1)

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
      val columnSampler = sampler(columns.size, config.featureSampleRate, rand)
      val rowSampler = sampler(columns.head.size, config.rowSampleRate, rand)
      val sampledColumns = columns.filter(c => columnSampler(c.columnId))
      val gradients = new mutable.ArraySeq[Double](data.size)
      data.zip(cost.gradient(data)) foreach { t => gradients(t._1.rowId) = t._2 }
      val residualData = for (c <- sampledColumns) yield {
        val regressionInstances = (for (row <- c.all(rootRegion)) yield {
          ObservationLabelFeature(rowId = row.rowId, featureValue = row.featureValue, label = -gradients(row.rowId))
        })
        new FeatureColumn[Double, ObservationLabelFeature[Double]](regressionInstances, c.columnId)
      }

      // Regression fit to the gradient
      val regressionTrainer = new RegressionTreeTrainer(config.treeConfig, residualData, data.size, rowSampler)
      val tree = regressionTrainer.train

      // Optimize constant values at leaves of the tree
      val (leaves, splits) = tree.nodes.partition(_.isInstanceOf[DecisionTreeLeaf])
      for (
        leaf <- leaves;
        row <- residualData.head.all(regressionTrainer.partition(leaf.regionId))
      ) {
        data(row.rowId).regionId = leaf.regionId
      }
      val regionIdToDelta = cost.optimalDelta(data.filter(r => rowSampler(r.rowId)))
      for (row <- data) {
        row.score += regionIdToDelta(row.regionId) * config.learningRate
      }
      val replacedLeaves = for (l <- leaves) yield {
        val delta = regionIdToDelta(l.regionId)
        new DecisionTreeLeaf(regionId = l.regionId, value = delta * config.learningRate)
      }
      val deltaModel = new DecisionTreeModel(splits ++ replacedLeaves)

      // Add tree to ensemble
      trees = trees :+ deltaModel
    }
  }

  private def rowIdsInAscendingOrder(data: Seq[Observation]) = {
    data.head.rowId == 0 && data.take(data.size - 1).zip(data.drop(1)).forall(t => t._2.rowId == t._1.rowId + 1)
  }

}