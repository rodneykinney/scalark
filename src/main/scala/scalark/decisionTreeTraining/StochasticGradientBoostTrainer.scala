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

class StochasticGradientBoostTrainer(
  config: StochasticGradientBoostTrainConfig,
  cost: CostFunction[Boolean, Observation with Label[Boolean]],
  labelData: Seq[Observation with Label[Boolean]],
  columns: immutable.Seq[FeatureColumn[Boolean]]) {

  require(rowIdsInAscendingOrder(labelData))

  private var trees = Vector.empty[Model]
  private val rootRegion = new TreeRegion(0, 0, labelData.size)
  private var _model: Model = _
  private val modelScores = new mutable.ArraySeq[Double](labelData.size)
  private val rand = new util.Random(config.randomSeed)

  def model = _model

  def train = {
    while (trees.size < config.iterationCount) {
      nextIteration()
    }
    model
  }

  def nextIteration() = {
    if (trees.size == 0) {
      // Initialize with constant value
      val mean = cost.optimalConstant(labelData)
      (0 until labelData.size) foreach (i => modelScores(i) = mean)
      _model = new DecisionTreeModel(Vector(new DecisionTreeLeaf(0, mean)))
      trees = trees :+ _model
    } else {
      val currentModel = model

      // Build training data to fit regression tree to gradient of the cost function
      val sampleSeed = rand.nextInt
      val columnSampler = sampler(sampleSeed, config.featureSampleRate)
      val rowSampler = sampler(sampleSeed, config.rowSampleRate)
      val sampledColumns = columns.filter(c => columnSampler(c.columnId))
      val residualData = for (c <- sampledColumns) yield {
        val data = c.all(rootRegion)
        val regressionInstances = mutable.ArraySeq.empty[Observation with Label[Double] with Feature] ++ data.zip(cost.gradient(data, modelScores)).
          map(t => Instance(t._1.rowId, t._1.featureValue, -t._2))
        new FeatureColumn[Double](regressionInstances, c.columnId)
      }

      // Regression fit to the gradient
      val regressionTrainer = new RegressionTreeTrainer(config.treeConfig, residualData, labelData.size, rowSampler)
      val tree = regressionTrainer.train

      // Optimize constant values at leaves of the tree
      val (leaves, splits) = tree.nodes.partition(_.isInstanceOf[DecisionTreeLeaf])
      val rowIdToRegionId = leaves.flatMap(l => residualData.head.all(regressionTrainer.partition(l.regionId)).map(row => (row.rowId, l.regionId))).toMap
      val regionIdToDelta = cost.optimalDelta(labelData.filter(r => rowSampler(r.rowId)), rowIdToRegionId, modelScores)
      for (row <- labelData) {
        modelScores(row.rowId) += regionIdToDelta(rowIdToRegionId(row.rowId)) * config.learningRate
      }
      val replacedLeaves = for (l <- leaves) yield {
        val delta = regionIdToDelta(l.regionId)
        new DecisionTreeLeaf(regionId = l.regionId, value = delta * config.learningRate)
      }
      val deltaModel = new DecisionTreeModel(splits ++ replacedLeaves)

      // Add tree to ensemble
      trees = trees :+ deltaModel
      _model = new AdditiveModel(trees)
    }
  }

  private def rowIdsInAscendingOrder(data: Seq[Observation]) = {
    data.head.rowId == 0 && data.take(data.size - 1).zip(data.drop(1)).forall(t => t._2.rowId == t._1.rowId + 1)
  }

}