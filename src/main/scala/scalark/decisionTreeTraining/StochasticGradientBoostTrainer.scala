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
  cost: CostFunction[Boolean, LabelInstance[Boolean]],
  columns: immutable.Seq[FeatureColumn[Boolean]]) {

  private var trees = Vector.empty[Model]
  private val rootRegion = new TreeRegion(0, 0, columns.head.size)
  private var _model: Model = _
  private val modelScores:mutable.Map[Int,Double] = new mutable.HashMap[Int,Double]()
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
      val mean = cost.optimalConstant(columns.head.all(rootRegion))
      (0 until columns.head.size) foreach (i => modelScores(i) = mean)
      _model = new DecisionTreeModel(Vector(new DecisionTreeLeaf(0, mean)))
      trees = trees :+ _model
    } else {
      val currentModel = model

      // Build training data to fit regression tree to gradient of the cost function
      val sampledColumns = columns.filter(c => rand.nextDouble() < config.featureSampleRate)
      val sampledRows = columns.head.all(rootRegion).filter(r => rand.nextDouble() < config.rowSampleRate).map(_.rowId).toSet 
      val residualData = for (c <- sampledColumns) yield {
        val instances = c.all(rootRegion).filter(r => sampledRows.contains(r.rowId))
        val regressionInstances = mutable.ArraySeq.empty[FeatureInstance[Double]] ++ instances.zip(cost.gradient(instances, modelScores)).
          map(t => { val (i, g) = t; new FeatureInstanceDelegate[Double, Boolean](i, -g) })
        new FeatureColumn[Double](regressionInstances, c.columnId)
      }
      
      // Regression fit to the gradient
      val regressionTrainer = new RegressionTreeTrainer(config.treeConfig, residualData, columns.head.size)
      val tree = regressionTrainer.train

      // Optimize constant values at leaves of the tree
      val (leaves, splits) = tree.nodes.partition(_.isInstanceOf[DecisionTreeLeaf])
//      val instancesInLeafRegions = leaves.map(l => residualData.head.all(regressionTrainer.partition(l.regionId)).map(_.asInstanceOf[FeatureInstanceDelegate[Double, Boolean]]._2))
//      val deltas = cost.optimalDelta(instancesInLeafRegions, modelScores)
//      val replacedLeaves = leaves.zip(deltas).map(t => {
//        val (l,v) = t
//        residualData.head.all(regressionTrainer.partition(l.regionId)).foreach(i => modelScores(i.rowId) += v * config.learningRate)
//        new DecisionTreeLeaf(regionId = l.regionId, value = v * config.learningRate)
//      })
      val rowIdToRegionId = leaves.flatMap(l => residualData.head.all(regressionTrainer.partition(l.regionId)).map(row => (row.rowId,l.regionId))).toMap
      val regionIdToDelta = cost.optimalDelta(columns.head.all(rootRegion).filter(sampledRows contains _.rowId), rowIdToRegionId, modelScores)
      for (row <- columns.head.all(rootRegion)) {
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

}