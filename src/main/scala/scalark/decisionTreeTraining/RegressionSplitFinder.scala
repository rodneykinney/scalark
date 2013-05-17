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

/**
 * Finds best threshold at which to split a feature, based on mean-squared error
 */
class RegressionSplitFinder(minLeafSize: Int) {

  /**
   * Given a column of input data, scan through all possible threshold values and pick the split with the smallest loss
   */
  def findSplitCandidate[T <: Observation with Label[Double] with Weight with Feature](column: FeatureColumn[Double, T], node: TreeRegion) = {
    // Statistics of points in the left split
    val stats = new Stats[T]
    var batchSize = column.iterateBatch(node, 0, minLeafSize, stats.accumulate _)
    var lCount = batchSize
    var lWgt = stats.wgt
    var lSum = stats.sum
    var threshold = stats.lastFeatureValue

    // Statistics of points in the right split
    stats.reset
    batchSize = column.iterateBatch(node, lCount, node.size, stats.accumulate _)
    var rCount = batchSize
    var rWgt = stats.wgt
    var rSum = stats.sum

    if (lWgt < minLeafSize || rWgt < minLeafSize)
      None
    else {

      // Loss function before the split
      val originalLoss = -((lSum + rSum) * (lSum + rSum)) / (lWgt + rWgt)

      // Loss at the current split
      var loss = -lSum * lSum / lWgt - rSum * rSum / rWgt
      var minLoss = loss

      // Split candidate with the lowest current loss
      var candidate = new SplitCandidate(node.regionId, column.columnId, threshold, originalLoss - loss, lSum / lWgt, rSum / rWgt)

      // Update loss while we move points from right to left
      while (rWgt >= minLeafSize) {
        stats.reset
        batchSize = column.iterateBatch(node, lCount, 1, stats.accumulate _)
        lCount += batchSize
        lWgt += stats.wgt
        lSum += stats.sum
        rCount -= batchSize
        rWgt -= stats.wgt
        rSum -= stats.sum
        loss = -lWgt * lSum * lSum / (lWgt * lWgt) - rWgt * rSum * rSum / (rWgt * rWgt)
        if (loss < minLoss) {
          minLoss = loss
          candidate = new SplitCandidate(node.regionId, column.columnId, stats.lastFeatureValue, originalLoss - loss, lSum / lWgt, rSum / rWgt)
        }
      }
      Some(candidate)
    }
  }

  class Stats[T <: Label[Double] with Weight with Feature] {
    var wgt = 0.0
    var sum = 0.0
    var lastFeatureValue = 0.0
    def accumulate(f: T) = {
      wgt += f.weight
      sum += f.label * f.weight
      lastFeatureValue = f.featureValue
    }
    def reset = {
      wgt = 0.0
      sum = 0.0
      lastFeatureValue = 0
    }
  }
}