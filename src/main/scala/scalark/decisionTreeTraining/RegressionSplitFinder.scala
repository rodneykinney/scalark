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
    var batch = column.batch(node, 0, minLeafSize)
    val scLeft = sumAndCount(batch)
    var lCount = scLeft._1
    var lWgt = scLeft._2
    var lSum = scLeft._3

    // Statistics of points in the right split
    val scRight = sumAndCount(column.range(node, lCount, node.size))
    var rCount = scRight._1
    var rWgt = scRight._2
    var rSum = scRight._3

    if (lWgt < minLeafSize || rWgt < minLeafSize)
      None
    else {

      // Loss function before the split
      val originalLoss = - ((lSum + rSum) * (lSum + rSum)) / (lWgt + rWgt)

      // Loss at the current split
      var loss = -lSum * lSum / lWgt - rSum * rSum / rWgt
      var minLoss = loss

      // Split candidate with the lowest current loss
      var candidate = new SplitCandidate(node.regionId, column.columnId, scLeft._4, originalLoss - loss, lSum / lWgt, rSum / rWgt)

      // Update loss while we move points from right to left
      var stats = scLeft
      while (rWgt >= minLeafSize) {
        batch = column.batch(node, lCount, 1)
        stats = sumAndCount(batch)
        lCount += stats._1
        lWgt += stats._2
        lSum += stats._3
        rCount -= stats._1
        rWgt -= stats._2
        rSum -= stats._3
        loss = -lWgt * lSum * lSum / (lWgt * lWgt) - rWgt * rSum * rSum / (rWgt * rWgt)
        if (loss < minLoss) {
          minLoss = loss
          candidate = new SplitCandidate(node.regionId, column.columnId, stats._4, originalLoss - loss, lSum / lWgt, rSum / rWgt)
        }
      }
      Some(candidate)
    }
  }

  private def sumAndCount[T <: Label[Double] with Weight with Feature](s: Seq[T]) = {
    var count = 0
    var wgt = 0.0
    var sum = 0.0
    var lastfeatureValue = 0
    for (f <- s) {
      count += 1
      wgt += f.weight
      sum += f.label * f.weight
      lastfeatureValue = f.featureValue
    }
    (count, wgt, sum, lastfeatureValue)
  }
}