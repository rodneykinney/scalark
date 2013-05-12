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
/**
 * A column of data instances, each with a label and feature value
 *
 * Together with a TreePartition instance, this defines the data instances that exist within a given node of a decision tree
 */
class FeatureColumn[L, T <: Observation with Weight with Feature](val instances: mutable.IndexedSeq[T], val columnId: Int) {

  def this(immutableInstances: Seq[T], columnId: Int) = this(mutable.ArraySeq.empty[T] ++ immutableInstances, columnId)
  def size = instances.size

  /** All instances within the given region */
  def all(node: TreeRegion): IndexedSeq[T] = range(node, 0, node.size)

  /**
   * A subsequence of data instances within the given region
   */
  def range(node: TreeRegion, start: Int, end: Int): IndexedSeq[T] = {
    instances.slice(node.start + start, node.start + end)
  }

  /**
   * Retrieve a batch of instances from the given starting point
   * At least minimumSize instances (with weight > 0) will be returned, if available
   * All instances with the same feature value will be included in the batch
   */
  def batch(node: TreeRegion, start: Int, minimumSize: Int): List[T] = {
    var nodes = Vector.empty[T]
    def appendNode = (node: T) => nodes = nodes :+ node
    iterateBatch(node, start, minimumSize, appendNode)
    nodes.toList
  }

  def iterateBatch(node: TreeRegion, start: Int, minimumSize: Int, visitor: T => Any) = {
    val minSize = math.min(minimumSize, node.size - start)
    var currentIndex = node.start + start
    val lastIndex = node.start + node.size
    var nProcessed = 0
    while (nProcessed < minSize && currentIndex < lastIndex) {
      var current = instances(currentIndex)
      if (current.weight == 0) {
        currentIndex += 1
      } else {
        visitor(current)
        nProcessed += 1
        while (currentIndex + 1 < lastIndex
          && instances(currentIndex + 1).featureValue == current.featureValue) {
          current = instances(currentIndex + 1)
          visitor(current)
          nProcessed += 1
          currentIndex += 1
        }
        currentIndex += 1
      }
    }
    nProcessed
  }
  /**
   * Repartition the data within the parent node into the two child nodes
   * Maintain sort order of rows within each of the child nodes
   */
  def repartition(parent: TreeRegion, leftChild: TreeRegion, rightChild: TreeRegion, partition: Int => Boolean) = {
    val size = leftChild.size

    val tmp = instances.slice(parent.start, parent.start + parent.size).toIndexedSeq

    var iLeft = leftChild.start; var iRight = rightChild.start
    for (fi <- tmp) {
      if (partition(fi.rowId)) {
        instances(iLeft) = fi
        iLeft += 1
      } else {
        instances(iRight) = fi
        iRight += 1
      }
    }
  }

}