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
class FeatureColumn[L, T <: Observation with Weight with Feature](val instances: mutable.IndexedSeq[T], val columnId: Int) extends Serializable {

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
    var currentIndex = node.start + start
    val endIndex = node.start + node.size
    val lastIndex = node.start + math.min(start + minimumSize, node.size)
    while (currentIndex < lastIndex) {
      var current = instances(currentIndex)
      if (current.weight == 0) {
        currentIndex += 1
      } else {
        visitor(current)
        currentIndex += 1
        while (currentIndex < endIndex
          && instances(currentIndex).featureValue == current.featureValue) {
          current = instances(currentIndex)
          visitor(current)
          currentIndex += 1
        }
      }
    }
    currentIndex - (node.start + start)
  }
  /**
   * Repartition the data within the parent node into the two child nodes
   * Maintain sort order of rows within each of the child nodes
   */
  def repartition(parent: TreeRegion, leftChild: TreeRegion, rightChild: TreeRegion, moveToLeft: Int => Boolean) = {
    val (leftInstances, rightInstances) = all(parent).partition(i => moveToLeft(i.rowId))
    for (index <- (0 until leftInstances.size)) instances(leftChild.start + index) = leftInstances(index)
    for (index <- (0 until rightInstances.size)) instances(rightChild.start + index) = rightInstances(index)
    this
  }

  // Seems like this ought to be faster, but it isn't
  private def repartitionAlternate(parent: TreeRegion, leftChild: TreeRegion, rightChild: TreeRegion, moveToLeft: Int => Boolean) = {
    val queue = new mutable.ListBuffer[T]()
    var index = leftChild.start
    for (i <- (parent.start until parent.start + parent.size)) {
      val instance = instances(i)
      if (moveToLeft(instance.rowId)) {
        instances(index) = instance
        index += 1
      } else {
        queue += instance
      }
    }
    index = rightChild.start
    for (i <- queue) {
      instances(index) = i
      index += 1
    }
  }

}

class FeatureColumn2[L, T <: Observation with Weight with Feature](regionContents: Map[Int, IndexedSeq[T]], columnId: Int) extends Serializable {

  def this(instances: IndexedSeq[T], columnId: Int) = this(Map(0 -> instances), columnId)
  def size = regionContents.values.map(_.size).sum

  /** All instances within the given region */
  def all(node: TreeRegion): Iterable[T] = regionContents(node.regionId)

  /**
   * A subsequence of data instances within the given region
   */
  def range(node: TreeRegion, start: Int, end: Int) = {
    regionContents(node.regionId).slice(start, end)
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
    val instances = regionContents(node.regionId)
    var currentIndex = start
    val endIndex = node.size
    val lastIndex = math.min(start + minimumSize, node.size)
    while (currentIndex < lastIndex) {
      var current = instances(currentIndex)
      if (current.weight == 0) {
        currentIndex += 1
      } else {
        visitor(current)
        currentIndex += 1
        while (currentIndex < endIndex
          && instances(currentIndex).featureValue == current.featureValue) {
          current = instances(currentIndex)
          visitor(current)
          currentIndex += 1
        }
      }
    }
    currentIndex - start
  }
  /**
   * Repartition the data within the parent node into the two child nodes
   * Maintain sort order of rows within each of the child nodes
   */
  def repartition(parent: TreeRegion, leftChild: TreeRegion, rightChild: TreeRegion, moveToLeft: Int => Boolean) = {
    val (leftInstances, rightInstances) = all(parent).partition(i => moveToLeft(i.rowId))
    val newMap = regionContents - parent.regionId + ((leftChild.regionId, leftInstances.toIndexedSeq), (rightChild.regionId, rightInstances.toIndexedSeq))
    new FeatureColumn2(newMap, columnId)
  }

}