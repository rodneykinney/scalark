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
class FeatureColumn[L, +T <: Observation with Feature with Label[L]](val data: Seq[T], val columnId: Int) {
  private[this] val instances = mutable.ArraySeq.empty[T] ++ data

  def size = instances.size

  /** All instances within the given region */
  def all(node: TreeRegion): IndexedSeq[T] = range(node, 0, node.size)

  /**
   * A subsequence of data instances within the given region
   */
  def range(node: TreeRegion, start: Int, length: Int): IndexedSeq[T] = {
    for (i <- (node.start + start until node.start + length))
      yield instances(i)
  }

  /**
   * Retrieve a batch of instances from the given starting point
   * At least minimumSize instances (with weight > 0) will be returned, if available
   * All instances with the same feature value will be included in the batch
   */
  def batch(node: TreeRegion, start: Int, minimumSize: Int): List[T] = {
    val minSize = math.min(minimumSize, node.size - start)
    val current = instances(node.start + start)
    if (current.weight == 0) {
      minSize match {
        case 0 => Nil
        case _ => batch(node, start + 1, minSize)
      }
    } else {
      minSize match {
        case 0 => Nil
        case 1 => {
          val next = {
            if (node.start + start + 1 < node.start + node.size
              && instances(node.start + start + 1).featureValue
              == current.featureValue)
              batch(node, start + 1, 1)
            else Nil
          }
          current :: next
        }
        case _ => {
          current :: batch(node, start + 1, minSize - 1)
        }
      }
    }
  }

  /**
   * Repartition the data within the parent node into the two child nodes
   * Maintain sort order of rows within each of the child nodes
   */
  def repartition(parent: TreeRegion, leftChild: TreeRegion, rightChild: TreeRegion, partition: Int => Boolean) = {
    val size = leftChild.size

    val tmp = (parent.start until parent.start + parent.size).map(instances(_)).toIndexedSeq

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