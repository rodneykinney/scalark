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
 * Partitions a set of rows into tree nodes
 */
class TreePartition(size: Int) {
  private var nodes = Vector(new TreeRegion(0, 0, size))

  def apply(i: Int) = nodes(i)

  def root = apply(0)

  def nodeCount = nodes.size

  def left(node: TreeRegion) = apply(node.leftChildId)
  def right(node: TreeRegion) = apply(node.rightChildId)

  /**
   * Split a parent region into two child regions
   */
  def split(parent: TreeRegion, leftSize: Int) = {
    val leftId = nodes.size
    val left = new TreeRegion(leftId, parent.start, leftSize)
    nodes = nodes :+ left
    parent.leftChildId = leftId

    val right = new TreeRegion(leftId + 1, parent.start + leftSize, parent.size - leftSize)
    nodes = nodes :+ right
    parent.rightChildId = leftId + 1
    (left, right)
  }
}

/**
 * Describes a range of data within an array representing one node of a decision tree
 */
@serializable
case class TreeRegion(val regionId: Int, val start: Int, val size: Int) {
  var leftChildId: Int = -1
  var rightChildId: Int = -1
}
