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

/* Classes for evaluating a decision tree model at run-time */

abstract class DecisionTreeNode {
  val regionId: Int
  def isLeaf: Boolean
}

case class DecisionTreeLeaf(regionId: Int, value: Double) extends DecisionTreeNode {def isLeaf = true}

case class DecisionTreeSplit(regionId: Int, leftId: Int, rightId: Int, split: Split) extends DecisionTreeNode { def isLeaf = false}

class DecisionTreeModel(val nodes: Seq[DecisionTreeNode]) extends Model {
  private val nodesByIndex = nodes.map(n => (n.regionId, n)).toMap

  def Root = nodesByIndex(0)

  def eval(features: Seq[Double]) = eval(features, Root).value

  def regionId(features: Seq[Double]) = eval(features, Root).regionId

  private def eval(features: Seq[Double], node: DecisionTreeNode): DecisionTreeLeaf = node match {
    case leaf: DecisionTreeLeaf => leaf
    case split: DecisionTreeSplit =>
      if (features(split.split.columnId) <= split.split.threshold) eval(features, nodesByIndex(split.leftId))
      else eval(features, nodesByIndex(split.rightId))
  }

  def merge(model: DecisionTreeModel) = {
    val replacedIds = model.nodes.map(_.regionId).toSet
    new DecisionTreeModel(nodes.filter(n => !replacedIds.contains(n.regionId)) ++ model.nodes)
  }

  def leafCount = nodes.count(_.isLeaf)
}

case class Split(columnId: Int, threshold: Double)

