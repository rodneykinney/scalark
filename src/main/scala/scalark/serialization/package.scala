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
package scalark

import scalark.decisionTreeTraining._
import spray.json._
import DefaultJsonProtocol._

/**
 * Serialize different kinds of models using spray-json
 */
package object serialization extends DefaultJsonProtocol {
  implicit val leafFormat = jsonFormat2(DecisionTreeLeaf)
  implicit val splitFormat = jsonFormat2(Split)
  implicit val splitNodeFormat = jsonFormat4(DecisionTreeSplit)
  implicit val gaussianFormat = jsonFormat(GaussianModel, "means", "variance", "featureIndices", "range")

  implicit object decisionTreeNodeFormat extends RootJsonFormat[DecisionTreeNode] {
    def write(n: DecisionTreeNode) = n match {
      case l: DecisionTreeLeaf => l.toJson 
      case s: DecisionTreeSplit => s.toJson
    }

    def read(value: JsValue) = value match {
      case v if v.asJsObject.getFields("split").length > 0 => v.convertTo[DecisionTreeSplit]
      case v if v.asJsObject.getFields("value").length > 0 => v.convertTo[DecisionTreeLeaf]
      case _ => deserializationError("DecisionTreeNode expected")
    }
  }

  implicit object decisionTreeModelFormat extends RootJsonFormat[DecisionTreeModel] {
    def write(m: DecisionTreeModel) = JsObject("nodes" -> m.nodes.toList.toJson)

    def read(value: JsValue) = value.asJsObject.getFields("nodes") match {
      case Seq(JsArray(nodes)) => new DecisionTreeModel(nodes.map(_.convertTo[DecisionTreeNode]))
      case _ => deserializationError("DecisionTreeModel expected")
    }
  }

  implicit object genericModelFormat extends RootJsonFormat[Model] {
    def write(m: Model) = m match {
      case tree: DecisionTreeModel => tree.toJson
      case gaussian: GaussianModel => gaussian.toJson
      case add: AdditiveModel => add.toJson
      case _ => serializationError("Unknown model type:  " + m)
    }
    def read(value: JsValue) = value match {
      case v if v.asJsObject.getFields("models").length > 0 => v.convertTo[AdditiveModel]
      case v if v.asJsObject.getFields("means").length > 0 => v.convertTo[GaussianModel]
      case v if v.asJsObject.getFields("nodes").length > 0 => v.convertTo[DecisionTreeModel]
      case _ => deserializationError("Unknown Model format: " + value)
    }
  }

  implicit val additiveModelFormat = jsonFormat1(AdditiveModel)

}