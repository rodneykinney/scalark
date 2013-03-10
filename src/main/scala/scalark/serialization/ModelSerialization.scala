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
package scalark.serialization

import scalark.decisionTreeTraining._
import spray.json._
import DefaultJsonProtocol._

/**
 * Serialize decision tree models using spray-json
 */
object ModelSerialization extends DefaultJsonProtocol {
  implicit val leafFormat = jsonFormat2(DecisionTreeLeaf)
  implicit val splitFormat = jsonFormat2(Split)
  implicit val splitNodeFormat = jsonFormat4(DecisionTreeSplit)
  implicit val gaussianFormat = jsonFormat4(GaussianModel)

  implicit object decisionTreeNodeFormat extends RootJsonFormat[DecisionTreeNode] {
    def write(n: DecisionTreeNode) = n match {
      case l: DecisionTreeLeaf => l.toJson
      case s: DecisionTreeSplit => s.toJson
    }
    def read(value: JsValue) = value.asJsObject.getFields("regionId", "value", "leftId", "rightId", "split") match {
      case Seq(JsNumber(regionId), JsNumber(leftId), JsNumber(rightId), s: JsObject) => new DecisionTreeSplit(regionId.toInt, leftId.toInt, rightId.toInt, s.convertTo[Split])
      case Seq(JsNumber(regionId), JsNumber(value)) => new DecisionTreeLeaf(regionId.toInt, value.toDouble)
      case _ => deserializationError("DecisionTreeNode expected")
    }
  }

  implicit object decisionTreeModelFormat extends RootJsonFormat[DecisionTreeModel] {
    def write(m: DecisionTreeModel) = m.nodes.toList.toJson

    def read(value: JsValue) = value match {
      case JsArray(nodes) => new DecisionTreeModel(nodes.map(_.convertTo[DecisionTreeNode]))
      case _ => deserializationError("DecisionTreeModel expected")
    }
  }

  implicit object genericModelFormat extends RootJsonFormat[Model] {
    def write(m: Model) = m match {
      case tree: DecisionTreeModel => Map("modelType" -> "DecisionTreeModel", "model" -> tree.nodes.toList.toJson.toString).toJson
      case gaussian: GaussianModel => Map("modelType" -> "GaussianModel", "model" -> gaussian.toJson.toString).toJson
      case _ => serializationError("Unknown model type:  " + m)
    }
    def read(value: JsValue) = value match {
      case m: Map[String, String] => m("modelType") match {
        case "DecisionTreeModel" => m("model").asJson.convertTo[DecisionTreeModel]
        case "GaussianModel" => m("model").asJson.convertTo[GaussianModel]
        case _ => deserializationError("Unknown model type: " + m("model"))
      }
      case _ => deserializationError("Unknown Model format: " + value)
    }
  }

  implicit val additiveModelFormat = jsonFormat1(AdditiveModel)
  implicit val bayesOptimalBinaryFormat = jsonFormat2(BayesOptimalBinaryModel)

}