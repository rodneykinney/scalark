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
package scalark.apps
import spray.json._
import DefaultJsonProtocol._
import scalark.decisionTreeTraining._
import scalark.serialization.ModelSerialization._

object SprayJsonTest {
  def main(args: Array[String]) = {
    val source = """{ "some": "JSON source" }"""
    val jsonAst = source.asJson // or JsonParser(source)
    println(jsonAst)
    val argsAsJson = List(1, 2, 3).toJson
    println(argsAsJson)

    val tree = new DecisionTreeModel(List(new DecisionTreeSplit(0, 1, 2, new Split(0, 1)), new DecisionTreeLeaf(1, 0.0), new DecisionTreeLeaf(2, 1.0)))
    val jsonTree = tree.toJson
    println(jsonTree)
    val decoded = jsonTree.convertTo[DecisionTreeModel]
    println(decoded)
    
    val jsonTreeList = List(tree).toJson
  }
}