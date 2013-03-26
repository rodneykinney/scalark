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

import scalark.decisionTreeTraining._
import spray.json._
import DefaultJsonProtocol._
import scalark.serialization._

object EvalModel {
  def main(args: Array[String]) {
    val config = new EvalModelConfig()
    if (!config.parse(args))
      System.exit(0)
    this(config.dataFile, config.modelFile, new BinaryAccuracy)
  }

  def apply[L,T <: Label[L]](dataFile: String, modelFile: String, metric:Metric[L,T])= {
    val rows = new java.io.File(dataFile).readRows.toSeq
    val trees = io.Source.fromFile(modelFile).getLines.map(_.asJson.convertTo[Model]).toList
    var treesAtIteration = Vector.empty[Model]
    val accuracy = new BinaryAccuracy
    val pr = new PrecisionRecall
    for ((tree, iter) <- trees.zipWithIndex) {
      treesAtIteration = treesAtIteration :+ tree
      val model = new AdditiveModel(treesAtIteration)
      val scoredRows = rows.map(r => r.withScore(model.eval(r.features)))
      println("Iteration #%d, accuracy = %f, precision/recall = ".format(iter, accuracy.compute(scoredRows), pr.compute(scoredRows)))
    }
  }
}

class EvalModelConfig extends CommandLineParameters {
  var dataFile: String = _
  var modelFile: String = _

  def usage = {
    required("dataFile", "Input TSV file containing test data") ::
      required("modelFile", "JSON-format file containing model") ::
      Nil
  }
}