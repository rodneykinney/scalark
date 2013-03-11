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
    this(config.dataFile, config.modelFile)
  }

  def apply(dataFile: String, modelFile: String) = {
    val rows = new java.io.File(dataFile).readRows.toSeq
    val models = io.Source.fromFile(modelFile).getLines.map(_.asJson.convertTo[Model]).toList
    val rowCount = rows.length
    val cost = new LogLogisticLoss
    for ((iter, m) <- (1 to models.length).zip(models)) {
      println("Iteration #%d accuracy = %f".format(iter, rows.count(r => m.eval(r.features) < 0 ^ r.label).toDouble / rowCount))
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