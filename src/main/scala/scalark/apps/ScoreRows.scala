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

import java.io.{FileWriter, PrintWriter, File}

import spray.json._
import scala.io.Source
import scalark.decisionTreeTraining._
import scalark.serialization._

object ScoreRows {
  def main(args: Array[String]) {
    val config = new ScoreRowsConfig
    if (!config.parse(args))
      System.exit(0)

    apply(dataFile = config.data, modelFile = config.model, output = config.output)
  }
  def apply(dataFile: String, modelFile: String, output: String) = {
    val rows = new File(dataFile).readRows()
    val model = io.Source.fromFile(modelFile).getLines.mkString.asJson.convertTo[Model]
    using(new PrintWriter(new FileWriter(output))) { writer =>
      val lines = Source.fromFile(dataFile).getLines.toList
      writer.println(s"${lines.head}\tScore")
      for ((row, line) <- rows.zip(lines.tail)) {
        writer.println(s"$line\t${model.eval(row.features)}")
      }
//      writer.println("#Label\tScore")
//      for (row <- rows) writer.println(row.label+"\t"+model.eval(row.features))
    }
  }
}

class ScoreRowsConfig extends CommandLineParameters {
  var data: String = _
  var model: String = _
  var output: String = _

  def usage = {
    required("data", "Input TSV containing features") ::
      required("model", "Model file in JSON format") ::
      required("output", "Output TSV containing scores") ::
      Nil
  }
}