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
import scalark.serialization._
import spray.json._
import java.io._

object GenerateTSV {
  def main(args: Array[String]) {
    val config = new GenerateTSVConfig
    if (!config.parse(args))
      System.exit(0)

    this(nDim = config.nDim,
      rowCount = config.rowCount,
      outputFile = config.output,
      modelFile = config.modelFile,
      labelCreator = config.labelCreator,
      fileFormat = config.format)
  }

  def apply[L](nDim: Int, rowCount: Int,
    outputFile: String,
    modelFile: String,
    minFeatureValue: Int = 0,
    maxFeatureValue: Int = 1000,
    labelCreator: Int => L,
    fileFormat: String = "TSV") = {
    val models = io.Source.fromFile(modelFile).getLines().map(_.asJson.convertTo[Model]).toSeq
    val synthesizer = new DataSynthesizer(nDim, minFeatureValue = minFeatureValue, maxFeatureValue = maxFeatureValue)
    val rows = synthesizer.generateData(rowCount, new GenerativeModel(models, labelCreator))
    fileFormat match {
      case "TSV" => {
        val featureFormat = List.fill(rows.head.features.size)("%d").mkString("\t")
        val lineFormat = "%d\t%s\t%s"
        val header = lineFormat.format("rowId", "label", (0 until rows.head.features.size).map(i => "feature" + i).mkString("\t"))
        using(new PrintWriter(new FileWriter(outputFile))) { writer =>
          writer.println(header)
          for (row <- rows) writer.println(lineFormat.format(row.rowId, row.label, featureFormat.format(row.features: _*)))
        }
      }
      case "ARFF" => {
        val writer = new java.io.PrintWriter(new java.io.FileWriter(outputFile))
        writer.println("@RELATION auto-generated-gaussian-mixture\n")
        (0 until nDim) foreach (i => writer.println("@ATTRIBUTE Column" + i + "\tNUMERIC"))
        writer.println("@ATTRIBUTE Label\t{true,false}")
        writer.println("\n@DATA")
        rows.foreach(row => writer.println(row.features.mkString(",") + "," + row.label))
        writer.close()
      }
    }
  }
}

class GenerateTSVConfig extends CommandLineParameters {
  var output: String = _
  var nDim: Int = 2
  var rowCount: Int = 100
  var format: String = "TSV"
  var labelType: String = "binary"
  var modelFile: String = _

  def usage = {
    required("modelFile", "File containing serialized model used to generate data") ::
      required("output", "Output TSV File") ::
      optional("nDim", "Number of dimensions") ::
      optional("rowCount", "Number of instances") ::
      optional("labelType", "Boolean | Int") ::
      optional("format", "TSV | ARFF") ::
      Nil
  }

  def labelCreator = labelType match {
    case "Boolean" => i: Int => i > 0
    case "Int" => i: Int => i
  }
}
