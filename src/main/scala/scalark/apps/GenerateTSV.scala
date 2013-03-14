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
      format = config.format)
  }

  def apply(nDim: Int, rowCount: Int,
    outputFile: String,
    modelFile:String,
    minFeatureValue: Int = 0,
    maxFeatureValue: Int = 1000,
    format: String = "TSV") = {
    val model = io.Source.fromFile(modelFile).getLines.mkString(" ").asJson.convertTo[Model]
    val synthesizer = new DataSynthesizer(nDim, minFeatureValue = minFeatureValue, maxFeatureValue = maxFeatureValue)
    val rows = synthesizer.binaryClassificationData(rowCount, model)
    format match {
      case "TSV" => {
        val format = (1 to rows.head.features.size).map(i => "%d").mkString("\t")
        rows.map(row => row.rowId + "\t" + 1 + "\t" + row.label + "\t" + format.format(row.features: _*)).writeToFile(outputFile)
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
  var modelFile: String = _

  def usage = {
    required("modelFile", "File containing serialized model used to generate data") ::
      required("output", "Output TSV File") ::
      optional("nDim", "Number of dimensions") ::
      optional("rowCount", "Number of instances") ::
      optional("format", "TSV | ARFF") ::
      Nil
  }
}
