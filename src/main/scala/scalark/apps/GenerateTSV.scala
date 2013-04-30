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
      rowCount = config.nRows,
      outputFile = config.output,
      modelFile = config.model,
      labelCreator = config.labelCreator,
      fileFormat = config.format,
      minFeatureValue = config.minFeatureValue,
      maxFeatureValue = config.maxFeatureValue,
      seed = config.seed,
      predictability = config.predictability,
      labelColumnName = config.labelColumnName)
  }

  def apply[L](nDim: Int, rowCount: Int,
    outputFile: String,
    modelFile: String,
    minFeatureValue: Int = 0,
    maxFeatureValue: Int = 1000,
    seed: Int = 117,
    labelCreator: Int => L,
    fileFormat: String = "TSV",
    predictability: Double = 0.0,
    labelColumnName:String = "#Label") = {
    val models = io.Source.fromFile(modelFile).mkString.asJson.convertTo[List[Model]]
    val synthesizer = new DataSynthesizer(nDim, minFeatureValue = minFeatureValue, maxFeatureValue = maxFeatureValue, seed = seed)
    val genModel = new GenerativeModel(models, labelCreator)
    val optimalModel = new MaximumLikelihoodModel(models, labelCreator)
    val labelGenerator = (features: IndexedSeq[Int], rand:util.Random) => {
      if (rand.nextDouble < predictability) {
        optimalModel.assignLabel(features)
      }
      else {
        genModel.assignLabel(features, rand)
      }
    }
    val rows = synthesizer.generateData(rowCount, labelGenerator)
    fileFormat match {
      case "TSV" => {
        val featureFormat = List.fill(rows.head.features.size)("%d").mkString("\t")
        val lineFormat = "%s\t%s"
        val header = labelColumnName+"\t" + (0 until rows.head.features.size).map(i => "feature" + i).mkString("\t")
        using(new PrintWriter(new FileWriter(outputFile))) { writer =>
          writer.println(header)
          for (row <- rows) writer.println(lineFormat.format(row.label, featureFormat.format(row.features: _*)))
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
  var nRows: Int = 100
  var format: String = "TSV"
  var labelType: String = "Boolean"
  var model: String = _
  var minFeatureValue: Int = 0
  var maxFeatureValue: Int = 1000
  var seed: Int = 117
  var predictability: Double = 0.0
  var labelColumnName: String = "#Label"

  def usage = {
    required("model", "File containing serialized model used to generate data") ::
      required("output", "Output TSV File") ::
      optional("nDim", "Number of dimensions") ::
      optional("nRows", "Number of instances") ::
      optional("labelType", "Boolean | Int") ::
      optional("minFeatureValue", "lower limit of range of feature values") ::
      optional("maxFeatureValue", "upper limit of range for feature values") ::
      optional("seed", "random seed") ::
      optional("predictability","Probability of assigning label via maximum likelihood") ::
      optional("format", "TSV | ARFF") ::
      optional("labelColumnName","Name of column containing label in output file") ::
      Nil
  }

  def labelCreator = labelType match {
    case "Boolean" => i: Int => i > 0
    case "Int" => i: Int => i
  }
}
