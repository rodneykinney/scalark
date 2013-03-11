package scalark.apps

import scalark.decisionTreeTraining._
import scalark.serialization._
import ModelSerialization._
import Extensions._
import spray.json._

object GenerateModel {
  def main(args: Array[String]) {
    val config = new GenerateModelConfig
    if (!config.parse(args))
      System.exit(0)

    this(nDim = config.nDim,
      nModesPerClass = config.nModesPerClass,
      outputFile = config.output)
  }

  def apply(nDim: Int, nModesPerClass: Int, outputFile: String, minFeatureValue: Int = 0, maxFeatureValue: Int = 1000) = {
    val synthesizer = new DataSynthesizer(nDim, minFeatureValue = minFeatureValue, maxFeatureValue = maxFeatureValue)
    val model = new BayesOptimalBinaryModel(synthesizer.gaussianMixtureModel(nModesPerClass), synthesizer.gaussianMixtureModel(nModesPerClass))
    using(new java.io.PrintWriter(new java.io.File(outputFile))) {
      w => w.println(model.toJson)
    }
  }
}

class GenerateModelConfig extends CommandLineParameters {
  var output: String = _
  var nDim: Int = 2
  var nModesPerClass: Int = 6

  def usage = {
    required("output", "Output TSV File") ::
      optional("nDim", "Number of dimensions") ::
      optional("rowCount", "Number of instances") ::
      optional("nModesPerClass", "Number of peaks in join distribution") ::
      Nil
  }
}
