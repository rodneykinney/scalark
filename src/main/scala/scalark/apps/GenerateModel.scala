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

object GenerateModel {
  def main(args: Array[String]) {
    val config = new GenerateModelConfig
    if (!config.parse(args))
      System.exit(0)

    this(nDim = config.nDim,
      nModesPerClass = config.nModesPerClass,
      outputFile = config.output,
      seed = config.seed,
      spikiness = config.spikiness)
  }

  def apply(nDim: Int, nModesPerClass: Int, outputFile: String, minFeatureValue: Int = 0, maxFeatureValue: Int = 1000, seed: Int = 117, spikiness: Double = 1.0) = {
    val synthesizer = new DataSynthesizer(nDim, minFeatureValue = minFeatureValue, maxFeatureValue = maxFeatureValue, seed)
    val models = List(synthesizer.gaussianMixtureModel(nModesPerClass, spikiness), synthesizer.gaussianMixtureModel(nModesPerClass, spikiness))
    using(new java.io.PrintWriter(new java.io.File(outputFile))) {
      w => for (m <- models) w.println(m.toJson)
    }
  }
}

class GenerateModelConfig extends CommandLineParameters {
  var output: String = _
  var nDim: Int = 2
  var nModesPerClass: Int = 6
  val seed: Int = 117
  val spikiness: Double = 2.5

  def usage = {
    required("output", "Output TSV File") ::
      optional("nDim", "Number of dimensions") ::
      optional("nModesPerClass", "Number of peaks in single-class distribution") ::
      optional("seed", "Random seed") ::
      optional("spikiness", "Inverse width of peaks in single-class distribution") ::
      Nil
  }
}
