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
import breeze.util.logging._
import scalark.decisionTreeTraining._
import scalark.serialization._

import spray.json._

object TrainModel extends ConfiguredLogging {

  def main(args: Array[String]) {
    val config = new TrainModelConfig
    if (!config.parse(args))
      System.exit(0)

    val sgbConfig = new StochasticGradientBoostTrainConfig(iterationCount = config.numIterations,
      learningRate = config.learningRate,
      leafCount = config.leafCount,
      minLeafSize = config.minLeafSize,
      rowSampleRate = config.rowSampleRate,
      sampleRowsWithReplacement = config.withReplacement)

    this(trainConfig = sgbConfig, input = config.train, output = config.output)
  }

  def apply(trainConfig: StochasticGradientBoostTrainConfig, input: String, output: String) {
    log.info("Reading data from "+input)
    val rows = new java.io.File(input).readRows.toList
    val columns = rows.toSortedColumns
    val labels = rows.map(r => ObservationLabel(r.rowId, 1.0, r.label)).toList
    log.info("Read "+labels.size+" rows")
    log.info("Training configuration: "+trainConfig)
    var iter = 0
    val trainer = new StochasticGradientBoostTrainer(trainConfig, new LogLogisticLoss(), labels, columns)
    val start = new java.util.Date()
    val trees = trainer.train({log.info("Iteration #"+iter) ; iter += 1})
    val end = new java.util.Date()
    log.info("Training complete in "+(end.getTime-start.getTime).toDouble/1000 + " seconds")
    val treesJson = trees.toJson
    log.info("Saving model to "+output)
    using (new java.io.PrintWriter(new java.io.File(output))) {
      p => p.println(treesJson)
    }
  }
}

class TrainModelConfig extends CommandLineParameters {
  var train: String = "train.tsv"
  var numIterations: Int = 100
  var output: String = "trees.json"
  var learningRate = 0.2
  var leafCount = 10
  var minLeafSize = 20
  var rowSampleRate = 1.0
  var featureSampleRate = 1.0
  var withReplacement = false

  def usage = {
    required("train", "In TSV file to use for training") ::
      required("numIterations", "Number of training iterations") ::
      required("output", "Output file containing trained trees") ::
      optional("learningRate", "Learning rate for gradient descent") ::
      optional("leafCount", "Number of leaf nodes per tree") ::
      optional("minLeafSize", "Minimum number of instances per leaf node") ::
      optional("rowSampleRate", "Number of rows to sample at each iteration") ::
      optional("withReplacement","Sample rows with replacement") ::
      optional("featureSampleRate", "Number of features to sample at each iteration") ::
      Nil
  }
}