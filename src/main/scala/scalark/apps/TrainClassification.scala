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
import breeze.util._
import scalark.decisionTreeTraining._
import scalark.serialization._
import spray.json._

object TrainClassification extends SerializableLogging {

  def main(args: Array[String]) {
    val config = new TrainClassificationConfig
    if (!config.parse(args))
      System.exit(0)

    val sgbConfig = new StochasticGradientBoostTrainConfig(iterationCount = config.numIterations,
      learningRate = config.learningRate,
      leafCount = config.leafCount,
      minLeafSize = config.minLeafSize,
      rowSampleRate = config.rowSampleRate,
      featureSampleRate = config.featureSampleRate,
      sampleRowsWithReplacement = config.withReplacement)

    this(trainConfig = sgbConfig, input = config.train, output = config.output, config.initialModel)
  }

  def apply(trainConfig: StochasticGradientBoostTrainConfig, input: String, output: String, initialModel: String) {
    logger.info("Training configuration: " + trainConfig)
    val rows = new java.io.File(input).readRows().toList
    val columns = rows.toSortedColumnData
    val labels = rows.map(_.asTrainable).toIndexedSeq
    logger.info("Read " + labels.size + " rows from " + input)
    val startingTrees =
      if (initialModel == null)
    	Vector.empty[Model]
      else {
        logger.info("Reading initial model from "+initialModel)
        val model = io.Source.fromFile(initialModel).getLines.mkString.asJson.convertTo[AdditiveModel]
        // Initialize score
        for ((row,label) <- rows.zip(labels))
          label.score = model.eval(row.features)
        model.models
      }
    var iter = startingTrees.size
    val trainer = new StochasticGradientBoostTrainer(trainConfig, new LogLogisticLoss(), labels, columns, startingTrees)
    val start = System.currentTimeMillis()
    val trees = trainer.train({ logger.info("Iteration #" + iter); iter += 1 })
    val end = System.currentTimeMillis()
    logger.info("Training complete in " + (end - start).toDouble / 1000 + " seconds")
    val treesJson = trees.toJson
    using(new java.io.PrintWriter(new java.io.File(output))) {
      p => p.println(treesJson)
    }
    logger.info("Saved model to " + output)
  }
}

class TrainClassificationConfig extends CommandLineParameters {
  var train: String = "train.tsv"
  var numIterations: Int = 100
  var output: String = "trees.json"
  var initialModel: String = _
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
      optional("initialModel", "File containing model for starting point") ::
      optional("learningRate", "Learning rate for gradient descent") ::
      optional("leafCount", "Number of leaf nodes per tree") ::
      optional("minLeafSize", "Minimum number of instances per leaf node") ::
      optional("rowSampleRate", "Number of rows to sample at each iteration") ::
      optional("withReplacement", "Sample rows with replacement") ::
      optional("featureSampleRate", "Number of features to sample at each iteration") ::
      Nil
  }
}