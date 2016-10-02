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

object TrainRanking extends SerializableLogging {

  def main(args: Array[String]) {
    val config = new TrainRankingConfig
    if (!config.parse(args))
      System.exit(0)

    val sgbConfig = new StochasticGradientBoostTrainConfig(iterationCount = config.numIterations,
      learningRate = config.learningRate,
      leafCount = config.leafCount,
      minLeafSize = config.minLeafSize,
      rowSampleRate = config.rowSampleRate,
      featureSampleRate = config.featureSampleRate,
      sampleRowsWithReplacement = config.withReplacement)

    this(trainConfig = sgbConfig, labelOrder = config.labelOrder.split(','), input = config.train, output = config.output, config.initialModel)
  }

  def apply(trainConfig: StochasticGradientBoostTrainConfig, labelOrder: IndexedSeq[String], input: String, output: String, initialModel: String) {
    logger.info("Training configuration: " + trainConfig)
    val rows = new java.io.File(input).readQueryRows(labelOrder).toList
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
    val trainer = new StochasticGradientBoostTrainer(trainConfig, new RankingCost(), labels, columns, startingTrees)
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

class TrainRankingConfig extends TrainClassificationConfig {
  var labelOrder: String = ""

  override def usage = {
    super.usage.takeWhile(_.required) ++
      List(required("labelOrder", "CSV list of expected relevance labels, ordered from best to worst")) ++
      super.usage.dropWhile(_.required)
  }
}