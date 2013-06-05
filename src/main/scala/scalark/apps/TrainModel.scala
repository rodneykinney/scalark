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
      featureSampleRate = config.featureSampleRate,
      sampleRowsWithReplacement = config.withReplacement)

    this(trainConfig = sgbConfig,
      outputModelFile = config.output,
      columnOps = config.columnOps,
      labels = config.labels)
  }

  def apply(trainConfig: StochasticGradientBoostTrainConfig,
    outputModelFile: String,
    columnOps: ColumnOperationsFactory,
    labels: IndexedSeq[Label[Boolean] with MutableWeight with MutableScore with MutableRegion]) {
    log.info("Training configuration: " + trainConfig)
    var iter = 0
    val trainer = new StochasticGradientBoostTrainer(trainConfig, new LogLogisticLoss(), labels, columnOps)
    val start = System.currentTimeMillis()
    val trees = trainer.train({ log.info("Iteration #" + iter); iter += 1 })
    val end = System.currentTimeMillis()
    log.info("Training complete in " + (end - start).toDouble / 1000 + " seconds")
    val treesJson = trees.toJson
    using(new java.io.PrintWriter(new java.io.File(outputModelFile))) {
      p => p.println(treesJson)
    }
    log.info("Saved model to " + outputModelFile)
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
  var format = "rows"
  var distributed = false

  private lazy val rows = {
    val rowList = new java.io.File(train).readRows.toList
    TrainModel.log.info("Read " + rowList.size + " rows from " + train)
    rowList
  }

  def columnOps = format match {
    case "rows" => {
      val columns = rows.toSortedColumnData
      new ParallelColumnOperationsFactory(columns.par)
    }
    case "columns" if distributed => {
      //TODO: Support distributed column loading
      throw new IllegalArgumentException("Distributed not supported")
      //      val sc = new spark.SparkContext("local[4]", "TrainModel")
      //      val columns = sc.textFile(train) map(parseColumn(_))
    }
    case "columns" => {
      val src = io.Source.fromFile(train)
      val columns = src.getLines.map(_.parseColumnData).toIndexedSeq
      val colOps = new ParallelColumnOperationsFactory(columns.par)
      src.close
      colOps
    }
    case _ => throw new IllegalArgumentException("Illegal format name: " + format)
  }

  def labels = format match {
    case "rows" => rows.map(_.asTrainable).toIndexedSeq
    case "columns" => {
      val src = io.Source.fromFile(train+".labels.tsv")
      val l = src.getLines.map(l => new TrainableLabel(l.toBoolean)).toIndexedSeq
      src.close
      l
    }
    case _ => throw new IllegalArgumentException("Illegal format name: " + format)
  }

  def usage = {
    required("train", "In TSV file to use for training") ::
      required("numIterations", "Number of training iterations") ::
      required("output", "Output file containing trained trees") ::
      optional("format", "Input data format [rows|columns]") ::
      optional("learningRate", "Learning rate for gradient descent") ::
      optional("leafCount", "Number of leaf nodes per tree") ::
      optional("minLeafSize", "Minimum number of instances per leaf node") ::
      optional("rowSampleRate", "Number of rows to sample at each iteration") ::
      optional("withReplacement", "Sample rows with replacement") ::
      optional("featureSampleRate", "Number of features to sample at each iteration") ::
      Nil
  }
}