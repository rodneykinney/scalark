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

object TrainModel {
  def main(args: Array[String]) {
    val config = new TrainModelConfig
    if (!config.parse(args))
      System.exit(0)

    val sgbConfig = new StochasticGradientBoostTrainConfig(iterationCount = config.numIterations,
      learningRate = config.learningRate,
      leafCount = config.leafCount,
      minLeafSize = config.minLeafSize)

    this(sgbConfig, config.train, config.output)
  }

  def apply(trainConfig: StochasticGradientBoostTrainConfig, input: String, output: String) {
    val rows = new java.io.File(input).readRows
    val columns = rows.toSeq.toSortedColumns[Boolean, Observation with Feature with Label[Boolean]]
    val dummy = Vector.empty[ObservationLabelRowQuery[Int]].toSortedColumns[Int, Observation with Query with Feature with Label[Int]]
    val labels = rows.map(r => ObservationLabel(r.rowId, r.label)).toList
    val trees = new StochasticGradientBoostTrainer(trainConfig, new LogLogisticLoss().asInstanceOf[CostFunction[Boolean, Observation with Label[Boolean]]], labels, columns).train 
    val treesJson = trees.toJson
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

  def usage = {
    required("train", "In TSV file to use for training") ::
      required("numIterations", "Number of training iterations") ::
      required("output", "Output file containing trained trees") ::
      optional("learningRate", "Learning rate for gradient descent") ::
      optional("leafCount", "Number of leaf nodes per tree") ::
      optional("minLeafSize", "Minimum number of instances per leaf node") ::
      optional("rowSampleRate", "Number of rows to sample at each iteration") ::
      optional("featureSampleRate", "Number of features to sample at each iteration") ::
      Nil
  }
}