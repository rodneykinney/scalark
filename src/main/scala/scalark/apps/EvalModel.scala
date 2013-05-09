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
import spray.json._
import DefaultJsonProtocol._
import scalark.serialization._

object EvalModel {
  def main(args: Array[String]) {
    val config = new EvalModelConfig()
    if (!config.parse(args))
      System.exit(0)
    this(dataFile = config.data, modelFile = config.model, outputWriter = config.outputWriter)
  }

  def apply[L, T <: Label[L], W <: { def println(s: String); def close() }](dataFile: String, modelFile: String, outputWriter: W) = {
    val rows = new java.io.File(dataFile).readRows.toList
    val modelJson = io.Source.fromFile(modelFile).getLines.mkString.asJson
    val models = modelJson match {
      case l:JsArray => modelJson.convertTo[List[Model]]
      case _ => List(modelJson.convertTo[Model])
    }
    val eval = new EvalModel(models, rows)
    val accuracy = eval(BinaryAccuracy)
    val pr = eval(PrecisionRecall)
    outputWriter.println("Iteration\tAccuracy\tPrecision\tRecall")
    for ((((p, r), a), iter) <- pr.zip(accuracy).zipWithIndex) {
      outputWriter.println("%d\t%f\t%f\t%f".format(iter, a, p, r))
    }
    outputWriter.close()
  }
}

class EvalModel[T <: Observation with Weight with Label[Boolean] with RowOfFeatures](models: List[Model], rows: Seq[T]) {
  val scoredRowSets = models match {
    case (a: AdditiveModel) :: Nil => {
      var cumulative = new AdditiveModel(Vector.empty[Model])
      for (m <- a.models) yield {
        cumulative = new AdditiveModel(cumulative.models :+ m)
        val scoredRows = rows.map(r => r.withScore(cumulative.eval(r.features)))
        scoredRows
      }
    }
    case _ => {
      List(rows.map(r => r.withScore(models(1).eval(r.features) - models(0).eval(r.features))))
    }
  }

  def apply[MetricResult](metric: Metric[Boolean, Label[Boolean], MetricResult]) = {
    for (rowSet <- scoredRowSets) yield metric.compute(rowSet)
  }
}

class EvalModelConfig extends CommandLineParameters {
  var data: String = _
  var model: String = _
  var output: String = _

  def outputWriter = {
    output match {
      case file if file != null => {
        val writer = new java.io.PrintWriter(new java.io.FileWriter(file))
        new {
          def println(s: String) = writer.println(s)
          def close() = writer.close()
        }
      }
      case _ =>
        new {
          def println(s: String) = Predef.println(s)
          def close() = ()
        }
    }
  }

  def usage = {
    required("data", "Input TSV file containing test data") ::
      required("model", "JSON-format file containing model") ::
      optional("output", "Output file containing results") ::
      Nil
  }
}