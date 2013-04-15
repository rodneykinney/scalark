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
    this(config.dataFile, config.modelFile, new BinaryAccuracy)
  }

  def apply[L, T <: Label[L]](dataFile: String, modelFile: String, metric: Metric[L, T]) = {
    val rows = new java.io.File(dataFile).readRows.toSeq
    val models = io.Source.fromFile(modelFile).getLines.toList.map(_.asJson.convertTo[Model])
    val accuracy = new BinaryAccuracy
    val pr = new PrecisionRecall
    models match {
      case (a: AdditiveModel) :: Nil => {
        var cumulative = new AdditiveModel(Vector.empty[Model])
        for ((m, iter) <- a.models.zipWithIndex) {
          cumulative = new AdditiveModel(cumulative.models :+ m)
          val scoredRows = rows.map(r => r.withScore(cumulative.eval(r.features)))
          val a = accuracy.compute(scoredRows)
          val (p, r) = pr.compute(scoredRows)
          println("Iteration %d, Accuracy = %f, precision = %f, recall = %f".format(iter, a, p, r))
        }
      }
      case _ => {
        val scoredRows = rows.map(r => r.withScore(models(1).eval(r.features) - models(0).eval(r.features)))
        val a = accuracy.compute(scoredRows)
        val (p, r) = pr.compute(scoredRows)
        println("Accuracy = %f, precision = %f, recall = %f".format(a, p, r))
      }
    }
  }
}

class EvalModelConfig extends CommandLineParameters {
  var dataFile: String = _
  var modelFile: String = _

  def usage = {
    required("dataFile", "Input TSV file containing test data") ::
      required("modelFile", "JSON-format file containing model") ::
      Nil
  }
}