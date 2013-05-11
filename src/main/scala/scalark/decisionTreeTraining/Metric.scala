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
package scalark.decisionTreeTraining

trait Metric[L, T <: Label[L], MetricResult] {
  def compute[T1 <: T with Score](rows: Seq[T1]): MetricResult
}

object BinaryAccuracy extends Metric[Boolean, Label[Boolean], Double] {
  def compute[T <: Label[Boolean] with Score](rows: Seq[T]) = {
    var rowCount, errorCount = 0
    for (row <- rows) {
      rowCount += 1
      if (row.label ^ row.score > 0) errorCount += 1
    }
    1 - errorCount.toDouble / rowCount
  }
}

object PrecisionRecall extends Metric[Boolean, Label[Boolean], Tuple2[Double,Double]] {
  def compute[T <: Label[Boolean] with Score](rows: Seq[T]) = {
    val confusion = Array(0, 0, 0, 0)
    for (row <- rows) {
      val index = (if (row.label) 0 else 2) + (if (row.score > 0) 0 else 1)
      confusion(index) += 1
    }
    val precision = confusion(0).toDouble / (confusion(0) + confusion(2))
    val recall = confusion(0).toDouble / (confusion(0) + confusion(1))
    (precision, recall)
  }
}
