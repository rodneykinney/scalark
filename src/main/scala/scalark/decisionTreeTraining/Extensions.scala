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

import scala.collection._
import java.io._

object Extensions {

  /**
   * Add method toSortedColumns to sequences of rows, to produce column-wise data from row-wise data
   */
  implicit def RowsToSortedColumns[LabelType](rows: Seq[LabeledFeatureRow[LabelType]]) = new {
    def toSortedColumns = {
      for (col <- (0 until rows.head.features.length)) yield {
        val data = mutable.ArraySeq.empty[FeatureInstance[LabelType]] ++ rows.map(r => FeatureInstance[LabelType](rowId = r.rowId, featureValue = r.features(col), weight = r.weight, label = r.label)).sortBy(_.featureValue)
        new FeatureColumn[LabelType](data, col)
      }
    }
  }

  implicit def FileWriter(lines: Seq[String]) = new {
    def writeToFile(path: String) = {
      val writer = new PrintWriter(new FileWriter(path))
      try {
        lines.foreach(l => writer.println(l))
      } finally {
        writer.close()
      }
    }
  }

  implicit def FileRowReader(file: java.io.File) = new {
    def readRows = {
      for (line <- io.Source.fromFile(file).getLines()) yield {
        val fields = line.split('\t')
        new LabeledFeatureRow[Boolean](rowId = fields(0).toInt, weight = fields(1).toDouble, label = fields(2).toBoolean, features = fields.drop(3).map(_.toInt))
      }
    }
  }

  def using[A, B <: { def close(): Unit }](closeable: B)(f: B => A): A =
    try { f(closeable) } finally { closeable.close() }
}