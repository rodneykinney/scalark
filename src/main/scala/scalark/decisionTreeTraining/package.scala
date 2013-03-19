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
package scalark

import scala.collection._
import java.io._

package object decisionTreeTraining {

  /**
   * Add method toSortedColumns to sequences of rows, to produce column-wise data from row-wise data
   */
  implicit def RowsToSortedColumns[RowType <: RowOfFeatures](rows: Seq[RowType]) = new {
    def toSortedColumns[LabelType, ColumnType <: Observation with Feature with Label[LabelType]](columnBuilder: (RowType, Int) => ColumnType) = {
      for (col <- (0 until rows.head.features.length)) yield {
        val data = mutable.ArraySeq.empty[ColumnType] ++ rows.sortBy(_.features(col)).map(r => columnBuilder(r, col))
        new FeatureColumn[LabelType, ColumnType](data, col)
      }
    }
  }

  def rowsToColumns[LabelType, RowType <: RowOfFeatures, ColumnType <: Observation with Feature with Label[LabelType]](columnBuilder: (RowType, Int) => ColumnType)(rows: Seq[RowType]) = {
    for (col <- (0 until rows.head.features.length)) yield {
      val data = mutable.ArraySeq.empty[ColumnType] ++ rows.sortBy(_.features(col)).map(r => columnBuilder(r, col))
      new FeatureColumn[LabelType, ColumnType](data, col)
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
        ObservationRowLabel(rowId = fields(0).toInt, label = fields(2).toBoolean, features = fields.drop(3).map(_.toInt))
      }
    }
  }

  def using[A, B <: { def close(): Unit }](closeable: B)(f: B => A): A =
    try { f(closeable) } finally { closeable.close() }

  def sampler(seed: Int, sampleRate: Double) = {
    i: Int =>
      {
        new util.Random(java.nio.ByteBuffer.allocate(8).putInt(seed).putInt(i).getLong(0)).nextDouble < sampleRate
      }
  }
}
