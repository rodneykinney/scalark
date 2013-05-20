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
  implicit def rowsToSortedColumns[RowType <: RowOfFeatures](rows: Seq[RowType]) = new {
    def toSortedColumnData = {
      for (col <- (0 until rows.head.features.length)) yield {
        rows.zipWithIndex.view.sortBy(_._1.features(col)).map { case (r, i) => TrainableFeatureValue(rowId = i, featureValue = r.features(col)) }.force.toIndexedSeq
      }
    }
  }

  //TODO: Cleanup
  implicit def addWeightAndLabel(columns: immutable.Seq[Seq[TrainableFeatureValue]]) = new {
    def toFeatureColumns(weightFinder: Int => Double, labelFinder: Int => Double) = {
      for ((col, columnId) <- columns.zipWithIndex) yield {
        col.foreach{c => c.weight = weightFinder(c.rowId) ; c.label = labelFinder(c.rowId)}
        new FeatureColumn[Double, TrainableFeatureValue](col, columnId)
      }
    }
  }

  implicit def ValidateRowIds(data: Seq[Observation]) = new {
    def validate = data.head.rowId == 0 && data.zip(data.drop(1)).forall { case (first, second) => second.rowId == first.rowId + 1 }
  }

  implicit def ValidateRowAndQueryIds(data: Seq[Observation with Query with Label[Int]]) = new {
    def validate = data.head.rowId == 0 &&
      data.zip(data.drop(1)).forall {
        case (row, nextRow) =>
          nextRow.rowId == row.rowId + 1 &&
            (nextRow.queryId == row.queryId + 1 ||
              nextRow.label >= row.label)
      }
  }

  implicit def FileWriter(lines: Iterable[String]) = new {
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
    def readRows() = {
      val lines = io.Source.fromFile(file).getLines.toBuffer
      val columnNames = lines.head.split("\t")
      val labelIndex = columnNames.indexOf("#Label")
      require(labelIndex >= 0, "No column found with name #Label")
      val dropColumn = Set(columnNames.zipWithIndex.filter { case (name, index) => name.startsWith("#") }.map(_._2): _*)
      var rowId = -1
      for (line <- lines.drop(1)) yield {
        val fields = line.split('\t')
        val features = (0 until fields.size).filter(!dropColumn(_)).map(fields(_).toInt)
        rowId += 1
        LabeledRow(label = fields(labelIndex).toBoolean, features = features)
      }
    }
  }

  def using[A, B <: { def close(): Unit }](closeable: B)(f: B => A): A =
    try { f(closeable) } finally { closeable.close() }

  def sampler(size: Int, sampleRate: Double, rand: util.Random) = sampleRate match {
    case 1.0 => i: Int => true
    case _ => {
      val keepIds = new collection.mutable.BitSet(size)
      for (i <- (0 until size)) {
        if (rand.nextDouble < sampleRate) keepIds += i
      }
      keepIds
    }
  }
}