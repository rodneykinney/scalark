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

  implicit def addWeightAndLabel(columns: immutable.Seq[Seq[Observation with Feature]]) = new {
    def toFeatureColumns[L](weightFinder: Int => Double, labelFinder: Int => L) = {
      for ((col, columnId) <- columns.zipWithIndex) yield {
        val instances = mutable.ArraySeq.empty[Observation with Feature with Weight with Label[L]] ++
          (for (c <- col) yield new Observation with Feature with MutableWeight with MutableLabel[L] {
            def rowId = c.rowId
            def featureValue = c.featureValue
            var weight = weightFinder(c.rowId)
            var label = labelFinder(c.rowId)
          })
        new FeatureColumn[L, Observation with Feature with Weight with Label[L]](instances, columnId)
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
    def readRows(labelColumnName:String = "#Label") = {
      val lines = io.Source.fromFile(file).getLines.toBuffer
      val columnNames = lines.head.split("\t")
      val labelIndex = columnNames.indexOf(labelColumnName)
      require(labelIndex >= 0, "No column found with name "+labelColumnName)
      val dropColumn = columnNames.zipWithIndex.filter { case (name, index) => name.startsWith("#") }.map(_._2).toSet
      var rowId = -1
      for (line <- lines.drop(1)) yield {
        val fields = line.split('\t')
        val features = (0 until fields.size).filter(!dropColumn(_)).map(fields(_).toDouble)
        rowId += 1
        val label = if (fields(labelIndex) == "1")
          true
        else if (fields(labelIndex) == "0")
          false
        else fields(labelIndex).toBoolean
        LabeledRow(label, features = features)
      }
    }
    def readQueryRows(labelOrder: IndexedSeq[String], labelColumnName:String = "#Label", queryColumnName: String = "#Query") = {
      val lines = io.Source.fromFile(file).getLines.toBuffer
      val columnNames = lines.head.split("\t")
      val labelIndex = columnNames.indexOf(labelColumnName)
      val queryIndex = columnNames.indexOf(queryColumnName)
      val queryToId = mutable.HashMap.empty[String, Int]
      require(labelIndex >= 0, "No column found with name "+labelColumnName)
      val dropColumn = columnNames.zipWithIndex.filter { case (name, index) => name.startsWith("#") }.map(_._2).toSet
      var rowId = -1
      val rows = for (line <- lines.drop(1)) yield {
        val fields = line.split('\t')
        val features = fields.indices.filter(!dropColumn(_)).map(fields(_).toDouble)
        rowId += 1
        val label = {
          val l = labelOrder.indexOf(fields(labelIndex))
          require(l >= 0, s"Unrecognized label ${fields(labelIndex)}. Expected one of ${labelOrder.mkString("[",",","]")}")
          l
        }
        val queryId = queryToId.getOrElseUpdate(fields(queryIndex), queryToId.size)
        LabeledQueryRow(queryId, features, label)
      }
      rows.sortBy(r => (r.queryId, labelOrder.indexOf(r.label)))
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