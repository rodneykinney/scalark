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
        rows.zipWithIndex.view.sortBy(_._1.features(col)).map { case (r, i) => new ObservationFeature(i, r.features(col)) }.force.toIndexedSeq
      }
    }
  }

  implicit def addWeightAndLabel(columns: immutable.Seq[Seq[Observation with Feature]]) = new {
    def toFeatureColumns[L](weightFinder: Int => Double, labelFinder: Int => L) = {
      for ((col, columnId) <- columns.zipWithIndex) yield {
        val instances = mutable.ArraySeq.empty[Observation with Feature with Weight with Label[L]] ++
          (for (c <- col) yield new Observation with Feature with Weight with Label[L] {
            def rowId = c.rowId
            def featureValue = c.featureValue
            def weight = weightFinder(c.rowId)
            def weight_=(value:Double) = ()
            def label = labelFinder(c.rowId)
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

  trait DecorateWithScore[T] {
    def withScore(score: Double): T with Score
  }

  implicit def decorateObservationWithScore[L](row: Observation with Weight with Label[L]) = new DecorateWithScore[Observation with Label[L]] {
    def withScore(score: Double) = ObservationLabelScore(rowId = row.rowId, weight = row.weight, label = row.label, score = score)
  }

  implicit def decorateRowWithScore[L](row: Observation with Weight with RowOfFeatures with Label[L]) = new DecorateWithScore[Observation with RowOfFeatures with Label[L]] {
    def withScore(score: Double) = ObservationRowLabelScore(rowId = row.rowId, weight = row.weight, label = row.label, score = score, features = row.features)
  }
  implicit def decorateQueryWithScore[L](row: Observation with Weight with Query with Label[L]) = new DecorateWithScore[Observation with Query with Label[L]] {
    def withScore(score: Double) = ObservationLabelQueryScore(rowId = row.rowId, weight = row.weight, label = row.label, score = score, queryId = row.queryId)
  }

  trait DecorateWithScoreAndRegion[T] {
    def withScoreAndRegion(score: Double, regionId: Int): T with Score with Region
  }

  implicit def decorateObservationWithScoreAndRegion[L](row: Observation with Weight with Label[L]) = new DecorateWithScoreAndRegion[Observation with Label[L]] {
    def withScoreAndRegion(score: Double, regionId: Int) = ObservationLabelScoreRegion(rowId = row.rowId, weight = row.weight, label = row.label, score = score, regionId = regionId)
  }

  implicit def decorateQueryWithScoreAndRegion[L](row: Observation with Weight with Query with Label[L]) = new DecorateWithScoreAndRegion[Observation with Query with Label[L]] {
    def withScoreAndRegion(score: Double, regionId: Int) = ObservationLabelQueryScoreRegion(rowId = row.rowId, weight = row.weight, label = row.label, score = score, regionId = regionId, queryId = row.queryId)
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
        ObservationRowLabel(rowId = rowId, weight = 1.0, label = fields(labelIndex).toBoolean, features = features)
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