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
    def toSortedColumns[LabelType, ColumnType <: Observation](implicit featureSelector: RowType => SelectSingleFeature[LabelType, ColumnType with Feature with Label[LabelType]]) = {
      for (col <- (0 until rows.head.features.length)) yield {
        val data = rows.view.sortBy(_.features(col)).map(_.selectSingleFeature(col)).force
        new FeatureColumn[LabelType, ColumnType with Feature with Label[LabelType]](data, col)
      }
    }
  }

  trait SelectSingleFeature[L, T] {
    def selectSingleFeature(col: Int): T
  }

  implicit def singleFeatureConvert[L](row: Observation with RowOfFeatures with Label[L]) = new SelectSingleFeature[L, Observation with Feature with Label[L]] {
    def selectSingleFeature(col: Int) = new ObservationLabelFeature(rowId = row.rowId, weight = row.weight, featureValue = row.features(col), label = row.label)
  }
  implicit def singleFeatureConvertScore[L](row: Observation with RowOfFeatures with Score with Label[L]) = new SelectSingleFeature[L, Observation with Feature with Score with Label[L]] {
    def selectSingleFeature(col: Int) = new ObservationLabelFeatureScore(rowId = row.rowId, weight = row.weight, featureValue = row.features(col), label = row.label, score = row.score)
  }
  implicit def singleFeatureConvertQuery[L](row: Observation with RowOfFeatures with Query with Label[L]) = new SelectSingleFeature[L, Observation with Feature with Query with Label[L]] {
    def selectSingleFeature(col: Int) = new ObservationLabelFeatureQuery(rowId = row.rowId, weight = row.weight, featureValue = row.features(col), label = row.label, queryId = row.queryId)
  }
  implicit def singleFeatureConvertQueryScore[L](row: Observation with RowOfFeatures with Score with Query with Label[L]) = new SelectSingleFeature[L, Observation with Feature with Score with Query with Label[L]] {
    def selectSingleFeature(col: Int) = new ObservationLabelFeatureQueryScore(rowId = row.rowId, weight = row.weight, featureValue = row.features(col), label = row.label, queryId = row.queryId, score = row.score)
  }

  implicit def ValidateRowIds(data: Seq[Observation]) = new {
    def validate = data.head.rowId == 0 && data.zip(data.drop(1)).forall(t => t._2.rowId == t._1.rowId + 1)
  }
  implicit def ValidateRowAndQueryIds(data: Seq[Observation with Query with Label[Int]]) = new {
    def validate = data.head.rowId == 0 &&
      data.zip(data.drop(1)).forall(t => {
        val (row, nextRow) = t
        nextRow.rowId == row.rowId + 1 &&
          (nextRow.queryId == row.queryId + 1 ||
            nextRow.label >= row.label)
      })
  }

  trait DecorateWithScore[T] {
    def withScore(score: Double): T with Score
  }

  implicit def decorateObservationWithScore[L](row: Observation with Label[L]) = new DecorateWithScore[Observation with Label[L]] {
    def withScore(score: Double) = ObservationLabelScore(rowId = row.rowId, weight = row.weight, label = row.label, score = score)
  }

  implicit def decorateRowWithScore[L](row: Observation with RowOfFeatures with Label[L]) = new DecorateWithScore[Observation with RowOfFeatures with Label[L]] {
    def withScore(score: Double) = ObservationRowLabelScore(rowId = row.rowId, weight = row.weight, label = row.label, score = score, features = row.features)
  }

  implicit def decorateQueryWithScore[L](row: Observation with Query with Label[L]) = new DecorateWithScore[Observation with Query with Label[L]] {
    def withScore(score: Double) = ObservationLabelQueryScore(rowId = row.rowId, weight = row.weight, label = row.label, score = score, queryId = row.queryId)
  }

  trait DecorateWithScoreAndRegion[T] {
    def withScoreAndRegion(score: Double, regionId: Int): T with Score with Region
  }

  implicit def decorateObservationWithScoreAndRegion[L](row: Observation with Label[L]) = new DecorateWithScoreAndRegion[Observation with Label[L]] {
    def withScoreAndRegion(score: Double, regionId: Int) = ObservationLabelScoreRegion(rowId = row.rowId, weight = row.weight, label = row.label, score = score, regionId = regionId)
  }

  implicit def decorateQueryWithScoreAndRegion[L](row: Observation with Query with Label[L]) = new DecorateWithScoreAndRegion[Observation with Query with Label[L]] {
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
      val dropColumn = Set(columnNames.zipWithIndex.filter(t => t._1.startsWith("#")).map(_._2): _*)
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