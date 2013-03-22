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
        val data = rows.sortBy(_.features(col)).map(_.selectSingleFeature(col))
        new FeatureColumn[LabelType, ColumnType with Feature with Label[LabelType]](data, col)
      }
    }
  }

  trait SelectSingleFeature[L, T] {
    def selectSingleFeature(col: Int): T
  }

  implicit def singleFeatureConvert[L](row: Observation with RowOfFeatures with Label[L]) = new SelectSingleFeature[L, Observation with Feature with Label[L]] {
    def selectSingleFeature(col: Int) = new ObservationLabelFeature(rowId = row.rowId, featureValue = row.features(col), label = row.label)
  }
  implicit def singleFeatureConvertScore[L](row: Observation with RowOfFeatures with Score with Label[L]) = new SelectSingleFeature[L, Observation with Feature with Score with Label[L]] {
    def selectSingleFeature(col: Int) = new ObservationLabelFeatureScore(rowId = row.rowId, featureValue = row.features(col), label = row.label, score = row.score)
  }
  implicit def singleFeatureConvertQuery[L](row: Observation with RowOfFeatures with Query with Label[L]) = new SelectSingleFeature[L, Observation with Feature with Query with Label[L]] {
    def selectSingleFeature(col: Int) = new ObservationLabelFeatureQuery(rowId = row.rowId, featureValue = row.features(col), label = row.label, queryId = row.queryId)
  }
  implicit def singleFeatureConvertQueryScore[L](row: Observation with RowOfFeatures with Score with Query with Label[L]) = new SelectSingleFeature[L, Observation with Feature with Score with Query with Label[L]] {
    def selectSingleFeature(col: Int) = new ObservationLabelFeatureQueryScore(rowId = row.rowId, featureValue = row.features(col), label = row.label, queryId = row.queryId, score = row.score)
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

  trait DecorateWithScoreAndRegion[T] {
    def withScoreAndRegion(score: Double, regionId: Int): T with Score with Region
  }

  implicit def decorateObservation[L](row: Observation with Label[L]) = new DecorateWithScoreAndRegion[Observation with Label[L]] {
    def withScoreAndRegion(score: Double, regionId: Int) = ObservationLabelScoreRegion(rowId = row.rowId, label = row.label, score = score, regionId = regionId)
  }

  implicit def decorateObservationWithQuery[L](row: Observation with Query with Label[L]) = new DecorateWithScoreAndRegion[Observation with Query with Label[L]] {
    def withScoreAndRegion(score: Double, regionId: Int) = ObservationLabelQueryScoreRegion(rowId = row.rowId, label = row.label, score = score, regionId = regionId, queryId = row.queryId)
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