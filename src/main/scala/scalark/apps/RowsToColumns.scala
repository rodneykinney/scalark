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

import java.io._
import scalark.decisionTreeTraining._
import scala.collection._

/**
 * Converts a TSV file of feature rows (where each line is a single training example)
 * into a TSV file of columns (where each line is a column, containing feature value and row number, sorted by feature value)
 */
object RowsToColumns {
  def main(args: Array[String]) {
    val config = new RowsToColumnsConfig
    if (!config.parse(args)) {
      System.exit(0)
    }
    this(config.input, config.output, config.output + ".labels.tsv")
  }
  def apply(input: String, outputColumns: String, outputLabels: String) = {
    val columnFiles = partitionByColumn(new File(input).readRows, new PrintWriter(new File(outputLabels)))
    val writer = new PrintWriter(new File(outputColumns))
    for ((f, col) <- columnFiles.zipWithIndex) {
      val featureValues = io.Source.fromFile(f).getLines.map(_.toInt).toIndexedSeq.zipWithIndex.sortBy(_._1)
      writer.println(col + "\t" + featureValues.map { case (x, i) => i + "," + x }.mkString("\t"))
    }
    writer.close
  }
  private def partitionByColumn(rows: Iterator[LabeledRow[Boolean]], labelWriter: PrintWriter) = {
    val tmpDir = createTempDirectory
    val firstRow = rows.next
    val columnCount = firstRow.features.size
    val columnFiles = for (i <- 0 until columnCount) yield {
      val f = new File(tmpDir, "column" + i)
      f.deleteOnExit()
      f
    }
    val columnWriters = columnFiles.map(new PrintWriter(_))
    def processRow[L] =
      (row: RowOfFeatures with Label[L]) => {
        labelWriter.println(row.label)
        for ((f, i) <- row.features.zipWithIndex) {
          columnWriters(i).println(f)
        }
      }
    processRow(firstRow)
    for (row <- rows) {
      processRow(row)
    }
    labelWriter.close
    for (w <- columnWriters) w.close
    columnFiles
  }
  def createTempDirectory = {
    var count = 0
    var dir = new File(System.getProperty("java.io.tmpdir"), "columns-" + count)
    while (dir.exists) {
      count += 1
      dir = new File(System.getProperty("java.io.tmpdir"), "columns-" + count)
    }
    if (!dir.mkdirs())
      throw new IOException("Could not create directory " + dir)
    dir.deleteOnExit()
    dir
  }
}

class RowsToColumnsConfig extends CommandLineParameters {
  var input: String = _
  var output: String = _

  def usage = {
    required("input", "Input TSV file") ::
      required("output", "Output TSV file") ::
      Nil
  }
}