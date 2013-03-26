package scalark.decisionTreeTraining

trait Metric[L, T <: Label[L]] {
  type ResultType
  def compute[T1 <: T with Score](rows: Seq[T1]): ResultType
}

class BinaryAccuracy extends Metric[Boolean, Label[Boolean]] {
  type ResultType = Double
  def compute[T <: Label[Boolean] with Score](rows: Seq[T]) = {
    var rowCount, errorCount = 0
    for (row <- rows) {
      rowCount += 1
      if (row.label ^ row.score > 0) errorCount += 1
    }
    errorCount.toDouble / rowCount
  }
}

class PrecisionRecall extends Metric[Boolean, Label[Boolean]] {
  type ResultType = Tuple2[Double, Double]
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

object foo {
  val m = new BinaryAccuracy
  val mm:Metric[Boolean,Label[Boolean]] = m
  val value = mm.compute(Vector.empty[Label[Boolean] with Score])
  //val mm; Metric[Boolean, Label[Boolean]] = m
}