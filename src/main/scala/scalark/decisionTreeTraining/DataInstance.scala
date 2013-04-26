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

trait Observation {
  def rowId: Int
  var weight: Double
}

trait Label[LabelType] {
  def label: LabelType
}

trait Feature {
  def featureValue: Int
}

trait RowOfFeatures {
  val features: IndexedSeq[Int]
}

trait Score {
  var score: Double
}

trait Region {
  var regionId: Int
}

case class ObservationLabel[LabelType](val rowId: Int, var weight: Double, val label: LabelType) extends Observation with Label[LabelType]
case class ObservationLabelScore[LabelType](val rowId: Int, var weight: Double, val label: LabelType, var score: Double) extends Observation with Label[LabelType] with Score
case class ObservationLabelScoreRegion[LabelType](val rowId: Int, var weight: Double, val label: LabelType, var score: Double, var regionId: Int) extends Observation with Label[LabelType] with Score with Region
case class ObservationLabelFeature[LabelType](val rowId: Int, var weight: Double, val label: LabelType, val featureValue: Int) extends Observation with Label[LabelType] with Feature
case class ObservationLabelFeatureScore[LabelType](val rowId: Int, var weight: Double, val label: LabelType, val featureValue: Int, var score: Double) extends Observation with Label[LabelType] with Feature with Score
case class ObservationLabelFeatureScoreRegion[LabelType](val rowId: Int, var weight: Double, val label: LabelType, val featureValue: Int, var score: Double, var regionId: Int) extends Observation with Label[LabelType] with Feature with Score with Region
case class ObservationRowLabel[LabelType](val rowId: Int, var weight: Double, val features: IndexedSeq[Int], val label: LabelType) extends Observation with RowOfFeatures with Label[LabelType]
case class ObservationRowLabelScore[LabelType](val rowId: Int, var weight: Double, val features: IndexedSeq[Int], val label: LabelType, var score: Double) extends Observation with RowOfFeatures with Label[LabelType] with Score
