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
}

trait Weight {
  var weight: Double
}

trait Label[LabelType] {
  var label: LabelType
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

case class ObservationLabelFeatureWeightScore(val rowId: Int, var label: Double = 0.0, val featureValue: Int, var weight: Double = 1.0, var score: Double = 0.0) extends Observation with Feature with Weight with Score with Label[Double]
case class LabeledRow[LabelType](var label: LabelType, val features: IndexedSeq[Int]) extends Label[LabelType] with RowOfFeatures {
  def forTraining = new Label[LabelType] with Weight with Score with Region {
    var label = LabeledRow.this.label; 
    var weight = 1.0; 
    var score = 0.0; 
    var regionId = -1
    }
}
