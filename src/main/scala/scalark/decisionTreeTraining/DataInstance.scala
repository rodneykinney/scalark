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

trait Label[LabelType] {
  def label: LabelType
}

trait Weight {
  def weight: Double
}

trait Feature {
  def featureValue: Int
}

trait RowOfFeatures {
  val features: IndexedSeq[Int]
}

trait LabelInstance[LabelType] extends Observation with Label[LabelType]
trait WeightedLabelInstance[LabelType] extends Observation with Label[LabelType] with Weight

trait FeatureInstance[LabelType] extends LabelInstance[LabelType] with Feature
trait WeightedFeatureInstance[LabelType] extends FeatureInstance[LabelType] with Weight

trait FeatureRow extends Observation with RowOfFeatures
trait LabeledFeatureRow[LabelType] extends LabelInstance[LabelType] with RowOfFeatures 

object Instance {
  def apply(id: Int) = new Observation { val rowId = id }
  def apply[LabelType](id: Int, l: LabelType) = new LabelInstance[LabelType] { val rowId = id; val label = l }
  def apply[LabelType](id: Int, v: Int, l: LabelType) = new FeatureInstance[LabelType] { val rowId = id; val featureValue = v; val label = l }
}

object Row {
  def apply(id: Int, v: IndexedSeq[Int]) = new FeatureRow { val rowId = id; val features = v }
  def apply[LabelType](id: Int, v: IndexedSeq[Int], l: LabelType) = new LabeledFeatureRow[LabelType] { val rowId = id; val features = v; val label = l }
}