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

/** Single data instance with a label */
trait LabelInstance[LabelType] {
  def label: LabelType
  def rowId: Int
  def weight: Double
}

object LabelInstance {
  def apply[LabelType](rowId: Int, label: LabelType, weight: Double) = new ConcreteLabelInstance(rowId, label, weight)
  def apply[LabelType](rowId: Int, label: LabelType) = new FixedWeightLabelInstance(rowId, label)
}

class ConcreteLabelInstance[LabelType](val rowId: Int, val label: LabelType, val weight: Double) extends LabelInstance[LabelType] {}

class FixedWeightLabelInstance[LabelType](val rowId: Int, val label: LabelType) extends LabelInstance[LabelType] {
  val weight = 1.0
}

/** Single data instance with one feature value */
trait FeatureInstance[LabelType] extends LabelInstance[LabelType] {
  def featureValue: Int
}

object FeatureInstance {
  def apply[LabelType](labelInstance: LabelInstance[LabelType], featureValue: Int) = new FeatureInstanceSharedLabel[LabelType](labelInstance, featureValue)
  def apply[LabelType](rowId: Int, label: LabelType, weight: Double, featureValue: Int) = new ConcreteFeatureInstance(rowId, label, weight, featureValue)
}

class FeatureInstanceSharedLabel[LabelType](private val labelInstance: LabelInstance[LabelType], val featureValue: Int) extends FeatureInstance[LabelType] {
  def rowId = labelInstance.rowId
  def label = labelInstance.label
  def weight = labelInstance.weight
}

case class ConcreteFeatureInstance[LabelType](val rowId: Int, val label: LabelType, val weight: Double, val featureValue: Int) extends FeatureInstance[LabelType] {}

/** Instance with two labels */
class FeatureInstanceDelegate[LabelType1, LabelType2](val _2:FeatureInstance[LabelType2], val label:LabelType1) extends FeatureInstance[LabelType1] {
  def rowId = _2.rowId
  def weight = _2.weight
  def featureValue = _2.featureValue
  override def toString = "FeatureInstanceDelegate("+rowId+","+label+","+weight+","+featureValue+","+_2.label+")"
}

trait FeatureRow {
  val rowId: Int
  val weight: Double
  val features: IndexedSeq[Int]
}

class LabeledFeatureRow[LabelType](val rowId: Int, val features: IndexedSeq[Int], val weight: Double, val label: LabelType)
  extends FeatureRow with LabelInstance[LabelType] {

  override def toString = "Row[id=" + rowId + ", label=" + label + "]"
}