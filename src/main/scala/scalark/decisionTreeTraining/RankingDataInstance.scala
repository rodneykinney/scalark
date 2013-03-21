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

trait Query {
  val queryId: Int
}

case class ObservationLabelQuery[LabelType](val rowId:Int, val queryId:Int, label:LabelType) extends Observation with Label[LabelType] with Query
case class ObservationLabelRowQuery[LabelType](val rowId:Int, val queryId:Int, val features:IndexedSeq[Int], label:LabelType) extends Observation with Label[LabelType] with RowOfFeatures with Query 
case class ObservationLabelFeatureQuery[LabelType](val rowId:Int, val queryId:Int, label:LabelType, featureValue:Int) extends Observation with Label[LabelType] with Query with Feature
case class ObservationLabelFeatureQueryScore[LabelType](val rowId:Int, val queryId:Int, label:LabelType, featureValue:Int, var score:Double) extends Observation with Label[LabelType] with Query with Feature with Score
case class ObservationLabelQueryScore[LabelType](val rowId:Int, val queryId:Int, label:LabelType, var score:Double) extends Observation with Label[LabelType] with Query with Score
case class ObservationLabelQueryScoreRegion[LabelType](val rowId:Int, val queryId:Int, label:LabelType, var score:Double, var regionId:Int) extends Observation with Label[LabelType] with Query with Score with Region

