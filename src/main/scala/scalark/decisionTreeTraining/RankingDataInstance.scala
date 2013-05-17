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
  def queryId: Int
}

case class LabeledQueryRow[LabelType](val queryId: Int, val features: IndexedSeq[Double], var label: LabelType) extends Label[LabelType] with RowOfFeatures with Query {
  def asTrainable = new Query with Label[LabelType] with MutableWeight with MutableScore with MutableRegion {
    val queryId = LabeledQueryRow.this.queryId;
    val label = LabeledQueryRow.this.label;
    var weight = 1.0;
    var score = 0.0;
    var regionId = -1
  }
}

