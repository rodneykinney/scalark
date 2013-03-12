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

trait QueryDocPair[LabelType] extends LabelInstance[LabelType] {
  val queryId:Int
}

case class ConcreteQueryDocPair[LabelType](
val rowId: Int, 
val label: LabelType, 
val weight: Double, 
val queryId:Int) 
extends QueryDocPair[LabelType]

