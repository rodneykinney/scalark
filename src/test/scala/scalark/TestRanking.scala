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

import scalark.decisionTreeTraining._
import org.scalatest._
import scala.util._

class TestRanking extends FunSuite {
  test("TotalCost") {
val c = new RankingCost()

val rows = List(new ConcreteQueryDocPair[Int](rowId=0,label=0,weight=0,queryId=0),
new ConcreteQueryDocPair[Int](rowId=1,label=0,weight=0,queryId=0),
new ConcreteQueryDocPair[Int](rowId=2,label=1,weight=0,queryId=0),
new ConcreteQueryDocPair[Int](rowId=3,label=0,weight=0,queryId=1),
new ConcreteQueryDocPair[Int](rowId=4,label=1,weight=0,queryId=1))

    assert(math.abs(c.totalCost(rows, id => rows(id).label) - 3 * math.log(1 + math.exp(-1))) < 1.0e-9)
  }
}
