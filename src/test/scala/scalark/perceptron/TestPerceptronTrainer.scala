/*
Copyright 2014 Rodney Kinney

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
package scalark.perceptron

import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.FunSuite
import org.scalatest.matchers.ShouldMatchers

@RunWith(classOf[JUnitRunner])
class TestPerceptronTrainer extends FunSuite with ShouldMatchers {
  test("One step logistic regression") {
    // Build network with no hidden layers
    val net = new Network(IndexedSeq(new DenseConnection(3, 1, new Array[Double](3), Identity)))

    // Example with label=true
    val xPos = (new DenseLayer(Array(1.0, 0.5, 2.0)), new DenseLayer(Array(1.0)))
    val nextNetPos = net.trainStep(List(xPos), 1.0, SoftMaxCrossEntropy)
    nextNetPos.connections.head.asInstanceOf[DenseConnection].weights.toArray should be(Array(0.5, 0.25, 1.0))

    // Example with label=false
    val xNeg = (new DenseLayer(Array(1.0, 0.5, 2.0)), new DenseLayer(Array(0.0)))
    val nextNetNeg = net.trainStep(List(xNeg), 1.0, SoftMaxCrossEntropy)
    nextNetNeg.connections.head.asInstanceOf[DenseConnection].weights.toArray should be(Array(-0.5, -0.25, -1.0))
  }
}