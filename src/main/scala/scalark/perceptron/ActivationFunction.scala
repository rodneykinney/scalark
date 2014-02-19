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

/**
 * Activation function for computing node values
 */
trait ActivationFunction {
  def apply(x: Double): Double
  def derivative(x: Double): Double
}

object Logistic extends ActivationFunction {
  def apply(x: Double) = 1.0 / (1.0 + math.exp(-x))
  def derivative(x: Double) = x*(1-x)
}

object Identity extends ActivationFunction {
  def apply(x: Double) = x
  def derivative(x: Double) = 1
}