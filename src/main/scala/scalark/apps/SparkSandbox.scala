package scalark.apps

import spark._
object SparkSandbox {
  def main(args:Array[String]) {
    def context = new SparkContext("local","sandbox")
    val data = (0 until 9999).toArray
    val parData = context.parallelize(data)
  }

}