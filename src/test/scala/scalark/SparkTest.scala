package scalark

import org.scalatest._
import spark._

object SparkTest extends org.scalatest.Tag("scalark.SparkTest")

trait SparkTestUtils extends FunSuite {
  var sc: SparkContext = _

  /**
   * Convenience method for tests that use spark.  Creates a local spark context, and cleans
   * it up even if your test fails.  Also marks the test with the tag SparkTest, so you can
   * turn it off
   *
   * @param name the name of the test
   */
  def sparkTest(name: String)(body: => Unit) {
    test(name, SparkTest){
      sc = new SparkContext("local[4]", name)
      try {
        body
      }
      finally {
        sc.stop
        sc = null
        // To avoid Akka rebinding to the same port, since it doesn't unbind immediately on shutdown
        System.clearProperty("spark.master.port")
      }
    }
  }
}
