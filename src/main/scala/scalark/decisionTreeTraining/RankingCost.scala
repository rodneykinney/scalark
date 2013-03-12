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

class RankingCost extends CostFunction[Int] {


  def optimalConstant(labels: Seq[LabelInstance[Int]]) = {
  0.0
}

  def gradient(labels: Seq[LabelInstance[Int]], modelEval: Int => Double) = {
  Seq(0.0)
}

  def totalCost(ids: Seq[LabelInstance[Int]], modelEval: Int => Double) = {
    val queries = ids.asInstanceOf[Seq[QueryDocPair[Int]]]
      var cost = 0.0
      for (query <- groupBySorted(queries, splitQueries)) {
        val labelGroups = groupBySorted(query, splitLabels).toIndexedSeq
	for (label <- (0 until labelGroups.length)) {
	  for (betterLabel <- label + 1 until labelGroups.length) {
	    cost += (for (worse <- labelGroups(label);
	    better <- labelGroups(betterLabel)) yield {
	      math.log(1 + math.exp(modelEval(worse.rowId)-modelEval(better.rowId)))
	    }).sum
	  }
	}
      }
      cost
  }

  def optimalDelta(regions:Seq[Seq[LabelInstance[Int]]], modelEval:Function[Int,Double]) = {
  Seq(0.0)
  }

private val splitQueries = (l:Seq[QueryDocPair[Int]]) => l.indexWhere(_.queryId != l.head.queryId)
private val splitLabels = (l:Seq[QueryDocPair[Int]]) => l.indexWhere(_.label != l.head.label)

  def groupBySorted[T](elems:Seq[T],splitFunction:Seq[T] => Int) : List[Seq[T]] = {
    val splitIndex = splitFunction(elems)
    val (thisGroup,nextGroup) = elems.splitAt(splitIndex)
    thisGroup match {
    case Nil => List(nextGroup)
    case _ => thisGroup :: groupBySorted(nextGroup,splitFunction)
}
    
  }
}