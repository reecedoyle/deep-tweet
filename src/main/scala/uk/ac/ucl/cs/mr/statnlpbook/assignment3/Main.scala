package uk.ac.ucl.cs.mr.statnlpbook.assignment3

import scala.collection.mutable

/**
 * @author rockt
 */
object Main extends App {
  /**
   * Example training of a model
   *
   * Problems 2/3/4: perform a grid search over the parameters below
   */

  val learningRate = 0.01
  val vectorRegularizationStrength = 0.01
  val matrixRegularizationStrength = 0.0
  val wordDim = 10
  val hiddenDim = 10


  val trainSetName = "train"
  val validationSetName = "dev"
  //var model: Model = new SumOfWordVectorsModel(wordDim, vectorRegularizationStrength)
  var model: Model = new SumOfWordVectorsModelWithDropout(wordDim, vectorRegularizationStrength, 0.75)
  //val model: Model = new RecurrentNeuralNetworkModel(wordDim, hiddenDim, vectorRegularizationStrength, matrixRegularizationStrength)

  def epochHook(iter: Int, accLoss: Double): Unit = {
    val weights = model.vectorParams("param_w")
    val testModel = new SumOfWordVectorsModel(wordDim, vectorRegularizationStrength)
    testModel.vectorParams ++= model.vectorParams
    testModel.vectorParams += "param_w" -> weights
    model.vectorParams += "param_w" -> weights
    println("Epoch %4d\tLoss %8.4f\tTrain Acc %4.2f\tDev Acc %4.2f".format(
      iter, accLoss, 100 * Evaluator(model, trainSetName), 100*Evaluator(testModel, validationSetName)))
  }


  StochasticGradientDescentLearner(model, trainSetName, 100, learningRate, epochHook)


 /*val accuracyMatrix:mutable.HashMap[(Int,Double,Double), Double] = new mutable.HashMap[(Int,Double,Double), Double]()

  val range = List(0.005,0.01, 0.05)

  for (i <- 7 to 10 by 1){
    for (j <- range)
    {
      model = new SumOfWordVectorsModel(i, j)
      for (k <- range)
      {
        StochasticGradientDescentLearner(model, trainSetName, 50, k, epochHook)
        accuracyMatrix += (i,j,k) -> 100*Evaluator(model,validationSetName)
        model.vectorParams.clear()
        model.vectorParams += "param_w" -> VectorParam(i)
      }
    }
    println(i)
  }
  println(accuracyMatrix.maxBy(_._2))*/

  //println(accuracyMatrix)







  /**
   * Comment this in if you want to look at trained parameters
   */
  /*
  for ((paramName, paramBlock) <- model.vectorParams) {
    println(s"$paramName:\n${paramBlock.param}\n")
  }
  for ((paramName, paramBlock) <- model.matrixParams) {
    println(s"$paramName:\n${paramBlock.param}\n")
  }
  */
}