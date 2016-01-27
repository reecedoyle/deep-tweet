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
  def epochHook(iter: Int, accLoss: Double): Unit = {
//    println("Epoch %4d\tLoss %8.4f\tTrain Acc %4.2f\tDev Acc %4.2f".format(
//      iter, accLoss, 100 * Evaluator(model, trainSetName), 100*Evaluator(model, validationSetName)))
  }



  val learningRate = 0.01
  val vectorRegularizationStrength = 0.01
  val matrixRegularizationStrength = 0.0
  val wordDim = 10
  val hiddenDim = 10


  val trainSetName = "train"
  val validationSetName = "dev"
   //var model: Model = new SumOfWordVectorsModel(wordDim, vectorRegularizationStrength)
  var model: Model = new RecurrentNeuralNetworkModel(wordDim, hiddenDim, vectorRegularizationStrength, matrixRegularizationStrength)

  val accuracyMatrix:mutable.HashMap[(Int,Int,Double,Double,Double), Double] = new mutable.HashMap[(Int,Int,Double,Double,Double), Double]()

  val range = List(0.005,0.01, 0.05)

  for (i <- 7 to 10 by 1) {
    for (j <- 7 to 10 by 1) {
      for (k <- range) {
        for (l <- range) {
          model = new RecurrentNeuralNetworkModel(i, j, k, l)
          for (m <- range) {
            StochasticGradientDescentLearner(model, trainSetName, 50, m, epochHook)
            accuracyMatrix += (i, j, k, l,m) -> 100 * Evaluator(model, validationSetName)
            model.vectorParams.clear()
            model.vectorParams += "param_w" -> VectorParam(i)
          }
        }
      }
      println(i)
    }
  }
  println(accuracyMatrix.maxBy(_._2))

  println(accuracyMatrix)




  /*  RELEVANT TO SUM OF WORDS GRID SEARCH
  val accuracyMatrix:mutable.HashMap[(Int,Double,Double), Double] = new mutable.HashMap[(Int,Double,Double), Double]()

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
  println(accuracyMatrix.maxBy(_._2))

  //println(accuracyMatrix)

*/





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