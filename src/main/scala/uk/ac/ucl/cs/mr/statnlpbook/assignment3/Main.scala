package uk.ac.ucl.cs.mr.statnlpbook.assignment3

import breeze.stats.distributions.Uniform

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
  val probability = 0.6


  val trainSetName = "train"
  val validationSetName = "dev"
//  var model: Model = new SumOfWordVectorsModel(wordDim, vectorRegularizationStrength)
  //var model: Model = new SumOfWordVectorsModelWithDropout(wordDim, vectorRegularizationStrength, probability)
//  var model: Model = new SumOfWordVectorsModelWithTrainedVectors(9, vectorRegularizationStrength)
  //var model: Model = new ProductOfWordVectorsModel(wordDim, vectorRegularizationStrength)
  //val model: Model = new RecurrentNeuralNetworkModel(wordDim, hiddenDim, vectorRegularizationStrength, matrixRegularizationStrength)


  def epochHook(iter: Int, accLoss: Double): Unit = {
    // The commented code below is needed for to calculate the accuracy on the training and development set for the
    // Dropout model
    /*val weights = model.vectorParams("param_w")
    val adjustedWeights = weights.copy()
    adjustedWeights.param :* probability
    val testModel = new SumOfWordVectorsModel(wordDim, vectorRegularizationStrength)
    testModel.vectorParams ++= model.vectorParams
    testModel.vectorParams += "param_w" -> adjustedWeights
    model.vectorParams += "param_w" -> weights*/
    println("Epoch %4d\tLoss %8.4f\tTrain Acc %4.2f\tDev Acc %4.2f".format(
      iter, accLoss, 100 * Evaluator(model, trainSetName), 100*Evaluator(model, validationSetName)))
  }


//  StochasticGradientDescentLearner(model, trainSetName, 10, learningRate, epochHook)


  // Hyoer-parameter searching

  // Uncomment first for SumOfWordVectors____ models, second for RNN and LSTM
//  val accuracyMatrix:mutable.HashMap[(Int,Double,Double), Double] = new mutable.HashMap[(Int,Double,Double), Double]()
  val accuracyMatrix:mutable.HashMap[(Int, Int, Double, Double, Double), Double] =
    new mutable.HashMap[(Int, Int, Double, Double,Double), Double]()

//  val range = List(0.005,0.01, 0.05)
//  for (i <- 7 to 10 by 1){
//    for (j <- range)
//    {
//      model = new SumOfWordVectorsModel(i, j)
//      for (k <- range)
//      {
//        StochasticGradientDescentLearner(model, trainSetName, 50, k, epochHook)
//        accuracyMatrix += (i,j,k) -> 100*Evaluator(model,validationSetName)
//        model.vectorParams.clear()
//        model.vectorParams += "param_w" -> VectorParam(i)
//      }
//    }
//    println(i)
//  }
//  println(accuracyMatrix.maxBy(_._2))
//
//  println(accuracyMatrix)

  // CODE FOR GRID SEARCH ON SUM OF WORDS WITH DROPOUT MODEL
//  val range = List(0.005,0.01, 0.05)
//  for (i <- 7 to 10 by 1){
//    for (j <- range)
//    {
//      model = new SumOfWordVectorsModelWithDropout(i, j, probability)
//      for (k <- range)
//      {
//        StochasticGradientDescentLearner(model, trainSetName, 50, k, epochHook)
//
//        val weights = model.vectorParams("param_w")
//        weights.param :* probability
//        val testModel = new SumOfWordVectorsModel(i, j)
//        testModel.vectorParams ++= model.vectorParams
//        testModel.vectorParams += "param_w" -> weights
//
//        accuracyMatrix += (i,j,k) -> 100*Evaluator(model,validationSetName)
//        model.vectorParams.clear()
//        model.vectorParams += "param_w" -> VectorParam(i)
//      }
//    }
//    println(i)
//  }
//  println(accuracyMatrix.maxBy(_._2))
//
//  println(accuracyMatrix)

//   CODE FOR GRID SEARCH FOR RNN/LSTM
//    val range = List(0.005,0.01, 0.05)
//    for (i <- 7 to 10 by 1) {
//      for (j <- 7 to 10 by 1) {
//        for (k <- range) {
//          for (l <- range) {
//            model = new RecurrentNeuralNetworkModel(i, j, k, l)
//            for (m <- range) {
//              StochasticGradientDescentLearner(model, trainSetName, 50, m, epochHook)
//              accuracyMatrix += (i, j, k, l, m) -> 100 * Evaluator(model, validationSetName)
//              model.vectorParams.clear()
//              model.vectorParams += "param_w" -> VectorParam(i)
//            }
//          }
//        }
//        println(i)
//      }
//    }
//    println(accuracyMatrix.maxBy(_._2))
//    println(accuracyMatrix)


//  RANDOM GRID SEARCH CODE FOR RNN/LSTM
  val embeddingDist = new Uniform(4,11) // embedding size distribution
  val hiddenDist = new Uniform(6,13) // hidden size distribution
  val vectorRegDist = new Uniform(-5,-3) // vector regularisation strength distribution
  val matrixRegDist = new Uniform(-5,-3) // matrix regularisation strength distribution
  val learningRateDist = new Uniform(-5,-3) // learning rate distribution
  var bestConfig = (0.0,0.0,0.0,0.0,0.0,0.0)

  try {
    while (true) {
      val i = embeddingDist.sample().floor.toInt
      val j = hiddenDist.sample().floor.toInt
      val k = /*0.0001*/math.pow(10.0,vectorRegDist.sample())
      val l = /*0.0001*/math.pow(10.0,matrixRegDist.sample())
      val m = 0.002//math.pow(10.0,learningRateDist.sample())
//      val model = new RecurrentNeuralNetworkModel(i, j, k, l)
      println("Current: " + (i,j,k,l,m))
      StochasticGradientDescentLearner(model, trainSetName, 20, m, epochHook)
      val accuracy = 100.0 * Evaluator(model, validationSetName)
      val trainAccuracy = 100.0 * Evaluator(model, trainSetName)
      accuracyMatrix += (i, j, k, l, m) -> accuracy
      if (accuracy > bestConfig._6)
        bestConfig = (i,j,k,l,m,accuracy)
      println("Current: " + (i,j,k,l,m,accuracy) + ", Best: " + bestConfig)
      println("Train set accuracy: " + trainAccuracy)
      model.vectorParams.clear()
      model.matrixParams.clear()
    }
  }
//  catch{
//    case e:Exception => {
//      println(accuracyMatrix.maxBy(_._2))
//      println(accuracyMatrix)
//    }
//  }



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