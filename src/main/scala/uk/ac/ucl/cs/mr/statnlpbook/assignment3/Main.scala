package uk.ac.ucl.cs.mr.statnlpbook.assignment3

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
  
  val model: Model = new SumOfWordVectorsModel(wordDim, vectorRegularizationStrength)
  //val model: Model = new RecurrentNeuralNetworkModel(wordDim, hiddenDim, vectorRegularizationStrength, matrixRegularizationStrength)

  def epochHook(iter: Int, accLoss: Double): Unit = {
    println("Epoch %4d\tLoss %8.4f\tTrain Acc %4.2f\tDev Acc %4.2f".format(
      iter, accLoss, 100 * Evaluator(model, trainSetName), 100*Evaluator(model, validationSetName)))
  }

  StochasticGradientDescentLearner(model, trainSetName, 100, learningRate, epochHook)



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