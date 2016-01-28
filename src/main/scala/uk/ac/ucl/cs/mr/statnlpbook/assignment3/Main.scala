package uk.ac.ucl.cs.mr.statnlpbook.assignment3

import java.io.{FileOutputStream, File, PrintWriter}

/**
 * @author rockt
 */
object Main extends App {
  /**
   * Example training of a model
   *
   * Problems 2/3/4: perform a grid search over the parameters below
   */
  val learningRate = 0.05
  val vectorRegularizationStrength = 0.05
  val matrixRegularizationStrength = 0.0
  val wordDim = 9
  val hiddenDim = 10

  val trainSetName = "train"
  val validationSetName = "dev"
  
  val model: Model = new SumOfWordVectorsModel(wordDim, vectorRegularizationStrength)
  //val model: Model = new RecurrentNeuralNetworkModel(wordDim, hiddenDim, vectorRegularizationStrength, matrixRegularizationStrength)

  // Put in word2vec representations for vectors into hashmap


  def epochHook(iter: Int, accLoss: Double): Unit = {
    println("Epoch %4d\tLoss %8.4f\tTrain Acc %4.2f\tDev Acc %4.2f".format(
      iter, accLoss, 100 * Evaluator(model, trainSetName), 100*Evaluator(model, validationSetName)))
  }

//  StochasticGradientDescentLearner(model, trainSetName, 100, learningRate, epochHook)
  StochasticGradientDescentLearner(model, trainSetName, 100, learningRate, epochHook)

  // Output words, vector representations of words and weights in separate text files
//  model.vectorParams.remove("param_w") // This is not a word

//  // Vector representations
//  val vectorWriter = new PrintWriter(new FileOutputStream("vectorOutput.txt", false))
//  val vectorStrings = model.vectorParams.map(e => e._2).map(f => f.forward().toArray.map(g => g.toString).mkString(", "))
//  vectorStrings.foreach(e => vectorWriter.append(e + "\n"))
//  vectorWriter.close()
//
//  // Scores
//  val scoreWriter = new PrintWriter(new FileOutputStream("scoreOutput.txt", false))
//  val scoreStrings = model.vectorParams.map(e => e._2).map(f => model.scoreSentence(f).forward().toString)
//  scoreStrings.foreach(e => scoreWriter.append(e + "\n"))
//  scoreWriter.close()
//
//  val tokenWriter = new PrintWriter(new FileOutputStream("tokenOutput.txt", false))
//  val tokenStrings = model.vectorParams.map(e => e._1)
//  tokenStrings.foreach(e => tokenWriter.append(e + "\n"))
//  tokenWriter.close()
//
//  model.vectorParams.get("")

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