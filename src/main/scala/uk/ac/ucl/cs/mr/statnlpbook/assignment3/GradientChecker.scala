package uk.ac.ucl.cs.mr.statnlpbook.assignment3

import breeze.linalg.{DenseVector, QuasiTensor, TensorLike, sum}
import breeze.numerics._

/**
 * Problem 1
 */
object GradientChecker extends App {
  val EPSILON = 1e-6

  /**
   * For an introduction see http://cs231n.github.io/neural-networks-3/#gradcheck
   *
   * This is a basic implementation of gradient checking.
   * It is restricted in that it assumes that the function to test evaluates to a double.
   * Moreover, another constraint is that it always tests by backpropagating a gradient of 1.0.
   */
  def apply[P](model: Block[Double], paramBlock: ParamBlock[P]) = {
    paramBlock.resetGradient()
    model.forward()
    model.backward(1.0)

    var avgError = 0.0

    val gradient = paramBlock.gradParam match {
      case m: Matrix => m.toDenseVector
      case v: Vector => v
    }

    /**
     * Calculates f_theta(x_i + eps)
     * @param index i in x_i
     * @param eps value that is added to x_i
     * @return
     */
    def wiggledForward(index: Int, eps: Double): Double = {
      var result = 0.0
      paramBlock.param match {
        case v: Vector =>
          val tmp = v(index)
          v(index) = tmp + eps
          result = model.forward()
          v(index) = tmp
        case m: Matrix =>
          val (row, col) = m.rowColumnFromLinearIndex(index)
          val tmp = m(row, col)
          m(row, col) = tmp + eps
          result = model.forward()
          m(row, col) = tmp
      }
      result
    }

    for (i <- 0 until gradient.activeSize) {
      //todo: your code goes here!
      val gradientExpected: Double = (wiggledForward(i,EPSILON) - wiggledForward(i,-1*EPSILON))/(2*EPSILON)

      avgError = avgError + math.abs(gradientExpected - gradient(i))

      assert(
        math.abs(gradientExpected - gradient(i)) < EPSILON,
        "Gradient check failed!\n" +
          s"Expected gradient for ${i}th component in input is $gradientExpected but I got ${gradient(i)}"
      )
    }

    println("Average error: " + avgError)
  }

  /**
    * A very silly block to test if gradient checking is working.
    * Will only work if the implementation of the Dot block is already correct
    */

  // Checking blocks
  val a = vec(-1.5, 1.0, 1.5, 0.5)
  val b = VectorParam(4)
  b.set(vec(1.0, 2.0, -0.5, 2.5))
  val dotBlock = Dot(a, b)
  GradientChecker(dotBlock, b)
  val sigmoidBlock = Sigmoid(dotBlock)
  val c = VectorParam(4)
  c.set(vec(2.0, 4.0, 5.5, -3.0))
  val sumBlock = Sum(Seq(b,c))
  val negLoss = NegativeLogLikelihoodLoss(sigmoidBlock,1.0)
//  GradientChecker(L2Regularization(10, b), c) // L2 regularisation on a vector
  val matrixBlock = MatrixParam(4,4)
  val regularizationBlock = L2Regularization(99, matrixBlock)
  GradientChecker(regularizationBlock, matrixBlock) // L2 reg on a matrix
  val productBlock = Product(Seq(b,c))
  GradientChecker(Dot(productBlock,c),b)
  val mulBlock = Mul(matrixBlock, b)
  GradientChecker(Dot(mulBlock, mulBlock), matrixBlock)
  GradientChecker(Dot(mulBlock, mulBlock), b)
  val tanhBlock = Tanh(b)
  GradientChecker(Dot(tanhBlock,c),b)
  val sigBlock = VecSig(b)
  GradientChecker(Dot(sigBlock,c),b)
  //println(sigmoid(b.output))
  val concatBlock1 = Concat(b,c)
  val concatBlock2 = Concat(c,b)
  GradientChecker(Dot(concatBlock1, concatBlock2), b)
  val pointBlock = PointMul(b,c)
  GradientChecker(Dot(pointBlock,c),c)


  // Checking full model
  val sumVectorsModelWithDropout = new SumOfWordVectorsModelWithDropout(4, 0.1, 0.6)
  val sumVectorsModelBlockWithDropout = sumVectorsModelWithDropout.scoreSentence(sumVectorsModelWithDropout.wordVectorsToSentenceVector(Seq(sumVectorsModelWithDropout.wordToVector("Reece"),sumVectorsModelWithDropout.wordToVector("wins"))))
  GradientChecker(sumVectorsModelBlockWithDropout, sumVectorsModelWithDropout.vectorParams("Reece"))
  val sumVectorsModel = new SumOfWordVectorsModel(4, 0.1)
  val sumVectorsModelBlock = sumVectorsModel.scoreSentence(sumVectorsModel.wordVectorsToSentenceVector(Seq(sumVectorsModel.wordToVector("Reece"), sumVectorsModel.wordToVector("wins"))))
  GradientChecker(sumVectorsModelBlock, sumVectorsModel.vectorParams("Reece"))
  val productVectorsModel = new ProductOfWordVectorsModel(4, 0.1)
  val productVectorsModelBlock = productVectorsModel.scoreSentence(productVectorsModel.wordVectorsToSentenceVector(Seq(productVectorsModel.wordToVector("Reece"),productVectorsModel.wordToVector("wins"))))
  GradientChecker(productVectorsModelBlock, productVectorsModel.vectorParams("Reece"))
  val rnnModel = new RecurrentNeuralNetworkModel(4, 6, 0.1, 0.1)
  val rnnModelBlock = rnnModel.scoreSentence(rnnModel.wordVectorsToSentenceVector(Seq(rnnModel.wordToVector("Reece"),rnnModel.wordToVector("wins"))))
  GradientChecker(rnnModelBlock, rnnModel.vectorParams("Reece"))
  val lstmModel = new LSTMNetworkModel(4, 6, 0.1, 0.1)
  val lstmModelBlock = lstmModel.scoreSentence(lstmModel.wordVectorsToSentenceVector(Seq(lstmModel.wordToVector("Reece"),lstmModel.wordToVector("wins"))))
  GradientChecker(lstmModelBlock, lstmModel.vectorParams("Reece"))
}
