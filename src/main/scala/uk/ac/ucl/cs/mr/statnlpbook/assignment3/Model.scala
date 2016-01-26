package uk.ac.ucl.cs.mr.statnlpbook.assignment3

import scala.collection.mutable

/**
 * @author rockt
 */
trait Model {
  /**
   * Stores all vector parameters
   */
  val vectorParams = new mutable.HashMap[String, VectorParam]()
  /**
   * Stores all matrix parameters
   */
  val matrixParams = new mutable.HashMap[String, MatrixParam]()
  /**
   * Maps a word to its trainable or fixed vector representation
   * @param word the input word represented as string
   * @return a block that evaluates to a vector/embedding for that word
   */
  def wordToVector(word: String): Block[Vector]
  /**
   * Composes a sequence of word vectors to a sentence vectors
   * @param words a sequence of blocks that evaluate to word vectors
   * @return a block evaluating to a sentence vector
   */
  def wordVectorsToSentenceVector(words: Seq[Block[Vector]]): Block[Vector]
  /**
   * Calculates the score of a sentence based on the vector representation of that sentence
   * @param sentence a block evaluating to a sentence vector
   * @return a block evaluating to the score between 0.0 and 1.0 of that sentence (1.0 positive sentiment, 0.0 negative sentiment)
   */
  def scoreSentence(sentence: Block[Vector]): Block[Double]
  /**
   * Predicts whether a sentence is of positive or negative sentiment (true: positive, false: negative)
   * @param sentence a tweet as a sequence of words
   * @param threshold the value above which we predict positive sentiment
   * @return whether the sentence is of positive sentiment
   */
  def predict(sentence: Seq[String])(implicit threshold: Double = 0.5): Boolean = {
    val wordVectors = sentence.map(wordToVector)
    val sentenceVector = wordVectorsToSentenceVector(wordVectors)
    scoreSentence(sentenceVector).forward() >= threshold
  }
  /**
   * Defines the training loss
   * @param sentence a tweet as a sequence of words
   * @param target the gold label of the tweet (true: positive sentiement, false: negative sentiment)
   * @return a block evaluating to the negative log-likelihod plus a regularization term
   */
  def loss(sentence: Seq[String], target: Boolean): Loss = {
    val targetScore = if (target) 1.0 else 0.0
    val wordVectors = sentence.map(wordToVector)
    val sentenceVector = wordVectorsToSentenceVector(wordVectors)
    val score = scoreSentence(sentenceVector)
    new LossSum(NegativeLogLikelihoodLoss(score, targetScore), regularizer(wordVectors))
  }
  /**
   * Regularizes the parameters of the model for a given input example
   * @param words a sequence of blocks evaluating to word vectors
   * @return a block representing the regularization loss on the parameters of the model
   */
  def regularizer(words: Seq[Block[Vector]]): Loss
}


/**
 * Problem 2
 * A sum of word vectors model
 * @param embeddingSize dimension of the word vectors used in this model
 * @param regularizationStrength strength of the regularization on the word vectors and global parameter vector w
 */
class SumOfWordVectorsModel(embeddingSize: Int, regularizationStrength: Double = 0.0) extends Model {
  /**
   * We use a lookup table to keep track of the word representations
   */
  override val vectorParams: mutable.HashMap[String, VectorParam] =
    LookupTable.trainableWordVectors
  /**
   * We are also going to need another global vector parameter
   */
  vectorParams += "param_w" -> VectorParam(embeddingSize)

  def wordToVector(word: String): Block[Vector] = {
    LookupTable.addTrainableWordVector(word, embeddingSize)
  }

  def wordVectorsToSentenceVector(words: Seq[Block[Vector]]): Block[Vector] = Sum(words)

  def scoreSentence(sentence: Block[Vector]): Block[Double] = Sigmoid(Dot(vectorParams("param_w"), sentence))

  def regularizer(words: Seq[Block[Vector]]): Loss = L2Regularization[Vector](regularizationStrength, words :+ vectorParams("param_w") :_*)
}


/**
 * Problem 3
 * A recurrent neural network model
 * @param embeddingSize dimension of the word vectors used in this model
 * @param hiddenSize dimension of the hidden state vector used in this model
 * @param vectorRegularizationStrength strength of the regularization on the word vectors and global parameter vector w
 * @param matrixRegularizationStrength strength of the regularization of the transition matrices used in this model
 */
class RecurrentNeuralNetworkModel(embeddingSize: Int, hiddenSize: Int,
                                  vectorRegularizationStrength: Double = 0.0,
                                  matrixRegularizationStrength: Double = 0.0) extends Model {
  override val vectorParams: mutable.HashMap[String, VectorParam] =
    LookupTable.trainableWordVectors
  vectorParams += "param_w" -> VectorParam(hiddenSize) // i think this is the size & we do dot of this with sentence vector for score?
  vectorParams += "param_h0" -> VectorParam(hiddenSize)
  vectorParams += "param_b" -> VectorParam(hiddenSize)

  override val matrixParams: mutable.HashMap[String, MatrixParam] =
    new mutable.HashMap[String, MatrixParam]()
  matrixParams += "param_Wx" -> MatrixParam(hiddenSize, embeddingSize)
  matrixParams += "param_Wh" -> MatrixParam(hiddenSize, hiddenSize)

  def wordToVector(word: String): Block[Vector] = LookupTable.addTrainableWordVector(word, embeddingSize)

  def wordVectorsToSentenceVector(words: Seq[Block[Vector]]): Block[Vector] =
    words.foldLeft[Block[Vector]](vectorParams("param_h0")){(h,x) =>
      Tanh(
        Sum(
          Seq(
            Mul(matrixParams("param_Wh"), h),
            Mul(matrixParams("param_Wx"), x),
            vectorParams("param_b")
          )
        )
      )
    }

  def scoreSentence(sentence: Block[Vector]): Block[Double] = Sigmoid(Dot(vectorParams("param_w"), sentence))

  def regularizer(words: Seq[Block[Vector]]): Loss =
    new LossSum(
      L2Regularization(vectorRegularizationStrength, words :+ vectorParams("param_w") :_*),
      L2Regularization(matrixRegularizationStrength, matrixParams("param_Wh"), matrixParams("param_Wx"))
    )
}

/**
 * Problem 4
 * A Long Short Term Memory Recurrent Neural Network model
 * @param embeddingSize dimension of the word vectors used in this model
 * @param hiddenSize dimension of the hidden state vector used in this model
 * @param vectorRegularizationStrength strength of the regularization on the word vectors and global parameter vector w
 * @param matrixRegularizationStrength strength of the regularization of the transition matrices used in this model
 */
class LSTMNetworkModel(embeddingSize: Int, hiddenSize: Int,
                                  vectorRegularizationStrength: Double = 0.0,
                                  matrixRegularizationStrength: Double = 0.0) extends Model {
  override val vectorParams: mutable.HashMap[String, VectorParam] =
    LookupTable.trainableWordVectors
  vectorParams += "param_w" -> VectorParam(hiddenSize) // i think this is the size & we do dot of this with sentence vector for score?
  vectorParams += "param_h0" -> VectorParam(hiddenSize)
  vectorParams += "param_C0" -> VectorParam(hiddenSize)
  vectorParams += "param_bf" -> VectorParam(hiddenSize)
  vectorParams += "param_bi" -> VectorParam(hiddenSize)
  vectorParams += "param_bc" -> VectorParam(hiddenSize)
  vectorParams += "param_bo" -> VectorParam(hiddenSize)

  override val matrixParams: mutable.HashMap[String, MatrixParam] =
    new mutable.HashMap[String, MatrixParam]()
  matrixParams += "param_Wf" -> MatrixParam(hiddenSize, embeddingSize+hiddenSize)
  matrixParams += "param_Wi" -> MatrixParam(hiddenSize, embeddingSize+hiddenSize)
  matrixParams += "param_Wc" -> MatrixParam(hiddenSize, embeddingSize+hiddenSize)
  matrixParams += "param_Wo" -> MatrixParam(hiddenSize, embeddingSize+hiddenSize)

  def wordToVector(word: String): Block[Vector] = LookupTable.addTrainableWordVector(word, embeddingSize)

  def wordVectorsToSentenceVector(words: Seq[Block[Vector]]): Block[Vector] =
    words.foldLeft[(Block[Vector],Block[Vector])]((vectorParams("param_h0"),vectorParams("param_C0"))){(h,x) =>
      {
        val hx = Concat(h._1, x)
        val ft = VecSig(Sum(Seq(Mul(matrixParams("param_Wf"), hx), vectorParams("param_bf"))))
        val it = VecSig(Sum(Seq(Mul(matrixParams("param_Wi"), hx), vectorParams("param_bi"))))
        val Ctwiddlet = Tanh(Sum(Seq(Mul(matrixParams("param_Wc"), hx), vectorParams("param_bc"))))
        val Ct = Sum(Seq(PointMul(ft, h._2), PointMul(it, Ctwiddlet)))
        val ot = VecSig(Sum(Seq(Mul(matrixParams("param_Wo"), hx), vectorParams("param_bo"))))
        val ht = PointMul(ot, Tanh(Ct))
        (ht,Ct)
      }
    }._1

  def scoreSentence(sentence: Block[Vector]): Block[Double] = Sigmoid(Dot(vectorParams("param_w"), sentence))

  def regularizer(words: Seq[Block[Vector]]): Loss =
    new LossSum(
      L2Regularization(vectorRegularizationStrength, words ++ vectorParams.values.toSeq :_*),
      L2Regularization(matrixRegularizationStrength, matrixParams.values.toSeq :_*)
    )
}