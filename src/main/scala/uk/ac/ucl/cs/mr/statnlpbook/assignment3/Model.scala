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

  def loadVectorRepresentationConstant(path: String): mutable.HashMap[String, VectorConstant] = {
    val fixedParams = new mutable.HashMap[String, VectorConstant]()
    val bufferedSource = io.Source.fromFile("vecs.txt")
      val lines = bufferedSource.getLines().drop(1)
      for (line <- lines) {
        val splitLine = line.split(" ")
        val word = splitLine(0)
        if (word != "") {
          val vectorised = vec(splitLine.tail.map(e => e.toDouble):_*)
          val entry = VectorConstant(vectorised)
          fixedParams += word -> entry
        }
      }
    fixedParams
  }

  def loadVectorRepresentationTrainable(embeddingSize: Int, path: String): mutable.HashMap[String, VectorParam] = {
    val trainableParams = new mutable.HashMap[String, VectorParam]()
    val bufferedSource = io.Source.fromFile("vecs.txt")
    val lines = bufferedSource.getLines().drop(1)
    for (line <- lines) {
      val splitLine = line.split(" ")
      val word = splitLine(0)
      if (word != "") {
        val vectorised = vec(splitLine.tail.map(e => e.toDouble):_*)
        val entry = VectorParam(embeddingSize)
        entry.param = vectorised
        trainableParams += word -> entry
      }
    }
    trainableParams
  }

}




// Load the vector representations obtained from word2vec as a map. Input is a text file with a word and its
// vector representation seperated by spaces on each line


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

// Model where we supply vectors from word2vec as constant vectors
class SumOfWordVectorsModelWithConstantVectors(embeddingSize: Int, regularizationStrength: Double = 0.0) extends Model {
  /**
   * We use a lookup table to keep track of the word representations
   */
  override val vectorParams: mutable.HashMap[String, VectorParam] =
    LookupTable.trainableWordVectors


//    var fixedParams = LookupTable.fixedWordVectors
    val fixedParams = loadVectorRepresentationConstant("vecs.txt")

  /**
   * We are also going to need another global vector parameter
   */
  vectorParams += "param_w" -> VectorParam(embeddingSize)

  // If we've seen the word at training time, take its vector representation from fixedParams,
  // else take it from vectorParams
  def wordToVector(word: String): Block[Vector] = {
    if (fixedParams.contains(word)) {
      return fixedParams.get(word).get
    }
    else return LookupTable.addTrainableWordVector(word, embeddingSize)
  }

  def wordVectorsToSentenceVector(words: Seq[Block[Vector]]): Block[Vector] = Sum(words)

  def scoreSentence(sentence: Block[Vector]): Block[Double] = Sigmoid(Dot(vectorParams("param_w"), sentence))

  def regularizer(words: Seq[Block[Vector]]): Loss = L2Regularization[Vector](regularizationStrength, words :+ vectorParams("param_w") :_*)
}

// Model where we supply vectors from word2vec as trainable vectors
class SumOfWordVectorsModelWithTrainableVectors(embeddingSize: Int, regularizationStrength: Double = 0.0) extends Model {
  /**
   * We use a lookup table to keep track of the word representations
   */
  override val vectorParams: mutable.HashMap[String, VectorParam] =
    LookupTable.trainableWordVectors

  vectorParams += "param_w" -> VectorParam(embeddingSize)
  val vecReps = loadVectorRepresentationTrainable(embeddingSize, "vecs.txt")
  vecReps.foreach(kv => vectorParams.put(kv._1, kv._2))

  // If we've seen the word at training time, take its vector representation from fixedParams,
  // else take it from vectorParams
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

class SumOfWordVectorsModelWithDropout(embeddingSize: Int, regularizationStrength: Double = 0.0, prob: Double) extends Model {
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

  def wordVectorsToSentenceVector(words: Seq[Block[Vector]]): Block[Vector] = Sum(words.map(x => DropoutSum(prob, x)))

  def scoreSentence(sentence: Block[Vector]): Block[Double] = Sigmoid(Dot(vectorParams("param_w"), sentence))

  def regularizer(words: Seq[Block[Vector]]): Loss = L2Regularization[Vector](regularizationStrength, words :+ vectorParams("param_w") :_*)
}

/**
  * Problem 4
  * A product of word vectors model
  * @param embeddingSize dimension of the word vectors used in this model
  * @param regularizationStrength strength of the regularization on the word vectors and global parameter vector w
  */
class ProductOfWordVectorsModel(embeddingSize: Int, regularizationStrength: Double = 0.0) extends Model {
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

  def wordVectorsToSentenceVector(words: Seq[Block[Vector]]): Block[Vector] = Product(words)

  def scoreSentence(sentence: Block[Vector]): Block[Double] = Sigmoid(Dot(vectorParams("param_w"), sentence))

  def regularizer(words: Seq[Block[Vector]]): Loss = L2Regularization[Vector](regularizationStrength, words :+ vectorParams("param_w") :_*)
}