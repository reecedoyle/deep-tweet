package uk.ac.ucl.cs.mr.statnlpbook.assignment3

/**
 * Problem 2
 */
object StochasticGradientDescentLearner extends App {
  def apply(model: Model, corpus: String, maxEpochs: Int = 10, learningRate: Double, epochHook: (Int, Double) => Unit): Unit = {
    val iterations = SentimentAnalysisCorpus.numExamples(corpus)
    for (i <- 0 until maxEpochs) {
      var accLoss = 0.0
      for (j <- 0 until iterations) {
        if (j % 1000 == 0) print(s"Iter $j\r")
        val (sentence, target) = SentimentAnalysisCorpus.getExample(corpus)
        val lossBlock = model.loss(sentence,target)
        accLoss += lossBlock.forward()
        lossBlock.backward()
        lossBlock.update(learningRate)

      }
      epochHook(i, accLoss)
    }
  }
}
