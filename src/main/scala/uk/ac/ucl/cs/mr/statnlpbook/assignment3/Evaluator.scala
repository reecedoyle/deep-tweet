package uk.ac.ucl.cs.mr.statnlpbook.assignment3

/**
 * @author rockt
 */
object Evaluator {
  def apply(model: Model, corpus: String): Double = {
    val total = SentimentAnalysisCorpus.numExamples(corpus)
    var correct = 0.0
    for (i <- 0 until total) {
      val (sentence, target) = SentimentAnalysisCorpus.getExample(corpus)
      val predict = model.predict(sentence)
      if (target == predict) correct = correct + 1
    }
    correct / total
  }
}
