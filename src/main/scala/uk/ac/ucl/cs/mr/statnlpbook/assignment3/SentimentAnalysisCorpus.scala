package uk.ac.ucl.cs.mr.statnlpbook.assignment3

import java.io.PrintWriter

import com.github.tototoshi.csv._

import scala.util.Random

/**
 * @author rockt
 */
object SentimentAnalysisCorpus {
  val random = new Random(0L)
  def tokenize(sentence: String): Seq[String] = sentence.split(" ")

  def loadCorpus(lines: Iterator[String]): Corpus = lines.map(s => {
    val (target, tweet) = s.splitAt(1)
    tokenize(tweet) -> (target.toInt == 1)
  }).toIndexedSeq

  var train: Corpus = loadCorpus(io.Source.fromFile("./data/assignment3/train.tsv", "ISO-8859-1").getLines())
  var dev: Corpus = loadCorpus(io.Source.fromFile("./data/assignment3/dev.tsv", "ISO-8859-1").getLines())
  var debug: Corpus = loadCorpus(io.Source.fromFile("./data/assignment3/debug.tsv", "ISO-8859-1").getLines())
  var test: Corpus = loadCorpus(io.Source.fromFile("./data/assignment3/test.tsv", "ISO-8859-1").getLines())

  var trainCounter = 0
  var devCounter = 0
  var debugCounter = 0
  var testCounter = 0

  def numExamples(corpus: String) = corpus match {
    case "train" => train.size
    case "dev" => dev.size
    case "debug" => debug.size
    case "test" => test.size
  }

  def getExample(corpus: String): Example = corpus match {
    case "train" =>
      if (trainCounter == train.length - 1) {
        train = random.shuffle(train)
        trainCounter = -1
      }
      trainCounter = trainCounter + 1
      train(trainCounter)
    case "dev" =>
      if (devCounter == dev.length - 1) {
        dev = random.shuffle(dev)
        devCounter = -1
      }
      devCounter = devCounter + 1
      dev(devCounter)
    case "debug" =>
      if (debugCounter == debug.length - 1) {
        debug = random.shuffle(debug)
        debugCounter = -1
      }
      debugCounter = debugCounter + 1
      debug(debugCounter)
    case "test" =>
      if (testCounter == test.length - 1) {
        test = random.shuffle(test)
        testCounter = -1
      }
      testCounter = testCounter + 1
      test(debugCounter)
  }
}