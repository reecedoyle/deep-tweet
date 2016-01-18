package uk.ac.ucl.cs.mr.statnlpbook.assignment3

import scala.collection.mutable

/**
 * @author rockt
 */
object LookupTable {
  val fixedWordVectors = new mutable.HashMap[String, VectorConstant]()
  val trainableWordVectors = new mutable.HashMap[String, VectorParam]()

  def addFixedWordVector(word: String, vector: Vector): Block[Vector] = {
    fixedWordVectors.getOrElseUpdate(word, vector)
  }

  def addTrainableWordVector(word: String, dim: Int = 300): Block[Vector] = {
    trainableWordVectors.getOrElseUpdate(word, VectorParam(dim))
  }

  def addTrainableWordVector(word: String, vector: Vector): Block[Vector] = {
    val param = VectorParam(vector.activeSize)
    param.set(vector)
    trainableWordVectors.getOrElseUpdate(word, param)
  }

  def get(word: String): Block[Vector] = trainableWordVectors.getOrElse(word, fixedWordVectors(word))
}
