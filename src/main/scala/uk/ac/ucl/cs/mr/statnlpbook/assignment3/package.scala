package uk.ac.ucl.cs.mr.statnlpbook

import breeze.linalg.{DenseMatrix, DenseVector}

import scala.language.implicitConversions

import scala.util.Random

/**
 * @author rockt
 */
package object assignment3 {
  type Example = (Seq[String], Boolean)
  type Corpus = Seq[Example]

  type Vector = DenseVector[Double]
  type Matrix = DenseMatrix[Double]
  val random = new Random(0L)

  implicit def vectorToVectorConstant(vec: Vector): VectorConstant = VectorConstant(vec)
  def doubleToVector(d: Double): Vector = new Vector(Array(d))
  implicit def doubleToVectorConstant(d: Double): VectorConstant = VectorConstant(doubleToVector(d))
  implicit def doubleToDoubleConstant(d: Double): DoubleConstant = DoubleConstant(d)
  
  def vec(values: Double*): Vector = new Vector(values.toArray)
  def mat(dim1: Int, dim2: Int)(values: Double*): Matrix = new Matrix(dim1, dim2, values.toArray)

  def randVec(dim: Int, dist: () => Double): Vector = vec((0 until dim).map(i => dist()):_*)
  def randMat(dim1: Int, dim2: Int, dist: () => Double): Matrix = new Matrix(dim1, dim2, (0 until dim1 * dim2).map(i => dist()).toArray)

  def outer(arg1: Vector, arg2: Vector): Matrix = arg1 * arg2.t

  def eye(dim1: Int, dim2: Int, value: Double = 1.0): Matrix = {
    val W = new Matrix(dim1, dim2)
    for (i <- 0 until dim1) {
      W(i,i) = value
    }
    W
  }
}
