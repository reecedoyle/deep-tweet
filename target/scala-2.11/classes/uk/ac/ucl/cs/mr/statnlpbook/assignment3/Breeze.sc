import breeze.linalg.{DenseMatrix, DenseVector}

//creation of a vector
val x = new DenseVector[Double](Array(1.0, 2.0, -0.5))
val y = new DenseVector[Double](Array(-0.5, 3.0, 2.0))
//creation of a vector with a specific value
val z = Vector.fill[Double](6){2.7}
//vector dot product
x dot x
//vector addition
x+x
//element-wise vector multiplication
x :* x
//outer product
x * x.t
//in-place vector addition, subtraction and multiplication
x :+= y
x :-= y
x :*= y
//scalar vector addition, subtraction and multiplication
x + 1.0
1.0 - x
x * 2.0

//creation of a matrix
val W = new DenseMatrix[Double](2,3, Array(1.0, -2.0, 3.0, -4.0, 5.0, -6.0))
W.rows
W.cols
//matrix-vector multiplication
W*x
//vector transpose
x.t
//matrix transpose
W.t.rows == W.cols

//element-wise functions
breeze.numerics.sigmoid(x)
breeze.numerics.tanh(x)
breeze.linalg.clip(x, -1.0, 1.0)
