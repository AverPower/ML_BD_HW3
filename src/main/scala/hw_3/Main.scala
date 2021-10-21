package hw_3

import breeze.linalg._
import breeze.numerics._
import java.io._
import java.nio.file._




case class linearRegression(data: DenseMatrix[Double], values: DenseMatrix[Double]){
  val eps: Double = 0.0001
  val lr: Double = 0.1
  var coef: DenseMatrix[Double] = DenseMatrix.zeros(data.cols, 1)
  var bias: Double = 0
  def fit(): Unit ={
    var difCoef: DenseMatrix[Double] = DenseMatrix.zeros(data.cols, 1)
    var difBias: Double = 0
    while (true) {
      difCoef = DenseMatrix.zeros(data.cols, 1)
      difBias = 0
      var prediction: DenseMatrix[Double] = data * coef + bias
      var y_dif: DenseMatrix[Double] = values - prediction
      for (i <- 0 until data.rows) {
        difCoef = difCoef + (data(i, ::) * y_dif(i, 0)).t.toDenseMatrix.t
      }

      difCoef = -2.0 * difCoef / data.rows.toDouble
      difBias = -1.0 * sum(y_dif(::, *)).t.apply(0) / data.rows

      var counter: Int = 0
      for (i <- 0 until data.cols){
        if (abs(lr * difCoef.apply(i, 0)) < eps){
          counter += 1
        }
      }
      if (counter == data.cols){
        return
      }

      coef = coef - difCoef * lr
      bias = bias - difBias * lr
    }
  }

  def predict(dataPredict: DenseMatrix[Double]): DenseMatrix[Double] ={
    var prediction: DenseMatrix[Double] = dataPredict * coef + bias
    return prediction
  }
}


object Main {

  def meanSquaredError(x: DenseMatrix[Double], y: DenseMatrix[Double]): Double={
    return sum((x - y).map(a => math.pow(a, 2))) / x.rows
  }
  def meanAbsoluteError(x: DenseMatrix[Double], y: DenseMatrix[Double]): Double={
    return sum(abs(x - y)) / x.rows
  }


  def crossValidate(data:DenseMatrix[Double], values:DenseMatrix[Double], folds: Int = 5): Unit ={

    val logPath: String = Paths.get("").toAbsolutePath.toString + "/logs/log.txt"
    val logWriter = new PrintWriter(new File(logPath ))
    logWriter.write("Cross validation started\n")
    val foldSize: Int = (data.rows / folds).toInt
    for (curFold <- 0 until folds){
      val dataTrain:DenseMatrix[Double] = data((0 until (folds - curFold - 1) * foldSize)++((folds - curFold) * foldSize until data.rows), ::).toDenseMatrix
      val dataTest:DenseMatrix[Double] = data( (folds - curFold - 1) * foldSize until (folds - curFold) * foldSize, ::)
      val valueTrain:DenseMatrix[Double] = values((0 until (folds - curFold - 1) * foldSize)++((folds - curFold) * foldSize until data.rows), ::).toDenseMatrix
      val valueTest:DenseMatrix[Double] = values((folds - curFold - 1) * foldSize until (folds - curFold) * foldSize, ::)
      val linReg: linearRegression = linearRegression(dataTrain, valueTrain)
      linReg.fit()
      val valuePrediction = linReg.predict(dataTest)
      val errorMAE: Double = meanAbsoluteError(valueTest, valuePrediction)
      val errorMSE: Double = meanSquaredError(valueTest, valuePrediction)
      logWriter.write(f"On ${curFold + 1} fold MAE = $errorMAE, MSE = $errorMSE\n")

    }
    logWriter.write("Cross validation finished\n")
    logWriter.close()

  }

  def main(args: Array[String]): Unit = {

    var pathTrain: String = ""
    var pathTest: String = ""

    if (args.length == 2){
     pathTrain= Paths.get("").toAbsolutePath.toString + args(0)
     pathTest = Paths.get("").toAbsolutePath.toString + args(1)
    }
    else{
      pathTrain = Paths.get("").toAbsolutePath.toString + "/data/data_train.csv"
      pathTest = Paths.get("").toAbsolutePath.toString + "/data/data_test.csv"
    }

    var dataTrain: DenseMatrix[Double] = csvread(new File(pathTrain), separator=',')
    var dataTest: DenseMatrix[Double] = csvread(new File(pathTest), separator=',')
    var values: DenseMatrix[Double] = dataTrain(::, -1 to -1)
    dataTrain = dataTrain(::, 0 to -2)

    crossValidate(dataTrain, values)

    val linearRegressor: linearRegression = linearRegression(dataTrain, values)
    linearRegressor.fit()
    val prediction: DenseMatrix[Double] = linearRegressor.predict(dataTest)
    val pathOutput: String = Paths.get("").toAbsolutePath.toString + "/data/prediction.csv"
    csvwrite(new File(pathOutput), prediction)
  }
}
