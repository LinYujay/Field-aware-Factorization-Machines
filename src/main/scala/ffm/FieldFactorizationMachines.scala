package ffm

import java.text.SimpleDateFormat
import java.util.Date

import breeze.numerics.{abs, exp}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.sql.DataFrame

import scala.collection.mutable.ArrayBuffer

/***********************************************************************
  * Field-aware Factorization Machines model implemented in spark,
  * Stochastic gradient descent was applied in this first version.
  * Following features are planed, including early stopping for over-fitting,
  * momentum and self-adaption learning rate type gradient methods for speeding
  * up train procedure, etc...
  *
  * See <<Field-aware Factorization Machines for CTR prediction, Y Juan, Y Zhuang, WS Chin, CJ Lin>>
  *     for mathematical derivation.
  *
  * Please feel free to contact me if you have any question about the source code.
  * Email: Yujay.w.lin@gmail.com
  *
  * Created by yjlin on 2016/12/31.
  **********************************************************************/

class FieldFactorizationMachines(fieldsConfig:Array[Int], latentVariableNumber:Int) extends Serializable{

    /***********************************************************************
      * lr         - learning rate
      * steps      - max steps to iterate
      * lambda     - parameter for ridge regression(L2)
      * partitions - number of partitions
      * weights    - weights of Field-aware-Factorization-Machines
      * batchSize  - batch size for SGD
      *
      **********************************************************************/
    var lr = 1.0
    var steps = 10
    var lambda = 0.0001
    var batchSize = 1024

    var weights:Map[Int, Map[Int, Feature]] = {
        var _weights = Map[Int, Map[Int, Feature]]()

        for((i, n) <- (0 until fieldsConfig.length) zip fieldsConfig)
        {
            var _temp = Map[Int, Feature]()
            for(i <- 0 until n) _temp += (i -> new Feature(fieldsConfig.length, latentVariableNumber))
            _weights += (i -> _temp)
        }
        _weights
    }

    /********************************************************************
      * setting interface
      *
      *******************************************************************/
    def setLr(_lr:Double) = lr = 1.0

    def setLambda(_lambda:Double) = lambda = _lambda

    def setBatchSize(_batchSize:Int) = batchSize = _batchSize

    /**********************************************************************
      * train Field-aware Factorization Machines
      *
      * input - DataFrame("label", "features")
      *         label = 1 or -1
      *         features = element of "#field:#bucket#", splitted by space
      *
      *********************************************************************/
    def fit(df:DataFrame) = {
        val count:Double = df.count()

        var sampleRate = batchSize * 1.0 / count
        if (sampleRate > 1.0) sampleRate = 1.0

        for(step <- 0 until steps)
        {
            _log("step: " + step)
            val gradient = _gradientDescent(
                df.sample(false, sampleRate))

            _applyGradient(gradient)
        }
    }

    /**********************************************************************
      * calculate batch gradient descent for Field-aware-Factorization-Machines
      *
      *********************************************************************/
    private def _gradientDescent(df:DataFrame) = {
        val _weights = weights

        val results = df.flatMap(row =>{
            val label = row.getAs[String]("label").toInt
            val features = row.getAs[String]("features").split(" ")
            val gradients = _calGradient(features, _weights, label)
            gradients.toList
        })
          .reduceByKey{
              case (g1, g2) => (_reduceGradient(g1, g2))
          }
          .collect()

        results
    }

    /**********************************************************************
      * calculate gradient for one instance
      *
      *********************************************************************/
    private def _calGradient(features:Array[String], _weights:Map[Int, Map[Int, Feature]], label:Int) = {
        var gradients = ArrayBuffer[Tuple2[String, String]]()

        val logits = _FFM(features, _weights)
        val ldf = _ldf(logits, label)
        val loss = _loss(logits, label)

        gradients += Tuple2("AE", loss.toString)

        features.foreach(feature1 => {
            val field1 = feature1.split(":")(0).toInt
            val bucket1 = feature1.split(":")(1).toInt

            features.foreach(feature2 => {
                val field2 = feature2.split(":")(0).toInt
                val bucket2 = feature2.split(":")(1).toInt

                if(field1 < field2)
                {
                    val fdw12 = _fdw(field1, bucket1, field2, _weights)
                    val fdw21 = _fdw(field2, bucket2, field1, _weights)
                    val g12 = _add(_dot(fdw12, lambda), _dot(fdw21, ldf))
                    val g21 = _add(_dot(fdw21, lambda), _dot(fdw12, ldf))

                    gradients += Tuple2(field1 + ":" + bucket1 + ":" + field2, g12.mkString("@"))
                    gradients += Tuple2(field2 + ":" + bucket2 + ":" + field1, g21.mkString("@"))
                }
            })
        })
        gradients
    }

    /**********************************************************************
      * loss derivate Field-aware-Factorization-Machines
      *
      *********************************************************************/
    private def _ldf(logits:Double, label:Int) = - label.toDouble / (1 + exp(label.toDouble * logits))


    /**********************************************************************
      * Field-aware-Factorization-Machines derivate weight
      *
      *********************************************************************/
    private def _fdw(field1:Int, bucket1:Int, field2:Int, _weights:Map[Int, Map[Int, Feature]]) = {
        _weights(field1)(bucket1).weights(field2)
    }

    /**********************************************************************
      * apply gradient for Field-aware-Factorization-Machines
      *
      *********************************************************************/
    private def _applyGradient(gradients:Array[Tuple2[String, String]]) = {

        gradients.foreach(gradient => {

            if(gradient._1 != "AE")
            {
                val field1 = gradient._1.split(":")(0).toInt
                val bucket1 = gradient._1.split(":")(1).toInt
                val field2 = gradient._1.split(":")(2).toInt

                val v = weights(field1)(bucket1).weights(field2)

                for((g, i) <- gradient._2.split("@") zip (0 until gradient._2.split("@").length))
                {
                    val w = weights(field1)(bucket1).weights(field2)(i)
                    weights(field1)(bucket1).weights(field2)(i) -= g.toDouble * (1.0 / batchSize) * lr
                }
            }
            else{
                _log("MAE: " + gradient._2.toDouble / batchSize)
            }
        })
    }

    /**********************************************************************
      * reduce gradients for weight ij
      *
      *********************************************************************/
    private def _reduceGradient(g1:String, g2:String) = {
        var s = ""
        for((i, j) <- g1.split("@") zip g2.split("@"))
        {
            s += (i.toDouble + j.toDouble).toString + "@"
        }

        s.substring(0, s.length() - 1)
    }

    /*********************************************************************
      * Field-aware-Factorization-Machines score
      *
      **********************************************************************/
    private def _FFM(features:Array[String], _weights:Map[Int, Map[Int, Feature]]) = {
        var result = 0.0
        features.foreach(feature1 => {
            val field1 = feature1.split(":")(0).toInt
            val bucket1 = feature1.split(":")(1).toInt

            features.foreach(feature2 => {
                val field2 = feature2.split(":")(0).toInt
                val bucket2 = feature2.split(":")(1).toInt

                if(field1 < field2)
                {
                    val weights1 = _weights(field1)(bucket1).weights(field2)
                    val weights2 = _weights(field2)(bucket2).weights(field1)
                    for((i, j) <- weights1 zip weights2) result += i * j
                }
            })
        })

        result
    }

    /***********************************************************************
      * math ops
      *
      ************************************************************************/
    private def _dot(v1:Array[Double], value:Double) = {
        val v = ArrayBuffer[Double]()
        for(i <- v1)
            v += i * value
        v.toArray
    }

    private def _add(v1:Array[Double], v2:Array[Double]) = {
        val v = ArrayBuffer[Double]()
        for((i, j) <- v1 zip v2)
            v += (i + j)
        v.toArray
    }

    private def _sigmoid(logits:Double) = 1.0 / 1.0 + exp(-logits)

    private def _loss(logits:Double, label:Int) = abs((exp(logits) - exp(-logits)) / (exp(logits) + exp(-logits)) - label)

    /************************************************************************
      * evaluate trained model
      *
      *************************************************************************/
    def auc(df:DataFrame) = {
        val m = new BinaryClassificationMetrics(_predict(df))
        val a = m.areaUnderROC()
        a
    }

    private def _predict(df:DataFrame) = {
        val _weights = weights

        df.map(row => {
            val features = row.getAs[String]("features").split(" ")
            val logits = _FFM(features, _weights)
            val score = _sigmoid(logits)

            var label = 1.0
            if (row.getAs[String]("label").toInt != -1) label = 0.0

            (score, label)
        })
    }

    /************************************************************************
      * log with timestamp
      *
      *************************************************************************/
    private def _log(info:String) = {
        var timeTemplate:SimpleDateFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss")
        println(timeTemplate.format(new Date()) + "|" + info)
    }

    /*************************************************************************
      * optional, generate learning rate based on step & max step
      *
      **************************************************************************/

    //    def lr(epoch:Int) = {
    //
    //    }
}