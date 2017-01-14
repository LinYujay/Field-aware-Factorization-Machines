package ffm

import java.text.SimpleDateFormat
import java.util.Date

import breeze.numerics.abs
import org.apache.spark.{SparkContext, SparkConf}

import scala.collection.mutable.ListBuffer
import scala.util.Random

/**
 * Created by yjlin on 2016/12/31.
 */
object test {
    val conf = new SparkConf().setAppName("Field-aware-Factorization-Machines")
    .setMaster("local")

    val sc = new SparkContext(conf)
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)

    def main(args: Array[String]) {

        val lr = 10
        val epochs = 10
        val lambda = 0.001
        val partitionNUM = 1

        var now:Date = new Date()
        var dateFormat:SimpleDateFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss")

        val start = dateFormat.format(now)
        val data = testData()

        val fieldsConfig = Array[Int](10, 10, 10, 10, 10, 10, 10, 10, 10, 10)
        val ffm = new FieldFactorizationMachines(fieldsConfig, 10)

        // ffm.show()

        ffm.fit(data)

        // ffm.show()

        println("auc: " + ffm.auc(data))

        println("start at: " + start)
        println("end at: " + dateFormat.format(new Date()))
    }

    def testData() = {
        import sqlContext.implicits._
        val rand = new Random()
        val l = new ListBuffer[Array[String]]()
        for(i <- 0 until 1000)
        {
            var label = -1
            if(abs(rand.nextInt())%2 == 1)
                label = 1

            if (label == 1)
            {
                l += Array(label.toString
                    ,abs(rand.nextInt()%5) + ":" + abs(rand.nextInt()%10)
                      + " " + abs(rand.nextInt()%5) + ":" + abs(rand.nextInt()%10)
                      + " " + abs(rand.nextInt()%5) + ":" + abs(rand.nextInt()%10)
                      + " " + abs(rand.nextInt()%5) + ":" + abs(rand.nextInt()%10)
                      + " " + abs(rand.nextInt()%5) + ":" + abs(rand.nextInt()%10))

            }
            else{
                l += Array(label.toString
                    ,(abs(rand.nextInt()%5) + 5) + ":" + abs(rand.nextInt()%10)
                      + " " + (abs(rand.nextInt()%5) + 5) + ":" + abs(rand.nextInt()%10)
                      + " " + (abs(rand.nextInt()%5) + 5) + ":" + abs(rand.nextInt()%10)
                      + " " + (abs(rand.nextInt()%5) + 5) + ":" + abs(rand.nextInt()%10)
                      + " " + (abs(rand.nextInt()%5) + 5) + ":" + abs(rand.nextInt()%10))

            }
        }

        sc.parallelize(l.toList).map(row => (row(0), row(1))).toDF("label", "features")
    }
}
