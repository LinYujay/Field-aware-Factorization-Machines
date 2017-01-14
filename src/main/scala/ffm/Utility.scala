package ffm

import scala.util.Random

/**
 * Created by yjlin on 2016/12/31.
 */
object Utility {

    def matrix(rows:Int, cols:Int):Array[Array[Double]] = {
        val rand = new Random()
        var _matrix:Array[Array[Double]] = Array.ofDim[Double](rows, cols)
        for (i <-0 until rows)
            for ( j <- 0 until cols)
                {
                    _matrix(i)(j) = rand.nextDouble() / 15.0
                    if (rand.nextInt()%2 == 0) _matrix(i)(j) *= -1
                }
        _matrix
    }
}
