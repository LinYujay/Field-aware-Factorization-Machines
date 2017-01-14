package ffm

/**
 * Created by yjlin on 2016/12/31.
 */
class Feature(fieldsNumber:Int, latentVariableNumber:Int) extends Serializable{
    var weights = Utility.matrix(fieldsNumber, latentVariableNumber)

    def show() = {
        weights.foreach(__weights => {
            __weights.foreach(w => print(w + " "))
            println()
        })
    }
}
