package Utils

import Clasificadores.Processing.balancedDF
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.{Column, DataFrame, SparkSession}
import org.apache.spark.sql.functions.{col, lit, when}
import org.apache.spark.sql.types.FloatType

object Utils {

   def prepareData(args:Array[String], configs: Map[String, Any])(implicit spark:SparkSession): (DataFrame, DataFrame)= {
     val dfStart=if(args.length>1 && args(1)=="1") {
       val dfAux=spark.sqlContext.read.parquet(configs("dataset").toString)
       val lisCols = dfAux.columns.map(i => (i, col(i).cast("Double"))).toMap[String, Column]
       dfAux.withColumns(lisCols)
     }
     else {
       spark.sqlContext.read.parquet(configs("dataset").toString)
       //balancedDF(df, "class")
     }
    // dfStart.show()
     val cols=dfStart.columns.filter(_!="class")

     val dfFeatures=dfStart
       .withColumn("label", col("class"))
     println("Prepare data")
     val assembler=new VectorAssembler()
       .setInputCols(cols)
       .setOutputCol("features")

     val featureDf: DataFrame = assembler.transform(dfFeatures).select("features", "label")
     println(configs("test").toString)
     val test=spark.sqlContext.read.parquet(configs("test").toString)
     //val lisCols = test1.columns.map(i => (i, col(i).cast("Double"))).toMap[String, Column]
     //val test=test1.withColumns(lisCols)

     //val colsTest=test.columns.filter(_!="class")
     val dfTest=test
       .withColumn("label",col("class"))
     //dfTest.show(false)
     val assemblerTest=new VectorAssembler()
       .setInputCols(cols)
       .setOutputCol("features")

     val dfTestFeatures: DataFrame = assemblerTest.transform(dfTest).select("features", "label")

     (featureDf, dfTestFeatures)
   }

}
