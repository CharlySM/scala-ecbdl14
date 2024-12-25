package Utils

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.{Column, DataFrame, SparkSession}
import org.apache.spark.sql.functions.{col, lit, when}

object Utils {

  /**This functions evaluate the result obtained in df predictions, where this dataframe contain the next columns:
   * prediction and label. This dataframe is transformed in rdd and then is used Multiclass metrics to obtain
   * the confusion matrix and from here is obtained tax positives, false positives, false negatives and tax negatives.
   * The is calculated the specificity and sensivity to calculate our metric.
   * @param predictions
   * @return It is returned the tax positives, tax negatives and multiplication of Sensitivity by specificity
   */
  def evaluate(predictions: DataFrame): (Double, Double, Double) = {
    val pred = predictions.select("prediction", "label").rdd.map(x => (x.getDouble(0), x.getDouble(1)))
    val metrics = new MulticlassMetrics(pred)
    val matriz_confusion = metrics.confusionMatrix.toArray
    val tp = matriz_confusion(0)
    val fp = matriz_confusion(1)
    val fn = matriz_confusion(2)
    val tn = matriz_confusion(3)

    val TPR = tp / (tp + fn)
    val TNR = tn / (tn + fp)
    val score = TPR * TNR
    (TPR, TNR, score)
  }

  /**
   * This function balance a dataframe desbalanced, it is balanced by undersampling.
   * First is calculated the number of positive class and negative class, then
   * divide then minority class by majority class and the result is save in ratio variable.
   * The balancing is done with sample function from spark with ratio variable calculated before.
   * Then is returned the union of minority class with the result of
   * the call to sample function with ratio variable.
   * @param data
   * @param ratio
   * @param column
   * @return
   */
  def random_undersampling(data: DataFrame, ratio: Double, column:String): DataFrame = {
    val df_minority = data.filter(data(column) === 1.0)
    val df_majority = data.filter(data(column) === 0.0)
    val counMinority=df_minority.count()
    val counMajority=df_majority.count()
    println("count minority: ", counMinority)
    println("count Majority: ", counMajority)
    var ratio_real = df_minority.count().toDouble / df_majority.count().toDouble
    println("Ratio real antes de modificacion: ", ratio_real)
    ratio_real = ratio_real * ratio.toLong
    println("Ratio real: ", ratio_real)
    val df_majority_under = df_majority.sample(withReplacement = true, ratio_real, seed = 12345678)
    val union=df_majority_under.union(df_minority)
    val unionCount1s=union.filter(union(column) === 1.0).count()
    val unionCount0s=union.filter(union(column) === 0.0).count()
    println("count 1s: ", unionCount1s)
    println("count 0s: ", unionCount0s)
    union
  }

  /**
   * This function balance a dataframe desbalanced, it is balanced by undersampling.
   * First is calculated the number of positive class and negative class, then
   * divide then minority class by majority class and the result is save in ratio variable.
   * The balancing is done with sample function from spark with ratio variable calculated before.
   * Then is returned the union of minority class with the result of
   * the call to sample function with ratio variable.
   * @param df
   * @param column
   * @return
   */
  def balancedDF(df:DataFrame, column:String): DataFrame = {
    val dfP=df.filter(s"$column=1")
    val dfN=df.filter(s"$column=0")
    val nP=dfP.count().toDouble
    val nN=dfN.count().toDouble
    val major= if(nP>nN) nP else nN
    val minor=if(nP<nN) nP else nN
    val dfMinor=if(nP<nN) dfP else dfN
    val ratio = minor/major
    val dfNSample=dfN.sample(withReplacement = true, ratio)
    dfMinor.unionAll(dfNSample)
  }

  /**
   * This function read the training dataset and test datasets, if the datasets contains features and label columns
   * return this datasets else it is created the features and label column with vectorAssambler and is selected this columns
   * and returned the news datasets
   * @param args
   * @param configs
   * @param spark
   * @return The dtasets train and test
   */
   def prepareData(args:Array[String], configs: Map[String, Any])(implicit spark:SparkSession): (DataFrame, DataFrame)= {
     val dfStart=if(args.length>1 && args(1)=="1") {
       val dfAux=spark.sqlContext.read.parquet(configs("dataset").toString)
       val lisCols = dfAux.columns.map(i => (i, col(i).cast("Double"))).toMap[String, Column]
       dfAux.withColumns(lisCols)
     }
     else {
       val df=spark.sqlContext.read.parquet(configs("dataset").toString)
       random_undersampling(df, 1.0, "class")
     }

     //dfStart.printSchema()

     val featureDf=if(dfStart.columns.contains("label") && dfStart.columns.contains("features")) dfStart
     else {

       val cols = dfStart.columns.filter(_ != "class")

       val dfFeatures = dfStart
         .withColumn("label", col("class").cast("Double"))
       println("Prepare data")
       val assembler = new VectorAssembler()
         .setInputCols(cols)
         .setOutputCol("features")

       assembler.transform(dfFeatures).select("features", "label")
     }
     println(configs("test").toString)
     val test= spark.sqlContext.read.parquet(configs("test").toString)

     val featuresTestDf=if(test.columns.contains("label") && test.columns.contains("features")) test
     else{
     val cols = dfStart.columns.filter(_ != "class")
     val dfTest=test
       .withColumn("label",col("class").cast("Double"))
     val assemblerTest=new VectorAssembler()
       .setInputCols(cols)
       .setOutputCol("features")

      assemblerTest.transform(dfTest).select("features", "label")
     }
     (featureDf, featuresTestDf)
   }

}
