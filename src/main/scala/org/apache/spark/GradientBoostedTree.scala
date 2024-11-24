package org.apache.spark

import YamlConfig.LoadYaml.parseYaml
import org.apache.spark.ml.classification.GBTClassifier
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.{Column, SparkSession}
import org.apache.spark.sql.functions.{col, lit}

object GradientBoostedTree extends App{

  val spark: SparkSession = SparkSession
    .builder()
    .appName(name = "Preprocessing")
    .master(master = "spark://atlas:7077")
  //.master("local[*]")
    .getOrCreate()

  val configs=parseYaml(args(0))


  val sc: SparkContext = spark.sparkContext
  sc.setLogLevel("ERROR")

  spark.conf.set("spark.sql.adaptive.enabled", "true")

  //val dfStart = spark.sqlContext.read.parquet(configs("dataset").toString)
  //val dfStart = spark.sqlContext.read.parquet("./src/main/resources/data/proteinasNormalized.parquet").limit(4000000).cache()

  val dfStart=if(args(1)=="1") {
    val dfAux=spark.sqlContext.read.parquet(configs("dataset").toString)
    val lisCols = dfAux.columns.map(i => (i, col(i).cast("Double"))).toMap[String, Column]
    dfAux.withColumns(lisCols)

    //balancedDF(dfRes)
  }
  else spark.sqlContext.read.parquet(configs("dataset").toString)
  println(dfStart.count())

  val cols=dfStart.columns.filter(_!="class")
  val mapCols=cols.map(i=> (i, col(i)+lit(10))).toMap
  val dfStartModified=dfStart.withColumns(mapCols)

  val dfFeatures=dfStartModified
    .withColumn("label", col("class"))

  val assembler=new VectorAssembler()
    .setInputCols(cols)
    .setOutputCol("features")

  val featureDf = assembler.transform(dfFeatures)

  val Array(trainingData, testData) = featureDf.randomSplit(Array(0.75, 0.25), seed = 23789L)

  val model = new GBTClassifier()
    .setLabelCol("label")
    .setFeaturesCol("features")
    .setMaxIter(10)
    .setFeatureSubsetStrategy("auto")
    .fit(trainingData)

  val predictionDf = model.transform(testData)
  val evaluator = new BinaryClassificationEvaluator()
    .setLabelCol("label")
    .setMetricName("areaUnderROC")

  val accuracy = evaluator.evaluate(predictionDf)
  println(accuracy)

}
