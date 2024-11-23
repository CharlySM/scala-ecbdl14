package org.apache

import YamlConfig.LoadYaml.parseYaml
import org.apache.spark.Main.args
import org.apache.spark.SparkContext
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, VectorAssembler}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{col, lit}

object RandomForest extends App{

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

  val dfStart = spark.sqlContext.read.parquet(configs("dataset").toString)


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

  val model = new RandomForestClassifier()
    .setImpurity("gini")
    .setMaxDepth(3)
    .setNumTrees(20)
    .setFeatureSubsetStrategy("auto")
    .setSeed(23789L)
    .fit(trainingData)

  val predictionDf = model.transform(testData)

  val evaluator = new BinaryClassificationEvaluator()
    .setLabelCol("label")
    .setMetricName("areaUnderROC")

  val accuracy = evaluator.evaluate(predictionDf)
  println(accuracy)


}
