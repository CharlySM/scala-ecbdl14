package Clasificadores

import Utils.Utils.{evaluate, prepareData}
import YamlConfig.LoadYaml.{getParams, parseYaml}
import org.apache.spark.SparkContext
import org.apache.spark.ml.classification.GBTClassifier
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.{Column, SparkSession}
import org.apache.spark.sql.functions.{col, lit}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.rdd.RDD


object GradientBoostedTree extends App{

  implicit val spark: SparkSession = SparkSession
    .builder()
    .config("spark.driver.extraJavaOptions", "-Xss1024m")
    .config("spark.executor.extraJavaOptions", "-Xss1024m")
    .config("spark.memory.offHeap.enabled", true)
    .config("spark.memory.offHeap.size", "9g")
    .appName(name = "GradientBoost")
    .master(master = "spark://atlas:7077")
    //.master("local[*]")
    .getOrCreate()

  val configs=parseYaml(args(0))

  val sc: SparkContext = spark.sparkContext
  sc.setLogLevel("ERROR")

  spark.conf.set("spark.sql.adaptive.enabled", "true")

  val (featureDf, dfTestFeatures)=prepareData(args, configs)

  val params: Map[String, Any] = getParams(configs("params"))
  println("Building model")

  val model = new GBTClassifier()
    .setLabelCol("label")
    .setFeaturesCol("features")
    .setMaxIter(100)
    .setMaxDepth(10)
    .setFeatureSubsetStrategy("auto")
    .fit(featureDf)

  val predictionDf= model.transform(dfTestFeatures)

  val scores=evaluate(predictionDf)

  print(scores)
}
