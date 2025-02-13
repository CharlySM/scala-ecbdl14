package Clasificadores

import Clasificadores.Noise.randomForest
import Utils.Utils.{evaluate, prepareData, random_oversampling}
import YamlConfig.LoadYaml.{getParams, parseYaml}
import org.apache.spark.SparkContext
import org.apache.spark.ml.classification.{GBTClassifier, RandomForestClassifier}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{ChiSqSelector, MaxAbsScaler, MinMaxScaler, PCA, VectorAssembler}
import org.apache.spark.ml.linalg
import org.apache.spark.sql.{Column, DataFrame, SparkSession}
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
  val params: Map[String, Any] = getParams(configs("params"))
  val (featureDf1, dfTestFeatures)=prepareData(args, configs)
  /*val featureDf: DataFrame =random_oversampling(featureDf1, 1.5)

  println("Building model")

  println("Inicio Max Abs Scaler")
  val scaler = new MaxAbsScaler()
    .setInputCol("features")
    .setOutputCol("scaledFeatures")

  val scalerModel = scaler.fit(featureDf)
  // rescale each feature to range [-1, 1]
  val featureDfScaler = scalerModel.transform(featureDf)

  val scalerModelTest = scaler.fit(dfTestFeatures)
  // rescale each feature to range [-1, 1]
  val dfTestFeaturesScaler = scalerModelTest.transform(dfTestFeatures)

  val selector = new ChiSqSelector()
    .setPercentile(params("percentil").asInstanceOf[Double])
    //.setFdr(params("percentil").asInstanceOf[Double])
    .setFeaturesCol("features")
    .setLabelCol("label")
    .setOutputCol("selectedFeatures")

  val modelChi = selector.fit(featureDfScaler)

  val featureDfSelected = modelChi
    .transform(featureDfScaler).selectExpr("selectedFeatures as features", "label")

  val testSelected=modelChi
    .transform(dfTestFeaturesScaler).selectExpr("selectedFeatures as features", "label")*/
  /*val pca = new PCA()
    .setInputCol("features")
    .setOutputCol("pcaFeatures")
    .setK(params("k").asInstanceOf[Int])
    .fit(featureDf)

  val pcafeatures = pca.transform(featureDf).select("label","pcaFeatures").withColumnRenamed("pcaFeatures", "features")
  val pcaTest = pca.transform(dfTestFeatures).select("label","pcaFeatures").withColumnRenamed("pcaFeatures", "features")
*/

  println("Inicio gradient boost tree")
  val model = new GBTClassifier()
    .setLabelCol("label")
    .setFeaturesCol("features")
    .setMaxIter(params("maxIter").asInstanceOf[Int])
    .setMaxDepth(params("depth").asInstanceOf[Int])
    .setFeatureSubsetStrategy("auto")
    .setStepSize(params("stepsize").asInstanceOf[Double])
    .fit(featureDf1)
  val predictionDf= model.transform(dfTestFeatures)

  val scores=evaluate(predictionDf)

  print(scores)
}
