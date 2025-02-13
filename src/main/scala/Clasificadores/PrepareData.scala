package Clasificadores

import Utils.Utils.{prepareData, random_oversampling}
import YamlConfig.LoadYaml.{getParams, parseYaml}
import org.apache.spark.SparkContext
import org.apache.spark.ml.feature.{ChiSqSelector, MaxAbsScaler}
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}


object PrepareData extends App{

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

  val (featureDf1, dfTestFeatures)=prepareData(args, configs)
  val featureDf: DataFrame =random_oversampling(featureDf1, 1.5)
  val params: Map[String, Any] = getParams(configs("params"))
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
    .transform(dfTestFeaturesScaler).selectExpr("selectedFeatures as features", "label")

  featureDfSelected.write.mode(SaveMode.Overwrite).parquet(configs("trainOut").toString)
  testSelected.write.mode(SaveMode.Overwrite).parquet(configs("testOut").toString)


}
