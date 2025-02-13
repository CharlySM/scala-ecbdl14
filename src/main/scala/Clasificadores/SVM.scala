package Clasificadores

import Utils.Utils.{evaluate, prepareData}
import YamlConfig.LoadYaml.{getParams, parseYaml}
import org.apache.spark.SparkContext
import org.apache.spark.ml.classification.{FMClassifier, LinearSVC}
import org.apache.spark.sql.SparkSession

object SVM extends App {

  implicit val spark: SparkSession = SparkSession
    .builder()
    .config("spark.driver.extraJavaOptions", "-Xss1024m")
    .config("spark.executor.extraJavaOptions", "-Xss1024m")
    .config("spark.memory.offHeap.enabled", true)
    .config("spark.memory.offHeap.size", "9g")
    .appName(name = "SVM")
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

  val lsvc = new FMClassifier()
    .setMaxIter(params("maxiter").asInstanceOf[Int])
    .setRegParam(params("regParam").asInstanceOf[Double])

  // Fit the model
  val lsvcModel = lsvc.fit(featureDf)

  val predict = lsvcModel.transform(dfTestFeatures).select("label", "prediction")

  val scores=evaluate(predict)

  print(scores)

}
