package Clasificadores

import Utils.Utils.{evaluate, prepareData}
import YamlConfig.LoadYaml.{getParams, parseYaml}
import org.apache.spark.SparkContext
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{col, lit, when}

import java.util
import scala.collection.JavaConverters._
import scala.collection.mutable

object Multiperceptron extends App {

  val configs=parseYaml(args(0))

  implicit val spark: SparkSession = SparkSession
    .builder()
    .appName(name = "Multiperceptron")
    .master(master = configs("cluster").toString)
    //.master("local[*]")
    .getOrCreate()


  val sc: SparkContext = spark.sparkContext
  sc.setLogLevel("ERROR")

  spark.conf.set("spark.sql.adaptive.enabled", "true")

  val (featureDf, dfTestFeatures)=prepareData(args, configs)

  println(featureDf.count())
  println(dfTestFeatures.count())
  featureDf.printSchema()
  dfTestFeatures.printSchema()

  val params: Map[String, Any] = getParams(configs("params"))
  println("Building model")


  val layers = params("layers").asInstanceOf[java.util.ArrayList[Int]].asScala.toArray

  // create the trainer and set its parameters
  val trainer = new MultilayerPerceptronClassifier()
    .setLayers(layers)
    .setStepSize(params("stepsize").asInstanceOf[Double])
    .setBlockSize(params("blockSize").asInstanceOf[Int])
    .setSeed(params("seed").asInstanceOf[Int])
    .setMaxIter(params("iter").asInstanceOf[Int])
  // train the model
  val model = trainer.fit(featureDf)

  // compute accuracy on the test set
  val predictionDf= model.transform(dfTestFeatures).select("prediction", "label")
  val scores=evaluate(predictionDf)

  print(scores)

}
