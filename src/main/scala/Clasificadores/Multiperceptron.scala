package Clasificadores

import Utils.Utils.prepareData
import YamlConfig.LoadYaml.{getParams, parseYaml}
import org.apache.spark.SparkContext
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.col

import java.util
import scala.collection.JavaConverters._
import scala.collection.mutable

object Multiperceptron extends App {

  implicit val spark: SparkSession = SparkSession
    .builder()
    .config("spark.driver.extraJavaOptions", "-Xss1024m")
    .config("spark.executor.extraJavaOptions", "-Xss1024m")
    .config("spark.memory.offHeap.enabled", value = true)
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


  val layers = params("layers").asInstanceOf[java.util.ArrayList[Int]].asScala.toArray

  // create the trainer and set its parameters
  val trainer = new MultilayerPerceptronClassifier()
    .setLayers(layers)
    .setBlockSize(params("blockSize").asInstanceOf[Int])
    .setSeed(params("seed").asInstanceOf[Int])
    .setMaxIter(params("iter").asInstanceOf[Int])

  // train the model
  val model = trainer.fit(featureDf.withColumn("label", col("label").cast("Int")))

  // compute accuracy on the test set
  val predictionDf= model.transform(dfTestFeatures).select("prediction", "label")
    val predictionRDD=predictionDf.rdd
    .map(r=>(r.getAs[Double]("prediction"), r.getAs[Double]("label")))

  val metrics = new MulticlassMetrics(predictionRDD)
  val evaluator=new MulticlassClassificationEvaluator()
  val metrics2=evaluator.getMetrics(predictionDf)//.//.evaluate(predictionDf)

  val (fp2, tp2)=(metrics2.falsePositiveRate(1.0), metrics2.truePositiveRate(1.0))
  val (fn2, tn2)=(1-fp2, 1-tp2)
  val TPR2 = tp2/(tp2+fn2)
  val TNR2 = tn2/(tn2+fp2)
  val score2 = TPR2 * TNR2
  val scores2 = (TPR2, TNR2, score2)

  print(scores2)

  val (fp, tp) = (metrics.falsePositiveRate(1.0), metrics.truePositiveRate(1.0))
  val (fn, tn)=(1-fp, 1-tp)

  val TPR = tp/(tp+fn)
  val TNR = tn/(tn+fp)
  val score = TPR * TNR

  val scores = (TPR, TNR, score)

  print(scores)

}
