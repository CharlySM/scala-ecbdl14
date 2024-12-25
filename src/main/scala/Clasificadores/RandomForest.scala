package Clasificadores

import Utils.Utils.prepareData
import YamlConfig.LoadYaml.{getParams, parseYaml}
import org.apache.spark.SparkContext
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions.{col, lit}
import org.apache.spark.sql.{Column, DataFrame, SparkSession}

object RandomForest extends App{



  implicit val spark: SparkSession = SparkSession
    .builder()
    .config("spark.driver.extraJavaOptions", "-Xss1024m")
    .config("spark.executor.extraJavaOptions", "-Xss1024m")
    .config("spark.memory.offHeap.enabled", true)
    .config("spark.memory.offHeap.size", "9g")
    .appName(name = "Random forest")
    //.master(master = "spark://atlas:7077")
    .master("local[*]")
    .getOrCreate()

  val configs: Map[String, Any] =parseYaml(args(0))

  val sc: SparkContext = spark.sparkContext
  sc.setLogLevel("ERROR")

  spark.conf.set("spark.sql.adaptive.enabled", "true")

  println("Reading data")

  val (featureDf, dfTestFeatures)=prepareData(args, configs)

  val params: Map[String, Any] = getParams(configs("params"))
  println("Building model")
  val model = new RandomForestClassifier()
    .setImpurity(params.getOrElse("impurity", "gini").asInstanceOf[String])
    .setMaxDepth(params.getOrElse("maxDepth", 0).asInstanceOf[Int])
    .setNumTrees(params.getOrElse("nTrees", 0).asInstanceOf[Int])
    .setFeatureSubsetStrategy(params.getOrElse("strategy", 0).asInstanceOf[String])
    .setSeed(params.getOrElse("seed", 0).asInstanceOf[Int])
    .fit(featureDf)

  println("Prediting data")
  val result = model.transform(dfTestFeatures).select("prediction", "label").rdd.map(r=>(r(0), r(1)))
  println(result.first())
  val metrics = new MulticlassMetrics(result)

  println(metrics.confusionMatrix.toArray.mkString("Array(", ", ", ")"))
  println(metrics.confusionMatrix)
  val matrix=metrics.confusionMatrix

  val (fp, tp) = (matrix.apply(0, 1), matrix.apply(1,1))
  val (fn, tn)=(matrix.apply(1, 0), matrix.apply(0,0))

  val TPR = tp/(tp+fn)
  val TNR = tn/(tn+fp)
  val score = TPR * TNR

  val scores = (TPR, TNR, score)

  print(scores)

}
