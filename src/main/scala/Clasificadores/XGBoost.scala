package Clasificadores

import Utils.Utils.{evaluate, prepareData}
import YamlConfig.LoadYaml.{getParams, parseYaml}
import ml.dmlc.xgboost4j.scala.spark.XGBoostClassifier
import org.apache.spark.SparkContext
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.{Column, SparkSession}
import org.apache.spark.sql.functions.{col, lit}

object XGBoost extends App {

  val configs=parseYaml(args(0))

  implicit val spark: SparkSession = SparkSession
    .builder()
    .appName(name = "Preprocessing")
    .master(master = configs("cluster").toString)
    //.master("local[*]")
    .getOrCreate()

  val sc: SparkContext = spark.sparkContext
  sc.setLogLevel("ERROR")

  spark.conf.set("spark.sql.adaptive.enabled", "true")

  //val dfStart = spark.sqlContext.read.parquet(configs("dataset").toString)
  //val dfStart = spark.sqlContext.read.parquet("./src/main/resources/data/proteinasNormalized.parquet").limit(4000000).cache()

  val (featureDf, dfTestFeatures)=prepareData(args, configs)
  //featureDf.filter("label=1").show()
  //featureDf.filter("label=0").show()
  println(featureDf.filter("label=1").count())
  println(featureDf.filter("label=0").count())
  //val Array(trainingData, testData) = featureDf.randomSplit(Array(0.75, 0.25), seed = 23789L)

  val params: Map[String, Any] = getParams(configs("params"))

  val xgbParam = Map(
    "features_col" -> "features",
    "label_col" -> "label",
    "prediction_col" -> "prediction",
    "scale_pos_weight" -> params("scale_pos_weight").asInstanceOf[Double],
    "num_workers" -> params("num_workers").asInstanceOf[Int],
    "num_round" -> params.getOrElse("num_round", "30").asInstanceOf[Int])


  val xgbClassifier = new XGBoostClassifier(xgbParam)
  .setNumClass(2)//.setFeaturesCol("features").setLabelCol("label")
  val xgbClassificationModel = xgbClassifier.fit(featureDf)
  //dfTestFeatures.filter("label=1").show()
  //dfTestFeatures.filter("label=0").show()
  //println(dfTestFeatures.filter("label=1").count())
  //println(dfTestFeatures.filter("label=0").count())
  val results1 = xgbClassificationModel.transform(dfTestFeatures)
  // val results= results1.select("prediction", "label").rdd.map(r=>(r(0), r(1)))
  //println(results1.filter("prediction=1").count())
 // println(results1.filter("prediction=0").count())
 /* val tp=results1.select("prediction", "label").filter("label=1 and prediction=1.0").count().asInstanceOf[Double]
  val tn=results1.select("prediction", "label").filter("label=0 and prediction=0.0").count().asInstanceOf[Double]
  val fn=results1.select("prediction", "label").filter("label=1 and prediction=0.0").count().asInstanceOf[Double]
  val fp=results1.select("prediction", "label").filter("label=0 and prediction=1.0").count().asInstanceOf[Double]*/
 /* val metrics = new MulticlassMetrics(results)

  ///val matrix=metrics.confusionMatrix
  //println(metrics.confusionMatrix.toArray.mkString("Array(", ", ", ")"))

  //val (fp, tp) = (matrix.apply(0, 1), matrix.apply(1,1))
  //val (fn, tn)=(matrix.apply(1, 0), matrix.apply(0,0))*/
  /*println("tp", tp, " tn", tn, " fp", fp, " fn", fn)
  val TPR = tp/(tp+fn)
  val TNR = tn/(tn+fp)
  val score = TPR * TNR

  val scores = (TPR, TNR, score)*/
  val scores = evaluate(results1)

  print(scores)



}
