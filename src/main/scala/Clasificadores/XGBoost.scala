package Clasificadores

import Utils.Utils.prepareData
import YamlConfig.LoadYaml.{getParams, parseYaml}
import ml.dmlc.xgboost4j.scala.spark.XGBoostClassifier
import org.apache.spark.SparkContext
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.{Column, SparkSession}
import org.apache.spark.sql.functions.{col, lit}

object XGBoost extends App {

  implicit val spark: SparkSession = SparkSession
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

  val (featureDf, dfTestFeatures)=prepareData(args, configs)
  featureDf.filter("label=1").show()
  featureDf.filter("label=0").show()
  //val Array(trainingData, testData) = featureDf.randomSplit(Array(0.75, 0.25), seed = 23789L)

  val params: Map[String, Any] = getParams(configs("params"))

  val xgbParam = Map(
    "features_col" -> "features",
    "label_col" -> "label",
    "prediction_col" -> "prediction",
    "scale_pos_weight" -> params("scale_pos_weight").asInstanceOf[Double],
    "num_workers" -> params("num_workers").asInstanceOf[Int],
    "num_round" -> params("num_round").asInstanceOf[Int])


  val xgbClassifier = new XGBoostClassifier(xgbParam).setFeaturesCol("features").setLabelCol("label")
  val xgbClassificationModel = xgbClassifier.fit(featureDf)
  dfTestFeatures.filter("label=1").show()
  dfTestFeatures.filter("label=0").show()
  val results1 = xgbClassificationModel.transform(dfTestFeatures)
   val results= results1.select("prediction", "label").rdd.map(r=>(r(0), r(1)))
  results1.show()
  val metrics = new MulticlassMetrics(results)
  val matrix=metrics.confusionMatrix
  println(metrics.confusionMatrix.toArray.mkString("Array(", ", ", ")"))

  val (fp, tp) = (matrix.apply(0, 1), matrix.apply(1,1))
  val (fn, tn)=(matrix.apply(1, 0), matrix.apply(0,0))

  val TPR = tp/(tp+fn)
  val TNR = tn/(tn+fp)
  val score = TPR * TNR

  val scores = (TPR, TNR, score)

  print(scores)



}
