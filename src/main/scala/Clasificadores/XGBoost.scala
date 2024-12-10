package Clasificadores

import YamlConfig.LoadYaml.parseYaml
import ml.dmlc.xgboost4j.scala.spark.XGBoostClassifier
import org.apache.spark.SparkContext
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.{Column, SparkSession}
import org.apache.spark.sql.functions.{col, lit}

object XGBoost extends App {

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

  //val dfStart = spark.sqlContext.read.parquet(configs("dataset").toString)
  //val dfStart = spark.sqlContext.read.parquet("./src/main/resources/data/proteinasNormalized.parquet").limit(4000000).cache()

  val dfStart=if(args(1)=="1") {
    val dfAux=spark.sqlContext.read.parquet(configs("dataset").toString)
    val lisCols = dfAux.columns.map(i => (i, col(i).cast("Double"))).toMap[String, Column]
    dfAux.withColumns(lisCols)

    //balancedDF(dfRes)
  }
  else spark.sqlContext.read.parquet(configs("dataset").toString)
  println(dfStart.count())

  val test=spark.sqlContext.read.parquet(configs("test").toString)

  val cols=dfStart.columns.filter(_!="class")
  val mapCols=cols.map(i=> (i, col(i))).toMap
  val dfStartModified=dfStart.withColumns(mapCols)

  val dfFeatures=dfStartModified
    .withColumn("label", col("class"))

  val assembler=new VectorAssembler()
    .setInputCols(cols)
    .setOutputCol("features")

  val featureDf = assembler.transform(dfFeatures)

  val colsTest=test.columns.filter(_!="class")
  val mapColsTest=colsTest.map(i=> (i, col(i))).toMap
  val dfStartModifiedTest=test.withColumns(mapColsTest)

  val dfTest=dfStartModifiedTest
    .withColumn("label", col("class"))

  val assemblerTest=new VectorAssembler()
    .setInputCols(colsTest)
    .setOutputCol("features")

  val dfTestFeatures = assemblerTest.transform(dfTest)
  test
  //val Array(trainingData, testData) = featureDf.randomSplit(Array(0.75, 0.25), seed = 23789L)


  val xgbParam = Map(
    "features_col" -> "features",
    "label_col" -> "label",
    "prediction_col" -> "prediction",
    "scale_pos_weight" -> 1.0,
    "num_workers" -> 31,
  "eval_metric" -> "auc")


  val xgbClassifier = new XGBoostClassifier(xgbParam).setFeaturesCol("features").setLabelCol("label")
  val xgbClassificationModel = xgbClassifier.fit(featureDf)

  val results = xgbClassificationModel.transform(dfTestFeatures).select("prediction", "label").rdd.map(r=>(r(0), r(1)))

  val metrics = new MulticlassMetrics(results)

  val (fp, tp) = (metrics.falsePositiveRate(1.0), metrics.truePositiveRate(1.0))
  val (fn, tn)=(1-fp, 1-tp)

  val TPR = tp/(tp+fn)
  val TNR = tn/(tn+fp)
  val score = TPR * TNR

  val scores = (TPR, TNR, score)

  print(scores)



}
