package org.apache.spark

import YamlConfig.LoadYaml.parseYaml
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.functions.{col, lit}
import org.apache.spark.sql.{Column, DataFrame, SparkSession}

object RandomForest extends App{

  def balancedDF(df:DataFrame): DataFrame = {
    val dfP=df.filter("class=1")
    val dfN=df.filter("class=0")
    val nP=dfP.count()
    val nN=dfN.count()
    val ratio = nP/nN
    val dfNSample=dfN.sample(false, ratio)
    dfP.unionAll(dfNSample)
  }

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

  println("Reading data")
  //val dfStart = spark.sqlContext.read.parquet(configs("dataset").toString)
 // val dfStart = spark.sqlContext.read.parquet("./src/main/resources/data/treatedProteinasHME_BD.parquet").limit(400000).cache()

  val dfStart=if(args(1)=="1") {
    val dfAux=spark.sqlContext.read.parquet(configs("dataset").toString)
    val lisCols = dfAux.columns.map(i => (i, col(i).cast("Double"))).toMap[String, Column]
    dfAux.withColumns(lisCols)

    //balancedDF(dfRes)
  }
  else spark.sqlContext.read.parquet(configs("dataset").toString)
  println(dfStart.count())
  val cols=dfStart.columns.filter(_!="class")
  val mapCols=cols.map(i=> (i, col(i)+lit(10))).toMap
  val dfStartModified=dfStart.withColumns(mapCols)

  val dfFeatures=dfStartModified
    .withColumn("label", col("class"))
  println("Prepare data")
  val assembler=new VectorAssembler()
    .setInputCols(cols)
    .setOutputCol("features")

  val featureDf = assembler.transform(dfFeatures)

  val Array(trainingData, testData) = featureDf.randomSplit(Array(0.8, 0.2), seed = 2457843L)

  println("Building model")
  val model = new RandomForestClassifier()
    .setImpurity("gini")
    .setMaxDepth(5)
    .setNumTrees(50)
    .setFeatureSubsetStrategy("auto")
    .setSeed(23789L)
    .fit(trainingData)

  println("Prediting data")
  val predictionDf = model.transform(testData)

  val evaluator = new BinaryClassificationEvaluator()
    .setLabelCol("label")
    .setMetricName("areaUnderROC")
  println("Evaluating data")
  val accuracy = evaluator.evaluate(predictionDf)
  println(accuracy)


}
