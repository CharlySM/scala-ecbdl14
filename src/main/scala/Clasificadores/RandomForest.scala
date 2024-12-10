package Clasificadores

import Utils.Utils.prepareData
import YamlConfig.LoadYaml.{getParams, parseYaml}
import org.apache.spark.SparkContext
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.functions.{col, lit}
import org.apache.spark.sql.{Column, DataFrame, SparkSession}

object RandomForest extends App{

  def balancedDF(df:DataFrame): DataFrame = {
    val dfP=df.filter("class=1")
    val dfN=df.filter("class=0")
    val nP=dfP.count().toDouble
    val nN=dfN.count().toDouble
    val major= if(nP>nN) nP else nN
    val minor=if(nP<nN) nP else nN
    val dfMinor=if(nP<nN) dfP else dfN
    val ratio = minor/major
    val dfNSample=dfN.sample(withReplacement = true, ratio)
    dfMinor.unionAll(dfNSample)
  }

  implicit val spark: SparkSession = SparkSession
    .builder()
    .config("spark.driver.extraJavaOptions", "-Xss1024m")
    .config("spark.executor.extraJavaOptions", "-Xss1024m")
    .config("spark.memory.offHeap.enabled", true)
    .config("spark.memory.offHeap.size", "9g")
    .appName(name = "Random forest")
    .master(master = "spark://atlas:7077")
    //.master("local[*]")
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
  val predictionDf = model.transform(dfTestFeatures).select("prediction", "label")//.rdd.map(r=>(r(0), r(1)))
   val metrics = new MulticlassMetrics(predictionDf.rdd.map(r=>(r(0), r(1))))

  val evaluator=new MulticlassClassificationEvaluator()
  metrics.confusionMatrix.toArray

  val (fp, tp) = Tuple2(metrics.falsePositiveRate(1.0), metrics.truePositiveRate(1.0))
  val (fn, tn)=(1-fp, 1-tp)

  val TPR = tp/(tp+fn)
  val TNR = tn/(tn+fp)
  val score = TPR * TNR

  val scores = (TPR, TNR, score)

  print(scores)

}
