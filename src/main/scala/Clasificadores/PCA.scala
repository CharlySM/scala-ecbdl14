package Clasificadores

import Utils.Utils.prepareData
import YamlConfig.LoadYaml.{getParams, parseYaml}
import org.apache.spark.SparkContext
import org.apache.spark.ml.feature.PCA
import org.apache.spark.sql.{SaveMode, SparkSession}

object PCA extends App {
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


  val pca = new PCA()
    .setInputCol("features")
    .setOutputCol("pcaFeatures")
    .setK(params("k").asInstanceOf[Int])
    .fit(featureDf)

  val pcafeatures = pca.transform(featureDf).select("label","pcaFeatures").withColumnRenamed("pcaFeatures", "features")
  val pcaTest = pca.transform(dfTestFeatures).select("label","pcaFeatures").withColumnRenamed("pcaFeatures", "features")

  pcafeatures.show(false)
  pcaTest.show(false)

  println("Write data")
  //result.write.mode(SaveMode.Overwrite).parquet(configs("output").toString)


  /*val metrics = new MulticlassMetrics(result.rdd.map(r=>(r(0), r(1))))

  println(metrics.confusionMatrix.toArray.mkString("Array(", ", ", ")"))
  println(metrics.confusionMatrix)
  val matrix=metrics.confusionMatrix

  val (fp, tp) = (matrix.apply(0, 1), matrix.apply(1,1))
  val (fn, tn)=(matrix.apply(1, 0), matrix.apply(0,0))

  val TPR = tp/(tp+fn)
  val TNR = tn/(tn+fp)
  val score = TPR * TNR

  val scores = (TPR, TNR, score)

  print(scores)*/

}
