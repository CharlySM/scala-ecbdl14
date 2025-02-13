import YamlConfig.LoadYaml.{getParams, parseYaml}
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.col

import scala.collection.JavaConverters._



object MiPrueba2 extends App{

  //val configs=parseYaml(args(0))
  val spark: SparkSession = SparkSession
    .builder()
    .config("spark.driver.extraJavaOptions", "-Xss1024m")
    .config("spark.executor.extraJavaOptions", "-Xss1024m")
    .config("spark.memory.offHeap.enabled", true)
    .config("spark.memory.offHeap.size", "9g")
    .appName(name = "Processing")
    .master(master = "local[*]")
    .getOrCreate()

  spark.conf.set("spark.sql.adaptive.enabled", "true")

  val df=spark.sqlContext.read.option("header", true)
    .option("Sep", "|")
    .option("inferSchema", true)
    .csv("src/main/resources/data.csv").drop("layers")

  val cols = df.columns.filter(_ != "class")

  val dfFeatures = df
    .withColumn("label", col("media").cast("Double"))

  df.show()
  df.printSchema()

  println("idexer data")

  //val indexer=new StringIndexer().setInputCol("layers").setOutputCol("layerIndexer")
  //val dfIndexer=indexer.fit(dfFeatures).transform(dfFeatures).withColumn("layers", col("layerIndexer")).drop("layerIndexer")

  println("assamble data")
  val assembler = new VectorAssembler()
    .setInputCols(cols)
    .setOutputCol("features")

  val assenblerDf=assembler.transform(dfFeatures).select("features", "label")
  println("params of data")
  val configs=parseYaml(args(0))


  val params: Map[String, Any] = getParams(configs("params"))

  val layers = params("layers").asInstanceOf[java.util.ArrayList[Int]].asScala.toArray

  println("train model data")
  val trainer = new MultilayerPerceptronClassifier()
    .setLayers(layers)
    .setStepSize(params("stepsize").asInstanceOf[Double])
    .setBlockSize(params("blockSize").asInstanceOf[Int])
    .setSeed(params("seed").asInstanceOf[Int])
    .setMaxIter(params("iter").asInstanceOf[Int])

  val model=trainer.fit(assenblerDf)




}
