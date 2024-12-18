package Clasificadores

import YamlConfig.LoadYaml.parseYaml
import org.apache.spark.SparkContext
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{col, lit}

/**
 * @author ${user.name}
 */
object Main extends App {

  val spark: SparkSession = SparkSession
    .builder()
    .appName(name = "Preprocessing")
    .master(master = "spark://atlas:7077")
    //.master("local[*]")
    .getOrCreate()

  val sc: SparkContext = spark.sparkContext
  spark.sparkContext.setLogLevel("ERROR")

  // spark.conf.set("spark.sql.shuffle.partitions", 1120) // Adjusto to avoid OOM
  spark.conf.set("spark.sql.adaptive.enabled", "true")
  //spark.conf.set(SQLConf.CONSTRAINT_PROPAGATION_ENABLED.key, value = false)

  //val path = "hdfs://atlas:9000/user/djgarcia/SUSY"
  val configs=parseYaml(args(0))

  ///val selectCols=spark.sqlContext.read.parquet("hdfs://atlas:9000/user/carsan/proteinasNormalized.parquet").columns
  //val schema=spark.sqlContext.read.parquet("hdfs://atlas:9000/user/carsan/proteinasNormalized.parquet").schema
  val dfStart = spark.sqlContext.read.parquet("hdfs://atlas:9000/user/carsan/proteinasNormalized.parquet")


  val cols=dfStart.columns.filter(_!="class")
  val mapCols=cols.map(i=> (i, col(i)+lit(10))).toMap
  val dfStartModified=dfStart.withColumns(mapCols)

  val dfFeatures=dfStartModified
    .withColumn("label", col("class"))

  val assembler=new VectorAssembler()
    .setInputCols(cols)
    .setOutputCol("features")

  val featureDf = assembler.transform(dfFeatures)
   // .select(selectCols.head, selectCols.tail:_*) // SUSY only has 18 blocks

  val Array(trainingData, testData) = featureDf.randomSplit(Array(0.7, 0.3), seed = 1234L)

  val model = new NaiveBayes()
    .fit(trainingData)

  val predictions = model.transform(testData)

  val evaluator = new MulticlassClassificationEvaluator()
    .setLabelCol("label")
    .setPredictionCol("prediction")
    .setMetricName("accuracy")

  val accuracy = evaluator.evaluate(predictions)
  predictions.show()
  println(s"Test set accuracy = $accuracy")
}
