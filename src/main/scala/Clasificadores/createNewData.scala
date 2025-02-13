package Clasificadores

import YamlConfig.LoadYaml.parseYaml
import org.apache.spark.SparkContext
import org.apache.spark.ml.classification.GBTClassifier
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.functions.{array, col, element_at, lit, rand, udf, when}
import org.apache.spark.sql.types.ArrayType
import org.apache.spark.sql.{SaveMode, SparkSession}
import org.apache.spark.ml.linalg.{SparseVector, Vector}
import org.apache.spark.mllib.linalg.{Vector => OldVector}

object CreateNewData extends App{

  private val vectorToArrayUdf = udf { vec: Any =>
    vec match {
      case v: Vector => v.toArray
      case v: OldVector => v.toArray
      case v => throw new IllegalArgumentException(
        "function vector_to_array requires a non-null input argument and input type must be " +
          "`org.apache.spark.ml.linalg.Vector` or `org.apache.spark.mllib.linalg.Vector`, " +
          s"but got ${ if (v == null) "null" else v.getClass.getName }.")
    }
  }.asNonNullable()

  val params=Array(Map("maxIter"->65, "depth"->7, "stepSize"->0.4),
    Map("maxIter"->70, "depth"->7, "stepSize"->0.4),
    Map("maxIter"->100, "depth"->6, "stepSize"->0.4),
    Map("maxIter"->90, "depth"->7, "stepSize"->0.4),
    Map("maxIter"->95, "depth"->8, "stepSize"->0.4))

  implicit val spark: SparkSession = SparkSession
    .builder()
    .config("spark.driver.extraJavaOptions", "-Xss1024m")
    .config("spark.executor.extraJavaOptions", "-Xss1024m")
    .config("spark.memory.offHeap.enabled", true)
    .config("spark.memory.offHeap.size", "9g")
    .appName(name = "GradientBoost")
    .master(master = "spark://atlas:7077")
    //.master("local[*]")
    .getOrCreate()

  val configs=parseYaml(args(0))

  val sc: SparkContext = spark.sparkContext
  sc.setLogLevel("ERROR")

  spark.conf.set("spark.sql.adaptive.enabled", "true")

  val dataset=spark.sqlContext.read.parquet(configs("datasetTrain").toString)
  //.withColumn("label", when(col("label")===lit(0.0), lit(-1)).otherwise(col("label")))//.withColumnRenamed("class", "label")
  //var dfTest=spark.sqlContext.read.parquet(configs("datasetTest").toString)//.withColumnRenamed("class", "label")

  var Array(dfTrain, dfTest)=dataset.randomSplit(Array(0.3, 0.7), 123456789)
  Range(0, 30).foreach(i=>{
    println("iteracion:", i)
    val model = new GBTClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setMaxIter(params((i%5))("maxIter").asInstanceOf[Int])
      .setMaxDepth(params((i%5))("depth").asInstanceOf[Int])
      .setFeatureSubsetStrategy("auto")
      .setStepSize(params((i%5))("stepSize").asInstanceOf[Double])
      .fit(dfTrain)

    var predictionDf= model.transform(dfTest)

    predictionDf=predictionDf.withColumn("probability",vectorToArrayUdf(col("probability")))
    dfTest=predictionDf.filter((element_at(col("probability"), 1)>lit(0.6) and col("label")===lit(0)) or (element_at(col("probability"), 2)>lit(0.6) and col("label")===1))
     // .withColumn(s"label$i", when(col("label") === lit(0), col("prediction")).otherwise((col("probability").getItem(2))))
      .withColumnRenamed("prediction", s"label$i")
      .drop("rawPrediction", "probability", "prediction")

  })
  println("termina entrenamiento")
  dfTest=dfTest.drop("features")
  val cols=dfTest.columns.filter(_!="label")
  //val assembler = new VectorAssembler()
    //.setInputCols(cols)
   // .setOutputCol("features")

  dfTest=dfTest.withColumn("features", array(cols.map(col):_*))

  //dfTest=assembler.transform(dfTest).select("features", "label")
  dfTest.show(50,truncate = false)
  dfTest.write.mode(SaveMode.Overwrite).parquet(configs("output").toString)

}
