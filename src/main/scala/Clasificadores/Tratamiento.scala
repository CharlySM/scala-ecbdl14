package Clasificadores

import Clasificadores.Tratamiento.f1
import Utils.Utils.{evaluate, prepareData}
import YamlConfig.LoadYaml.{getParams, parseYaml}
import org.apache.spark.SparkContext
import org.apache.spark.ml.classification.{GBTClassifier, RandomForestClassifier}
import org.apache.spark.ml.feature.{ChiSqSelector, MaxAbsScaler, VectorAssembler}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.mllib.linalg.{Vector => OldVector}
import org.apache.spark.sql.{Column, DataFrame, Dataset, Row, SaveMode, SparkSession}
import org.apache.spark.sql.functions.{col, lit, rand, udf, when}

import scala.Function.chain
import scala.collection.immutable

object Tratamiento extends App{

  val list0 = immutable.Seq.range(0,30).map(i=>(s"label$i" -> when(rand() >= lit(0.3), lit(0)).otherwise(1))).toMap
  val list1= immutable.Seq.range(0,30).map(i=>(s"label$i" -> when(rand() >= lit(0.3), lit(1)).otherwise(0))).toMap

  def f1()(df:DataFrame): DataFrame = {
    df.withColumns(list0)
      .withColumns(list1)
  }

  implicit val spark: SparkSession = SparkSession
    .builder()
    .config("spark.driver.extraJavaOptions", "-Xss1024m")
    .config("spark.executor.extraJavaOptions", "-Xss1024m")
    .config("spark.memory.offHeap.enabled", true)
    .config("spark.memory.offHeap.size", "20g")
    .appName(name = "GradientBoost")
    .master(master = "spark://atlas:7077")
    //.master("local[*]")
    .getOrCreate()

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

  val configs=parseYaml(args(0))

  val sc: SparkContext = spark.sparkContext
  sc.setLogLevel("ERROR")

  spark.conf.set("spark.sql.adaptive.enabled", "true")

  val featureDf=spark.sqlContext.read.parquet(configs("dataset").toString).cache()//prepareData(args, configs)

  //val params: Map[String, Any] = getParams(configs("params"))
  println("Building model")

  val vecArr=Range(0, 30).map(i=>s"label$i")
  val sqlExpr= vecArr.zipWithIndex.map{ case (alias, idx) => vectorToArrayUdf(col("features")).getItem(idx).as(alias) }

  val res=featureDf.select(sqlExpr:+col("label"):_*)

  //val listF1=Range(0,2000).map(i=>f1()(_))
  //val chained=chain(listF1)
  val dataset= f1()(res)
  val cols=dataset.columns.filter(_!="label")
  val assembler = new VectorAssembler()
    .setInputCols(cols)
    .setOutputCol("features")

  val datasetFinal=assembler.transform(dataset).select("features", "label")
  datasetFinal.write.mode(SaveMode.Overwrite).parquet(configs("test").toString)
  // res.foreach(_.show())
}
