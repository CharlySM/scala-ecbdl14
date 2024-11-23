package org.apache.spark

import YamlConfig.LoadYaml.parseYaml
import org.apache.spark.Main.args
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{Column, DataFrame, SaveMode, SparkSession, functions}

import scala.Function.chain
import scala.collection.immutable
import scala.util.Random




object Processing extends App {

  def balancedDF(df:DataFrame): DataFrame = {
    val dfP=df.filter("class=1")
    val dfN=df.filter("class=0")
    val nP=dfP.count()
    val nN=dfN.count()
    val ratio = nP/nN
    val dfNSample=dfN.sample(false, ratio)
    dfP.unionAll(dfNSample)
  }

  def initializingWeight(df:DataFrame, columns:Array[String]): DataFrame = {
    val dictWeights=columns.map(i=> (i, struct(rand().alias("weight"), col(i).alias("value")))).toMap
    df.withColumns(dictWeights)
  }

  def calculate(df: DataFrame, columns: Array[String]): DataFrame = {
    val dictSum = columns.map(i => (i, struct(col(s"$i.weight").alias("weight"), col(s"$i.value").alias("value"),
      (col(s"$i.weight") * col(s"$i.value")).alias("res")))).toMap

    val sum = columns.map(i => col(s"$i.res")).reduce(_ + _)
    val sumTotal = columns.map(i => col(s"$i.value")).reduce(_ + _)

    df.withColumns(dictSum)
      .withColumn("sum", sum)
      .withColumn("sumTotal", sumTotal)

  }

  def getRandom= (n: Float, m:Float) => {
    val rand=scala.util.Random.nextFloat()
    val max=m
    val min=n
    val range = max-min
    val adjustment=range*rand
    min+adjustment

  }

  def f1()(df:DataFrame): DataFrame = {
    df.withColumns(dictUpdateWeights)
  }

  import org.apache.spark.sql.functions.udf
  val getRandomNumber = udf(getRandom)

  val spark: SparkSession = SparkSession
    .builder()
    .appName(name = "Preprocessing")
    .master(master = "spark://atlas:7077")
    //.master("local[*]")
    .getOrCreate()

  val sc: SparkContext = spark.sparkContext
  spark.sparkContext.setLogLevel("ERROR")

  spark.conf.set("spark.sql.adaptive.enabled", "true")

  println("Reading data")
  val dfStart = spark.sqlContext.read.parquet("hdfs://atlas:9000/user/carsan/proteinasNormalized.parquet").cache()

  //val dfStart = spark.sqlContext.read.parquet("./src/main/resources/proteinasNormalized.parquet").limit(1000).cache()
  println("Balancing data")
  val dfBalanced=balancedDF(dfStart)

  val columns=dfBalanced.columns.filter(_!="class")

  val dfWeights=initializingWeight(dfBalanced, columns)
    .withColumn("sum", lit(0))
    .withColumn("sumTotal", lit(0))

  val dfCalculated=calculate(dfWeights, columns)
    .withColumn("total", col("sumTotal")*col("class"))

  val maxim=dfCalculated.filter("total>0").select(max(col("sumTotal")).alias("MAX")).first().getDouble(0).toFloat

  val corte: Float=maxim/columns.length.toFloat

  val dictCols=columns.map(i=>(i, col(i).withField("max", lit(1)).withField("min", lit(0)))).toMap
  val dfPrepared=dfCalculated.withColumns(dictCols)
  println("Evolutive  transformations")
  val dictUpdateWeights=columns.map(c=> (c, col(c).withField("max", when(col(c + ".res") > lit(corte), col(c + ".weight")).otherwise(
      col(c + ".max"))).withField("min", when(col(c + ".res") < lit(corte), col(c + ".weight"))
      .otherwise(col(c + ".min"))).withField("weight", getRandomNumber(col(c + ".min"), col(c + ".max")))
    .withField("res", col(c + ".weight") * col(c + ".value")))).toMap

  val listF1=immutable.Seq.range(0,3000).map(i=>f1()(_))
  val chained=chain(listF1)
  val dfWeightsCalculated=dfPrepared.transform(chained).cache()

  val listCols=columns.map(c=>struct(col(s"${c}.weight"), col(s"${c}.value")).alias(c) )

  val dfSumaFinal = dfWeightsCalculated.select(listCols:_*).cache()

  val listProds=columns.map(i=>sum(col(s"$i.weight")).alias(i+"_sum"))

  val prods=dfSumaFinal.select(listProds:_*)
  val row=prods.first()
  val mapsRow=row.getValuesMap[Double](row.schema.fieldNames)
  val selectColumns=mapsRow.toSeq.sortWith(_._2 > _._2).toMap.take(100)

  val colM: Column =columns.map(i=>col(i+".weight")).reduce((x, y) => x+y)

  val dfProd=dfSumaFinal.withColumn("valueRow", colM)

  val min_value = dfProd.select(min(col("valueRow")).alias("MIN")).first().getFloat(0)
  val avg_value = dfProd.select(avg(col("valueRow")).alias("MAX")).first().getDouble(0)
  println("Select columns and instances")
  val columnsSelected=selectColumns.map(i=>s"`${i._1.replace("_sum", "")}`.value as `${i._1.replace("_sum", "")}`").toList

  val dfFinal=dfProd.filter(col("valueRow")>=lit(avg_value-min_value))
  .selectExpr(columnsSelected:_*)
  println("Write data")
  dfFinal.write.mode(SaveMode.Overwrite).parquet("hdfs://atlas:9000/user/carsan/processingDataPersonalized.parquet")


}
