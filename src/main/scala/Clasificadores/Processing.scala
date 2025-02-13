package Clasificadores

import Utils.Utils.random_undersampling
import YamlConfig.LoadYaml.parseYaml
import org.apache.spark.SparkContext
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{Column, DataFrame, SaveMode, SparkSession, functions}

import scala.Function.chain
import scala.collection.immutable


object Processing extends App {


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

  def f1()(df:DataFrame): DataFrame = {
    df.withColumns(dictUpdateWeights)
    //dfres.select(columns.map(i=>col(s"$i.weight")):_*).show()
    //dfres
   // null
  }

  import org.apache.spark.sql.functions.udf
  //val getRandomNumber = udf(getRandom)
  val configs=parseYaml(args(0))
  val spark: SparkSession = SparkSession
    .builder()
    .config("spark.driver.extraJavaOptions", "-Xss1024m")
    .config("spark.executor.extraJavaOptions", "-Xss1024m")
    .config("spark.memory.offHeap.enabled", true)
    .config("spark.memory.offHeap.size", "9g")
    .appName(name = "Processing")
    .master(master = configs("cluster").toString)
    .getOrCreate()

  val sc: SparkContext = spark.sparkContext
  spark.sparkContext.setLogLevel("ERROR")

  spark.conf.set("spark.sql.adaptive.enabled", "true")

  println("Reading data")

  val dfStart = spark.sqlContext.read.parquet(configs("dataset").toString).cache()

 // println("dfStart show")
  //dfStart.filter("class=0").show()
  //val dfStart = spark.sqlContext.read.parquet("./src/main/resources/proteinasNormalized.parquet").limit(1000).cache()
  println("Balancing data")
  val dfBalanced=random_undersampling(dfStart, 1.0, "class")

  val columns=dfBalanced.columns.filter(_!="class")
  //println("dfBalanced show")
  //dfBalanced.filter("class=0").show()
  val dfWeights=initializingWeight(dfBalanced, columns)
    .withColumn("sum", lit(0))
    .withColumn("sumTotal", lit(0))
  //println("dfWeights show")
 // dfWeights.filter("class=0").show()
  val dfCalculated=calculate(dfWeights, columns)
    .withColumn("total", col("sumTotal")*col("class"))

 /* val maxim=dfCalculated.select(functions.max(col("sumTotal")).alias("MAX")).first().getDouble(0).toFloat
  val minim=dfCalculated.select(min(col("sumTotal")).alias("MIN")).first().getDouble(0).toFloat
  val mediana=dfCalculated.select(median(col("sumTotal")).alias("median")).first().getDouble(0).toFloat
  val medmax=(maxim-mediana).abs
  val medmin=(mediana-minim).abs

  val margen=(medmax-medmin)/2.0

  val sup=mediana+margen
  val inf=mediana-margen
*/
  val maxim1s=dfCalculated.filter("class=1").select(functions.max(col("sumTotal")).alias("MAX")).first().getDouble(0).toFloat
  val maxim0s=dfCalculated.filter("class=0").select(functions.max(col("sumTotal")).alias("MAX")).first().getDouble(0).toFloat
  val const1s=100000
  val const0s=10000
  val corte1s=const1s/columns.length.toFloat
  val corte0s=const0s/columns.length.toFloat
  //val max=mediana
  //val corte=mediana/4
  //val corte: Float=mediana/columns.length.toFloat
  val mapCorte=map(lit(0), lit(corte0s),
    lit(1), lit(corte1s))
  val mapMax=map(lit(1), lit(maxim1s),
    lit(0), lit(maxim0s))
  //val dfFiltered=dfCalculated.filter(s"sumTotal>$inf and sumTotal<$sup")

  //val dictCols=columns.map(i=>(i, col(i).withField("max", lit(1)).withField("min", lit(0)))).toMap
  //val dfPrepared=dfCalculated.withColumns(dictCols)
  println("Evolutive  transformations")
  val dictUpdateWeights=columns.map(c=> (c, col(c).withField("weight", col(c+".weight")+(mapCorte(col("class"))- (col(c+".res")+mapCorte(col("class"))))/mapMax(col("class")))/*col(c).withField("max", when(col(c + ".res") > lit(corte), col(c + ".weight")).otherwise(
      col(c + ".max"))).withField("min", when(col(c + ".res") < lit(corte), col(c + ".weight"))
      .otherwise(col(c + ".min"))).withField("weight", getRandomNumber(col(c + ".min"), col(c + ".max")))*/
    .withField("res", col(c + ".weight") * col(c + ".value")))).toMap
  //println("dfCalculated show")
  //dfCalculated.filter("class=0").show()
  //val niter=(maxim-minim).toInt*5
 /* println(s"maximo: $maxim")
  println(s"minimo: $minim")
  println(s"numero iteraciones: $niter" )*/
  val listF1=immutable.Seq.range(0,10).map(i=>f1()(_))
  val chained=chain(listF1)
  //dfFiltered.select(columns.map(i=>col(s"$i.weight")):_*).show()
  val dfWeightsCalculated=dfCalculated.transform(chained).cache()
  //dfWeightsCalculated.select(columns.map(i=>col(s"$i.weight")):_*).show()
  val listCols=columns.map(c=>struct(col(s"$c.weight"), col(s"$c.value")).alias(c) ):+col("class")
  //println("dfWeightsCalculated show")
  //dfWeightsCalculated.filter("class=0").show()
  val dfSumaFinal = dfWeightsCalculated.select(listCols:_*).cache()

  val listProds=columns.map(i=>sum(col(s"$i.weight")).alias(i+"_sum")) //:+col("class")

  val prods=dfSumaFinal.select(listProds:_*)
  val row=prods.first()
  val mapsRow=row.getValuesMap[Double](row.schema.fieldNames)
  val selectColumns=mapsRow.toSeq.sortWith(_._2 > _._2).toMap.take(200)

  //val colM: Column =columns.map(i=>col(i+".weight")).reduce((x, y) => x+y)
  //println("dfSumaFinal show")
  //dfSumaFinal.filter("class=0").show()
  //val dfProd=dfSumaFinal.withColumn("valueRow", colM)

  //val min_value = dfProd.select(min(col("valueRow")).alias("MIN")).first().getDouble(0)
 // val avg_value = dfProd.select(avg(col("valueRow")).alias("AVG")).first().getDouble(0)
  println("Select columns and instances")
  val columnsSelected=selectColumns.map(i=>s"`${i._1.replace("_sum", "")}`.value as `${i._1.replace("_sum", "")}`").toList:+"class"

  val dfFinal=dfSumaFinal//.filter(col("valueRow")>=lit(avg_value-min_value))
  .selectExpr(columnsSelected:_*)
  println("Write data")
  dfFinal.write.mode(SaveMode.Overwrite).parquet(configs("output").toString)

}
