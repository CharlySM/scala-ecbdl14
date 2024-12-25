import sun.security.ec.point.ProjectivePoint.Mutable

import scala.collection.immutable
import scala.collection.mutable.ArrayBuffer
import scala.util.Random

object MiPrueba extends App {

  val r = scala.util.Random
  var a=Array(ArrayBuffer[Tuple4[Double, Double, Double, Double]]((r.nextFloat(), 10, 10, 30), (r.nextFloat(), 8, 8, 30), (r.nextFloat(), 12, 12, 30)),
    ArrayBuffer[Tuple4[Double, Double, Double, Double]]((r.nextFloat(), 10, 10, 18), (r.nextFloat(), 5, 5, 18), (r.nextFloat(), 3, 3, 18)),
    ArrayBuffer[Tuple4[Double, Double, Double, Double]]((r.nextFloat(), 2, 2, 26), (r.nextFloat(), 7, 7, 26), (r.nextFloat(), 15, 15, 26)),
    ArrayBuffer[Tuple4[Double, Double, Double, Double]]((r.nextFloat(), 20, 20, 27), (r.nextFloat(), 5, 5, 27), (r.nextFloat(), 2, 2, 27)))

  val maximo=30
  val min=18
  val mediana=26.5

  val medmax=maximo-mediana
  val medmin=mediana-min

  val margen=(medmax-medmin)/2.0

  val sup=mediana+margen
  val inf=mediana-margen

  val max=mediana
  val corte=mediana/4

  println(medmax)
  println(medmin)
  println(sup)
  println(inf)
  println(margen)
  var a2=a.map(i=>i.filter(j=>j._4>sup && j._4<inf))

  a.foreach(i=> i.map(j=>println(j)))
  immutable.Seq.range(0,5*(maximo-min)).foreach(k=> {
    a2=a2.map(i=> i.map(j=> j.copy(_2=j._3*j._1)))
    a2.foreach(println(_))
    a2=a2.map(i=> i.map(j=> j.copy(_1=j._1+(corte-j._2)/max)))
    println("\n\n")})


  //a.foreach(i=> i.foreach(j=> println(j)))
}
