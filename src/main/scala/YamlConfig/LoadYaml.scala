package YamlConfig

import java.io.{File, FileInputStream}
import org.yaml.snakeyaml.Yaml
import scala.collection.JavaConverters._

object LoadYaml {

  def parseYaml(file:String): Map[String, Any] = {
    val ios = new FileInputStream(new File(file))
    val yaml = new Yaml()
    val obj: Map[String, Any] = yaml.load(ios).asInstanceOf[java.util.Map[String, Any]]
      .asScala.map(kv => (kv._1,kv._2)).toMap
    obj
  }

  def getParams(m:Any): Map[String, Any] = {
      m.asInstanceOf[java.util.Map[String, Any]]
        .asScala.map(kv => (kv._1,kv._2)).toMap
  }

}
