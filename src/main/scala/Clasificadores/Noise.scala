package Clasificadores


import org.apache.spark.ml.classification.{RandomForestClassifier}
import org.apache.spark.sql.{DataFrame, Dataset, Row}

object Noise {

  def randomForest(trainingData:DataFrame): DataFrame = {

    val folds=trainingData.randomSplit(Array(0.25, 0.25, 0.25, 0.25), 123456789L)
    println("count traindata: ", trainingData.count())
    println(folds.length)
    val model = new RandomForestClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setNumTrees(10)

    folds.map(i=>{
      val train=folds.diff(Array(i)).reduce(_ union _)

      val model1=model.fit(train)

      val predictions = model1.transform(i)

      predictions.filter("label=1 or (label=0 and label=prediction)").select("features", "label")

    }).reduce(_ union _)


  }

}
