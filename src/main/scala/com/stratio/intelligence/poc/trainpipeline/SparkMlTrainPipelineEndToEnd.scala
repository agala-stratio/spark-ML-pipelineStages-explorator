package com.stratio.intelligence.poc.trainpipeline

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.{Model, Pipeline, PipelineModel, PipelineStage}
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.param.{BooleanParam, DoubleParam}
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.sql.{Column, DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.DoubleType
import org.json4s.DefaultFormats
import org.json4s.jackson.Serialization.writePretty

import scala.util.Try


object SparkMlTrainPipelineEndToEnd extends App {

  Logger.getLogger("org").setLevel(Level.OFF)
  Logger.getLogger("akka").setLevel(Level.OFF)
  val spark: SparkSession = getSparkSession()

  import spark.implicits._

  // => Input data
  //      Five Gaussian r.v [data_1,..,data_5] --> data_4 and data_5 uninformative
  val inputDf = getBinaryClassDf(
    10000, // Number of samples
    Array((1, 1), (1, 1), (1, 1), (1, 1), (1, 1)), // Array[(std_H0, std_H1)]
    Array((0, 0.5), (0, 0.5), (0, 0.5), (0, 0), (0.5, 0.5))  // Array[(mu_H0, std_H1)]
  )
  inputDf.show()
  val featuresColNames: Array[String] = inputDf.columns.filter(_ != "label" )

  // => Pipeline
  val pipeline = getPipeline(featuresColNames)

  // => Train all

  // => Train and eval

  // => Hyper-parameter tuning - CrossValidation
  val evaluator = new BinaryClassificationEvaluator()
  val grid = new ParamGridBuilder()
  val crossValidation = new CrossValidator()

  // · Split input data
  val trainDf = inputDf

  // · Defining crossValidation training process
  evaluator
    .setLabelCol("label")
    .setRawPredictionCol("rawPrediction")
    .setMetricName("areaUnderROC")

  val gridParamMap = grid
    // - with or without intercept term
    .addGrid( pipeline.getStages(1).getParam("fitIntercept").asInstanceOf[BooleanParam])
    // - L1-regularization parameter
    .addGrid( pipeline.getStages(1).getParam("regParam").asInstanceOf[DoubleParam],
              Array(0, 0.00001, 0.0001, 0.001, 0.01, 1) )
    .build()

  crossValidation
    .setEstimator(pipeline)
    .setEvaluator(evaluator)
    .setEstimatorParamMaps(gridParamMap)
    .setNumFolds(3)

  // · Training process
  val crossValidatorModel: CrossValidatorModel = crossValidation.fit(trainDf)

  val bestModel: PipelineModel = crossValidatorModel.bestModel.asInstanceOf[PipelineModel]
  val trainAvgMetrics: Array[Double] = crossValidatorModel.avgMetrics

  // => Summary training
  val gridMetricMapper = (gridParamMap zip trainAvgMetrics).map(
    x => Map("grid" -> x._1.toString(), "metric" -> x._2))

  implicit val formats = DefaultFormats
  val tuningSummaryJson = writePretty(gridMetricMapper)
  println(tuningSummaryJson)

  println(bestModel.stages.map(
    x => s"${x.asInstanceOf[PipelineStage].uid}"+
      Try(s"\n${x.asInstanceOf[LogisticRegressionModel].summary.toString}").getOrElse("")
  ).mkString("\n\n"))

  print("a")


  def getSparkSession(): SparkSession = {
    SparkSession.builder.appName("SparkMl - Training").master("local").getOrCreate()
  }

  def getBinaryClassDf(n: Int,
                       std: Array[(Double, Double)],
                       mu: Array[(Double, Double)]): DataFrame = {

    def gaussianDistCol(std: Double, mu: Double): Column = {
      randn() * lit(std) + lit(mu)
    }

    val gaussianDescriptors: Array[((Double, Double), (Double, Double))] = std zip mu

    val labelDf: DataFrame = spark.range(n).withColumn("label", (rand() > 0.5).cast(DoubleType)).drop("id")

    gaussianDescriptors.foldLeft((labelDf, 1))((currState, newGaussian) => {
      val currDf = currState._1;
      val currIdx = currState._2
      val std = newGaussian._1;
      val mu = newGaussian._2
      val newDf = currDf.withColumn(s"data_$currIdx",
        when($"label" === 1, gaussianDistCol(std._2, mu._2))
          .otherwise(gaussianDistCol(std._1, mu._1)))
      (newDf, currIdx + 1)
    })._1
  }

  /**
    * L1-regularized Binary Logistic Regression
    */
  def getPipeline(features: Array[String]): Pipeline = {
    new Pipeline().setStages(
      Array(
        new VectorAssembler()
          .setInputCols(features)
          .setOutputCol("features"),
        new LogisticRegression()
          .setFeaturesCol("features")
          .setRawPredictionCol("rawPrediction")
          .setElasticNetParam(1) // For alpha = 1, it is an L1 penalty.
      ))
  }
}

