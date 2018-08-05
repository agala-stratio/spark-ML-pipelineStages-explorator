package com.stratio.intelligence.poc.trainpipeline

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml._
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, Evaluator, MulticlassClassificationEvaluator, RegressionEvaluator}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.param._
import org.apache.spark.ml.tuning._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.{Column, DataFrame, SparkSession}
import org.json4s.DefaultFormats
import org.json4s.jackson.Serialization.writePretty

import scala.util.{Success, Try}


object SparkMlTrainPipelineCustomizable extends App {

  Logger.getLogger("org").setLevel(Level.OFF)
  Logger.getLogger("akka").setLevel(Level.OFF)
  val spark: SparkSession = getSparkSession()

  import spark.implicits._

  // => Input data
  //      Five Gaussian r.v [data_1,..,data_5] --> data_4 and data_5 uninformative
  val inputDf = getBinaryClassDf(
    10000, // Number of samples
    Array((1, 1), (1, 1), (1, 1), (1, 1), (1, 1)), // Array[(std_H0, std_H1)]
    Array((0, 0.5), (0, 0.5), (0, 0.5), (0, 0), (0.5, 0.5)) // Array[(mu_H0, std_H1)]
  )
  inputDf.show()
  val featuresColNames: Array[String] = inputDf.columns.filter(_ != "label")

  // => Pipeline
  val pipeline: Pipeline = getPipeline(featuresColNames)
  // - Id -> stage mapper
  val idStageMapper: Map[String, PipelineStage] = pipeline.getStages.map(s => (s.uid, s)).toMap


  // · Split input data
  val trainDf = inputDf


  // => Train all
  val trainAllStrategy = TrainProperties()

  // => Train and eval

  // => Hyper-parameter tuning - CrossValidation
  val cvHyperParamTuningStrategy = TrainProperties(
    enableHyperparameterTuning = true,
    labelCol = Option("label"),
    predictionCol = Option("rawPrediction"),
    tuningEvaluatorType = Option("BinaryClassificationEvaluator"),
    tuningEvaluatorMetric = Option("areaUnderROC"),
    numFoldsOrTrainTestRatio = Option(3),
    paramGrid = Option(
      Map("lg.fitIntercept" -> "",
          "lg.regParam" -> "0, 0.00001, 0.0001, 0.001, 0.01, 1")
    )
  )

  // · Building grid
  val grid: Array[ParamMap] = paramGridBuilder(
    cvHyperParamTuningStrategy.paramGrid.get, idStageMapper)

  // · Building evaluator
  val evaluator = getEvaluator(
    cvHyperParamTuningStrategy.labelCol.get,
    cvHyperParamTuningStrategy.predictionCol.get,
    cvHyperParamTuningStrategy.tuningEvaluatorType.get,
    cvHyperParamTuningStrategy.tuningEvaluatorMetric.get
  )

  // · Building tuning training executor instance
  val tuningHyperParams = getTrainer(pipeline, evaluator, grid, cvHyperParamTuningStrategy.numFoldsOrTrainTestRatio.get)

  val crossValidatorModel: CrossValidatorModel = tuningHyperParams.asInstanceOf[CrossValidator].fit(trainDf)

  val bestModel: PipelineModel = crossValidatorModel.bestModel.asInstanceOf[PipelineModel]
  val trainAvgMetrics: Array[Double] = crossValidatorModel.avgMetrics

  // => Summary training
  val gridMetricMapper = (grid zip trainAvgMetrics).map(
    x => Map("grid" -> x._1.toString(), "metric" -> x._2))

  implicit val formats = DefaultFormats
  val tuningSummaryJson = writePretty(gridMetricMapper)
  println(tuningSummaryJson)

  println(bestModel.stages.map(
    x => s"${x.asInstanceOf[PipelineStage].uid}" +
      Try(s"\n${x.asInstanceOf[LogisticRegressionModel].summary.toString}").getOrElse("")
  ).mkString("\n\n"))



  print("a")


  def getSparkSession(): SparkSession = {
    SparkSession.builder.appName("SparkMl - Training").master("local").getOrCreate()
  }

  def getBinaryClassDf(n: Int,
                       std: Array[(Double, Double)],
                       mu: Array[(Double, Double)]): DataFrame = {

    def gaussianDistCol(std: Double, mu: Double): Column = randn() * lit(std) + lit(mu)

    val gaussianDescriptors: Array[((Double, Double), (Double, Double))] = std zip mu

    val labelDf: DataFrame = spark.range(n).withColumn("label", (rand() > 0.5).cast(DoubleType)).drop("id")

    gaussianDescriptors.foldLeft((labelDf, 1))((currState, newGaussian) => {
      val currDf = currState._1
      val currIdx = currState._2
      val std = newGaussian._1
      val mu = newGaussian._2
      val newDf = currDf.withColumn(s"data_$currIdx",
        when($"label" === 1, gaussianDistCol(std._2, mu._2))
          .otherwise(gaussianDistCol(std._1, mu._1)))
      (newDf, currIdx + 1)
    })._1
  }

  /**
    * L1-regularized Logistic Regression for Binary Classification
    */
  def getPipeline(features: Array[String]): Pipeline = {
    new Pipeline().setStages(
      Array(
        new VectorAssembler("va")
          .setInputCols(features)
          .setOutputCol("features"),
        new LogisticRegression("lg")
          .setFeaturesCol("features")
          .setRawPredictionCol("rawPrediction")
          .setElasticNetParam(1) // For alpha = 1, it is an L1 penalty.
      ))
  }

  def paramGridBuilder( paramMap: Map[String, String],
                        idStageMapper: Map[String, PipelineStage]
                      ): Array[ParamMap] = {

    def decodeParamValue(paramClassName:String, value: String): Try[Any] = Try {
      paramClassName match {
        case "LongParam" => value.toLong
        case "DoubleParam" => value.toDouble
        case "FloatParam" => value.toFloat
        case "IntParam" => value.toInt
        case "StringArrayParam" => value.split(",").map(_.trim)
        case "DoubleArrayParam" => value.split(",").map(_.trim.toDouble)
        case "IntArrayParam" => value.split(",").map(_.trim.toInt)
        case "Param" => value
        case _ => throw new Exception("Unknown parameter type")
      }
    }

    val gridBuilder = new ParamGridBuilder()
    paramMap.foreach {
      case (paramId, paramValues) => {
        val Array(stageId, paramName) = paramId.split("\\.")
        val stageParam = idStageMapper(stageId).asInstanceOf[Params].getParam(paramName)
        stageParam.getClass().getSimpleName match {
          case "BooleanParam" => gridBuilder.addGrid(stageParam.asInstanceOf[BooleanParam])
          case paramClassName => {
            val paramGrid = paramValues.split(",").map(x => decodeParamValue(paramClassName, x))
            gridBuilder.addGrid(stageParam, paramGrid.map{case Success(x) => x})
          }
        }
      }
    }

    gridBuilder.build()
  }

  def getEvaluator(
                    labelCol:String,
                    predictionCol:String,
                    evaluatorType:String,
                    evaluatorMetric:String
                  ): Evaluator ={

    evaluatorType match {
      case TrainProperties.REGRESSION_EVALUATOR => {
        new RegressionEvaluator()
          .setLabelCol(labelCol).setPredictionCol(predictionCol).setMetricName(evaluatorMetric)
      }
      case TrainProperties.BINARY_CLASSIFIER_EVALUATOR => {
        new BinaryClassificationEvaluator()
          .setLabelCol(labelCol).setRawPredictionCol(predictionCol).setMetricName(evaluatorMetric)
      }
      case TrainProperties.MULTICLASS_CLASSIFIER_EVALUATOR => {
        new MulticlassClassificationEvaluator()
          .setLabelCol(labelCol).setPredictionCol(predictionCol).setMetricName(evaluatorMetric)
      }
    }
  }

  def getTrainer( pipeline: Pipeline,
                  evaluator: Evaluator,
                  paramGrid:Array[ParamMap],
                  numFoldsOrTrainTestRatio:Double
                ): Estimator[_] ={

    if(numFoldsOrTrainTestRatio >2){
      new CrossValidator()
        .setNumFolds(numFoldsOrTrainTestRatio.toInt)
        .setEstimator(pipeline)
        .setEstimatorParamMaps(paramGrid)
        .setEvaluator(evaluator)
    }else{
      new TrainValidationSplit()
        .setTrainRatio(numFoldsOrTrainTestRatio)
        .setEstimator(pipeline)
        .setEstimatorParamMaps(paramGrid)
        .setEvaluator(evaluator)
    }
  }

}

case class TrainProperties(
                            labelCol: Option[String] = None,
                            predictionCol: Option[String] = None,

                            // - Evaluation process
                            enableEvaluation: Boolean = false,
                            trainEvalRatio: Option[Double] = None,
                            trainEvalSplitSeed: Option[Long] = None,
                            evaluatorType: Option[String] = None,
                            metrics: Option[Array[String]] = None,

                            // - Hyper-Parameter tuning process
                            enableHyperparameterTuning: Boolean = false,
                            numFoldsOrTrainTestRatio: Option[Double] = None,
                            tuningSeed: Option[Long] = None,
                            paramGrid: Option[Map[String, String]] = None,
                            tuningEvaluatorType: Option[String] = None,
                            tuningEvaluatorMetric: Option[String] = None
                          ) {

  // - Evaluation mode on
  if (enableEvaluation) {
    assert(trainEvalRatio.isDefined)
    assert(labelCol.isDefined)
    assert(predictionCol.isDefined)
    assert(evaluatorType.isDefined && TrainProperties.validateEvaluatorType(evaluatorType.get))
    assert(metrics.isDefined)
    assert(TrainProperties.validateEvaluatorMetric(evaluatorType.get, metrics.get))
  }

  // - Hyper-parameters tuning mode on
  if(enableHyperparameterTuning){
    assert(numFoldsOrTrainTestRatio.isDefined)
    assert(
      (numFoldsOrTrainTestRatio.get>0.0 && numFoldsOrTrainTestRatio.get<1.0) ||
      (numFoldsOrTrainTestRatio.get>2)
    )
    assert(paramGrid.isDefined && paramGrid.get.size > 0)
    assert(labelCol.isDefined)
    assert(predictionCol.isDefined)
    assert(tuningEvaluatorType.isDefined &&
      TrainProperties.validateEvaluatorType(tuningEvaluatorType.get))
    assert(tuningEvaluatorMetric.isDefined &&
      TrainProperties.validateEvaluatorMetric(
        tuningEvaluatorType.get, Array(tuningEvaluatorMetric.get)))
  }

}

object TrainProperties {

  // => Evaluators
  val BINARY_CLASSIFIER_EVALUATOR = "BinaryClassificationEvaluator"
  val BINARY_CLASSIFIER_EVALUATOR_METRICS = Array("areaUnderROC","areaUnderPR")

  val MULTICLASS_CLASSIFIER_EVALUATOR = "MulticlassClassificationEvaluator"
  val MULTICLASS_CLASSIFIER_EVALUATOR_METRICS = Array("f1", "weightedPrecision", "weightedRecall", "accuracy")

  val REGRESSION_EVALUATOR = "RegressionEvaluator"
  val REGRESSION_EVALUATOR_METRICS = Array("mse","rmse","r2","mae")

  val EVALUATOR_TYPES = Array(
    BINARY_CLASSIFIER_EVALUATOR, MULTICLASS_CLASSIFIER_EVALUATOR, REGRESSION_EVALUATOR)

  def validateEvaluatorType(evaluator:String):Boolean = {
    EVALUATOR_TYPES.contains(evaluator)
  }

  def validateEvaluatorMetric(evaluator:String, metrics:Array[String]):Boolean = {
    evaluator match {
      case TrainProperties.BINARY_CLASSIFIER_EVALUATOR => {
        metrics.forall(metric =>
          TrainProperties.BINARY_CLASSIFIER_EVALUATOR_METRICS.contains(metric))
      }
      case TrainProperties.MULTICLASS_CLASSIFIER_EVALUATOR => {
        metrics.forall(metric =>
          TrainProperties.MULTICLASS_CLASSIFIER_EVALUATOR_METRICS.contains(metric))
      }
      case TrainProperties.REGRESSION_EVALUATOR => {
        metrics.forall(metric =>
          TrainProperties.REGRESSION_EVALUATOR_METRICS.contains(metric))
      }
      case _ => false
    }
  }

}
