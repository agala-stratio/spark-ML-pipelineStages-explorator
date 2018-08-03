package com.stratio.intelligence.poc

import org.json4s.{DefaultFormats, JObject}
import org.json4s.jackson.Serialization.read
import org.json4s._
import org.json4s.jackson.JsonMethods._


import scala.io.Source
import scala.util.Try

// - Evaluators
// ------------------------------------------------------
// metricType = binaryClassifier
// metricType = regression
// metricType = multiClassClassifier

case class EvalEvaluator(metricType: String,
                         metricName: String,
                         labelCol: String,
                         rawPredictionCol: Option[String], // only binaryClassifier
                         predictionCol: Option[String]
                        )

// - Training strategies
// ------------------------------------------------------
// strategy = onlyTrain
// strategy = trainAndEval
// strategy = tuningHyperParamsTraining
//                validationMode = trainValidationSplit
//                validationMode = crossValidation

case class TrainingStrategy(strategy: String)

case class TrainAndEvalStrategy(strategy: String,
                           seed: Long,
                           trainRatio: Double,
                           evaluators: Array[EvalEvaluator]
                          )

case class TuningTrainValidationSplit(strategy: String,
                                 validationMode: String,
                                 seed: Long,
                                 paramGrid: Map[String, String],
                                 evaluator: EvalEvaluator,
                                 trainRatio: Double
                                )

case class TuningCrossValidation(strategy: String,
                                 validationMode: String,
                                 seed: Long,
                                 paramGrid: Map[String, String],
                                 evaluator: EvalEvaluator,
                                 numFolds: Int
                                )


object SparkMlTrainingStrategies extends App {

  implicit val formats = DefaultFormats

  val readedJson = Source.fromFile("examples/example_training_strategy.json").getLines.mkString

  val myObj = parse(readedJson).asInstanceOf[JObject]
  val d = (myObj \ "strategy").extract[String] match {
    case "onlyTrain" => Try(read[TrainingStrategy](readedJson))
    case "trainAndEval" => Try(read[TrainAndEvalStrategy](readedJson))
    case "tuningHyperParamsTraining" => {
      (myObj \ "validationMode").extract[String] match {
        case "trainValidationSplit" => Try(read[TuningTrainValidationSplit](readedJson))
        case "crossValidation" => Try(read[TuningCrossValidation](readedJson))
      }
    }
  }


  print("a")
}
