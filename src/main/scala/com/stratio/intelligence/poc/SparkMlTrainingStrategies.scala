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
// strategy = tuningHyperParams
//                validationMode = trainValidationSplit
//                validationMode = crossValidation

trait TrainingStrategy

case class OnlyTrain(strategy: String) extends TrainingStrategy{
  assert("onlyTrain"==strategy)
}

case class TrainAndEvalStrategy(strategy: String,
                                seed: Long,
                                trainRatio: Double,
                                evaluators: Array[EvalEvaluator]
                               ) extends TrainingStrategy{
  assert("trainAndEval"==strategy)
}

case class TuningTrainValidationSplit(strategy: String,
                                      validationMode: String,
                                      seed: Long,
                                      paramGrid: Map[String, String],
                                      testEvaluator: EvalEvaluator,
                                      trainRatio: Double,
                                      evalRatio: Option[Double],
                                      evaluators: Option[Array[EvalEvaluator]]
                                     ) extends TrainingStrategy{
  assert("tuningHyperParams"==strategy)
  assert("trainValidationSplit"==validationMode)
  if(evalRatio.isDefined) assert( evaluators.isDefined && evaluators.get.length > 1)

}

case class TuningCrossValidation(strategy: String,
                                 validationMode: String,
                                 seed: Long,
                                 paramGrid: Map[String, String],
                                 testEvaluator: EvalEvaluator,
                                 numFolds: Int,
                                 evalRatio: Option[Double],
                                 evaluators: Option[Array[EvalEvaluator]]
                                ) extends TrainingStrategy{
  assert("tuningHyperParams"==strategy)
  assert("crossValidation"==validationMode)
  if(evalRatio.isDefined) assert( evaluators.isDefined && evaluators.get.length > 1)
}


object SparkMlTrainingStrategies extends App {

  implicit val formats = DefaultFormats

  val onlyTrainingJson = Source.fromFile("examples/trainStrategy/only_train_strategy.json").getLines.mkString
  val onlyTrainingStrategy = getTrainingStrategy(onlyTrainingJson)
  assert(onlyTrainingStrategy.isSuccess)

  val trainEvalJson = Source.fromFile("examples/trainStrategy/trainEvalSplit_train_strategy.json").getLines.mkString
  val trainEvalStrategy = getTrainingStrategy(trainEvalJson)
  assert(trainEvalStrategy.isSuccess)

  val tuningTrainEvalJson = Source.fromFile("examples/trainStrategy/trainEvalSplit_tuning_strategy.json").getLines.mkString
  val tuningTrainEvalStrategy = getTrainingStrategy(tuningTrainEvalJson)
  assert(tuningTrainEvalStrategy.isSuccess)

  val tuningCrossValidationJson = Source.fromFile("examples/trainStrategy/crossvalidation_tuning_strategy.json").getLines.mkString
  val tuningCrossValidationStrategy = getTrainingStrategy(tuningCrossValidationJson)
  assert(tuningCrossValidationStrategy.isSuccess)

  // Errors
  val onlyTrainingBadJson = Source.fromFile("examples/trainStrategy/withErrors/only_train_bad_strategy.json").getLines.mkString
  val onlyTrainingBadStrategy = getTrainingStrategy(onlyTrainingBadJson)
  assert(onlyTrainingBadStrategy.isFailure)


  def getTrainingStrategy(trainStrategyJson: String): Try[TrainingStrategy] = Try{
    val parsedJson = parse(trainStrategyJson).asInstanceOf[JObject]
    (parsedJson \ "strategy").extract[String] match {
      case "onlyTrain" => read[OnlyTrain](trainStrategyJson)
      case "trainAndEval" => read[TrainAndEvalStrategy](trainStrategyJson)
      case "tuningHyperParams" => {
        (parsedJson \ "validationMode").extract[String] match {
          case "trainValidationSplit" => read[TuningTrainValidationSplit](trainStrategyJson)
          case "crossValidation" => read[TuningCrossValidation](trainStrategyJson)
        }
      }
    }
  }

  print("a")
}
