package com.stratio.intelligence.poc

import java.io.File
import java.lang.reflect.Field

import org.apache.spark.ml.param._
import org.clapper.classutil.{ClassFinder, ClassInfo}
import org.json4s._
import org.json4s.jackson.Serialization.writePretty

import scala.util.{Success, Try}

// => Pipeline stages descriptors classes
case class PipelineStageParam(name: String, paramType: String, paramCategory: String, description: String, defaultValue: String, restriction: String)

case class PipelineStageDescriptor(name: String, className: String, stageType: String, parameters: Seq[PipelineStageParam])


object SparkMlPipelineStagesDescriptors extends App {

  // => Getting current classpath
  val cl = ClassLoader.getSystemClassLoader
  val classpath = cl.asInstanceOf[java.net.URLClassLoader].getURLs.map(x => new File(x.getFile))

  // => Spark related jars in classpath
  val spark_classpath = classpath.filter(_.toString.contains("spark"))

  // => Getting all Estimator subclasses
  val finder = ClassFinder(spark_classpath)
  val classes = finder.getClasses
  val estimatorsClasses: Seq[ClassInfo] = ClassFinder.concreteSubclasses(
    "org.apache.spark.ml.Estimator", classes
  ).toList.filter(_.toString() != "org.apache.spark.ml.Pipeline")
  val estimatorsInstances: Seq[Any] = estimatorsClasses.map(c => Class.forName(c.toString()).newInstance)


  // => Getting all Transformer subclasses -> Only transformers that not have an estimator associated
  val transformersClasses = ClassFinder.concreteSubclasses(
    "org.apache.spark.ml.Transformer", classes
  ).toList
  val transformerWithoutEstimatorInstances = transformersClasses.map(
    c => Try(Class.forName(c.toString()).newInstance)).collect { case Success(x) => x }

  // => Getting all evaluator subclasses
  val evaluatorClasses: Seq[ClassInfo] = ClassFinder.concreteSubclasses(
    "org.apache.spark.ml.evaluation.Evaluator", classes
  ).toList
  val evaluatorInstances = evaluatorClasses.map(c => Class.forName(c.toString()).newInstance)


  // => All possible pipelineStages to include in front
  val pipelineStagesInstances = transformerWithoutEstimatorInstances ++ estimatorsInstances ++ evaluatorInstances

  // => Explain params
  pipelineStagesInstances.foreach(e =>
    println(s"\n${"*" * 200}\n=> ${e.getClass.getName}\n${"*" * 200}\n${e.asInstanceOf[Params].explainParams()}")
  )

  val estimatorsParamMap: Seq[ParamMap] = pipelineStagesInstances.map(e =>
    e.asInstanceOf[Params].extractParamMap()
  )

  val estimatorsParams: Seq[(String, Array[Param[_]])] = pipelineStagesInstances.map(e =>
    (e.getClass.getSimpleName, e.asInstanceOf[Params].params)
  )

  val estimatorsParamsProcessed: Seq[(String, Map[String, (String, String)])] = estimatorsParams.map(e =>
    (e._1, e._2.map(x => (x.name, (x.doc, x.getClass.getSimpleName))).toMap)
  )

  val pipelineStagesDescriptors = pipelineStagesInstances.map(e => {
    PipelineStageDescriptor(name = e.getClass.getSimpleName,
      className = e.getClass.getName,
      stageType = e.getClass.getName.replaceAll("org.apache.spark.ml.", "").split("\\.")(0),
      parameters = e.asInstanceOf[Params].params.map(p =>
        PipelineStageParam(name = p.name,
          description = p.doc,
          paramType = p.getClass.getSimpleName,
          paramCategory = getParameterCategory(e, p),
          defaultValue = getDefaultValue(e, p),
          restriction = getParameterRestriction(p)
        )
      )
    )
  })

  implicit val formats = DefaultFormats

  val pipelineStagesDescriptorsJson = pipelineStagesDescriptors.map(d => (d.name, writePretty(d), d.stageType))


  // Saving jsons
  pipelineStagesDescriptorsJson.foreach(
    x => {
      scala.tools.nsc.io.File(s"outputs/${x._3}").createDirectory(failIfExists = false)
      scala.tools.nsc.io.File(s"outputs/${x._3}/${x._1}.json").writeAll(x._2)
    }
  )


  def getDefaultValue(e: Any, p: Param[_]): String = {
    e.asInstanceOf[Params].getDefault(p) match {
      case Some(x) => {
        p match {
          case _: StringArrayParam => {
            if (x.asInstanceOf[Array[String]].length == 0) "[]" else x.asInstanceOf[Array[String]].mkString("[", ",", "]")
          }
          case _: DoubleArrayParam => {
            if (x.asInstanceOf[Array[Double]].length == 0) "[]" else x.asInstanceOf[Array[Double]].mkString("[", ",", "]")
          }
          case _: IntArrayParam => {
            if (x.asInstanceOf[Array[Int]].length == 0) "[]" else x.asInstanceOf[Array[Int]].mkString("[", ",", "]")
          }
          case _ =>{
            val uid = e.asInstanceOf[org.apache.spark.ml.util.Identifiable].uid
            if(!x.toString.contains(uid)) x.toString else ""
          }
        }
      }
      case _ => ""
    }
  }

  def getParameterCategory(e: Any, p: Param[_]): String = {
    p.name match {
      case "labelCol" => "input";
      case "inputCol" => "input";
      case "inputCols" => "input";
      case "featuresCol" => "input";

      case "outputCol" => "output";
      case "predictionCol" => "output";
      case "rawPredictionCol" => "output";
      case "probabilityCol" => "output";
      case _ => "parameter"
    }
  }

  def getParameterRestriction(p: Param[_]): String = {

    val validatorClass = p.isValid.getClass().getName
    if (validatorClass.startsWith("org.apache.spark.ml.param.ParamValidators")) {
      validatorClass match {
        case s if s.contains("inArray") => {
          val allowed = getDeclaredFieldValue[Array[String]](p, "allowed")
          s"in ${allowed.mkString("[", ",", "]")}"
        }
        case s if s.contains("gt") => {
          val lowerBound = getDeclaredFieldValue[Double](p, "lowerBound")
          s">${lowerBound}"
        }
        case s if s.contains("gtEq") => {
          val lowerBound = getDeclaredFieldValue[Double](p, "lowerBound")
          s">=${lowerBound}"
        }
        case s if s.contains("inRange") => {
          val lowerBound = getDeclaredFieldValue[Double](p, "lowerBound")
          val upperBound = getDeclaredFieldValue[Double](p, "upperBound")
          val lowerInclusive = getDeclaredFieldValue[Boolean](p, "lowerInclusive")
          val upperInclusive = getDeclaredFieldValue[Boolean](p, "upperInclusive")
          s"${if (lowerInclusive) ">=" else ">"}${lowerBound} && ${if (upperInclusive) "<=" else "<"}${upperBound}"
        }
        case s if s.contains("lt") => {
          val upperBound = getDeclaredFieldValue[Double](p, "upperBound")
          s"<${upperBound}"
        }
        case s if s.contains("ltEq") => {
          val upperBound = getDeclaredFieldValue[Double](p, "upperBound")
          s"<=${upperBound}"
        }
        case s if s.contains("arrayLengthGt") => {
          val lowerBound = getDeclaredFieldValue[Double](p, "lowerBound")
          s"ArrayLength >${lowerBound}"
        }

        case _ => ""
      }
    } else ""
  }

  def getDeclaredFieldValue[T](p: Param[_], field: String): T = {
    val declaredFieldArray: Array[Field] = p.isValid.getClass().getDeclaredFields.filter(_.getName.contains(field))
    assert(declaredFieldArray.length == 1)
    val declaredField = declaredFieldArray(0)
    declaredField.setAccessible(true)
    declaredField.get(p.isValid).asInstanceOf[T]
  }

}


