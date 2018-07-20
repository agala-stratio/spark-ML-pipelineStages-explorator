package com.stratio.intelligence.poc

import java.io.File

import org.apache.spark.ml.Estimator
import org.clapper.classutil.{ClassFinder, ClassInfo}
import org.apache.spark.ml.param.{Param, ParamMap, Params, StringArrayParam}
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods._
import org.json4s._
import org.json4s.jackson.Serialization
import org.json4s.jackson.Serialization.write

import scala.util.{Success, Try}

// => Pipeline stages descriptors classes
case class PipelineStageParam(name:String, paramType:String, paramCategory:String, description:String, defaulValue:String)
case class PipelineStageDescriptor(name:String, className:String, stageType:String, parameters:Seq[PipelineStageParam])


object SparkMlInfo extends App{

  // => Getting current classpath
  val cl = ClassLoader.getSystemClassLoader
  val classpath = cl.asInstanceOf[java.net.URLClassLoader].getURLs.map(x=>new File(x.getFile))

  // => Spark related jars in classpath
  val spark_classpath = classpath.filter(_.toString.contains("spark"))

  // => Getting all Estimator subclasses
  val finder = ClassFinder(spark_classpath)
  val classes = finder.getClasses
  val estimatorsClasses: Seq[ClassInfo] = ClassFinder.concreteSubclasses(
      "org.apache.spark.ml.Estimator", classes
    ).toList.filter(_.toString()!="org.apache.spark.ml.Pipeline")

  // => Getting all Transformer subclasses -> Only transformers that not have an estimator associated
  val transformersClasses = ClassFinder.concreteSubclasses(
    "org.apache.spark.ml.Transformer", classes
  ).toList
  val transformerWithoutEstimatorInstances = transformersClasses.map(
    c => Try(Class.forName(c.toString()).newInstance)).collect{case Success(x) => x}

  // => Getting instances of all Pipeline subclasses
  val estimatorsInstances: Seq[Any] = estimatorsClasses.map(c => Class.forName(c.toString()).newInstance)

  // => All possible pipelineStages to include in front
  val pipelineStagesInstances = transformerWithoutEstimatorInstances ++ estimatorsInstances

  // => Explain params
  pipelineStagesInstances.foreach( e =>
    println(s"\n${"*"*200}\n=> ${e.getClass.getName}\n${"*"*200}\n${e.asInstanceOf[Params].explainParams()}")
  )

  val estimatorsParamMap: Seq[ParamMap] = pipelineStagesInstances.map(e =>
    e.asInstanceOf[Params].extractParamMap()
  )

  val estimatorsParams: Seq[(String, Array[Param[_]])] = pipelineStagesInstances.map(e =>
    (e.getClass.getSimpleName, e.asInstanceOf[Params].params)
  )

  val estimatorsParamsProcessed: Seq[(String, Map[String, (String, String)])] = estimatorsParams.map( e =>
    (e._1, e._2.map( x => (x.name, (x.doc, x.getClass.getSimpleName))).toMap )
  )

  val pipelineStagesDescriptors = pipelineStagesInstances.map( e => {
    PipelineStageDescriptor(name = e.getClass.getSimpleName,
                            className = e.getClass.getName,
                            stageType = e.getClass.getName.replaceAll("org.apache.spark.ml.","").split("\\.")(0),
                            parameters = e.asInstanceOf[Params].params.map( p =>
                              PipelineStageParam( name = p.name,
                                                  description = p.doc,
                                                  paramType = p.getClass.getSimpleName,
                                                  paramCategory = getParameterCategory(e,p),
                                                  defaulValue = getDefaultValue(e,p)
                              )
                            )
    )
  })

  implicit val formats = DefaultFormats

  val pipelineStagesDescriptorsJson: Seq[String] = pipelineStagesDescriptors.map(d => write(d))


  def getDefaultValue(e:Any, p:Param[_]):String ={


    e.asInstanceOf[Params].getDefault(p) match {
      case Some(x) =>{
        var out = "";
        val uid = e.asInstanceOf[org.apache.spark.ml.util.Identifiable].uid

        if(p.isInstanceOf[StringArrayParam]) {
          if (x.asInstanceOf[Array[String]].length == 0)
            out = "[]"
          else
            out = x.asInstanceOf[Array[String]].mkString("[", ",", "]")
        }
        if(out =="" && !x.toString.contains(uid))
            out = x.toString

        out
      }
      case _ => ""
    }
  }

  def getParameterCategory(e:Any, p:Param[_]): String ={
    p.name match {
      case "labelCol" => "input";
      case "inputCol" => "input";
      case "inputCols" => "input";
      case "featuresCol" => "input";

      case "outputCol" => "output";
      case "predictionCol" => "output";
      case "rawPredictionCol" => "output";
      case "probabilityCol" => "output";
      case "probabilityCol" => "output";
      case "probabilityCol" => "output";
      case _ => "parameter"
    }
  }


  print("a")
}


