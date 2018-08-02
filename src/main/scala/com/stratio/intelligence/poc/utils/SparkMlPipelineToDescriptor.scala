package com.stratio.intelligence.poc.utils

import java.io.InputStream

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature._
import org.apache.spark.ml.param._
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.json4s.DefaultFormats
import org.json4s.jackson.Serialization.{read, writePretty}

import scala.io.Source
import scala.util.Try


case class PipelineStageParamDeser(name: String, value: String)


case class PipelineStageDescriptorDeser( name: String,
                                         uid: String,
                                         parameters: Seq[PipelineStageParamDeser])



object SparkMlPipelineToDescriptor extends App {

  val tokenizer = new Tokenizer().setInputCol("text").setOutputCol("words")
  val hashingTF = new HashingTF().setNumFeatures(1000).setInputCol(tokenizer.getOutputCol).setOutputCol("features")
  val lr = new LogisticRegression().setMaxIter(10).setRegParam(0.001)
  val remover = new StopWordsRemover().setInputCol("raw").setOutputCol("filtered").setStopWords(Array("a","b","ccc"))
  val vl = new VectorSlicer().setIndices(Array(1,2,33,4444))
  val buk = new Bucketizer().setSplits(Array(0.1,0.2,0.33,0.444))
  val pipeline = new Pipeline()
    .setStages(Array(tokenizer, hashingTF, lr,remover,vl,buk))

  val descriptor: Array[PipelineStageDescriptorDeser] = SparkMlUtils.getDescriptorFromPipeline(pipeline)

  implicit val formats = DefaultFormats
  println(writePretty(descriptor))


  val reconstructedPipeline = SparkMlUtils.getPipelineFromDescriptor(descriptor)

  print("a")

}

object SparkMlUtils{

  lazy val stageNameToClassMapper:Try[Map[String, String]] = Try {
    implicit val formats = DefaultFormats

    val stream: InputStream = getClass.getResourceAsStream("/mlpipeline-classes-mapper.json")
    val json: String = Source.fromInputStream(stream).mkString
    read[Map[String, String]](json)
  }

  def getPipelineFromDescriptor(pipelineDescriptor: Array[PipelineStageDescriptorDeser]): Pipeline = {

    def decodeParamValue(param:Param[Any], value:String): Try[Any] = Try{
      param.getClass().getSimpleName match {
        case "BooleanParam" => value.toBoolean
        case "LongParam" => value.toLong
        case "DoubleParam" => value.toDouble
        case "FloatParam" => value.toFloat
        case "IntParam" => value.toInt
        case "StringArrayParam" => value.split(",").map(_.trim)
        case "DoubleArrayParam" => value.split(",").map(_.trim.toDouble)
        case "IntArrayParam" => value.split(",").map(_.trim.toInt)
        case "Param"=> value
        case _ => throw new Exception("Unknown parameter type")
      }
    }

    val stages: Array[PipelineStage] = for(stageDescriptor <- pipelineDescriptor) yield{

      // · Instantiate class
      val stage = Class.forName(
        // Getting class from mapper
        stageNameToClassMapper.get(stageDescriptor.name)
      ).getConstructor(classOf[String]).newInstance(stageDescriptor.uid)

      // · Set parameters
      stageDescriptor.parameters.foreach( paramDescriptor => {
        val paramToSet: Param[Any] = stage.asInstanceOf[Params].getParam(paramDescriptor.name)
        val valueToSet =  decodeParamValue(paramToSet, paramDescriptor.value)

        stage.asInstanceOf[Params].set(paramToSet, valueToSet.get)
      })

      stage.asInstanceOf[PipelineStage]
    }

    new Pipeline().setStages(stages)
  }

  def getDescriptorFromPipeline(pipeline: Pipeline): Array[PipelineStageDescriptorDeser] ={

    def encodeParamValue(params:Params, param:Param[_]) = Try{
        val x = params.get(param).get
        param match {
          case _: StringArrayParam | _:DoubleArrayParam| _:IntArrayParam => {
            val value = x.asInstanceOf[Array[_]]
            if (value.length == 0) "" else value.mkString(",")
          }
          case _ => x.toString()
      }
    }
    pipeline.getStages.map( e => {
      PipelineStageDescriptorDeser(
        name = e.getClass.getSimpleName,
        uid = e.uid,
        parameters = e.asInstanceOf[Params].params.flatMap(p => Try{
          PipelineStageParamDeser(name = p.name, value = encodeParamValue(e.asInstanceOf[Params],p).get)
        }.toOption).toSeq
      )
    })
  }

}