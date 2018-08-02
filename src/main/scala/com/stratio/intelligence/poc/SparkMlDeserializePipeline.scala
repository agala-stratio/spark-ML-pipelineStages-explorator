package com.stratio.intelligence.poc

import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.ml.param.{DoubleParam, Params}
import org.apache.spark.sql.SparkSession
import org.json4s.DefaultFormats
import org.json4s._
import org.json4s.jackson.Serialization
import org.json4s.jackson.Serialization.read

import scala.io.Source
import scala.util.{Failure, Try}


// => Pipeline stages descriptors classes
case class PipelineStageParamDeser(name:String, paramType:String, paramCategory:String, description:String, value:String)
case class PipelineStageDescriptorDeser(name:String, className:String, stageType:String, parameters:Seq[PipelineStageParamDeser])
case class PipelineDescriptor(pipeline:Array[PipelineStageDescriptorDeser])

object SparkMlDeserializePipeline extends App {

 /* **************************
      Example to replicate
    **************************

    // Prepare training documents from a list of (id, text, label) tuples.
    val training = spark.createDataFrame(Seq(
      (0L, "a b c d e spark", 1.0),
      (1L, "b d", 0.0),
      (2L, "spark f g h", 1.0),
      (3L, "hadoop mapreduce", 0.0)
    )).toDF("id", "text", "label")

    // Configure an ML pipeline, which consists of three stages:
    //    tokenizer, hashingTF, and lr.
    val tokenizer = new Tokenizer()
      .setInputCol("text")
      .setOutputCol("words")
    val hashingTF = new HashingTF()
      .setNumFeatures(1000)
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("features")
    val lr = new LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.001)
    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, hashingTF, lr))

    // Fit the pipeline to training documents.
    val model = pipeline.fit(training)

   */


  implicit val formats = DefaultFormats

  // val readedJson = Source.fromFile("examples/nlp_pipeline_example.json").getLines.mkString

  val readedJson = Source.fromFile("examples/nlp_pipeline_example.json").getLines.mkString
  val pipelineDescriptor: Try[PipelineDescriptor] = Try(read[PipelineDescriptor](readedJson))

  pipelineDescriptor match {
    case Failure(e) => {println(e); System.exit(1)}
    case _ => None
  }
  val deserializedPipeline: Pipeline = getPipelineFromDescriptor(pipelineDescriptor.get)


  val spark = SparkSession.builder.appName("Simple Application").master("local").getOrCreate()

  val training = spark.createDataFrame(Seq(
    (0L, "a b c d e spark", 1.0),
    (1L, "b d", 0.0),
    (2L, "spark f g h", 1.0),
    (3L, "hadoop mapreduce", 0.0)
  )).toDF("id", "text", "label")

  val model = deserializedPipeline.fit(training)

  val outDf = model.transform(training)
  outDf.show()

  print("Okk")



  def getPipelineFromDescriptor(pipelineDescriptor: PipelineDescriptor): Pipeline = {

    val stages: Array[PipelineStage] = for(stageDescriptor <- pipelineDescriptor.pipeline) yield{
      val stage = Class.forName(stageDescriptor.className).newInstance
      stageDescriptor.parameters.foreach( paramDescriptor => {
        val paramToSet = stage.asInstanceOf[Params].getParam(paramDescriptor.name)


        stage.asInstanceOf[Params].set(paramToSet, getParamType(paramDescriptor.paramType, paramDescriptor.value))
      })
      stage.asInstanceOf[PipelineStage]
    }

    new Pipeline().setStages(stages)
  }

  def getParamType(pType:String, value:String): Any ={
    pType match {
      case "DoubleParam" => value.toDouble
      case "IntParam" => value.toInt
      case _ => value
    }
  }

}


