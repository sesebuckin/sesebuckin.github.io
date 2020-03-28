---
layout:     post
title:      XGBoost4j-Spark实战
subtitle:   
date:       2020-03-28
author:     Schoko
header-img: 
catalog: true
tags:
    - Spark
    - Scala
    - XGBoost4j-Spark
---

### 前言

最近做项目，首次使用了XGBoost4j-Spark，踩了不少坑，日常就是查Spark、XGBoost、Scala的API文档......
所以简单总结一下遇到的问题和解决方法。

### 需要预备的知识

Scala：

[https://www.scala-lang.org](https://www.scala-lang.org/) 

Spark：

[https://spark.apache.org/docs/2.3.0/](https://spark.apache.org/docs/2.3.0/)

XGBoost4j-Spark：

[https://xgboost.readthedocs.io/en/release_0.82/jvm/xgboost4j_spark_tutorial.html](https://xgboost.readthedocs.io/en/release_0.82/jvm/xgboost4j_spark_tutorial.html)

本文代码版本：

- 工程: Maven

- Scala: 2.11.8

- Spark: 2.3.1

- XGBoost4j-Spark: 0.82

那咱们开始吧————

### 配置依赖

```xml
<dependencies>
        <!-- Spark-core -->
        <dependency>
            <groupId>org.apache.spark</groupId>
            <artifactId>spark-core_2.11</artifactId>
            <version>2.3.1</version>
        </dependency>
        <!-- SparkSQL -->
        <dependency>
            <groupId>org.apache.spark</groupId>
            <artifactId>spark-sql_2.11</artifactId>
            <version>2.3.1</version>
        </dependency>
        <!-- SparkSQL  ON  Hive-->
        <dependency>
            <groupId>org.apache.spark</groupId>
            <artifactId>spark-hive_2.11</artifactId>
            <version>2.3.1</version>
        </dependency>
        <!-- Scala 包-->
        <dependency>
            <groupId>org.scala-lang</groupId>
            <artifactId>scala-library</artifactId>
            <version>2.11.7</version>
        </dependency>
        <dependency>
            <groupId>org.scala-lang</groupId>
            <artifactId>scala-compiler</artifactId>
            <version>2.11.7</version>
        </dependency>
        <dependency>
            <groupId>org.scala-lang</groupId>
            <artifactId>scala-reflect</artifactId>
            <version>2.11.7</version>
        </dependency>
        <!-- https://mvnrepository.com/artifact/junit/junit -->
        <dependency>
            <groupId>junit</groupId>
            <artifactId>junit</artifactId>
            <version>4.12</version>
            <scope>test</scope>
        </dependency>

        <!-- https://mvnrepository.com/artifact/org.apache.spark/spark-mllib-local -->
        <dependency>
            <groupId>org.apache.spark</groupId>
            <artifactId>spark-mllib_2.11</artifactId>  <!--spark-mllib-local_2.11-->
            <version>2.3.1</version>
        </dependency>
        <dependency>
            <groupId>ml.dmlc</groupId>
            <artifactId>xgboost4j</artifactId>
            <version>0.82</version>
        </dependency>
        <dependency>
            <groupId>ml.dmlc</groupId>
            <artifactId>xgboost4j-spark</artifactId>
            <version>0.82</version>
        </dependency>

</dependencies>
```

### 数据读取、预处理

#### 数据读取
把打好的jar包放到接入Spark集群的机器上就可以轻松运行，所以从Hive里读取模型的训练数据也很容易：

```scala
val spark = SparkSession.builder().enableHiveSupport().appName("CtrXgboost").getOrCreate()
val df = spark.table(inputTable).select("*").orderBy(rand(2020))  // limit(1000)
```

df的数据类型为Spark DataFrame，orderBy(rand(seed))是对DataFrame进行Row shuffle。

#### 数据预处理

获取sparseFeatures和desneFeatures列：

```scala
def getEncCols(inputDF: DataFrame): (Array[String], Array[String]) = {
    val textCols = new ArrayBuffer[String]()
    val res = new ArrayBuffer[String]()

    for (col <- inputDF.columns) {
        if (col.contains("rightname") || col.contains("producttitle")) {
            textCols += col
        } else {
            if (col != "label") {res += col}
        }
    }

    val sparseFeatures = new ArrayBuffer[String]()
    val denseFeatures = new ArrayBuffer[String]()

    for (col <- res) {
        if (inputDF.schema(col).dataType.isInstanceOf[StringType]) {
            sparseFeatures += col
        } else {
            denseFeatures += col
        }
    }
    (sparseFeatures.toArray, denseFeatures.toArray)
}
```

填补缺失值：

```scala
def fillNan(inputDF: DataFrame, categoricalColumns: Array[String], numericalColumns: Array[String]): DataFrame = {
    val fillCate = inputDF.na.replace(categoricalColumns, Map("" -> "-1"))
    val fillNum = fillCate.na.fill(0, numericalColumns)
    fillNum
}
```

对于categoricalColumns，一开始也是用的na.fill方法，但不知道为什么没有用。
猜测原因可能是原本数据中的空值是Spark不支持的类型（因为数据处理是用python做的）。

### 特征编码

对类别型特征，采用了One-Hot Encoding进行编码。
一开始我用了Label Encoding，但思考了下，Label Encoding会依赖于训练数据中各项出现的频次，
而训练数据每天都有新增，可能导致统计结果发生变化，而unique value相对来说变化稍小。
所以在unique value不超过一定范围的前提下，还是使用One-Hot Encoding。

首先构建Pipeline：
```scala
import org.apache.spark.ml.{Pipeline, PipelineModel, PipelineStage}

def buildStages(categoricalColumns: Array[String], numericalColumns: Array[String]): Array[PipelineStage]= {

    val stagesArray = new ListBuffer[PipelineStage]()

    for (cate <- categoricalColumns) {
        println("-------------------------")
        println(cate)
        val indexer = new StringIndexer()
                .setInputCol(cate)
                .setOutputCol(s"${cate}Index")
                .setHandleInvalid("keep")
        val encoder = new OneHotEncoder()
                .setInputCol(s"${cate}Index")
                .setOutputCol(s"${cate}classVec")

        stagesArray.append(indexer, encoder)
    }

    val assemblerInputs = categoricalColumns.map(_ + "classVec") ++ numericalColumns
    val assembler = new VectorAssembler().setInputCols(assemblerInputs).setOutputCol("features")
    stagesArray.append(assembler)

    stagesArray.toArray
}
```

进行fit与transform，返回transform后的数据与训练好的pipeline模型：
```scala
def pipelineProcess(inputDF: DataFrame, stages: Array[PipelineStage]): (DataFrame, PipelineModel)= {
    print(">> pipeline fit")
    val model = new Pipeline().setStages(stages).fit(inputDF)
    println(">> pipeline transform")
    val dataSet = model.transform(inputDF)
    val encoderParamMap: ParamMap = model.extractParamMap()

    (dataSet, model)
}
```

### 数据集划分

和python中的sklearn模块的train_test_split用法类似，不过没法设置分层划分。
```scala
val Array(train, test) = dataSet.randomSplit(Array(0.8, 0.2), seed = 4444)
val Array(trainDF, valDF) = train.randomSplit(Array(0.8, 0.2), seed = 4444)
```

当然想达到分层划分的效果也是可以的。先把posDF和NegDF独立出来，
按设定好的比例进行按行采样，然后把两个DataFrame进行concatation，最后shuffle rows。

### 模型

和Python里的API差不多，但python版本有对接sklearn的接口，可以使用的API更加丰富。
```scala
val paramMap = Map(
        "eta" -> 0.13,
        "verbosity" -> 2,
        "disable_default_eval_metric " -> 1,
        "max_depth" -> 5,
        "subsample" -> 0.85,
        "colsample_bytree" -> 0.85,
        "objective" -> "binary:logistic",
        "num_round" -> 300,
        "seed" -> 2020
    )

val Classifier = new XGBoostClassifier(paramMap)
        .setLabelCol("label")
        .setFeaturesCol("features")
        .setEvalSets(Map("eval" -> valDF))  //setEvalSets(Map("train" -> trainDF, "eval" -> valDF))
        .setEvalMetric("logloss")
        .setNumEarlyStoppingRounds(10)
        .setMaximizeEvaluationMetrics(false)
        .setSeed(2020)  // seed for C++ codes
```
训练：
```scala
val model = Classifier.fit(trainDF)
println(model.summary.toString())  // 输出train set和val set在训练过程中的历史metrics
```
特征重要性：
```scala
val featureScoreMap_gain = model.nativeBooster.getScore("", "gain")
val sortedScoreMap = featureScoreMap_gain.toSeq.sortBy(-_._2) // descending order
println(sortedScoreMap)
```

### 评估
通过transform方法，得到模型在测试集上的预测结果：
```scala
val oriPrediction = model.transform(testDF)
```
需要注意的是，tranform的输出格式：
```scala
/**
  * When we get a model, either XGBoostClassificationModel or XGBoostRegressionModel,
  * it takes a DataFrame, read the column containing feature vectors,
  * predict for each feature vector, and output a new DataFrame
  * XGBoostClassificationModel will output margins (rawPredictionCol),
  * probabilities(probabilityCol) and the eventual prediction labels (predictionCol) for each possible label.
  * */
```
对于二分类任务，我们需要关系的字段是模型预测的log probability以及真实标签。这是在集群运行时打印出的结果：

![prediction](/img/post-xgb4jspark-prediction.PNG)

建立evaluator对象，评估AUC：

```scala
val evaluator = new BinaryClassificationEvaluator()
evaluator.setLabelCol("label")
evaluator.setRawPredictionCol("probability")
evaluator.setMetricName("areaUnderROC")

/* AUC */
val AUC = evaluator.evaluate(prediction)
```

有个容易踩坑的地方需注意：

BinaryClassificationEvaluator类默认的setMetricName只能选择areaUnderROC或者areaUnderPR，
并没有提供计算logloss的方法。所以计算logloss需要自己手动实现。

rawPrediction与probability都是Vector（Dense Vector），
Vector类定义在org.apache.spark.ml.linalg.Vector包中。
如果需要计算logloss，需要把Vector中我们需要的值拿出来，转为Double或者Float类型，再与label列进行计算。

幸运的是，Vector提供了toArray的方法。因此我们可以通过自定义udf来实现上诉需求：
```scala
val vectorToArray = udf((row: Vector) => row.toArray)
```
然后把udf运用到DataFrame上就可以了。

我们可以把每次训练后得到的auc与logloss落表。在下一次训练后，首先进行评估，然后拉取历史auc与logloss，计算平均值，
再与本次的值进行比较。如果差值超过自定义的阈值，则判断本次训练模型存在问题，可能会影响线上效果。

核心逻辑：
```scala
if ((meanAUCRow != null && meanAUCRow.length > 0) && (meanLoglossRow != null && meanLoglossRow.length > 0)) {
    val meanAUCRowHead = meanAUCRow.head
    val meanLoglossRowHead = meanLoglossRow.head
    if ((meanAUCRowHead != null && meanAUCRowHead.length > 0 && meanAUCRowHead.get(0) != null)
            && (meanLoglossRowHead != null && meanLoglossRowHead.length > 0 && meanLoglossRowHead.get(0) != null)) {
        val meanAUC = meanAUCRowHead.getFloat(0)
        val meanLogloss = meanLoglossRowHead.getFloat(0)
        if (((meanAUC - testAUCFloat) <= thresh) || ((meanLogloss - testLoglossFloat) <= thresh)) {1}  // succeed
        else {0}  // failed
    } else {0}  // failed
} else{0}  // failed
```

根据评估情况，决定是否进行model persistence：
```scala
/* evaluating stage */
println(">> Evaluating start...")
val status = evaluation(test, model, spark, targetTable, evalModelThresh)

// if status is 1, then do model persistence; else, do nothing
if (status == 1) {
    println(">> model persistence...")
    val hdfs: FileSystem =  FileSystem.get(new Configuration())
    val outputStreamPipeline: FSDataOutputStream = hdfs.create(new Path(xgbPath))
    model.nativeBooster.saveModel(outputStreamPipeline)

    println(">> pipeline persisitence...")
    serializePipelineModel(pipelineModel, pipelinePath)

} else {
    println("The auc or logloss of the trained model is under history mean value!")
}
```


tbc