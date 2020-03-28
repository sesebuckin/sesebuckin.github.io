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

Scala：https://www.scala-lang.org/ 

Spark：https://spark.apache.org/docs/2.3.0/

XGBoost4j-Spark：https://xgboost.readthedocs.io/en/release_0.82/jvm/xgboost4j_spark_tutorial.html

本文代码版本：

工程: Maven

Scala: 2.11.8

Spark: 2.3.1

XGBoost4j-Spark: 0.82

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


tbc