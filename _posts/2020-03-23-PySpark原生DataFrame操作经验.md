---
layout:     post
title:      PySpark原生DataFrame操作经验
subtitle:   
date:       2020-03-23
author:     Schoko
header-img: 
catalog: true
tags:
    - PySpark
---

日常项目重度依赖PySpark。之前一直只把pyspark作为读取写入数据的client，主要数据处理逻辑在pandas中实现。

然而pandas的确只适合处理小数据量的场景。toPandas()一次性将数据全部加载到内存，这是其弊端之一；其二，当数据量较大时，用pandas处理耗时相对也会更长。

如果不需要强依赖一些第三方包，那么可以选择scala编写spark来进行数据处理；在必须依赖第三方python包的脚本里，例如需要nlp模块对文本进行处理的时候，为了提升大数据处理的效率，则可以采用spark原生的DataFrame来进行支持，在分布式集群中进行计算。

## PySpark DataFrame

详细信息可查看[官方API](https://spark.apache.org/docs/2.3.0/api/python/pyspark.sql.html#module-pyspark.sql.types)

![spark_api](/img/post-pyspark-sparkapi.PNG)

## 结合word2vec的示例

这里主要记录一个结合word2vec模块的例子，从中总结一些word2vec与pyspark的应用。

先放上主要代码：

```python

from pyspark.sql.types import *
from pyspark.sql import functions as F
from configparser import ConfigParser
import time
import pandas as pd

from features.nlp_modules import *
from utils.SparkHelper import SparkHelper

def nlp_process(df, helper):
    text_model = TextModel(cfg)
    text_cols = cfg.get('word2vec', 'cols').split(',')

    # select unique values in each column
    texts = set()
    for col in text_cols:
        tmp_data = [i[col] for i in df.select(col).distinct().collect()]
        texts.update(tmp_data)
    na_v = [np.nan, 'NaN', '']
    for i in na_v:
        texts.discard(i)
    texts = list(texts)

    data_map = dict()
    texts_trans = [text_model.re_module(item) for item in texts]
    texts_trans = text_model.jieba_module(texts_trans, 'word2vec')

    model = text_model.build(texts_trans, 'word2vec')  # build word2vec model
    vectors = [text_model.text2vec(model, 'word2vec', t) for t in texts_trans]
    vector_columns = ["vector_{}".format(i) for i in range(len(vectors[0]))]

    # build schema for each column
    struct_fields = [StructField('text_content', StringType(), True)]
    struct_fields.extend([StructField(name, FloatType(), True) for name in vector_columns])
    schema = StructType(struct_fields)

    # create text DataFrame by adding each column
    # spark do not support numpy data type
    # need to do transferring before assign to spark
    # otherwise it will occur error
    tmp_df = pd.DataFrame()
    tmp_df['text_content'] = texts

    text_vectors = np.array(vectors)  # (36, 50)

    for i in range(text_vectors.shape[1]):
        values = list(text_vectors[:, i])
        values = [float(v) for v in values]  # from np.float64 to float
        tmp_df[vector_columns[i]] = values
    tmp_df = helper.spark.createDataFrame(tmp_df, schema)

    # join df with tmp_df
    df = join_with_text_trans(df, tmp_df, text_cols, vector_columns)

    return df
```

tbc