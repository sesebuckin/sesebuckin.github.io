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

### 获取每一列unique values
```python
# select unique values in each column
texts = set()
for col in text_cols:
    tmp_data = [i[col] for i in df.select(col).distinct().collect()]
    texts.update(tmp_data)
```

在SQL中可以通过
```text
select distinct(colname) from table
```
实现，
在pandas中是
```python
df[colname].unique()
```
Spark的操作与SQL类似，通过distinct()与collect()，得到列col下的unique值列表，
通过遍历需要查询的col，将所有值update到一个set中，保证不出现重复值。

### 准备Word2Vec预料

为了后续训练分类模型，针对数据中的文本特征，我们希望将其转换为词向量表示，这里我们使用Word2Vec来训练词向量。

通过去重操作得到的文本list，首先通过正则表达式去除不需要的内容与符号，然后调用中文分词包jieba进行分词。
```python
texts_trans = [text_model.re_module(item) for item in texts]
texts_trans = text_model.jieba_module(texts_trans, 'word2vec')
```
可以根据业务需要，事先人为规定特殊词（即希望jieba在分词的时候，保留这些特殊词，
例如“无门槛”可能会被分为“无”、“门槛”，如果希望保留“无门槛”这个语义，则可以通过jieba.suggest_freq(*)来实现）：
```python
import jieba
import jieba.analyse

for word in jieba_words:
    jieba.suggest_freq(word, True)
```
分词结果示例：
```text
分词前： 机票报销凭证
分词后： 机票 报销 凭证
```
### 构建Word2Vec模型
准备好分词后的预料，则可以通过gensim.models.Word2Vec模块来构建模型。

完整流程示例如下：
```python
from gensim.models import Word2Vec
import numpy as np
import jieba
import jieba.analyse

class TextModel:

    def __init__(self, cfg):
        self.cfg = cfg

    def jieba_module(self, data, section):
        jieba_words = self.cfg.get(section, "jieba_words").split(',')
        for word in jieba_words:
            jieba.suggest_freq(word, True)
        data = [" ".join(jieba.cut(str(e))) for e in data]
        return data

    def build(self, data, section):
        min_count = int(self.cfg.get(section, "min_count"))
        size = int(self.cfg.get(section, "size"))
        window = int(self.cfg.get(section, "window"))
    
        model = Word2Vec(data, min_count=min_count, size=size, window=window, workers=4)
        return model
    
    def text2vec(self, model, section, row):
        vec = np.zeros(50)  # here uses numpy dtype
        count = 0
        for word in row:
            try:
                vec += model[word]
                count += 1
            except:
                pass
        if count == 0:
            return vec
        return vec/count

text_model = TextModel(cfg)

texts_trans = [text_model.re_module(item) for item in texts]
texts_trans = text_model.jieba_module(texts_trans, 'word2vec')

model = text_model.build(texts_trans, 'word2vec')  # build word2vec model
vectors = [text_model.text2vec(model, 'word2vec', t) for t in texts_trans]
vector_columns = ["vector_{}".format(i) for i in range(len(vectors[0]))]
```
我们来学习一下Word2Vec的具体用法，首先是类定义：
```python
class Word2Vec(utils.SaveLoad):
    def __init__(
            self, sentences=None, size=100, alpha=0.025, window=5, min_count=5,
            max_vocab_size=None, sample=1e-3, seed=1, workers=3, min_alpha=0.0001,
            sg=0, hs=0, negative=5, cbow_mean=1, hashfxn=hash, iter=5, null_word=0,
            trim_rule=None, sorted_vocab=1, batch_words=MAX_WORDS_IN_BATCH):
```
部分参数含义（具体文档请转[gensim官方API](https://radimrehurek.com/gensim/models/word2vec.html)）：
```text
size：词向量维度，一般在50以上
window：窗口大小，表示当前词与预测词在句子中的最大距离
min_count：词频小于此值的词会被忽略
```
训练完毕后得到model。在原始数据集上，text2vec函数通过model计算文本词向量的平均值，
即先求和得到vec，再平均vec/count，作为文本的最终表示。

![word2vec](/img/post-pyspark-word2vec.PNG)

### 词向量序列化、持久化
我们得到了文本经过转换后的词向量表示。为了方便online预测时能快速得到文本的向量化表示，
可以在每次离线模型训练完后，将得到的词向量表示序列化后，再进行持久化保存到某个db；
online预测时，拉取并进行反序列化，这样是最快速的方法。

一般来说，如果只是用于Python，可以采用pickle来进行序列化。为了支持跨语言读取
（这里很重要，模型训练与线上预测很多时候是Scala、Java、C++等其他语言），
这里采用**json**来进行序列化与反序列化。需要注意NaN问题（见注释）：
```python
# word2vec will generate NaN, after json serialization and setting to redis
# the value is still NaN (which is only recognizable in numpy, not in Java or C++ etc.)
# so we need to fill the nan with correct value (here we fill in with float 0.0)

# serialize the dict object using JSON, not pickle
# in case we can load this json obj through any other programming language

json_obj = json.dumps(res)
```

然后采用自定义的RedisHelper对象，将序列化后的数据缓存到redis集群：
```python
Model2RedisHelper().setFile2redis(f=json_obj, name=name) 
```

![redis](/img/post-pyspark-redis.PNG)

### DataFrame存储到hive
在一系列操作之后，df对象各个字段的值类型或许不是我们想要的结果。
因此，可以根据自己需要，更改DataFrame每个字段的数值类型：
```python
for col in df.columns:
    df = df.withColumn(col, F.col(col).cast(type_dict[col]))
```
这里的type_dict是预先定义好的“字段名-类型”的字典。

在将df落表到Hive之前，需要注意：在全部的数据处理过程中，
如果使用了numpy模块，则有可能产生numpy类型的数据格式，如numpy.float64等。
Spark不支持numpy的数据类型，因此会报错。Spark对python内置的数据类型是支持的。
因此有必要过滤、处理df中数值字段的type。

示例：
```python
# spark do not support numpy data type
# need to do transferring before assign to spark
# otherwise it will occur error

values = [float(v) for v in values]  # from np.float64 to float
```

### 后记

这篇博客的产生原因，是我最开始全称用Pandas来处理数据，然而数据量一上去，效率就很低，原先job平均要跑20分钟。
改写为PySpark处理后，job平均运行时长缩短为2分钟。

你可能会问，既然是Spark为什么不直接写Scala。我最开始用PySpark的原因，是因为想用word2vec的python第三方包。
然而这篇的代码写完了没多久，我发现原来Spark的mllib也有word2vec......下次试试看吧。

在操作数据的时候，我们检查代码的合理性，一是通过debug来查看每一步的中间值；二是校验最终落的数据是否有问题。
操作Spark的时候，因为需要提交代码到集群运行，在本地很难做debug，所以校验数据的正确性相对来说需要花更多时间。但只要保持耐心，认真做review，
相信bug会越来越少的！

