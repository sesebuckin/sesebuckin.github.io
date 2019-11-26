---
layout:     post
title:      使用flask+gunicorn+supervisor进行python web部署
subtitle:   
date:       2019-11-26
author:     Schoko
header-img: 
catalog: true
tags:
    - 机器学习
    - 模型部署
    - python
    - xgboost
    - Flask
    - Gunicore
    - Supervisor
---

最近需要直接在服务器上部署xgboost模型。之前也做过模型部署方式的调研，具体需要根据实际情况来选择合适的部署方式。由于项目使用python进行开发，所以选择部署为python web service是最方便的做法。

之前实习的时候，开发帮忙部署过一次python web，我对其中细节其实并不太清楚。这次正好自己在公司的云平台上实验了整个流程，在此记录步骤。

主要参考了这篇[教程](https://juejin.im/post/5a30f7f0f265da43346fe8b5)。

### 部署方式

- flask：python的服务器框架
- gunicore：webservice，WSGI的容器
- supervisor：进程管理工具

### 流程

#### 安装依赖包

```shell script
sudo pip install Flask
sudo pip install Flask-RESTful
sudo pip install gunicorn
sudo pip install supervisor

pip install -r requirements.txt # 具体项目所依赖的包
```

#### wsgi.py脚本

具体的例子如下，有了这个框架，以后部署python web service可以照葫芦画瓢。

```python
from coupon_xgb import CouponPredict
from sklearn.externals import joblib
from flask import Flask, jsonify, request
from flask_restful import Api, Resource
import datetime
import json
import os
import time

app = Flask(__name__)
maps = {}

cur_date = datetime.datetime.now().strftime("%Y_%m_%d")
model_path = './models'

@app.route('/vi/health')
def index():
    return "hello world"

@app.route('/train', methods=['POST'])
def train():
    if request.method == 'POST':
        start = time.time()
        data = request.get_data()
        json_data = json.loads(data)
        promotion_id = json_data.get('promotion_id')
        index_name = json_data.get('index_name')
        if promotion_id == "":
            return jsonify({"result_no": 2, "results_msg": "策略编号不能为空"})
        else:
            if not os.path.isfile('{0}/{1}-{2}.model'.format(model_path, promotion_id, cur_date)):
                instance = CouponPredict(index_name, promotion_id)
                instance.train()
                return jsonify({"result_no": 0, "result_msg": "模型训练完成", "time(s)": round(time.time()-start, 2)})
            else:
                return jsonify({"result_no": 3, "result_msg": "模型已存在"})
    else:
        return jsonify({"result_no": 1, "result_msg": "请求方式不正确"})
           
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        start = time.time()
        data = request.get_data()
        json_data = json.loads(data)
        promotion_id = json_data.get('promotion_id')
        index_name = json_data.get('index_name')
        future_steps = json_data.get('future_steps')
        if promotion_id == "":
            return jsonify({"result_no": 2, "results_msg": "策略编号不能为空"})
        else:
            if not os.path.isfile('{0}/{1}-{2}.model'.format(model_path, promotion_id, cur_date)):
                return jsonify({"result_no": 3, "result_msg": "模型文件不存在"})
            model = joblib.load('{0}/{1}-{2}.model'.format(model_path, promotion_id, cur_date))
            instance = CouponPredict(index_name, promotion_id)
            predictions = instance.predict(model, future_steps)
            return jsonify({"result_no": 0, "result_msg": "预测成功", "predictions": predictions,"time(s)": round(time.time()-start, 2)})
    else:
        return jsonify({"result_no": 1, "result_msg": "请求方式不正确"})
    
if __name__ == "__main__":
    app.run()
```

到这一步为止是我以前熟知的部署方式，即只使用flask。然而flask自带的服务器一般用于开发环境，而不适用于直接部署到生产环境，主要原因是性能问题。

实际生产环境中，使用gunicore作为wsgi的容器，进而来进行部署python web。

#### 配置gunicore

直接通过gunicore启动flask：

```shell script
gunicore -w 4 -b 0.0.0.0:5000 wsgi:app
```

这里设置ip为0.0.0.0，在访问时输入服务器实际ip地址即可，端口号设为5000。**-w 4**表示开启了4个worker，**wsgi**为.py脚本名，**app**
为脚本中开启的app程序。

结束gunicore进程可以通过pkill gunicore，有时需要知道PID，实际操作起来比较麻烦。因此，可以进一步通过进程管理工具supervisor来统一管理。

#### 配置supervisor

首先生成默认配置文件：

```shell script
echo_supervisord_conf > supervisor.conf
vim supervisor.conf
```

在底部添加新增的program信息：

```editorconfig
[program:wsgi]
command=/opt/app/conda/bin/gunicorn -w4 -b 0.0.0.0:5000 wsgi:app                        ; supervisor启动命令
directory=/home/powerop/work/project/coupon/mkt_coupon_predict/coupon_xgboost           ; 项目的文件夹路径
startsecs=0                                                                             ; 启动时间
stopwaitsecs=0                                                                          ; 终止等待时间
autostart=false                                                                         ; 是否自动启动
autorestart=false                                                                       ; 是否自动重启
stdout_logfile=/home/powerop/work/gunicorn.log                                          ; log日志
stderr_logfile=/home/powerop/work/gunicorn.err  
```

注意文件路径不要出错。

#### 通过supervisor开启/关闭进程

```shell script
supervisord -c supervisor.conf                             # 通过配置文件启动supervisor
supervisorctl -c supervisor.conf status                    # 察看supervisor的状态
supervisorctl -c supervisor.conf reload                    # 重新载入 配置文件
supervisorctl -c supervisor.conf start [all]|[appname]     # 启动指定/所有 supervisor管理的程序进程
supervisorctl -c supervisor.conf stop [all]|[appname]      # 关闭指定/所有 supervisor管理的程序进程
```

打开web service：

```shell script
supervisorctl -c supervisor.conf start wsgi   # 开启
```

此时可以通过postman进行测试，请求train：

![train](/img/post-pythonweb-pic1.PNG)

得到response：

![response](/img/post-pythonweb-pic2.PNG)

请求predict并返回结果：

![predict](/img/post-pythonweb-pic3.PNG)

关闭web service：

```shell script
supervisorctl -c supervisor.conf stop wsgi   # 关闭
```



