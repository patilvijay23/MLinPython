#!/usr/bin/env python
# coding: utf-8

# # Using XGBoost with PySpark
# Following notebook showcases an example for using XGBoost with PySpark.
# 
# XGBoost does not provide a PySpark API in Spark, it only provides Scala and other APIs.
# Hence we will be using a custom python wrapper for XGBoost [from here](https://github.com/dmlc/xgboost/pull/4656#issuecomment-510693296).
# 
# We will be using Spark 2.4.5 with XGBoost 0.9 as it is one the working version pairs.
# 
# This uses dummy sales data.
# 
# ***
# 
# <b>Spark 2.4.5</b> (with Python 3.7) has been used for this notebook- <b>Notebook run on AWS EMR cluster with EMR version 5.30.2.</b><br>
# Refer to [spark documentation](https://spark.apache.org/docs/2.4.5/api/sql/index.html) for help with <b>data ops functions</b>.<br>
# Refer to [this article](https://medium.com/analytics-vidhya/installing-and-using-pyspark-on-windows-machine-59c2d64af76e) to <b>install and use PySpark on Windows machine</b> and [this article](https://patilvijay23.medium.com/installing-and-using-pyspark-on-linux-machine-e9f8dddc0c9a) to <b>install and use PySpark on Linux machine</b>.

# ## Getting the required files
# XGBoost 0.9 Jar files:
# - `wget https://repo1.maven.org/maven2/ml/dmlc/xgboost4j/0.90/xgboost4j-0.90.jar`
# - `wget https://repo1.maven.org/maven2/ml/dmlc/xgboost4j-spark/0.90/xgboost4j-spark-0.90.jar`
# 
# OR
# 
# - `curl -o xgboost4j-0.90.jar https://repo1.maven.org/maven2/ml/dmlc/xgboost4j/0.90/xgboost4j-0.90.jar`
# - `curl -o xgboost4j-spark-0.90.jar https://repo1.maven.org/maven2/ml/dmlc/xgboost4j-spark/0.90/xgboost4j-spark-0.90.jar`
# 
# Wrapper code
# - `wget -O pyspark-xgboost_0.90.zip https://github.com/dmlc/xgboost/files/3384356/pyspark-xgboost_0.90_261ab52e07bec461c711d209b70428ab481db470.zip`
# 
# Setup:
# - Note the path to the two jar files and the zipped package file

# ### Building a spark session
# To create a SparkSession, use the following builder pattern:
#  
# `spark = SparkSession\
#     .builder\
#     .master("local")\
#     .appName("Word Count")\
#     .config("spark.some.config.option", "some-value")\
#     .getOrCreate()`
# 
# We will use `.config("spark.jars", "/path/jar1.jar,/path/jar2.jar")` to add the required XGBoost jars to the session.

# CLI command: pyspark --executor-cores=1 --executor-memory=7g --driver-memory=4g --name vp --jars /home/hadoop/xgboost4j-0.90.jar,/home/hadoop/xgboost4j-spark-0.90.jar

# In[1]:
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator

import numpy as np
import pandas as pd
from time import time

# In[2]:
#initiating spark session
# spark.stop()

# In[3]:
# spark = SparkSession    .builder    .appName("xgboost")    .config("spark.executor.memory", "1g")    .config("spark.driver.memory", "1g")    .config("spark.jars", "/home/hadoop/xgboost4j-0.90.jar,/home/hadoop/xgboost4j-spark-0.90.jar")    .getOrCreate()

# In[4]:
spark

# In[5]:
spark.sparkContext.setLogLevel('WARN')


# <b>Import the wrapper</b>

# In[30]:
# add zipped module to session
spark.sparkContext.addPyFile("/home/hadoop/pyspark-xgboost_0.90.zip")

# In[31]:
from sparkxgb import XGBoostClassifier, XGBoostRegressor

# ## Data prep
# 
# We will use the model training dataset created in the Rolling Window features notebook- 3_rolling_window_features.
# 
# The dataset has two y variables, one for each classification and regression tasks. We will use those to build classification and regression models using XGBoost.

# ### Read input dataset

# In[9]:
path_pre = 's3://<s3 path>'

# In[10]:
df_features = spark.read.parquet(path_pre+'rw_features/')

# ### Dataset for modeling

# <b>Sample one week_end per month</b>

# In[11]:
df_wk_sample = df_features.select('week_end').withColumn('month', F.substring(F.col('week_end'), 1,7))
df_wk_sample = df_wk_sample.groupBy('month').agg(F.max('week_end').alias('week_end'))

df_wk_sample = df_wk_sample.repartition(1).persist()
df_wk_sample.count()

# In[12]:
df_wk_sample.sort('week_end').show(5)

# In[13]:
# join back to filer
df_model = df_features.join(F.broadcast(df_wk_sample.select('week_end')), on=['week_end'], how='inner')

# <b>Eligibility filter</b>: Customer should be active in last year w.r.t the reference date

# In[14]:
# use sales_52w for elig. filter
df_model = df_model.where(F.col('sales_52w')>0)

# <b>Removing latest 4 week_end dates</b>: As we have a look-forward period of 4 weeks, latest 4 week_end dates in the data cannot be used for our model as these do not have 4 weeks ahead of them for the y-variable.

# In[15]:
# see latest week_end dates (in the dataframe prior to monthly sampling)
df_features.select('week_end').drop_duplicates().sort(F.col('week_end').desc()).show(5)

# In[16]:
# filter
df_model = df_model.where(F.col('week_end')<'2020-11-14')

# In[17]:
# fillna
df_model = df_model.fillna(0)

# ### Train-Test Split
# 
# 80-20 split

# In[18]:
train, test = df_model.randomSplit([0.8, 0.2], seed=125)

# In[19]:
train.columns

# ## Classification

# ### Model Dataset Summary
# Let's look at event rate for our dataset and also get a quick summary of all features.
# 
# The y-variable is balanced here because it is a dummy dataset. <mark>In most actual scenarios, this will not be balanced and the model build exercise will involving sampling for balancing.</mark>

# In[20]:
train.groupBy('purchase_flag_next_4w').count().sort('purchase_flag_next_4w').show()

# In[21]:
test.groupBy('purchase_flag_next_4w').count().sort('purchase_flag_next_4w').show()

# ### Pre-Processing pipeline
# The below pipeline only does pre-processing and saves it to be used for scoring.
# 
# You can also add the model step to this to have a single pipeline instead of two that I have created. Though having two pipelines makes it easier to iterate through just the model step during training.

# In[22]:
# list of features: remove identifier columns and the y-var
col_list = df_model.drop('week_end','customer_id','min_week','max_week','purchase_flag_next_4w','sales_next_4w').columns

stages = []
assembler = VectorAssembler(inputCols=col_list, outputCol='features')
stages.append(assembler)

pipe = Pipeline(stages=stages)
pipe_model = pipe.fit(train)

pipe_model.write().overwrite().save('./files/model_objects/xgb_clf_pipe/')

# In[23]:
pipe_model = PipelineModel.load('./files/model_objects/xgb_clf_pipe/')

# <b>Apply the transformation pipeline</b>
# 
# Also keep the identifier columns and y-var in the transformed dataframe.
# 
# <mark>We are keeping both the classification and regression y-vars here as we will be re-using the same processed dataset for the regression section.</mark>

# In[24]:
train_pr = pipe_model.transform(train)
train_pr = train_pr.select('customer_id','week_end','purchase_flag_next_4w','sales_next_4w','features')
train_pr = train_pr.persist()
train_pr.count()

# In[25]:
test_pr = pipe_model.transform(test)
test_pr = test_pr.select('customer_id','week_end','purchase_flag_next_4w','sales_next_4w','features')
test_pr = test_pr.persist()
test_pr.count()

# ### Model Training
# We will train one iteration of XGBoost classification model as showcase.
# 
# In actual scenario, you will have to iterate through the training step multiple times for feature selection and model hyper parameter tuning to get a good final model.

# In[26]:
train_pr.show(5)

# In[27]:
# XGBoost model requires a PipeLine object for the save and load steps to work properly
stages = []

# In[32]:
xgboost=XGBoostClassifier(
    featuresCol="features",
    labelCol="purchase_flag_next_4w",
    predictionCol="prediction",
    objective="binary:logistic",
    maxDepth=8,
    missing=0.0,
    subsample=0.7,
    numRound=16,
    numWorkers=8)

# In[ ]:
stages.append(xgboost)
pipe = Pipeline(stages=stages)

start_time = time()
model = pipe.fit(train_pr)
print('time elapsed: ', np.round(time()-start_time,2),'s',sep='')

model.write().overwrite().save('./files/model_objects/xgb_clf_model/')

model = PipelineModel.load(path='./files/model_objects/xgb_clf_model/')

# In[69]:
## Feature importance
feat_imp = str(model.stages[0].nativeBooster.getScore("","gain"))
feature_importance_list = [x.split(' -> ') for x in feat_imp[4:-1].split(', ')]
feature_importance_list = pd.DataFrame(data=feature_importance_list, columns=['idx','gain_importance'])
feature_importance_list['idx'] = feature_importance_list['idx'].apply(lambda x: int(x[1:]))
feature_importance_list['gain_importance'] = feature_importance_list['gain_importance'].apply(lambda x: float(x))

try:
    feature_list = pd.DataFrame(
        train_pr.schema['features'].metadata['ml_attr']['attrs']['numeric'] +
        train_pr.schema['features'].metadata['ml_attr']['attrs']['nominal']).sort_values('idx')
except:
    feature_list = pd.DataFrame(
        train_pr.schema['features'].metadata['ml_attr']['attrs']['numeric']).sort_values('idx')

feature_list = feature_list.merge(feature_importance_list, on=['idx'], how='left')
feature_list = feature_list.sort_values('gain_importance', ascending=False)

feature_list.to_csv("./files/feature_importance_cl_xgb.csv", index=False)

# ### Predict on train and test
# In[72]:
secondelement = F.udf(lambda v: float(v[1]), FloatType())

train_pred = model.transform(train_pr).withColumn('score',secondelement(F.col('probability')))
test_pred =  model.transform(test_pr).withColumn('score', secondelement(F.col('probability')))

# In[73]:
test_pred.show(5)

# ### Test Set Evaluation

# In[74]:
evaluator = BinaryClassificationEvaluator(
        rawPredictionCol='rawPrediction',
        labelCol='purchase_flag_next_4w',
        metricName='areaUnderROC')

# In[75]:
# areaUnderROC
evaluator.evaluate(train_pred)

# In[76]:
evaluator.evaluate(test_pred)

# In[77]:
# cm
test_pred.groupBy('purchase_flag_next_4w','prediction').count().sort('purchase_flag_next_4w','prediction').show()

# In[78]:
# accuracy
test_pred.where(F.col('purchase_flag_next_4w')==F.col('prediction')).count()/test_pred.count()

# ## Scoring
# We will take the records for latest week_end from df_features and score it using our trained model.

# In[81]:
df_features = spark.read.parquet(path_pre+'rw_features/')

# In[82]:
max_we = df_features.selectExpr('max(week_end)').collect()[0][0]
max_we

# In[88]:
df_scoring = df_features.where(F.col('week_end')==max_we)

# In[89]:
df_scoring.count()

# In[90]:
# fillna
df_scoring = df_scoring.fillna(0)

# transformation pipeline
pipe_model = PipelineModel.load('./files/model_objects/xgb_clf_pipe/')

# apply
df_scoring = pipe_model.transform(df_scoring)
df_scoring = df_scoring.select('customer_id','week_end','features')

# xgb model
model = PipelineModel.load(path='./files/model_objects/xgb_clf_model/')

#apply
secondelement = F.udf(lambda v: float(v[1]), FloatType())

df_scoring = model.transform(df_scoring).withColumn('score',secondelement(F.col('probability')))

# In[91]:
df_scoring.show(5)

# In[92]:
# save scored output
df_scoring.repartition(8).write.parquet(path_pre+'rw_scored_clf_xgb/', mode='overwrite')

# In[ ]:
# ## Regression: PENDING

