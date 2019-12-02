# Author: Sina Mahdipour Saravani
# Date: Nov 22, 2019
# Advanced Big Data Analytics Course Final Project

import sys
from pyspark import SQLContext
from pyspark.context import SparkContext
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql.functions import col
from pyspark.ml.linalg import Vectors
from pyspark.ml.regression import LinearRegression
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import VectorAssembler, MinMaxScaler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import rand
import pandas as pd
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from texttable import Texttable
import os
# os.environ['PYSPARK_PYTHON'] = '/usr/bin/python3'

sc = SparkContext('local', 'test')
sqlContext = SQLContext(sc)

args = sys.argv

# creating a numpy array of the whole dataset
data_set = np.zeros((1, 9))
directory = os.fsencode('datafolder/S1_Dataset')
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith("M") or filename.endswith("F"):
        content = np.loadtxt(open(os.fsdecode(directory) + "/" + filename, "r"), delimiter=",")
        # print(content[0:5])
        # data_set = np.append(data_set, content, axis=0)
        data_set = np.concatenate((data_set, content), axis=0)
        # print(os.path.join(directory, filename))
        #continue
    else:
        continue
# remove the zeros row that we first initialized the array with
data_set = data_set[1:]
# np.random.shuffle(data_set)
pdarr = pd.DataFrame(data_set, columns=['_c0', '_c1', '_c2', '_c3', '_c4', '_c5', '_c6', '_c7', '_c8'])


# read all files with spark
# singleread = sqlContext.read.format("csv").option("header", "false").load("datafolder/*")
paths = "datafolder/S1_Dataset/*,datafolder/S2_Dataset/*"
singleread = sqlContext.read.format("com.databricks.spark.csv") \
    .options(header='false', inferschema=True).load(paths.split(","))
# print("# of samples read: " + str(singleread.count()))

# singleread = sqlContext.read.format("com.databricks.spark.csv") \
    # .options(header='false', inferschema=True).load("dt/combined")

# #
# # train_df = sqlContext.read.format("com.databricks.spark.csv") \
# #     .options(header='false', inferschema=True).load(args[1])
# #
# # test_df = sqlContext.read.format("com.databricks.spark.csv") \
# #     .options(header='false', inferschema=True).load(args[2])
# #
# #

# use the numpy array to create spark dataframe:
# data_df = sqlContext.createDataFrame(pdarr, schema=None)
# use the spark dataframe read by single reading:
data_df = singleread

# assembling the dataframe:
column_names = ['_c' + str(i) for i in range(8)]
assembler = VectorAssembler(inputCols=column_names[:8], outputCol="features")
data_df = assembler.transform(data_df)
# # test_df = assembler.transform(test_df)
datadf = data_df.select(['features', '_c8'])
datadf.show(5)
mmScaler = MinMaxScaler(inputCol="features", outputCol="scaled", min=0, max=1)
scaledM = mmScaler.fit(datadf)
scaledM = scaledM.transform(datadf)
print("sssss")
scaledM.show(5)
print(scaledM)
datadf = scaledM.select(['scaled', '_c8'])
datadf = datadf.withColumnRenamed('scaled', 'features')
datadf.show(5)

# tdmlp = datadf
# tdmlp = copy.deepcopy(datadf)
# # testData = test_df.select(['features', '_c13'])


# things need to be initialized before this function
def train_test_models(lr, mlp, dtree, acc_eval, f1_eval, trainset, testset):
    lr_model = lr.fit(trainset)
    lr_predictions = lr_model.transform(testset)
    lr_acc = acc_eval.evaluate(lr_predictions)
    lr_f1 = f1_eval.evaluate(lr_predictions)

    mlp_model = mlp.fit(trainset)
    mlp_predictions = mlp_model.transform(testset)
    mlp_acc = acc_eval.evaluate(mlp_predictions)
    mlp_f1 = f1_eval.evaluate(mlp_predictions)

    dtree_model = dtree.fit(trainset)
    dtree_predictions = dtree_model.transform(testset)
    dtree_acc = acc_eval.evaluate(dtree_predictions)
    dtree_f1 = f1_eval.evaluate(dtree_predictions)

    return lr_acc, lr_f1, mlp_acc, mlp_f1, dtree_acc, dtree_f1

# # # Creating a multi class logistic regression model
# lr = LogisticRegression(maxIter=100, regParam=0, elasticNetParam=0, labelCol="_c8")
# lr_model = lr.fit(datadf)
# lr_predictions = lr_model.transform(datadf)
# print('First 5 predictions of LR')
# lr_predictions.select("prediction", "_c8", "features").show(5)


# df = sqlContext.createDataFrame([(0.0, Vectors.dense([0.0, 0.0])),
# (1.0, Vectors.dense([0.0, 1.0])),
# (1.0, Vectors.dense([1.0, 0.0])),
# (0.0, Vectors.dense([1.0, 1.0]))], ["label", "features"])
# mlp2 = MultilayerPerceptronClassifier(maxIter=100, layers=[2, 2, 2], blockSize=1, seed=123)
# model2 = mlp2.fit(df)


# # # Creating a MLP model
print("count of traindata")
print(datadf.count())
layers = [8, 10, 9, 5]
# mlp = MultilayerPerceptronClassifier(maxIter=200, layers=layers, seed=12, blockSize=128, featuresCol="features", labelCol="_c8")
# mlp_model = mlp.fit(datadf)
# mlp_predictions = mlp_model.transform(datadf)
# print('First 5 predictions of MLP')
# mlp_predictions.select("prediction", "_c8", "features").show(5)


# # # Creating a DecisionTree model
# dtree = DecisionTreeClassifier(labelCol="_c8")
# dtree_model = dtree.fit(datadf)
# dtree_predictions = dtree_model.transform(datadf)

# #
# # # print the coefficients table:
# # t = Texttable()
# # t.set_max_width(0)
# # headers = ['c_' + str(i) for i in range(1, 14)]
# # headers = ['Intercept'] + headers
# # vals = np.append(model.intercept, np.array(model.coefficients))
# # t.add_rows([headers, vals])
# # print(t.draw())
# #
# # predictions = model.transform(testData)
# # # print('First 5 predictions...')
# # # predictions.select("prediction","_c13","features").show(5)
#
#
# # draw the figure for predictions vs. ground truth
# dt = predictions.toPandas()
# plt.scatter(dt['prediction'], dt['_c13'])
# plt.xlabel("Prediction")
# plt.ylabel("Ground truth")
# x1 = [0, np.maximum(np.max(dt['prediction']), np.max(dt['_c13']))]
# y1 = [0, np.maximum(np.max(dt['prediction']), np.max(dt['_c13']))]
# plt.plot(x1, y1)
# plt.show()
#

# Cross Validation Implementation:
datadf.orderBy(rand())
print("xxxxxxxxx")
print(datadf.count())
kfolds = 10
splits = datadf.randomSplit([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], 1)
splits[0].show(5)
splits[1].show(5)
# print(splits[2])
# print(splits[2].count())
x = splits[0].filter((col('_c8') < 0))
print("empty x is like this:")
x.show(5)
for k in range(kfolds):
    x = x.filter((col('_c8') < 0))
    for i in range(k):
        # print("i am in here")
        x = x.union(splits[i])
    for j in range(k+1, kfolds):
        # print("i am in there")
        x = x.union(splits[j])
    if k == 0 or k == 1:
        print("number of train samples:")
        print(x.count())
        print("number of test samples:")
        print(splits[k].count())
        print("train samples:")
        x.show(5)
        print("test samples:")
        splits[k].show(5)
    # count = (datadf.count()/kfolds)
    # t1 = datadf.head(count)
    # d2 = datadf.subtract(t1)
    # t2 = d2.head(count)




# evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="_c13", metricName="rmse")
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="_c8", metricName="accuracy")
# print("Accuracy for LR model = %g" % evaluator.evaluate(lr_predictions))
# print("Accuracy for MLP model = %g" % evaluator.evaluate(mlp_predictions))
# print("Accuracy for Decision Tree model = %g" % evaluator.evaluate(dtree_predictions))

# trying the spark cross validation module:
# paramGrid = ParamGridBuilder() \
#     .addGrid(mlp.maxIter, [1, 200]) \
#     .build()
# crossval = CrossValidator(estimator=mlp,
#                           estimatorParamMaps=paramGrid,
#                           evaluator=evaluator,
#                           numFolds=5)
# cvModel = crossval.fit(trainingData)
# res = cvModel.transform(trainingData)
# print("Accuracy for CV MLP model = %g" % evaluator.evaluate(res))

sc.stop()
