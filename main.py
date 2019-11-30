# Author: Sina Mahdipour Saravani
# Date: Nov 22, 2019
# Advanced Big Data Analytics Course Final Project

import sys
from pyspark import SQLContext
from pyspark.context import SparkContext
from pyspark.sql.functions import col
from pyspark.ml.linalg import Vectors
from pyspark.ml.regression import LinearRegression
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import pandas as pd
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from texttable import Texttable
import os
#os.environ['PYSPARK_PYTHON'] = '/usr/bin/python3'

sc = SparkContext('local', 'test')
sqlContext = SQLContext(sc)


args = sys.argv
train_set = np.zeros((1, 9))
directory = os.fsencode('datafolder/S1_Dataset')
# singleread = sqlContext.read.format("csv").option("header", "false").load("datafolder/*")
paths = "datafolder/S1_Dataset/*,datafolder/S2_Dataset/*"
singleread = sqlContext.read.format("com.databricks.spark.csv") \
    .options(header='false', inferschema=True).load(paths.split(","))
# print("readdddddddddddddddddd")
# print(singleread.show(5))
print("# of samples read: " + str(singleread.count()))
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith("M") or filename.endswith("F"):
        content = np.loadtxt(open(os.fsdecode(directory) + "/" + filename, "r"), delimiter=",")
        # print(content[0:5])
        # train_set = np.append(train_set, content, axis=0)
        train_set = np.concatenate((train_set, content), axis=0)
        # print(os.path.join(directory, filename))
        #continue
    else:
        continue
# remove the zeros row that we first initialized the array with
train_set = train_set[1:]
# print(train_set.shape)
# np.random.shuffle(train_set)
# print(train_set[0:5])
pdarr = pd.DataFrame(train_set, columns=['_c0', '_c1', '_c2', '_c3', '_c4', '_c5', '_c6', '_c7', '_c8'])
# print(pdarr.head())



# #
# # train_df = sqlContext.read.format("com.databricks.spark.csv") \
# #     .options(header='false', inferschema=True).load(args[1])
# #
# # test_df = sqlContext.read.format("com.databricks.spark.csv") \
# #     .options(header='false', inferschema=True).load(args[2])
# #
# #
column_names = ['_c' + str(i) for i in range(8)]
# train_df = sqlContext.createDataFrame(pdarr, schema=None)
train_df = singleread
assembler = VectorAssembler(inputCols=column_names[:8], outputCol="features")
train_df = assembler.transform(train_df)
# # test_df = assembler.transform(test_df)
trainingData = train_df.select(['features', '_c8'])
# tdmlp = trainingData
# tdmlp = copy.deepcopy(trainingData)
# # testData = test_df.select(['features', '_c13'])
# #
# # # Creating a multi class logistic regression model
lr = LogisticRegression(maxIter=100, regParam=0, elasticNetParam=0, labelCol="_c8")
lr_model = lr.fit(trainingData)
lr_predictions = lr_model.transform(trainingData)
# print(predictions.dtype)
# narr = np.array(predictions.select('prediction'))
# print(narr)
# narr = np.rint(narr)
# predictions['prediction'] = narr
print('First 5 predictions of LR')
lr_predictions.select("prediction", "_c8", "features").show(5)


# df = sqlContext.createDataFrame([(0.0, Vectors.dense([0.0, 0.0])),
# (1.0, Vectors.dense([0.0, 1.0])),
# (1.0, Vectors.dense([1.0, 0.0])),
# (0.0, Vectors.dense([1.0, 1.0]))], ["label", "features"])
# mlp2 = MultilayerPerceptronClassifier(maxIter=100, layers=[2, 2, 2], blockSize=1, seed=123)
# model2 = mlp2.fit(df)


# # # Creating a multi layer perceptron model
layers = [8, 10, 10, 8, 6, 10, 5]
mlp = MultilayerPerceptronClassifier(maxIter=10000, layers=layers, seed=326, blockSize=128, featuresCol="features", labelCol="_c8")
mlp_model = mlp.fit(trainingData)
mlp_predictions = mlp_model.transform(trainingData)
print('First 5 predictions of MLP')
mlp_predictions.select("prediction", "_c8", "features").show(5)
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
# evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="_c13", metricName="rmse")
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="_c8")
print("Accuracy for LR model = %g" % evaluator.evaluate(lr_predictions))
print("Accuracy for MLP model = %g" % evaluator.evaluate(mlp_predictions))

sc.stop()
