# Author: Sina Mahdipour Saravani
# Date: Nov 22, 2019
# Advanced Big Data Analytics Course Final Project

import sys
from pyspark import SQLContext
from pyspark.context import SparkContext
from pyspark.sql.functions import col
from pyspark.ml.linalg import Vectors
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from texttable import Texttable
import os
#os.environ['PYSPARK_PYTHON'] = '/usr/bin/python3'

args = sys.argv
train_set = np.zeros((1, 9))
directory = os.fsencode('datafolder/S1_Dataset')
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
print(train_set.shape)
print(train_set[0:5])

sc = SparkContext('local', 'test')
sqlContext = SQLContext(sc)

# #
# # train_df = sqlContext.read.format("com.databricks.spark.csv") \
# #     .options(header='false', inferschema=True).load(args[1])
# #
# # test_df = sqlContext.read.format("com.databricks.spark.csv") \
# #     .options(header='false', inferschema=True).load(args[2])
# #
# #
column_names = ['c_' + str(i) for i in range(10)]
train_df = sqlContext.createDataFrame(train_set, schema=column_names)
assembler = VectorAssembler(inputCols=column_names[:9], outputCol="features")
# # train_df = assembler.transform(train_df)
# # test_df = assembler.transform(test_df)
trainingData = train_df.select(['features', 'c_9'])
# # testData = test_df.select(['features', '_c13'])
# #
# # # Creating a linear regression model
lr = LinearRegression(maxIter=100, regParam=0, elasticNetParam=0, labelCol="c_9")
model = lr.fit(trainingData)

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
# print("Root Mean Squared Error (RMSE) on test data = %g" % evaluator.evaluate(predictions))
