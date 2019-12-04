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
def read_in_np():
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
    data_df = sqlContext.createDataFrame(pdarr, schema=None)
    return data_df


def normalize_data(dataset):
    scaler_model = MinMaxScaler(inputCol="features", outputCol="scaled", min=0, max=1)
    scaled_data = scaler_model.fit(dataset)
    scaled_data = scaled_data.transform(dataset)
    norm_data = scaled_data.select(['scaled', '_c8'])
    norm_data = norm_data.withColumnRenamed('scaled', 'features')
    return norm_data


# read all files with spark
# singleread = sqlContext.read.format("csv").option("header", "false").load("datafolder/*")
paths = "datafolder/S1_Dataset/*,datafolder/S2_Dataset/*"
path_s1 = "datafolder/S1_Dataset/*"
path_s2 = "datafolder/S2_Dataset/*"
path_s1b = "dt/s1balanced/s1combb"
# singleread = sqlContext.read.format("com.databricks.spark.csv") \
    # .options(header='false', inferschema=True).load(paths.split(","))
singleread = sqlContext.read.format("com.databricks.spark.csv") \
    .options(header='false', inferschema=True).load(path_s1)
print("# of samples read: " + str(singleread.count()))

# singleread = sqlContext.read.format("com.databricks.spark.csv") \
    # .options(header='false', inferschema=True).load("dt/combined")


# use the numpy array to create spark dataframe:
# data_df = read_in_np()
# use the spark dataframe read by single reading:
data_df = singleread

# assembling the dataframe:
column_names = ['_c' + str(i) for i in range(8)]
assembler = VectorAssembler(inputCols=column_names[:8], outputCol="features")
data_df = assembler.transform(data_df)
datadf = data_df.select(['features', '_c8'])
# scaler_model = MinMaxScaler(inputCol="features", outputCol="scaled", min=0, max=1)
# scaled_data = scaler_model.fit(datadf)
# scaled_data = scaled_data.transform(datadf)
# datadf = scaled_data.select(['scaled', '_c8'])
# datadf = datadf.withColumnRenamed('scaled', 'features')

# datadf = normalize_data(datadf)


# things need to be initialized before this function
def train_test_models(lr, mlp, dtree, acc_eval, f1_eval, trainset, testset):
    lr_acc, lr_f1, mlp_acc, mlp_f1, dtree_acc, dtree_f1 = 0, 0, 0, 0, 0, 0

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


# # # Creating a MLP model
# layers = [8, 10, 9, 5]
# mlp = MultilayerPerceptronClassifier(maxIter=200, layers=layers, seed=12, blockSize=128, featuresCol="features", labelCol="_c8")
# mlp_model = mlp.fit(datadf)
# mlp_predictions = mlp_model.transform(datadf)
# print('First 5 predictions of MLP')
# mlp_predictions.select("prediction", "_c8", "features").show(5)


# # # Creating a DecisionTree model
# dtree = DecisionTreeClassifier(labelCol="_c8")
# dtree_model = dtree.fit(datadf)
# dtree_predictions = dtree_model.transform(datadf)


# evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="_c8", metricName="accuracy")
# print("Accuracy for LR model = %g" % evaluator.evaluate(lr_predictions))
# print("Accuracy for MLP model = %g" % evaluator.evaluate(mlp_predictions))
# print("Accuracy for Decision Tree model = %g" % evaluator.evaluate(dtree_predictions))



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


# Models Initialization:
lr_model = LogisticRegression(maxIter=200, regParam=0, elasticNetParam=0, labelCol="_c8")
layers = [8, 10, 9, 8, 6, 5]
mlp_model = MultilayerPerceptronClassifier(maxIter=1000, layers=layers, seed=12, blockSize=128, featuresCol="features", labelCol="_c8")
dtree_model = DecisionTreeClassifier(seed=12, labelCol="_c8")

acc_eval_model = MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="_c8", metricName="accuracy")
f1_eval_model = MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="_c8", metricName="f1")


# Cross Validation Implementation:
def cross_validation(lr, mlp, dtree, acc_eval, f1_eval, datadf):
    datadf.orderBy(rand(seed=12))
    kfolds = 10
    splits = datadf.randomSplit([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], 1)
    train_split = splits[0].filter((col('_c8') < 0))
    lras, lrfs, mlpas, mlpfs, dtreeas, dtreefs = 0, 0, 0, 0, 0, 0
    for k in range(kfolds):
        train_split = train_split.filter((col('_c8') < 0))
        # joining the ones before k
        for i in range(k):
            train_split = train_split.union(splits[i])
        # joining the ones after k
        for j in range(k+1, kfolds):
            train_split = train_split.union(splits[j])
        test_split = splits[k]
        train_split = normalize_data(train_split)
        test_split = normalize_data(test_split)
        lr_acc, lr_f1, mlp_acc, mlp_f1, dtree_acc, dtree_f1 = train_test_models(lr, mlp, dtree, acc_eval, f1_eval,
                                                                                train_split, test_split)
        lras = lras + lr_acc
        lrfs = lrfs + lr_f1
        mlpas = mlpas + mlp_acc
        mlpfs = mlpfs + mlp_f1
        dtreeas = dtreeas + dtree_acc
        dtreefs = dtreefs + dtree_f1

    return lras/kfolds, lrfs/kfolds, mlpas/kfolds, mlpfs/kfolds, dtreeas/kfolds, dtreefs/kfolds


def holdout(lr, mlp, dtree, acc_eval, f1_eval, datadf):
    datadf.orderBy(rand(seed=12))
    splits = datadf.randomSplit([0.8, 0.2], 1)
    train_split = normalize_data(splits[0])
    test_split = normalize_data(splits[1])
    lr_acc, lr_f1, mlp_acc, mlp_f1, dtree_acc, dtree_f1 = train_test_models(lr, mlp, dtree, acc_eval, f1_eval,
                                                                            train_split, test_split)
    return lr_acc, lr_f1, mlp_acc, mlp_f1, dtree_acc, dtree_f1


lra, lrf, mlpa, mlpf, dtreea, dtreef = cross_validation(lr_model, mlp_model, dtree_model, acc_eval_model,
                                                        f1_eval_model, datadf)

# lra, lrf, mlpa, mlpf, dtreea, dtreef = holdout(lr_model, mlp_model, dtree_model, acc_eval_model, f1_eval_model, datadf)
print("acc for lr model is " + str(lra) + " f1 for it is " + str(lrf))
print("acc for mlp model is " + str(mlpa) + " f1 for it is " + str(mlpf))
print("acc for dtree model is " + str(dtreea) + " f1 for it is " + str(dtreef))

sc.stop()
