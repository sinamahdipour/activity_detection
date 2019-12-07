# Author: Sina Mahdipour Saravani
# Date: Nov 22, 2019
# Advanced Big Data Analytics Course Final Project

import sys
from pyspark import SQLContext
from pyspark.context import SparkContext
from pyspark.sql.functions import col
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import VectorAssembler, MinMaxScaler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import rand
import pandas as pd
import numpy as np
import os
# os.environ['PYSPARK_PYTHON'] = '/usr/bin/python3'

sc = SparkContext('local', 'test')
sqlContext = SQLContext(sc)

args = sys.argv
if len(args) != 3:
    print("\nCalling arguments are incorrect. Use the sample below:")
    print("python3 main.py [-s1/-s2/-s1b/-s2b] [-ho/-cv]")
    print("-s1: use room setting 1 imbalanced data")
    print("-s2: use room setting 2 imbalanced data")
    print("-s1b: use room setting 1 balanced data")
    print("-s2b: use room setting 2 balanced data")
    print("-ho: use holdout method for evaluation")
    print("-cv: use 10-fold cross validation method for evaluation")
    exit(0)

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
paths = "datafolder/S1_Dataset/*,datafolder/S2_Dataset/*"
path_s1 = "datafolder/S1_Dataset/*"
path_s2 = "datafolder/S2_Dataset/*"
path_s1b = "dt/s1balanced/s1combb"
path_s2b = "dt/s2balanced/s2combb"
# singleread = sqlContext.read.format("com.databricks.spark.csv").options(header='false', inferschema=True).load(
# paths.split(","))

# check input arguments to figure out what dataset user wants to work with
if args[1] == '-s1':
    singleread = sqlContext.read.format("com.databricks.spark.csv") \
        .options(header='false', inferschema=True).load(path_s1)
elif args[1] == '-s2':
    singleread = sqlContext.read.format("com.databricks.spark.csv") \
        .options(header='false', inferschema=True).load(path_s2)
elif args[1] == '-s1b':
    singleread = sqlContext.read.format("com.databricks.spark.csv") \
        .options(header='false', inferschema=True).load(path_s1b)
elif args[1] == '-s2b':
    singleread = sqlContext.read.format("com.databricks.spark.csv") \
        .options(header='false', inferschema=True).load(path_s2b)
else:
    print("\nCalling arguments are incorrect. Use the sample below:")
    print("python3 main.py [-s1/-s2/-s1b/-s2b] [-ho/-cv]")
    print("-s1: use room setting 1 imbalanced data")
    print("-s2: use room setting 2 imbalanced data")
    print("-s1b: use room setting 1 balanced data")
    print("-s2b: use room setting 2 balanced data")
    print("-ho: use holdout method for evaluation")
    print("-cv: use 10-fold cross validation method for evaluation")
    exit(0)
print("# of samples read: " + str(singleread.count()))

# use the numpy array to create spark dataframe:
# data_df = read_in_np()
# use the spark dataframe read by single reading:
data_df = singleread

# assembling the dataframe:
column_names = ['_c' + str(i) for i in range(8)]
assembler = VectorAssembler(inputCols=column_names[:8], outputCol="features")
data_df = assembler.transform(data_df)
datadf = data_df.select(['features', '_c8'])


# things need to be initialized before this function
def train_test_models(lr, mlp, dtree, acc_eval, f1_eval, trainset, testset):
    lr_acc, lr_f1, mlp_acc, mlp_f1, dtree_acc, dtree_f1 = 0, 0, 0, 0, 0, 0
    # train and test the logistic regression model
    lr_model = lr.fit(trainset)
    lr_predictions = lr_model.transform(testset)
    lr_acc = acc_eval.evaluate(lr_predictions)
    lr_f1 = f1_eval.evaluate(lr_predictions)
    # train asd test MLP model
    mlp_model = mlp.fit(trainset)
    mlp_predictions = mlp_model.transform(testset)
    mlp_acc = acc_eval.evaluate(mlp_predictions)
    mlp_f1 = f1_eval.evaluate(mlp_predictions)
    # train and test decision tree model
    dtree_model = dtree.fit(trainset)
    dtree_predictions = dtree_model.transform(testset)
    dtree_acc = acc_eval.evaluate(dtree_predictions)
    dtree_f1 = f1_eval.evaluate(dtree_predictions)
    # return evaluation metrics
    return lr_acc, lr_f1, mlp_acc, mlp_f1, dtree_acc, dtree_f1


# Models Initialization:
# classification models:
lr_model = LogisticRegression(maxIter=200, labelCol="_c8", regParam=0, elasticNetParam=0)
layers = [8, 50, 30, 8, 6, 5]
mlp_model = MultilayerPerceptronClassifier(maxIter=400, layers=layers, seed=12, blockSize=128, featuresCol="features", labelCol="_c8")
dtree_model = DecisionTreeClassifier(seed=12, labelCol="_c8")
# evaluation models:
acc_eval_model = MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="_c8", metricName="accuracy")
f1_eval_model = MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="_c8", metricName="f1")


# Cross Validation Implementation:
def cross_validation(lr, mlp, dtree, acc_eval, f1_eval, datadf):
    # shuffle data
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
        # sum the results for 10 folds to compute the average:
        lras = lras + lr_acc
        lrfs = lrfs + lr_f1
        mlpas = mlpas + mlp_acc
        mlpfs = mlpfs + mlp_f1
        dtreeas = dtreeas + dtree_acc
        dtreefs = dtreefs + dtree_f1

    return lras/kfolds, lrfs/kfolds, mlpas/kfolds, mlpfs/kfolds, dtreeas/kfolds, dtreefs/kfolds


# holout method implementation
def holdout(lr, mlp, dtree, acc_eval, f1_eval, datadf):
    # shuffl the data
    datadf.orderBy(rand(seed=12))
    splits = datadf.randomSplit([0.8, 0.2], 1)
    # normalize the data:
    train_split = normalize_data(splits[0])
    test_split = normalize_data(splits[1])
    lr_acc, lr_f1, mlp_acc, mlp_f1, dtree_acc, dtree_f1 = train_test_models(lr, mlp, dtree, acc_eval, f1_eval,
                                                                            train_split, test_split)
    return lr_acc, lr_f1, mlp_acc, mlp_f1, dtree_acc, dtree_f1


# check input arguments to figure out if user wants holdout or cross validation
if args[2] == '-cv':
    lra, lrf, mlpa, mlpf, dtreea, dtreef = cross_validation(lr_model, mlp_model, dtree_model, acc_eval_model,
                                                        f1_eval_model, datadf)
elif args[2] == '-ho':
    lra, lrf, mlpa, mlpf, dtreea, dtreef = holdout(lr_model, mlp_model, dtree_model, acc_eval_model, f1_eval_model, datadf)
else:
    print("\nCalling arguments are incorrect. Use the sample below:")
    print("python3 main.py [-s1/-s2/-s1b/-s2b] [-ho/-cv]")
    print("-s1: use room setting 1 imbalanced data")
    print("-s2: use room setting 2 imbalanced data")
    print("-s1b: use room setting 1 balanced data")
    print("-s2b: use room setting 2 balanced data")
    print("-ho: use holdout method for evaluation")
    print("-cv: use 10-fold cross validation method for evaluation")
    exit(0)
print("acc for lr model is " + str(lra) + " f1 for it is " + str(lrf))
print("acc for mlp model is " + str(mlpa) + " f1 for it is " + str(mlpf))
print("acc for dtree model is " + str(dtreea) + " f1 for it is " + str(dtreef))

sc.stop()
