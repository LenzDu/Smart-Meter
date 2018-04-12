
# coding: utf-8

import numpy as np
import pandas as pd
from datetime import date, timedelta
from time import time 

from pyspark import SparkContext, SparkConf
from pyspark.sql import Row
from pyspark.sql.types import *
from pyspark.sql import SQLContext
import numpy as np
from pyspark.sql.functions import *
import math
import pyspark.sql.functions as func
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from datetime import date, timedelta

start = time()

block = pd.read_csv('file:////mnt/sparklingwater/smart_wide/wide_combine.csv', parse_dates=['day'], index_col=[0,1])
# We tried to import from mongodb (see screenshot on presentation) but it was too slow. Thus, we chose to read directly from disk for final model.

household = pd.read_csv('file:////mnt/sparklingwater/other/informations_households.csv', index_col=0)
weather = pd.read_csv('file:////mnt/sparklingwater/other/weather_daily_darksky.csv', parse_dates=['time'])

block.fillna(0, inplace=True)
household['file'] = household.file.astype('category').cat.codes

pred_date = date(2014,2,1)
start_date = pred_date - timedelta(days=3)
df_part = block.loc[(block.index.get_level_values(1)>=pd.to_datetime(start_date))&                  (block.index.get_level_values(1)<pd.to_datetime(pred_date))]
df_part

def prepare_dataset(df, window_day_num, first_pred_date, n_range=1, day_skip=7, pred_num_period=4):
    
    user_index = df.index.get_level_values(0).unique()
    
    for n in range(n_range):
        
        pred_date = first_pred_date - timedelta(days=day_skip*n)
        
        # get day mean       
        data_df = pd.DataFrame(index=user_index)
        for i in range(1, window_day_num+1):
            current_date = pred_date - timedelta(days=i)
            subset = df.xs(current_date, level=1).mean(axis=1).reindex(user_index).values
            data_df['day%dmean'%i] = subset
        
        # get hourly mean
        start_date = pred_date - timedelta(days=window_day_num)
        df_part = df.loc[(df.index.get_level_values(1)>=pd.to_datetime(start_date))&                          (df.index.get_level_values(1)<pd.to_datetime(pred_date))]
        hourly_mean = df_part.groupby('LCLid').mean().reindex(user_index)
        data_df = pd.concat([data_df, hourly_mean], axis=1)
        
        # get weather data
        for i in range(1, window_day_num+1):
            current_date = pred_date - timedelta(days=i+1)
            w_data = weather.loc[weather.time==current_date]
            if w_data.shape[0] == 0:
                data_df['temp_max_%d'%i] = np.nan
                data_df['temp_min_%d'%i] = np.nan
            else:
                data_df['temp_max_%d'%i] = w_data['temperatureMax'].iloc[0]
                data_df['temp_min_%d'%i] = w_data['temperatureMin'].iloc[0]

        # household data
        data_df = data_df.join(household[['file']])
        
        # get label
        pred_length = int(48 / pred_num_period)
        for i in range(pred_num_period):
            period_cols = ['hh_%d'%x for x in range(pred_length*i, pred_length*(i+1))]
            pred_period_mean = df.xs(pred_date, level=1)[period_cols].mean(axis=1).reindex(user_index)
            data_df['pred_period_%d'%i] = pred_period_mean.values
        
        if n == 0: data_df_combine = data_df
        elif n > 0: data_df_combine = pd.concat([data_df_combine, data_df], axis=0)
    
    return data_df_combine

pred_num_period = 12
train_df = prepare_dataset(block, 10, date(2014,2,1), n_range=20, pred_num_period=pred_num_period)
val_df = prepare_dataset(block, 10, date(2014,2,8), pred_num_period=pred_num_period)

# train_df.fillna(0, inplace=True)
# val_df.fillna(0, inplace=True)
train_df.dropna(inplace=True)
val_df.dropna(inplace=True)

conf = SparkConf().setMaster("local").setAppName('smartcity')
sc = SparkContext.getOrCreate()
sqlContext = SQLContext(sc)

train_spark_df = sqlContext.createDataFrame(train_df)
val_spark_df = sqlContext.createDataFrame(val_df)

pred_list = []
for i in range(pred_num_period):
    va = VectorAssembler(outputCol='features', inputCols=train_spark_df.columns[:-pred_num_period])
    label_col = 'pred_period_%d'%i
    train_va = va.transform(train_spark_df).select('features', label_col).withColumnRenamed(label_col, 'label').cache()
    val_va = va.transform(val_spark_df).select('features', label_col).withColumnRenamed(label_col, 'label').cache()

    train_va.count(); val_va.count();

    rf = RandomForestRegressor(maxDepth=10, numTrees=10, maxBins=128)
    rfmodel = rf.fit(train_va)

    pred_val = rfmodel.transform(val_va)
    pred_list.append(pred_val.select('prediction').rdd.map(lambda x: x[0]).collect())
    evaluator = RegressionEvaluator(labelCol='label', predictionCol='prediction', metricName="rmse")
    accuracy = evaluator.evaluate(pred_val)
    print 'RMSE for period %d: %.4f'%(i+1, accuracy)

pred = np.stack(pred_list, axis=1)

sc.stop()

print 'time: ' + str(time()-start)

# % matplotlib inline
# from matplotlib import pyplot as plt
# user_id += 1
# plt.plot(np.repeat(pred[user_id, :], 2), color='red') # prediction
# plt.plot(np.repeat(val_df.iloc[user_id, -12:], 2).values) # true value

