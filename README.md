# Distributed Energy Usage Forecasting
Author: [Chris Dong](https://github.com/dongchris), [Lingzhi Du](https://github.com/LenzDu), [Kyna Ji](https://github.com/feiran-kyna-ji), [Amber Song](https://github.com/ambersongzz), [Vanessa Zheng](https://github.com/VZ0624)
  
"Forecasting Smart Meter Energy Usage using Distributed Systems and Machine Learning". *Under Review*, ***The 16th IEEE International Conference on Smart City***

## Goal
The EU aims to replace at least 80% of electricity meters with smart meters by 2020. We aimed to leverage the vast amount of smart meter data and build scalable machine learning model to forecast future energy usage. This will make people more informed on energy consumption patterns and benefit energy companies management.

## Project Overview
Following is our model building process:
* Produced automated data pipeline
  - Store data into Amazon S3
  - Import from Amazon S3 to MongoDB running on AWS EC2 instance 
  - Import from MongoDB to Amazon EMR cluster (YARN)
* Preprocessed data and implemented feature engineering using Pandas and Spark SQL
* Forecasted bi-hourly London smart meter usage one day ahead with a scalable random forest model (SparkML)
* Implemented the model on Amazon EMR clusters
* Optimized computational performance by tuning configurations for Yarn cluster
  - Level of parallelism
  - Caching
  - Memory settings


