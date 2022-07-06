# MLinPython

This repository has Jupyter Notebooks showcasing code examples for the purpose of Data Analysis and Machine Learning in Python.
The notebooks contain code blocks that can be used as reference for required tasks.

Conda environment and requirements.txt details are at the end of this readme.

# Notebooks
Python and PySpark notebooks.

## Python

Created and run on Python 3.8.
 - **1_Data_operations.ipynb**: Covers major data operations required before getting into any analysis or model building
 - **2_Pandas_apply_optimization.ipynb**: Shows the comparison between various ways of applying functions to a pandas df. Helps in optimizing pandas codes
 - **3_Clustering_kmeans.ipynb**: Showcases the flow of a clustering exercise using customer sales data

## PySpark
These notebooks have been built using Spark 3.1.2 installed on Windows, unless specified.
 - **pyspark/1_Clustering_kmeans.ipynb**: Showcases the flow of a clustering exercise using customer sales data
 - **pyspark/2_Spark_data_ops.ipynb**: Covers major data operations in PySpark
 - **pyspark/3_rolling_window_features.ipynb**: Classification model using rolling window features
 - **pyspark/4_xgboost.ipynb**: Notebook showcasing how to use XGBoost with Spark 2.4.5

For Spark 2.4.4 based notebooks, see `./pyspark/pyspark_2_4_4`.

For Spark installation process, refer to [this Medium article](https://medium.com/analytics-vidhya/installing-and-using-pyspark-on-windows-machine-59c2d64af76e).

# Pipeline
## Planned for immediate future
 - Linear regression
 - Logistic regression

## Planned for later (list is WIP as well; suggestions welcome)
 - Decision Trees and Random forests
 - More nbs for PySpark

I plan to keep updating existing notebooks as well along the way.

# Requirements for Python
Create a Python 3.8 environment using the requirements.txt file.

## Commands for conda

Create env
> `conda create -n mlInPython python=3.8`

Switch to the env
> `conda activate mlInPython`

Install dependencies
> `pip install -r requirements.txt`