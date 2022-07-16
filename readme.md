# Insurance Fraud Detection

Implementation of  **Insurance fraud detection** task.

## Get Started

Data set has been taken from [2018 UMN STAT 8051 Modeling](https://www.kaggle.com/competitions/2018-trv-statistical-modeling-competition-umn/overview) Kaggle competition. Train and test sets can be found at `./data/train.csv` and `./data/test.csv`

To run the application locally, you require [Python>3.8](https://www.python.org/) and [pip](https://pypi.org/project/pip/) 

Use pip to install the packages with below command:

`pip install -r requirements.txt`

## Run the application

This application requires the following packages to start a development server:

1. [fastAPI](https://fastapi.tiangolo.com/) 
2. [Uvicorn](https://www.uvicorn.org/)

Run the below command to start the application:

`uvicorn api:app`

Note: Use `--reload` to reload the server on code changes

## API

Currently there are two APIs that can train and infer the model. Below are the two API description:

1. `/train`

    **train** is a HTTP Post call that takes **params** such as *depth, features and number of estimators* to name a few. 
Pickle has been used to save the best model at `./models/store_best_model.pickle`.


2. `/test` 

    Uses the saved model and finds out if the claim is a fraud or not. 

More info and API documentation can be found at:

`http://localhost:8000/docs`. 

## Zipped file contents

After training, the best model will be saved in `models`. The `report.json` file containing the results will be saved in the output folder after testing using test data.

## Bonus Task

This pipeline can also be executed with AWS, specifically AWS Sagemaker.