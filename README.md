# Project Title
The project aims to develop a machine learning application for predicting loan approval using data from historical loan applications. Leveraging machine learning techniques, the application aims to provide insights into whether a loan application is likely to be approved or rejected based on various factors such as income, credit history, loan amount, etc.

## Overview

Explain the purpose and goals of your project.

## MLflow Integration

This project utilizes [MLflow](https://www.mlflow.org/) for experiment tracking, packaging code into reproducible runs, and sharing and deploying models. MLflow helps streamline the machine learning lifecycle.

## Dagshub Credentials
MLFLOW_TRACKING_URI=https://dagshub.com/naimurborno/Loan_prediction_tracking_using_mlflow.mlflow \
MLFLOW_TRACKING_USERNAME=naimurborno \
MLFLOW_TRACKING_PASSWORD=94ebc629914b4f17304744d67eec0b421a8f74d1  \

MLFLOW_TRACKING_URI=https://dagshub.com/naimurborno/Loan_prediction_tracking_using_mlflow.mlflow \
MLFLOW_TRACKING_USERNAME=naimurborno \
MLFLOW_TRACKING_PASSWORD=94ebc629914b4f17304744d67eec0b421a8f74d1 \
python script.py

## MLflow on Aws

    Login to AWS console.
    Create IAM user with AdministratorAccess
    Export the credentials in your AWS CLI by running "aws configure"
    Create a s3 bucket
    Create EC2 machine (Ubuntu) & add Security groups 5000 port

Run the following command on EC2 machine
sudo apt update

sudo apt install python3-pip

sudo pip3 install pipenv

sudo pip3 install virtualenv

mkdir mlflow

cd mlflow

pipenv install mlflow

pipenv install awscli

pipenv install boto3

pipenv shell


## Then set aws credentials
aws configure


#Finally 
mlflow server -h 0.0.0.0 --default-artifact-root s3://mlflow-test-23

#open Public IPv4 DNS to the port 5000


#set uri in your local terminal and in your code 
export MLFLOW_TRACKING_URI=http://ec2-54-147-36-34.compute-1.amazonaws.com:5000/

