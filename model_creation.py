from cmath import sqrt
import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import sys
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.ensemble import RandomForestClassifier

def metrices(true_val,predicted):
    rmse=np.sqrt(mean_squared_error(true_val,predicted))
    mae=mean_absolute_error(true_val,predicted)
    r2=r2_score(true_val,predicted)
    return rmse,mae,r2


if __name__=="__main__":
    dataset=pd.read_csv("cleaned_data.csv")
    X=dataset.drop(columns=["Loan_Status"])
    y=dataset['Loan_Status']
    X_train,X_test,y_train,y_test=train_test_split(X,y,shuffle=True,test_size=0.2)
    alpha=float(sys.argv[1]) if len(sys.argv[1])>1 else 0.5
    l1=float(sys.argv[2]) if len(sys.argv[2])>1 else 1
    with mlflow.start_run(run_name="another run") as run:
        model=ElasticNet(alpha=alpha,l1_ratio=l1)
        model.fit(X_train,y_train)
        model1=RandomForestClassifier()
        model1.fit (X_train,y_train)
        predict1=model.predict(X_test)
        predict=model.predict(X_test)
        rmse,mae,r2=metrices(y_test,predict)
        rmse1,mae1,r21=metrices(y_test,predict1)
        testing_acc_rf=model.score(X_test,y_test)
        training_acc=model.score(X_train,y_train)
        testing_acc=model.score(X_test,y_test)
        mlflow.log_metric("RMSE",rmse)
        mlflow.log_metric("MAE",mae)
        mlflow.log_metric("R2",r2)
        mlflow.log_metric("RMSE1",rmse1)
        mlflow.log_metric("MAE1",mae1)
        mlflow.log_metric("R21",r21)
        mlflow.log_param("Alpha",alpha)
        mlflow.log_param("l1_ratio",l1)
        mlflow.log_metric("Training Accuracy",training_acc)
        mlflow.log_metric("Testing Accuracy",testing_acc)
        mlflow.log_metric("the randomforest acc",testing_acc_rf)
        # remote_server_uri="https://dagshub.com/naimurborno/Loan_prediction_tracking_using_mlflow.mlflow"
        # mlflow.set_tracking_uri(remote_server_uri)

        #for using mlflow using AWS
        remote_server_uri="http://ec2-54-175-131-61.compute-1.amazonaws.com:5000/"
        mlflow.set_tracking_uri(remote_server_uri)

        mlflow.sklearn.log_model(model,"ElasticNet Model",registered_model_name="ElasticNet Model")
        #mlflow.skleran.log_model(model1,"RandomForestClassifier")
        
        #this is good


