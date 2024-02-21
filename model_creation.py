from urllib.parse import urlparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import ElasticNet
import mlflow
import sys
from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

if __name__=="__main__":
    dataset=pd.read_csv("cleaned_data.csv")
    X=dataset.drop(columns=["Loan_Status"])
    y=datset['Loan_Status']
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,shuffle=True)
    alpha=float(sys.argv[1]) if len(sys.argv[1]>1) else 0.5
    l1=float(sys.argv[2]) if len(sys.argv[2]>1) else 1
    with mlflow.start_run():
        model=ElasticNet(alpha=alpha,l1_ration=l1)
        model.fit(X_train,y_train)

        predicted=model.predict(X_test)
        (rmse,mae,r2)=eval_metrics(y_test,predicted)
        print(f"the rmse value is:{rmse}")
        print(f"the mae value is:{mae}")
        print(f"the r2 score value is:{r2}")
        mlflow.log_param("alpha",alpha)
        mlflow.log_param("l1_ratio,l1")
        mlflow.log_metric("rmse",rmse)
        mlflow.log_metric("mae",mae)
        mlflow.log_metric("r2_score",r2)

        remote_server_uri="https://dagshub.com/naimurborno/Loan_prediction_tracking_using_mlflow.mlflow"
        mlflow.set_tracking_uri(remote_server_uri)

        tracking=urlparse(mlflow.get_tracking_uri()).scheme
        if tracking !="file":
            mlflow.log_model(model,"model")
        else:
            mlflow.sklearn.log_model(model,"model")



    

        








