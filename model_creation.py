import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import ElasticNet
import mlflow
import sys

if __name__=="__main__":
    with mlflow.start_run(run_name="training",experiment_id="501253800075241540") as run:
        #reading the cleaned dataset
        dataset=pd.read_csv("cleaned_data.csv")
        X=dataset.drop(columns=['Loan_Status'])
        y=dataset['Loan_Status']

        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,shuffle=True)
        # n_estimator=float(sys.argv[1] if len(sys.argv[1]>1 else 10))
        alpha=float(sys.argv[1] if len(sys.argv[1])>1 else 1)
        l1=float(sys.argv[2] if len(sys.argv[2])>1 else 0.5)


        mlflow.sklearn.autolog()
        # model=RandomForestClassifier(n_estimators=n_estimator)
        model=ElasticNet(alpha=alpha,l1_ratio=l1)
        model.fit(X_train,y_train)
        








