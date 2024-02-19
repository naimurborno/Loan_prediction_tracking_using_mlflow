import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow

if __name__=="__main__":
    with mlflow.start_run(run_name="training",experiment_id="501253800075241540") as run:
        #reading the cleaned dataset
        dataset=pd.read_csv("cleaned_data.csv")
        X=dataset.drop(columns=['Loan_Status'])
        y=dataset['Loan_Status']

        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,shuffle=True)
        mlflow.sklearn.autolog()
        model=RandomForestClassifier()
        model.fit(X_test,y_test)








