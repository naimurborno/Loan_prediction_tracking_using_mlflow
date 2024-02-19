import mlflow
from utils import create_mlflow_experiment
experiment=create_mlflow_experiment(experiment_name="Load_prediction",artifact_location="mlflow1")
print(experiment)