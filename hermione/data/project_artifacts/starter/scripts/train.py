import pandas as pd
import json

import os
from pathlib import Path

from src.data_source.spreadsheet import Spreadsheet
from src.preprocessing._preprocessing import Preprocessing
from src.model import TrainerSklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

import mlflow
import mlflow.sklearn

PROJECT_ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = os.path.abspath(os.path.join(PROJECT_ROOT_DIR, "data"))

with open(os.path.join(PROJECT_ROOT_DIR, "config", "config.json"), "r") as file:
    project_name = json.load(file)["project_name"]

mlflow.set_experiment(project_name)

df = Spreadsheet().get_data(os.path.join(DATA_DIR, "raw", "train.csv"))
p = Preprocessing()
df = p.clean_data(df)
df = p.categ_encoding(df)

X = df.drop(columns=["Survived"])
y = df["Survived"]

algos = {
    "rf": RandomForestClassifier,
    "gb": GradientBoostingClassifier,
    "log": LogisticRegression,
}

for algo in algos.keys():
    with mlflow.start_run() as run:
        model = TrainerSklearn().train(
            X,
            y,
            classification=True,
            algorithm=algos[algo],
            data_split=("cv", {"cv": 8}),
            preprocessing=p,
        )
        mlflow.log_params({"algorithm": algo})
        mlflow.log_metrics(model.get_metrics())
        mlflow.sklearn.log_model(model.get_model(), "model")
        # Salva o modelo na pasta output
        model.save_model(os.path.join(DATA_DIR, "output", f"titanic_model_{algo}.pkl"))
