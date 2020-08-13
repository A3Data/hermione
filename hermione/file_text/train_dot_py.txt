import pandas as pd
import json

from ml.data_source.spreadsheet import Spreadsheet
from ml.preprocessing.preprocessing import Preprocessing
from ml.model.trainer import TrainerSklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

import mlflow
import mlflow.sklearn

with open('config/config.json', 'r') as file:
    project_name = json.load(file)['project_name']

mlflow.set_experiment(project_name)

df = Spreadsheet().get_data('../data/raw/train.csv')
p = Preprocessing()
df = p.clean_data(df)
df = p.categ_encoding(df)

X = df.drop(columns=["Survived"])
y = df["Survived"]

algos = {
    'rf':RandomForestClassifier,
    'gb':GradientBoostingClassifier,
    'log':LogisticRegression
}

for algo in algos.keys():
    with mlflow.start_run() as run:
        model = TrainerSklearn().train(X, y, 
                            classification=True, 
                            algorithm=algos[algo],
                            data_split=('cv', {'cv': 8}),
                            preprocessing=p)
        mlflow.log_params({'algorithm': algo})
        mlflow.log_metrics(model.get_metrics())
        mlflow.sklearn.log_model(model.get_model(), 'model')
        # Salva o modelo na pasta output
        model.save_model(f'../output/titanic_model_{algo}.pkl')
