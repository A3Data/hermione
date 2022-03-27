import streamlit as st
import json
import pandas as pd
import os
import glob
import re
from pathlib import Path
from joblib import load
from src.data_source.spreadsheet import Spreadsheet
from src.preprocessing._preprocessing import Preprocessing
from src.model import TrainerSklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import mlflow
import mlflow.sklearn
from ..graphic_elements.st_functions import *

PROJECT_ROOT_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = os.path.abspath(os.path.join(PROJECT_ROOT_DIR, "data"))


CLASS_NAME = {1: "1st Class", 2: "2nd Class", 3: "3rd Class"}

ALGO_NAME = {
    "rf": "Random Forest",
    "gb": "Gradient Boost",
    "log": "Logistic Regression",
}


class ModelPage:
    def __init__(self):
        st.title("Model Execution")
        with open(os.path.join(PROJECT_ROOT_DIR, "config", "config.json"), "r") as file:
            project_name = json.load(file)["project_name"]
        mlflow.set_experiment(project_name)
        self.df = Spreadsheet().get_data(os.path.join(DATA_DIR, "raw", "train.csv"))
        self.algos = {
            "rf": RandomForestClassifier,
            "gb": GradientBoostingClassifier,
            "log": LogisticRegression,
        }
        self.p = Preprocessing()
        self.mode = st.sidebar.radio("", ["Model Fitting", "Predict"])

    def preprocess(self):

        df_clean = self.p.clean_data(self.df)
        df_encoded = self.p.categ_encoding(df_clean)
        return df_encoded

    def fit_model(self, df, algos):
        X = df.drop(columns=["Survived"])
        y = df["Survived"]
        my_bar = st.progress(0)
        for algo, object, index in zip(
            algos.keys(), algos.values(), range(len(algos.keys()))
        ):
            st.write("### Fitting {}".format(ALGO_NAME[algo]))
            with mlflow.start_run() as run:
                model = TrainerSklearn().train(
                    X,
                    y,
                    classification=True,
                    algorithm=object,
                    data_split=("cv", {"cv": 8}),
                    preprocessing=self.p,
                )
                mlflow.log_params({"algorithm": algo})
                mlflow.log_metrics(model.get_metrics())
                mlflow.sklearn.log_model(model.get_model(), "model")
                # Salva o modelo na pasta output
                model.save_model(
                    os.path.join(DATA_DIR, "output", f"titanic_model_{algo}.pkl")
                )
                my_bar.progress((1 / len(algos.keys())) * (index + 1))
        my_bar.empty()

    def model_page(self):
        st.write("## Model Fitting")
        st.write("""#### Train the developed machine learning algorithm""")
        with st.form(key="model_form"):
            algo_options = ["All"] + list(self.algos.keys())
            chosen_algo = st.selectbox(
                "Choose algorithm",
                algo_options,
                format_func=lambda x: x if x == "All" else ALGO_NAME[x],
            )
            if chosen_algo == "All":
                model_algo = self.algos
            else:
                model_algo = {
                    key: value
                    for key, value in self.algos.items()
                    if key == chosen_algo
                }
            st.write("Execute model?")
            fit = st.form_submit_button("Fit")
        if fit:
            with st.spinner("Running algorithm ..."):
                df_model = self.preprocess()
                self.fit_model(df_model, model_algo)
            st.write("### Success! Models fitted!")

    def predict(self, X, model, probs):
        try:
            model = load(os.path.join(DATA_DIR, "output", f"titanic_model_{model}.pkl"))
        except:
            st.error("Model not loaded")
        X = self.p.clean_data(X)
        X = self.p.categ_encoding(X)
        columns = model.get_columns()
        for col in columns:
            if col not in X.columns:
                X[col] = 0
        if probs:
            return model.predict_proba(X)[:, 1]
        else:
            return model.predict(X)

    def predict_page(self):
        st.write("## Simulator")
        st.write(
            """Predict the outcome or probability of survival by customizing the input data of individuals"""
        )
        st.write("")
        model_list = glob.glob(os.path.join(DATA_DIR, "output", "*.pkl"))
        avail_algo = [
            re.search("(?<=model_).+(?=\.pkl)", m).group(0) for m in model_list
        ]
        with st.form(key="predict_form"):
            # New Data
            col1, col2, col3 = st.columns(3)
            pclass = col1.radio(
                "Passage Class", (1, 2, 3), format_func=lambda x: CLASS_NAME[x]
            )
            sex = col2.selectbox(
                "Sex", ("male", "female"), format_func=lambda x: x.title()
            )
            age = col3.number_input("Age", step=5)
            new_data = pd.DataFrame({"Pclass": [pclass], "Sex": [sex], "Age": [age]})
            # Prediction Options
            col4, col5 = st.columns([2, 1])
            algorithm = col4.selectbox(
                "Algorithm", avail_algo, format_func=lambda x: ALGO_NAME[x]
            )
            probs = col5.radio(
                "Predict Probability?",
                (True, False),
                format_func=lambda x: "Yes" if x else "No",
            )
            predict = st.form_submit_button(label="Predict")
        if predict:
            pred = self.predict(new_data, algorithm, probs)
            if probs:
                velocimeter_chart(pred[0])
            else:
                outcome = "SURVIVED" if pred == 1 else "DIED"
                st.write(
                    f"""
                         The model predicted that this individual would have 
                         ### {outcome}

                         in the Titanic tragic accident
                         """
                )

    def write(self):
        if self.mode == "Model Fitting":
            self.model_page()
        elif self.mode == "Predict":
            self.predict_page()
