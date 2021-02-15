from joblib import dump, load
from datetime import date
import mlflow.pyfunc
from mlflow import pyfunc
from interpret.ext.blackbox import TabularExplainer, MimicExplainer
from interpret.ext.glassbox import *
import pandas as pd

from util import load_yaml, load_json


class Wrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model=None, preprocessing=None, metrics=None, columns=None):
        """
        Constructor

        Parameters
        ----------
        model         :   object
                          If it's just a model: enter all parameters
                          if it is more than one model: do not enter parameters and use
                          the add method to add each of the models
        preprocessing :   Preprocessamento
                          Preprocessing used in training
        metrics       :   dict
                          Dictionary with the metrics of the result of the model
        columns       :   list
                          list with columns names
        Returns
        -------
        WrapperModel
        """
        self.artifacts = dict()
        self.artifacts["model"] = model
        self.artifacts["preprocessing"] = preprocessing
        self.artifacts["metrics"] = metrics
        self.artifacts["columns"] = columns
        self.artifacts["creation_date"] = date.today()

    def predict(self, model_input):
        """
        Method that returns the result of the prediction on a dataset

        Parameters
        ----------
        df : pd.DataFrame
             Data to be predicted

        Returns
        -------
        list
        """
        df_processed = model_input.copy()
        model = self.artifacts["model"]
        columns = self.artifacts["columns"]
        return model.predict(df_processed[columns])

    def predict_proba(self, model_input, binary=False):
        """
        Method that returns the result of the prediction on a dataset

        Parameters
        ----------
        df : pd.DataFrame
             data to be predicted

        Returns
        -------
        list
        """
        df_processed = model_input.copy()
        model = self.artifacts["model"]
        columns = self.artifacts["columns"]
        if binary:
            return model.predict_proba(df_processed[columns])[:, 1]
        else:
            return model.predict_proba(df_processed[columns])

    def save_model(self, path):
        """
        Saves the model object to a specific path

        Parameters
        ----------
        path : str
               path where the model object will be saved

        Returns
        -------
        None
        """
        dump(self, path)

    @staticmethod
    def load_model(path):
        """
        Loads the model object in a specific path

        Parameters
        ----------
        path : str
               path where the model object will be loaded.

        Returns
        -------
        None
        """
        model = pyfunc.load_model(path)
        return model

    def save(self, path):
        """
        Save model as a Wrapper class

        Parameters
        ----------
        path : str
               path where the model object will be loaded.

        Returns
        -------
        None
        """
        path_artifacts = path + "_artifacts.pkl"
        dump(self.artifacts, path_artifacts)
        content = load_json("config/arquivos.json")
        conda_env = load_yaml(content["path_yaml"])
        mlflow.pyfunc.save_model(
            path=path,
            python_model=self,
            artifacts={"model": path_artifacts},
            conda_env=conda_env,
        )

    def get_metrics(self):
        """
        Return metrics

        Parameters
        ----------
        self : object Wrapper

        Returns
        -------
        dict
        """
        return self.artifacts["metrics"]

    def get_columns(self):
        """
        Return columns

        Parameters
        ----------
        self : object Wrapper

        Returns
        -------
        list
        """
        return self.artifacts["columns"]

    def get_model(self):
        """
        Return model

        Parameters
        ----------
        self : object Wrapper

        Returns
        -------
        dict
        """
        return self.artifacts["model"]

    def get_preprocessing(self):
        """
        Return preprocessing instance

        Parameters
        ----------
        self : object Wrapper

        Returns
        -------
        Preprocessing instance
        """
        return self.artifacts["preprocessing"]

    def train_interpret(self, X, model="tabular"):
        """
        Train a interpret model

        Parameters
        ----------
        self    : object Wrapper
        X       : pd.DataFrame
                  Data that were used in the train for interpret
        model   : string, optional
                  Model to use for the interpret [tabular,mimic_LGBME,
                  mimic_Linear,mimic_SGDE,mimic_Dec_Tree]
        Returns
        -------
        None
        """
        mimic_models = {
            "mimic_LGBME": LGBMExplainableModel,
            "mimic_Linear": LinearExplainableModel,
            "mimic_SGDE": SGDExplainableModel,
            "mimic_Dec_Tree": DecisionTreeExplainableModel,
        }
        if model == "tabular":
            explainer = TabularExplainer(
                self.artifacts["model"], X, features=self.artifacts["columns"]
            )
        else:
            explainer = MimicExplainer(
                self.artifacts["model"],
                X,
                mimic_models[model],
                augment_data=True,
                max_num_of_augmentations=10,
                features=self.artifacts["columns"],
            )
        self.artifacts["explainer"] = explainer

    def local_interpret(self, X, n_feat=3, norm=True):
        """
        Return a local interpret for each row in data

        Parameters
        ----------
        self    : object Wrapper
        X       : array[array], shape (n_linha, n_colunas)
                  Matrix with the data that were used to return interpret
        n_feat  : int, optional
                  Number of features to return
        norm    : bool, optional
                  if True, do normalization in the features importances

        Returns
        -------
        pd.DataFrame
        """
        local_explanation = self.artifacts["explainer"].explain_local(X)
        n_obs = X.shape[0]
        predictions = self.artifacts["model"].predict(X)
        local_values = local_explanation.get_ranked_local_values()
        local_values = [local_values[predictions[i]][i] for i in range(n_obs)]
        local_names = local_explanation.get_ranked_local_names()
        local_names = [local_names[predictions[i]][i] for i in range(n_obs)]
        if norm:
            local_values = [
                [(i - min(l)) / (max(l) - min(l)) for i in l] for l in local_values
            ]
        result = [
            (local_names[i][:n_feat] + local_values[i][:n_feat]) for i in range(n_obs)
        ]
        column_names = [
            f"Importance_{item}_{str(i)}"
            for item in ["Name", "Value"]
            for i in range(n_feat)
        ]
        return pd.DataFrame(result, columns=column_names)