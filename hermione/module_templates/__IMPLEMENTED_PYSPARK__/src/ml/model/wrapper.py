from joblib import dump, load
from datetime import date
import mlflow.spark
from mlflow import pyfunc

from src.util import load_yaml, load_json


class Wrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model=None, metrics=None):
        """
        Constructor

        Parameters
        ----------
        model         :   object
                          If it's just a model: enter all parameters
                          if it is more than one model: do not enter parameters and use
                          the add method to add each of the models
        metrics       :   dict
                          Dictionary with the metrics of the result of the model
        Returns
        -------
        WrapperModel
        """
        self.artifacts = dict()
        self.artifacts["model"] = model
        self.artifacts['model_instance'] = type(model)
        self.artifacts["metrics"] = metrics
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
        model = self.artifacts["model"]
        df_pred = model.transform(model_input)
        pred_row = df_pred.select('prediction').collect()
        return [c['prediction'] for c in pred_row]

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
        model = self.artifacts["model"]
        df_pred = model.transform(model_input)
        pred_row = df_pred.select('probability').collect()
        if binary:
            return [c['probability'][1] for c in pred_row]
        else:
            return[c['probability'][1] for c in pred_row]
        
    def load(self, path):
        """
        Load the model object to a specific path

        Parameters
        ----------
        path : str
               path where the model object will be saved

        Returns b
        -------
        None
        """
        return self.artifacts['model_instance'].load(path)

    def save(self, path):
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
        self.artifacts['model'].save(path)

    @staticmethod
    def load_mlflow(path):
        """
        Loads the model object in a specific path (pyfunc)

        Parameters
        ----------
        path : str
               path where the model object will be loaded.

        Returns
        -------
        None
        """
        return mlflow.spark.load_model(path)

    def save_mlflow(self, path):
        """
        Save model as a Wrapper class (pyfunc)

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
        mlflow.spark.save_model(
            self.artifacts['model'],
            path=path,
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

    def get_model_instance(self):
        """
        Return model

        Parameters
        ----------
        self : object Wrapper

        Returns
        -------
        dict
        """
        return self.artifacts["model_instance"]

