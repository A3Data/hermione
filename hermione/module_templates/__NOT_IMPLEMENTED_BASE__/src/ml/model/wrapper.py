from joblib import dump, load
from datetime import date
import mlflow.pyfunc
from mlflow import pyfunc

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
        df_processed  = model_input.copy()
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
        df_processed  = model_input.copy()
        model = self.artifacts["model"]
        columns = self.artifacts["columns"]
        if binary:
            return model.predict_proba(df_processed[columns])[:,1]
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
        path_artifacts = path+'_artifacts.pkl'
        dump(self.artifacts, path_artifacts)
        content = load_json("config/arquivos.json") 
        conda_env = load_yaml(content["path_yaml"])
        mlflow.pyfunc.save_model(
                    path= path, 
                    python_model= self, 
                    artifacts= {'model': path_artifacts},
                    conda_env= conda_env)
    
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
        return self.artifacts['metrics']

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
        return self.artifacts['columns']
    
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
        return self.artifacts['model']

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
        return self.artifacts['preprocessing']
