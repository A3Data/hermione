from abc import ABC, abstractmethod
from src.ml.model.wrapper import Wrapper
from src.ml.model.metrics import Metrics
from pyspark.ml.pipeline import Pipeline
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split, LeaveOneOut
import numpy as np

# Add custom method
from pyspark.ml import Estimator

def fit_transform(self, df):
    self.model = self.fit(df)
    self.transform = self.model.transform
    return self.transform(df)

Estimator.fit_transform = fit_transform

class Trainer(ABC):
    def __init__(self):
        """
    	Constructor
    
    	Parameters
    	----------    
        None
             
    	Returns
    	-------
    	Trainer
        """
    
    @abstractmethod
    def train(self):
        """
        Abstract method that should be implemented in every class that inherits TrainerModel
    	Parameters
    	----------    
        None
             
    	Returns
    	-------
    	None
        """
        pass
        
class SparkTrainer(Trainer):
        
    def train(self, df,
                classification, 
                algorithm,
                data_split=('train_test', {'test_size': 0.2}), 
                **params):
        """
    	Method that builds the Sklearn model
    
    	Parameters
    	----------    
        classification    : bool
                            if True, classification model training takes place, otherwise Regression
        model_name        : str
                            model name
        preprocessing     : Preprocessing
                            preprocessed object to be applied
        data_split        : tuple (strategy: str, params: dict)
                            strategy of split the data to train your model. 
                            Strategy: ['train_test', 'cv']
                            Ex: ('cv', {'numFolds': 9, 'agg': np.median})
             
    	Returns
    	-------
    	Wrapper
        """
        model = algorithm(**params) #model
        if data_split[0] == 'train_test':
            test_size = data_split[1]['test_size']
            df_train, df_test = df.randomSplit([1 - test_size, test_size], seed = 13)
            fitted_model = model.fit(df_train)
            df_pred = fitted_model.transform(df_test)
            labelCol = fitted_model.getLabelCol()
            if classification:
                res_metrics = Metrics.classification(df_pred, labelCol)
            else:
                res_metrics = Metrics.regression(df_pred, labelCol)
        elif data_split[0] == 'cv':
            fitted_model, res_metrics = Metrics.crossvalidation(model, df, classification, **data_split[1])
        final_model = Wrapper(fitted_model, res_metrics)
        return final_model

class TrainerSklearnUnsupervised(Trainer):
        
    def train(self, X,
                algorithm, 
                preprocessing=None,
                **params):
        """
    	Method that builds the Sklearn model
    
    	Parameters
    	----------    
        model_name        : str
                            model name
        preprocessing     : Preprocessing
                            preprocessed object to be applied
             
    	Returns
    	-------
    	Wrapper
        """
        model = algorithm(**params) #model
        columns = list(X.columns)
        model.fit(X)
        labels = model.predict(X)
        res_metrics = Metrics.clusterization(X, labels)
        
        model = Wrapper(model, preprocessing, res_metrics, columns)
        return model
