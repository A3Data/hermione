from abc import ABC, abstractmethod
from ml.model.wrapper import Wrapper
from ml.model.metrics import Metrics
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
import numpy as np

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
        
class TrainerSklearn(Trainer):
        
    def train(self, X, y,
                classification: bool, 
                algorithm, 
                preprocessing=None,
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
        data_split        : tuple (strategy: str, params: dict)
                            strategy of split the data to train your model. 
                            Strategy: ['train_test', 'cv']
                            Ex: ('cv', {'cv': 9, 'agg': np.median})
        preprocessing     : Preprocessing
                            preprocessed object to be applied
             
    	Returns
    	-------
    	Wrapper
        """
        model = algorithm(**params) #model
        columns = list(X.columns)
        if data_split[0] == 'train_test':
            X_train, X_test, y_train, y_test = train_test_split(X, y, **data_split[1])
            model.fit(X_train,y_train)
            y_pred = model.predict(X_test[columns])
            y_probs = model.predict_proba(X_test[columns])[:,1]
            if classification:
                res_metrics = Metrics.classification(y_test.values, y_pred, y_probs)
            else:
                res_metrics = Metrics.regression(y_test.values, y_pred)
        elif data_split[0] == 'cv':
            cv = data_split[1]['cv'] if 'cv' in data_split[1] else 5
            agg_func = data_split[1]['agg'] if 'agg' in data_split[1] else np.mean
            res_metrics = Metrics.crossvalidation(model, X, y, classification, cv, agg_func)
            model.fit(X,y)
        model = Wrapper(model, preprocessing, res_metrics, columns)
        return model
