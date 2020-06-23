from sklearn.metrics import *
import numpy as np
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate

class Metrics:
    
    @classmethod
    def smape(cls, A, F):
        """
    	Calculates the smape value between the real and the predicted
    
    	Parameters
    	----------            
        A : array
            Target values
        F : array
            Predicted values
    
    	Returns
    	-------
    	float: smape value
    	"""
        return 100/len(A) * np.sum(np.abs(F - A) / (np.abs(A) + np.abs(F)))
    
    @classmethod
    def __custom_score(cls, y_true, y_pred):
        """
    	Creates a custom metric
    
    	Parameters
    	----------            
        y_true : array
                 Target values
        y_pred : array
                 Predicted values
    
    	Returns
    	-------
    	sklearn.metrics
    	"""
        #return sklearn.metrics.fbeta_score(y_true, y_pred, 2)
        pass
     
    @classmethod
    def customized(cls, y_true, y_pred):
        """
    	Creates a custom metric
    
    	Parameters
    	----------            
        y_true : array
                 Target values
        y_pred : array
                 Predicted values
    
    	Returns
    	-------
    	float
    	"""
        custom_metric = make_scorer(cls.__custom_score, greater_is_better=True)
        return custom_metric
        
    @classmethod
    def mape(cls, y_true, y_pred):
        """
    	Calculates the map value between the real and the predicted
    
    	Parameters
    	----------            
        y_true : array
                 Target values
        y_pred : array
                 Predicted values
    
    	Returns
    	-------
    	float : value of mape
    	"""
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs(((y_true+1) - (y_pred+1)) / (y_true+1))) * 100

    
    @classmethod
    def regression(cls, y_true, y_pred):
        """
    	Calculates some metrics for regression problems
    
    	Parameters
    	----------            
        y_true : array
                 Target values
        y_pred : array
                 Predicted values
    
    	Returns
    	-------
    	dict : metrics results
    	"""
        results =   {'mean_absolute_error': round(mean_absolute_error(y_true, y_pred), 7),
                      'root_mean_squared_error': round(np.sqrt(mean_squared_error(y_true, y_pred)), 7),
                      'r2': round(r2_score(y_true, y_pred), 7),
                      'smape': round(cls.smape(y_true, y_pred), 7),
                      'mape': round(cls.mape(y_true, y_pred), 7)
                     }        
        return results
        
    @classmethod
    def crossvalidation(cls, model, X, y, classification: bool, cv=5, agg=np.mean):
        if classification:
            if len(set(y)) > 2:
                metrics = ['accuracy','f1_weighted', 'recall_weighted','precision_weighted']
            else:
                metrics = ['accuracy','f1', 'recall','precision', 'roc_auc']
        else:
            metrics = ['mean_absolute_error', 'r2', 'root_mean_squared_error', 'smape', 'mape']
        res_metrics = cross_validate(model, X, y, cv=cv, return_train_score=False, scoring=metrics)
        results = {metric.replace("test_", ""): round(agg(res_metrics[metric]),7) for metric in res_metrics}
        return results

    @classmethod
    def __multiclass_classification(cls, y_true, y_pred):
        """
    	Calculates some metrics for multiclass classification problems
    
    	Parameters
    	----------            
        y_true    : array
                    Target values
        y_pred    : array
                    Predicted values
    
    	Returns
    	-------
    	dict : metrics results
    	"""
        results =   {'accuracy': accuracy_score(y_true, y_pred),
                      'f1': f1_score(y_true, y_pred, average='weighted'),
                      'precision': precision_score(y_true, y_pred, average='weighted'),
                      'recall': recall_score(y_true, y_pred, average='weighted'),
                     }
        return results
    
    @classmethod
    def __binary_classification(cls, y_true, y_pred, y_probs):
        """
    	Calculates some metrics for binary classification problems
    
    	Parameters
    	----------            
        y_true    : array
                    Target values
        y_pred    : array
                    Predicted values
    
    	Returns
    	-------
    	dict : metrics results
    	"""
        results =    {'accuracy': accuracy_score(y_true, y_pred),
                      'f1': f1_score(y_true, y_pred),
                      'precision': precision_score(y_true, y_pred),
                      'recall': recall_score(y_true, y_pred),
                      'roc_auc': roc_auc_score(y_true, y_probs)
                     }        
        return results
    
    @classmethod
    def classification(cls, y_true, y_pred, y_probs):
        """
    	Checks which classification method will be applied: binary or multiclass
    
    	Parameters
    	----------            
        y_true    : array
                    Target values
        y_pred    : array
                    Predicted values
        y_probs   : array
                    Probabilities values
    
    	Returns
    	-------
    	dict: metrics results
    	"""
        if len(set(y_true)) > 2:
            results = cls.__multiclass_classification(y_true, y_pred)
        else:
            results = cls.__binary_classification(y_true, y_pred, y_probs)
        return results
            
        
    @classmethod
    def clusterization(cls, X, labels):
        """
    	Calculates some metrics on clustering quality
    
    	Parameters
    	----------            
        X      : array[array], shape (n_linha, n_colunas)
                 Matrix with the values that were used in the cluster
        labels : array, shape (n_linha, 1)
                 Vector with labels selected by the clustering method (eg KMeans)
    
    	Returns
    	-------
    	dict : metrics results
    	"""
        results = {'silhouette': silhouette_score(X, labels, metric='euclidean'),
                   'calinski_harabaz': calinski_harabaz_score(X, labels)
                  }
        return results