from ._wrapper import SparkWrapper
from ._metrics import SparkMetrics
from .._base import Trainer
        
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
        algorithm        : str
                            model name
        data_split        : tuple (strategy: str, params: dict)
                            strategy of split the data to train your model. 
                            Strategy: ['train_test', 'cv']
                            Ex: ('cv', {'numFolds': 9, 'agg': np.median})
             
    	Returns
    	-------
    	SparkWrapper
        """
        model = algorithm(**params) #model
        if data_split[0] == 'train_test':
            test_size = data_split[1]['test_size']
            df_train, df_test = df.randomSplit([1 - test_size, test_size], seed = 13)
            model = model.fit(df_train)
            df_pred = model.transform(df_test)
            labelCol = model.getLabelCol()
            if classification:
                res_metrics = SparkMetrics.classification(df_pred, labelCol)
            else:
                res_metrics = SparkMetrics.regression(df_pred, labelCol)
        elif data_split[0] == 'cv':
            model, res_metrics = SparkMetrics.crossvalidation(model, df, classification, **data_split[1])
        final_model = SparkWrapper(model, res_metrics)
        return final_model

class SparkUnsupTrainer(Trainer):
        
    def train(self, df, algorithm, metric_params=None, **params):
        """
    	Method that builds the Sklearn model
    
    	Parameters
    	----------
        algorithm   : str
            model name
             
    	Returns
    	-------
    	SparkWrapper
        """
        model = algorithm(**params) #model
        model= model.fit(df)
        df_pred = model.transform(df)
        metric_params = metric_params if metric_params else {}
        res_metrics = SparkMetrics.clusterization(df_pred, **metric_params)
        final_model = SparkWrapper(model, res_metrics)
        return final_model
