from .._base import CustomEstimator
from pyspark.ml.feature import (
    VectorAssembler,
    MinMaxScaler,
    MaxAbsScaler,
    StandardScaler,
    RobustScaler,
)
from pyspark.ml.pipeline import Pipeline

class SparkScaler(CustomEstimator):

    def __init__(self, mapping):
        """ 
        Constructor
        
    	Parameters
    	----------            
        mapping : Dict[str]
            Receives dict with the normalization method as keys and columns on 
            which it will be applied as values
            Ex: norm_cols = {'zscore': ['salary', 'price'], 
                             'min-max': ['heigth', 'age']}

    	Returns
    	-------
        """
        self.mapping = {
            key: (value if type(value) is list else [value]) 
            for key, value in mapping.items()
        }
        self.norm_methods = {
            'min_max': MinMaxScaler(),
            'max_abs': MaxAbsScaler(),
            'zscore': StandardScaler(withMean=True),
            'robust': RobustScaler(withCentering=True),
        }
        for method in mapping.keys():
            self.assert_method(self.norm_methods.keys(), method)
        self.estimator_cols = list(set([c for sbl in self.mapping.values() for c in sbl]))
        self.final_cols = [f'{method}_scaled' for method in self.mapping.keys()]
    
    def _fit(self):
        """
        Prepare the estimators
        
    	Parameters
    	----------            
    	Returns
    	-------
        """
        scalers = []
        for method, col_list in self.mapping.items():
            assembler = (
                VectorAssembler(handleInvalid='skip')
                .setInputCols(col_list)
                .setOutputCol(f'{method}_vec')
            )
            scaler = (
                self.norm_methods[method]
                .setInputCol(f'{method}_vec')
                .setOutputCol(f'{method}_scaled')
            )
            pipeline = Pipeline(stages=[assembler, scaler])
            scalers.append(pipeline)
        return Pipeline(stages=scalers)
