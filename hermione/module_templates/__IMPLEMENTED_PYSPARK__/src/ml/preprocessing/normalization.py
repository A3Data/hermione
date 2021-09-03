from pyspark.ml.base import Estimator
from pyspark.sql.dataframe import DataFrame
from pyspark.ml.feature import (
    VectorAssembler,
    MinMaxScaler,
    StandardScaler,
    RobustScaler,
)
from pyspark.ml.util import MLWriter
from pyspark.ml.pipeline import Pipeline

class SparkScaler(Estimator, MLWriter):

    def __init__(self, cols, method: str, vec_name: str = None):

        super(SparkScaler, self).__init__()
        self.cols = cols if type(cols) is list else [cols]
        self.vec_name = method + '_vec' if not vec_name else vec_name
        norm_methods = {
            'min_max': MinMaxScaler,
            'zscore': StandardScaler,
            'robust': RobustScaler,
        }
        if method not in norm_methods.keys():
            raise Exception('Method not supported. Choose one of the available normalization methods.')
        self.method_name = method
        self.scale_method = norm_methods[method]
    
    def _fit(self):
        super()._fit()

    def create_vector(self):

        self.assembler = VectorAssembler(
            inputCols=self.cols,
            outputCol=self.vec_name, 
            handleInvalid='skip'
        )

    def create_scaler(self, **kwargs):

        self.scaler = (
            self.scale_method(**kwargs)
            .setInputCol(self.vec_name)
            .setOutputCol(f'{self.method_name}_scaled')
        )

    def fit(self, df: DataFrame, **kwargs):
        """
        Apply normalization to the selected columns
        
    	Parameters
    	----------            
        df         : pd.DataFrame
                     dataframe with columns to be normalized             
        step_train : bool
                     if True, the Normalizer is created and applied,
                     otherwise it is only applied
                     
    	Returns
    	-------
        pd.DataFrame
            Normalized dataframe
        """
        self.create_vector()
        self.create_scaler(**kwargs)
        pipeline = Pipeline(stages=[self.assembler, self.scaler])
        self.model = pipeline.fit(df)
        return self.model

    def transform(self, df: DataFrame):
        """
        Apply normalization to the selected columns
        
    	Parameters
    	----------            
        df         : pd.DataFrame
                     dataframe with columns to be normalized             
        step_train : bool
                     if True, the Normalizer is created and applied,
                     otherwise it is only applied
                     
    	Returns
    	-------
        pd.DataFrame
            Normalized dataframe
        """
        try:
            return self.model.transform(df)
        except:
            raise Exception('Model no fit! Run `fit()` method before trying to `transform()`')

    def fit_transform(self, df: DataFrame, **kwargs):
        """
        Apply normalization to the selected columns
        
    	Parameters
    	----------            
        df         : pd.DataFrame
                     dataframe with columns to be normalized             
        step_train : bool
                     if True, the Normalizer is created and applied,
                     otherwise it is only applied
                     
    	Returns
    	-------
        pd.DataFrame
            Normalized dataframe
        """
        self.create_vector()
        self.create_scaler(**kwargs)
        pipeline = Pipeline(stages=[self.assembler, self.scaler])
        self.model = pipeline.fit(df)
        return self.model.transform(df)
        