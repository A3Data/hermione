
from pyspark.ml.base import Estimator
from pyspark.ml.util import MLWriter, MLReader, Identifiable
from pyspark.ml.pipeline import Pipeline


class CustomEstimator(Estimator, MLWriter, MLReader, Identifiable):

    def __init__(self, pipeline) -> None:
        self.__pipeline = pipeline

    def __repr__(self):
        return f'{self.__class__}'
    
    def _fit(self):
        """
        Implements abstract method
        
    	Parameters
    	----------            
    	Returns
    	-------
        SparkScaler
        """
        super()._fit()

    def fit(self, df):
        """
        Fits estimator
        
    	Parameters
    	----------            
        df  : pyspark.sql.dataframe.DataFrame
            input Spark DataFrame to be used in fitting

    	Returns
    	-------
        self
        """
        self.__pipeline = self.__pipeline.fit(df)
        return self

    def transform(self, df):
        """
        Transform the Dataframe based on the previously fitted estimators
        
    	Parameters
    	----------            
        df  : pyspark.sql.dataframe.DataFrame
            input Spark DataFrame to be transformed
                     
    	Returns
    	-------
        pyspark.sql.dataframe.DataFrame
        """
        if isinstance(self.__pipeline, Pipeline):
            raise Exception('Estimator not yet fitted.')
        return self.__pipeline.transform(df)

    def fit_transform(self, df):
        """
        Fit estimators and transform the Dataframe
        
    	Parameters
    	----------            
        df  : pyspark.sql.dataframe.DataFrame
            input Spark DataFrame to be transformed

    	Returns
    	-------
        pyspark.sql.dataframe.DataFrame
        """
        if isinstance(self.__pipeline, Pipeline):
            self.__pipeline = self.__pipeline.fit(df)
        return self.__pipeline.transform(df)