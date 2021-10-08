from pyspark.sql.dataframe import DataFrame
from pyspark.ml.base import Estimator
from pyspark.ml.util import MLWriter, MLReader
from abc import ABC, abstractmethod

class DataSource(ABC):
    
    @abstractmethod
    def get_data(self) -> DataFrame:
        """
        Abstract method that is implemented in classes that inherit it
        """
        pass

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
         

class CustomEstimator(Estimator, MLWriter, MLReader):

    def __repr__(self):
        return f'{self.__class__}'

    def assert_method(self, valid_methods, method):
        """ 
        Assert if it the input method is valid.
        
    	Parameters
    	----------          
        method_dict : Iterable[str]
            iterable of valid methods

        method  : str
            input method

    	Returns
    	-------
        """
        options = '`' + '`, `'.join(valid_methods) + '`.'
        if method not in valid_methods:
            raise Exception(f'Method not supported. Choose one from {options}')

    def assert_columns(self, df_columns):
        """ 
        Assert if it the DataFrame has the columns to be normalized.
        
    	Parameters
    	----------            
        df  : pyspark.sql.dataframe.DataFrame
            input Spark DataFrame

    	Returns
    	-------
        """
        options = '`' + '`, `'.join(df_columns) + '`.'
        for col in self.estimator_cols:
            if col not in df_columns:
                raise Exception(f'Column `{col}` not present in the DataFrame. Avaliable columns are {options}')

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
        self.assert_columns(df.columns)
        self.estimator = self._fit().fit(df)
        return self

    def transform(self, df) -> DataFrame:
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
        if hasattr(self, 'estimator'):
            df_cols = df.columns
            self.assert_columns(df_cols)
            df = self.estimator.transform(df)
            return df.select(*df_cols, *self.final_cols)
        else:
            raise Exception('Estimator not yet fitted.')

    def fit_transform(self, df) -> DataFrame:
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
        df_cols = df.columns
        self.assert_columns(df_cols)
        self.fit(df)
        df = self.estimator.transform(df)
        return df.select(*df_cols, *self.final_cols)
    