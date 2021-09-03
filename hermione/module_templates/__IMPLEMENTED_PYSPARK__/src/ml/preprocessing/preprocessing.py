from pyspark.sql.dataframe import DataFrame
from pyspark.ml.feature import (
    VectorAssembler, 
    StringIndexer, 
    OneHotEncoder
)
from pyspark.ml.pipeline import Pipeline
from src.ml.preprocessing.normalization import SparkScaler
import logging

logging.getLogger().setLevel(logging.INFO)

# Add custom method
from pyspark.ml import Estimator

def fit_transform(self, df):
    self.model = self.fit(df)
    self.transform = self.model.transform
    return self.transform(df)

Estimator.fit_transform = fit_transform

class SparkPreprocessor:
    """
    Class to perform data preprocessing before training
    """
    
    def __init__(self, num_cols: dict = None, cat_cols: str = None):
        """
        Constructor

        Parameters
        ----------
        num_cols   : dict
                      Receives dict with the name of the normalization to be 
                      performed and which are the columns
                      Ex: norm_cols = {'zscore': ['salary', 'price'], 
                                       'min-max': ['heigth', 'age']}
        cat_cols : array
                      Receives an array with columns names to be categorized with One Hot Encoding 
        Returns
        -------
        Preprocessing
        """
        self.num_cols = num_cols
        self.cat_cols = cat_cols if not cat_cols or type(cat_cols) is list else [cat_cols]

    def categoric(self):
        """
        Constructor

        Parameters
        ----------
        norm_cols   : dict
                      Receives dict with the name of the normalization to be 
                      performed and which are the columns
                      Ex: norm_cols = {'zscore': ['salary', 'price'], 
                                       'min-max': ['heigth', 'age']}
        oneHot_cols : array
                      Receives an array with columns names to be categorized with One Hot Encoding 
        Returns
        -------
        Preprocessing
        """
        indexed_cols = [c + '_indexed' for c in self.cat_cols]
        ohe_cols = [c + '_ohe' for c in self.cat_cols]
        self.indexer = StringIndexer(
            inputCols = self.cat_cols,
            outputCols=indexed_cols,
            handleInvalid = 'keep'
        )
        self.ohe = OneHotEncoder(
            inputCols = indexed_cols,
            outputCols=ohe_cols
        )
        self.ohe_cols = ohe_cols
        return [self.indexer, self.ohe]
    
    def numeric(self):
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
        logging.info("Normalizing")
        scalers = []
        for method, col in self.num_cols.items():
            scalers.append(SparkScaler(col, method))
        return scalers
    
    def execute(self, df: DataFrame, pipeline: bool = True, step_train: bool = False, val_size: float = 0.2):
        """
        Apply all preprocessing steps on the Dataframe
        
    	Parameters
    	----------            
        df         : pd.DataFrame
                     dataframe with columns to be normalized             
        step_train : bool
                     if True, data is splited in train and val
        val_size : val_size
                     Size of the validation dataset
                     
    	Returns
    	-------
        pd.DataFrame
            - One Preprocessed dataframe, if step_train is False
            - Two Preprocessed dataframes, if step_train is True 
        """
        estimators = []
        input_cols = []
        if self.cat_cols:
            estimators = estimators + self.categoric()
            input_cols = input_cols + self.ohe_cols
        if self.num_cols:
            estimators = estimators + self.numeric()
            num_input_cols = [method + '_scaled' for method in self.num_cols.keys()]
            input_cols = input_cols + num_input_cols
        self.assembler = VectorAssembler(
            inputCols=input_cols, 
            outputCol="features", 
            handleInvalid = 'skip'
        )
        estimators.append(self.assembler)
        if pipeline:
            pipeline = Pipeline(stages=estimators)
            if step_train:
                df_train, df_test = df.randomSplit([1 - val_size, val_size], seed=13)
                df_train = pipeline.fit_transform(df_train)
                df_test = pipeline.transform(df_test)
                self.pipeline = pipeline.model
                return (df_train, df_test)
            else:
                df = pipeline.fit_transform(df)
                self.pipeline = pipeline.model
                return df
        else:
            if step_train:
                df_train, df_test = df.randomSplit([1 - val_size, val_size], seed=13)
                for model in estimators:
                    df_train = model.fit_transform(df_train)
                    df_test = model.transform(df_test)
                    model = model.model
                return (df_train, df_test)
            else:
                for model in estimators:
                    df = model.fit_transform(df)
                    model = model.model
                return df
            
