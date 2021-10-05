from pyspark.ml.feature import (
    VectorAssembler, 
    StringIndexer, 
    OneHotEncoder,
    Imputer
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
    
    def __init__(self, num_cols = None, cat_cols = None, impute_strategy=None):
        """
        Constructor

        Parameters
        ----------
        num_cols   : dict
            Receives dict with the normalization method as keys and columns on 
            which it will be applied as values
            Ex: norm_cols = {'zscore': ['salary', 'price'], 
                            'min-max': ['heigth', 'age']}

        cat_cols : Union[list, str]
            Categorical columns present in the model

        impute_strategy: str
            Strategy for completing missing values on numerical columns. Supports `mean`, `median` and `mode`.

        Returns
        -------
        SparkPreprocessor
        """
        self.num_cols = num_cols
        self.cat_cols = cat_cols if not cat_cols or type(cat_cols) is list else [cat_cols]
        if impute_strategy:
            self.imputer = (
                Imputer(strategy=impute_strategy)
                .setInputCols(self.num_cols)
                .setOutputCols(self.num_cols)
            )

    def categoric(self):
        """
        Creates the model responsible to transform strings in categories with `StringIndexer` 
        and then one-hot-encodes them all using `OneHotEncoder`

        Parameters
        ----------
        Returns
        -------
        list[Estimator]
            Returns a list of estimators 
        """
        logging.info("Treating categorical data...")
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
        Creates the model responsible to normalize numerical columns

        Parameters
        ----------   
    	Returns
    	-------
        list[Estimator]
            Returns a list of estimators
        """
        logging.info("Normalizing")
        scalers = []
        for method, col in self.num_cols.items():
            scalers.append(SparkScaler(col, method))
        return scalers
    
    def execute(self, df, pipeline = True, step_train = False, val_size = 0.2):
        """
        Apply all preprocessing steps on the Dataframe
        
    	Parameters
    	----------            
        df         : pyspark.sql.dataframe.DataFrame
            dataframe with columns to be preprocessed
        pipeline   : bool
            if True, the estimators are wrapped in a Pipeline  
        step_train : bool
            if True, data is splited in train and val
        val_size : val_size
            Size of the validation dataset
                     
    	Returns
    	-------
        pyspark.sql.dataframe.DataFrame
            - One Preprocessed dataframe, if step_train is False
            - Two Preprocessed dataframes, if step_train is True 
        """
        estimators = []
        input_cols = []
        if self.cat_cols and None not in self.cat_cols:
            estimators = estimators + self.categoric()
            input_cols = input_cols + self.ohe_cols
        if self.num_cols:
            if self.imputer:
                estimators.append(self.imputer)
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
            
