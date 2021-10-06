from pyspark.ml.feature import (
    VectorAssembler, 
    StringIndexer, 
    OneHotEncoder,
    Imputer
)
from ._base import CustomEstimator
from pyspark.ml.util import MLWriter, MLReader
from pyspark.ml.pipeline import Pipeline
from src.ml.preprocessing.normalization import SparkScaler
import logging

logging.getLogger().setLevel(logging.INFO)

class SparkPreprocessor(CustomEstimator):
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
        self.num_cols = {key: (value if type(value) is list else [value]) for key, value in num_cols.items()}
        self.cat_cols = cat_cols if not cat_cols or type(cat_cols) is list else [cat_cols]
        pipeline = self.__prepare_transformers(impute_strategy)
        super().__init__(pipeline)

    def __prepare_transformers(self, impute_strategy):
        """
        Prepare the transformer for preprocessing the DataFrames
        
    	Parameters
    	----------            
        impute_strategy: str
            Strategy for completing missing values on numerical columns. Supports `mean`, `median` and `mode`.

    	Returns
    	-------
        pyspark.ml.pipeline.Pipeline | list[pyspark.ml.Estimator]
        """
        estimators = []
        input_cols = []
        if self.cat_cols and None not in self.cat_cols:
            estimators = estimators + self.__categoric()
            input_cols = input_cols + self.ohe_cols
        if self.num_cols:
            if impute_strategy:
                cols = [c for sublist in self.num_cols.values() for c in sublist]
                imputer = (
                    Imputer(strategy=impute_strategy)
                    .setInputCols(cols)
                    .setOutputCols(cols)
                )
                estimators.append(imputer)
            estimators = estimators + self.__numeric()
            num_input_cols = [method + '_scaled' for method in self.num_cols.keys()]
            input_cols = input_cols + num_input_cols
        self.assembler = VectorAssembler(
            inputCols=input_cols, 
            outputCol="features", 
            handleInvalid = 'skip'
        )
        estimators.append(self.assembler)
        return Pipeline(stages=estimators)

    def __categoric(self):
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
    
    def __numeric(self):
        """
        Creates the model responsible to normalize numerical columns

        Parameters
        ----------   
    	Returns
    	-------
        list[Estimator]
            Returns a list of estimators
        """
        scalers = []
        for method, col in self.num_cols.items():
            scalers.append(SparkScaler(col, method))
        return scalers
    