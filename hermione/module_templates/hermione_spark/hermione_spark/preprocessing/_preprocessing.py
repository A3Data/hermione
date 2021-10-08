from pyspark.ml.feature import (
    VectorAssembler, 
    StringIndexer, 
    OneHotEncoder,
    Imputer
)
from pyspark.ml.pipeline import Pipeline
from .._base import CustomEstimator
from ._normalization import SparkScaler
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
        num_cols   : Dict[str]
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
        """
        self.num_cols = {
            key: (value if type(value) is list else [value]) 
            for key, value in num_cols.items()
        } if num_cols else None
        self.cat_cols = cat_cols if not cat_cols or type(cat_cols) is list else [cat_cols]
        self.impute_strategy = impute_strategy
        input_cols = [c for sbl in self.num_cols.values() for c in sbl] + self.cat_cols
        self.estimator_cols = list(set(input_cols))

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
        scaler = SparkScaler(self.num_cols)
        return scaler._fit()

    def _fit(self):
        """
        Prepare the estimators
        
    	Parameters
    	----------            
    	Returns
    	-------
        pyspark.ml.pipeline.Pipeline
        """
        estimators = []
        input_cols = []
        if self.cat_cols and None not in self.cat_cols:
            estimators.extend(self.__categoric())
            input_cols.extend(self.ohe_cols)
        if self.num_cols:
            if self.impute_strategy:
                cols = list(set([c for sublist in self.num_cols.values() for c in sublist]))
                imputer = (
                    Imputer(strategy=self.impute_strategy)
                    .setInputCols(cols)
                    .setOutputCols(cols)
                )
                estimators.append(imputer)
            estimators.append(self.__numeric())
            num_input_cols = [method + '_scaled' for method in self.num_cols.keys()]
            input_cols.extend(num_input_cols)
        self.assembler = VectorAssembler(
            inputCols=input_cols, 
            outputCol="features", 
            handleInvalid = 'skip'
        )
        estimators.append(self.assembler)
        self.final_cols = input_cols + ['features']
        return Pipeline(stages=estimators)
