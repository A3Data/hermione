from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder, Imputer
from pyspark.ml.pipeline import Pipeline
from .._base import CustomEstimator
from hermione.core.base import Asserter
from ._normalization import SparkScaler
import logging

logging.getLogger().setLevel(logging.INFO)


class SparkPreprocessor(CustomEstimator, Asserter):
    """
    Class used to preprocess data before training, using Spark ML

    Parameters
    ----------
    num_cols   : Dict[str]
        Receives dict with the normalization method as keys and columns on which it will be applied as values
        Ex: norm_cols = {'zscore': ['salary', 'price'], 'min-max': ['heigth', 'age']}

    cat_cols : Union[list, str]
        Categorical columns present in the model

    input_strategy: str
        Strategy for completing missing values on numerical columns. Supports "mean", "median" and "mode".

    Attributes
    ----------
    cat_cols : Union[list, str]
        Categorical columns present in the model

    estimator_cols : list[str]
        List of strings with the columns that are necessary to execute the model. Used to assert if columns are in the DataFrame to be fitted or transformed.

    final_cols : list[str]
        List of strings with the columns that should be appended to the resulting DataFrame.

    input_strategy: str
        Strategy for completing missing values on numerical columns. Supports "mean", "median" and "mode".

    num_cols   : Dict[str]
        Receives dict with the normalization method as keys and columns on which it will be applied as values

    ohe_cols   : list[str]
        List of strings with the names of the categorical columns that where one-hot encoded.

    Examples
    --------
    >>> data = [(1, 0.0, 5.0, 'a', 'c'), (2, 1.0, 54.0, 'b', 'd'), (3, 9.0, 27.0, 'a', 'd'), (4, 8.0, 9.0, 'b', 'e')]
    >>> df = spark.createDataFrame(data, ['id', "num_col1", "num_col2", 'cat_col1', 'cat_col2'])
    >>> preprocessor = SparkPreprocessor({'zscore': ['num_col1', 'num_col2']}, ['cat_col1', 'cat_col2'])
    >>> preprocessor.fit_transform(df).show()
    +---+--------+--------+--------+--------+-------------+-------------+--------------------+--------------------+
    | id|num_col1|num_col2|cat_col1|cat_col2| cat_col1_ohe| cat_col2_ohe|       zscore_scaled|            features|
    +---+--------+--------+--------+--------+-------------+-------------+--------------------+--------------------+
    |  1|     0.0|     5.0|       a|       c|(2,[0],[1.0])|(3,[1],[1.0])|[-0.9667550799532...|[1.0,0.0,0.0,1.0,...|
    |  2|     1.0|    54.0|       b|       d|(2,[1],[1.0])|(3,[0],[1.0])|[-0.7519206177414...|[0.0,1.0,1.0,0.0,...|
    |  3|     9.0|    27.0|       a|       d|(2,[0],[1.0])|(3,[0],[1.0])|[0.96675507995323...|[1.0,0.0,1.0,0.0,...|
    |  4|     8.0|     9.0|       b|       e|(2,[1],[1.0])|(3,[2],[1.0])|[0.75192061774140...|[0.0,1.0,0.0,0.0,...|
    +---+--------+--------+--------+--------+-------------+-------------+--------------------+--------------------+
    """

    def __init__(self, num_cols=None, cat_cols=None, input_strategy=None):

        input_cols = []
        if num_cols:
            self.assert_type(num_cols, dict, "num_cols")
            self.num_cols = {
                key: (value if type(value) is list else [value])
                for key, value in num_cols.items()
            }
            input_cols.extend([c for sbl in self.num_cols.values() for c in sbl])
        else:
            self.num_cols = None
        if cat_cols:
            self.assert_type(cat_cols, (list, str), "cat_cols")
            self.cat_cols = cat_cols if type(cat_cols) is list else [cat_cols]
            input_cols.extend(self.cat_cols)
        else:
            self.cat_cols = None
        if not cat_cols and not num_cols:
            raise Exception("Provide atleast one set of columns to preprocess.")
        self.input_strategy = input_strategy
        self.estimator_cols = list(set(input_cols))

    def __categoric(self):
        """
        Creates the model responsible to transform strings in categories with `StringIndexer` and then one-hot-encodes them using `OneHotEncoder`.

        Parameters
        ----------
        Returns
        -------
        list[Estimator]
            Returns a list of estimators
        """
        indexed_cols = [c + "_indexed" for c in self.cat_cols]
        ohe_cols = [c + "_ohe" for c in self.cat_cols]
        indexer = StringIndexer(
            inputCols=self.cat_cols, outputCols=indexed_cols, handleInvalid="keep"
        )
        ohe = OneHotEncoder(inputCols=indexed_cols, outputCols=ohe_cols)
        self.ohe_cols = ohe_cols
        return [indexer, ohe]

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
            if self.input_strategy:
                cols = list(
                    set([c for sublist in self.num_cols.values() for c in sublist])
                )
                imputer = (
                    Imputer(strategy=self.input_strategy)
                    .setInputCols(cols)
                    .setOutputCols(cols)
                )
                estimators.append(imputer)
            estimators.append(self.__numeric())
            num_input_cols = [method + "_scaled" for method in self.num_cols.keys()]
            input_cols.extend(num_input_cols)
        assembler = VectorAssembler(
            inputCols=input_cols, outputCol="features", handleInvalid="skip"
        )
        estimators.append(assembler)
        self.final_cols = input_cols + ["features"]
        return Pipeline(stages=estimators)
