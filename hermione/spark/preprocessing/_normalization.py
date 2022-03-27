from hermione.core.base import Asserter
from .._base import CustomEstimator
from pyspark.ml.feature import (
    VectorAssembler,
    MinMaxScaler,
    MaxAbsScaler,
    StandardScaler,
    RobustScaler,
)
from pyspark.ml.pipeline import Pipeline


class SparkScaler(CustomEstimator, Asserter):
    """
    Class used to scale numerical data using Spark ML

    Parameters
    ----------
    mapping : Dict[str]
        Dict with the normalization method as keys and columns on which it will be applied as values.
        Ex: norm_cols = {'zscore': ['salary', 'price'], 'min-max': ['heigth', 'age']}

    Attributes
    ----------
    estimator_cols : list[str]
        List of strings with the columns that are necessary to execute the model. Used to assert if columns are in the DataFrame to be fitted or transformed.

    final_cols : list[str]
        List of strings with the columns that should be appended to the resulting DataFrame.

    mapping : Dict[object]
        Dict with the normalization method as keys and columns on which it will be applied as values.

    norm_methods : Dict[Estimator]
        Dict with the supported normalization Estimators.

    Examples
    --------
    >>> data = [(1, 0.0, 5.0, 12.0, 102.0), (2, 1.0, 54.0, 27.0, 142.0), (3, 9.0, 27.0, 34.0, 98.0), (4, 8.0, 9.0, 52.0, 112.0)]
    >>> df = spark.createDataFrame(data, ['id', "num_col1", "num_col2", 'num_col3', 'num_col4'])
    >>> scaler = SparkScaler({'zscore': ['num_col1', 'num_col2'], 'min_max': 'num_col3', 'robust': 'num_col4'})
    >>> scaler.fit_transform(df).show()
    +---+--------+--------+--------+--------+--------------------+--------------+--------------------+
    | id|num_col1|num_col2|num_col3|num_col4|       zscore_scaled|min_max_scaled|       robust_scaled|
    +---+--------+--------+--------+--------+--------------------+--------------+--------------------+
    |  1|     0.0|     5.0|    12.0|   102.0|[-0.9667550799532...|         [0.0]|               [0.0]|
    |  2|     1.0|    54.0|    27.0|   142.0|[-0.7519206177414...|       [0.375]|[2.8571428571428568]|
    |  3|     9.0|    27.0|    34.0|    98.0|[0.96675507995323...|        [0.55]|[-0.2857142857142...|
    |  4|     8.0|     9.0|    52.0|   112.0|[0.75192061774140...|         [1.0]|[0.7142857142857142]|
    +---+--------+--------+--------+--------+--------------------+--------------+--------------------+
    """

    def __init__(self, mapping):

        self.assert_type(mapping, dict, "mapping")
        self.mapping = {
            key: (value if type(value) is list else [value])
            for key, value in mapping.items()
        }
        self.norm_methods = {
            "min_max": MinMaxScaler(),
            "max_abs": MaxAbsScaler(),
            "zscore": StandardScaler(withMean=True),
            "robust": RobustScaler(withCentering=True),
        }
        for method in mapping.keys():
            self.assert_method(self.norm_methods.keys(), method)
        self.estimator_cols = list(
            set([c for sbl in self.mapping.values() for c in sbl])
        )
        self.final_cols = [f"{method}_scaled" for method in self.mapping.keys()]

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
                VectorAssembler(handleInvalid="skip")
                .setInputCols(col_list)
                .setOutputCol(f"{method}_vec")
            )
            scaler = (
                self.norm_methods[method]
                .setInputCol(f"{method}_vec")
                .setOutputCol(f"{method}_scaled")
            )
            pipeline = Pipeline(stages=[assembler, scaler])
            scalers.append(pipeline)
        return Pipeline(stages=scalers)
