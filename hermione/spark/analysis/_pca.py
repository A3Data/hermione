from pyspark.ml.feature import PCA, Imputer
from pyspark.ml.pipeline import Pipeline
from pyspark.ml.functions import vector_to_array
from hermione.core.base import Asserter
from ..preprocessing._normalization import SparkScaler


class SparkPCA(Asserter):
    """
    Class used to perform Principal Components Analysis with Spark ML

    Parameters
    ----------
    inputCols : str | list[str]
        Columns that should be used in the dimensionality reduction

    k : int | float
        Number of desired components. If integer, it is equal to `min(k, _n_features)`.
        If `0 < k < 1`, select the number of components such that the amount of variance that needs to be explained is greater than the percentage specified by `k`.

    input_strategy: str
        Strategy for completing missing values on numerical columns. Supports "mean", "median" and "mode".

    Attributes
    ----------
    estimator_cols : list[str]
        List of strings with the columns that are necessary to execute the model. Used to assert if columns are in the DataFrame to be fitted or transformed.

    estimator : pyspark.ml.pipeline.PipelineModel
        Fitted pipeline with a PCAModel

    k : int | float
        Number of desired components

    pca : pyspark.ml.feature.PCA
        Estimator used to fit the model

    _n_features : int
        Number of features in `inputCol`

    Examples
    --------
    >>> data = [(1, 0.0, 5.0, 12.0, 102.0), (2, 1.0, 54.0, 27.0, 142.0), (3, 9.0, 27.0, 34.0, 98.0), (4, 8.0, 9.0, 52.0, 112.0)]
    >>> df = spark.createDataFrame(data, ['id', "num_col1", "num_col2", 'num_col3', 'num_col4'])
    >>> pca = SparkPCA(["num_col1", "num_col2", 'num_col3', 'num_col4'], .7)
    >>> pca.fit_transform(df).show()
    Explained variance (1): 0.6742215494358871
    Explained variance (2): 0.2439515744953364
    +---+--------+--------+--------+--------+------------------+-------------------+
    | id|num_col1|num_col2|num_col3|num_col4|             cmp_1|              cmp_2|
    +---+--------+--------+--------+--------+------------------+-------------------+
    |  1|     0.0|     5.0|    12.0|   102.0| 70.07001321358567|-20.436957033967012|
    |  2|     1.0|    54.0|    27.0|   142.0|132.72695970656463|  -37.4676157300962|
    |  3|     9.0|    27.0|    34.0|    98.0| 82.98490516179737| -42.84278493554484|
    |  4|     8.0|     9.0|    52.0|   112.0| 78.12025454767982|-61.822109086408844|
    +---+--------+--------+--------+--------+------------------+-------------------+
    """

    def __init__(self, inputCols, k=2, input_strategy="mean"):

        self.assert_type(k, (int, float), "k")
        self.k = k
        self.assert_type(inputCols, (list, str), "inputCols")
        inputCols = inputCols if type(inputCols) is list else inputCols
        self._n_features = len(inputCols)
        self.assert_type(input_strategy, str, "input_strategy")
        self.assert_method(["mean", "median", "mode"], input_strategy)
        imputer = (
            Imputer(strategy=input_strategy)
            .setInputCols(inputCols)
            .setOutputCols(inputCols)
        )
        scaler = SparkScaler({"zscore": inputCols})
        self.preproc = Pipeline(stages=[imputer, scaler])
        self.pca = PCA(inputCol="zscore_scaled", outputCol="components")
        self.estimator_cols = inputCols

    def __find_k(self, df, threshold):
        """
        Find how many k dimensions will be reduced

        Parameters
        ----------
        df : pyspark.sql.dataframe.DataFrame
            input Spark DataFrame to be fitted

        threshold : float
            minimum percentage of explained variance the components should reach

        Returns
        -------
        int
            Minimum number of components that reach `threshold` percentage of explained variance
        """
        df = self.preproc.fit(df).transform(df)
        explained_var = self.pca.setK(self._n_features).fit(df).explainedVariance
        sum_var = 0
        for i, var in enumerate(explained_var):
            sum_var += var
            if sum_var < threshold:
                continue
            else:
                return i + 1

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
        if self.k >= 0 and self.k <= 1:
            self.k = self.__find_k(df, self.k)
        else:
            self.k = min(self._n_features, self.k)
        pca = self.pca.setK(self.k)
        self.estimator = Pipeline(stages=[self.preproc, pca]).fit(df)
        self.__report()

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
        self.assert_columns(df.columns)
        if hasattr(self, "estimator"):
            raise Exception("Estimator not fitted.")
        df_pred = self.estimator.transform(df)
        for comp in range(self.k):
            comp_number = comp + 1
            df_pred = df_pred.withColumn(
                f"cmp_{comp_number}", vector_to_array("components").getItem(comp)
            )
        return df_pred.drop("components", "features")

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
        self.fit(df)
        df_pred = self.estimator.transform(df)
        for comp in range(self.k):
            comp_number = comp + 1
            df_pred = df_pred.withColumn(
                f"cmp_{comp_number}", vector_to_array("components").getItem(comp)
            )
        return df_pred.drop("components", "features")

    def __report(self):
        """
        Returns explained variance

        Parameters
        ----------
        Returns
        -------
        """
        for col, ratio in zip(
            range(self.k), self.estimator.stages[-1].explainedVariance
        ):
            print(f"Explained variance ({col + 1}): {ratio}")
