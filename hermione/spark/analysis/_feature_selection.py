from pyspark.ml.feature import UnivariateFeatureSelector, VectorAssembler
from pyspark.ml.pipeline import Pipeline
from hermione.core.base import Asserter


class SparkFS(Asserter):
    """
    Class used to perform Feature Selection with Spark ML

    Parameters
    ----------
    mapping : Dict[object]
        Dictionary with information concerning the selection. It should be in the following format:

        * <key> str : column types to be selected. One of "categorical" or "continuous".

        * <value>: another dictionary, with the following possible keys:

            * selectionMode (str): The mode of selection. Possible values are "numTopFeatures", "percentile", "fpr", "fwe", "fdr".

            * selectionThreshold (int | float): The threshold for the number of columns selected. If `selectionMode` is "numTopFeatures" it should be an integer, while for the other available modes it should be float.

            * cols (str | list[str]): the column or columns that should be selected.

        More information at http://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.UnivariateFeatureSelector.html

    labelCol : str
        The column of interest, with which each column to be selected will be tested

    labelType : str
        The type of `labelCol`. Available values are "categorical" or "continuous".

    Attributes
    ----------
    estimator_cols : list[str]
        List of strings with the columns that are necessary to execute the model. Used to assert if columns are in the DataFrame to be fitted or transformed.

    labelCol : str
        The column of interest, with which each column to be selected will be tested

    labelType : str
        The type of `labelCol`. Available values are "categorical" or "continuous".

    mapping : Dict[object]
        Dictionary with information concerning the selection

    Examples
    --------
    >>> data = [(1, 0.0, 5.0, 0, 0, 0), (2, 1.0, 54.0, 0, 1, 1), (3, 9.0, 27.0,1, 0, 1), (4, 8.0, 9.0, 1, 1, 0)]
    >>> df = spark.createDataFrame(data, ['id', "num_col1", "num_col2", 'cat_col1', 'cat_col2', 'label'])
    >>> mapping = {
    ...     'continuous': {
    ...         'selectionMode': 'fpr',
    ...         'selectionThreshold': 0.6,
    ...         'cols': ["num_col1", "num_col2"]
    ...     },
    ...     'categorical': {
    ...         'selectionMode': 'numTopFeatures',
    ...         'selectionThreshold': 1,
    ...         'cols': ["cat_col1", "cat_col2"]
    ...     }
    ... }
    >>> selector = SparkFS(mapping, 'label', 'categorical')
    >>> selector.transform(df).show()
    +---+--------+--------+--------+--------+-----+-----------------+
    | id|num_col1|num_col2|cat_col1|cat_col2|label|selected_features|
    +---+--------+--------+--------+--------+-----+-----------------+
    |  1|     0.0|     5.0|       0|       0|    0|        [5.0,0.0]|
    |  2|     1.0|    54.0|       0|       1|    1|       [54.0,0.0]|
    |  3|     9.0|    27.0|       1|       0|    1|       [27.0,1.0]|
    |  4|     8.0|     9.0|       1|       1|    0|        [9.0,1.0]|
    +---+--------+--------+--------+--------+-----+-----------------+
    """

    def __init__(self, mapping, labelCol, labelType):

        self.labelCol = labelCol
        self.estimator_cols = [labelCol]
        available_methods = ["numTopFeatures", "percentile", "fpr", "fwe", "fdr"]
        self.assert_type(mapping, dict, "mapping")
        for col_type in mapping.keys():
            # Assert column types of test
            self.assert_method(["continuous", "categorical"], col_type)
            # Assert Selection mode
            if "selectionMode" not in mapping[col_type]:
                mapping[col_type]["selectionMode"] = "numTopFeatures"
            self.assert_method(available_methods, mapping[col_type]["selectionMode"])
            # Assert Columns
            self.assert_type(mapping[col_type], dict, f"mapping[{col_type}]")
            value = mapping[col_type]["cols"]
            mapping[col_type]["cols"] = value if type(value) is list else value
            self.estimator_cols.extend(mapping[col_type]["cols"])
        self.mapping = mapping
        self.assert_method(["continuous", "categorical"], labelType)
        self.labelType = labelType

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
        estimators = []
        for col_type, params in self.mapping.items():
            assembler = VectorAssembler(
                inputCols=params["cols"], outputCol=f"{col_type}_features"
            )
            selector = (
                UnivariateFeatureSelector()
                .setLabelCol(self.labelCol)
                .setLabelType(self.labelType)
                .setSelectionMode(params["selectionMode"])
                .setFeatureType(col_type)
                .setFeaturesCol(f"{col_type}_features")
                .setOutputCol(f"selected_{col_type}")
            )
            if "selectionThreshold" in params:
                self.assert_type(
                    params["selectionThreshold"], (int, float), "selectionThreshold"
                )
                selector = selector.setSelectionThreshold(params["selectionThreshold"])
            pipeline = Pipeline(stages=[assembler, selector])
            estimators.append(pipeline)
        selected_cols = [f"selected_{col_type}" for col_type in self.mapping.keys()]
        assembler = VectorAssembler(
            inputCols=selected_cols, outputCol="selected_features"
        )
        estimators.append(assembler)
        final_pipeline = Pipeline(stages=estimators).fit(df)
        return final_pipeline.transform(df).select(*df.columns, "selected_features")
