from pyspark.ml.evaluation import (
    Evaluator,
    BinaryClassificationEvaluator as BCEval,
    MulticlassClassificationEvaluator as MCEval,
    RegressionEvaluator,
    ClusteringEvaluator,
)
from pyspark.ml.param import Param
import pyspark.sql.functions as f
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.util import MLWriter, MLReader
from tabulate import tabulate


class SparkMetrics:
    """Class used to calculate metrics of fitted estimators using Spark ML"""

    @classmethod
    def regression(cls, df, labelCol):
        """
        Calculates metrics to evaluate regression models

        Parameters
        ----------
        df : pyspark.sql.dataframe.DataFrame
            Spark DataFrame with model predictions

        labelCol : str
            Name of the outcome column

        Returns
        -------
        Dict[float]
            Returns a dict with metric results
        """
        metrics = ["rmse", "mse", "mae", "mape", "smape", "weighted_mape", "r2", "var"]
        results = dict()
        for metric in metrics:
            if metric in ["mape", "smape", "weighted_mape"]:
                evaluator = CustomRegressionEvaluator(
                    labelCol=labelCol, metricName=metric
                )
            else:
                evaluator = RegressionEvaluator(labelCol=labelCol, metricName=metric)
            results[metric] = evaluator.evaluate(df)
        metric_list = [[key, value] for key, value in results.items()]
        # Results
        print("Results")
        print("")
        print(tabulate(metric_list, headers=["Metric", "Value"], tablefmt="grid"))
        return results

    @classmethod
    def crossvalidation(
        cls,
        model,
        df,
        classification,
        numFolds=5,
        param_grid=None,
        evaluator=None,
        **kwargs
    ):
        """
        Calculates metrics to evaluate multiclass classification models

        Parameters
        ----------
        model : Estimator
            Estimator to be used in cross validation.

        df : pyspark.sql.dataframe.DataFrame
            Spark DataFrame that should be used in the cross validation

        classification : bool
            Boolean indicating whether `model` is a classification model or not.

        numFolds : int
            Number of folds for cross validation.

        param_grid : Dict[object]
            Dict in which the keys are params of the `model` and the values are the values the params should assume in the grid search.

        evaluator : Evaluator
            Evaluator used to select hyper-parameters that maximize the validator metric

        **kwargs : object
            Other arguments passed to the cross validator estimator

        Returns
        -------
        Tuple[Estimator, Dict[object]]
            Returns both the best model and a dict with its metric and params

        Examples
        --------
        >>> from hermione_spark.preprocessing import SparkPreprocessor
        >>> from pyspark.ml.classification import LinearRegression
        >>> data = [(1, 0.0, 5.0), (2, 1.0, 54.0), (3, 9.0, 27.0), (4, 8.0, 9.0)]
        >>> df = spark.createDataFrame(data, ['id', "num_col1", "label"])
        >>> df = SparkPreprocessor({'zscore': 'num_col1'}).fit_transform(df)
        >>> df.show()
        +---+--------+-----+--------------------+--------------------+
        | id|num_col1|label|       zscore_scaled|            features|
        +---+--------+-----+--------------------+--------------------+
        |  1|     0.0|  5.0|[-0.9667550799532...|[-0.9667550799532...|
        |  2|     1.0| 54.0|[-0.7519206177414...|[-0.7519206177414...|
        |  3|     9.0| 27.0|[0.9667550799532344]|[0.9667550799532344]|
        |  4|     8.0|  9.0|[0.7519206177414045]|[0.7519206177414045]|
        +---+--------+-----+--------------------+--------------------+
        >>> lr = LinearRegression()
        >>> SparkMetrics.crossvalidation(lr, df, False, numFolds=3, param_grid={'regParam': [0, 1, 2]})
        """
        if param_grid:
            grid = ParamGridBuilder()
            for param, values in param_grid.items():
                param_instance = Param(model, param, None)
                grid = grid.addGrid(param_instance, values)
            grid = grid.build()
        else:
            grid = ParamGridBuilder().build()

        if not evaluator:
            labelCol = model.getLabelCol()
            if classification:
                label_rows = df.select(labelCol).distinct().collect()
                metricLabels = sorted([int(c[labelCol]) for c in label_rows])
                if len(metricLabels) > 2:
                    evaluator = MCEval(labelCol=labelCol)
                else:
                    evaluator = BCEval(labelCol=labelCol)
            else:
                evaluator = RegressionEvaluator(labelCol=labelCol)
        cv = CrossValidator(
            estimator=model,
            evaluator=evaluator,
            numFolds=numFolds,
            estimatorParamMaps=grid,
            **kwargs
        )
        cv = cv.fit(df)
        bestModel = cv.bestModel
        metricName = evaluator.getMetricName()
        results = {
            metricName: max(cv.avgMetrics),
            "params": {
                param: bestModel.getOrDefault(param) for param in param_grid.keys()
            },
        }
        return (bestModel, results)

    @classmethod
    def __multiclass_classification(cls, df, labelCol, metricLabels):
        """
        Calculates metrics to evaluate multiclass classification models

        Parameters
        ----------
        df : pyspark.sql.dataframe.DataFrame
            Spark DataFrame with model predictions

        labelCol : str
            Name of the outcome column

        metricLabels : str
            Unique list of possible outcome values

        Returns
        -------
        Dict[float]
            Returns a dict with metric results
        """
        confusion_matrix = (
            df.withColumnRenamed(labelCol, "Outcome")
            .groupby("Outcome")
            .pivot("prediction", values=metricLabels)
            .count()
            .orderBy("Outcome")
            .fillna(0)
        )
        # Evaluate predicitons
        metrics = dict(
            f1="fMeasureByLabel", precision="precisionByLabel", recall="recallByLabel"
        )
        results = dict()
        for out in metricLabels:
            results[out] = dict()
            for name, metric in metrics.items():
                evaluator = MCEval(
                    labelCol=labelCol, metricName=metric, metricLabel=out
                )
                results[out][name] = evaluator.evaluate(df)
        metric_list = [
            [key, value["precision"], value["recall"], value["f1"]]
            for key, value in results.items()
        ]
        accuracy = MCEval(
            labelCol=labelCol, metricName="accuracy", metricLabel=metricLabels[0]
        ).evaluate(df)
        # Results
        print("Confusion Matrix")
        confusion_matrix.show()
        print("")
        print("Results")
        print(tabulate([[accuracy]], headers=["Accuracy"], tablefmt="grid"))
        print("")
        print(
            tabulate(
                metric_list,
                headers=["Outcome", "Precision", "Recall", "F1"],
                tablefmt="grid",
            )
        )
        return results

    @classmethod
    def __binary_classification(cls, df, labelCol, metricLabels):
        """
        Calculates metrics to evaluate binary classification models

        Parameters
        ----------
        df : pyspark.sql.dataframe.DataFrame
            Spark DataFrame with model predictions

        labelCol : str
            Name of the outcome column

        metricLabels : str
            Unique list of possible outcome values

        Returns
        -------
        Dict[float]
            Returns a dict with metric results
        """
        metrics = ["accuracy", "roc_auc", "precision", "recall", "f1"]
        results = dict()
        results["labels"] = {key: {} for key in metricLabels}
        for metric in metrics:
            evaluator = CustomBinaryEvaluator(metric, labelCol)
            res = evaluator.evaluate(df)
            if metric in ["accuracy", "roc_auc"]:
                results[metric] = res
            else:
                for index, label in enumerate(metricLabels):
                    results["labels"][label][metric] = res[index]
        metric_list = [
            [key, value["precision"], value["recall"], value["f1"]]
            for key, value in results["labels"].items()
        ]
        # Results
        print("Confusion Matrix")
        cm = evaluator._create_confusion_matrix(df)
        cm.show()
        print("")
        print("Results")
        print(
            tabulate(
                [[results["accuracy"], results["roc_auc"]]],
                headers=["Accuracy", "ROC AUC"],
                tablefmt="grid",
            )
        )
        print("")
        print(
            tabulate(
                metric_list,
                headers=["Outcome", "Precision", "Recall", "F1"],
                tablefmt="grid",
            )
        )
        return results

    @classmethod
    def classification(cls, df, labelCol):
        """
        Checks whether it should be applied the binary or multiclass classification evaluator

        Parameters
        ----------
        df : pyspark.sql.dataframe.DataFrame
            Spark DataFrame with model predictions

        labelCol : str
            Name of the outcome column

        Returns
        -------
        Dict[float]
            Returns a dict with metric results
        """
        label_rows = df.select(labelCol).distinct().collect()
        metricLabels = sorted([int(c[labelCol]) for c in label_rows])
        if len(metricLabels) > 2:
            return cls.__multiclass_classification(df, labelCol, metricLabels)
        else:
            return cls.__binary_classification(df, labelCol, metricLabels)

    @classmethod
    def clusterization(cls, df, distanceMeasure="squaredEuclidean"):
        """
        Calculates metrics to evaluate a clustrization

        Parameters
        ----------
        df : pyspark.sql.dataframe.DataFrame
            Spark DataFrame with model predictions

        Returns
        -------
        Dict[float]
            Returns a dict with metric results
        """
        evaluator = ClusteringEvaluator(distanceMeasure=distanceMeasure)
        results = {
            "silhouette": evaluator.evaluate(df),
            "distanceMeasure": distanceMeasure,
        }
        return results


class CustomBinaryEvaluator(Evaluator, MLWriter, MLReader):
    """
    Class used evaluate binary classification using metrics not default in Spark ML

    Parameters
    ----------
    metricName : str
        The metric that should be calculated. Available metrics are "accuracy", "precision", "recall", "f1", "roc_auc".

    labelCol : str
        Name of the outcome column

    metricLabel : str
        The label from which the metrics should be returned. It must be a level of `labelCol`.

    Attributes
    ----------
    metricLabel : str
        The label from which the metrics should be returned. It must be a level of `labelCol`.

    metricLabels : str
        The levels of `labelCol`.

    metricName : str
        The metric that should be calculated. Available metrics are "accuracy", "precision", "recall", "f1", "roc_auc".

    labelCol : str
        Name of the outcome column

    Examples
    --------
    >>> from hermione_spark.preprocessing import SparkPreprocessor
    >>> from pyspark.ml.classification import LogisticRegression
    >>> data = [(1, 0.0, 0), (2, 1.0, 1), (3, 9.0, 1), (4, 8.0, 0)]
    >>> df = spark.createDataFrame(data, ['id', "num_col1", "label"])
    >>> df = SparkPreprocessor({'zscore': 'num_col1'}).fit_transform(df)
    >>> df.show()
    +---+--------+-----+--------------------+--------------------+
    | id|num_col1|label|       zscore_scaled|            features|
    +---+--------+-----+--------------------+--------------------+
    |  1|     0.0|    0|[-0.9667550799532...|[-0.9667550799532...|
    |  2|     1.0|    1|[-0.7519206177414...|[-0.7519206177414...|
    |  3|     9.0|    1|[0.9667550799532344]|[0.9667550799532344]|
    |  4|     8.0|    0|[0.7519206177414045]|[0.7519206177414045]|
    +---+--------+-----+--------------------+--------------------+
    >>> lr = LogisticRegression()
    >>> eva = CustomBinaryEvaluator('precision', 'label')
    >>> df_pred = lr.fit(df).transform(df)
    >>> eva.evaluate(df_pred)
    """

    def __init__(self, metricName, labelCol, metricLabel=None):

        self.metricName = metricName
        self.labelCol = labelCol
        self.metricLabel = metricLabel
        if metricName not in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            raise Exception(
                "Metric not available. Please, choose one from accuracy, precision, recall, F1 or ROC AUC."
            )

    def _evaluate(self):
        super()._evaluate()

    def isLargerBetter(self):
        return True

    def getMetricName(self):
        """
        Returns the chosen metric

        Parameters
        ----------
        Returns
        -------
        str
        """
        return self.metricName

    def _create_confusion_matrix(self, df):
        """
        Creates the model confusion matrix based on predictions

        Parameters
        ----------
        df : pyspark.sql.dataframe.DataFrame
            Spark DataFrame with model predictions

        Returns
        -------
        pyspark.sql.dataframe.DataFrame
        """
        metricLabels = df.select(self.labelCol).distinct().collect()
        self.metricLabels = sorted([int(c[self.labelCol]) for c in metricLabels])
        cm = (
            df.withColumnRenamed(self.labelCol, "Outcome")
            .groupby("Outcome")
            .pivot("prediction", values=self.metricLabels)
            .count()
            .orderBy("Outcome")
            .fillna(0)
        )
        return cm

    def _get_metric(self, df):
        """
        Calculates the metric.

        Parameters
        ----------
        df : pyspark.sql.dataframe.DataFrame
            Spark DataFrame with model predictions

        Returns
        -------
        list[float] | float
        """
        cm = self._create_confusion_matrix(df)
        metrics = [c.asDict() for c in cm.collect()]
        true_vec = [
            metrics[0]["0"] if value == 0 else metrics[1]["1"]
            for value in self.metricLabels
        ]
        false_vec = [
            metrics[1]["0"] if value == 0 else metrics[0]["1"]
            for value in self.metricLabels
        ]
        if self.metricName == "accuracy":
            return _safe_division(sum(true_vec), (sum(false_vec) + sum(true_vec)))
        elif self.metricName == "precision":
            return [_safe_division(t, t + f) for t, f in zip(true_vec, false_vec)]
        elif self.metricName == "recall":
            return [_safe_division(t, t + f) for t, f in zip(true_vec, false_vec[::-1])]
        elif self.metricName == "f1":
            precision_vec = [
                _safe_division(t, t + f) for t, f in zip(true_vec, false_vec)
            ]
            recall_vec = [
                _safe_division(t, t + f) for t, f in zip(true_vec, false_vec[::-1])
            ]
            return [
                _safe_division(2 * precision * recall, precision + recall)
                for recall, precision in zip(recall_vec, precision_vec)
            ]
        elif self.metricName == "roc_auc":
            evaluator = BCEval(labelCol=self.labelCol)
            return evaluator.evaluate(df)

    def evaluate(self, df):
        """
        Wrapper for returning the metric

        Parameters
        ----------
        df : pyspark.sql.dataframe.DataFrame
            Spark DataFrame with model predictions

        Returns
        -------
        list[float] | float
        """
        if self.metricLabel is None:
            return self._get_metric(df)
        else:
            return self._get_metric(df)[self.metricLabel]


def _safe_division(numerator, denominator):
    denominator = denominator if denominator != 0 else 1
    return numerator / denominator


class CustomRegressionEvaluator(Evaluator, MLWriter, MLReader):
    """
    Class used evaluate binary classification using metrics not default in Spark ML

    Parameters
    ----------
    metricName : str
        The metric that should be calculated. Available metrics are "mape", "smape", "weighted_mape".

    labelCol : str
        Name of the outcome column

    Attributes
    ----------
    metricName : str
        The metric that should be calculated. Available metrics are "mape", "smape", "weighted_mape".

    labelCol : str
        Name of the outcome column

    Examples
    --------
    >>> from hermione_spark.preprocessing import SparkPreprocessor
    >>> from pyspark.ml.regression import LinearRegression
    >>> data = [(1, 0.0, 5.0), (2, 1.0, 54.0), (3, 9.0, 27.0), (4, 8.0, 9.0)]
    >>> df = spark.createDataFrame(data, ['id', "num_col1", "label"])
    >>> df = SparkPreprocessor({'zscore': 'num_col1'}).fit_transform(df)
    >>> df.show()
    +---+--------+-----+--------------------+--------------------+
    | id|num_col1|label|       zscore_scaled|            features|
    +---+--------+-----+--------------------+--------------------+
    |  1|     0.0|  5.0|[-0.9667550799532...|[-0.9667550799532...|
    |  2|     1.0| 54.0|[-0.7519206177414...|[-0.7519206177414...|
    |  3|     9.0| 27.0|[0.9667550799532344]|[0.9667550799532344]|
    |  4|     8.0|  9.0|[0.7519206177414045]|[0.7519206177414045]|
    +---+--------+-----+--------------------+--------------------+
    >>> lr = LinearRegression()
    >>> eva = CustomRegressionEvaluator('mape', 'label')
    >>> df_pred = lr.fit(df).transform(df)
    >>> eva.evaluate(df_pred)
    """

    def __init__(self, metricName, labelCol):

        self.metricName = metricName
        self.labelCol = labelCol
        if metricName not in ["mape", "smape", "weighted_mape"]:
            raise Exception(
                "Metric not available. Please, choose one from mape or smape"
            )

    def _evaluate(self):
        super()._evaluate()

    def isLargerBetter(self):
        return False

    def getMetricName(self):
        """
        Returns the chosen metric

        Parameters
        ----------
        Returns
        -------
        str
        """
        return self.metricName

    def _get_metric(self, df):
        """
        Calculates the metric.

        Parameters
        ----------
        df : pyspark.sql.dataframe.DataFrame
            Spark DataFrame with model predictions

        Returns
        -------
        float
        """
        metrics = (
            df.withColumn(
                "mape",
                f.abs(
                    ((f.col(self.labelCol) + 1) - (f.col("prediction") + 1))
                    / (f.col(self.labelCol) + 1)
                )
                * 100,
            )
            .withColumn("abs_dif", f.abs(f.col("prediction") - f.col(self.labelCol)))
            .withColumn("abs_label", f.abs(f.col(self.labelCol)))
            .withColumn(
                "smape",
                (f.col("abs_dif") / (f.col("abs_label") + f.abs(f.col("prediction"))))
                * 100,
            )
            .agg(
                f.mean("mape").alias("mape"),
                f.mean("smape").alias("smape"),
                (100 * (f.sum("abs_dif") / f.sum("abs_label"))).alias("weighted_mape"),
            )
            .collect()[0]
        )
        if self.metricName == "mape":
            return metrics["mape"]
        elif self.metricName == "smape":
            return metrics["smape"]
        elif self.metricName == "weighted_mape":
            return metrics["weighted_mape"]

    def evaluate(self, df):
        """
        Wrapper for returning the metric

        Parameters
        ----------
        df : pyspark.sql.dataframe.DataFrame
            Spark DataFrame with model predictions

        Returns
        -------
        float
        """
        return self._get_metric(df)
