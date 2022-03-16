from ._wrapper import SparkWrapper
from ._metrics import SparkMetrics
from hermione.core.base import Trainer


class SparkTrainer(Trainer):
    """Class used to train supervised models with Spark ML"""

    def train(
        self,
        df,
        classification,
        algorithm,
        data_split=("train_test", {"test_size": 0.2}),
        **kwargs
    ):
        """
        Builds the Spark ML model

        Parameters
        ----------
        df : pyspark.sql.dataframe.DataFrame
            Spark DataFrame with the data used in training

        classification : bool
            Boolean indicating whether `model` is a classification model or not.

        algorithm : Estimator
            Estimator to be used in fitting the model.

        data_split : Tuple[str, Dict[object]]
            Tuple with the data split strategy to train the model. Available strategies are "train_test" and "cv".
            The second item of the tuple is a dict with arguments used in each strategy.
            Ex: ('train_test', {'test_size': 0.2})
                ('cv', {'numFolds': 4, 'param_grid': {'regParam': [0, 1, 2]}})

        **kwargs : object
            Other arguments passed to the `algorithm`

        Returns
        -------
        SparkWrapper
        """
        model = algorithm(**kwargs)  # model
        if data_split[0] == "train_test":
            test_size = data_split[1]["test_size"]
            df_train, df_test = df.randomSplit([1 - test_size, test_size], seed=13)
            model = model.fit(df_train)
            df_pred = model.transform(df_test)
            labelCol = model.getLabelCol()
            if classification:
                res_metrics = SparkMetrics.classification(df_pred, labelCol)
            else:
                res_metrics = SparkMetrics.regression(df_pred, labelCol)
        elif data_split[0] == "cv":
            model, res_metrics = SparkMetrics.crossvalidation(
                model, df, classification, **data_split[1]
            )
        final_model = SparkWrapper(model, res_metrics)
        return final_model


class SparkUnsupTrainer(Trainer):
    """Class used to train unsupervised models with Spark ML"""

    def train(self, df, algorithm, metric_params=None, **kwargs):
        """
        Builds the Spark ML model

        Parameters
        ----------
        df : pyspark.sql.dataframe.DataFrame
            Spark DataFrame with the data used in training

        algorithm : Estimator
            Estimator to be used in fitting the model.

        metric_params : Dict[object]
            Dict with arguments to be passed to the metric evaluator

        **kwargs : object
            Other arguments passed to the `algorithm`

        Returns
        -------
        SparkWrapper
        """
        model = algorithm(**kwargs)  # model
        model = model.fit(df)
        df_pred = model.transform(df)
        metric_params = metric_params if metric_params else {}
        res_metrics = SparkMetrics.clusterization(df_pred, **metric_params)
        final_model = SparkWrapper(model, res_metrics)
        return final_model
