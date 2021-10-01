from pyspark.ml.evaluation import (
    Evaluator,
    BinaryClassificationEvaluator as BCEval, 
    MulticlassClassificationEvaluator as MCEval
)
from pyspark.ml.util import MLWriter, MLReader
from pyspark.sql.dataframe import DataFrame
from tabulate import tabulate
import numpy as np
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate


class Metrics:

    @classmethod
    def smape(cls, A, F):
        """
        Calculates the smape value between the real and the predicted

        Parameters
        ----------
        A : array
            Target values
        F : array
            Predicted values

        Returns
        -------
        float: smape value
        """
        return 100/len(A) * np.sum(np.abs(F - A) / (np.abs(A) + np.abs(F)))

    @classmethod
    def __custom_score(cls, y_true, y_pred):
        """
        Creates a custom metric

        Parameters
        ----------
        y_true : array
                 Target values
        y_pred : array
                 Predicted values

        Returns
        -------
        sklearn.metrics
        """
        #return sklearn.metrics.fbeta_score(y_true, y_pred, 2)
        pass

    @classmethod
    def customized(cls, y_true, y_pred):
        """
        Creates a custom metric

        Parameters
        ----------
        y_true : array
                 Target values
        y_pred : array
                 Predicted values

        Returns
        -------
        float
        """
        custom_metric = make_scorer(cls.__custom_score, greater_is_better=True)
        return custom_metric

    @classmethod
    def mape(cls, y_true, y_pred):
        """
        Calculates the map value between the real and the predicted

        Parameters
        ----------
        y_true : array
                 Target values
        y_pred : array
                 Predicted values

        Returns
        -------
        float : value of mape
        """
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs(((y_true+1) - (y_pred+1)) / (y_true+1))) * 100

    @classmethod
    def regression(cls, y_true, y_pred):
        """
        Calculates some metrics for regression problems

        Parameters
        ----------
        y_true : array
                 Target values
        y_pred : array
                 Predicted values

        Returns
        -------
        dict : metrics results
        """
        results = {'mean_absolute_error': round(mean_absolute_error(
            y_true, y_pred), 7),
                    'root_mean_squared_error': round(np.sqrt(
                          mean_squared_error(y_true, y_pred)), 7),
                    'r2': round(r2_score(y_true, y_pred), 7),
                    'smape': round(cls.smape(y_true, y_pred), 7),
                    'mape': round(cls.mape(y_true, y_pred), 7)
                     }
        return results

    @classmethod
    def crossvalidation(cls, model, X, y, classification: bool,
                        cv=5, agg=np.mean):
        if classification:
            if len(set(y)) > 2:
                metrics = ['accuracy', 'f1_weighted',
                           'recall_weighted', 'precision_weighted']
            else:
                metrics = ['accuracy', 'f1', 'recall', 'precision', 'roc_auc']
        else:
            metrics = ['mean_absolute_error', 'r2', 'root_mean_squared_error',
                       'smape', 'mape']
        res_metrics = cross_validate(model, X, y, cv=cv,
                                     return_train_score=False,
                                     scoring=metrics)
        results = {metric.replace("test_", ""): round(agg(
            res_metrics[metric]), 7)
                   for metric in res_metrics}
        return results

    @classmethod
    def __multiclass_classification(cls, df, labelCol, metricLabels):
        """
        Calculates some metrics for multiclass classification problems

        Parameters
        ----------
        df         : DataFrame
            Dataframe with model predictions
        labelCol   : str
            Name of the outcome column
        metricLabels   : str
            Unique list of possible outcome values

        Returns
        -------
        None
        """
        confusion_matrix = (
            df.withColumnRenamed(labelCol, 'Outcome')
            .groupby('Outcome')
            .pivot('prediction', values=metricLabels)
            .count()
            .orderBy('Outcome')
            .fillna(0)
        )
        # Evaluate predicitons
        metrics = dict(f1='fMeasureByLabel', precision='precisionByLabel', recall='recallByLabel')
        results = dict()
        for out in metricLabels:
            results[out] = dict()
            for name, metric in metrics.items():
                evaluator = MCEval(labelCol=labelCol, metricName=metric, metricLabel=out)
                results[out][name] = evaluator.evaluate(df)
        metric_list = [[key, value['precision'], value['recall'], value['f1']] for key, value in results.items()]
        accuracy = MCEval(labelCol=labelCol,  metricName='accuracy', metricLabel=metricLabels[0]).evaluate(df)
        # Results
        print("Confusion Matrix")
        confusion_matrix.show()
        print("")
        print("Results")
        print(tabulate([[accuracy]], headers=['Accuracy'], tablefmt='grid'))
        print("")
        print(tabulate(metric_list, headers=['Outcome', 'Precision', 'Recall', 'F1'], tablefmt='grid'))
        return results

    @classmethod
    def __binary_classification(cls, df, labelCol, metricLabels):
        """
        Calculates some metrics for binary classification problems

        Parameters
        ----------
        df         : DataFrame
            Dataframe with model predictions
        labelCol   : str
            Name of the outcome column
        metricLabels   : str
            Unique list of possible outcome values

        Returns
        -------
        None
        """
        metrics = ['accuracy', 'roc_auc', 'precision', 'recall', 'f1']
        results = dict()
        results['labels'] = {key:{} for key in metricLabels}
        for metric in metrics:
            evaluator = BinaryEvaluator(metric, labelCol)
            res = evaluator.evaluate(df)
            if metric in ['accuracy', 'roc_auc']:
                results[metric] = res
            else:
                for index, label in enumerate(metricLabels):
                    results['labels'][label][metric] = res[index]
        metric_list = [[key, value['precision'], value['recall'], value['f1']] for key, value in results['labels'].items()]
        # Results
        print("Confusion Matrix")
        cm = evaluator._create_confusion_matrix(df)
        cm.show()
        print("")
        print("Results")
        print(tabulate([[results['accuracy'], results['roc_auc']]], headers=['Accuracy', 'ROC AUC'], tablefmt='grid'))
        print("")
        print(tabulate(metric_list, headers=['Outcome', 'Precision', 'Recall', 'F1'], tablefmt='grid'))
        return results

    @classmethod
    def classification(cls, df: DataFrame, labelCol: str):
        """
        Checks which classification method will be applied:
        binary or multiclass

        Parameters
        ----------
        df         : DataFrame
            Dataframe with model predictions
        labelCol   : str
            Name of the outcome column

        Returns
        -------
        None
        """
        pred_rows = df.select(labelCol).distinct().collect()
        metricLabels = sorted([int(c[labelCol]) for c in pred_rows])
        if len(metricLabels) > 2:
            return cls.__multiclass_classification(df, labelCol, metricLabels)
        else:
            return cls.__binary_classification(df, labelCol, metricLabels)

    @classmethod
    def clusterization(cls, X, labels):
        """
        Calculates some metrics on clustering quality

        Parameters
        ----------
        X      : array[array], shape (n_linha, n_colunas)
                 Matrix with the values that were used in the cluster
        labels : array, shape (n_linha, 1)
                 Vector with labels selected by the clustering method
                 (eg KMeans)

        Returns
        -------
        dict : metrics results
        """
        results = {'silhouette': silhouette_score(X, labels,
                                                  metric='euclidean'),
                   'calinski_harabaz': calinski_harabaz_score(X, labels)}
        return results

class BinaryEvaluator(Evaluator, MLWriter, MLReader):
    
    def __init__(self, metricName, labelCol, metricLabel=None):
        
        self.metricName = metricName
        self.labelCol = labelCol
        self.metricLabel = metricLabel
        if metricName not in ['accuracy', 'precision', 'recall', 'f1','roc_auc']:
            raise Exception('Metric not available. Please, choose one from accuracy, precision, recall, F1 or ROC AUC.')
        
    def _evaluate(self):
        super()._evaluate()

    def isLargerBetter(self):
        return True
    
    def _create_confusion_matrix(self, df):
        
        metricLabels = df.select(self.labelCol).distinct().collect()
        self.metricLabels = sorted([int(c[self.labelCol]) for c in metricLabels])
        cm = (
            df.withColumnRenamed(self.labelCol, 'Outcome')
            .groupby('Outcome')
            .pivot('prediction', values=self.metricLabels)
            .count()
            .orderBy('Outcome')
            .fillna(0)
        )
        return cm
    
    def _get_metric(self, df):
        
        cm = self._create_confusion_matrix(df)
        metrics = [c.asDict() for c in cm.collect()]
        true_vec = [metrics[0]['0'] if value == 0 else metrics[1]['1'] for value in self.metricLabels]
        false_vec = [metrics[1]['0'] if value == 0 else metrics[0]['1'] for value in self.metricLabels]
        if self.metricName == 'accuracy':
            return sum(true_vec) / (sum(false_vec) + sum(true_vec))
        elif self.metricName == 'precision':
            return [_safe_division(t, t + f) for t, f in zip(true_vec, false_vec)]
        elif self.metricName == 'recall':
            return [_safe_division(t, t + f) for t, f in zip(true_vec, false_vec[::-1])]
        elif self.metricName == 'f1':
            precision_vec = [_safe_division(t, t + f) for t, f in zip(true_vec, false_vec)]
            recall_vec = [_safe_division(t, t + f) for t, f in zip(true_vec, false_vec[::-1])]
            return [_safe_division(2 * precision * recall, precision + recall) for recall, precision in zip(recall_vec, precision_vec)]
        elif self.metricName == 'roc_auc':
            evaluator = BCEval(labelCol=self.labelCol)
            return evaluator.evaluate(df)
        
    def evaluate(self, df):
        
        if self.metricLabel is None:
            return self._get_metric(df)
        else:
            return self._get_metric(df)[self.metricLabel]


def _safe_division(numerator, denominator):
    denominator = denominator if denominator != 0 else 1
    return numerator / denominator