from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
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
    def __multiclass_classification(cls, df, labelCol, pred_values):
        """
        Calculates some metrics for multiclass classification problems

        Parameters
        ----------
        y_true    : array
                    Target values
        y_pred    : array
                    Predicted values

        Returns
        -------
        dict : metrics results
        """
        confusion_matrix = (
            df.withColumnRenamed(labelCol, 'Outcome')
            .groupby('Outcome')
            .pivot('prediction', values=pred_values)
            .count()
            .orderBy('Outcome')
        )
        # Evaluate predicitons
        results = {
            'accuracy': [],
            'f1': [],
            'weightedPrecision': [],
            'weightedRecall': [],
        }
        for metric in results.keys():
            metric_calc = []
            for out in pred_values:
                evaluator = MulticlassClassificationEvaluator(
                    labelCol=labelCol, 
                    metricName=metric, 
                    metricLabel=out
                )
                metric_calc.append(evaluator.evaluate(df))
            results[metric] = metric_calc
        metric_list = [
            [out, precision, recall, f1] 
            for out, precision, recall, f1 
            in zip(pred_values, results['weightedPrecision'], results['weightedRecall'], results['f1'])
        ]
        # Results
        print("Confusion Matrix")
        confusion_matrix.show()
        print("")
        print("Results")
        print(tabulate([results['accuracy']], headers=['Accuracy'], tablefmt='grid'))
        print("")
        print(tabulate(metric_list, headers=['Outcome', 'Precision', 'Recall', 'F1'], tablefmt='grid'))

    @classmethod
    def __binary_classification(cls, df, labelCol, pred_values):
        """
        Calculates some metrics for binary classification problems

        Parameters
        ----------
        y_true    : array
                    Target values
        y_pred    : array
                    Predicted values

        Returns
        -------
        dict : metrics results
        """
        confusion_matrix = (
            df.withColumnRenamed(labelCol, 'Outcome')
            .groupby('Outcome')
            .pivot('prediction', values=pred_values)
            .count()
            .orderBy('Outcome')
        )
        # Evaluate predicitons
        evaluator = BinaryClassificationEvaluator(labelCol=labelCol)
        metrics = [c.asDict() for c in confusion_matrix.collect()]
        tp_vec = [metrics[0]['0'] if value == 0 else metrics[1]['1'] for value in pred_values]
        fn_vec = [metrics[0]['1'] if value == 0 else metrics[1]['0'] for value in pred_values]
        fp_vec = [metrics[1]['0'] if value == 0 else metrics[0]['1'] for value in pred_values]
        # Compute metrics
        accuracy = sum(tp_vec) / (sum(fn_vec) + sum(fp_vec))
        recall_vec = [tp / (tp + fn) for tp, fn in zip(tp_vec, fn_vec)]
        precision_vec = [tp / (tp + fp) for tp, fp in zip(tp_vec, fp_vec)]
        f1 = [(2 * precision * recall) / (precision + recall) for recall, precision in zip(recall_vec, precision_vec)]
        roc_auc = evaluator.evaluate(df)
        metric_list = [
            [out, precision, recall, f1] 
            for out, precision, recall, f1 
            in zip(pred_values, precision_vec, recall_vec, f1)
        ]
        # Results
        print("Confusion Matrix")
        confusion_matrix.show()
        print("")
        print("Results")
        print(tabulate([[accuracy, roc_auc]], headers=['Accuracy', 'ROC AUC'], tablefmt='grid'))
        print("")
        print(tabulate(metric_list, headers=['Outcome', 'Precision', 'Recall', 'F1'], tablefmt='grid'))

    @classmethod
    def classification(cls, df, labelCol):
        """
        Checks which classification method will be applied:
        binary or multiclass

        Parameters
        ----------
        y_true    : array
                    Target values
        y_pred    : array
                    Predicted values
        y_probs   : array
                    Probabilities values

        Returns
        -------
        dict: metrics results
        """
        pred_rows = df.select('prediction').distinct().collect()
        pred_values = sorted([int(c['prediction']) for c in pred_rows])
        if len(pred_values) > 2:
            results = cls.__multiclass_classification(df, labelCol, pred_values)
        else:
            results = cls.__binary_classification(df, labelCol, pred_values)
        return results

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
