from abc import ABC, abstractmethod
from ml.model.wrapper import Wrapper
from ml.model.metrics import Metrics


class Trainer(ABC):
    def __init__(self):
        """
        Constructor

        Parameters
        ----------
        None

        Returns
        -------
        Trainer
        """

    @abstractmethod
    def train(self):
        """
        Abstract method that should be implemented in every class
        that inherits TrainerModel
        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        pass


class TrainerSklearn(Trainer):
    def train(
        self,
        train,
        val,
        y_name,
        classification: bool,
        algorithm,
        columns=None,
        **params
    ):
        """
        Method that builds the Sklearn model

        Parameters
        ----------
        train             : pd.Dataframe
                            data to train the model
        val               : pd.Dataframe
                            data to validate the model
        y_name            : str
                            target name
        algorithm         : Sklearn algorithm
                            algorithm to be trained
        classification    : bool
                            if True, classification model training takes place,
                            otherwise Regression
        columns           : array
                            columns name to be used in the train

        Returns
        -------
        Wrapper
        """
        model = algorithm(**params)
        y_train = train[y_name]
        y_val = val[y_name]
        X_train = train[columns]
        X_val = val[columns]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        y_probs = model.predict_proba(X_val)[:, 1]
        if classification:
            res_metrics = Metrics.classification(y_val.values, y_pred, y_probs)
        else:
            res_metrics = Metrics.regression(y_val.values, y_pred)
        model = Wrapper(model, res_metrics, X_train.columns)
        return model


class TrainerSklearnUnsupervised(Trainer):
    def train(self, X, algorithm, **params):
        """
        Method that builds the Sklearn model

        Parameters
        ----------
        model_name        : str
                            model name

        Returns
        -------
        Wrapper
        """
        model = algorithm(**params)
        columns = list(X.columns)
        model.fit(X)
        labels = model.predict(X)
        res_metrics = Metrics.clusterization(X, labels)
        model = Wrapper(model, res_metrics, columns)
        return model
