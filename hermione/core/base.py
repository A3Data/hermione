from pyspark.sql.dataframe import DataFrame
from pyspark.ml.base import Estimator
from pyspark.ml.util import MLWriter, MLReader
from abc import ABC, abstractmethod


class DataSource(ABC):
    @abstractmethod
    def get_data(self) -> DataFrame:
        """
        Abstract method that is implemented in classes that inherit it
        """
        pass


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
        Abstract method that should be implemented in every class that inherits TrainerModel
        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        pass


class Asserter:
    def assert_type(self, value, _type, name):
        """
        Assert if it the input value is of the correct type.

        Parameters
        ----------
        value : obj
            Any python object

        _type  : any
            the desired type of the value

        name : str
            Name of the input

        Returns
        -------
        """
        if not isinstance(value, _type):
            raise TypeError(f"Input `{name}` must be of type {_type}.")

    def assert_method(self, valid_methods, method):
        """
        Assert if it the input method is valid.

        Parameters
        ----------
        valid_methods : Iterable[str]
            iterable of valid methods

        method  : str
            input method

        Returns
        -------
        """
        options = "`" + "`, `".join(valid_methods) + "`."
        if method not in valid_methods:
            raise ValueError(f"Method not supported. Choose one from {options}")

    def assert_columns(self, df_columns):
        """
        Assert if it the DataFrame has the columns to be normalized.

        Parameters
        ----------
        df_columns  : List[str]
            input Spark DataFrame

        Returns
        -------
        """
        options = "`" + "`, `".join(df_columns) + "`."
        for col in self.estimator_cols:
            if col not in df_columns:
                raise ValueError(
                    f"Column `{col}` not present in the DataFrame. Avaliable columns are {options}"
                )
