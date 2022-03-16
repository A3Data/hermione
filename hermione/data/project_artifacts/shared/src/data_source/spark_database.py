from pyspark.sql.dataframe import DataFrame
from hermione.base import DataSource


class SparkDataBase(DataSource):
    """
    Class used to read data from databases

    Parameters
    ----------
    spark_session  :   pyspark.sql.session.SparkSession
        SparkSession used to read data

    Attributes
    ----------
    spark  :   pyspark.sql.session.SparkSession
        SparkSession used to read data

    Examples
    --------
    >>> db_source = SparkDataBase(spark_session)
    >>> df = db_source.get_data()
    """

    def __init__(self, spark_session) -> None:

        self.spark = spark_session

    def get_data(self) -> DataFrame:
        """
        Returns a flat table in Dataframe

        Parameters
        -----------
        Returns
        -------
        pyspark.sql.dataframe.DataFrame
        """
        pass

    def open_connection(self, connection):
        """
        Opens the connection to the database

        Parameters
        -----------
        connection : string
                     Connection with database

        Returns
        -------
        bool
            Check if connection is open or not

        """
        pass

    def close_connection(self, connection):
        """
        Close the connection database

        Parameters
        -----------
        connection : string
                     Connection with database

        Returns
        -------
        bool
            Check if connection was closed

        """
        pass
