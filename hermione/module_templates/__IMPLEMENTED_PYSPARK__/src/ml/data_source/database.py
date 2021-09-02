from pyspark.sql.dataframe import DataFrame
from ml.data_source.base import DataSource

class DataBase(DataSource):
    """
    Class for data from databases with Spark
    """

    def __init__(self, spark_session) -> None:
        """
        Instantiate class
        
        Parameters
        ----------            
        spark_session  :   pyspark.sql.session.SparkSession
            SparkSession used to read data

        Returns
    	-------
        self:
            returns an instance of the object
        """
        self.spark = spark_session
    
    def get_data(self)-> DataFrame:
        """
        Returns a flat table in Dataframe
        
        Parameters
        -----------         
        arg : type
              description
        
        Returns
        -------
        pd.DataFrame    
            Dataframe with data
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
    
    def close_connection(self, connection ):
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
