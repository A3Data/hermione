import pandas as pd

from ml.data_source.base import DataSource

class DataBase(DataSource):
    
    def __init__(self):
        """
        Constructor.
        
        Parameters
        -----------       
        arg : type
              description
        
        Returns
        -------
        class Object
        """
        pass
    
    def get_data(self)->pd.DataFrame:
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
