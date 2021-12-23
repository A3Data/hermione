import pandas as pd

from ml.data_source.base import DataSource

class Spreadsheet(DataSource):
    """
    Class to read files from spreadsheets or raw text files
    """
    
    def get_data(self, path)->pd.DataFrame:
        """
        Returns a flat table in Dataframe
        
        Parameters
        ----------            
        arg : type
              description
        
        Returns
        -------
        pd.DataFrame
            Dataframe with data
        """
        return pd.read_csv(path)[['Survived', 'Pclass', 'Sex', 'Age']]