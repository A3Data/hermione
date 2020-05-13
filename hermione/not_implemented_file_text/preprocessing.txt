import pandas as pd

from ml.preprocessing.normalization import Normalizer
from util import wrap_text

class Preprocessing:
    """
    Class to perform data preprocessing before training
    """
    
    def process(self, df: pd.DataFrame):
        """
        Perform data cleansing.
        
        Parameters
        ----------            
        df  :   pd.Dataframe
                Dataframe to be processed

        Returns
    	-------
        pd.Dataframe
            Cleaned Data Frame
        """
        pass