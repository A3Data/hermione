import pandas as pd

from ml.preprocessing.normalization import Normalizer
from category_encoders import *
import logging

logging.getLogger().setLevel(logging.INFO)

class Preprocessing:
    """
    Class to perform data preprocessing before training
    """
    
    def clean_data(self, df: pd.DataFrame):
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
        logging.info("Cleaning data")
        df_copy = df.copy()
        df_copy['Pclass'] = df_copy.Pclass.astype('object')
        df_copy = df_copy.dropna()
        return df_copy

    def categ_encoding(self, df: pd.DataFrame):
        """
        Perform encoding of the categorical variables

        Parameters
        ----------            
        df  :   pd.Dataframe
                Dataframe to be processed

        Returns
    	-------
        pd.Dataframe
            Cleaned Data Frame
        """
        logging.info("Category encoding")
        df_copy = df.copy()
        df_copy = pd.get_dummies(df_copy)
        return df_copy
