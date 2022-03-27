import pandas as pd

from ml.preprocessing.normalization import Normalizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from category_encoders import *
import logging

logging.getLogger().setLevel(logging.INFO)


class Preprocessing:
    """
    Class to perform data preprocessing before training
    """

    def __init__(self, norm_cols=None, oneHot_cols=None):
        """
        Constructor

        Parameters
        ----------
        norm_cols   : dict
                      Receives dict with the name of the normalization to be
                      performed and which are the columns
                      Ex: norm_cols = {'zscore': ['salary', 'price'],
                                       'min-max': ['heigth', 'age']}
        oneHot_cols : array
                      Receives an array with columns names to be categorized with One Hot Encoding
        Returns
        -------
        Preprocessing
        """
        self.norm_cols = norm_cols
        self.oneHot_cols = oneHot_cols
        self.ohe = OneHotEncoder(handle_unknown="ignore")

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
        df_copy["Pclass"] = df_copy.Pclass.astype("object")
        df_copy = df_copy.dropna()
        return df_copy

    def categ_encoding_oneHot(self, df: pd.DataFrame, step_train=False):
        """
        Perform encoding of the categorical variables using One Hot Encoding

        Parameters
        ----------
        df           : pd.Dataframe
                       Dataframe to be processed
        step_train   : bool
                       if True, the fit function is executed

        Returns
        -------
        pd.Dataframe
            Encoded Data Frame
        """
        logging.info("One hot encoding")
        df_copy = df.copy()

        if step_train:
            self.ohe.fit(df_copy[self.oneHot_cols])

        arr = self.ohe.transform(df_copy[self.oneHot_cols])
        df_copy = df_copy.join(arr).drop(self.oneHot_cols, axis=1)
        return df_copy

    def normalize(self, df: pd.DataFrame, step_train=False):
        """
        Apply normalization to the selected columns

        Parameters
        ----------
        df         : pd.DataFrame
                     dataframe with columns to be normalized
        step_train : bool
                     if True, the Normalizer is created and applied,
                     otherwise it is only applied

        Returns
        -------
        pd.DataFrame
            Normalized dataframe
        """
        logging.info("Normalizing")
        if step_train:
            self.norm = Normalizer(self.norm_cols)
            df = self.norm.fit_transform(df)
        else:
            df = self.norm.transform(df.copy())
        return df

    def execute(self, df, step_train=False, val_size=0.2):
        """
        Apply all preprocessing steps on the Dataframe

        Parameters
        ----------
        df         : pd.DataFrame
                     dataframe with columns to be normalized
        step_train : bool
                     if True, data is splited in train and val
        step_train : val_size
                     Size of the validation dataset

        Returns
        -------
        pd.DataFrame
            - One Preprocessed dataframe, if step_train is False
            - Two Preprocessed dataframes, if step_train is True
        """
        df = self.clean_data(df)
        df = self.categ_encoding_oneHot(df, step_train)

        if step_train:
            logging.info("Divide train and test")
            X_train, X_val = train_test_split(df, test_size=val_size, random_state=123)
            X_train = self.normalize(X_train, step_train=True)
            X_val = self.normalize(X_val, step_train=False)
            logging.info(f"shape train {X_train.shape} val {X_val.shape}")
            return X_train, X_val
        else:
            X = self.normalize(df, step_train=False)
            logging.info(f"shape {X.shape}")
            return X
