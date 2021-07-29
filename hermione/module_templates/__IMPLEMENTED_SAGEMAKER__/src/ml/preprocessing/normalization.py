import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, normalize, MinMaxScaler
from scipy.stats import zscore

class Normalizer:

    def __init__(self, norm_cols: dict):
        """
        Constructor
        
    	Parameters
    	----------            
        norm_cols : dict
                    Receives dict with the name of the normalization to be 
                    performed and which are the columns
                    Ex: norm_cols = {'zscore': ['salary', 'price'], 
                                     'min-max': ['heigth', 'age']}
                    
    	Returns
    	-------
        Normalization
        """
        self.norm_cols = norm_cols
        self.col_names = [name for norm in norm_cols for name in norm_cols[norm]]
        self.norms = {'min-max': MinMaxScaler, 
                      'standard': StandardScaler}
        self.fitted = False
        
    def statistics(self, df : pd.DataFrame):
        """
        Calculates dataframe statistics
        
    	Parameters
    	----------            
        df : dataframe to calculate the statistics for each column
                    
    	Returns
    	-------
        None
        """
        zip_cols = lambda result: zip(result.index.values, result.values)
        self.col_min = {col: value for col, value in zip_cols(df[self.col_names].min())}
        self.col_max = {col: value for col, value in zip_cols(df[self.col_names].max())}
        self.col_std = {col: value for col, value in zip_cols(df[self.col_names].std())}
        self.col_mean = {col: value for col, value in zip_cols(df[self.col_names].mean())}
        self.col_median = {col: value for col, value in zip_cols(df[self.col_names].median())}

    def __apply_func(self, X, normalization):
        """
        Creates the normalization object
        
    	Parameters
    	----------            
        X             : array
                        Data to be normalized
        normalization : Normalization
                        Normalization to be applied
                    
    	Returns
    	-------
        Normalization
        """
        normalization.fit(X)
        return normalization

    def fit(self, df: pd.DataFrame):
        """
        Generates normalization object for each column
        
    	Parameters
    	----------            
        df : pd.DataFrame
             dataframe with columns to be normalized
                    
    	Returns
    	-------
        None
        """
        self.statistics(df)
        self.normalization = dict()
        for norm in self.norm_cols:
            if norm in ['zscore', 'log10']:
                continue
            for col in self.norm_cols[norm]:
                self.normalization[col] = self.__apply_func(df[col].values.reshape(-1, 1), self.norms[norm]())
        self.fitted = True

    def transform(self, df: pd.DataFrame):
        """
        Apply normalization to each column
        
    	Parameters
    	----------            
        df : pd.DataFrame
             dataframe with columns to be normalized
                    
    	Returns
    	-------
        pd.DataFrame
        """
        if not self.fitted:
            raise Exception("Not yet fitted.")
        
        for norm in self.norm_cols:
            if norm == 'zscore':
                for col in self.norm_cols[norm]:
                    df.loc[:,col] = (df[col].values - self.col_mean[col])/self.col_std[col]
            elif norm == 'log10':
                for col in self.norm_cols[norm]:
                    df.loc[:,col] = np.log10(df[col].values)
            else:
                for col in self.norm_cols[norm]:
                    df.loc[:,col] = self.normalization[col].transform(df[col].values.reshape(-1, 1))
        return df
    
    def inverse_transform(self, df: pd.DataFrame):
        """
        Apply the denormalized to each column
        
    	Parameters
    	----------            
        df : pd.DataFrame
             dataframe with columns to be denormalized
                    
    	Returns
    	-------
        pd.DataFrame
        """
        if not self.fitted:
            raise Exception("Not yet trained.")
        
        for norm in self.norm_cols:
            if norm == 'zscore':
                for col in self.norm_cols[norm]:
                    df.loc[:,col] = df[col].apply(lambda z: self.col_std[col]*z + self.col_mean[col])
            elif norm == 'log10':
                for col in self.norm_cols[norm]:
                    df.loc[:,col] = df[col].apply(lambda x: 10 ** x)
            else:
                for col in self.norm_cols[norm]:
                    df.loc[:,col] = self.normalization[col].inverse_transform(df[col].values.reshape(-1, 1))
        return df
    
    def fit_transform(self, df: pd.DataFrame):
        """
        Creates object and apply it normalization
        
    	Parameters
    	----------            
        df : pd.DataFrame
             dataframe with columns to be normalized
                    
    	Returns
    	-------
        pd.DataFrame
        """
        self.fit(df)
        return self.transform(df)
