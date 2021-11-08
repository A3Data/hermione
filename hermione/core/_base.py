from abc import ABC, abstractmethod
import pandas as pd

class SelectAlgorithm(ABC):
    """
        Abstract class for feature selection algorithms
    """
    def transform(self, df: pd.DataFrame):
        """
        Select features based on fit
        
    	Parameters
    	----------            
        df : pd.DataFrame
             dataframe with features to be selected
                    
    	Returns
    	-------
        pd.DataFrame
        dataframe with selected features only
        """
        return df[df.columns[self.selected_columns]]

    def get_support(self):
        """
        Get a mask, or integer index, of the features selected
        
    	Parameters
    	----------            
                    
    	Returns
    	-------
        np.array     
        """
        return self.selected_columns

    @abstractmethod
    def fit(self) -> None:
        """
        Abstract method that is implemented in classes that inherit it
        """
        pass