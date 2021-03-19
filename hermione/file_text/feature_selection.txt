from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SequentialFeatureSelector
from mlxtend.feature_selection import ExhaustiveFeatureSelector
from abc import ABC, abstractmethod
import numpy as np
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

class SelectCoefficients(SelectAlgorithm):
    """
        Class to select features based on model coefficients
    """
    def __init__(self, model, num_feat = None):
        """
            Constructor

            Parameters
    	    ----------            
            model     :    
                        should be an instance of a classification or regression model class from scikit-learn and have coef_.ravel method

            num_feats : int
                        number of features to be selected
    	    Returns
    	    -------
            SelectCoefficients
        """
        self.model = model
        self.num_feat = num_feat

    def fit(self, X: pd.DataFrame, y = None):
        """
        Identify the features to be selected.
        
    	Parameters
    	----------            
        X : pd.DataFrame
             features to be selected

        y : pd.DataFrame
            target values
                    
    	Returns
    	-------
        None
        """
        self.num_feat = int(X.shape[1]/2) if self.num_feat == None else self.num_feat
        trained_model = self.model.fit(X,y)
        self.selected_columns = np.argsort(np.abs(trained_model.coef_.ravel()))[-self.num_feat:]

class SelectCorrelation(SelectAlgorithm):
    """
        Class to select features based on correlation between features
    """
    def __init__(self, threshold = 1.0):
        """
        Constructor

        Parameters
    	----------            
        threshold     : float   
                        correlation threshold
    	Returns
    	-------
        SelectCorrelation
        """
        self.threshold = threshold
    def fit(self, X: pd.DataFrame, y = None):
        """
        Identify the features to be selected.
        
    	Parameters
    	----------            
        X : pd.DataFrame
             features to be selected

        y : pd.DataFrame
            target values
                    
    	Returns
    	-------
        None
        """
        corr = X.corr()
        self.selected_columns = np.full((corr.shape[0],), True, dtype=bool)
        [self.check_correlation(corr.iloc[i,j],j) for i in range(corr.shape[0]) for j in range(i+1, corr.shape[0])]
        
    def check_correlation(self,corr,j):
        """
        Auxiliar method to check if correlation between features is above threshold
        Parameters
      	----------            
          corr : float
               correlation between two atributes

          j : int
              index of column to be removed in case corr >= self.threshold
                      
      	Returns
      	-------
        None
        """
        if np.abs(corr) >= self.threshold and self.selected_columns[j]:
            self.selected_columns[j] = False

class MyExhaustiveFeatureSelector(ExhaustiveFeatureSelector):
    """
        Class that inherits from ExhaustiveFeatureSelector (from mlxtend) and implements get_support method for
        compatibility issues
    """
    def get_support(self):
        return list(self.best_idx_)

class SelectEnsemble(SelectAlgorithm):
    """
        Class to select features based on ensemble of methods
    """
    def __init__(self, dic_selection: dict, num_feat = None):
        """
            Constructor

            Parameters
    	    ----------            
            dic_selection : dict
                        dict with name of the algorithm as keys and dicts of parameters as values 
                        Ex: dic_selection = { 'variance': {'threshold' : 0.3},
                                              'recursive': {'estimator' : LinearSVC(), 'n_features_to_select' : 2}}
            num_feats : int
                        number of features to be selected
    	    Returns
    	    -------
            SelectCoefficients
        """
        self.dic_selection = dic_selection
        self.num_feat = num_feat

    def fit(self, X: pd.DataFrame, y = None):
          """
          Identify the features to be selected.
          
      	Parameters
      	----------            
          X : pd.DataFrame
               features to be selected

          y : pd.DataFrame
              target values
                      
      	Returns
      	-------
          None
          """
          self.num_feat = int(X.shape[1]/2) if self.num_feat == None else self.num_feat
          self.column_dic = {}
          for i,column in enumerate(X.columns):
              self.column_dic[column] = i
          self.column_count = [0 for column in X.columns]
          selections = [FeatureSelector(selector,**self.dic_selection[selector]) for selector in self.dic_selection]
          [selection.fit(X,y) for selection in selections]
          [self.increment_count(column) for selection in selections for column in selection.selected_columns]
          self.selected_columns = np.argsort(self.column_count)[-self.num_feat:]
    
    def increment_count(self,column):
        """
        Auxiliar method to increment the count of a column
        Parameters
      	----------            
          column : int
               column which the count will be incremented 
                      
      	Returns
      	-------
        None
        """
        self.column_count[self.column_dic[column]]+=1

class FeatureSelector:
    
    def __init__(self, selector, **kwargs):
        """
        Constructor
        
    	Parameters
    	----------            
        selector : str
                   name of algorithm to be applied
        **kwargs : 
                   optional and positional arguments of the choosen algorithm (selector)
    	Returns
    	-------
        FeatureSelector
      Examples
      ---------
      variance thresholding:      f = FeatureSelector('variance', threshold=0.3) #Instantiating 
                                  f.fit(X[,y]) #fitting (y is optional for variance thresholding)
                                  X = f.transform(X) #transforming

      filter-based, k best (MAD): f = FeatureSelector('univariate_kbest', score_func=FeatureSelector.mean_abs_diff, k=2) #Instantiating 
                                  #score_func can be any function f: R^n -> R^n (n = number of columns)
                                  f.fit(X,y) #fitting 
                                  X = f.transform(X) #transforming

      wrapper, recursive:         f = FeatureSelector('recursive', estimator = LinearSVC(), n_features_to_select=2) #Instantiating 
                                  #estimator should be an instance of a classification or regression model class from scikit-learn 
                                  #one can use a custom class but it must be compatible with scikit-learn arquitecture  
                                  f.fit(X,y) #fitting 
                                  X = f.transform(X) #transforming

     wrapper, sequential:          f = FeatureSelector('sequential', estimator = LinearSVC(), direction='forward') #Instantiating 
                                  #estimator should be an instance of a classification or regression model class from scikit-learn 
                                  #one can use a custom class but it must be compatible with scikit-learn arquitecture  
                                  f.fit(X,y) #fitting 
                                  X = f.transform(X) #transforming
   
      to better understand the optional arguments of each algorithm see: https://scikit-learn.org/stable/modules/feature_selection.html                         
        """
        self.selector = selector
        self.selectors = {'variance': VarianceThreshold, 
                          'univariate_kbest': SelectKBest,
                          'univariate_percentile': SelectPercentile,
                          'recursive': RFE,
                          'model':SelectFromModel,
                          'sequential':SequentialFeatureSelector,
                          'exaustive':MyExhaustiveFeatureSelector,
                          'correlation':SelectCorrelation,
                          'coefficients':SelectCoefficients,
                          'ensemble':SelectEnsemble}
        self.kwargs = kwargs 
        self.fitted = False
    
    def fit(self, X: pd.DataFrame, y = None):
        """
        Identify the features to be selected.
        
    	Parameters
    	----------            
        X : pd.DataFrame
             features to be selected

        y : pd.DataFrame
            target values
                    
    	Returns
    	-------
        None
        """
        self.columns = X.columns
        self.selection = self.selectors[self.selector](**self.kwargs)
        self.selection.fit(X,y)
        self.selected_columns = self.columns[self.selection.get_support()]
        self.fitted = True

    def transform(self, df: pd.DataFrame):
        """
        Select features based on fit
        
    	Parameters
    	----------            
        pd.DataFrame
        dataframe with features to be selected
                    
    	Returns
    	-------
        df : pd.DataFrame
             dataframe with selected features only
        """
        if not self.fitted:
            raise Exception("Not yet trained.")

        
        #return self.selection.transform(df)
        return df[self.selected_columns]
    
    def inverse_transform(self, df: pd.DataFrame):
        """
        Apply the invese_transform of vectorizer to each column
        Options: index, bag_of_words and tf_idf
        
    	Parameters
    	----------            
        df : pd.DataFrame
             dataframe with columns to be unvectorizer
                    
    	Returns
    	-------
        pd.DataFrame
        """
        pass

        #return df

    @staticmethod
    def mean_abs_diff(X, y=None):
        """
        method to compute the mean absolute difference (MAD) of all atributes of X
        
    	Parameters
    	----------            
        X : pd.DataFrame
             dataframe 
        y: any type
            not necessary, used only for compatibility issues
                    
    	Returns
    	-------
        pd.DataFrame
        """
        return np.sum(np.abs(X - np.mean(X, axis = 0)), axis = 0)/X.shape[0]

    @staticmethod
    def variance(X, y=None):
        """
        method to compute the mean variance of all atributes of X
        
    	Parameters
    	----------            
        X : pd.DataFrame
             dataframe 
        y: any type
            not necessary, used only for compatibility issues
                    
    	Returns
    	-------
        pd.DataFrame
        """
        return np.sum((X - np.mean(X, axis = 0)**2), axis = 0)/X.shape[0]
  
    @staticmethod
    def disp_ratio(X, y=None):
        """
        method to compute the dispersion ratio of all atributes od X
        
    	Parameters
    	----------            
        X : pd.DataFrame
             dataframe
        y: any type
            not necessary, used only for compatibility issues
                    
    	Returns
    	-------
        pd.DataFrame
        """
        return np.mean(X, axis = 0)/np.power(np.prod(X, axis = 0),1/X.shape[0])
