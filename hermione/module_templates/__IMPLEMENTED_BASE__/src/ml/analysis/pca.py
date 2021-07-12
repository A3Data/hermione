import pandas as pd
from sklearn.decomposition import PCA as PCA_sklearn
from sklearn import metrics

class PCA:
    
    def __init__(self, columns, prefix="prefix", k=2):
        """
        Constructor
        
    	Parameters
    	----------            
        columns : list
           Columns for dimensionality reduction
        prefix : bool
            column prefix
        k : int
            Number of dimensions
            
    	Returns
    	-------
    	PCA
        """
        self.columns = columns
        self.prefix = prefix
        self.k = k

    
    def __find_k(self, df, threshold):
        """
        Find how many k dimensions will be reduced 
        
    	Parameters
    	----------            
        df : pd.Dataframe
            dataframe to be reduced
            
    	Returns
    	-------
    	int
        """
        self.pca = PCA_sklearn(n_components=len(self.columns))
        self.pca.fit(df[ self.columns ].values)
        for i in range(len(self.columns)-1):
            if self.pca.explained_variance_ratio_[i]+self.pca.explained_variance_ratio_[i+1] < threshold:
                if i == 0:
                    raise Expecption("Not reduced by poor explicability")
                return i+1
    
    def __check(self, df: pd.DataFrame):
        """
        Check dataframe contains all columns

    	Parameters
    	----------            
        df : pd.Dataframe
            dataframe to be reduced
            
    	Returns
    	-------
    	bool
        """
        if not all(col in list(df.columns) for col in self.columns):
            raise Exception('Missing columns') 
        return True


    def transform(self, df: pd.DataFrame):
        """
        Transform the data
        
    	Parameters
    	----------            
        df : pd.Dataframe
            dataframe to be reduced
            
    	Returns
    	-------
    	None
        """
        self.__check(df)
        if self.pca is None:
            raise Exception("Error - object not fitted")
        reduced = self.pca.transform(df[self.columns].values)
        for col in range(self.k):
            df[self.prefix+"_"+str(col)] = [line[col] for line in reduced]
        return df.drop(columns = self.columns.values)


    def fit(self, df : pd.DataFrame, threshold=0.4):
        """
        Compute PCA object

    	Parameters
    	----------            
        df : pd.Dataframe
            dataframe to be reduced
            
    	Returns
    	-------
    	None
        """
        self.__check(df)
        if self.k is None:
            self.k = self.__find_k(df,threshold)
        self.pca = PCA_sklearn(n_components=self.k)
        self.pca.fit(df[ self.columns ].values)
       

    def fit_transform (self, df : pd.DataFrame, threshold=0.4):
        """
        Fit to data, then transform it.

    	Parameters
    	----------            
        df : pd.Dataframe
            dataframe to be reduced
            
    	Returns
    	-------
    	None
        """
        self.__check(df)
        if self.k is None:
            self.k = self.__find_k(df,threshold)
        self.pca = PCA_sklearn(n_components=self.k)
        self.pca.fit(df[ self.columns ].values)
        transformed = self.transform(df)
        self.report()
        return transformed



    
    def report(self):
        """
        Returns explained variance

    	Parameters
    	----------            
        None
            
    	Returns
    	-------
    	None
        """
        for col in range(self.k):
            print("Explained variance ({col}): {ratio}".
                  format(col = self.prefix+"_"+str(col),
                         ratio = str(self.pca.explained_variance_ratio_[col])))
