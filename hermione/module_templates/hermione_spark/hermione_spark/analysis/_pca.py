from pyspark.ml.feature import PCA
from pyspark.ml.functions import vector_to_array
from .._base import CustomEstimator

class SparkPCA(CustomEstimator):
    
    def __init__(self, inputCol, k=2):
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
        self.k = k
        self.inputCol = inputCol
        self.pca = PCA(inputCol=inputCol, outputCol = "components")
        self.estimator_cols = inputCol

    
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
        n_features = len(df.select(self.inputCol).limit(1).collect()[0][self.inputCol])
        for i in range(n_features):
            pca = self.pca.setK(i + 1)
            ev = sum(pca.fit(df).explainedVariance)
            if ev < threshold:
                continue
            else:
                return i + 1
    
    def transform(self, df):
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
        self.assert_columns(df.columns)
        if isinstance(self.model, PCA):
            raise Exception('Estimator not fitted.')
        df_pred = self.model.transform(df)
        for comp in range(self.model.getK()):
            comp_number = comp + 1
            df_pred = df_pred.withColumn(f'cmp_{comp_number}', vector_to_array('components').getItem(comp))
        return df_pred.drop('components')


    def fit(self, df):
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
        self.assert_columns(df.columns)
        if type(self.k) is float and (self.k >= 0 and self.k <= 1):
            self.k = self.__find_k(df, self.k)
        pca = self.pca.setK(self.k)
        self.model = pca.fit(df)
        self.__report()
       

    def fit_transform (self, df):
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
        self.assert_columns(df.columns)
        if type(self.k) is float and (self.k >= 0 and self.k <= 1):
            self.k = self.__find_k(df, self.k)
        pca = self.pca.setK(self.k)
        self.model = pca.fit(df)
        self.__report()
        df_pred = self.model.transform(df)
        for comp in range(self.model.getK()):
            comp_number = comp + 1
            df_pred = df_pred.withColumn(f'cmp_{comp_number}', vector_to_array('components').getItem(comp))
        return df_pred.drop('components')
    
    def __report(self):
        """
        Returns explained variance

    	Parameters
    	----------            
        None
            
    	Returns
    	-------
    	None
        """
        for col, ratio in zip(range(self.k), self.model.explainedVariance):
            print(f"Explained variance ({col + 1}): {ratio}")