from pyspark.ml.feature import UnivariateFeatureSelector, VectorAssembler

class SparkFS:
    
    def __init__(
        self, 
        numeric_features, 
        categoric_features, 
        labelCol, labelType, 
        selectionMode='numTopFeatures', 
        selectionThreshold=None
    ):
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
        """
        self.selector = UnivariateFeatureSelector(labelCol=labelCol, selectionMode=selectionMode).setLabelType(labelType)
        if selectionThreshold:
            self.selector = self.selector.setSelectionThreshold(selectionThreshold)
        self.numeric_features = numeric_features
        self.categoric_features = categoric_features
        
    def __numeric_selector(self, df):
        
        selector = (
            self.selector
            .setFeatureType('continuous')
            .setFeaturesCol(self.numeric_features)
            .setOutputCol('selected_continuous')
        )
        return selector.fit(df).transform(df).select(*df.columns, 'selected_continuous')
    
    def __categorical_selector(self, df):
        
        selector = (
            self.selector
            .setFeatureType('categorical')
            .setFeaturesCol(self.categoric_features)
            .setOutputCol('selected_categorical')
        )
        return selector.fit(df).transform(df).select(*df.columns, 'selected_categorical')
        
    def transform(self, df):
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
        df_selected = self.__numeric_selector(df)
        df_selected = self.__categorical_selector(df_selected)
        assembler = VectorAssembler(inputCols=['selected_continuous', 'selected_categorical'], outputCol='features')
        return assembler.transform(df_selected)