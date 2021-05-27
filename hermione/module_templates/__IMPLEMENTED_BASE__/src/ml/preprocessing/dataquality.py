import pandas as pd
import great_expectations as ge

class DataQuality:
    """
    Class to perform data quality before training
    """
    def __init__(self, continuous_cols=None, discrete_cat_cols=None):
        """
        Constructor

        Parameters
        ----------
        continuous_cols       : array
                              Receives an array with the name of the continuous columns 
        discrete_cat_cols     : array
                              Receives an array with the name of the dicrete/categorical columns
        Returns
        -------
        DataQuality
        """
        self.continuous_cols = continuous_cols
        self.discrete_cat_cols = discrete_cat_cols
        
    def perform(self, df: pd.DataFrame, target=None, cut_off = 2):
        """
        Perform data quality

        Parameters
        ----------            
        df  :   pd.Dataframe
                Dataframe to be processed

        Returns
    	-------
        json
        """
        if target != None:
            df.drop(columns=[target], inplace=True)
        df_ge = ge.dataset.PandasDataset(df)
        cols = df_ge.columns
        df_ge.expect_table_columns_to_match_ordered_list(cols)
        for col in cols:
            df_ge.expect_column_values_to_not_be_null(col)
        if self.continuous_cols != None:
            for col in self.continuous_cols:
                measures = df_ge[col].describe() 
                df_ge.expect_column_values_to_be_of_type(col, 'int64')
                df_ge.expect_column_mean_to_be_between(col, measures['mean'] - cut_off * measures['std'], measures['mean'] + cut_off * measures['std'])
                df_ge.expect_column_max_to_be_between(col, measures['max'] - cut_off * measures['std'], measures['max'] + cut_off * measures['std'])
                df_ge.expect_column_min_to_be_between(col, measures['min'] - cut_off * measures['std'], measures['min'] + cut_off * measures['std'])
                expected_partition = ge.dataset.util.continuous_partition_data(df_ge[col])
                df_ge.expect_column_bootstrapped_ks_test_p_value_to_be_greater_than(col, expected_partition)
        if len(self.discrete_cat_cols) != None:
            for col in self.discrete_cat_cols:
                possible_cat = df_ge[col].unique()
                df_ge.expect_column_values_to_be_in_set(col, possible_cat)
                expected_partition = ge.dataset.util.categorical_partition_data(df_ge[col])
                df_ge.expect_column_chisquare_test_p_value_to_be_greater_than(col, expected_partition)         
        return df_ge