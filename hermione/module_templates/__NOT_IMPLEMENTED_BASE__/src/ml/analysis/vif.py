import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor 


class AnlyzeVIF:
    
    @classmethod    
    def calculate_vif(cls, df: pd.DataFrame, thresh=5.0, verbose=True):
        """
        Multicollinearity analysis
    
        Parameters
        ----------    
        df      : pd.DataFrame
                  Dataframe must have the following columns (x, y, cluster)
        thresh  : int 
                  value of cut
        verbose : bool
                  if true prints possible variables to be removed
                    
        
        Return
        -------
        pd.DataFrame
        """
        variables = list(range(df.shape[1]))
        dropped = True
        while dropped:
            dropped = False
            vif = [variance_inflation_factor(df.iloc[:, variables].values, ix)
                   for ix in range(df.iloc[:, variables].shape[1])]

            maxloc = vif.index(max(vif))
            if max(vif) > thresh:
                m = max(vif)
                index_max = [i for i, j in enumerate(vif) if j == m]
                if verbose:
                    cols_possibles_remove = [df.iloc[:, variables].columns[i] for i in index_max]
                    print("Columns that can be removed -> " + ", ".join(cols_possibles_remove))
                    print("------")
                print('dropping \'' + df.iloc[:, variables].columns[maxloc] +
                      '\' at index: ' + str(maxloc))
                print("_____________________________________________________________")
                del variables[maxloc]
                dropped = True

        print('Remaining variables:')
        print(df.columns[variables])
        return df.iloc[:, variables]
