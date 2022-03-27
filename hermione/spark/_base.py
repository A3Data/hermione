from pyspark.sql.dataframe import DataFrame
from pyspark.ml.base import Estimator
from pyspark.ml.util import MLWriter, MLReader


class CustomEstimator(Estimator, MLWriter, MLReader):
    def __repr__(self):
        return f"{self.__class__}"

    def fit(self, df):
        """
        Fits estimator

        Parameters
        ----------
        df  : pyspark.sql.dataframe.DataFrame
            input Spark DataFrame to be used in fitting

        Returns
        -------
        self
        """
        self.assert_columns(df.columns)
        self.estimator = self._fit().fit(df)
        return self

    def transform(self, df) -> DataFrame:
        """
        Transform the Dataframe based on the previously fitted estimators

        Parameters
        ----------
        df  : pyspark.sql.dataframe.DataFrame
            input Spark DataFrame to be transformed

        Returns
        -------
        pyspark.sql.dataframe.DataFrame
        """
        if hasattr(self, "estimator"):
            df_cols = df.columns
            self.assert_columns(df_cols)
            df = self.estimator.transform(df)
            return df.select(*df_cols, *self.final_cols)
        else:
            raise Exception("Estimator not yet fitted.")

    def fit_transform(self, df) -> DataFrame:
        """
        Fit estimators and transform the Dataframe

        Parameters
        ----------
        df  : pyspark.sql.dataframe.DataFrame
            input Spark DataFrame to be transformed

        Returns
        -------
        pyspark.sql.dataframe.DataFrame
        """
        df_cols = df.columns
        self.assert_columns(df_cols)
        self.fit(df)
        df = self.estimator.transform(df)
        return df.select(*df_cols, *self.final_cols)
