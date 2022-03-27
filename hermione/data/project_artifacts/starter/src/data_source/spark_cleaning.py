import pyspark.sql.functions as f
from hermione.spark.data_source import SparkSpreadsheet
from hermione.spark.spark_utils import unidecode_udf, convert_decimal_udf

FILE_PATHS = {"data": "../../../data/raw/train.csv"}


class SparkCleaner:
    """
    Class used to clean perform basic cleaning with Spark

    Parameters
    ----------
    spark_session  :   pyspark.sql.session.SparkSession
        SparkSession used to read data

    Attributes
    ----------
    df : pyspark.sql.dataframe.DataFrame
        Base Spark DataFrame read from raw

    df_cleaned : pyspark.sql.dataframe.DataFrame
        Cleaned Spark DataFrame

    read_options : dict
        Dict with options passed to SparkSession.DataFrameWriter

    save_path : str
        Path to where the final files should be written

    spark  :   pyspark.sql.session.SparkSession
        SparkSession used to read data

    ss_source : SparkSpreadsheet
        Object used to read and write data

    Examples
    --------
    >>> cleaner = SparkCleaner(spark_session)
    >>> cleaner.clean()
    """

    def __init__(self, spark_session) -> None:

        self.spark = spark_session
        self.ss_source = SparkSpreadsheet(spark_session)
        self.read_options = {"header": True}
        self.save_path = "../../../data/refined/train"

    def read_data(self, format) -> None:
        """
        Reads raw data for cleaning.

        Parameters
        ----------
        format : str
            name of the destination file format,  e.g. 'csv', 'parquet'.

        Returns
        -------
        self:
            returns an instance of the object
        """
        self.df = self.ss_source.get_data(
            FILE_PATHS["data"], format, **self.read_options
        )

    def clean_types(self) -> None:
        """
        Cleans DataFrame column types.

        Parameters
        ----------
        Returns
        -------
        self:
            returns an instance of the object
        """
        str_cols = ["Sex", "Name"]
        for c in str_cols:
            self.df = self.df.withColumn(c, unidecode_udf(f.initcap(f.trim(c))))

        dbl_cols = ["Fare"]
        for c in dbl_cols:
            self.df = self.df.withColumn(c, convert_decimal_udf(f.col(c)))

        int_cols = ["PassengerId", "Pclass", "Age", "Survived", "SibSp", "Parch"]
        for c in int_cols:
            self.df = self.df.withColumn(c, f.col(c).cast("int"))

        date_cols = []
        for c in date_cols:
            self.df = self.df.withColumn(c, f.to_date(c, "dd/MM/yyyy"))

    def clean_specific(self) -> None:
        """
        Performs cleaning operations specific to this DataFrame.

        Parameters
        ----------
        Returns
        -------
        self:
            returns an instance of the object
        """
        predicate = """
            CASE WHEN Embarked = 'S' THEN 'Southampton'
                 WHEN Embarked = 'Q' THEN 'Queenstown'
                 WHEN Embarked = 'C' THEN 'Cherbourg'
                 ELSE 'Unknown'
            END
        """
        self.df_cleaned = self.df.withColumn("Embarked", f.expr(predicate)).withColumn(
            "total_relatives", f.col("SibSp") + f.col("Parch")
        )

    def write_data(self, format, mode) -> None:
        """
        Saves intermediate DataFrames generated in this process.

        Parameters
        ----------
        format : str
            name of the destination file format,  e.g. 'csv', 'parquet'.
        mode : str
            specify the mode of writing data, if data already exist in the designed path
            * append: Append the contents of the DataFrame to the existing data
            * overwrite: Overwrite existing data
            * ignore: Silently ignores this operation
            * error or errorifexists (default case): Raises an error

        Returns
        -------
        self:
            returns an instance of the object
        """
        self.ss_source.write_data(self.df_cleaned, self.save_path, mode, format)

    def clean(self, mode="error") -> None:
        """
        Wrapper for running cleaner.

        Parameters
        ----------
        mode : str
            specify the mode of writing data, if data already exist in the designed path
            * append: Append the contents of the DataFrame to the existing data
            * overwrite: Overwrite existing data
            * ignore: Silently ignores this operation
            * error or errorifexists (default case): Raises an error

        Returns
        -------
        self:
            returns an instance of the object
        """
        self.read_data("csv")
        self.clean_types()
        self.clean_specific()
        self.write_data("parquet", mode)
