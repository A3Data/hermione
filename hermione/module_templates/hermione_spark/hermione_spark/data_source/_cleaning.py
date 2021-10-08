import pyspark.sql.functions as f
from ._spreadsheet import SparkSpreadsheet
from ..utils import unidecode_udf, convert_decimal_udf

FILE_PATHS = {
    'data': '../../../data/raw/raw_train.csv'
}

class SparkCleaner:
    """ 
    Class used to clean perform basic cleaning with Spark
    """

    def __init__(self, spark_session) -> None:
        """
        Instantiates class
        
        Parameters
        ----------     
        spark_session  :   pyspark.sql.session.SparkSession
            SparkSession that is used to manipulate data.
        
        Returns
    	-------
        self:
            returns an instance of the object
        """
        self.spark = spark_session
        self.ss_source = SparkSpreadsheet(spark_session)
        self.read_options = {
            'header': True
        }
        self.save_path = 'teste'

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
        self.df = self.ss_source.get_data(FILE_PATHS['data'], format, **self.read_options)

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
        str_cols = ['Sex']
        for c in str_cols:
            self.df = self.df.withColumn(c, unidecode_udf(f.initcap(f.trim(c))))

        dbl_cols = []
        for c in dbl_cols:
            self.df = self.df.withColumn(c, convert_decimal_udf(f.col(c)))

        int_cols = ['Pclass', 'Age', 'Survived']
        for c in int_cols:
            self.df = self.df.withColumn(c, f.col(c).cast('int'))

        date_cols = []
        for c in date_cols:
            self.df = self.df.withColumn(c, f.to_date(c, 'dd/MM/yyyy'))

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
        self.df_cleaned = self.df

    def write_data(self, format, mode='error') -> None:
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
    
    def clean(self) -> None:
        """
        Wrapper for running cleaner.
        
        Parameters
        ----------     
        Returns
    	-------
        self:
            returns an instance of the object
        """
        self.read_data('csv')
        self.clean_types()
        self.clean_specific()
        self.write_data('parquet')