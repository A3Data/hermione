from pyspark.sql.dataframe import DataFrame


class FlatFile:
    """
    Class for reading files with Spark
    """

    def __init__(self, spark_session) -> None:
        """
        Instantiate class
        
        Parameters
        ----------            
        spark_session  :   pyspark.sql.session.SparkSession
            SparkSession used to read data

        Returns
    	-------
        self:
            returns an instance of the object
        """
        self.spark = spark_session

    def read_data(self, file_path, format, **kwargs) -> DataFrame:
        """
        Read a Spark DataFrame
        
        Parameters
        ----------            
        file_path : str
            File path
        format: str
            name of the file format,  e.g. 'csv', 'parquet'.
        **kwargs: 
            Additional reading options passed to `options()`.
        Returns
        -------
        pyspark.sql.DataFrame
        """
        return self.spark.read.format(format).options(**kwargs).load(file_path)

    def write_data(self, df, save_path, format, mode, partitions=None) -> None:
        """
        Save a Spark DataFrame
        
        Parameters
        ----------            
        save_path : str
            Destination path
        format : str
            name of the destination file format,  e.g. 'csv', 'parquet'.
        mode : str
            specify the mode of writing data, if data already exist in the designed path
            * append: Append the contents of the DataFrame to the existing data
            * overwrite: Overwrite existing data
            * ignore: Silently ignores this operation
            * error or errorifexists (default case): Raises an error
        partitions : int
            number of partitions desired in output files. If greater than `df`'s partitions, `rapartition()` will be used.
            Else, `coalesce()` will be used.

        Returns
    	-------
        self:
            returns an instance of the object
        """
        if partitions:
            df_partitions = df.rdd.getNumPartitions()
            if df_partitions >= partitions:
                df.coalesce(partitions).write.format(format).save(save_path, mode=mode)
            else:
                df.repartition(partitions).write.format(format).save(save_path, mode=mode)
        else:
            df.write.format(format).save(save_path, mode=mode)
