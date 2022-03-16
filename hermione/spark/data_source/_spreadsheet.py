from pyspark.sql.dataframe import DataFrame
from hermione.core.base import DataSource


class SparkSpreadsheet(DataSource):
    """
    Class used to read data from flat files

    Parameters
    ----------
    spark_session  :   pyspark.sql.session.SparkSession
        SparkSession used to read data

    Attributes
    ------------
    spark  :   pyspark.sql.session.SparkSession
        SparkSession used to read data

    Examples
    --------
    >>> ss_source = SpaekSpreadsheet(spark_session)
    >>> df = ss_source.get_data('/path/to/file', 'csv', header=True)
    """

    def __init__(self, spark_session) -> None:

        self.spark = spark_session

    def get_data(self, file_path, format, **kwargs) -> DataFrame:
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
        pyspark.sql.dataframe.DataFrame
        """
        return self.spark.read.format(format).options(**kwargs).load(file_path)

    def write_data(
        self,
        df,
        save_path,
        mode,
        format=None,
        partition_col=None,
        n_partitions=None,
        **kwargs
    ) -> None:
        """
        Writes DataFramein the specified destination

        Parameters
        ----------
        df : pyspark.sql.dataframe.DataFrame
            Spark DataFrameto be written

        save_path : str
            Path to where data should be written

        mode : str
            Specify the mode of writing data, if data already exist in the designed path
            * append: Append the contents of the DataFrame to the existing data
            * overwrite: Overwrite existing data
            * ignore: Silently ignores this operation
            * error or errorifexists (default): Raises an error

        format : str
            File format of data being written

        n_partitions : int
            Number of DataFrame partitions

        partition_col : str
                Column to partition DataFrame on writing

        **kwargs:
            Other options passed to DataFrameWriter.options

        Returns
        -------
        """
        if n_partitions:
            df_partitions = df.rdd.getNumPartitions()
            if df_partitions >= n_partitions:
                df = df.coalesce(n_partitions)
            else:
                df = df.repartition(n_partitions)
        writer = df.write.options(**kwargs).mode(mode)
        if partition_col:
            writer = writer.partitionBy(partition_col)
        if format:
            writer = writer.format(format)
        writer.save(save_path)
