from pyspark.sql import SparkSession
from ml.data_source.cleaning import SparkCleaner

if __name__ == "__main__":
    spark = (
        SparkSession
        .builder
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .getOrCreate()
    )
    cleaner = SparkCleaner(spark)
    cleaner.clean()
    spark.stop()