import pyspark.sql.functions as f
import pyspark.sql.types as t
from pyspark.sql.dataframe import DataFrame
from unidecode import unidecode
import numpy as np
import os
import re

def get_spark_versions() -> 'list[str]':
    spark_home = os.environ['SPARK_HOME']
    spark_version = re.search('(?<=spark-).+(?=-bin)', spark_home).group(0)
    hadoop_version = re.search('(?<=hadoop).+', spark_home).group(0)
    return (spark_version, hadoop_version)

spark_version, hadoop_version = get_spark_versions()

if int(spark_version[0]) < 3:
    def transform(self, f) -> DataFrame:
        return f(self)

    DataFrame.transform = transform

# UDFs

@f.udf(returnType=t.StringType())
def unidecode_udf(string):
    if not string:
        return None
    else:
        return unidecode(string)

@f.udf(returnType=t.DoubleType())
def convert_decimal_udf(string):
    if string is None:
        return None
    else:
        string = string.replace(",", ".")
        return float(string.replace(".", "", string.count(".") - 1))

@f.udf(returnType=t.FloatType())
def array_product_udf(array):
    if not array:
        return None
    else:
        array = [e for e in array if e is not None]
    return float(np.prod(array))


@f.udf(returnType=t.FloatType())
def array_sum_udf(array):
    if not array:
        return None
    else:
        array = [e for e in array if e is not None]
    return sum(array)


# Custom methods

def df_from_struct(cols, extract_col, explode) -> DataFrame:
    def _(df):
        if explode:
            df = df.withColumn(extract_col, f.explode(extract_col))
        struct_cols = df.select(f'{extract_col}.*').columns
        renamed_cols = []
        for c in struct_cols:
            col_ref = f.col(f'{extract_col}.' + c)
            if c in cols:
                renamed_cols.append(col_ref.alias(c + '_struct'))
            else:
                renamed_cols.append(col_ref)
        return df.select(*cols, *renamed_cols)
    return _

def renamer(dict) -> DataFrame:
    def _(df):
        for c, n in dict.items():
            df = df.withColumnRenamed(c, n)
        return df
    return _


def unpivot(*args, col_name="categorias", value_name="valor") -> DataFrame:
    if not args[0]:
        key_cols = []
    else:
        key_cols = args[0] if type(args[0]) is list else args

    def _(df):
        unpivot_cols = [c for c in df.columns if c not in key_cols]
        groups_str = [f"'{i}', `{i}`" for i in unpivot_cols]
        unpivot_string = ", ".join(groups_str)
        unpivot_query = "stack({}, {}) as ({}, {})".format(
            len(unpivot_cols), unpivot_string, col_name, value_name
        )
        return df.selectExpr(*key_cols, unpivot_query)

    return _