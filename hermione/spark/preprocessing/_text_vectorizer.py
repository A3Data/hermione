from pyspark.ml.feature import Word2Vec, Tokenizer, HashingTF, IDF, CountVectorizer
from pyspark.ml.pipeline import Pipeline
from .._base import CustomEstimator
from hermione.core.base import Asserter


class SparkVectorizer(CustomEstimator, Asserter):
    """
    Class used to vectorize text data using Spark ML

    Parameters
    ----------
    inputCol : str
        Column that should be used in the vectorization

    method : str
        The vectorizer algorithm. Supported methods are "tfidf", "hashing_tfidf" and "word2vec".

    tokenized : bool
        Boolean indicating if the text column already is tokenized or not.

    **kwargs : object
        Other arguments passed to the vectorizer algorithm

    Attributes
    ----------
    algorithm: str
        Strategy for completing missing values on numerical columns. Supports "mean", "median" and "mode".

    estimator_cols : list[str]
        List of strings with the columns that are necessary to execute the model. Used to assert if columns are in the DataFrame to be fitted or transformed.

    final_cols : list[str]
        List of strings with the columns that should be appended to the resulting DataFrame.

    inputCol : str
        Column that should be used in the vectorization

    tokenized : bool
        Boolean indicating if the text column already is tokenized or not.

    Examples
    --------
    >>> data = [(1,'Using Spark is really cool'), (2, 'Hermione is such a great tool'), (3, 'Data science is nice'), ]
    >>> df = spark.createDataFrame(data, ['id', "text"])
    >>> vectorizer = SparkVectorizer('text', 'tfidf')
    >>> vectorizer.fit_transform(df).show()
    +---+--------------------+--------------------+--------------------+
    | id|                text|              tokens|        word_vectors|
    +---+--------------------+--------------------+--------------------+
    |  1|Using Spark is re...|[using, spark, is...|(13,[0,2,4,8,9],[...|
    |  2|Hermione is such ...|[hermione, is, su...|(13,[0,1,3,5,7,11...|
    |  3|Data science is nice|[data, science, i...|(13,[0,6,10,12],[...|
    +---+--------------------+--------------------+--------------------+
    """

    def __init__(self, inputCol, method, tokenized=False, **kwargs):

        self.inputCol = inputCol
        self.tokenized = tokenized
        methods_dict = {
            "hashing_tfidf": HashingTF,
            "tfidf": CountVectorizer,
            "word2vec": Word2Vec,
        }
        self.assert_method(methods_dict.keys(), method)
        self.algorithm = methods_dict[method](
            inputCol=inputCol, outputCol="word_vectors", **kwargs
        )
        self.estimator_cols = [inputCol]
        self.final_cols = (
            ["tokens", "word_vectors"] if not tokenized else ["word_vectors"]
        )

    def _fit(self):
        """
        Prepare the estimators

        Parameters
        ----------
        Returns
        -------
        pyspark.ml.pipeline.Pipeline
        """
        stages = []
        if not self.tokenized:
            tokenizer = Tokenizer(inputCol=self.inputCol, outputCol="tokens")
            algorithm = self.algorithm.setInputCol("tokens")
            stages.append(tokenizer)
        stages.append(algorithm)
        if isinstance(self.algorithm, (HashingTF, CountVectorizer)):
            algorithm = algorithm.setOutputCol("unscaled_vectors")
            idf = IDF(inputCol="unscaled_vectors", outputCol="word_vectors")
            stages.append(idf)
        return Pipeline(stages=stages)
