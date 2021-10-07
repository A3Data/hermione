from pyspark.ml.feature import Word2Vec, Tokenizer, HashingTF, IDF, CountVectorizer
from pyspark.ml.pipeline import Pipeline
from ._base import CustomEstimator

class SparkVectorizer(CustomEstimator):
    
    def __init__(self, inputCol, method, tokenized = False, **kwargs):
        """
        Constructor
        
    	Parameters
    	----------            
        vectorizer_cols : dict
                       Receives a dict with the name of the vectorizer to be 
                       performed and which are the columns
                       Ex: vectorizer_cols = {'embedding_median': ['col'], 
                                              'embedding_mean': ['col'],
                                              'tf_idf': ['col'],
                                              'bag_of_words' : [col]}
    	Returns
    	-------
        Normalization
        """
        self.inputCol = inputCol
        self.tokenized = tokenized
        methods_dict = {
            'hashing_tfidf': HashingTF, 
            'tfidf': CountVectorizer,
            'word2vec': Word2Vec,
        }
        self.assert_method(methods_dict.keys(), method)
        self.algorithm = methods_dict[method](inputCol=inputCol, outputCol='word_vectors', **kwargs)
        self.estimator_cols = [inputCol]
        self.final_cols = ['tokens', 'word_vectors'] if not tokenized else ['word_vectors']

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
            tokenizer = Tokenizer(inputCol=self.inputCol, outputCol='tokens')
            algorithm = self.algorithm.setInputCol('tokens')
            stages.append(tokenizer)
        stages.append(algorithm)
        if isinstance(self.algorithm, (HashingTF, CountVectorizer)):
            algorithm = algorithm.setOutputCol('unscaled_vectors')
            idf = IDF(inputCol="unscaled_vectors", outputCol="word_vectors")
            stages.append(idf)
        return Pipeline(stages=stages)
