from pyspark.ml.feature import Word2Vec, Tokenizer, HashingTF, IDF, CountVectorizer
from pyspark.ml.pipeline import Pipeline
from ._base import CustomEstimator

class TextVectorizer(CustomEstimator):
    
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
        if method not in methods_dict.keys():
            options = '`' + '`, `'.join(methods_dict.keys()) + '`.'
            raise Exception(f'Method not supported. Choose one from {options}')
        algorithm = methods_dict[method](inputCol=inputCol, outputCol='word_vectors', **kwargs)
        stages = []
        if not tokenized:
            tokenizer = Tokenizer(inputCol=inputCol, outputCol='tokens')
            algorithm = algorithm.setInputCol('tokens')
            stages.append(tokenizer)
        stages.append(algorithm)
        if method in ['hashing_tfidf', 'tfidf']:
            algorithm = algorithm.setOutputCol('unscaled_vectors')
            idf = IDF(inputCol="unscaled_vectors", outputCol="word_vectors")
            stages.append(idf)
        pipeline = Pipeline(stages=stages)
        super().__init__(pipeline)
