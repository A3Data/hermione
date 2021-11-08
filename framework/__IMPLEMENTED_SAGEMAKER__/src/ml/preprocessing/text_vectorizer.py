from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd

class TextVectorizer:
    
    def __init__(self, vectorizer_cols : dict, word2vec=None):
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
        self.word2vec = word2vec
        self.index_ini_fim = len(self.word2vec.index2word) if word2vec != None else 0
        self.vectorizer_cols = vectorizer_cols
        self.vectorizer_vects = {'bag_of_words': self.bag_of_words,
                                 'tf_idf': self.tf_idf_vect}
        self.fitted = False
    
    def fit(self, df: pd.DataFrame):
        """
        Generates the vectorizer object for each column. The text must be preprocessed.
        
    	Parameters
    	----------            
        df : pd.DataFrame
             dataframe with columns to be vectorizer
                    
    	Returns
    	-------
        None
        """
        self.vectorizers_fitted = dict()
        for vectorizer in self.vectorizer_cols:
            if vectorizer in ['index', 'embedding_median', 'embedding_mean']:
                continue
            for col in self.vectorizer_cols[vectorizer]:
                self.vectorizers_fitted[vectorizer] = {}
                self.vectorizers_fitted[vectorizer][col] =  self.vectorizer_vects[vectorizer](df[col].values)
        self.fitted = True

    def transform(self, df: pd.DataFrame):
        """
        Apply the vectorizer object for each column. The text must be preprocessed.
        
    	Parameters
    	----------            
        df : pd.DataFrame
             dataframe with columns to be vectorizer
                    
    	Returns
    	-------
        pd.DataFrame
        """
        if not self.fitted:
            raise Exception("Not yet trained.")
        
        for vectorizer in self.vectorizer_cols:
            if vectorizer == 'index':
                for col in self.vectorizer_cols[vectorizer]:
                    df.loc[:, col+"_"+vectorizer] = df[col].apply(lambda x: self.embedding(x, 3))
            elif vectorizer == 'embedding_median':
                for col in self.vectorizer_cols[vectorizer]:
                    df.loc[:, col+"_"+vectorizer] = df[col].apply(lambda x: self.embedding(x, 1))
            elif vectorizer == 'embedding_mean':
                for col in self.vectorizer_cols[vectorizer]:
                    df.loc[:, col+"_"+vectorizer] = df[col].apply(lambda x: self.embedding(x, 2))
            elif (vectorizer == 'bag_of_words') | (vectorizer == 'tf_idf'):
                for col in self.vectorizer_cols[vectorizer]:
                    values = self.vectorizers_fitted[vectorizer][col].transform(df[col])
                    df.loc[:,col+"_"+vectorizer] = pd.Series(values.toarray().tolist())

        return df
    
    def embedding(self, X, typ_transform=1):
        """
        Apply the embedding in X. The text must be preprocessed.
        
    	Parameters
    	----------            
        X             : pd.Series
                        row to be encoded
        typ_transform : int
                        type of transformation
                        1 - apply embedding median
                        2 - apply embedding mean
                        2 - apply index
                    
    	Returns
    	-------
        pd.DataFrame
        """
        if X is None or type(X) == float:
            return None
        vector = []
        if typ_transform == 1: # mediana
            vector = np.median([self.word2vec[x] for x in X.split() if x in self.word2vec], axis=0)
        elif typ_transform == 2: # média
            vector = np.mean([self.word2vec[x] for x in X.split() if x in self.word2vec], axis=0)#[0]
        elif typ_transform == 3: # indexação
            idx = self.word2vec.index2word
            set_idx = set(idx)
            indexes = [idx.index(token) for token in X.split() if token in set_idx]
            indexes = [self.index_ini_fim] + indexes + [self.index_ini_fim]
            # Create vector
            X_length = len(indexes)
            vector = np.zeros(X_length, dtype=np.int64)
            vector[:len(indexes)] = indexes
        else:
            vector = []
        return vector

    def bag_of_words(self, corpus):
        """
        Generate object bag of words
        
    	Parameters
    	----------            
        corpus   : str
                   text to generate object bag of words
    	Returns
    	-------
        model
        """
        vectorizer = CountVectorizer()
        model = vectorizer.fit(corpus)
        return model
    
    def tf_idf_vect(self, corpus):
        """
        Generate object td idf
        
    	Parameters
    	----------            
        corpus   : str
                   text to generate object tf idf
    	Returns
    	-------
        model
        """
        vectorizer = TfidfVectorizer()
        model = vectorizer.fit(corpus)
        return model
    
    def inverse_transform(self, df: pd.DataFrame):
        """
        Apply the invese_transform of vectorizer to each column
        Options: index, bag_of_words and tf_idf
        
    	Parameters
    	----------            
        df : pd.DataFrame
             dataframe with columns to be unvectorizer
                    
    	Returns
    	-------
        pd.DataFrame
        """
        if not self.fitted:
            raise Exception("Not yet trained.")
            
        for vectorizer in self.vectorizer_cols:
            if vectorizer == 'index':
                for col in self.vectorizer_cols[vectorizer]:
                    df.loc[:, col+"_remove_"+vectorizer] = df[col].apply(lambda x: self.unvectorize(x))
            elif (vectorizer == 'bag_of_words') | (vectorizer == 'tf_idf'):
                for col in self.vectorizer_cols[vectorizer]:
                    values = self.vectorizers_fitted[vectorizer][col].inverse_transform(df[col])
                    df.loc[:,col+"_remove_"+vectorizer] = pd.Series(values.toarray().tolist())

        return df
    
    def unvectorize(self, vector):
        """
        Apply unvectorize in vector index
        
    	Parameters
    	----------            
        vector : array
                 array with index
                    
    	Returns
    	-------
        array
        """
        idx = self.word2vec.index2word
        tokens = [idx[index] for index in vector if index != self.index_ini_fim]
        X = " ".join(token for token in tokens)
        return X