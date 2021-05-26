import sys
sys.path.append("..")

import os
import logging
import pandas as pd
from joblib import load
from six import StringIO

from ml.model.wrapper import Wrapper
from sagemaker_inference.default_inference_handler import DefaultInferenceHandler
from sagemaker_inference.default_handler_service import DefaultHandlerService
from sagemaker_inference import content_types, errors, transformer, encoder, decoder

logging.getLogger().setLevel('INFO')

# Path to access the model
MODEL_DIR = '/opt/ml/model'

def _csv_to_pandas(string_like):  # type: (str) -> pd.DataFrame
    """Convert a CSV object to a pandas DataFrame.
    Args:
        string_like (str): CSV string.
        
    Returns:
        (pd.DataFrame): pandas DataFrame
    """    
    stream = StringIO(string_like)
    res = pd.read_csv(stream)
    return res

class HandlerService(DefaultHandlerService, DefaultInferenceHandler):
    """
        Execute the inference step in the virtual environment
        
    """
    def __init__(self):
        op = transformer.Transformer(default_inference_handler=self)
        super(HandlerService, self).__init__(transformer=op)
    
    # Loads the model from the disk
    def default_model_fn(self, model_dir):
        logging.info('Loading the model')   
        return load(os.path.join(MODEL_DIR, "model.pkl"))
    
    # Parse and check the format of the input data
    def default_input_fn(self, input_data, content_type):
        global colunas
        if content_type != "text/csv":
            raise Exception("Invalid content-type: %s" % content_type)
        return _csv_to_pandas(input_data)                           
    
    # Run our model and do the prediction
    def default_predict_fn(self, df, model):
        logging.info('Predicting...')        
        resultados = model.predict(df,included_input=True)
        logging.info('Prediction Complete')     
        return resultados.reset_index(drop=True).T.reset_index().T
    
    # Gets the prediction output and format it to be returned to the user
    def default_output_fn(self, prediction, accept):
        logging.info('Saving') 
        if accept != "text/csv":
            raise Exception("Invalid accept: %s" % accept)
        return encoder.encode(prediction, accept)