import sys

sys.path.append("..")

import os
import logging
from joblib import load
from six import StringIO
import pandas as pd

from ml.model.wrapper import Wrapper
from sagemaker_inference.default_inference_handler import DefaultInferenceHandler
from sagemaker_inference.default_handler_service import DefaultHandlerService
from sagemaker_inference import content_types, errors, transformer, encoder, decoder

logging.getLogger().setLevel("INFO")

# Path to access the model
MODEL_DIR = "/opt/ml/model"


def _csv_to_pandas(string_like):
    """
    Convert a CSV object to a pandas DataFrame.

    Parameters
    ----------
    string_like : String
                  CSV string.

    Returns
    -------
    pd.DataFrame : pandas DataFrame
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

    def default_model_fn(self, model_dir):
        """
        Loads the model from the disk

        Parameters
        ----------
        model_dir   : string
                      Path of the model

        Returns
        -------
        pkl : model
        """
        logging.info("Loading the model")
        return load(os.path.join(MODEL_DIR, "model.pkl"))

    def default_input_fn(self, input_data, content_type):
        """
        Parse and check the format of the input data

        Parameters
        ----------
        input_data   : string
                       CSV string
        content_type : string
                       Type of the file

        Returns
        -------
        pd.DataFrame : pandas DataFrame
        """
        global colunas
        if content_type != "text/csv":
            raise Exception("Invalid content-type: %s" % content_type)
        return _csv_to_pandas(input_data)

    def default_predict_fn(self, df, model):
        """
        Run our model and do the prediction

        Parameters
        ----------
        df    : pd.DataFrame
                Data to be predicted
        model : pkl
                Model to predict the data

        Returns
        -------
        pd.DataFrame : pandas DataFrame
        """
        logging.info("Predicting...")
        resultados = model.predict(df, included_input=True)
        logging.info("Prediction Complete")
        return resultados.reset_index(drop=True).T.reset_index().T

    def default_output_fn(self, prediction, accept):
        """
        Gets the prediction output and format it to be returned to the user

        Parameters
        ----------
        prediction    : pd.DataFrame
                        Predicted dataset
        accept        : string
                        Output type

        Returns
        -------
        CSV : CSV file
        """
        logging.info("Saving")
        if accept != "text/csv":
            raise Exception("Invalid accept: %s" % accept)
        return encoder.encode(prediction, accept)
