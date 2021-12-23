import sys
import os
import argparse
import logging
from sagemaker_inference import model_server

logging.getLogger().setLevel(logging.INFO)

if __name__ == "__main__":
    model_server.start_model_server(handler_service="serving.handler")
