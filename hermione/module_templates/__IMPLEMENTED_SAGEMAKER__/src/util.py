import os
import sys
import collections
import copy
import json
import numpy as np
import pandas as pd
import re
from shutil import copyfile
import time
import yaml
import io

def create_dirs(dirpath):
    """Creating directories."""
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

def load_yaml(filepath):
    with open(filepath, 'r') as stream:
        return yaml.safe_load(stream)


def load_json(filepath):
    """Load a json file."""
    with open(filepath, "r", encoding='utf8') as fp:
        obj = json.load(fp)
    return obj


def save_json(obj, filepath):
    """Save a dictionary to a json file."""
    with open(filepath, "w") as fp:
        json.dump(obj, fp, indent=4)

def wrap_text(text):
    """Pretty box print."""
    box_width = len(text) + 2
    print ('\n╒{}╕'.format('═' * box_width))
    print ('│ {} │'.format(text.upper()))
    print ('╘{}╛'.format('═' * box_width))


def load_data(data_csv):
    """Load data from CSV to Pandas DataFrame."""
    df = pd.read_csv(data_csv, header=0)
    wrap_text("Raw data")
    print (df.head(5))
    return df
