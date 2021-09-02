from dateutil.relativedelta import relativedelta
from datetime import datetime
import pandas as pd
import os
import sys
import collections
import copy
import json
import re
from shutil import copyfile
import time
import yaml
import io
import pickle

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

def join_path(*args):
    """Concatenate file paths"""
    return os.path.join(*args).replace("\\", "/")

def excel_to_csv(file_path, save_path):
    """Save an excel file as csv"""
    pd.read_excel(file_path).to_csv(save_path, header=True, index=False)

def get_date_array(date, n_months):
    """
    Get an array sequence of dates starting from `date` and with `n_months` legnth.

    """
    d = datetime.strptime(date, "%Y-%m-%d")
    a = [d.strftime("%Y-%m-%d")]
    for i in range(n_months):
        a.append((d + relativedelta(months=(i + 1))).strftime("%Y-%m-%d"))
    return a

def months_between(date1, date2=None):
    """
    Computes the number of months between two dates. If the second the date is null, uses today.

    """
    if date2:
        delta = relativedelta(
            datetime.strptime(date1, "%Y-%m-%d"), datetime.strptime(date2, "%Y-%m-%d")
        )
    else:
        today = datetime.today().date().replace(day = 1)
        delta = relativedelta(
            today, datetime.strptime(date1, "%Y-%m-%d")
        )
    return delta.months + (12 * delta.years)

def add_months(date, n_months, format="%Y-%m-%d"):
    """
    Adds a number of months equal to `n_months` to `date`

    """
    d = datetime.strptime(date, "%Y-%m-%d")
    return (d + relativedelta(months=(n_months))).strftime(format)


def write_pickle(obj, file_path):
    """Pickle a python object"""
    pickle.dump(obj, open(file_path, "wb"))

