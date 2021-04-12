import os
import pytest
import pandas as pd
import sys
sys.path.append('..')

def test_tree():
    """
    Test if the project has a good minimum structure
    """
    assert os.path.exists(os.path.join('..','..', 'data', 'raw'))
    assert os.path.exists(os.path.join('..','..', 'output'))
    assert os.path.exists(os.path.join('..','..', 'src', 'api'))
    assert os.path.exists(os.path.join('..','..', 'src', 'config'))
    assert os.path.exists(os.path.join('..','..', 'src', 'ml', 'data_source'))
    assert os.path.exists(os.path.join('..','..', 'src', 'ml', 'model'))
    assert os.path.exists(os.path.join('..','..', 'src', 'ml', 'notebooks'))
    assert os.path.exists(os.path.join('..','..', 'src', 'ml', 'preprocessing'))
    assert os.path.exists(os.path.join('..','..', 'src', 'tests'))
