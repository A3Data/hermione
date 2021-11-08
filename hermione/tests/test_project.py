import os
import pytest
import pandas as pd
import sys
sys.path.append('..')

@pytest.fixture(scope='module')
def read_data():
    from ml.data_source.spreadsheet import Spreadsheet
    yield Spreadsheet().get_data('../../data/raw/train.csv')

@pytest.fixture(scope='module')
def cleaned_data(read_data):
    from ml.preprocessing.preprocessing import Preprocessing
    p = Preprocessing()
    yield p.clean_data(read_data)

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

def test_spreadsheet(read_data):
    """
    Test that spreadsheet is importing correctly
    """
    assert read_data.shape[0] > 1


def test_clean_data(cleaned_data):
    """
    Test that the df is cleaned correctly
    """
    assert cleaned_data.Pclass.dtype == 'object'
    assert pd.isnull(cleaned_data.Age).sum() == 0

def test_categ_encoding(cleaned_data):
    """
    Test if column PClass is 
    """
    from ml.preprocessing.preprocessing import Preprocessing
    p = Preprocessing()
    df = p.categ_encoding(cleaned_data)
    names = ['Survived', 'Age', 'Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male']
    assert_categ = [name in df.columns for name in names]
    assert sum(assert_categ) == len(assert_categ)