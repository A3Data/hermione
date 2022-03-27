import os
import pytest
import pandas as pd
import sys

sys.path.append("..")


@pytest.fixture(scope="module")
def read_data_train():
    from ml.data_source.spreadsheet import Spreadsheet

    yield Spreadsheet().get_data("../../data/raw/raw_train.csv")


@pytest.fixture(scope="module")
def read_data_test():
    from ml.data_source.spreadsheet import Spreadsheet

    yield Spreadsheet().get_data("../../data/raw/raw_test.csv")


@pytest.fixture(scope="module")
def cleaned_data_train(read_data_train):
    from ml.preprocessing.preprocessing import Preprocessing

    p = Preprocessing()
    yield p.clean_data(read_data_train)


@pytest.fixture(scope="module")
def cleaned_data_test(read_data_test):
    from ml.preprocessing.preprocessing import Preprocessing

    p = Preprocessing()
    yield p.clean_data(read_data_test)


def test_tree():
    """
    Test if the project has a good minimum structure
    """
    assert os.path.exists(os.path.join("..", "..", "data", "raw"))
    assert os.path.exists(os.path.join("..", "..", "output"))
    assert os.path.exists(os.path.join("..", "..", "src", "api"))
    assert os.path.exists(os.path.join("..", "..", "src", "config"))
    assert os.path.exists(os.path.join("..", "..", "src", "ml", "data_source"))
    assert os.path.exists(os.path.join("..", "..", "src", "ml", "model"))
    assert os.path.exists(os.path.join("..", "..", "src", "ml", "notebooks"))
    assert os.path.exists(os.path.join("..", "..", "src", "ml", "preprocessing"))
    assert os.path.exists(os.path.join("..", "..", "src", "tests"))


def test_spreadsheet(read_data_train):
    """
    Test that spreadsheet is importing correctly
    """
    assert read_data_train.shape[0] > 1


def test_clean_data(cleaned_data_train):
    """
    Test that the df is cleaned correctly
    """
    assert cleaned_data_train.Pclass.dtype == "object"
    assert pd.isnull(cleaned_data_train.Age).sum() == 0


def all_columns(df, names):
    """
    Test if df has all columns
    """
    array = [name in df.columns for name in names]
    return sum(array) == len(array)


def values_between(df, col, min_value, max_value):
    """
    Test if column has values between min and max
    """
    array = [value >= min_value and max_value <= 1 for value in df[col]]
    return sum(array) == len(array)


def test_categ_encoding(cleaned_data_train, cleaned_data_test):
    """
    Test if column PClass is encoding
    """
    from ml.preprocessing.preprocessing import Preprocessing

    names = ["Survived", "Age", "Pclass_1", "Pclass_2", "Pclass_3", "Sex_1", "Sex_2"]
    p = Preprocessing(oneHot_cols=["Pclass", "Sex"])
    df_train = p.categ_encoding_oneHot(cleaned_data_train, step_train=True)
    assert all_columns(df_train, names)
    df_test = p.categ_encoding_oneHot(cleaned_data_test, step_train=False)
    assert all_columns(df_test, names)


def test_normalize(cleaned_data_train, cleaned_data_test):
    """
    Test if column Age is normalized
    """
    from ml.preprocessing.preprocessing import Preprocessing

    p = Preprocessing(norm_cols={"min-max": ["Age"]})
    df_train = p.normalize(cleaned_data_train, step_train=True)
    assert values_between(df_train, "Age", 0, 1)
    df_test = p.normalize(cleaned_data_test, step_train=False)
    assert values_between(df_test, "Age", 0, 1)


def test_execute_train(read_data_train, read_data_test):
    """
    Test if execute is correct
    """
    from ml.preprocessing.preprocessing import Preprocessing

    names = ["Survived", "Age", "Pclass_1", "Pclass_2", "Pclass_3", "Sex_1", "Sex_2"]
    norm_cols = {"min-max": ["Age"]}
    oneHot_cols = ["Pclass", "Sex"]
    p = Preprocessing(norm_cols, oneHot_cols)
    X_train, X_val = p.execute(read_data_train, step_train=True)
    assert all_columns(X_train, names)
    assert values_between(X_train, "Age", 0, 1)
    assert all_columns(X_val, names)
    assert values_between(X_val, "Age", 0, 1)
    X_test = p.execute(read_data_test, step_train=False)
    assert all_columns(X_test, names)
    assert values_between(X_test, "Age", 0, 1)
