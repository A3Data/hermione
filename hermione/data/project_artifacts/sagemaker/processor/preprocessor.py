import argparse
import logging
from datetime import date

import pandas as pd
import glob
import json
from joblib import dump, load
import great_expectations as ge

from ml.preprocessing.preprocessing import Preprocessing
from ml.preprocessing.dataquality import DataQuality
from ml.data_source.spreadsheet import Spreadsheet

logging.getLogger().setLevel("INFO")

path_input = "/opt/ml/processing/input/"
path_output = "/opt/ml/processing/output/"
date = date.today().strftime("%Y%m%d")


def data_quality(df, step_train):
    """
    If True, it creates the DataQuality object,
    otherwise it loads an existing one

    Parameters
    ----------
    df          : pd.Dataframe
                  Train or test dataset
    step_train  : boolean
                  Train or test

    """
    if step_train:
        dq = DataQuality(discrete_cat_cols=["Sex", "Pclass", "Survived"])
        df_ge = dq.perform(df)
        df_ge.save_expectation_suite(path_output + "expectations/expectations.json")
    else:
        df_ge = ge.dataset.PandasDataset(df)
        ge_val = df_ge.validate(
            expectation_suite=path_input + "expectations/expectations.json",
            only_return_failures=False,
        )
        with open(f"{path_output}validations/{date}.json", "w") as f:
            json.dump(ge_val.to_json_dict(), f)


def preprocessing(df, step_train):
    """
    If True, it creates the Preprocessing object,
    otherwise it loads an existing one

    Parameters
    ----------
    df          : pd.Dataframe
                  Train or test dataset
    step_train  : boolean
                  Train or test

    """
    if step_train:
        norm_cols = {"min-max": ["Age"]}
        oneHot_cols = ["Pclass", "Sex"]
        p = Preprocessing(norm_cols, oneHot_cols)
        train, test_train = p.execute(df, step_train=True, val_size=0.2)
        logging.info("Saving")
        dump(p, path_output + "preprocessing/preprocessing.pkl")
        train.to_csv(path_output + "processed/train/train.csv", index=False)
        test_train.to_csv(path_output + "processed/val/val.csv", index=False)
    else:
        p = load(path_input + "preprocessing/preprocessing.pkl")
        test = p.execute(df, step_train=False)
        logging.info("Saving")
        test.to_csv(path_output + "processed/inference/inference.csv", index=False)


if __name__ == "__main__":
    """
    Execute the processor step in the virtual environment

    """
    logging.info("Starting the preprocessing")

    # Read the step argument (train or test)
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=str, default="train")
    args = parser.parse_args()
    step_train = True if args.step == "train" else False
    logging.info(f"step_train: {step_train}")

    logging.info("Reading the inputs")
    file = glob.glob(path_input + "raw_data/*.csv")[0]
    logging.info(f"Reading file: {file}")
    df = Spreadsheet().get_data(file)

    logging.info("Data Quality")
    data_quality(df, step_train)

    logging.info("Preprocessing")
    preprocessing(df, step_train)
