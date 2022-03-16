import sys

sys.path.append("src/")

import os
from util import *
import traceback

import logging
import pandas as pd

from sklearn.metrics import *
from ml.model.trainer import TrainerSklearn
from sklearn.ensemble import RandomForestClassifier

logging.getLogger().setLevel("INFO")

# Paths to access the datasets and salve the model
prefix = "/opt/ml/"

training_path = os.environ["SM_CHANNEL_TRAIN"]
val_path = os.environ["SM_CHANNEL_VALIDATION"]

error_path = os.path.join(prefix, "output")
model_path = os.environ["SM_MODEL_DIR"]


def read_input(file_path):
    """
    Take the set of train files and read them all
    into a single pandas dataframe

    Parameters
    ----------
    file_path   : string
                  Path of the file

    Returns
    -------
    pd.Dataframe : pandas DataFrame
    """
    input_files = [os.path.join(file_path, file) for file in os.listdir(file_path)]
    if len(input_files) == 0:
        raise ValueError(
            (
                "There are no files in {}.\n"
                + "This usually indicates that the channel ({}) was \
                          incorrectly specified,\n"
                + "the data specification in S3 was incorrectly \
                          specified or the role specified\n"
                + "does not have permission to access \
                          the data."
            ).format(file_path, channel_name)
        )
    raw_data = [pd.read_csv(file) for file in input_files]
    return pd.concat(raw_data)


def train():
    """
    Execute the train step in the virtual environment

    """
    logging.info("Starting the training")
    try:
        logging.info("Reading the inputs")
        train = read_input(training_path)
        val = read_input(val_path)

        # Define the target and columns to be used in the train
        target = "Survived"
        columns = train.columns.drop(target)

        logging.info("Training the model")
        model = TrainerSklearn().train(
            train,
            val,
            target,
            classification=True,
            algorithm=RandomForestClassifier,
            columns=columns,
        )

        # Salve the model and metrics
        logging.info("Saving")
        model.save_model(os.path.join(model_path, "model.pkl"))
        metrics = model.artifacts["metrics"]
        logging.info(
            f"accuracy={metrics['accuracy']}; \
                     f1={metrics['f1']}; \
                     precision={metrics['precision']}; \
                     recall={metrics['recall']};"
        )
        pd.DataFrame(
            model.artifacts["metrics"].items(), columns=["Metric", "Value"]
        ).to_csv(os.path.join(model_path, "metrics.csv"), index=False)
        logging.info("Training complete.")

    except Exception as e:
        # Write out an error file
        trc = traceback.format_exc()
        with open(os.path.join(error_path, "failure"), "w") as s:
            s.write("Exception during training: " + str(e) + "\n" + trc)
        logging.info(
            "Exception during training: " + str(e) + "\n" + trc, file=sys.stderr
        )
        # A non-zero exit code causes the training job to be marked as Failed
        sys.exit(255)


if __name__ == "__main__":
    train()
    # A zero exit code causes the job to be marked a Succeeded.
    sys.exit(0)
