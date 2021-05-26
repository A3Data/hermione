import sys
sys.path.append("src/")

import os
import traceback
import pandas as pd
import logging
from sklearn.metrics import *
from ml.model.trainer import TrainerSklearn
from sklearn.ensemble import RandomForestClassifier
from util import *

logging.getLogger().setLevel('INFO')

# Paths to access the datasets and salve the model
prefix = '/opt/ml/'

training_path = os.environ["SM_CHANNEL_TRAIN"]
val_path = os.environ["SM_CHANNEL_VALIDATION"]

error_path = os.path.join(prefix, 'output')
model_path = os.environ['SM_MODEL_DIR']

def train():
    """
        Execute the train step in the virtual environment
        
    """
    logging.info('Starting the training')
    try:
        logging.info('Reading the inputs')
        # Take the set of train files and read them all into a single pandas dataframe
        input_files = [ os.path.join(training_path, file) for file in os.listdir(training_path) ]
        if len(input_files) == 0:
            raise ValueError(('There are no files in {}.\n' +
                              'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                              'the data specification in S3 was incorrectly specified or the role specified\n' +
                              'does not have permission to access the data.').format(training_path, channel_name))
        raw_data = [ pd.read_csv(file) for file in input_files ]
        train = pd.concat(raw_data)

        # Take the set of val files and read them all into a single pandas dataframe
        input_files = [ os.path.join(val_path, file) for file in os.listdir(val_path) ]
        if len(input_files) == 0:
            raise ValueError(('There are no files in {}.\n' +
                              'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                              'the data specification in S3 was incorrectly specified or the role specified\n' +
                              'does not have permission to access the data.').format(val_path, channel_name))
        raw_data = [ pd.read_csv(file) for file in input_files ]
        val = pd.concat(raw_data)
        
        # Define the target and columns to be used in the train
        target = "Survived"
        columns = train.columns.drop(target)

        logging.info("Training the model")
        model = TrainerSklearn().train(train, val, target, classification=True, 
                                       algorithm=RandomForestClassifier,
                                       columns=columns)
        
        # Salve the model and metrics
        logging.info("Saving")
        model.save_model(os.path.join(model_path, 'model.pkl'))
        metrics = model.artifacts["metrics"]
        logging.info(f"accuracy={metrics['accuracy']}; f1={metrics['f1']};  precision={metrics['precision']};  recall={metrics['recall']};")
        pd.DataFrame(model.artifacts["metrics"].items(), columns=['Metric', 'Value']).to_csv(os.path.join(model_path, 'metrics.csv'), index=False)
        logging.info('Training complete.')
        
    except Exception as e:
        # Write out an error file. This will be returned as the failureReason in the
        # DescribeTrainingJob result.
        trc = traceback.format_exc()
        with open(os.path.join(error_path, 'failure'), 'w') as s:
            s.write('Exception during training: ' + str(e) + '\n' + trc)
        # Printing this causes the exception to be in the training job logs, as well.
        logging.info('Exception during training: ' + str(e) + '\n' + trc, file=sys.stderr)
        # A non-zero exit code causes the training job to be marked as Failed.
        sys.exit(255)

if __name__ == '__main__':
    train()

    # A zero exit code causes the job to be marked a Succeeded.
    sys.exit(0)
