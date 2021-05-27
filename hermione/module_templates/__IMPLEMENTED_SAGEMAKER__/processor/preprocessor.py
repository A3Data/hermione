from ml.preprocessing.preprocessing import Preprocessing
from ml.preprocessing.dataquality import DataQuality
from ml.data_source.spreadsheet import Spreadsheet
import great_expectations as ge
from datetime import date
import pandas as pd
import argparse
import logging
import glob
import json
from joblib import dump, load

logging.getLogger().setLevel('INFO')

if __name__=='__main__':
    """
        Execute the processor step in the virtual environment
        
    """
    logging.info('Starting the preprocessing')
    
    # Read the step argument (train or test)
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=str, default='train')
    args = parser.parse_args()    
    step_train = True if args.step == "train" else False
    logging.info(f'step_train: {step_train}')
    
    logging.info('Reading the inputs')
    file = glob.glob("/opt/ml/processing/input/raw_data/*.csv")[0]
    logging.info(f'Reading file: {file}')
    df = Spreadsheet().get_data(file)
    
    
    logging.info("Data Quality")
    # If True, it creates the DataQuality object, otherwise it loads an existing one
    if step_train:
        dq = DataQuality(discrete_cat_cols=['Sex', 'Pclass','Survived'])
        df_ge = dq.perform(df)
        df_ge.save_expectation_suite('/opt/ml/processing/output/expectations/expectations.json')
    else:
        date = date.today().strftime('%Y%m%d')
        df_ge = ge.dataset.PandasDataset(df)
        ge_val = df_ge.validate(expectation_suite='/opt/ml/processing/input/expectations/expectations.json', only_return_failures=False)
        with open(f'/opt/ml/processing/output/validations/{date}.json', 'w') as f: 
            json.dump(ge_val.to_json_dict(), f)

    logging.info("Preprocessing")
    # If True, it creates the Preprocessing object, otherwise it loads an existing one
    if step_train:
        norm_cols = {'min-max': ['Age']}
        oneHot_cols = ['Pclass','Sex']
        p = Preprocessing(norm_cols, oneHot_cols)
        train, test_train = p.execute(df, step_train = True, val_size = 0.2)
    else:
        p = load("/opt/ml/processing/input/preprocessing/preprocessing.pkl")
        test = p.execute(df, step_train = False)

    logging.info("Saving")
    # If True, it saves the Preprocessing to be used later in the inference step
    if step_train:
        dump(p, '/opt/ml/processing/output/preprocessing/preprocessing.pkl')
        
    # If True, it saves the train and val files, otherwise it saves only the inference file    
    if step_train:
        train.to_csv('/opt/ml/processing/output/processed/train/train.csv', index=False)
        test_train.to_csv('/opt/ml/processing/output/processed/val/val.csv', index=False)
    else:
        test.to_csv('/opt/ml/processing/output/processed/inference/inference.csv', index=False)