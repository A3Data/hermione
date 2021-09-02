import pandas as pd
import io
from joblib import load
import logging

logging.getLogger().setLevel(logging.INFO)

def generate_data():
    new_data = pd.DataFrame({
        'Pclass':[3,2,1],
        'Sex': ['male', 'female', 'male'],
        'Age':[4, 22, 28]
    })
    return new_data


def load_model():
    try:
        return load('../output/titanic_model_rf.pkl')
    except:
        try: 
            return load('../../output/titanic_model_rf.pkl')
        except:
            logging.error('Model not loaded')


def predict_new(X, probs=True):
    model = load_model()
    p = model.get_preprocessing()
    
    X = p.clean_data(X)
    X = p.categ_encoding(X)
    
    columns = model.get_columns()
    for col in columns:
        if col not in X.columns:
            X[col] = 0
    if probs:
        return model.predict_proba(X)[:,1]
    else:
        return model.predict(X)



if __name__ == "__main__":
    df = generate_data()
    preds = predict_new(df, probs=True)
    logging.info("Predictions:")
    print(preds)
