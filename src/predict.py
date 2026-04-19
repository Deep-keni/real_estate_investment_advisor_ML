import joblib
import pandas as pd
import numpy as np

classifier = joblib.load('models/classifier.pkl')
regressor = joblib.load('models/regressor.pkl')

def predict_investment(input_df):
    prediction = classifier.predict(input_df)
    probability = classifier.predict_proba(input_df)
    return prediction[0], probability[0]

def predict_price(input_df):
    prediction = regressor.predict(input_df)
    return prediction[0]