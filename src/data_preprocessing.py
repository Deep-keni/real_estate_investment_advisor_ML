import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def data_load():
    housing_df = pd.read_csv('data\india_housing_prices.csv')
    print(housing_df.shape)           # checking the shape of the dataset
    print(housing_df.head(5))           # viewing the dataset
    print(housing_df.info())            # we can see from the output that the datatypes are present in proper format 
    print(housing_df.describe())        # basic information about the numerical features
    for col in housing_df.columns:
        print(f"{col}: {housing_df[col].unique()}")          # prints unique values from a column
    return housing_df

def handle_nulls(df):
    print(df.isnull().sum())  # checking for null values in the dataset = 0 (no null values)
    print(df.duplicated().any())  # checking for duplicate values in the dataset = 0
    
# def encode_features():  # handles categorical columns



# def scale_features():   # scales numerical columns

# def engineer_features(): # creates new columns

# def save_cleaned_data(): # saves to processed folder

def run_pipeline():     # calls all above functions in order
    df = data_load()
    handle_nulls(df)

if __name__ == "__main__":
    df = data_load()        # only this runs now
    #df = handle_nulls(df)      
    # encode_features(df)   
