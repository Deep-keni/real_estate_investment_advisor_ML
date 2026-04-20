import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder, StandardScaler , OrdinalEncoder , LabelEncoder

def data_load():
    housing_df = pd.read_csv(r'data\raw_data\india_housing_prices.csv')
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
    return df
    

def encode_features(df1):  # handles categorical columns

    cols_to_convert_directly = ['Parking_Space', 'Security', 'Availability_Status', 'State', 'City']
    le = LabelEncoder()
    for col in cols_to_convert_directly:    # loop because LE cannot handle multiple cols at once
        df1[col] = le.fit_transform(df1[col])

    priority_orders = [ ['Unfurnished', 'Semi-furnished', 'Furnished'], 
    ['Low', 'Medium', 'High'] ]                      
    oe = OrdinalEncoder(categories=[priority_orders[0], priority_orders[1]])
    df1[['Furnished_Status','Public_Transport_Accessibility']] = oe.fit_transform(df1[['Furnished_Status','Public_Transport_Accessibility']])

    unordered_col_convert = ['Property_Type','Facing','Owner_Type']
    ohe = OneHotEncoder(drop='first', sparse_output=False)
    encoded = ohe.fit_transform(df1[unordered_col_convert])
    encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(unordered_col_convert))
    df1 = df1.drop(columns=unordered_col_convert)
    df1 = pd.concat([df1.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)
    return df1
    

def scale_features(df):   # scales numerical columns
    col = ['Size_in_SqFt', 'Age_of_Property', 'Price_per_SqFt']
    scaler = StandardScaler()
    df[col] = scaler.fit_transform(df[col])
    return df
        

def engineer_features(df1): # creates new columns
    backup = df1[['Locality','Year_Built']].copy()
    df1 = df1.drop(columns=['Locality','Year_Built'])

    df1['Amenities_Count'] = df1['Amenities'].apply(lambda x: len(x.split(',')))
    df1.drop('Amenities', axis=1, inplace=True)

    df1['Future_Price_5Y'] = df1['Price_in_Lakhs'] * ((1.08)**5)

    med = df1['Price_per_SqFt'].median()
    def classifier(row):
        if (row['Price_per_SqFt'] <= med and row['Availability_Status']==1) or (row['Availability_Status']==1 and row['BHK']>=2) or (row['Price_per_SqFt'] <= med and row['BHK']>=2 ):
            return 1 
        else:
            return 0
    df1['Good_Investment'] = df1.apply(classifier,axis=1)

    return df1


def save_cleaned_data(df):
    scaled_cols = ['Size_in_SqFt', 'Age_of_Property', 'Price_per_SqFt']
    float_cols = df.select_dtypes(include='float64').columns
    float_cols = [col for col in float_cols if col not in scaled_cols]
    df[float_cols] = df[float_cols].astype(int)
    
    cleaned_df = df.copy()
    os.makedirs('data/processed_data', exist_ok=True)
    cleaned_df.to_csv('data/processed_data/cleaned_data.csv', index=False)
    print("Saved successfully!")


if __name__ == "__main__":
    df = data_load()
    df = handle_nulls(df)
    df = encode_features(df)
    df = scale_features(df)
    df = engineer_features(df)
    save_cleaned_data(df)
