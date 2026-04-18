import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler , OrdinalEncoder , LabelEncoder

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
    

#def scale_features(df):   # scales numerical columns

# def engineer_features(): # creates new columns

# def save_cleaned_data(): # saves to processed folder

# def run_pipeline():     # calls all above functions in order
#     df = data_load()
#     df = handle_nulls(df)
#     df = encode_features(df)

if __name__ == "__main__":
    df = data_load()        # only this runs now
    df = handle_nulls(df)      
    df = encode_features(df)   
    print(df.sample(5))
    print(df.dtypes)
