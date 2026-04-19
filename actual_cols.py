import pandas as pd
df = pd.read_csv('data/processed_data/cleaned_data.csv')

# classification columns
xc = df.drop(columns=['Future_Price_5Y', 'Good_Investment', 'Price_per_SqFt'])
print(xc.columns.tolist())

# regression columns  
xr = df.drop(columns=['Future_Price_5Y', 'Good_Investment', 'Price_in_Lakhs'])
print(xr.columns.tolist())