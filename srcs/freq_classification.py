import pandas as pd

# Load the data from the CSV file all_wells_cleaned.csv
df_gw_st = pd.read_csv('data/all_well_imputation_cleaned.csv')
print(df_gw_st.head())
