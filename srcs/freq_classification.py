import numpy as np
import pandas as pd

# Load the data from the CSV file all_wells_cleaned.csv
df_gw_st = pd.read_csv('data/all_well_imputation_cleaned.csv')
# set the column 'date time' as the index
df_gw_st.set_index('date time', inplace=True)
print(df_gw_st.head())
