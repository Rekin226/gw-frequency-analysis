import numpy as np
import pandas as pd

# Load the data from the CSV file all_wells_cleaned.csv
df_gw_st = pd.read_csv('data/all_well_imputation_cleaned.csv')
print(df_gw_st.head())

# Assuming the signal is in a column named 'signal'

signal = df_gw_st['09200221'].values

# Detrend the signal
detrended_signal = detrend_signal(signal)

# Plot the detrended signal
plot_detrended_signal(signal, detrended_signal)

# Assuming the signal is in a column named '09200221'
signal = df_gw_st['09200221'].values

# Detrend the signal
detrended_signal = detrend_signal(signal)

# Apply a high-pass filter to df_gw_st['09200221']
cutoff = 0.1  # Cutoff frequency in cycles per day  
fs = 24  # Sampling frequency in cycles per day
filtered_signal = high_pass_filter(signal, cutoff, fs)

# Plot the detrended signal
plot_detrended_signal(signal, filtered_signal)

# Identify the top 5 dominant frequencies
top_freqs, top_powers = identify_top_dominant_frequencies(filtered_signal, top_n=5)
print('The top 5 dominant frequencies are:', top_freqs)
print('Their corresponding power values are:', top_powers)

# 