import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def detrend_signal(signal, degree=3):
    x = np.arange(len(signal))
    poly_coeffs = np.polyfit(x, signal, degree)
    trend = np.polyval(poly_coeffs, x)
    detrended_signal = signal - trend
    return detrended_signal

def plot_detrended_signal(original_signal, detrended_signal):
    plt.figure(figsize=(12, 6))
    plt.plot(original_signal, label='Original Signal')
    plt.plot(detrended_signal, label='Detrended Signal')
    plt.legend()
    plt.title('Original and Detrended Signal')
    plt.show()

# Load the data from the CSV file all_wells_cleaned.csv
df_gw_st = pd.read_csv('data/all_well_imputation_cleaned.csv')
print(df_gw_st.head())

# Assuming the signal is in a column named 'signal'

signal = df_gw_st['09200221'].values

# Detrend the signal
detrended_signal = detrend_signal(signal)

# Plot the detrended signal
plot_detrended_signal(signal, detrended_signal)
