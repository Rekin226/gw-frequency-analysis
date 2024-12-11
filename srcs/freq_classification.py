import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq
from scipy.signal import butter, filtfilt
import sys

# Create a high-pass filter function
def high_pass_filter(data, cutoff, fs, order=5):
    """
    Apply a high-pass filter to the data.

    Parameters:
    data (array-like): The input signal.
    cutoff (float): The cutoff frequency in cycles per day.
    fs (float): The sampling frequency in cycles per day.
    order (int): The order of the filter.

    Returns:
    array-like: The filtered signal.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def detrend_signal(signal, degree=1):
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

def identify_top_dominant_frequencies(signal, top_n=5):
    # Perform the FFT
    n = len(signal)
    T = 1.0  # Sampling interval (1 hour)

    fft_values = fft(signal)
    fft_values = 2.0 / n * np.abs(fft_values[:n // 2])
    freqs = fftfreq(n, T)[:n // 2]

    # Convert from cph to cpd
    freqs = freqs * 24

    # Find the power spectrum
    power_spectrum = np.abs(fft_values) ** 2

    # Identify the top N dominant frequencies
    top_indices = np.argsort(power_spectrum)[-top_n:][::-1]
    top_freqs = freqs[top_indices]
    top_powers = power_spectrum[top_indices]
    top_amplitudes = fft_values[top_indices]

    return top_freqs, top_powers, top_amplitudes


def find_dominant_frequency_in_intervals(freqs, power_spectrum, intervals):
    dominant_frequencies = []
    dominant_amplitudes = []
    for interval in intervals:
        mask = (freqs >= interval[0]) & (freqs < interval[1])
        interval_freqs = freqs[mask]
        interval_powers = power_spectrum[mask]
        if len(interval_powers) > 0:
            peak_idx = np.argmax(interval_powers)
            peak_freq = interval_freqs[peak_idx]
            peak_amplitude = np.sqrt(interval_powers[peak_idx])
            dominant_frequencies.append(peak_freq)
            dominant_amplitudes.append(peak_amplitude)
        else:
            dominant_frequencies.append(None)
            dominant_amplitudes.append(None)
    return dominant_frequencies, dominant_amplitudes


# Load the data from the CSV file all_wells_cleaned.csv
df_gw_st = pd.read_csv('data/all_well_imputation_cleaned.csv')
# set the column 'date time' as the index
df_gw_st.set_index('date time', inplace=True)
# drop the first 0 from the station values 
df_gw_st.columns = df_gw_st.columns.str.lstrip('0')
# print the first 5 rows
print(df_gw_st.head())

# Apply the analysis to the first 5 stations in the dataframe
results = []
for station in df_gw_st.columns:  # Skip the first column which is 'date time'
    #print(f'Processing station: {station}')
    
    # Extract the signal
    signal = df_gw_st[station].values

    # Apply a high-pass filter to df_gw_st['09200221']
    cutoff = 0.5  # Cutoff frequency in cycles per day  
    fs = 24  # Sampling frequency in cycles per day
    filtered_signal = high_pass_filter(signal, cutoff, fs)

    # Identify the top 5 dominant frequencies
    top_freqs, top_powers, top_amplitudes = identify_top_dominant_frequencies(filtered_signal, top_n=4)
    #print(f'The top 5 dominant frequencies for {station} are:', top_freqs)
    #print('Their corresponding power values are:', top_powers)

    # create a dataframe to store the dominant frequencies including the station name
    # station is the column name top_freqs is the column value
    df_dominant_freq = pd.DataFrame({station: top_freqs})
    #print(f'Top 4 dominant frequencies for each {station}:', df_dominant_freq)

    # Define the intervals
    intervals = [(0.5, 1.5), (1.5, 2.5), (2.5, 3.5), (3.5, 4.5)]

    # Perform the FFT
    n = len(filtered_signal)
    T = 1.0  # Sampling interval (1 hour)
    fft_values = fft(filtered_signal)
    fft_values = 2.0 / n * np.abs(fft_values[:n // 2])
    freqs = fftfreq(n, T)[:n // 2]

    # Convert from cph to cpd
    freqs = freqs * 24

    # Find the power spectrum
    power_spectrum = np.abs(fft_values) ** 2

    # Find the dominant frequency in each interval
    dominant_frequencies, dominant_amplitudes = find_dominant_frequency_in_intervals(freqs, power_spectrum, intervals)
    
    # Find the amplitude of each dominant frequency
    dominant_amplitudes = []
    for freq in dominant_frequencies:
        if freq is not None:
            idx = np.where(freqs == freq)[0][0]
            dominant_amplitudes.append(fft_values[idx])
        else:
            dominant_amplitudes.append(None)
    
    #print(f'The dominant frequencies for {station}') 

    # Classify the station
    tolerance = 0.003
    if any(abs(freq - 1.93) < tolerance for freq in dominant_frequencies if freq is not None) or \
        any(abs(freq - 1.93) < tolerance for freq in top_freqs if freq is not None):
        classification = 'tides'
    else:
        classification = 'pumping'

    # Store the results
    results.append({
        'Station': station,
        'Top 4 Dominant Frequencies': top_freqs,
        'Top 4 Dominant Powers': top_powers,
        'Top 4 Dominant Amplitudes': top_amplitudes,
        'Dominant Frequencies in Intervals': dominant_frequencies,
        'Dominant Amplitudes in Intervals': dominant_amplitudes,
        'Classification': classification
    })


# Convert results to DataFrame
df_results = pd.DataFrame(results)
print(df_results.head())

# create a dataframe for each station
for station in df_results['Station']:
    print(f'Processing station: {station}')
    # create dataframe 
    df_station = pd.DataFrame(
        {
            'Top 4 Dominant Frequencies': df_results[df_results['Station'] == station]['Top 4 Dominant Frequencies'].values[0],
            'Top 4 Dominant amplitudes': df_results[df_results['Station'] == station]['Top 4 Dominant Amplitudes'].values[0],
            'Dominant Frequencies in Intervals': df_results[df_results['Station'] == station]['Dominant Frequencies in Intervals'].values[0],
            'Dominant Amplitudes in Intervals': df_results[df_results['Station'] == station]['Dominant Amplitudes in Intervals'].values[0],
        }
    
    )
    print(df_station)
    sys.exit() 


# add new colum 'amplitudes' to the dataframe and store
# find the corresponding amplitudes for the top 5 dominant frequencies and the dominant frequencies in intervals from 
# the dataframe df_results
amplitudes = []
for idx, row in df_results.iterrows():
    top_amplitudes = row['Top 4 Dominant Amplitudes']
    interval_amplitudes = row['Dominant Amplitudes in Intervals']
    if any(abs(freq - 1.93) < tolerance for freq in row['Top 4 Dominant Frequencies'] if freq is not None):
        idx = np.where(row['Top 4 Dominant Frequencies'] == 1.93)[0][0]
        amplitudes.append(top_amplitudes[idx])
    else:
        idx = np.where(row['Dominant Frequencies in Intervals'] == 1.93)[0][0]
        amplitudes.append(interval_amplitudes[idx])
df_results['Amplitudes'] = amplitudes
print(df_results)
#sys.exit()

# load the data from the CSV file df_input.csv
df_input = pd.read_csv('data/df_input.csv')
# change column name
df_input = df_input.rename(columns={'ST_NO': 'Station'})
#print(df_input.head())

# Ensure the 'Station' column in both DataFrames is of the same type
df_results['Station'] = df_results['Station'].astype(str)
df_input['Station'] = df_input['Station'].astype(str)

# Merge df_results with df_input to add TM_X97 and TM_Y97 columns
df_results = df_results.merge(df_input[['Station', 'TM_X97', 'TM_Y97']], on='Station', how='left')
#print(df_results)

# drop columns top 5 dominant frequencies and dominant frequencies in intervals
df_results = df_results.drop(columns=['Top 4 Dominant Frequencies', 'Dominant Frequencies in Intervals'])
# bring column classification to the end3
df_results = df_results[['Station', 'TM_X97', 'TM_Y97', 'Classification']]
#  drop rows with nan values
df_results = df_results.dropna()
print(df_results)


# print tide stations
tide_stations = df_results[df_results['Classification'] == 'tides']
#print(tide_stations)

# Add the amplitude for the frequency corresponding to tide (1.93 cpd)
# applitudes are in the column 'Top 5 Dominant Amplitudes' and 'dominant amplitudes in intervals'
# Add a new column 'Tide Amplitude' to df_results
tide_amplitudes = []
for idx, row in tide_stations.iterrows():
    top_amplitudes = row['Top 5 Dominant Amplitudes']
    interval_amplitudes = row['Dominant Amplitudes in Intervals']
    if any(abs(freq - 1.93) < tolerance for freq in row['Top 5 Dominant Frequencies'] if freq is not None):
        idx = np.where(row['Top 5 Dominant Frequencies'] == 1.93)[0][0]
        tide_amplitudes.append(top_amplitudes[idx])
    else:
        idx = np.where(row['Dominant Frequencies in Intervals'] == 1.93)[0][0]
        tide_amplitudes.append(interval_amplitudes[idx])
tide_stations['Tide Amplitude'] = tide_amplitudes
print(tide_stations)
