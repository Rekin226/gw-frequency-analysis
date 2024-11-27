import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq
from scipy.signal import butter, filtfilt

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
    for interval in intervals:
        mask = (freqs >= interval[0]) & (freqs < interval[1])
        interval_freqs = freqs[mask]
        interval_powers = power_spectrum[mask]
        if len(interval_powers) > 0:
            peak_freq = interval_freqs[np.argmax(interval_powers)]
            dominant_frequencies.append(peak_freq)
        else:
            dominant_frequencies.append(None)
    return dominant_frequencies


# Load the data from the CSV file all_wells_cleaned.csv
df_gw_st = pd.read_csv('data/all_well_imputation_cleaned.csv')
# set the column 'date time' as the index
df_gw_st.set_index('date time', inplace=True)
print(df_gw_st.head())

# Apply the analysis to the first 5 stations in the dataframe
for station in df_gw_st.columns[1:6]:  # Skip the first column which is 'date time'
    print(f'Processing station: {station}')
    
    # Extract the signal
    signal = df_gw_st[station].values

    # Apply a high-pass filter to df_gw_st['09200221']
    cutoff = 0.5  # Cutoff frequency in cycles per day  
    fs = 24  # Sampling frequency in cycles per day
    filtered_signal = high_pass_filter(signal, cutoff, fs)

    # Identify the top 5 dominant frequencies
    top_freqs, top_powers, top_amplitudes = identify_top_dominant_frequencies(filtered_signal, top_n=5)
    #print(f'The top 5 dominant frequencies for {station} are:', top_freqs)
    #print('Their corresponding power values are:', top_powers)

    # create a dataframe to store the dominant frequencies including the station name
    # station is the column name top_freqs is the column value
    df_dominant_freq = pd.DataFrame({station: top_freqs})
    print(f'Top 4 dominant frequencies for each {station}:', df_dominant_freq)

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
    dominant_frequencies = find_dominant_frequency_in_intervals(freqs, power_spectrum, intervals)
    
    # Find the amplitude of each dominant frequency
    dominant_amplitudes = []
    for freq in dominant_frequencies:
        if freq is not None:
            idx = np.where(freqs == freq)[0][0]
            dominant_amplitudes.append(fft_values[idx])
        else:
            dominant_amplitudes.append(None)
    
    print(f'The dominant frequencies for {station}') 
    # Create a dataframe to store the dominant frequencies and their amplitudes in each interval
    df_dominant_freq_intervals = pd.DataFrame({
        f'{station}_frequency': dominant_frequencies,
        f'{station}_amplitude': dominant_amplitudes
    })
    print(df_dominant_freq_intervals)
    
