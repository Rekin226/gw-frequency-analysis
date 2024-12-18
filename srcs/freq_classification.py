import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq
from scipy.signal import butter, filtfilt
import sys
import shapefile as shp


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

    # create a dataframe to store the top frequencies and their corresponding amplitudes
    df_top_freqs = pd.DataFrame({'Frequency': top_freqs, 'Amplitude': top_amplitudes})

    return df_top_freqs


# Load the data from the CSV file all_wells_cleaned.csv
df_gw_st = pd.read_csv('data/all_well_imputation_cleaned.csv')
# set the column 'date time' as the index
df_gw_st.set_index('date time', inplace=True)
# drop the first 0 from the station values 
df_gw_st.columns = df_gw_st.columns.str.lstrip('0')
# print the first 5 rows
print(df_gw_st.head())


def process_shapefile(sf, zone):
    df_meta = pd.DataFrame(sf.records(), columns=[field[0] for field in sf.fields[1:]])
    df_meta = df_meta[df_meta['GW_ZONE'] == zone]
    df_meta = df_meta[['ST_NO', 'NAME_C', 'TM_X97', 'TM_Y97']]
    df_meta['ST_NO'] = df_meta['ST_NO'].str.lstrip('0')
    df_meta = df_meta[~df_meta['ST_NO'].str.startswith(('8', '1'))]
    return df_meta

zone = '濁水溪沖積扇'
df_meta1 = process_shapefile(shp.Reader('data/gwobwell_e/gwobwell_e.shp'), zone)
print('df_meta1:', df_meta1)

df_meta2 = process_shapefile(shp.Reader('data/gwobwell_a/gwobwell_a.shp'), zone)
print('df_meta2:', df_meta2)

# merge df_meta1 and df_meta2
df_meta = pd.concat([df_meta1, df_meta2], ignore_index=True)
print('df_meta:', df_meta)

# Create a dataframe df_input with the columns 'ST_NO', 'NAME_C', 'TM_X97', 'TM_Y97'
df_input = pd.DataFrame({
    'ST_NO': df_gw_st.columns,
    'NAME_C': [None] * len(df_gw_st.columns), # Initialize with None
    'TM_X97': [None] * len(df_gw_st.columns), # Initialize with None
    'TM_Y97': [None] * len(df_gw_st.columns) # Initialize with None
}
)

# assign the values of 'NAME_C', 'TM_X97', 'TM_Y97' from df_meta to df_input
for index, row in df_meta.iterrows():
    st_no = row['ST_NO']
    mask = df_input['ST_NO'] == st_no
    df_input.loc[mask, 'NAME_C'] = row['NAME_C']
    df_input.loc[mask, 'TM_X97'] = row['TM_X97']
    df_input.loc[mask, 'TM_Y97'] = row['TM_Y97']

# rename the column 'ST_NO' to 'Station'
df_input.rename(columns={'ST_NO': 'Station'}, inplace=True)
#print('df_input:', df_input)
# print the station with none values
#print('df_input with None values:', len(df_input[df_input.isnull().any(axis=1)]))
#sys.exit()



# Apply the analysis to the first 5 stations in the dataframe
results = []
df_tides = pd.DataFrame(columns=['Station', 'Frequency', 'Amplitude'])
df_top1_freqs = pd.DataFrame(columns=['station', 'Frequency', 'Amplitude'])

for station in df_gw_st.columns:  # Skip the first column which is 'date time'
    #print(f'Processing station: {station}')
    
    # Extract the signal
    signal = df_gw_st[station].values

    # Apply a high-pass filter to df_gw_st['09200221']
    cutoff = 0.5  # Cutoff frequency in cycles per day  
    fs = 24  # Sampling frequency in cycles per day
    filtered_signal = high_pass_filter(signal, cutoff, fs)

    # Identify the top 5 dominant frequencies
    df_top_freqs = identify_top_dominant_frequencies(filtered_signal, top_n=5)
    #print(f'The top 5 dominant frequencies for {station} are:', df_top_freqs)
    
    # store the top 1 dominant frequency and its amplitude in a dataframe df_top1_freqs
    df_top1 = pd.DataFrame({'station': station, 'Frequency': df_top_freqs['Frequency'][0], 'Amplitude': df_top_freqs['Amplitude'][0]}, index=[0])
    df_top1_freqs = pd.concat([df_top1_freqs, df_top1], ignore_index=True)

    # Find index of the frequency close to 1.93 with tolerance 0.003
    idx = np.where(np.isclose(df_top_freqs['Frequency'], 1.93, atol=0.003))[0]
    if len(idx) > 0:
        #print(f'Closest frequency to 1.93: {df_top_freqs["Frequency"][idx[0]]}')
        #print(f'Amplitude: {df_top_freqs["Amplitude"][idx[0]]}')

        # store the frequency and amplitude in a dataframe
        df_closest_freq = pd.DataFrame({'Station': station, 'Frequency': df_top_freqs['Frequency'][idx[0]], 'Amplitude': df_top_freqs['Amplitude'][idx[0]]}, index=[0])
        #print(df_closest_freq)
        df_tides = pd.concat([df_tides, df_closest_freq], ignore_index=True)

# print df_tides
#print('df_tides:', df_tides)

# print df_top1_freqs
#print('df_top1_freqs:', df_top1_freqs)

# merge df_top1_freqs with df_input to add TM_X97 and TM_Y97 columns
df_top1_freqs = df_top1_freqs.merge(df_input[['Station','NAME_C' , 'TM_X97', 'TM_Y97']], left_on='station', right_on='Station', how='left')
# bring column classification to the end
df_top1_freqs = df_top1_freqs[['station', 'TM_X97', 'TM_Y97', 'Frequency', 'Amplitude']]

# print the top five stations with the highest amplitude
df_top1_freqs = df_top1_freqs.sort_values(by='Amplitude', ascending=False).head(10)
print('df_top1_freqs:', df_top1_freqs)



# Ensure the 'Station' column in both DataFrames is of the same type
df_tides['Station'] = df_tides['Station'].astype(str)
df_input['Station'] = df_input['Station'].astype(str)

# Merge df_tides with df_input to add TM_X97 and TM_Y97 columns
df_tides = df_tides.merge(df_input[['Station','NAME_C' , 'TM_X97', 'TM_Y97']], on='Station', how='left')
# bring column classification to the end
df_tides = df_tides[['Station', 'TM_X97', 'TM_Y97', 'Frequency', 'Amplitude']]
#print(df_tides.head())

# classify df_tides based on the amplitude into 'sea tides' and 'earth tides'
df_tides['Classification'] = np.where(df_tides['Amplitude'] > 0.03, 'Sea Tides', 'Earth Tides')
#print('df_tides:', df_tides)

# print sea tides
sea_tides = df_tides[df_tides['Classification'] == 'Sea Tides']
#print('sea_tides:', sea_tides)

# save as csv file
#sea_tides.to_csv('workspace/sea_tides.csv', index=False)
    






