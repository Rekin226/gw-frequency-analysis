import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt
import shapefile as shp
import logging
import sys
import os
import jfft

logging.basicConfig(level=logging.INFO)

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

def detect_tidal_candidates(df_top_freqs, station, known_frequencies=[1.93, 2.0], tolerance=0.01, amplitude_threshold=0.001):
    """
    Identify tidal frequency candidates within a tolerance window.
    Returns:
        List of dicts containing candidate info.
    """
    candidates_list = []
    for freq_target in known_frequencies:
        lower_bound = freq_target - tolerance
        upper_bound = freq_target + tolerance
        mask = (df_top_freqs['Frequency'] >= lower_bound) & (df_top_freqs['Frequency'] <= upper_bound)
        subset = df_top_freqs[mask]
        if not subset.empty:
            candidate = subset.loc[subset['Amplitude'].idxmax()]
            if candidate['Amplitude'] > amplitude_threshold:
                candidates_list.append({'Station': station, 'Frequency': candidate['Frequency'],
                                        'Amplitude': candidate['Amplitude'], 'Target_Freq': freq_target})
                logging.debug("Tidal candidate for station %s at target %s cpd", station, freq_target)
    return candidates_list

def process_shapefile(sf, zone):
    """
    Process a shapefile and filter metadata by a given zone.
    """
    df_meta = pd.DataFrame(sf.records(), columns=[field[0] for field in sf.fields[1:]])
    df_meta = df_meta[df_meta['GW_ZONE'] == zone]
    df_meta = df_meta[['ST_NO', 'NAME_C', 'TM_X97', 'TM_Y97']]
    df_meta['ST_NO'] = df_meta['ST_NO'].str.lstrip('0')
    df_meta = df_meta[~df_meta['ST_NO'].str.startswith(('8', '1'))]
    return df_meta



def main():
    # Data Ingestion with robust input validation
    try:
        df_gw_st = pd.read_csv('../data/all_well_imputation_cleaned.csv')
    except Exception as e:
        logging.error("Error reading CSV file: %s", e)
        sys.exit(1)
    df_gw_st.set_index('date time', inplace=True)
    df_gw_st.columns = df_gw_st.columns.str.lstrip('0')
    

    zone = '濁水溪沖積扇'
    try:
        sf1 = shp.Reader('../data/gwobwell_e/gwobwell_e.shp')
    except Exception as e:
        logging.error("Error reading shapefile gwobwell_e: %s", e)
        sys.exit(1)
    try:
        sf2 = shp.Reader('../data/gwobwell_a/gwobwell_a.shp')
    except Exception as e:
        logging.error("Error reading shapefile gwobwell_a: %s", e)
        sys.exit(1)

    df_meta1 = process_shapefile(sf1, zone)
    #logging.info('df_meta1: %s', df_meta1)
    df_meta2 = process_shapefile(sf2, zone)
    #logging.info('df_meta2: %s', df_meta2)

    # Merge metadata
    df_meta = pd.concat([df_meta1, df_meta2], ignore_index=True)
    #logging.info('Merged metadata: %s', df_meta)
    # Remove stations that contain "(2)", "(3)", "(4)" or "(5)" in NAME_C
    df_meta = df_meta[~df_meta['NAME_C'].str.contains(r'\(2\)|\(3\)|\(4\)|\(5\)', regex=True, na=False)]
    # print df_meta
    #logging.info('df_meta after removing duplicates: %s', df_meta)
  
    # Create df_input from station identifiers (use only the key column to avoid merge conflicts)
    df_input = pd.DataFrame({'ST_NO': df_meta['ST_NO'].unique()})
    # print df_input
    
    # Normalize key and merge metadata
    df_input['ST_NO'] = df_input['ST_NO'].astype(str).str.lstrip('0')
    df_meta['ST_NO'] = df_meta['ST_NO'].astype(str).str.lstrip('0')
    df_input = df_input.merge(df_meta, on='ST_NO', how='left')
    missing = df_input['NAME_C'].isna()
    if missing.any():
        logging.warning("Unmatched station metadata for stations: %s", df_input.loc[missing, 'ST_NO'].tolist())
    # Rename key for consistency
    df_input.rename(columns={'ST_NO': 'Station'}, inplace=True)

    # create a list of stations in df_input
    stations = df_input['Station'].tolist()
    # Filter df_gw_st to include only the stations in df_input
    df_gw_st = df_gw_st[df_gw_st.columns.intersection(stations)]

    # filter df_input to include only the stations in df_gw_st
    df_input = df_input[df_input['Station'].isin(df_gw_st.columns)]
    # save df_input to csv file
    #df_input.to_csv('df_input.csv', index=False)
    
    # identitfy the stations with the same TM_X97 and TM_Y97 coordinates
    duplicates = df_input[df_input.duplicated(subset=['TM_X97', 'TM_Y97'], keep=False)]
    if not duplicates.empty:
        logging.info("Stations with duplicate coordinates:\n%s", duplicates.sort_values(['TM_X97', 'TM_Y97']))

    # Prepare DataFrames for frequency analysis
    df_tides = pd.DataFrame(columns=['Station', 'Frequency', 'Amplitude'])
    df_top1_freqs = pd.DataFrame(columns=['station', 'Frequency', 'Amplitude'])

    for station in df_gw_st.columns:  # Skip the first column which is 'date time'
        # Extract the signal
        signal = df_gw_st[station].values

        # Standardize filtering parameters based on station/region-specific noise levels
        cutoff = 0.5          # Cutoff frequency in cycles per day  
        fs = 24               # Sampling frequency in cycles per day
        filter_order = 5      # Filter order (adjust as needed)
        filtered_signal = high_pass_filter(signal, cutoff, fs, order=filter_order)

        # Identify the top 5 dominant frequencies
        df_top_freqs = identify_top_dominant_frequencies(filtered_signal, top_n=5)
        
        # store the top 1 dominant frequency and its amplitude
        df_top1 = pd.DataFrame({'station': station, 'Frequency': df_top_freqs['Frequency'][0],
                                'Amplitude': df_top_freqs['Amplitude'][0]}, index=[0])
        df_top1_freqs = pd.concat([df_top1_freqs, df_top1], ignore_index=True)

        # Identify tidal candidates using the helper function
        tidal_candidates = detect_tidal_candidates(df_top_freqs, station)
        if tidal_candidates:
            for candidate in tidal_candidates:
                df_tides = pd.concat([df_tides, pd.DataFrame([candidate])], ignore_index=True)

    # merge df_top1_freqs with df_input to add TM_X97 and TM_Y97 columns
    df_top1_freqs = df_top1_freqs.merge(df_input[['Station','NAME_C' , 'TM_X97', 'TM_Y97']], left_on='station', right_on='Station', how='left')
    # bring column classification to the end
    df_top1_freqs = df_top1_freqs[['station', 'TM_X97', 'TM_Y97', 'Frequency', 'Amplitude']]

    # print the top five stations with the highest amplitude
    df_top1_freqs = df_top1_freqs.sort_values(by='Amplitude', ascending=False)
    logging.info('df_top1_freqs: %s', df_top1_freqs)
    # print stations with frequency=2 in df_top1_freqs and length of the dataframe
    df_top1_freqs_2 = df_top1_freqs[df_top1_freqs['Frequency'] == 2]
    logging.info('length of df_top1_freqs_2: %s', len(df_top1_freqs_2))
    logging.info('df_top1_freqs_2: %s', df_top1_freqs_2)

    # Ensure the 'Station' column in both DataFrames is of the same type
    df_tides['Station'] = df_tides['Station'].astype(str)
    df_input['Station'] = df_input['Station'].astype(str)


    # Merge df_tides with df_input to add TM_X97 and TM_Y97 columns
    df_tides = df_tides.merge(df_input[['Station','NAME_C' , 'TM_X97', 'TM_Y97']], on='Station', how='left')
    # bring column classification to the end
    df_tides = df_tides[['Station', 'TM_X97', 'TM_Y97', 'Frequency', 'Amplitude']]
    #logging.info('df_tides head: %s', df_tides.head())

    # classify df_tides based on the amplitude into 'sea tides' and 'earth tides'
    #df_tides['Classification'] = np.where(df_tides['Amplitude'] > 0.03, 'Sea Tides', 'Earth Tides')
    # save the classified data to a csv file
    #df_tides.to_csv('df_tides_sea_earth.csv', index=False)



    # Calculate the median and interquartile range (IQR) of the amplitude
    median_amp = df_tides['Amplitude'].median()
    q1 = df_tides['Amplitude'].quantile(0.25)
    q3 = df_tides['Amplitude'].quantile(0.75)
    iqr = q3 - q1

    # Define the threshold as a value above the third quartile plus 1.5 times the IQR
    threshold = q3 + 1.5 * iqr

    # Classify df_tides based on the calculated threshold
    df_tides['Classification'] = np.where(df_tides['Amplitude'] > threshold, 'Sea Tides', 'Earth Tides')
    logging.info('df_tides after classification: %s', df_tides)
    # save the classified data to a csv file
    df_tides.to_csv('df_tides_sea_earth.csv', index=False)



    # print the df_sea_tides
    df_sea_tides = df_tides[df_tides['Classification'] == 'Sea Tides']
    # Group stations in df_sea_tides based on amplitude
    median_amp = df_sea_tides['Amplitude'].median()
    df_sea_tides['Amplitude_Group'] = np.where(df_sea_tides['Amplitude'] > median_amp, 'High Amplitude', 'Low Amplitude')
    logging.info('df_sea_tides: %s', df_sea_tides)

    # Remove duplicate stations with same TM_X97 and TM_Y97 by keeping station with maximum amplitude
    df_sea_tides = df_sea_tides.loc[df_sea_tides.groupby(['TM_X97', 'TM_Y97'])['Amplitude'].idxmax()]
    logging.info("df_sea_tides after removing duplicates by max amplitude:\n%s", df_sea_tides)

    # Identify stations in df_sea_tides with duplicate TM_X97 and TM_Y97 coordinates
    duplicates_sea_tides = df_sea_tides[df_sea_tides.duplicated(subset=['TM_X97', 'TM_Y97'], keep=False)]
    if not duplicates_sea_tides.empty:
        logging.info("Stations in df_sea_tides with duplicate coordinates:\n%s", duplicates_sea_tides.sort_values(['TM_X97', 'TM_Y97']))

    # print the classified sea tides and earth tides
    #logging.info('df_tides: %s', df_tides)

    # print sea tides
    sea_tides = df_tides[df_tides['Classification'] == 'Sea Tides'].copy()  # Use .copy() to avoid SettingWithCopyWarning
    # print the sea tides
    logging.info('sea_tides: %s', sea_tides)
    # save the sea tides to a csv file in tides_analysis folder
    sea_tides.to_csv('sea_tides.csv', index=False)
    sea_tides["active"] = 0
    # drop frequency, classification, and amplitude columns
    sea_tides.drop(columns=["Frequency", "Amplitude", "Classification"], inplace=True)
    # add a column 'tank_size' to the dataframe and set it to 3 and freq_type to 'DAILY'
    #sea_tides["tank_size"] = 3
    sea_tides["freq_type"] = 'DAILY'
    #logging.info('sea_tides: %s', sea_tides)

    # add a column 'tank_size' to df_input and set it to 3 and freq_type to 'DAILY' and remove the column 'NAME_C'
    #df_input["tank_size"] = 3
    #df_input["freq_type"] = 'DAILY'
    # add a column 'active' to df_input and set it to 0
    df_input["active"] = 1
    df_input.drop(columns=["NAME_C"], inplace=True)
    #print(df_input)
    # save df_input to csv file
    #df_input.to_csv('df_input.csv', index=False)


if __name__ == "__main__":
    main()