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

# Define tidal target dictionaries for separate analyses
SEMIDIURNAL_FREQS = {
    'M2': 1.9323,
    'S2': 2.0000
}

DIURNAL_FREQS = {
    'K1': 1.0027,
    'O1': 0.9295
}

# Create a high-pass filter function
def high_pass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def fft_plot(station_data, station_name, cutoff, fs, order, candidates=None, ax=None):
    if isinstance(station_data, pd.Series):
        station_data_array = station_data.to_numpy()
    else:
        station_data_array = station_data
        
    filtered_data = high_pass_filter(station_data_array, cutoff, fs, order)
    n = len(filtered_data)
    T = 1.0  # Sampling interval in hours
    fft_values = fft(filtered_data)
    fft_values = 2.0/n * np.abs(fft_values[:n//2])
    freq = fftfreq(n, T)[:n//2] * 24  # Convert to cycles per day
    
    if ax is None:
        plt.figure()
        ax = plt.gca()
    ax.plot(freq, fft_values, label='FFT amplitude')
    
    if candidates:
        for candidate in candidates:
            freq_candidate = candidate['Frequency']
            tide_name = candidate['Tide_Name']
            ax.axvline(x=freq_candidate, linestyle='--', color='red', alpha=0.7)
            y_max = np.max(fft_values)
            ax.text(freq_candidate + 0.1, y_max * 0.8, f"{tide_name}\n{freq_candidate:.3f}", 
                    color='red', fontsize=8)
    
    ax.set_title(f'FFT of {station_name}', fontsize=12)
    ax.set_xlabel('Frequency (cycles per day)', fontsize=10)
    ax.set_ylabel('Amplitude', fontsize=10)
    ax.set_xticks([1, 2, 3, 4, 5, 6])
    ax.set_xlim(0, 6)
    ax.grid()
    return ax

def identify_top_dominant_frequencies(signal, top_n=5):
    n = len(signal)
    T = 1.0  # Sampling interval (hours)
    fft_values = fft(signal)
    fft_values = 2.0 / n * np.abs(fft_values[:n//2])
    freqs = fftfreq(n, T)[:n//2] * 24   # Convert to cycles per day
    power_spectrum = np.abs(fft_values) ** 2
    top_indices = np.argsort(power_spectrum)[-top_n:][::-1]
    top_freqs = freqs[top_indices]
    top_amplitudes = fft_values[top_indices]
    df_top_freqs = pd.DataFrame({'Frequency': top_freqs, 'Amplitude': top_amplitudes})
    return df_top_freqs

def detect_tidal_candidates_group(df_top_freqs, station, target_dict, tolerance=0.001, amplitude_threshold=0.001):
    """
    Identify tidal candidates for a given group (semi-diurnal or diurnal).
    For K1, the tolerance is forced to 0.0001 (exact match) to avoid confusion with a pumping signal.
    """
    candidates_list = []
    for tide_name, target_freq in target_dict.items():
        # Use an exact (or near-exact) match for K1 and provided tolerance for others.
        current_tolerance = 0.0001 if tide_name == 'K1' else tolerance
        lower_bound = target_freq - current_tolerance
        upper_bound = target_freq + current_tolerance
        mask = (df_top_freqs['Frequency'] >= lower_bound) & (df_top_freqs['Frequency'] <= upper_bound)
        subset = df_top_freqs[mask]
        if not subset.empty:
            candidate = subset.loc[subset['Amplitude'].idxmax()]
            if candidate['Amplitude'] > amplitude_threshold:
                candidate_data = {
                    'Station': station,
                    'Frequency': candidate['Frequency'],
                    'Amplitude': candidate['Amplitude'],
                    'Target_Freq': target_freq,
                    'Tide_Name': tide_name,
                    'Group': 'Semi-Diurnal' if tide_name in SEMIDIURNAL_FREQS else 'Diurnal'
                }
                candidates_list.append(candidate_data)
    return candidates_list

def red_flag_checks(df_top_freqs, candidates_list, station, flag_tolerance=0.001):
    red_flags = []
    # Flag peaks near 1.0 cpd (e.g., pumping/ET)
    mask_1 = (df_top_freqs['Frequency'] >= (1.0 - flag_tolerance)) & (df_top_freqs['Frequency'] <= (1.0 + flag_tolerance))
    if not df_top_freqs[mask_1].empty:
        red_flags.append("Peak at ~1.0 cpd detected (could be pumping/ET). Flag for manual review.")
    # Flag 2.0 cpd peak if no M2 candidate is found
    mask_2 = (df_top_freqs['Frequency'] >= (2.0 - flag_tolerance)) & (df_top_freqs['Frequency'] <= (2.0 + flag_tolerance))
    if not df_top_freqs[mask_2].empty and not any(c.get('Tide_Name') == 'M2' for c in candidates_list):
        red_flags.append("Peak at ~2.0 cpd detected without M2 candidate (could indicate solar thermal/noise).")
    return red_flags

def process_shapefile(sf, zone):
    df_meta = pd.DataFrame(sf.records(), columns=[field[0] for field in sf.fields[1:]])
    df_meta = df_meta[df_meta['GW_ZONE'] == zone]
    df_meta = df_meta[['ST_NO', 'NAME_C', 'TM_X97', 'TM_Y97']]
    df_meta['ST_NO'] = df_meta['ST_NO'].str.lstrip('0')
    df_meta = df_meta[~df_meta['ST_NO'].str.startswith(('8', '1'))]
    return df_meta

def main():
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
    df_meta2 = process_shapefile(sf2, zone)
    df_meta = pd.concat([df_meta1, df_meta2], ignore_index=True)
    df_meta = df_meta[~df_meta['NAME_C'].str.contains(r'\(2\)|\(3\)|\(4\)|\(5\)', regex=True, na=False)]
    
    df_input = pd.DataFrame({'ST_NO': df_meta['ST_NO'].unique()})
    df_input['ST_NO'] = df_input['ST_NO'].astype(str).str.lstrip('0')
    df_meta['ST_NO'] = df_meta['ST_NO'].astype(str).str.lstrip('0')
    df_input = df_input.merge(df_meta, on='ST_NO', how='left')
    missing = df_input['NAME_C'].isna()
    if missing.any():
        logging.warning("Unmatched station metadata for stations: %s", df_input.loc[missing, 'ST_NO'].tolist())
    df_input.rename(columns={'ST_NO': 'Station'}, inplace=True)
    
    stations = df_input['Station'].tolist()
    df_gw_st = df_gw_st[df_gw_st.columns.intersection(stations)]
    df_input = df_input[df_input['Station'].isin(df_gw_st.columns)]
    
    duplicates = df_input[df_input.duplicated(subset=['TM_X97', 'TM_Y97'], keep=False)]
    if not duplicates.empty:
        logging.info("Stations with duplicate coordinates:\n%s", duplicates.sort_values(['TM_X97', 'TM_Y97']))
    
    # Prepare DataFrame to store tidal candidate results
    df_tides = pd.DataFrame(columns=['Station', 'Frequency', 'Amplitude', 'Target_Freq', 
                                     'Tide_Name', 'Group', 'Red_Flag'])
    
    for station in df_gw_st.columns:
        signal = df_gw_st[station].values
        cutoff = 0.5   # cpd
        fs = 24        # cpd
        filter_order = 5
        filtered_signal = high_pass_filter(signal, cutoff, fs, order=filter_order)
        
        df_top_freqs = identify_top_dominant_frequencies(filtered_signal, top_n=5)
        
        # Detect candidates: for DIURNAL, K1 will be matched exactly.
        semi_candidates = detect_tidal_candidates_group(df_top_freqs, station, SEMIDIURNAL_FREQS, tolerance=0.001, amplitude_threshold=0.001)
        diurnal_candidates = detect_tidal_candidates_group(df_top_freqs, station, DIURNAL_FREQS, tolerance=0.001, amplitude_threshold=0.001)
        tidal_candidates = semi_candidates + diurnal_candidates
        
        red_flags = red_flag_checks(df_top_freqs, tidal_candidates, station, flag_tolerance=0.001)
        if red_flags:
            for flag in red_flags:
                logging.warning("Station %s: %s", station, flag)
            for candidate in tidal_candidates:
                candidate['Red_Flag'] = "; ".join(red_flags)
        else:
            for candidate in tidal_candidates:
                candidate['Red_Flag'] = ""
                
        if tidal_candidates:
            for candidate in tidal_candidates:
                df_tides = pd.concat([df_tides, pd.DataFrame([candidate])], ignore_index=True)
    
    df_tides['Station'] = df_tides['Station'].astype(str)
    df_input['Station'] = df_input['Station'].astype(str)
    df_tides = df_tides.merge(df_input[['Station', 'NAME_C', 'TM_X97', 'TM_Y97']], on='Station', how='left')
    df_tides = df_tides[['Station', 'TM_X97', 'TM_Y97', 'Frequency', 'Amplitude', 
                         'Target_Freq', 'Tide_Name', 'Group', 'Red_Flag']]
    
    #logging.info('Tidal candidates after detection: %s', df_tides[df_tides['Tide_Name'] == 'M2'])
    
    # === Classify M2 candidates into Sea Tide vs Earth Tide ===
    df_m2 = df_tides[df_tides['Tide_Name'] == 'M2']
    if not df_m2.empty:
        threshold = df_m2['Amplitude'].mean() * 0.5  # Set threshold to 1.5 times the mean amplitude
        logging.info("Increased threshold for M2 classification: %s", threshold)
        df_tides.loc[df_tides['Tide_Name'] == 'M2', 'Classification'] = np.where(
            df_tides.loc[df_tides['Tide_Name'] == 'M2', 'Amplitude'] > threshold,
            'Sea Tide',
            'Earth Tide'
        )
    
    # Create a dataframe to show the result M2 Classification
    df_m2_classif = df_tides[df_tides['Tide_Name'] == 'M2'][['Station', 'TM_X97', 'TM_Y97', 'Frequency', 'Amplitude', 'Classification']]
    logging.info("M2 Classification results:\n%s", df_m2_classif)
    
    # Write out tidal candidates classification, if desired
    # df_tides.to_csv('../workspace/df_tides_classif.csv', index=False)
    
    # Plot FFTs for stations with tidal candidates (using 75th percentile amplitude filter for "sea tides")
    sea_tides = df_tides[df_tides['Amplitude'] >= df_tides['Amplitude'].quantile(0.75)].copy()
    sea_tides.to_csv('../workspace/tides_analysis/classif_sea_tides.csv', index=False)
    sea_tides["active"] = 0
    sea_tides.drop(columns=["Frequency", "Amplitude", "Red_Flag"], inplace=True)
    sea_tides["freq_type"] = 'DAILY'
    
    df_input["active"] = 1
    df_input.drop(columns=["NAME_C"], inplace=True)
    
    num_stations = len(sea_tides['Station'])
    num_cols = 2
    num_rows = (num_stations + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 5 * num_rows), sharex=True, sharey=True)
    axes = axes.flatten()
    
    for i, station in enumerate(sea_tides['Station']):
        signal = df_gw_st[station].values
        cutoff = 0.5
        fs = 24
        filter_order = 5
        df_top_freqs = identify_top_dominant_frequencies(high_pass_filter(signal, cutoff, fs, filter_order), top_n=5)
        candidates_station = df_tides[df_tides['Station'] == station].to_dict('records')
        fft_plot(pd.Series(signal), station, cutoff, fs, filter_order, candidates=candidates_station, ax=axes[i])
    
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    # plt.show()

if __name__ == "__main__":
    main()
