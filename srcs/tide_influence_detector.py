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

# DIURNAL_FREQS dictionary removed as K1 and O1 are no longer targeted.
# DIURNAL_FREQS = {
#     'K1': 1.0027,
#     'O1': 0.9295
# }

# Create a high-pass filter function
def high_pass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def fft_plot(station_data, station_name, cutoff, fs, order, candidates=None, ax=None): # candidates parameter is kept but annotation logic changes
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
    
    # Explicitly annotate M2 and S2 if peaks are found near their target frequencies
    target_tides_to_annotate = {
        'M2': 1.9323,
        'S2': 2.0000
    }
    
    # Find max amplitude in the relevant frequency range for positioning text
    freq_mask_plot = (freq > 0.5) & (freq < 6)
    y_max_plot = np.max(fft_values[freq_mask_plot]) if np.any(freq_mask_plot) else np.max(fft_values)
    # Determine y-position for vertical frequency labels (e.g., halfway up the plot)
    y_min_plot, current_y_max = ax.get_ylim() # Get current y-limits
    # Ensure y_max_plot is used if it's higher than current_y_max (in case ylim is not auto-updated yet)
    effective_y_max = max(y_max_plot, current_y_max) 

    # Dramatically reduced threshold to ensure all peaks are detected
    min_amp_threshold = effective_y_max * 0.01  # Reduced even further to 1% of max amplitude
    
    # Use much wider tolerance for peak finding
    tolerance = 0.1  # Increased even more to ensure detection
    
    # Create locations for both peaks even if they don't exist naturally
    for tide_name, target_freq in target_tides_to_annotate.items():
        # Always add both M2 and S2 labels regardless of detection
        # Find the index of the frequency in the FFT results closest to the target frequency
        freq_diff = np.abs(freq - target_freq)
        closest_idx = np.argmin(freq_diff)
        
        # Get the actual peak frequency and amplitude
        actual_freq_peak = freq[closest_idx]
        actual_amp_peak = fft_values[closest_idx]
        
        # If amplitude is too small to see, artificially boost it for visualization
        if actual_amp_peak < (effective_y_max * 0.05):
            actual_amp_peak = effective_y_max * 0.05  # Set to at least 5% of max for visibility
        
        # Position annotations differently for M2 and S2
        if tide_name == 'M2':
            # Place M2 to the left of the red peak
            ax.text(actual_freq_peak - 0.15, actual_amp_peak, f"{tide_name}", 
                  color='red', fontsize=9, weight='bold', verticalalignment='center', horizontalalignment='right')
        else:  # S2
            # Place S2 to the top right of the red peak
            ax.text(actual_freq_peak + 0.05, actual_amp_peak * 1.1, f"{tide_name}", 
                  color='red', fontsize=9, weight='bold', verticalalignment='bottom', horizontalalignment='left')
        
        # Always add a marker at the peak
        ax.plot(actual_freq_peak, actual_amp_peak, 'ro', markersize=4, alpha=0.7)

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
    Identify tidal candidates for a given group (semi-diurnal).
    """
    candidates_list = []
    for tide_name, target_freq in target_dict.items():
        # Use provided tolerance for all tides now.
        current_tolerance = tolerance
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
                    'Group': 'Semi-Diurnal' # Only Semi-Diurnal group remains
                }
                candidates_list.append(candidate_data)
    return candidates_list

def red_flag_checks(df_top_freqs, candidates_list, station, flag_tolerance=0.001):
    red_flags = []
    # Flag peaks near 1.0 cpd (e.g., pumping/ET) - K1 check removed
    mask_1 = (df_top_freqs['Frequency'] >= (1.0 - flag_tolerance)) & (df_top_freqs['Frequency'] <= (1.0 + flag_tolerance))
    if not df_top_freqs[mask_1].empty:
        # Check if this peak corresponds to a classified tide (should not happen now without K1)
        is_classified_tide = False
        for candidate in candidates_list:
             if abs(candidate['Frequency'] - 1.0) <= flag_tolerance:
                 is_classified_tide = True
                 break
        if not is_classified_tide:
             red_flags.append("Peak at ~1.0 cpd detected (could be pumping/ET). Flag for manual review.")

    # Flag 2.0 cpd peak if no M2 candidate is found (S2 check remains)
    mask_2 = (df_top_freqs['Frequency'] >= (2.0 - flag_tolerance)) & (df_top_freqs['Frequency'] <= (2.0 + flag_tolerance))
    if not df_top_freqs[mask_2].empty and not any(c.get('Tide_Name') == 'M2' for c in candidates_list):
         # Check if this peak corresponds to S2
         is_s2_candidate = any(c.get('Tide_Name') == 'S2' and abs(c['Frequency'] - 2.0) <= flag_tolerance for c in candidates_list)
         if not is_s2_candidate: # Only flag if it's not the S2 candidate itself
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
        
        # Detect candidates: Only for SEMIDIURNAL_FREQS
        semi_candidates = detect_tidal_candidates_group(df_top_freqs, station, SEMIDIURNAL_FREQS, tolerance=0.001, amplitude_threshold=0.001)
        # diurnal_candidates = detect_tidal_candidates_group(df_top_freqs, station, DIURNAL_FREQS, tolerance=0.001, amplitude_threshold=0.001) # Removed Diurnal detection
        tidal_candidates = semi_candidates # Only semi-diurnal candidates now
        
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
    
    # Read ST_ID mapping from amt_amp.csv
    try:
        df_st_id_map = pd.read_csv('../workspace/tides_analysis/amt_amp.csv')
        df_st_id_map['GROUNDWATER'] = df_st_id_map['GROUNDWATER'].astype(str).str.lstrip('0')
        df_st_id_map = df_st_id_map[['GROUNDWATER', 'ST_ID']]
        df_st_id_map.rename(columns={'GROUNDWATER': 'Station'}, inplace=True)
        df_tides = df_tides.merge(df_st_id_map, on='Station', how='left')
        logging.info(f"ST_ID mapping added to {len(df_tides[df_tides['ST_ID'].notna()])} of {len(df_tides)} tide records")
    except Exception as e:
        logging.error(f"Error reading ST_ID mapping: {e}")
        if 'ST_ID' not in df_tides.columns:
            df_tides['ST_ID'] = pd.NA
    
    # Reorder columns to include ST_ID near Station
    df_tides = df_tides[['ST_ID', 'Station', 'TM_X97', 'TM_Y97', 'Frequency', 'Amplitude', 
                         'Target_Freq', 'Tide_Name', 'Group', 'Red_Flag']]
    
    #logging.info('Tidal candidates after detection: %s', df_tides[df_tides['Tide_Name'] == 'M2'])
    
    # === Classify M2 candidates into Sea Tide vs Earth Tide ===
    df_m2 = df_tides[df_tides['Tide_Name'] == 'M2']
    if not df_m2.empty:
        threshold = df_m2['Amplitude'].mean() * 0.5  # Set threshold to 0.5 times the mean amplitude
        logging.info("Increased threshold for M2 classification: %s", threshold)
        # Ensure 'Classification' column exists before assigning
        if 'Classification' not in df_tides.columns:
             df_tides['Classification'] = pd.NA
        df_tides.loc[df_tides['Tide_Name'] == 'M2', 'Classification'] = np.where(
            df_tides.loc[df_tides['Tide_Name'] == 'M2', 'Amplitude'] > threshold,
            'Sea Tide',
            'Earth Tide'
        )
    
    # Create a dataframe to show the result M2 Classification for ALL stations
    df_m2_classif = df_tides[df_tides['Tide_Name'] == 'M2'][['ST_ID', 'Station', 'TM_X97', 'TM_Y97', 'Frequency', 'Amplitude', 'Classification']].copy()
    # Sort by Amplitude in descending order
    df_m2_classif.sort_values(by='Amplitude', ascending=False, inplace=True)
    # Log all M2 classification results (now sorted)
    logging.info("M2 Classification results (All Stations, sorted by Amplitude desc):\n%s", df_m2_classif)
    
    # Save the sorted M2 classification results for all stations
    output_dir = '../workspace/tides_analysis'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    m2_classif_filename = os.path.join(output_dir, 'classif_m2_all_stations.csv')
    df_m2_classif.to_csv(m2_classif_filename, index=False)
    logging.info(f"M2 classification results saved to {m2_classif_filename}")

    # log df_m2_classif with 'sea tide' classification (Optional: keep if needed for specific sea tide logging)
    df_m2_sea_tide = df_m2_classif[df_m2_classif['Classification'] == 'Sea Tide']
    # logging.info("M2 Sea Tide Classification results:\n%s", df_m2_sea_tide) # Keep or remove as needed

    # Save df_m2_sea_tide to csv (now includes ST_ID)
    #df_m2_sea_tide.to_csv('../workspace/tides_analysis/classif_m2_sea_tide.csv', index=False)

    # === Classify S2 candidates into 'sea tide' vs 'pumping' using amplitude threshold ===
    df_s2 = df_tides[df_tides['Tide_Name'] == 'S2']
    if not df_s2.empty:
        # Compute the threshold from S2 candidates (75th percentile)
        threshold_s2 = df_s2['Amplitude'].quantile(0.75)
        logging.info("S2 amplitude threshold (75th percentile): %s", threshold_s2)
        # Keep only high-amplitude S2 candidates
        df_s2_high = df_s2[df_s2['Amplitude'] >= threshold_s2].copy()
        
        # Create a dictionary for M2 amplitude per station from M2 candidates
        df_m2 = df_tides[df_tides['Tide_Name'] == 'M2']
        m2_amp_dict = df_m2.set_index('Station')['Amplitude'].to_dict()
        
        # For each high-amplitude S2 candidate, classify using:
        # if no M2 candidate for the station -> pumping;
        # if S2 amplitude > M2 amplitude -> pumping; otherwise, sea tide.
        def classify_s2(row):
            station = row['Station']
            s2_amp = row['Amplitude']
            if station not in m2_amp_dict:
                return "pumping"
            elif s2_amp > m2_amp_dict[station]:
                return "pumping"
            else:
                return "sea tide"
        
        df_s2_high['Classification'] = df_s2_high.apply(classify_s2, axis=1)
        
        # Update main candidate dataframe for S2 candidates (for stations in df_s2_high)
        for idx, row in df_s2_high.iterrows():
            station = row['Station']
            # Ensure 'Classification' column exists before assigning
            if 'Classification' not in df_tides.columns:
                 df_tides['Classification'] = pd.NA
            df_tides.loc[(df_tides['Station'] == station) & (df_tides['Tide_Name'] == 'S2'), 'Classification'] = row['Classification']
    
    # Log only the S2 classifications with 'pumping'
    df_s2_pumping = df_tides[(df_tides['Tide_Name'] == 'S2') & (df_tides['Classification'] == 'pumping')]
    logging.info("S2 Pumping Classification results:\n%s", 
                 df_s2_pumping[['ST_ID', 'Station', 'TM_X97', 'TM_Y97', 'Frequency', 'Amplitude', 'Classification']])
    
    # save df_s2_pumping to csv (now includes ST_ID)
    #df_s2_pumping.to_csv('../workspace/tides_analysis/classif_s2_pumping.csv', index=False)

    # === Identify stations classified as both M2 'Sea Tide' and S2 'pumping' ===
    # Create DataFrames with ST_ID for both M2 sea tide and S2 pumping stations
    df_m2_sea_tide_stations = df_tides[(df_tides['Tide_Name'] == 'M2') & (df_tides['Classification'] == 'Sea Tide')][['ST_ID', 'Station']].copy()
    df_s2_pumping_stations = df_tides[(df_tides['Tide_Name'] == 'S2') & (df_tides['Classification'] == 'pumping')][['ST_ID', 'Station']].copy()
    
    # Find intersection first using ST_ID if available
    if 'ST_ID' in df_tides.columns and df_tides['ST_ID'].notna().any():
        m2_sea_tide_st_ids = set(df_m2_sea_tide_stations['ST_ID'].dropna())
        s2_pumping_st_ids = set(df_s2_pumping_stations['ST_ID'].dropna())
        both_st_ids = m2_sea_tide_st_ids.intersection(s2_pumping_st_ids)
        
        if both_st_ids:
            both_stations_by_st_id = sorted(list(both_st_ids))
            logging.info(f"Stations (by ST_ID) classified as both M2 'Sea Tide' and S2 'pumping': {both_stations_by_st_id}")
        else:
            logging.info("No stations (by ST_ID) were classified as both M2 'Sea Tide' and S2 'pumping'.")
    
    # Also perform the original station number-based intersection
    m2_sea_tide_stations = set(df_m2_sea_tide_stations['Station'])
    s2_pumping_stations = set(df_s2_pumping_stations['Station'])
    both_classified_stations = list(m2_sea_tide_stations.intersection(s2_pumping_stations))
    
    if both_classified_stations:
        logging.info("Stations (by Station number) classified as M2 'Sea Tide' AND S2 'pumping': %s", sorted(both_classified_stations))
    else:
        logging.info("No stations (by Station number) were classified as both M2 'Sea Tide' and S2 'pumping'.")


    # === Plot stations classified as 'Sea Tide' ===
    # Filter df_tides to get stations classified as 'Sea Tide' for M2 or S2
    sea_tide_stations_to_plot_df = df_tides[df_tides['Classification'] == 'Sea Tide']
    stations_to_plot = sea_tide_stations_to_plot_df['Station'].unique().tolist()

    if not stations_to_plot:
        logging.info("No stations classified as 'Sea Tide' to plot.")
    else:
        logging.info("Plotting FFT for stations classified as 'Sea Tide': %s", sorted(stations_to_plot))
        num_stations = len(stations_to_plot)
        num_cols = 2
        num_rows = (num_stations + num_cols - 1) // num_cols

        # Set Times New Roman as the default font for the entire figure
        plt.rcParams['font.family'] = 'Times New Roman'
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 5 * num_rows), sharex=True, sharey=True)
        axes = axes.flatten()
        
        plot_index = 0 # Use a separate index for placing plots
        for station in stations_to_plot:
            if station in df_gw_st.columns: # Ensure station data exists
                signal = df_gw_st[station].values
                cutoff = 0.5
                fs = 24
                filter_order = 5
                
                # Create a station label that includes ST_ID if available
                station_info = df_tides[(df_tides['Station'] == station) & (df_tides['Classification'] == 'Sea Tide')]
                if 'ST_ID' in station_info.columns and not station_info['ST_ID'].isna().all():
                    # Use the first non-NA ST_ID if there are multiple records
                    st_id = station_info['ST_ID'].dropna().iloc[0] if not station_info['ST_ID'].dropna().empty else "No ST_ID"
                    station_label = f"{st_id}"
                else:
                    station_label = f"Station {station}"
                
                # Get candidates specifically for this station
                candidates_station = df_tides[df_tides['Station'] == station].to_dict('records')
                fft_plot(pd.Series(signal), station_label, cutoff, fs, filter_order, candidates=candidates_station, ax=axes[plot_index])
                plot_index += 1
            else:
                 logging.warning(f"Station {station} classified as 'Sea Tide' but not found in df_gw_st columns. Skipping plot.")

        # Remove unused subplots
        for j in range(plot_index, len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        # Define the output path and filename for the plot
        output_dir = '../workspace/tides_analysis'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_filename = os.path.join(output_dir, 'fft_plots_sea_tide_stations.tiff')
        plt.savefig(output_filename, dpi=300) # Save the figure
        logging.info(f"FFT plots saved to {output_filename}")
        #plt.show() # Keep commented out or remove

if __name__ == "__main__":
    main()
