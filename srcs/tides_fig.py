# Additional Figure Plotting Script for Manuscript Visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt
import matplotlib as mpl

# Set Times New Roman as the default font for all text elements
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'  # Use STIX for math rendering (compatible with Times)
plt.rcParams['axes.titlesize'] = 14  # Larger title font
plt.rcParams['axes.labelsize'] = 12  # Larger label font
plt.rcParams['axes.titleweight'] = 'bold'  # Bold title
plt.rcParams['axes.labelweight'] = 'bold'  # Bold labels

# Load necessary data (ensure paths are correct)
df_m2 = pd.read_csv('../workspace/tides_analysis/classif_m2_all_stations.csv')
df_all_tides = pd.read_csv('../workspace/tides_analysis/classif_m2_all_stations.csv')  # Placeholder if full tide classification is saved


# === Figure 1 (New): Example Time Series and Frequency Spectrum ===

# Define the high-pass filter function (copied from tide_influence_detector.py for standalone use)
def high_pass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

# Load the main groundwater data
try:
    df_gw_st = pd.read_csv('../data/all_well_imputation_cleaned.csv', parse_dates=['date time'])
    df_gw_st.set_index('date time', inplace=True)
    df_gw_st.columns = df_gw_st.columns.str.lstrip('0') 
except FileNotFoundError:
    print("Error: Groundwater data file not found. Skipping Figure 1.")
    df_gw_st = None

# Select an example station ID (ensure this station exists and shows good tidal signals)
# You might want to pick a station classified as 'Sea Tide' from your previous analysis
example_station_id = '7230311' # Example: Replace with a suitable station ID from your data

if df_gw_st is not None and example_station_id in df_gw_st.columns:
    station_data_original = df_gw_st[example_station_id].dropna()
    
    # Filter parameters
    cutoff_freq = 0.5  # cpd
    sampling_freq = 24 # cpd (assuming hourly data, 1 sample per hour, 24 samples per day)
    filter_order = 5
    
    station_data_filtered = high_pass_filter(station_data_original.values, cutoff_freq, sampling_freq, filter_order)
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 12)) # Changed to 3 subplots
    
    # Subplot (a): Original groundwater level time series
    axes[0].plot(station_data_original.index, station_data_original.values, label='Original Data', alpha=0.7)
    axes[0].set_xlabel('Time', fontfamily='Times New Roman', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Groundwater Level (m)', fontfamily='Times New Roman', fontsize=12, fontweight='bold')
    axes[0].set_title(f'(a) Original Time Series for Station {example_station_id}', fontfamily='Times New Roman', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True)
    axes[0].autoscale(enable=True, axis='x', tight=True)
    # Add padding to y-axis to prevent cutting off maximum values
    y_min, y_max = np.min(station_data_original.values), np.max(station_data_original.values)
    y_range = y_max - y_min
    axes[0].set_ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)

    # Subplot (b): Filtered (high-pass) groundwater level time series
    axes[1].plot(station_data_original.index, station_data_filtered, label='Filtered Data (High-pass > 0.5 cpd)', color='red')
    axes[1].set_xlabel('Time', fontfamily='Times New Roman', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Filtered GWL (m)', fontfamily='Times New Roman', fontsize=12, fontweight='bold')
    axes[1].set_title(f'(b) Filtered Time Series for Station {example_station_id}', fontfamily='Times New Roman', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True)
    axes[1].autoscale(enable=True, axis='x', tight=True)
    # Add padding to y-axis to prevent cutting off maximum values
    y_min, y_max = np.min(station_data_filtered), np.max(station_data_filtered)
    y_range = y_max - y_min
    axes[1].set_ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)
    
    # Subplot (c): Corresponding FFT plot
    n_fft = len(station_data_filtered)
    T_fft = 1.0  # Sampling interval in hours
    
    fft_vals = fft(station_data_filtered)
    fft_vals_amp = 2.0/n_fft * np.abs(fft_vals[:n_fft//2])
    fft_freqs = fftfreq(n_fft, T_fft)[:n_fft//2] * 24  # Convert to cycles per day
    
    axes[2].plot(fft_freqs, fft_vals_amp)
    axes[2].set_xlabel('Frequency (cycles per day)', fontfamily='Times New Roman', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Amplitude', fontfamily='Times New Roman', fontsize=12, fontweight='bold')
    axes[2].set_title(f'(c) FFT Spectrum of Filtered Data for Station {example_station_id}', fontfamily='Times New Roman', fontsize=14, fontweight='bold')
    axes[2].grid(True)
    axes[2].set_xlim(0, 4.5) # Match x-limit with other FFT plots
    axes[2].set_xticks([1, 2, 3, 4])
    # Add padding to y-axis to prevent cutting off maximum values
    y_max = np.max(fft_vals_amp)
    axes[2].set_ylim(0, y_max * 1.1)  # 10% padding above maximum amplitude value


    # Annotate M2 and S2 (simplified from fft_plot)
    target_tides_to_annotate = {'M2': 1.9323, 'S2': 2.0000}
    y_min_fft, y_max_fft = axes[2].get_ylim() # Get ylim after autoscaling
    
    for tide_name, target_f in target_tides_to_annotate.items():
        axes[2].axvline(x=target_f, linestyle='--', color='grey', alpha=0.7)
        
        # Find closest peak in FFT
        freq_diff_fft = np.abs(fft_freqs - target_f)
        closest_idx_fft = np.argmin(freq_diff_fft)
        tolerance_fft = 0.05

        # Removed the vertical frequency annotations (1.9323 and 2.0000 cpd)
        # Position for vertical frequency label calculations still needed for later use
        vertical_label_y_pos = y_min_fft + (y_max_fft - y_min_fft) * 0.5 
        if freq_diff_fft[closest_idx_fft] <= tolerance_fft:
             peak_amp_fft = fft_vals_amp[closest_idx_fft]
             # Ensure label is within plot bounds if peak is very high
             vertical_label_y_pos = min(peak_amp_fft + (y_max_fft - peak_amp_fft) / 2, y_max_fft * 0.95)
             vertical_label_y_pos = max(vertical_label_y_pos, y_min_fft + (y_max_fft-y_min_fft)*0.05)


        if freq_diff_fft[closest_idx_fft] <= tolerance_fft:
            actual_f_peak = fft_freqs[closest_idx_fft]
            actual_a_peak = fft_vals_amp[closest_idx_fft]
            
            if tide_name == 'M2':
                axes[2].text(actual_f_peak - 0.05, actual_a_peak, tide_name, 
                             color='red', fontsize=9, weight='bold', 
                             verticalalignment='bottom', horizontalalignment='right',
                             fontfamily='Times New Roman')
            else:
                axes[2].text(actual_f_peak + 0.05, actual_a_peak, tide_name, 
                             color='red', fontsize=9, weight='bold', 
                             verticalalignment='bottom', horizontalalignment='left',
                             fontfamily='Times New Roman')
            axes[2].plot(actual_f_peak, actual_a_peak, 'ro', markersize=4, alpha=0.7)

    plt.tight_layout()
    plt.savefig('../workspace/tides_analysis/figure_example_timeseries_fft.tiff', dpi=400)
    plt.close()
    print(f"Figure 1: Example Time Series and FFT for station {example_station_id} saved.")
else:
    if df_gw_st is None:
        print("Skipping Figure 1 because groundwater data could not be loaded.")
    else:
        print(f"Error: Example station ID '{example_station_id}' not found in groundwater data. Skipping Figure 1.")
