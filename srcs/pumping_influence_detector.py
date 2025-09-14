'''
This module analyzes the influence of pumping on groundwater levels.
The target pumping frequencies to identify are 1 cycle per day (1cpd) and 2 cycles per day (2cpd).
The goal is to group each target frequency in categories based on the amplitude of the signal.
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt, detrend
import shapefile as shp
import logging
import sys
import os
import json
from scipy.signal import welch
from scipy.stats import spearmanr
from matplotlib.path import Path as MplPath
from matplotlib.colors import LogNorm, PowerNorm, Normalize
from matplotlib.colors import BoundaryNorm
import matplotlib.patches as mpatches
import geopandas as gpd
from scipy.interpolate import Rbf
from pyproj import CRS, Transformer
from matplotlib.patches import PathPatch

logging.basicConfig(level=logging.INFO)

# Define the pumping target dictionaries for separate analyses
pumping_targets = {
    '1cpd': 1,
    '2cpd': 2
}

# Optional high-pass filter (currently unused)
def high_pass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def process_shapefile(sf, zone):
    df_meta = pd.DataFrame(sf.records(), columns=[field[0] for field in sf.fields[1:]])
    df_meta = df_meta[df_meta['GW_ZONE'] == zone]
    df_meta = df_meta[['ST_NO', 'NAME_C', 'TM_X97', 'TM_Y97']]
    df_meta['ST_NO'] = df_meta['ST_NO'].str.lstrip('0')
    # Filter out stations starting with '8' or '1' as in the original logic
    df_meta = df_meta[~df_meta['ST_NO'].str.startswith(('8', '1'))]
    return df_meta

def identify_top_dominant_frequencies(signal, top_n=5, T_hours=1.0):
    """
    Identify top-N dominant frequencies using a Hann window and zero-padding.
    Returns a DataFrame with Frequency (cpd) and Amplitude.
    """
    signal = np.asarray(signal, dtype=float)
    n = len(signal)
    if n < 4:
        return pd.DataFrame(columns=['Frequency', 'Amplitude'])
    # Detrend and window
    try:
        sig = detrend(signal - np.nanmean(signal))
    except Exception:
        sig = signal - np.nanmean(signal)
    w = np.hanning(n)
    cg = w.mean() if w.mean() != 0 else 1.0  # coherent gain
    nfft = int(2 ** np.ceil(np.log2(n)) * 4)  # 4x zero-padding
    fft_values = fft(sig * w, n=nfft)
    # Frequency axis in cycles per day
    freqs = fftfreq(nfft, T_hours)[:nfft // 2] * 24.0
    fft_mags = (2.0 / (n * cg)) * np.abs(fft_values[:nfft // 2])
    power_spectrum = fft_mags ** 2
    top_indices = np.argsort(power_spectrum)[-top_n:][::-1]
    top_freqs = freqs[top_indices]
    top_amplitudes = fft_mags[top_indices]
    df_top_freqs = pd.DataFrame({'Frequency': top_freqs, 'Amplitude': top_amplitudes})
    return df_top_freqs

def get_amplitude_at_target(signal, target_cpd, T_hours=1.0):
    """
    Compute amplitude at a target cycles-per-day frequency using windowing and zero-padding.
    signal: 1D array-like (no NaNs)
    target_cpd: target cycles per day (e.g., 1 or 2)
    T_hours: sampling interval in hours
    returns: amplitude (float) or np.nan
    """
    signal = np.asarray(signal, dtype=float)
    n = len(signal)
    if n < 12:
        return np.nan
    # Detrend
    try:
        sig = detrend(signal - np.nanmean(signal))
    except Exception:
        sig = signal - np.nanmean(signal)
    # Hann window and zero-padding for better localization
    w = np.hanning(n)
    cg = w.mean() if w.mean() != 0 else 1.0  # coherent gain
    nfft = int(2 ** np.ceil(np.log2(n)) * 4)  # 4x zero-padding
    fft_values = fft(sig * w, n=nfft)
    # Frequency axis in cycles per day
    freqs = fftfreq(nfft, T_hours)[:nfft // 2] * 24.0
    fft_mags = (2.0 / (n * cg)) * np.abs(fft_values[:nfft // 2])
    # Nearest frequency bin to target
    idx = np.argmin(np.abs(freqs - target_cpd))
    return float(fft_mags[idx])

def categorize_with_thresholds(amplitudes: pd.Series, low_thr: float, high_thr: float) -> pd.Series:
    """
    Apply fixed thresholds so categories are comparable across targets.
    """
    def label(v):
        if pd.isna(v):
            return np.nan
        if v <= low_thr:
            return 'low'
        if v <= high_thr:
            return 'medium'
        return 'high'
    return amplitudes.map(label)

def _savefig(fig, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_category_counts(summary: pd.DataFrame, out_dir: str):
    # Count categories for each target
    targets = ['1cpd', '2cpd']
    categories = ['low', 'medium', 'high']
    counts = {t: summary[f'category_{t}'].value_counts().reindex(categories, fill_value=0) for t in targets}

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(categories))
    width = 0.35
    ax.bar(x - width/2, counts['1cpd'].values, width, label='1 cpd')
    ax.bar(x + width/2, counts['2cpd'].values, width, label='2 cpd')
    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in categories])
    ax.set_ylabel('Count')
    ax.set_title('Category counts by target')
    ax.legend()
    _savefig(fig, os.path.join(out_dir, 'category_counts.png'))

def plot_amplitude_scatter(summary: pd.DataFrame, low_thr: float, high_thr: float, out_dir: str):
    # Combined category for color (use 1cpd by default)
    cat = summary['category_1cpd']
    color_map = {'low': '#1f77b4', 'medium': '#ff7f0e', 'high': '#d62728'}
    colors = cat.map(color_map)

    fig, ax = plt.subplots(figsize=(5, 5))
    x = summary['amplitude_1cpd']
    y = summary['amplitude_2cpd']
    ax.scatter(x, y, c=colors, s=30, edgecolor='k', linewidths=0.3, alpha=0.8)
    ax.set_xlabel('Amplitude at 1 cpd')
    ax.set_ylabel('Amplitude at 2 cpd')
    ax.set_title('Amplitude relationship: 1 cpd vs 2 cpd')
    # Draw shared thresholds
    if np.isfinite(low_thr):
        ax.axvline(low_thr, color='gray', linestyle='--', linewidth=1)
        ax.axhline(low_thr, color='gray', linestyle='--', linewidth=1)
    if np.isfinite(high_thr):
        ax.axvline(high_thr, color='gray', linestyle='--', linewidth=1)
        ax.axhline(high_thr, color='gray', linestyle='--', linewidth=1)
    ax.grid(True, alpha=0.3)
    _savefig(fig, os.path.join(out_dir, 'amplitude_scatter.png'))

def plot_spatial_interpolation(summary: pd.DataFrame, out_dir: str,
                               boundary_shp: str = '../data/gis/choushi_edit.shp',
                               grid_size: int = 200):
    """
    Interpolates 1cpd and 2cpd amplitudes using RBF and plots them in two subplots,
    with station locations and labels. Properly clips interpolation to boundary.
    """
    # Define CRS and transformer
    twd97_crs = CRS.from_string("+proj=tmerc +lat_0=0 +lon_0=121 +k=0.9999 +x_0=250000 +y_0=0 +ellps=GRS80 +units=m +no_defs")
    wgs84_crs = CRS.from_string("+proj=longlat +datum=WGS84 +no_defs")
    transformer_to_wgs84 = Transformer.from_crs(twd97_crs, wgs84_crs, always_xy=True)

    # Read boundary shapefile
    try:
        boundary_gdf = gpd.read_file(boundary_shp)
        if boundary_gdf.crs.to_epsg() != 4326:
            boundary_gdf = boundary_gdf.to_crs(epsg=4326)
        xmin, ymin, xmax, ymax = boundary_gdf.total_bounds
        
        # Create a unified boundary geometry for masking
        from shapely.ops import unary_union
        boundary_union = unary_union(boundary_gdf.geometry)
        
    except Exception as e:
        logging.error("Failed to read boundary shapefile '%s': %s", boundary_shp, e)
        return

    # Prepare station data from summary, converting coordinates to WGS84
    df = summary.dropna(subset=['TM_X97', 'TM_Y97', 'station id']).copy()
    x_tm, y_tm = df['TM_X97'].values, df['TM_Y97'].values
    x_wgs, y_wgs = transformer_to_wgs84.transform(x_tm, y_tm)
    df['lon'] = x_wgs
    df['lat'] = y_wgs

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), constrained_layout=True)
    fig.suptitle('Spatial Interpolation of Pumping Amplitudes')

    data_to_plot = {'1cpd': 'amplitude_1cpd', '2cpd': 'amplitude_2cpd'}
    titles = {'1cpd': '1 cpd Amplitude', '2cpd': '2 cpd Amplitude'}

    # Generate interpolation grid
    xi = np.linspace(xmin, xmax, grid_size)
    yi = np.linspace(ymin, ymax, grid_size)
    Xi, Yi = np.meshgrid(xi, yi)

    for ax, (key, col) in zip(axes, data_to_plot.items()):
        sub_df = df.dropna(subset=[col])
        if sub_df.empty:
            logging.warning(f"No data available for {key} to plot interpolation.")
            ax.set_title(titles[key] + " (No data)")
            boundary_gdf.plot(ax=ax, facecolor='none', edgecolor='black')
            continue

        x_coords = sub_df['lon'].values
        y_coords = sub_df['lat'].values
        z_values = sub_df[col].values

        # RBF interpolation
        rbfi = Rbf(x_coords, y_coords, z_values, function='inverse')
        Zi = rbfi(Xi, Yi)

        # Mask interpolation outside boundary
        try:
            from shapely.geometry import Point
            # Create mask for points outside boundary
            mask = np.zeros_like(Zi, dtype=bool)
            for i in range(Xi.shape[0]):
                for j in range(Xi.shape[1]):
                    point = Point(Xi[i, j], Yi[i, j])
                    if not boundary_union.contains(point):
                        mask[i, j] = True
            
            # Apply mask to interpolated data
            Zi_masked = np.ma.masked_array(Zi, mask=mask)
            
        except Exception as e:
            logging.warning(f"Could not apply boundary mask for {key}: {e}")
            Zi_masked = Zi

        # Plot contour with masked data
        try:
            contour = ax.contourf(Xi, Yi, Zi_masked, levels=20, cmap='coolwarm', alpha=0.7, extend='both')
            cbar = fig.colorbar(contour, ax=ax, label='Amplitude')
        except Exception as e:
            logging.warning(f"Contour plot failed for {key}: {e}")
            # Fallback to regular contour without masking
            contour = ax.contourf(Xi, Yi, Zi, levels=20, cmap='coolwarm', alpha=0.7)
            cbar = fig.colorbar(contour, ax=ax, label='Amplitude')

        # Plot boundary
        boundary_gdf.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1.5)

        # Plot station points and labels
        ax.scatter(x_coords, y_coords, c='black', marker='.', s=50, label='Stations', zorder=5)
        for i, station_id in enumerate(sub_df['station id']):
            ax.annotate(station_id, (x_coords[i], y_coords[i]), fontsize=7, 
                       xytext=(2, 2), textcoords='offset points', zorder=6)

        ax.set_title(titles[key])
        ax.set_xlabel('Longitude')
        ax.ticklabel_format(useOffset=False, style='plain')
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel('Latitude')
    _savefig(fig, os.path.join(out_dir, 'spatial_interpolation.png'))


def plot_amplitude_hist(summary: pd.DataFrame, out_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    bins = 20
    for ax, target in zip(axes, ['1cpd', '2cpd']):
        vals = summary[f'amplitude_{target}'].dropna().values
        ax.hist(vals, bins=bins, color='#4c78a8', alpha=0.8)
        ax.set_title(f'Amplitude distribution — {target}')
        ax.set_xlabel('Amplitude')
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel('Count')
    _savefig(fig, os.path.join(out_dir, 'amplitude_histograms.png'))

def plot_sample_periodogram(df_gw_st: pd.DataFrame, stations_list: list, freq_minutes: int, dt_hours: float, out_dir: str):
    # Pick a representative station with >= 12 numeric samples
    sample = next((s for s in stations_list if pd.to_numeric(df_gw_st[s], errors='coerce').dropna().shape[0] >= 12), None)
    if sample is None:
        return
    series = pd.to_numeric(df_gw_st[sample], errors='coerce')
    try:
        rs = series.resample(f'{freq_minutes}min').mean().interpolate('time')
    except Exception:
        rs = series.resample(f'{freq_minutes}min').mean().ffill().bfill()
    y = rs.dropna().values
    n = len(y)
    if n < 16:
        return
    # Window + zero-pad FFT
    w = np.hanning(n)
    cg = w.mean() if w.mean() != 0 else 1.0
    nfft = int(2 ** np.ceil(np.log2(n)) * 4)
    Y = fft((y - np.mean(y)) * w, n=nfft)
    freqs_cpd = fftfreq(nfft, dt_hours)[:nfft // 2] * 24.0
    amp = (2.0 / (n * cg)) * np.abs(Y[:nfft // 2])
    power = amp ** 2

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(freqs_cpd, power, color='#2ca02c', lw=1.0)
    ax.set_xlim(0, 6)  # show diurnal band clearly
    ax.set_xlabel('Frequency (cycles per day)')
    ax.set_ylabel('Power')
    ax.set_title(f'Periodogram — {sample}')
    # Mark targets
    for f0, col in [(1.0, '#d62728'), (2.0, '#1f77b4')]:
        ax.axvline(f0, color=col, linestyle='--', linewidth=1.2, label=f'{f0:.0f} cpd')
    ax.legend()
    ax.grid(True, alpha=0.3)
    _savefig(fig, os.path.join(out_dir, f'periodogram_{sample}.png'))

def generate_figures(summary: pd.DataFrame, df_gw_st: pd.DataFrame, stations_list: list, low_thr: float, high_thr: float, freq_minutes: int, dt_hours: float):
    out_dir = '../results/figures'
    plot_category_counts(summary, out_dir)
    plot_amplitude_scatter(summary, low_thr, high_thr, out_dir)

    plot_spatial_interpolation(summary, out_dir)

    plot_amplitude_hist(summary, out_dir)
    plot_sample_periodogram(df_gw_st, stations_list, freq_minutes, dt_hours, out_dir)

def resample_uniform(series: pd.Series, freq_minutes: int) -> pd.Series:
    """
    Convert to numeric, resample to a uniform grid at freq_minutes, and fill gaps.
    Requires a DatetimeIndex on the parent DataFrame.
    """
    s = pd.to_numeric(series, errors='coerce')
    try:
        rs = s.resample(f'{freq_minutes}min').mean().interpolate('time')
    except Exception:
        rs = s.resample(f'{freq_minutes}min').mean().ffill().bfill()
    return rs

def cohen_kappa(a: pd.Series, b: pd.Series, labels=('low', 'medium', 'high')) -> float:
    """
    Cohen's kappa for categorical agreement between two equal-length label arrays.
    """
    x = pd.Series(a).astype('category')
    y = pd.Series(b).astype('category')
    mask = x.notna() & y.notna()
    if mask.sum() == 0:
        return np.nan
    x = x[mask].astype(pd.CategoricalDtype(categories=list(labels)))
    y = y[mask].astype(pd.CategoricalDtype(categories=list(labels)))
    cm = pd.crosstab(x, y).reindex(index=labels, columns=labels, fill_value=0).values.astype(float)
    n = cm.sum()
    if n == 0:
        return np.nan
    po = np.trace(cm) / n
    pe = (cm.sum(axis=1) @ cm.sum(axis=0)) / (n * n)
    if pe == 1:
        return 1.0
    return (po - pe) / (1 - pe)

def validate_negative_controls(summary: pd.DataFrame,
                               df_gw_st: pd.DataFrame,
                               stations_list: list,
                               control_cpd=(0.37, 1.37, 2.73, 3.41),
                               low_thr: float = np.nan,
                               high_thr: float = np.nan,
                               freq_minutes: int = 60,
                               dt_hours: float = 1.0):
    """
    Compute amplitudes at off-target control frequencies and summarize false-positive rates.
    Uses the same thresholds (low_thr, high_thr) defined from target amplitudes.
    """
    records = []
    for st in stations_list:
        rs = resample_uniform(df_gw_st[st], freq_minutes)
        y = rs.dropna().values
        if y.size < 16:
            continue
        for fcpd in control_cpd:
            amp = get_amplitude_at_target(y, target_cpd=fcpd, T_hours=dt_hours)
            records.append({'Station': st, 'frequency_cpd': fcpd, 'amplitude': amp})
    if not records:
        logging.warning("Negative control validation: no usable stations.")
        return
    df_ctrl = pd.DataFrame(records)
    # Categorize using the same thresholds
    df_ctrl['category'] = categorize_with_thresholds(df_ctrl['amplitude'], low_thr, high_thr)
    # Summaries
    summary_tbl = df_ctrl.groupby('frequency_cpd').agg(
        n=('amplitude', 'size'),
        median_amp=('amplitude', 'median'),
        iqr_amp=('amplitude', lambda x: np.subtract(*np.percentile(x.dropna(), [75, 25]))),
        high_rate=('category', lambda s: np.mean(s == 'high')),
        med_rate=('category', lambda s: np.mean(s == 'medium')),
    ).reset_index()
    out_dir = '../results/validation'
    os.makedirs(out_dir, exist_ok=True)
    df_ctrl.to_csv(os.path.join(out_dir, 'negative_controls_raw.csv'), index=False)
    summary_tbl.to_csv(os.path.join(out_dir, 'negative_controls_summary.csv'), index=False)
    logging.info("Negative controls saved: %s, %s",
                 os.path.join(out_dir, 'negative_controls_raw.csv'),
                 os.path.join(out_dir, 'negative_controls_summary.csv'))
    # Print concise summary
    print("\nNegative controls summary (false-positive rates):")
    print(summary_tbl[['frequency_cpd', 'n', 'high_rate', 'med_rate']].to_string(index=False))

def welch_amplitude_proxy(y: np.ndarray, target_cpd: float, dt_hours: float) -> float:
    """
    Compute a Welch PSD-based amplitude proxy at target frequency.
    Returns sqrt(PSD) at the nearest bin (scale-aligned later).
    """
    n = len(y)
    if n < 16:
        return np.nan
    try:
        sig = detrend(y - np.nanmean(y))
    except Exception:
        sig = y - np.nanmean(y)
    fs = 1.0 / dt_hours  # samples per hour
    nperseg = min(256, n)
    freqs, pxx = welch(sig, fs=fs, window='hann', nperseg=nperseg, noverlap=nperseg//2, detrend=False, scaling='density')
    # Convert to cycles per day
    freqs_cpd = freqs * 24.0
    idx = int(np.argmin(np.abs(freqs_cpd - target_cpd)))
    return float(np.sqrt(max(pxx[idx], 0.0)))

def validate_cross_method_welch(summary: pd.DataFrame,
                                df_gw_st: pd.DataFrame,
                                stations_list: list,
                                low_thr: float,
                                high_thr: float,
                                freq_minutes: int,
                                dt_hours: float):
    """
    Compare FFT-based amplitudes to Welch-based proxies.
    - Spearman correlations on amplitudes
    - Cohen's kappa on categories (Welch amplitudes aligned to FFT scale via median ratio)
    """
    targets = ['1cpd', '2cpd']
    fft_amps = {t: summary[f'amplitude_{t}'].astype(float).values for t in targets}
    welch_amps = {t: [] for t in targets}
    valid_st = []
    for st in stations_list:
        rs = resample_uniform(df_gw_st[st], freq_minutes)
        y = rs.dropna().values
        if y.size < 16:
            # keep alignment with stations_list
            for t in targets: welch_amps[t].append(np.nan)
            continue
        valid_st.append(st)
        for t in targets:
            fcpd = 1.0 if t == '1cpd' else 2.0
            welch_amps[t].append(welch_amplitude_proxy(y, fcpd, dt_hours))
    welch_amps = {t: np.asarray(welch_amps[t], dtype=float) for t in targets}
    # Align Welch scale to FFT by median ratio (robust scaling)
    scale = {}
    scaled_welch = {}
    for t in targets:
        med_fft = np.nanmedian(fft_amps[t])
        med_welch = np.nanmedian(welch_amps[t])
        scale[t] = (med_fft / med_welch) if np.isfinite(med_fft) and np.isfinite(med_welch) and med_welch != 0 else 1.0
        scaled_welch[t] = welch_amps[t] * scale[t]
    # Correlations
    metrics = {}
    for t in targets:
        mask = np.isfinite(fft_amps[t]) & np.isfinite(scaled_welch[t])
        if mask.sum() >= 3:
            rho, p = spearmanr(fft_amps[t][mask], scaled_welch[t][mask])
        else:
            rho, p = np.nan, np.nan
        metrics[f'spearman_{t}'] = {'rho': float(rho), 'p': float(p)}
    # Categories and kappa
    cats_fft = {t: summary[f'category_{t}'] for t in targets}
    cats_welch = {}
    for t in targets:
        s = pd.Series(scaled_welch[t])
        cats_welch[t] = categorize_with_thresholds(s, low_thr, high_thr)
    kappas = {}
    for t in targets:
        k = cohen_kappa(cats_fft[t], cats_welch[t])
        kappas[f'kappa_{t}'] = float(k) if np.isfinite(k) else np.nan
    # Save and print
    out_dir = '../results/validation'
    os.makedirs(out_dir, exist_ok=True)
    results = {'spearman': metrics, 'kappa': kappas, 'scale_factors': scale}
    with open(os.path.join(out_dir, 'cross_method_welch.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print("\nCross-method (Welch) agreement:")
    for t in targets:
        print(f"  {t}: Spearman rho={metrics[f'spearman_{t}']['rho']:.3f}, kappa={kappas[f'kappa_{t}']:.3f} (scale={scale[t]:.3g})")

def validate_threshold_cv(summary: pd.DataFrame, k: int = 5, random_state: int = 42):
    """
    Station-wise K-fold CV for thresholds.
    Train thresholds on (k-1)/k stations, apply to held-out 1/k, compare to reference categories.
    """
    rng = np.random.default_rng(random_state)
    stations = summary['Station'].values
    n = len(stations)
    idx = np.arange(n)
    rng.shuffle(idx)
    folds = np.array_split(idx, k)
    recs = []
    for i, test_idx in enumerate(folds, start=1):
        train_idx = np.setdiff1d(idx, test_idx)
        # Train thresholds on combined amplitudes from train
        train_combined = pd.concat([
            summary.iloc[train_idx]['amplitude_1cpd'],
            summary.iloc[train_idx]['amplitude_2cpd']
        ]).dropna().values
        if train_combined.size == 0:
            recs.append({'fold': i, 'low_thr': np.nan, 'high_thr': np.nan, 'kappa_1cpd': np.nan, 'kappa_2cpd': np.nan, 'pct_switch': np.nan})
            continue
        low_thr = float(np.percentile(train_combined, 33))
        high_thr = float(np.percentile(train_combined, 66))
        # Apply to test and compare with reference categories
        ref1 = summary.iloc[test_idx]['category_1cpd']
        ref2 = summary.iloc[test_idx]['category_2cpd']
        pred1 = categorize_with_thresholds(summary.iloc[test_idx]['amplitude_1cpd'], low_thr, high_thr)
        pred2 = categorize_with_thresholds(summary.iloc[test_idx]['amplitude_2cpd'], low_thr, high_thr)
        k1 = cohen_kappa(ref1, pred1)
        k2 = cohen_kappa(ref2, pred2)
        # Percent switches vs reference (where both notna)
        m1 = ref1.notna() & pred1.notna()
        m2 = ref2.notna() & pred2.notna()
        sw1 = (ref1[m1] != pred1[m1]).mean() if m1.any() else np.nan
        sw2 = (ref2[m2] != pred2[m2]).mean() if m2.any() else np.nan
        recs.append({'fold': i, 'low_thr': low_thr, 'high_thr': high_thr,
                     'kappa_1cpd': float(k1) if np.isfinite(k1) else np.nan,
                     'kappa_2cpd': float(k2) if np.isfinite(k2) else np.nan,
                     'pct_switch': float(np.nanmean([sw1, sw2]))})
    df_cv = pd.DataFrame(recs)
    out_dir = '../results/validation'
    os.makedirs(out_dir, exist_ok=True)
    df_cv.to_csv(os.path.join(out_dir, 'threshold_cv_summary.csv'), index=False)
    print("\nThreshold CV (station-wise) summary:")
    print(df_cv.to_string(index=False))

def main():
    try:
        df_gw_st = pd.read_csv('../data/all_well_imputation_cleaned.csv')
    except Exception as e:
        logging.error("Error reading CSV file: %s", e)
        sys.exit(1)

    # Ensure datetime index
    df_gw_st['date time'] = pd.to_datetime(df_gw_st['date time'])
    df_gw_st.set_index('date time', inplace=True)
    df_gw_st.columns = df_gw_st.columns.str.lstrip('0')

    # Infer sampling interval in hours from index (median diff)
    try:
        diffs = df_gw_st.index.to_series().diff().dt.total_seconds().dropna()
        median_sec = diffs.median() if not diffs.empty else 3600
        T_hours = float(median_sec) / 3600.0
        if T_hours <= 0 or not np.isfinite(T_hours):
            T_hours = 1.0
    except Exception:
        T_hours = 1.0
    logging.info("Inferred sampling interval: %.3f hours", T_hours)

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
    # Drop duplicated wells tagged as (2)(3)(4)(5)
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

    # Compute amplitudes for each station and each target frequency
    amplitudes = {name: [] for name in pumping_targets.keys()}
    stations_list = df_gw_st.columns.tolist()

    # Use a uniform resampling grid derived from inferred T_hours
    freq_minutes = max(1, int(round(T_hours * 60.0)))
    dt_hours = freq_minutes / 60.0
    logging.info("Resampling cadence: every %d minutes (dt_hours=%.3f)", freq_minutes, dt_hours)

    for st in stations_list:
        series = pd.to_numeric(df_gw_st[st], errors='coerce')
        # Resample to uniform grid, then interpolate missing timestamps
        try:
            rs = series.resample(f'{freq_minutes}min').mean().interpolate('time')
        except Exception:
            # Fallback to forward-fill if time interpolation fails
            rs = series.resample(f'{freq_minutes}min').mean().ffill().bfill()
        clean_series = rs.dropna().values

        # Require sufficient data length (at least ~48 samples or >= one day at the given cadence)
        min_samples = max(48, int(np.ceil(24.0 / max(dt_hours, 1e-6))))
        if clean_series.size < min_samples:
            for name in pumping_targets.keys():
                amplitudes[name].append(np.nan)
            continue

        for name, target in pumping_targets.items():
            amp = get_amplitude_at_target(clean_series, target_cpd=target, T_hours=dt_hours)
            amplitudes[name].append(amp)

    summary = pd.DataFrame({'Station': stations_list})
    for name in pumping_targets.keys():
        summary[f'amplitude_{name}'] = amplitudes[name]

    # Use shared thresholds across both targets so categories are comparable
    combined = pd.concat([summary['amplitude_1cpd'], summary['amplitude_2cpd']]).dropna()
    if not combined.empty:
        low_thr = float(np.nanpercentile(combined.values, 33))
        high_thr = float(np.nanpercentile(combined.values, 66))
    else:
        low_thr = high_thr = np.nan

    summary['category_1cpd'] = categorize_with_thresholds(summary['amplitude_1cpd'], low_thr, high_thr)
    summary['category_2cpd'] = categorize_with_thresholds(summary['amplitude_2cpd'], low_thr, high_thr)

    summary = summary.merge(df_input[['Station', 'NAME_C', 'TM_X97', 'TM_Y97']], on='Station', how='left')

    # Remove specific stations from the Pumping amplitude summary
    stations_to_drop = {'7050111', '7010111', '7100211', '9090111'}
    present_to_drop = sorted(stations_to_drop.intersection(set(summary['Station'].astype(str))))
    if present_to_drop:
        print("\nRemoving stations from Pumping amplitude summary:")
        for st in present_to_drop:
            print(f"  - {st}")
        summary = summary[~summary['Station'].astype(str).isin(stations_to_drop)].copy()
    else:
        print("\nNo specified stations to remove from Pumping amplitude summary.")

    # Add sequential station ids (w1, w2, ...) to the summary
    summary = summary.reset_index(drop=True)
    summary['station id'] = [f"w{i}" for i in range(1, len(summary) + 1)]

    out_path = '../results'
    os.makedirs(out_path, exist_ok=True)
    out_file = os.path.join(out_path, 'pumping_amplitude_summary.csv')
    summary.to_csv(out_file, index=False)
    logging.info("Saved amplitude summary to %s", out_file)

    # Print the summary to stdout
    try:
        print("\nPumping amplitude summary (full):")
        print(summary.to_string(index=False))
    except Exception:
        print("\nPumping amplitude summary (preview):")
        print(summary.head(50).to_string(index=False))

    # Print category counts for each target
    for name in pumping_targets.keys():
        counts = summary[f'category_{name}'].value_counts(dropna=True).to_dict()
        print(f"\nCategory counts for {name}: {counts}")

    # Generate and save figures for the paper
    try:
        generate_figures(summary, df_gw_st, stations_list, low_thr, high_thr, freq_minutes, dt_hours)
        logging.info("Saved figures to ../results/figures")
    except Exception:
        logging.debug("Figure generation failed", exc_info=True)

    # Validation: Negative controls and Agreement metrics
    try:
        validate_negative_controls(summary, df_gw_st, stations_list,
                                   control_cpd=(0.37, 1.37, 2.73, 3.41),
                                   low_thr=low_thr, high_thr=high_thr,
                                   freq_minutes=freq_minutes, dt_hours=dt_hours)
    except Exception:
        logging.debug("Negative control validation failed", exc_info=True)
    try:
        validate_cross_method_welch(summary, df_gw_st, stations_list,
                                    low_thr=low_thr, high_thr=high_thr,
                                    freq_minutes=freq_minutes, dt_hours=dt_hours)
    except Exception:
        logging.debug("Cross-method (Welch) validation failed", exc_info=True)
    try:
        validate_threshold_cv(summary, k=5, random_state=42)
    except Exception:
        logging.debug("Threshold CV failed", exc_info=True)

if __name__ == '__main__':
    main()

