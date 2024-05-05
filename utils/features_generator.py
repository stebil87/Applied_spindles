import numpy as np
from scipy.signal import butter, filtfilt, welch, hilbert
from scipy.stats import kurtosis, skew
import pandas as pd
from utils.processing import get_stft

def Welch(x, fs, nperseg=1024):
  from scipy.signal import welch
 
  f, Pxx = welch(x, fs, nperseg=nperseg)
  return f, Pxx
 
def calculate_peak_power(eeg_signal, fs, band_low, band_high):
  # Design a bandpass filter
  nyquist = fs / 2
  order = 5  # Adjust filter order as needed
  lowcut, highcut = band_low / nyquist, band_high / nyquist
  b, a = butter(order, [lowcut, highcut], btype='bandpass')
 
  # Filter the signal in the band of interest
  filtered_signal = filtfilt(b, a, eeg_signal)
 
  # Calculate power spectrum density (PSD) using Welch's method
  f, Pxx = welch(filtered_signal, fs, nperseg=50)
 
  # Find the peak power within the band
  peak_freq_idx = np.argmax(Pxx[(f >= band_low) & (f <= band_high)])
  peak_power = Pxx[peak_freq_idx]
 
  # Return the peak power
  return peak_power
 
def calculate_power_ratio(eeg_signal, fs, band_low_spindle, band_high_spindle, band_low_lf, band_high_lf):
 
  # Calculate PSD using Welch's method
  f, Pxx = welch(eeg_signal, fs, nperseg=50)  # Adjust nperseg for better resolution or faster computation
 
  # Calculate power in the sleep spindle band
  spindle_power = np.sum(Pxx[(f >= band_low_spindle) & (f <= band_high_spindle)])
 
  # Calculate power in the low-frequency band
  lf_power = np.sum(Pxx[(f >= band_low_lf) & (f <= band_high_lf)])
 
  # Avoid division by zero (if low-frequency band power is zero)
  if lf_power == 0:
    return 0  # Handle zero denominator case (set ratio to 0)
  else:
    # Calculate and return the power ratio
    power_ratio = spindle_power / lf_power
    return power_ratio
 
def sample_entropy(data, m=2, r=0.2):
    data = np.array(data)
    N = len(data)
 
    # Efficiently calculate distances using vectorization
    diffs = np.abs(data[:, None] - data)
 
    # Identify neighbors within radius and proximity (vectorized)
    neighbors_radius = diffs <= r
    neighbors_proximity = np.triu(np.ones((N, N)), k=1 - m)  # Upper triangle excluding diagonal
 
    # Combine neighbor masks using logical AND (vectorized)
    combined_mask = neighbors_radius * neighbors_proximity
 
    # Calculate counts (vectorized)
    num_a = np.sum(combined_mask)
    num_b = np.sum(neighbors_proximity) - N  # Subtract diagonal elements
 
    return -np.log(num_a / (num_b + 1e-10))  # Avoid d

def _find_closest_value_idx(arr, value):
    closest_index = np.argmin(np.abs(arr - value))
    return closest_index


# ================= Generation ===================

def compute_features(signal, fs, center_time, t, f, Zxx):
    features = {}

    # Time-domain features
    features['mean'] = np.mean(signal)
    features['std_dev'] = np.std(signal)
    features['skewness'] = skew(signal)
    features['kurtosis'] = kurtosis(signal)
    features['zero_crossings'] = len(np.where(np.diff(np.signbit(signal)))[0])

    features["spd_sigma_max"] = spectrogram_based(center_time, t, f, Zxx).max()
    features["spd_sigma_min"] = spectrogram_based(center_time, t, f, Zxx).min()
    features["spd_sigma_std"] = spectrogram_based(center_time, t, f, Zxx).std()

    features["spd_theta_max"] = spectrogram_based(center_time, t, f, Zxx, fmin=3.5, fmax=7.5).max()
    features["spd_theta_min"] = spectrogram_based(center_time, t, f, Zxx, fmin=3.5, fmax=7.5).min()
    features["spd_theta_std"] = spectrogram_based(center_time, t, f, Zxx, fmin=3.5, fmax=7.5).std()
    
    # Frequency-domain features
    f, Pxx = welch(signal, fs=fs)
    spindle_power = np.trapz(Pxx[(f >= 11) & (f <= 16)], f[(f >= 11) & (f <= 16)])
    total_power = np.trapz(Pxx, f)
    features['spindle_power_ratio'] = spindle_power / total_power
    features['peak_frequency'] = f[np.argmax(Pxx)]

    # Time-frequency features
    analytic_signal = hilbert(signal)
    amplitude_envelope = np.abs(analytic_signal)
    features['mean_amplitude_envelope'] = np.mean(amplitude_envelope)

    # # Nonlinear dynamics features
    # features['sample_entropy'] = ent.sample_entropy(signal, 2, 0.2 * np.std(signal))[0]
    
    # # Giorgio
    features["peak_power"] = calculate_peak_power(signal, fs, 11, 14)
    features["power_ratio"] = calculate_power_ratio(signal, fs, 11, 14, 0.3, 8)
    # features["se"] = sample_entropy(signal)
    
    return features

def generate_features(data, fs, step=0.1, low=0, up=0.5, verbose=False):
    featured_df = None
    times, freqs, Zxx = get_stft(data.y.values.flatten(), fs)

    i = 0
    while True:
        if data["time"].max() < up:
            break

        if verbose and i % 100:
            print(f"{i}/{len(data)}")
            
        curr_y = data.loc[(data["time"] <= up) & (data["time"] >= low), "y"].values.flatten()
        center_time = low+(up-low)/2
        f = compute_features(curr_y, fs, center_time, times, freqs, Zxx)
        f["center_time"] = center_time
        f = {k:[v] for k, v in f.items()}
        feats = pd.DataFrame(f)
        
        if featured_df is None:
            featured_df = feats
        else:
            featured_df = pd.concat([featured_df, feats], axis=0)
    
        low += step
        up += step
    
    return featured_df

#  =================  Custom featuers ==============
def spectrogram_based(center_time, t, f, Zxx, fmin=11, fmax=16):
    f_mask = (f >= fmin) & (f <= fmax)
    return Zxx[f_mask, _find_closest_value_idx(t, center_time)]