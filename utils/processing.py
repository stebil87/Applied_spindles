import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np
import math
from scipy.signal import butter, filtfilt, stft


def resample_signal(signal, original_fs, target_fs):
    duration = len(signal) / original_fs
    original_time = np.linspace(0, duration, len(signal), endpoint=False)
    resampled_time = np.linspace(0, duration, int(duration * target_fs), endpoint=False)
    
    if resampled_time[-1] > original_time[-1]:
        resampled_time = np.linspace(0, original_time[-1], int(original_time[-1] * target_fs), endpoint=False)
    interpolator = interp1d(original_time, signal, kind='cubic', bounds_error=False, fill_value="extrapolate") # np.resample was not used during the lab sessions
    resampled_signal = interpolator(resampled_time)
    return resampled_signal

def get_fft(signal, fs):
    # Compute the FFT
    n = len(signal)
    fft_result = np.fft.fft(signal)
    fft_freq = np.fft.fftfreq(n, 1/fs)

    # Compute the magnitudes of the FFT and keep the positive frequencies
    fft_magnitude = np.abs(fft_result)
    positive_freqs = fft_freq > 0

    return fft_freq[positive_freqs], fft_magnitude[positive_freqs]
    

def bandpass_filter(x, f_low, f_high, fs, n):
    b, a = butter(n, [f_low, f_high], btype="band", fs=fs)
    y = filtfilt(b, a, x)
    return y

def SNR_comp(signal, fs, noise_beg_time, noise_end_time):
    A = max(signal)-min(signal)
    noise = signal[math.ceil(noise_beg_time*fs):math.ceil(noise_end_time*fs)]
    noise_std = np.std(noise)
    SNR = 20*math.log10(A/(4*noise_std))
    return SNR

def plot_stft(signal, fs):
    t, f, Zxx = get_stft(signal, fs)
    
    # Plot the STFT as a spectrogram focusing on 0 to 35 Hz
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud', vmin=0, vmax=10, cmap="plasma")
    plt.hlines(y=[12, 14], xmin=np.min(t), xmax=np.max(t), colors=["r"])
    plt.title('STFT Magnitude Spectrogram (0-35 Hz)')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar(label='Magnitude')
    plt.tight_layout()
    plt.show()
    
def get_stft(signal, fs, window='hann', nperseg=200, noverlap=None):
    """
    Compute and plot the Short-Time Fourier Transform (STFT) of a signal with frequency range limited to 0-35 Hz.

    Parameters:
    - signal: 1D numpy array or list, the input signal.
    - fs: int or float, the sampling frequency of the signal.
    - window: string or tuple or array_like, desired window to use. Defaults to 'hann'.
    - nperseg: int, length of each segment. Defaults to 256.
    - noverlap: int, number of points to overlap between segments. If None, noverlap = nperseg // 2 is used.
    """
    
    if noverlap is None:
        noverlap = nperseg // 2
    # Compute the STFT
    f, t, Zxx = stft(signal, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap)

    # Find the indices of the frequency range 0 to 35 Hz
    freq_indices = (f >= 0) & (f <= 35)
    return t, f[freq_indices], np.abs(Zxx[freq_indices, :])
