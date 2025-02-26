import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.signal import butter, filtfilt, hilbert

# Function to apply a Butterworth filter
def butter_filter(signal, cutoff, fs, filter_type, order=4):
    nyq = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyq
    if normal_cutoff >= 1.0:  # Ensure valid cutoff frequency
        normal_cutoff = 0.99
    b, a = butter(order, normal_cutoff, btype=filter_type, analog=False)
    return filtfilt(b, a, signal)

# Function for downsampling
def downsample(signal, factor):
    return signal[::factor]

# Function to compute instantaneous frequency using Hilbert transform
def compute_instantaneous_frequency(analytic_signal, fs):
    phase = np.unwrap(np.angle(analytic_signal))  # Unwrap phase
    inst_freq = np.diff(phase) / (2.0 * np.pi) * fs  # Compute derivative
    return inst_freq

# Signal parameters
f0 = 880  # Fundamental frequency in Hz
harmonics = 4  # Number of harmonics
amplitudes = [0.1, 0.2, 0.3, 0.4]  # Amplitudes of the harmonics
fe = 8000  # Sampling frequency in Hz
T = 2  # Duration in seconds
fc1 = fe / 4  # First cutoff frequency (fe/4)
fc2 = fe / 8  # Second cutoff frequency (fe/8)

# Time vector
t = np.linspace(0, T, int(fe * T), endpoint=False)

# Construct the harmonic signal
signal = sum(amplitudes[k] * np.sin(2 * np.pi * (k + 1) * f0 * t) for k in range(harmonics))

# Normalize the signal
signal = signal / np.max(np.abs(signal))

# Apply low-pass and high-pass filters (fc = fe/4)
low_pass_signal = butter_filter(signal, fc1, fe, 'low')
high_pass_signal = butter_filter(signal, fc1, fe, 'high')

# Downsample both filtered signals (N=2)
low_pass_downsampled = downsample(low_pass_signal, 2)
high_pass_downsampled = downsample(high_pass_signal, 2)
t_down = t[::2]

# Apply low-pass and high-pass filters (fc = fe/8) on the downsampled signals
low_pass_filtered_down = butter_filter(low_pass_downsampled, fc2, fe // 2, 'low')
high_pass_filtered_down = butter_filter(low_pass_downsampled, fc2, fe // 2, 'high')
low_pass_filtered_high = butter_filter(high_pass_downsampled, fc2, fe // 2, 'low')
high_pass_filtered_high = butter_filter(high_pass_downsampled, fc2, fe // 2, 'high')

# Downsample again (N=2)
low_pass_filtered_down_ds = downsample(low_pass_filtered_down, 2)
high_pass_filtered_down_ds = downsample(high_pass_filtered_down, 2)
low_pass_filtered_high_ds = downsample(low_pass_filtered_high, 2)
high_pass_filtered_high_ds = downsample(high_pass_filtered_high, 2)
t_down_2 = t_down[::2]

# Apply Hilbert transform to obtain the analytic signals
analytic_low_pass_filtered_down_ds = hilbert(low_pass_filtered_down_ds)
analytic_high_pass_filtered_down_ds = hilbert(high_pass_filtered_down_ds)
analytic_low_pass_filtered_high_ds = hilbert(low_pass_filtered_high_ds)
analytic_high_pass_filtered_high_ds = hilbert(high_pass_filtered_high_ds)

# Compute the instantaneous frequencies
inst_freq_low_pass_filtered_down_ds = compute_instantaneous_frequency(analytic_low_pass_filtered_down_ds, fe // 4)
inst_freq_high_pass_filtered_down_ds = compute_instantaneous_frequency(analytic_high_pass_filtered_down_ds, fe // 4)
inst_freq_low_pass_filtered_high_ds = compute_instantaneous_frequency(analytic_low_pass_filtered_high_ds, fe // 4)
inst_freq_high_pass_filtered_high_ds = compute_instantaneous_frequency(analytic_high_pass_filtered_high_ds, fe // 4)

# Compute the average frequency for each signal
avg_freq_low_pass_filtered_down_ds = np.mean(inst_freq_low_pass_filtered_down_ds)
avg_freq_high_pass_filtered_down_ds = np.mean(inst_freq_high_pass_filtered_down_ds)
avg_freq_low_pass_filtered_high_ds = np.mean(inst_freq_low_pass_filtered_high_ds)
avg_freq_high_pass_filtered_high_ds = np.mean(inst_freq_high_pass_filtered_high_ds)

# Display the results
frequencies = {
    "Low-Pass Filtered Downsampled Low-Pass Signal (fc = fe/8)": avg_freq_low_pass_filtered_down_ds,
    "High-Pass Filtered Downsampled Low-Pass Signal (fc = fe/8)": avg_freq_high_pass_filtered_down_ds,
    "Low-Pass Filtered Downsampled High-Pass Signal (fc = fe/8)": avg_freq_low_pass_filtered_high_ds,
    "High-Pass Filtered Downsampled High-Pass Signal (fc = fe/8)": avg_freq_high_pass_filtered_high_ds,
}

# Print frequencies
for key, value in frequencies.items():
    print(f"{key}: {value:.2f} Hz")

# Plot the final downsampled signals
plt.figure(figsize=(12, 8))

# Downsampled low-pass filtered downsampled low-pass signal
plt.subplot(4, 1, 1)
plt.plot(t_down_2[:250], low_pass_filtered_down_ds[:250])
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Downsampled Low-Pass Filtered Downsampled Low-Pass Signal (fc = fe/8)")
plt.grid()

# Downsampled high-pass filtered downsampled low-pass signal
plt.subplot(4, 1, 2)
plt.plot(t_down_2[:250], high_pass_filtered_down_ds[:250])
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Downsampled High-Pass Filtered Downsampled Low-Pass Signal (fc = fe/8)")
plt.grid()

# Downsampled low-pass filtered downsampled high-pass signal
plt.subplot(4, 1, 3)
plt.plot(t_down_2[:250], low_pass_filtered_high_ds[:250])
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Downsampled Low-Pass Filtered Downsampled High-Pass Signal (fc = fe/8)")
plt.grid()

# Downsampled high-pass filtered downsampled high-pass signal
plt.subplot(4, 1, 4)
plt.plot(t_down_2[:250], high_pass_filtered_high_ds[:250])
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Downsampled High-Pass Filtered Downsampled High-Pass Signal (fc = fe/8)")
plt.grid()

plt.tight_layout()
plt.show()
