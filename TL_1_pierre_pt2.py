import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, butter, filtfilt
from scipy.fft import fft, fftfreq
import sounddevice as sd  

# Parameters
f0 = 880  # Fundamental frequency in Hz
A = 20    # Vibrato amplitude in Hz
fm = 5    # Vibrato frequency in Hz
fs = 8800 # Sampling frequency in Hz
T = 1     # Signal duration in seconds
N = 4     # Number of harmonics
amp = [0.1, 0.2, 0.3, 0.4]

# Time vector
t = np.linspace(0, T, int(fs * T), endpoint=False)
signal = np.zeros_like(t)

# Generate harmonic signal with vibrato (Modified A/fm)
for n in range(1, N + 1):
    phi_n = 2 * n * np.pi * f0 * t + (A / fm) * n * np.cos(2 * np.pi * fm * t)
    signal += amp[n-1] * np.sin(phi_n)

# Normalize the signal
signal /= np.max(np.abs(signal))

# **Compute the spectrogram**
f, t_spec, Sxx = spectrogram(signal, fs, nperseg=256, noverlap=128, nfft=1024, mode='magnitude')

# Convert to dB scale
Sxx_dB = 10 * np.log10(Sxx + 1e-12)  # Avoid log(0) issues

# Set dynamic range: from 0 dB (max) to -60 dB
vmax = np.max(Sxx_dB)
vmin = vmax - 60

# **Plot Spectrogram**
plt.figure(figsize=(10, 6))
plt.pcolormesh(t_spec, f, Sxx_dB, shading='gouraud', cmap='jet', vmin=vmin, vmax=vmax)
plt.colorbar(label='Amplitude (dB)')
plt.title("Spectrogram of the Signal")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.ylim(0, fs / 2)
plt.show()

# **Find peak frequency in 700-1100 Hz range**
freq_indices = np.where((f >= 700) & (f <= 1100))[0]

# Find the peak frequency in this range for each time step
peak_frequencies = []
for i in range(len(t_spec)):
    freq_slice = Sxx_dB[freq_indices, i]  # Extract power in the desired range
    max_index = np.argmax(freq_slice)     # Find the max amplitude index
    peak_frequencies.append(f[freq_indices[max_index]])  # Store corresponding frequency

# **Apply High-Pass Filter (Cutoff = 4 Hz)**
def highpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, data)

# Compute Fourier Transform BEFORE filtering
N_fft = len(peak_frequencies)
frequencies_fft = fftfreq(N_fft, d=(t_spec[1] - t_spec[0]))  # Frequency axis
fft_magnitude_before = np.abs(fft(peak_frequencies))  # Magnitude spectrum

# Apply filter
filtered_peak_frequencies = highpass_filter(peak_frequencies, cutoff=4, fs=1/(t_spec[1] - t_spec[0]))

# Compute Fourier Transform AFTER filtering
fft_magnitude_after = np.abs(fft(filtered_peak_frequencies))  # Magnitude spectrum

# **Plot the two Fourier Transform graphs side by side**
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

ax[0].plot(frequencies_fft[:N_fft//2], fft_magnitude_before[:N_fft//2], 'b')
ax[0].set_title("Fourier Transform BEFORE Filtering")
ax[0].set_xlabel("Frequency (Hz)")
ax[0].set_ylabel("Magnitude")
ax[0].grid()

ax[1].plot(frequencies_fft[:N_fft//2], fft_magnitude_after[:N_fft//2], 'r')
ax[1].set_title("Fourier Transform AFTER Filtering (4 Hz)")
ax[1].set_xlabel("Frequency (Hz)")
ax[1].set_ylabel("Magnitude")
ax[1].grid()

plt.tight_layout()
plt.show()

# **Play the signal**
sd.play(signal, samplerate=fs)
sd.wait()  # Wait for the sound to finish playing
