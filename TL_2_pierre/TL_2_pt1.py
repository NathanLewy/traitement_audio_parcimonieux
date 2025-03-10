import numpy as np
import pywt
import matplotlib.pyplot as plt

# Parameters
frequency = 500  # Frequency of the sine wave in Hz
processing_sampling_rate = 44100  # Sampling rate for processing and playback
duration = 1  # Duration of the signal in seconds
noise_level = 0.1  # Noise level
decomposition_level = 30  # Decomposition level
thresholds = np.linspace(0, 1, 30)  # Threshold values for analysis

# Generate time array for processing
t_processing = np.linspace(0, duration, int(processing_sampling_rate * duration), endpoint=False)

# Generate a sine wave
sine_wave = 0.5*np.sin(2 * np.pi * frequency * t_processing)

# Add Gaussian white noise
np.random.seed(0)  # For reproducibility
noise = noise_level * np.random.randn(len(sine_wave))
sine_wave_with_noise = sine_wave + noise

# Compute the original noise
original_noise = sine_wave_with_noise - sine_wave

# Function to filter and reconstruct the signal using pywt.threshold
def filter_and_reconstruct(signal, wavelet, threshold):
    coefficients = pywt.wavedec(signal, wavelet, level=decomposition_level)
    detail_coefficients = coefficients[1:]

    # Apply thresholding to detail coefficients
    filtered_coefficients = [coefficients[0]] + [
        pywt.threshold(detail_coefficients[i], value=threshold, mode='hard')
        for i in range(len(detail_coefficients))
    ]

    # Reconstruct the signal from filtered coefficients
    reconstructed_signal = pywt.waverec(filtered_coefficients, wavelet)
    reconstructed_signal = reconstructed_signal[:len(signal)]

    return reconstructed_signal

# Analyze all wavelets
wavelets = pywt.wavelist(kind='discrete')
best_ratio = -np.inf
best_wavelet = None
best_threshold = None

for wavelet in wavelets:
    for threshold in thresholds:
        reconstructed_signal = filter_and_reconstruct(sine_wave_with_noise, wavelet, threshold)
        reconstructed_noise = reconstructed_signal - sine_wave
        snr_ratio = np.sum(original_noise**2) / np.sum(reconstructed_noise**2)

        if snr_ratio > best_ratio:
            best_ratio = snr_ratio
            best_wavelet = wavelet
            best_threshold = threshold

# Output the best result
print(f"Best SNR Ratio: {best_ratio}")
print(f"Achieved with Wavelet: {best_wavelet}")
print(f"Achieved with Threshold: {best_threshold}")
