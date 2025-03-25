import numpy as np
import pywt
import matplotlib.pyplot as plt

# Parameters
frequency = 500  # Frequency of the sine wave in Hz
processing_sampling_rate = 44100  # Sampling rate for processing and playback
duration = 1  # Duration of the signal in seconds
noise_level = 0.1  # Noise level
decomposition_level = 30  # Decomposition level
thresholds = np.linspace(0, 1, 60)  # Threshold values for analysis

# Generate time array for processing
t_processing = np.linspace(0, duration, int(processing_sampling_rate * duration), endpoint=False)

# Generate a sine wave
sine_wave = 0.5*np.sin(2 * np.pi * frequency * t_processing)

# Add Gaussian white noise
np.random.seed(0)  # For reproducibility
noise = noise_level * np.random.randn(len(sine_wave))
sine_wave_with_noise = sine_wave + noise

#affichage de la sinusoide bruitée
plt.figure(figsize=(12, 8))
plt.plot(t_processing, sine_wave_with_noise)
plt.title('Sine Wave with Noise')
plt.show()

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

#Analyze with different wavelets 
wavelets = pywt.wavelist(kind='discrete')
for wavelet in ['haar','db4','dmey','coif17']:
        SNR = []
        for k in range(20):
            reconstructed_signal = filter_and_reconstruct(sine_wave_with_noise, wavelet, 0.4)
            reconstructed_noise = reconstructed_signal - sine_wave
            snr_ratio = np.sum(original_noise**2) / np.sum(reconstructed_noise**2)
            SNR.append(snr_ratio)
        plt.plot(t_processing, reconstructed_signal)
        plt.title('reconstructed Sine Wave' + str(wavelet))
        plt.show()
        print("moyenne du SNR : " + str(np.mean(np.array(SNR))))
        print("écart type du SNR : "  + str(np.mean(np.std(SNR))))

# Analyze different thresholds
thresholds = np.linspace(-3, 3, 60)
SNR_TOT = []
for threshold in thresholds:
        SNR = []
        SNR_STD = []
        for k in range(20):
            reconstructed_signal = filter_and_reconstruct(sine_wave_with_noise, 'db4', threshold)
            reconstructed_noise = reconstructed_signal - sine_wave
            snr_ratio = np.sum(original_noise**2) / np.sum(reconstructed_noise**2)
            SNR.append(snr_ratio)
        SNR_TOT.append(np.mean(SNR))
        SNR_STD.append(np.std(SNR))
plt.errorbar(thresholds, SNR_TOT,SNR_STD)
plt.title('SNR ratio for different thresholds, db4 wavelet')
plt.show()
SNR_TOT = []
for threshold in thresholds:
        SNR = []
        SNR_STD = []
        for k in range(20):
            reconstructed_signal = filter_and_reconstruct(sine_wave_with_noise, 'coif4', threshold)
            reconstructed_noise = reconstructed_signal - sine_wave
            snr_ratio = np.sum(original_noise**2) / np.sum(reconstructed_noise**2)
            SNR.append(snr_ratio)
        SNR_TOT.append(np.mean(SNR))
        SNR_STD.append(np.std(SNR))
plt.errorbar(thresholds, SNR_TOT,SNR_STD)
plt.title('SNR ratio for different thresholds, coif4 wavelet')
plt.show()
SNR_TOT = []
for threshold in thresholds:
        SNR = []
        SNR_STD = []
        for k in range(20):
            reconstructed_signal = filter_and_reconstruct(sine_wave_with_noise, 'dmey', threshold)
            reconstructed_noise = reconstructed_signal - sine_wave
            snr_ratio = np.sum(original_noise**2) / np.sum(reconstructed_noise**2)
            SNR.append(snr_ratio)
        SNR_TOT.append(np.mean(SNR))
        SNR_STD.append(np.std(SNR))
plt.errorbar(thresholds, SNR_TOT,SNR_STD)
plt.title('SNR ratio for different thresholds, dmey wavelet')
plt.show()
SNR_TOT = []
for threshold in thresholds:
        SNR = []
        SNR_STD = []
        for k in range(20):
            reconstructed_signal = filter_and_reconstruct(sine_wave_with_noise, 'haar', threshold)
            reconstructed_noise = reconstructed_signal - sine_wave
            snr_ratio = np.sum(original_noise**2) / np.sum(reconstructed_noise**2)
            SNR.append(snr_ratio)
        SNR_TOT.append(np.mean(SNR))
        SNR_STD.append(np.std(SNR))
plt.errorbar(thresholds, SNR_TOT,SNR_STD)
plt.title('SNR ratio for different thresholds, haar wavelet')
plt.show()



def filter_and_reconstruct_bis(signal, wavelet, threshold,decomp_level):
    coefficients = pywt.wavedec(signal, wavelet,level = decomp_level)
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


# Analyze different decomposition levels
SNR_TOT = []
for decomp in range(1,31):
        SNR = []
        SNR_STD = []
        for k in range(20):
            reconstructed_signal = filter_and_reconstruct_bis(sine_wave_with_noise, 'db4', 0.4,decomp)
            reconstructed_noise = reconstructed_signal - sine_wave
            snr_ratio = np.sum(original_noise**2) / np.sum(reconstructed_noise**2)
            SNR.append(snr_ratio)
        SNR_TOT.append(np.mean(SNR))
        SNR_STD.append(np.std(SNR))
plt.errorbar([k for k in range(1,31)], SNR_TOT,SNR_STD)
plt.title('SNR ratio for different decomposition levels, db4 wavelet')
plt.show()
SNR_TOT = []
for decomp in range(1,31):
        SNR = []
        SNR_STD = []
        for k in range(20):
            reconstructed_signal = filter_and_reconstruct_bis(sine_wave_with_noise, 'haar', 0.4,decomp)
            reconstructed_noise = reconstructed_signal - sine_wave
            snr_ratio = np.sum(original_noise**2) / np.sum(reconstructed_noise**2)
            SNR.append(snr_ratio)
        SNR_TOT.append(np.mean(SNR))
        SNR_STD.append(np.std(SNR))
plt.errorbar([k for k in range(1,31)], SNR_TOT,SNR_STD)
plt.title('SNR ratio for different decomposition levels, haar wavelet')
plt.show()
# Analyze different noise variance
variances = np.linspace(0, 0.5, 60)
SNR_TOT = []
SNR_STD = []
for variance in variances:
        SNR = []
        SNR_STD = []
        for k in range(20):
            # Generate a sine wave
            sine_wave = 0.5*np.sin(2 * np.pi * frequency * t_processing)

            # Add Gaussian white noise
            np.random.seed(0)  # For reproducibility
            noise = variance * np.random.randn(len(sine_wave))
            sine_wave_with_noise = sine_wave + noise    
            reconstructed_signal = filter_and_reconstruct(sine_wave_with_noise, 'db4', 0.4)
            reconstructed_noise = reconstructed_signal - sine_wave
            snr_ratio = np.sum(original_noise**2) / np.sum(reconstructed_noise**2)
            SNR.append(snr_ratio)
        SNR_TOT.append(np.mean(SNR))
        SNR_STD.append(np.std(SNR))
plt.errorbar(variances,SNR_TOT,SNR_STD)
plt.title("SNR ratio for different noise variances, db4 wavelet")
plt.show()

# Analyze all wavelets and thresholds to find the best SNR ratio
 # Generate a sine wave
sine_wave = 0.5*np.sin(2 * np.pi * frequency * t_processing)

# Add Gaussian white noise
np.random.seed(0)  # For reproducibility
noise = 0.1 * np.random.randn(len(sine_wave))
sine_wave_with_noise = sine_wave + noise 
thresholds = np.linspace(0, 1, 60)
wavelets = pywt.wavelist(kind='discrete')
best_ratio = -np.inf
best_wavelet = None
best_threshold = None

for wavelet in wavelets:
    for threshold in thresholds:
            for level in range(1,31):
                reconstructed_signal = filter_and_reconstruct_bis(sine_wave_with_noise, wavelet, threshold,level)
                reconstructed_noise = reconstructed_signal - sine_wave
                snr_ratio = np.sum(original_noise**2) / np.sum(reconstructed_noise**2)

                if snr_ratio > best_ratio:
                    best_ratio = snr_ratio
                    best_wavelet = wavelet
                    best_threshold = threshold
                    best_level = level


# Output the best result
print(f"Best SNR Ratio: {best_ratio}")
print(f"Achieved with Wavelet: {best_wavelet}")
print(f"Achieved with Threshold: {best_threshold}")
print(f"Achieved with level: {best_level}")
#affichage de la sinusoide reconstriute
plt.figure(figsize=(12, 8))
plt.plot(t_processing, filter_and_reconstruct_bis(sine_wave_with_noise, best_wavelet, best_threshold))
plt.title('reconstructed Sine Wave')
plt.show()
