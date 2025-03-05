import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy import interpolate



def pad_symmetric(arr, target_length):
    current_length = len(arr)
    if current_length >= target_length:
        return arr  # Rien à faire si le tableau est déjà assez grand

    # Nombre total de zéros à ajouter
    total_padding = target_length - current_length
    
    # Division en deux parts aussi égales que possible
    left_pad = total_padding // 2
    right_pad = total_padding - left_pad  # Assure que left + right = total_padding

    # Appliquer le padding avec np.pad
    padded_array = np.pad(arr, (left_pad, right_pad), mode='constant', constant_values=-float('inf'))

    return padded_array


# Paramètres
fs = 44000 # Fréquence d'échantillonnage (1000 Hz)
N=18 
T=2  # Durée du signal (2 secondes)
t = np.linspace(0, T, int(T*fs))  # Temps de 1 seconde
f_sinus = 500  # Fréquence du sinus (500 Hz)
amplitude_sinus = 0.5  # Amplitude du sinus
wave = 'db35'  # Type d'ondelette

# Génération du signal : un sinus + bruit blanc
signal_sinus = amplitude_sinus * np.sin(2 * np.pi * f_sinus * t)
bruit = np.random.normal(0, 0.1, t.shape)  # Bruit blanc avec écart-type de 0.5
signal_bruite = signal_sinus + bruit

# Application de la DWT (décomposition en 3 niveaux)
coeffs = pywt.wavedec(signal_bruite, wave, level=30)

# Afficher les coefficients ligne par ligne
G = [len(c) for c in coeffs]
maxsize = max(G)
img = []
threshold = 0.2
coeffs_detail = coeffs[1:]
for i, c in enumerate(coeffs_detail):
    c = pywt.threshold(np.abs(c), threshold, mode='soft')

    interpolated = interpolate.interp1d(np.linspace(0, 1, len(c)), c, kind='previous')
    c_ = interpolated(np.linspace(0, 1, maxsize))
    c_padded = pad_symmetric(c, maxsize)
    img.append(c_)

plt.figure(figsize=(12, 10))
plt.imshow(img, cmap='viridis', aspect='auto', interpolation='nearest')
plt.colorbar()
plt.show()

# Seuil de réduction de bruit
coeffs_thresholded = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]

# Reconstruction du signal à partir des coefficients modifiés
signal_reduit = pywt.waverec(coeffs_thresholded, wave)

# Plot the reconstructed signal
plt.plot(t, signal_reduit[:len(t)], label='Reconstructed Signal')
plt.plot(t, signal_bruite, label='Noisy Signal', alpha=0.5)
plt.legend()
plt.show()


#RSB
threshold_values = np.linspace(0.0001, 1, 100)
ratios = []
for i in range(len(threshold_values)):
    coeffs_thresholded = [pywt.threshold(c, threshold_values[i], mode='soft') for c in coeffs]
    signal_reduit = pywt.waverec(coeffs_thresholded, wave)
    erreur1=sum((signal_sinus-signal_bruite)**2)/np.sum(signal_sinus**2)
    erreur2=sum((signal_sinus-signal_reduit)**2)/sum(signal_sinus**2)
    ratio=erreur1/erreur2
    ratios.append(ratio)

plt.plot(threshold_values, ratios)
plt.xlabel('Threshold')
plt.ylabel('RSB')
plt.show()