import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

###Question 1

# Paramètres
T = 1.0  # Durée du signal en secondes
fs = 8000  # Fréquence d'échantillonnage
f0 = 880  # Fréquence fondamentale
harmonics = [1, 2, 3, 4]  # Multiples harmoniques

t = np.linspace(0, T, int(fs * T), endpoint=False)
signal = sum(np.sin(2 * np.pi * f0 * h * t) for h in harmonics)

# Normalisation du signal + conversion échelle 16 bits
signal = (signal / np.max(np.abs(signal)) * 32767).astype(np.int16)

# Sauvegarde du signal dans un fichier WAV
wav.write("./Partie1_4sinus.wav", fs, signal)


# Affichage du signal temporel
plt.figure()
plt.plot(t[:1000], signal[:1000])  # Affichage des 1000 premiers points
plt.xlabel('Temps (s)')
plt.ylabel('Amplitude')
plt.title('Signal temporel')
plt.show()

# Analyse spectrale
f, time, Sxx = spectrogram(signal, fs, window = "hamming", nfft = 2**18)
plt.pcolormesh(time, f, 10 * np.log10(Sxx))
plt.ylabel('Fréquence (Hz)')
plt.xlabel('Temps (s)')
plt.title('Spectrogramme du signal harmonique')
plt.colorbar(label='dB')
plt.show()

###Question 2 - Ajout du vibrato
# Paramètres
Av = 200    # Amplitude du vibrato
fv = 5     # Fréquence du vibrato

# Génération du signal avec 4 harmoniques et vibrato
signal_vibrato = sum(np.sin(2 * np.pi * (h * f0 * t - (Av / (2 * np.pi * fv)) * np.cos(2 * np.pi * fv * t))) for h in harmonics)

# Normalisation et conversion en int16 pour WAV
signal_vibrato = (signal_vibrato / np.max(np.abs(signal_vibrato)) * 32767).astype(np.int16)

# Sauvegarde du fichier audio
wav.write("./Partie1_vibrato.wav", fs, signal_vibrato)

# Affichage du signal temporel
plt.figure()
plt.plot(t[:1000], signal_vibrato[:1000])  # Affichage des 1000 premiers points
plt.xlabel("Temps (s)")
plt.ylabel("Amplitude")
plt.title("Signal avec vibrato")
plt.show()

# Calcul du spectrogramme
f, time, Sxx = spectrogram(signal_vibrato, fs, window = "hamming", nfft = 2**15)

# Affichage du spectrogramme
plt.figure()
plt.pcolormesh(time, f, 10 * np.log10(Sxx), shading='gouraud')
plt.xlabel("Temps (s)")
plt.ylabel("Fréquence (Hz)")
plt.title("Spectrogramme du signal avec vibrato")
plt.colorbar(label="Amplitude (dB)")
plt.show()