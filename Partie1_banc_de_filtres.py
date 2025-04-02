import numpy as np
import scipy.signal as sp_signal

def generate_signal(fs, duration=1.0):
    t = np.arange(0, duration, 1/fs)
    s = (np.cos(2*np.pi*880*t) +
         np.cos(2*np.pi*1760*t) +
         np.cos(2*np.pi*2640*t) +
         np.cos(2*np.pi*3520*t))
    return t, s

def apply_filter(sig, fs, lowcut=None, highcut=None, order=5):
    nyq = 0.5 * fs  # Fréquence de Nyquist
    if lowcut is not None:
        low = lowcut / nyq
    if highcut is not None:
        high = highcut / nyq

    if lowcut and highcut:
        b, a = sp_signal.butter(order, [low, high], btype='band')
    elif highcut:
        b, a = sp_signal.butter(order, high, btype='low')
    elif lowcut:
        b, a = sp_signal.butter(order, low, btype='high')
    else:
        raise ValueError("Il faut au moins une fréquence de coupure")

    return sp_signal.filtfilt(b, a, sig)

def downsample(sig, factor):
    return sig[::factor]

def extract_frequency(sig, fs):
    """ Utilise Hilbert pour extraire la fréquence moyenne """
    analytic_signal = sp_signal.hilbert(sig)
    phase = np.unwrap(np.angle(analytic_signal))
    freq_inst = np.diff(phase) / (2.0 * np.pi) * fs
    return np.median(freq_inst)  # On prend la médiane pour éviter le bruit

fs = 8000  # Fréquence d'échantillonnage initiale
T = 1.0    # Durée du signal
t, s = generate_signal(fs, T)

# Étape 1: Décomposition en basses et hautes fréquences
low_freqs = apply_filter(s, fs, highcut=1900)  # Passe-bas
high_freqs = apply_filter(s, fs, lowcut=2100, highcut=3900)  # Passe-bande

# Sous-échantillonnage à fe1 = 4000 Hz
low_freqs_ds = downsample(low_freqs, 2)
high_freqs_ds = downsample(high_freqs, 2)

# Étape 2: Nouvelle séparation
low_low = apply_filter(low_freqs_ds, 4000, highcut=900)  # Passe-bas
low_high = apply_filter(low_freqs_ds, 4000, lowcut=1100, highcut=1900)  # Passe-bande
high_low = apply_filter(high_freqs_ds, 4000, highcut=900)  # Passe-bas
high_high = apply_filter(high_freqs_ds, 4000, lowcut=1100, highcut=1900)  # Passe-bande

# Sous-échantillonnage à fe2 = 2000 Hz
low_low_ds = downsample(low_low, 2)
low_high_ds = downsample(low_high, 2)
high_low_ds = downsample(high_low, 2)
high_high_ds = downsample(high_high, 2)

# Extraction des fréquences moyennes
freq_low_low = extract_frequency(low_low_ds, 2000)
freq_low_high = extract_frequency(low_high_ds, 2000)
freq_high_low = extract_frequency(high_low_ds, 2000)
freq_high_high = extract_frequency(high_high_ds, 2000)

# Affichage des valeurs
print(f"Bande 1 : {freq_low_low:.2f} Hz (attendu ≈ 880 Hz)")
print(f"Bande 2 : {freq_low_high:.2f} Hz (attendu ≈ 240 Hz)")
print(f"Bande 3 : {freq_high_low:.2f} Hz (attendu ≈ 480 Hz)")
print(f"Bande 4 : {freq_high_high:.2f} Hz (attendu ≈ 640 Hz)")
