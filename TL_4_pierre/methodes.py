import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import OrthogonalMatchingPursuit

def extraire_et_afficher_omp_fourier_zeropad(fichier_mp3, frequence_echantillonnage=22100, taille_trame=1024, saut=256, zero_padding=1024):
    # Lire le fichier MP3
    y, sr = librosa.load(fichier_mp3, sr=frequence_echantillonnage)

    # Extraire les trames avec chevauchement
    trames = librosa.util.frame(y, frame_length=taille_trame, hop_length=saut).T

    # Ajouter du zero-padding aux trames
    taille_trame_zeropad = taille_trame + zero_padding
    trames_zeropad = np.zeros((trames.shape[0], taille_trame_zeropad))
    trames_zeropad[:, :taille_trame] = trames

    # Créer un dictionnaire de Fourier avec zero-padding
    t = np.arange(taille_trame_zeropad)
    D = np.zeros((taille_trame_zeropad, taille_trame_zeropad), dtype=np.complex128)
    for k in range(taille_trame_zeropad):
        D[:, k] = np.exp(2j * np.pi * k * t / taille_trame_zeropad)

    # Initialiser OMP
    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=10)

    # Appliquer OMP à chaque trame
    coefficients = []
    for trame in trames_zeropad:
        omp.fit(np.real(D), trame)
        coefficients.append(omp.coef_)

    coefficients = np.array(coefficients)

    # Afficher les coefficients obtenus
    plt.figure(figsize=(12, 6))
    plt.imshow(np.abs(coefficients.T), aspect='auto', origin='lower', cmap='viridis')
    plt.title('Coefficients OMP avec atomes de Fourier et Zero-Padding')
    plt.xlabel('Trame')
    plt.ylabel('Fréquence')
    plt.colorbar(label='Amplitude')
    plt.show()

    return trames

# Exemple d'utilisation
fichier_mp3 = '/home/pierres/ST7/traitement_audio_parcimonieux/TL_4_pierre/a.mp3'
trames = extraire_et_afficher_omp_fourier_zeropad(fichier_mp3)

# Afficher la forme des trames extraites
print("Forme des trames extraites :", trames.shape)
