import os
import wave
import numpy as np
import csv

def obtenir_frequence_et_amplitude(chemin_fichier, zero_padding_factor=100):
    # Ouvrir le fichier WAV
    with wave.open(chemin_fichier, 'rb') as fichier_wav:
        # Lire les paramètres du fichier WAV
        n_canaux = fichier_wav.getnchannels()
        largeur_echantillon = fichier_wav.getsampwidth()
        frequence_echantillonnage = fichier_wav.getframerate()
        n_echantillons = fichier_wav.getnframes()

        # Lire les échantillons audio
        donnees_audio = fichier_wav.readframes(n_echantillons)
        donnees_audio = np.frombuffer(donnees_audio, dtype=np.int16)

    # Appliquer le zero-padding
    #n_echantillons_padded = n_echantillons * zero_padding_factor
    #donnees_audio_padded = np.zeros(n_echantillons_padded, dtype=donnees_audio.dtype)
    #donnees_audio_padded[:n_echantillons] = donnees_audio

    # Calculer la transformée de Fourier
    fft_result = np.fft.fft(donnees_audio, n_fft=2**20)
    frequences = np.fft.fftfreq(len(fft_result), 1 / frequence_echantillonnage)

    # Trouver la fréquence avec l'amplitude maximale
    amplitude_max_index = np.argmax(np.abs(fft_result[:len(frequences)//2]))
    frequence_fondamentale = frequences[amplitude_max_index]
    amplitude_max = np.abs(fft_result[amplitude_max_index])

    return frequence_fondamentale, amplitude_max

def analyser_dossier(chemin_dossier, chemin_csv):
    # Créer le répertoire s'il n'existe pas
    os.makedirs(os.path.dirname(chemin_csv), exist_ok=True)

    # Lister tous les fichiers dans le dossier
    fichiers = [f for f in os.listdir(chemin_dossier) if f.endswith('.wav')]

    # Ouvrir le fichier CSV pour écrire les résultats
    with open(chemin_csv, mode='w', newline='') as fichier_csv:
        writer = csv.writer(fichier_csv)
        # Écrire l'en-tête du CSV
        writer.writerow(["Frequence Fondamentale (Hz)", "Amplitude Maximale"])

        # Analyser chaque fichier WAV
        for fichier in fichiers:
            chemin_fichier = os.path.join(chemin_dossier, fichier)
            frequence_fondamentale, amplitude_max = obtenir_frequence_et_amplitude(chemin_fichier)
            # Écrire les résultats dans le CSV
            writer.writerow([frequence_fondamentale, amplitude_max])

def traiter_oiseaux(base_path, sons_folder, features_folder, oiseaux):
    for oiseau in oiseaux:
        chemin_dossier = os.path.join(base_path, sons_folder, oiseau)
        chemin_csv = os.path.join(base_path, features_folder, oiseau, 'frequences_amplitudes.csv')
        analyser_dossier(chemin_dossier, chemin_csv)

# Spécifiez le chemin de base et les configurations
base_path = '/home/pierres/ST7/traitement_audio_parcimonieux/TL_3_pierre'
oiseaux = ['oiseau1']

# Traiter les configurations SONS et SONS_VC
#traiter_oiseaux(base_path, 'SONS', 'SONS_features', oiseaux)
#traiter_oiseaux(base_path, 'SONS-VC', 'SONS_VC_features', oiseaux)

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preparer_donnees(base_path, features_folder, oiseaux):
    data = []
    labels = []

    for i, oiseau in enumerate(oiseaux):
        chemin_csv = os.path.join(base_path, features_folder, oiseau, 'frequences_amplitudes.csv')
        # Lire le fichier CSV
        df = pd.read_csv(chemin_csv)

        # Ajouter les données et les étiquettes
        data.append(df)
        labels.append([i] * len(df))

    # Concaténer les données et les étiquettes
    data = pd.concat(data, ignore_index=True)
    labels = [item for sublist in labels for item in sublist]

    return data, labels

def creer_base_de_donnees(base_path, oiseaux):
    # Préparer les données d'entraînement
    X_train, y_train = preparer_donnees(base_path, 'SONS_features', oiseaux)

    # Préparer les données de test
    X_test, y_test = preparer_donnees(base_path, 'SONS_VC_features', oiseaux)

    # Normaliser les données
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Retourner les ensembles d'entraînement et de test
    return X_train, X_test, y_train, y_test

def sauvegarder_base_de_donnees(base_path, oiseaux, output_folder):
    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs(output_folder, exist_ok=True)

    # Créer la base de données
    X_train, X_test, y_train, y_test = creer_base_de_donnees(base_path, oiseaux)

    # Sauvegarder les ensembles d'entraînement et de test dans des fichiers CSV
    pd.DataFrame(X_train).to_csv(os.path.join(output_folder, 'X_SONS.csv'), index=False)
    pd.DataFrame(X_test).to_csv(os.path.join(output_folder, 'X_SONS-VC.csv'), index=False)
    pd.DataFrame(y_train, columns=['Label']).to_csv(os.path.join(output_folder, 'y_SONS.csv'), index=False)
    pd.DataFrame(y_test, columns=['Label']).to_csv(os.path.join(output_folder, 'y_SONS-VC.csv'), index=False)

# Spécifiez le chemin de base et les configurations
base_path = '/home/pierres/ST7/traitement_audio_parcimonieux/TL_3_pierre'
oiseaux = ['oiseau1', 'oiseau2', 'oiseau3']
output_folder = '/home/pierres/ST7/traitement_audio_parcimonieux/TL_3_pierre/database_finale'

# Sauvegarder la base de données finale
sauvegarder_base_de_donnees(base_path, oiseaux, output_folder)
