import numpy as np
import matplotlib.pyplot as plt
import os
import soundfile as sf
import pywt
import time
import sounddevice as sd
from scipy.signal import resample
from concurrent.futures import ThreadPoolExecutor


def create_wavelet_dictionary(signal_length,sr):
    wavelet_names = ['sym20']
    dict_matrix = []
    for wavelet_name in wavelet_names:
        wavelet = pywt.Wavelet(wavelet_name)
        wavelet_function = wavelet.wavefun(level=3)
        for ordre in range(len(wavelet_function)-2):
            for pad in range(signal_length - len(wavelet_function[ordre])):
                padded_wavelet = np.pad(wavelet_function[ordre], (pad, signal_length - len(wavelet_function[ordre]) - pad), mode='constant')
                dict_matrix.append(padded_wavelet / np.linalg.norm(padded_wavelet))
    #dct
    for k in range(signal_length):
        cos_k = np.cos([np.pi*(n + 1/2)* k / signal_length for n in range(signal_length)])
        dict_matrix.append(cos_k/np.linalg.norm(cos_k))
    return np.array(dict_matrix).T


def orthogonal_matching_pursuit(x, dictionary, max_iter, tol=1e-4):
    # Initialisation du résidu et des listes pour stocker les atomes sélectionnés
    r = x.copy()
    liste_proj = []  # Atomes bruts sélectionnés dans le dictionnaire
    liste_u = []     # Vecteurs orthonormaux issus de l'innovation
    for _ in range(max_iter):
        if np.sqrt(np.dot(r, r))/np.sqrt(np.dot(x, x)) < tol:
            break
        if np.sqrt(np.dot(r, r))/np.sqrt(np.dot(x, x)) > 1:
            print('RSB impossible')
        # Sélection du meilleur atome selon la corrélation avec le résidu
        projections = dictionary.T @ r
        k = np.argmax(np.abs(projections))
        meilleur_atome = dictionary[:, k]
        
        # Projection du résidu sur le meilleur atome
        proj_residu_sur_atome = np.dot(meilleur_atome, r) * meilleur_atome
        
        # Orthogonalisation de la projection par rapport aux vecteurs déjà sélectionnés
        if liste_u:
            proj_orthogonal = np.zeros_like(liste_u[0])
            for u in liste_u:
                proj_orthogonal += np.dot(proj_residu_sur_atome, u) * u
        else:
            proj_orthogonal = 0
        
        # Calcul du vecteur innovation et normalisation
        vecteur_innovation = proj_residu_sur_atome - proj_orthogonal
        norm_innovation = np.sqrt(np.dot(vecteur_innovation, vecteur_innovation))
        if norm_innovation == 0:
            break
        vecteur_orthogonal = vecteur_innovation / norm_innovation
        
        # Mise à jour du résidu par soustraction de sa composante sur le vecteur orthonormal
        r = r - np.dot(vecteur_orthogonal, r) * vecteur_orthogonal
        
        
        # Stockage de l'atome sélectionné et de son vecteur orthonormal associé
        liste_proj.append(meilleur_atome)
        liste_u.append(vecteur_orthogonal)
    
    # Construction de la matrice M reliant la base orthonormale aux atomes sélectionnés
    M = np.matmul(np.array(liste_u), np.array(liste_proj).T)
    # Construction du vecteur m contenant les corrélations du signal avec la base orthonormale
    m_vec = np.array([np.dot(x, u) for u in liste_u])
    
    # Résolution du système linéaire pour obtenir les coefficients finaux
    if M.shape[0] > 0:
        coeffs = np.linalg.solve(M, m_vec)
    else:
        coeffs = []
    
    # Reconstruction de l'approximation : somme pondérée (valeur absolue des coefficients)
    approx = np.sum([coeffs[i] * liste_proj[i] for i in range(len(coeffs))], axis=0)
    
    # Pour rester compatible, on renvoie aussi les indices (ici, les indices correspondant aux atomes sélectionnés)
    indices = [np.argmax(np.abs(dictionary.T @ atom)) for atom in liste_proj]
    
    return approx, coeffs, indices



def process_window(args):
    data, dictionary, window_size, step_size, max_iter, i = args
    x = data[i:i + window_size] * np.hanning(window_size)
    approx, coeffs, indices = orthogonal_matching_pursuit(x, dictionary, max_iter)
    return i, approx, len(coeffs)

def compress(data, dictionary, window_size, step_size, max_iter, verbose = True):
    t1 = time.time()
    signal_recomposed = np.zeros_like(data)
    n_coeffs = 0
    
    args_list = [(data, dictionary, window_size, step_size, max_iter, i) for i in range(0, len(data) - window_size, step_size)]
    
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_window, args_list))
    
    for i, approx, coeff_count in results:
        signal_recomposed[i:i + window_size] += approx
        n_coeffs += coeff_count


    plt.plot(data, label='Original')
    plt.plot(signal_recomposed, label='Recomposé')
    plt.legend()
    plt.show()
    
    print(f'RSB : {np.linalg.norm(data)/np.linalg.norm(data - signal_recomposed[:len(data)])}')
    print(f'Temps d’exécution : {time.time()-t1}')
    print(f'Taux de compression : {len(data) / n_coeffs}')

    sd.play(data, samplerate=sr)
    sd.wait()
    
    sd.play(signal_recomposed, samplerate=sr)
    sd.wait()

filepath = os.path.abspath('./audio_partie4/a.wav')
data, sr = sf.read(filepath)
data = resample(data, int(len(data) * 16000 / sr))
sr = 16000
window_size = 1024
step_size = 512
dictionary = create_wavelet_dictionary(window_size, sr)
compress(data, dictionary, window_size, step_size, max_iter=120)
