import numpy as np
import matplotlib.pyplot as plt
import os
import soundfile as sf
import pywt
import time
import sounddevice as sd
from scipy.signal import resample


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


def matching_pursuit(x, dictionary, max_iter, tol=1e-7, verbose=False):
    t1 = time.time()
    residual_list = [np.linalg.norm(x)]
    approx = np.zeros_like(x)
    coeffs = []
    indices = []
    
    for i in range(max_iter):
        projections = dictionary.T @ (x - approx)
        k = np.argmax(np.abs(projections))  # Atome le plus corrélé
        a_k = projections[k]
        
        
        # Mise à jour
        approx += a_k * dictionary[:, k]
        residual_list.append(np.linalg.norm(x - approx))
        
        coeffs.append(a_k)
        indices.append(k)
        
        if np.linalg.norm(x - approx) < tol:
            print(f"Arrêt à l'itération {i} : résiduel inférieur à la tolérance.")
            break
    
    if verbose:
        t2 = time.time()
        print(f"Arrêt à l'itération     : {i}")
        print(f'norme du résidu         : {np.linalg.norm(x - approx)}')
        print(f'RSB                     : {np.linalg.norm(x)/np.linalg.norm(x - approx)}')
        print(f'tolérance du résidu     : {tol}')
        print(f'taille du dictionnaire  : {len(dictionary[0])}')
        print(f'temps d execution       : {t2-t1}')
        print(f'taux de compression     : {len(x) / len(coeffs)}')

        plt.figure()
        plt.yscale('log')
        plt.plot(residual_list)
        plt.title('erreur en dB en fonction de l\'itération')
        plt.xlabel('n° iteration')
        plt.ylabel('erreur en db')
        plt.legend()
        plt.show()
    return approx, coeffs, indices, x - approx

def compress(data, dictionary, window_size, step_size, max_iter, tol=1e-7):
    i = 0
    t1 = time.time()
    n_coeffs=0
    if len(data)%window_size>0:
        signal_recomposed = np.zeros(len(data)+(window_size-len(data)%window_size))
    else:
        signal_recomposed = np.zeros(len(data))

    while i+window_size<len(data):
        x = data[i:window_size+i]*np.hamming(window_size)
        approx, coeffs, indices, residual = matching_pursuit(x, dictionary, max_iter=max_iter, verbose=False)
        signal_recomposed[i:window_size+i]+=approx
        n_coeffs += len(coeffs)
        print(len(coeffs))
        i+=step_size
        print(f'avancement du calcul {i/len(data)*100} %')

    x = np.pad(data[i:], (0, window_size-len(data[i:])), mode='constant')*np.hamming(window_size)
    approx, coeffs, indices, residual = matching_pursuit(x, dictionary, max_iter=max_iter ,verbose=False)
    
    signal_recomposed[i:]+=approx
    plt.plot(data)
    plt.plot(signal_recomposed)
    plt.show()

    t2 = time.time()
    print(f"Arrêt à l'itération     : {i}")
    print(f'RSB                     : {np.linalg.norm(data)/np.linalg.norm(data-signal_recomposed[:len(data)])}')
    print(f'tolérance du résidu     : {tol}')
    print(f'taille du dictionnaire  : {len(dictionary[0])}')
    print(f'temps d execution       : {t2-t1}')
    print(f'taux de compression     : {len(data) / n_coeffs}')

    sd.play(data, samplerate=sr)
    sd.wait() 

    sd.play(signal_recomposed, samplerate=sr)
    sd.wait()



# Chargement du signal a 16 kHz
filepath = os.path.abspath('./audio_partie4/a.wav')
data, sr = sf.read(filepath)
new_sr = 16000
num_samples = int(len(data) * new_sr / sr)
data = resample(data, num_samples)
sr = new_sr

# Exemple sur une fenetre
window_size = 1024
start = int(0.5*len(data))
x=  data[start:start+window_size]
x = x*np.hamming(window_size)
t = np.linspace(0, window_size/sr, window_size)
dictionary = create_wavelet_dictionary(window_size, sr)

approx, coeffs, indices, residual = matching_pursuit(x, dictionary, max_iter=400, verbose=True)

plt.figure(figsize=(10, 6))
plt.plot(t, x, label='Signal original')
plt.plot(t, approx, '--', label='Approximation par Matching Pursuit')
plt.xlabel("Temps")
plt.ylabel("Amplitude")
plt.legend()
plt.title("Approximation du signal par Matching Pursuit avec ondelettes")
plt.show()


#sur tout le signal
compress(data, dictionary, window_size, 512, max_iter=150, tol=1e-7)