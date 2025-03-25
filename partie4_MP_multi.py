import numpy as np
import matplotlib.pyplot as plt
import os
import soundfile as sf
import pywt
import time
import sounddevice as sd
from scipy.signal import resample
from concurrent.futures import ThreadPoolExecutor

def create_wavelet_dictionary(signal_length):
    wavelet_names = ['db5', 'db1']
    dict_matrix = []
    for wavelet_name in wavelet_names:
        wavelet = pywt.Wavelet(wavelet_name)
        wavelet_function = wavelet.wavefun(level=3)
        for ordre in range(len(wavelet_function)-2):
            for pad in range(signal_length - len(wavelet_function[ordre])):
                padded_wavelet = np.pad(wavelet_function[ordre], (pad, signal_length - len(wavelet_function[ordre]) - pad), mode='constant')
                dict_matrix.append(padded_wavelet / np.linalg.norm(padded_wavelet))
    return np.array(dict_matrix).T

def matching_pursuit(x, dictionary, max_iter, tol=1e-7):
    approx = np.zeros_like(x)
    coeffs, indices = [], []
    for _ in range(max_iter):
        projections = dictionary.T @ (x - approx)
        k = np.argmax(np.abs(projections))
        a_k = projections[k]
        approx += a_k * dictionary[:, k]
        coeffs.append(a_k)
        indices.append(k)
        if np.linalg.norm(x - approx) < tol:
            break
    return approx, coeffs, indices

def process_window(args):
    data, dictionary, window_size, step_size, max_iter, i = args
    x = data[i:i + window_size] * np.hamming(window_size)
    approx, coeffs, indices = matching_pursuit(x, dictionary, max_iter)
    return i, approx, len(coeffs)

def compress(data, dictionary, window_size, step_size, max_iter, tol=1e-7):
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
    print(f'Temps d’exécution : {time.time()}')
    print(f'Taux de compression : {len(data) / n_coeffs}')
    
    sd.play(signal_recomposed, samplerate=sr)
    sd.wait()

filepath = os.path.abspath('./audio_partie4/a.wav')
data, sr = sf.read(filepath)
data = resample(data, int(len(data) * 16000 / sr))
sr = 16000
window_size = 1024
dictionary = create_wavelet_dictionary(window_size)
compress(data, dictionary, window_size, 512, max_iter=500)
