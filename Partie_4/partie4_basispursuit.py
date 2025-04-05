import numpy as np
import matplotlib.pyplot as plt
import os
import soundfile as sf
import pywt
import time
import sounddevice as sd
from scipy.signal import resample
from concurrent.futures import ThreadPoolExecutor
from scipy.optimize import linprog
from scipy.signal import butter, filtfilt



class AudioCompressor:
    def __init__(self, file_path, window_size=1024, step_size=512, max_iter=80, sr=16000, wavelet_names=['rbio3.3', 'bior3.5', 'bior5.5', 'bior6.8'], solver_type='mp'):
        self.file_path = file_path
        self.window_size = window_size
        self.step_size = step_size
        self.max_iter = max_iter
        self.sr = sr
        self.wavelet_names = wavelet_names
        self.solver_type = solver_type
        self.data, self.sr = self.load_audio(file_path, sr)
        self.dictionary = self.create_wavelet_dictionary(window_size)
        self.solved = False
        
        # Choix du solveur
        if self.solver_type == 'bs':
            self.solver = BasisPursuitSolver(max_iter)
        else:
            raise ValueError("Solver non existant")

    def load_audio(self, file_path, target_sr):
        data, sr = sf.read(file_path)
        data = resample(data, int(len(data) * target_sr / sr))
        data = self.bandpass_filter(data, lowcut=20, highcut=7900, fs=target_sr)
        return data, target_sr
    
    def bandpass_filter(self, data, lowcut=20, highcut=15000, fs=44100, order=5):
        nyquist = 0.5 * fs  # Fréquence de Nyquist
        low = lowcut / nyquist
        high = highcut / nyquist

        if not (0 < low < 1 and 0 < high < 1):
            raise ValueError(f"lowcut ({lowcut} Hz) et highcut ({highcut} Hz) doivent être entre 0 et {nyquist} Hz")

        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data)




    def create_wavelet_dictionary(self, signal_length):
        dict_matrix = []
        for wavelet_name in self.wavelet_names:
            wavelet = pywt.Wavelet(wavelet_name)
            wavelet_function = wavelet.wavefun(level=3)
            for ordre in range(len(wavelet_function) - 2):
                for pad in range(signal_length - len(wavelet_function[ordre])):
                    padded_wavelet = np.pad(wavelet_function[ordre], (pad, signal_length - len(wavelet_function[ordre]) - pad), mode='constant')
                    dict_matrix.append(padded_wavelet / np.linalg.norm(padded_wavelet))

        # Ajout des cosinus
        for k in range(signal_length):
            cos_k = np.cos([np.pi*(n + 1/2)* k / signal_length for n in range(signal_length)])
            dict_matrix.append(cos_k / np.linalg.norm(cos_k))
        
        return np.array(dict_matrix).T

    def process_window(self, args):
        data, dictionary, window_size, step_size, i = args
        x = data[i:i + window_size] * np.hamming(window_size)
        
        # Applique le solveur sélectionné
        approx, coeffs, indices = self.solver.solve(x, dictionary)
        
        return i, approx, len(coeffs)

    def compress(self):
        self.t1 = time.time()
        signal_recomposed = np.zeros_like(self.data)
        n_coeffs = 0

        # Préparation des arguments pour chaque fenêtre de traitement
        args_list = [(self.data, self.dictionary, self.window_size, self.step_size, i) for i in range(0, len(self.data) - self.window_size, self.step_size)]
        
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(self.process_window, args_list))

        # Reconstruction du signal compressé
        for i, approx, coeff_count in results:
            signal_recomposed[i:i + self.window_size] += approx
            n_coeffs += coeff_count

        self.t2 = time.time()
        self.n_coeffs = n_coeffs
        self.signal_recomposed = signal_recomposed

        self.solved = True
        self.approx = signal_recomposed
        if self.n_coeffs!=0:
            self.tcomp = len(self.data) * 32 / (self.n_coeffs * (32 + np.log2(len(self.dictionary.T))))
            self.RSB = (np.linalg.norm(self.data) / np.linalg.norm(self.data - self.signal_recomposed[:len(self.data)]))**2
        else:
            self.tcomp=0
            self.RSB=0
        self.tex = self.t2 - self.t1

        return self.approx, self.tcomp, self.RSB, self.tex
    def compression_report(self):
        if self.solved:
            self.plot_results(self.signal_recomposed)

            print(f'RSB : {self.RSB}')
            print(f'Temps d’exécution : {self.tex}')
            print(f'Taux de compression : {self.tcomp}')

            # Lecture audio
            self.play_audio(self.data)
            self.play_audio(self.signal_recomposed)
        else:
            print("Error : no compression to report on")

    def plot_results(self, signal_recomposed):
        """Affiche les résultats de la compression."""
        plt.plot(self.data, label='Original')
        plt.plot(signal_recomposed, label='Recomposé')
        plt.legend()
        plt.show()

    def play_audio(self, audio_data):
        """Joue l'audio à partir des données fournies."""
        sd.play(audio_data, samplerate=self.sr)
        sd.wait()

    def change_maxit(self, newmaxit):
        self.max_iter = newmaxit
        self.solver.max_iter = newmaxit


class BasisPursuitSolver:
    def __init__(self, max_iter):
        pass
    
    def solve(self, x, dictionary_):
        dictionary = dictionary_[:,:20]
        m, n = dictionary.shape
        c = np.ones(2*n)  # Fonction objectif : somme des variables auxiliaires
        
        # Contrainte : x = D*alpha -> D * (alpha+ - alpha-) = x
        A_eq = np.hstack([dictionary, -dictionary])
        b_eq = x
        
        # Contraintes de positivité des variables auxiliaires alpha+ et alpha-
        bounds = [(0, None) for _ in range(2*n)]
        
        # Résolution du problème d'optimisation linéaire
        result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        
        if result.success:
            alpha = result.x[:n] - result.x[n:]  # Reconstruction du vecteur de coefficients
            approx = dictionary @ alpha  # Reconstruction du signal
            indices = np.nonzero(alpha)[0].tolist()  # Indices des atomes sélectionnés
            return approx, alpha, indices
        else:
            print("Basis Pursuit n'a pas convergé.")
            return np.zeros_like(x), [], []



if __name__ == "__main__":
    file_path = os.path.abspath('./Partie_4/audio_partie4/a.wav')
    liste_maxit = range(10,150, 10)

    print('\n basis pursuit :')

    #attention le maxiter ne sert pas pour la basis pursuit
    solver_type = 'bs' 
    compressor = AudioCompressor(file_path, solver_type=solver_type)
    approx, tcomp, RSB, tex = compressor.compress()
    compressor.compression_report()


    

    
