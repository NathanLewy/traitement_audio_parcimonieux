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




class MatchingPursuitSolver:
    def __init__(self, max_iter=120):
        self.max_iter = max_iter

    def solve(self, x, dictionary, tol=1e-7):
        approx = np.zeros_like(x)
        coeffs, indices = [], []
        for _ in range(self.max_iter):
            projections = dictionary.T @ (x - approx)
            k = np.argmax(np.abs(projections))
            a_k = projections[k]
            approx += a_k * dictionary[:, k]
            coeffs.append(a_k)
            indices.append(k)
            if np.linalg.norm(x - approx) < tol:
                break
        return approx, coeffs, indices

class OrthogonalMatchingPursuitSolver:
    def __init__(self, max_iter=120):
        self.max_iter = max_iter

    def solve(self, x, dictionary, tol=1e-4):
        r = x.copy()
        liste_proj = []  # Atomes bruts sélectionnés dans le dictionnaire
        liste_u = []     # Vecteurs orthonormaux issus de l'innovation
        
        for _ in range(self.max_iter):
            if np.sqrt(np.dot(r, r)) / np.sqrt(np.dot(x, x)) < tol:
                break
            if np.sqrt(np.dot(r, r)) / np.sqrt(np.dot(x, x)) > 1:
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

        # Résolution du système linéaire pour obtenir les coefficients finaux
        M = np.matmul(np.array(liste_u), np.array(liste_proj).T)
        m_vec = np.array([np.dot(x, u) for u in liste_u])
        if M.shape[0] > 0:
            coeffs = np.linalg.solve(M, m_vec)
        else:
            coeffs = []
        approx = np.sum([coeffs[i] * liste_proj[i] for i in range(len(coeffs))], axis=0)
        indices = [np.argmax(np.abs(dictionary.T @ atom)) for atom in liste_proj]

        return approx, coeffs, indices

class AudioCompressor:
    def __init__(self, file_path, window_size=1024, step_size=512, max_iter=120, sr=16000, wavelet_names=['rbio5.5','bior3.1'], solver_type='mp'):
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
        if self.solver_type == 'mp':
            self.solver = MatchingPursuitSolver(max_iter)
        elif self.solver_type == 'omp':
            self.solver = OrthogonalMatchingPursuitSolver(max_iter)
        else:
            raise ValueError("Solver must be either 'omp' or 'mp'.")

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

        return signal_recomposed, len(self.data) / self.n_coeffs, np.linalg.norm(self.data) / np.linalg.norm(self.data - self.signal_recomposed[:len(self.data)]), self.t2 - self.t1
    
    def compression_report(self):
        if self.solved:
            self.plot_results(self.signal_recomposed)

            print(f'RSB : {np.linalg.norm(self.data) / np.linalg.norm(self.data - self.signal_recomposed[:len(self.data)])}')
            print(f'Temps d’exécution : {self.t2 - self.t1}')
            print(f'Taux de compression : {len(self.data) / self.n_coeffs}')

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

if __name__ == "__main__":
    file_path = os.path.abspath('./audio_partie4/a.wav')
    liste_maxit = range(10,150, 10)

    print('Recherche meilleur dico :')
    wavelet_list = pywt.wavelist(kind='discrete')
    max=0
    best_dict=[]
    for loop in range(1):
        print(loop, best_dict)
        wavelet_names = np.random.choice(wavelet_list,2, replace=False)
        solver_type = 'mp' 
        compressor = AudioCompressor(file_path, solver_type=solver_type,wavelet_names=wavelet_names)
        compressor.change_maxit(50)
        approx, tcomp, RSB, tex = compressor.compress()
        if RSB >max:
            best_dict = wavelet_names
            max = RSB
    
    print(f'meilleur dico trouvé sur cette itération : {wavelet_names}')
    print(f'on prend finalement : [\'rbio5.5\' \'bior3.1\']')


    print('\n matching pursuit :')
    
    solver_type = 'mp' 
    compressor = AudioCompressor(file_path, solver_type=solver_type)
    approx, tcomp, RSB, tex = compressor.compress()
    compressor.compression_report()
    
    liste_RSB=[]
    liste_tex=[]
    liste_tcomp = []

    for i in liste_maxit:
        compressor.change_maxit(i)
        approx, tcomp, RSB, tex = compressor.compress()
        liste_RSB.append(RSB)
        liste_tex.append(tex)
        liste_tcomp.append(tcomp)

    plt.figure()
    plt.plot(liste_maxit, liste_RSB)
    plt.xlabel('maxit')
    plt.ylabel('RSB')
    plt.title('MP - RSB en fonction de maxit')
    plt.show()

    plt.figure()
    plt.plot(liste_maxit, liste_tex)
    plt.xlabel('maxit')
    plt.ylabel('temps d\'execution')
    plt.title('MP - temps d\'execution en fonction de maxit')
    plt.show()
    
    plt.figure()
    plt.plot(liste_maxit, liste_tcomp)
    plt.xlabel('maxit')
    plt.ylabel('taux de compression')
    plt.title('MP - taux de compression en fonction de maxit')
    plt.show()

    print('\n orthogonal matching pursuit :')
    solver_type = 'omp' 
    compressor = AudioCompressor(file_path, solver_type=solver_type)
    approx, tcomp, RSB, tex = compressor.compress()
    compressor.compression_report()

    liste_RSB=[]
    liste_tex=[]
    liste_tcomp=[]
    for i in liste_maxit:
        print(i)
        compressor.change_maxit(i)
        approx, tcomp, RSB, tex = compressor.compress()
        liste_RSB.append(RSB)
        liste_tex.append(tex)
        liste_tcomp.append(tcomp)


    plt.figure()
    plt.plot(liste_maxit, liste_RSB)
    plt.xlabel('maxit')
    plt.ylabel('RSB')
    plt.title('OMP - RSB en fonction de maxit')
    plt.show()

    plt.figure()
    plt.plot(liste_maxit, liste_tex)
    plt.xlabel('maxit')
    plt.ylabel('temps d\'execution')
    plt.title('OMP - temps d\'execution en fonction de maxit')
    plt.show()

    plt.figure()
    plt.plot(liste_maxit, liste_tcomp)
    plt.xlabel('maxit')
    plt.ylabel('taux de compression')
    plt.title('OMP - taux de compression en fonction de maxit')
    plt.show()


    
