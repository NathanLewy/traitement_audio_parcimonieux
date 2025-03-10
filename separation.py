import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy import interpolate
import matplotlib.colors as plc



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
fs = 10000 # Fréquence d'échantillonnage
N=18 
T=4  # Durée du signal (2 secondes)
t = np.linspace(0, T, int(T*fs))  # Temps de 1 seconde
f_sinus_1 = 400  # Fréquence du sinus (500 Hz)
f_sinus_2 = 100
amplitude_sinus = 0.5  # Amplitude du sinus
wave = 'sym20'  # Type d'ondelette
n_level= 30

# Génération du signal : un sinus + bruit blanc
sinus_1 = amplitude_sinus * np.sin(2 * np.pi * f_sinus_1 * t)
sinus_2 = amplitude_sinus * np.sin(2 * np.pi * f_sinus_2 * t)
signal_somme = sinus_1 + sinus_2

# Application de la DWT (décomposition en 3 niveaux)
coeffs = pywt.wavedec(signal_somme, wave, level=n_level)

# Afficher les coefficients ligne par ligne
G = [len(c) for c in coeffs]
maxsize = max(G)
img = []

coeffs_detail = coeffs[1:]
for i, c in enumerate(coeffs_detail):

    interpolated = interpolate.interp1d(np.linspace(0, 1, len(c)), c, kind='previous')
    c_ = interpolated(np.linspace(0, 1, maxsize))
    c_padded = pad_symmetric(c, maxsize)
    img.append(c_)


plt.figure(figsize=(12, 10))
plt.imshow(np.log10(img), cmap='viridis', aspect='auto', interpolation='nearest')
plt.colorbar()
plt.show()



# Séparation des deux sinus, et mise à 0 du scalogramme
energy_levels=[]
for i,ligne in enumerate(coeffs):
    energy_levels.append(np.sum(abs(ligne)/np.max(np.abs(ligne)))/len(ligne))
energy_levels[0]=0

plt.figure()
plt.plot(energy_levels)
plt.show()
sine_1 = np.argmax(energy_levels)
energy_levels[sine_1]=0
sine_2 = np.argmax(energy_levels)
separation_level = int((sine_1 + sine_2)/2)
print(separation_level)


c_sin1, c_sin2 = coeffs.copy(), coeffs.copy()
for i in range(separation_level):
    c_sin1[i] = np.zeros(len(coeffs[i]))
for i in range(separation_level, len(coeffs)):
    c_sin2[i] = np.zeros(len(coeffs[i]))



# Reconstruction du signal à partir des coefficients modifiés
sinus1_reduit = pywt.waverec(c_sin1, wave)
sinus2_reduit = pywt.waverec(c_sin2, wave)
somme_reduit = sinus1_reduit + sinus2_reduit

# Plot the reconstructed signal
plt.figure()
plt.subplot(3,1,1)
plt.plot(t, signal_somme, label='Deux sinus', alpha=0.5)
plt.plot(t, somme_reduit, label='Deux sinus reconstruits', alpha=0.5)
plt.legend()

plt.subplot(3,1,2)
plt.plot(t, sinus1_reduit[:len(t)], label='Sinus 1 reconstruit')
plt.plot(t, sinus_1, label = 'Sinus 1 original')
plt.legend()

plt.subplot(3,1,3)
plt.plot(t, sinus2_reduit[:len(t)], label='Sinus 2 reconstruit')
plt.plot(t, sinus_2, label = 'Sinus 2 original')

plt.legend()
plt.show()


# Calcul des erreurs
erreur_sinus1=sum((sinus_1-sinus1_reduit)**2)/sum(sinus_1**2)
erreur_sinus2=sum((sinus_2-sinus2_reduit)**2)/sum(sinus_2**2)
erreur_sinus_somme=sum((signal_somme-somme_reduit)**2)/sum(signal_somme**2)

print('erreur 1er sinus : '+str(erreur_sinus1))
print('erreur 2e sinus : '+str(erreur_sinus2))
print('erreur sinus somme : '+str(erreur_sinus_somme))
