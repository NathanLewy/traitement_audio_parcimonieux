import numpy as np
import pywt
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error  
# Paramètres du signal
fs = 44000  # Fréquence d'échantillonnage
t = np.linspace(0, 1, fs, endpoint=False)  # 1 seconde de signal

# Création des sinusoïdes
sinusoid_100Hz = np.sin(2 * np.pi * 100 * t)
sinusoid_500Hz = np.sin(2 * np.pi * 500 * t)

# Somme des signaux
composite_signal = sinusoid_100Hz + sinusoid_500Hz


# Décomposition en ondelettes
wavelet = 'db4'  # Choix de la famille d'ondelettes
coeffs = pywt.wavedec(composite_signal, wavelet)

# Calcul de la norme pour chaque array dans coeffs
norms = [np.average(abs(array)) for array in coeffs]

# Indices des 4 plus grandes arrays
largest_indices = np.argsort(norms)[-4:]    
print(largest_indices)

# Vérification des indices pour éviter les dépassements
largest_indices = [index for index in largest_indices if index < len(coeffs) - 1]

# Création de deux tableaux de zéros de la même taille que la sortie de wavedec
zero_array_1 = [np.zeros_like(array) for array in coeffs]
zero_array_2 = [np.zeros_like(array) for array in coeffs]

# Remplacement des arrays "i, i+1" et "k, k+1"
if len(largest_indices) >= 2:
    i, k = largest_indices[0], largest_indices[2]
    print(i,k)
    zero_array_1[i], zero_array_1[i+1] = coeffs[i], coeffs[i+1]
    zero_array_2[k], zero_array_2[k+1] = coeffs[k], coeffs[k+1]
else:
    raise ValueError("Pas assez d'indices valides pour effectuer la séparation des signaux.")

# Reconstruction des signaux à partir des tableaux modifiés
reconstructed_signal_1 = pywt.waverec(zero_array_1, wavelet)
reconstructed_signal_2 = pywt.waverec(zero_array_2, wavelet)

# Affichage des signaux
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(t, composite_signal)
plt.title('Signal Composite')

plt.subplot(3, 1, 2)
plt.plot(t, reconstructed_signal_1)
plt.title('Signal Reconstruit 1 (i, i+1)')

plt.subplot(3, 1, 3)
plt.plot(t, reconstructed_signal_2)
plt.title('Signal Reconstruit 2 (k, k+1)')

plt.tight_layout()
plt.show()

# Calcul de l'erreur quadratique moyenne (MSE) entre les signaux reconstruits et les sinusoïdes originaux
mse_100Hz = mean_squared_error(sinusoid_500Hz, reconstructed_signal_1)
mse_500Hz = mean_squared_error(sinusoid_100Hz, reconstructed_signal_2)

# Affichage des signaux
plt.figure(figsize=(12, 10))

plt.subplot(4, 1, 1)
plt.plot(t, composite_signal)
plt.title('Signal Composite')

plt.subplot(4, 1, 2)
plt.plot(t, reconstructed_signal_1, label='Reconstruit 1,db4')
plt.plot(t, sinusoid_500Hz, label='Original 100Hz', linestyle='--')
plt.title(f'Signal Reconstruit 1 (i, i+1)\nMSE: {mse_100Hz:.4f}')
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(t, reconstructed_signal_2, label='Reconstruit 2,db4')
plt.plot(t, sinusoid_100Hz, label='Original 500Hz', linestyle='--')
plt.title(f'Signal Reconstruit 2 (k, k+1)\nMSE: {mse_500Hz:.4f}')
plt.legend()

plt.subplot(4, 1, 4)
plt.plot(t, sinusoid_100Hz, label='Original 100Hz')
plt.plot(t, sinusoid_500Hz, label='Original 500Hz')
plt.title('Sinusoïdes Originaux')
plt.legend()

plt.tight_layout()
plt.show()

# Création des sinusoïdes
sinusoid_100Hz = np.sin(2 * np.pi * 100 * t)
sinusoid_500Hz = np.sin(2 * np.pi * 500 * t)

# Somme des signaux
composite_signal = sinusoid_100Hz + sinusoid_500Hz


# Décomposition en ondelettes
wavelet = 'haar'  # Choix de la famille d'ondelettes
coeffs = pywt.wavedec(composite_signal, wavelet)

# Calcul de la norme pour chaque array dans coeffs
norms = [np.average(abs(array)) for array in coeffs]

# Indices des 4 plus grandes arrays
largest_indices = np.argsort(norms)[-4:]    
print(largest_indices)

# Vérification des indices pour éviter les dépassements
largest_indices = [index for index in largest_indices if index < len(coeffs) - 1]

# Création de deux tableaux de zéros de la même taille que la sortie de wavedec
zero_array_1 = [np.zeros_like(array) for array in coeffs]
zero_array_2 = [np.zeros_like(array) for array in coeffs]

# Remplacement des arrays "i, i+1" et "k, k+1"
if len(largest_indices) >= 2:
    i, k = largest_indices[0], largest_indices[2]
    print(i,k)
    zero_array_1[i], zero_array_1[i+1] = coeffs[i], coeffs[i+1]
    zero_array_2[k], zero_array_2[k+1] = coeffs[k], coeffs[k+1]
else:
    raise ValueError("Pas assez d'indices valides pour effectuer la séparation des signaux.")

# Reconstruction des signaux à partir des tableaux modifiés
reconstructed_signal_1 = pywt.waverec(zero_array_1, wavelet)
reconstructed_signal_2 = pywt.waverec(zero_array_2, wavelet)

# Affichage des signaux
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(t, composite_signal)
plt.title('Signal Composite')

plt.subplot(3, 1, 2)
plt.plot(t, reconstructed_signal_1)
plt.title('Signal Reconstruit 1 (i, i+1)')

plt.subplot(3, 1, 3)
plt.plot(t, reconstructed_signal_2)
plt.title('Signal Reconstruit 2 (k, k+1)')

plt.tight_layout()
plt.show()

# Calcul de l'erreur quadratique moyenne (MSE) entre les signaux reconstruits et les sinusoïdes originaux
mse_100Hz = mean_squared_error(sinusoid_500Hz, reconstructed_signal_1)
mse_500Hz = mean_squared_error(sinusoid_100Hz, reconstructed_signal_2)

# Affichage des signaux
plt.figure(figsize=(12, 10))

plt.subplot(4, 1, 1)
plt.plot(t, composite_signal)
plt.title('Signal Composite')

plt.subplot(4, 1, 2)
plt.plot(t, reconstructed_signal_1, label='Reconstruit 1,haar')
plt.plot(t, sinusoid_500Hz, label='Original 100Hz', linestyle='--')
plt.title(f'Signal Reconstruit 1 (i, i+1)\nMSE: {mse_100Hz:.4f}')
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(t, reconstructed_signal_2, label='Reconstruit 2,haar')
plt.plot(t, sinusoid_100Hz, label='Original 500Hz', linestyle='--')
plt.title(f'Signal Reconstruit 2 (k, k+1)\nMSE: {mse_500Hz:.4f}')
plt.legend()

plt.subplot(4, 1, 4)
plt.plot(t, sinusoid_100Hz, label='Original 100Hz')
plt.plot(t, sinusoid_500Hz, label='Original 500Hz')
plt.title('Sinusoïdes Originaux')
plt.legend()

plt.tight_layout()
plt.show()






# Création des sinusoïdes
sinusoid_100Hz = np.sin(2 * np.pi * 100 * t)
sinusoid_500Hz = np.sin(2 * np.pi * 500 * t)

# Somme des signaux
composite_signal = sinusoid_100Hz + sinusoid_500Hz


# Décomposition en ondelettes
wavelet = 'coif17'  # Choix de la famille d'ondelettes
coeffs = pywt.wavedec(composite_signal, wavelet)

# Calcul de la norme pour chaque array dans coeffs
norms = [np.average(abs(array)) for array in coeffs]

# Indices des 4 plus grandes arrays
largest_indices = np.argsort(norms)[-4:]    
print(largest_indices)

# Vérification des indices pour éviter les dépassements
largest_indices = [index for index in largest_indices if index < len(coeffs) - 1]

# Création de deux tableaux de zéros de la même taille que la sortie de wavedec
zero_array_1 = [np.zeros_like(array) for array in coeffs]
zero_array_2 = [np.zeros_like(array) for array in coeffs]

# Remplacement des arrays "i, i+1" et "k, k+1"
if len(largest_indices) >= 2:
    i, k = largest_indices[0], largest_indices[2]
    print(i,k)
    zero_array_1[i], zero_array_1[i+1] = coeffs[i], coeffs[i+1]
    zero_array_2[k], zero_array_2[k+1] = coeffs[k], coeffs[k+1]
else:
    raise ValueError("Pas assez d'indices valides pour effectuer la séparation des signaux.")

# Reconstruction des signaux à partir des tableaux modifiés
reconstructed_signal_1 = pywt.waverec(zero_array_1, wavelet)
reconstructed_signal_2 = pywt.waverec(zero_array_2, wavelet)

# Affichage des signaux
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(t, composite_signal)
plt.title('Signal Composite')

plt.subplot(3, 1, 2)
plt.plot(t, reconstructed_signal_1)
plt.title('Signal Reconstruit 1 (i, i+1)')

plt.subplot(3, 1, 3)
plt.plot(t, reconstructed_signal_2)
plt.title('Signal Reconstruit 2 (k, k+1)')

plt.tight_layout()
plt.show()

# Calcul de l'erreur quadratique moyenne (MSE) entre les signaux reconstruits et les sinusoïdes originaux
mse_100Hz = mean_squared_error(sinusoid_500Hz, reconstructed_signal_1)
mse_500Hz = mean_squared_error(sinusoid_100Hz, reconstructed_signal_2)

# Affichage des signaux
plt.figure(figsize=(12, 10))

plt.subplot(4, 1, 1)
plt.plot(t, composite_signal)
plt.title('Signal Composite')

plt.subplot(4, 1, 2)
plt.plot(t, reconstructed_signal_1, label='Reconstruit 1,coif17')
plt.plot(t, sinusoid_500Hz, label='Original 100Hz', linestyle='--')
plt.title(f'Signal Reconstruit 1 (i, i+1)\nMSE: {mse_100Hz:.4f}')
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(t, reconstructed_signal_2, label='Reconstruit 2,coif17')
plt.plot(t, sinusoid_100Hz, label='Original 500Hz', linestyle='--')
plt.title(f'Signal Reconstruit 2 (k, k+1)\nMSE: {mse_500Hz:.4f}')
plt.legend()

plt.subplot(4, 1, 4)
plt.plot(t, sinusoid_100Hz, label='Original 100Hz')
plt.plot(t, sinusoid_500Hz, label='Original 500Hz')
plt.title('Sinusoïdes Originaux')
plt.legend()

plt.tight_layout()
plt.show()








# Paramètres du signal
fs = 44000  # Fréquence d'échantillonnage
t = np.linspace(0, 1, fs, endpoint=False)  # 1 seconde de signal

# Création des sinusoïdes
sinusoid_100Hz = np.sin(2 * np.pi * 100 * t)
sinusoid_500Hz = np.sin(2 * np.pi * 500 * t)

# Somme des signaux
composite_signal = sinusoid_100Hz + sinusoid_500Hz


MSE_1 = []
MSE_2 = []


for level_ond in range(3,31):

    # Décomposition en ondelettes
    wavelet = 'coif17'  # Choix de la famille d'ondelettes
    coeffs = pywt.wavedec(composite_signal, wavelet,level = level_ond)

    # Calcul de la norme pour chaque array dans coeffs 
    norms = [np.average(abs(array)) for array in coeffs]

    # Indices des 4 plus grandes arrays
    largest_indices = np.argsort(norms)[-4:]    
    print(largest_indices)

    # Vérification des indices pour éviter les dépassements
    largest_indices = [index for index in largest_indices if index < len(coeffs) - 1]

    # Création de deux tableaux de zéros de la même taille que la sortie de wavedec
    zero_array_1 = [np.zeros_like(array) for array in coeffs]
    zero_array_2 = [np.zeros_like(array) for array in coeffs]

    # Remplacement des arrays "i, i+1" et "k, k+1"
    if len(largest_indices) >= 2:
        i, k = largest_indices[0], largest_indices[2]
        print(i,k)
        zero_array_1[i], zero_array_1[i+1] = coeffs[i], coeffs[i+1]
        zero_array_2[k], zero_array_2[k+1] = coeffs[k], coeffs[k+1]
    else:
        raise ValueError("Pas assez d'indices valides pour effectuer la séparation des signaux.")

    # Reconstruction des signaux à partir des tableaux modifiés 
    reconstructed_signal_1 = pywt.waverec(zero_array_1, wavelet)
    reconstructed_signal_2 = pywt.waverec(zero_array_2, wavelet)

    # Calcul de l'erreur quadratique moyenne (MSE) entre les signaux reconstruits et les sinusoïdes originaux
    mse_100Hz = mean_squared_error(sinusoid_500Hz, reconstructed_signal_1)
    mse_500Hz = mean_squared_error(sinusoid_100Hz, reconstructed_signal_2)

    MSE_1.append(mse_100Hz)
    MSE_2.append(mse_500Hz)


plt.plot([k for k in range(3,31)],MSE_1)
plt.plot([k for k in range(3,31)],MSE_2)
plt.show()









# Paramètres du signal
fs = 44000  # Fréquence d'échantillonnage
t = np.linspace(0, 1, fs, endpoint=False)  # 1 seconde de signal

# Création des sinusoïdes
sinusoid_100Hz = np.sin(2 * np.pi * 100 * t)
sinusoid_500Hz = np.sin(2 * np.pi * 500 * t)

# Somme des signaux
composite_signal = sinusoid_100Hz + sinusoid_500Hz


MSE_1 = []
MSE_2 = []


for level_ond in range(3,31):

    # Décomposition en ondelettes
    wavelet = 'db4'  # Choix de la famille d'ondelettes
    coeffs = pywt.wavedec(composite_signal, wavelet,level = level_ond)

    # Calcul de la norme pour chaque array dans coeffs 
    norms = [np.average(abs(array)) for array in coeffs]

    # Indices des 4 plus grandes arrays
    largest_indices = np.argsort(norms)[-4:]    
    print(largest_indices)

    # Vérification des indices pour éviter les dépassements
    largest_indices = [index for index in largest_indices if index < len(coeffs) - 1]

    # Création de deux tableaux de zéros de la même taille que la sortie de wavedec
    zero_array_1 = [np.zeros_like(array) for array in coeffs]
    zero_array_2 = [np.zeros_like(array) for array in coeffs]

    # Remplacement des arrays "i, i+1" et "k, k+1"
    if len(largest_indices) >= 2:
        i, k = largest_indices[0], largest_indices[2]
        print(i,k)
        zero_array_1[i], zero_array_1[i+1] = coeffs[i], coeffs[i+1]
        zero_array_2[k], zero_array_2[k+1] = coeffs[k], coeffs[k+1]
    else:
        raise ValueError("Pas assez d'indices valides pour effectuer la séparation des signaux.")

    # Reconstruction des signaux à partir des tableaux modifiés 
    reconstructed_signal_1 = pywt.waverec(zero_array_1, wavelet)
    reconstructed_signal_2 = pywt.waverec(zero_array_2, wavelet)

    # Calcul de l'erreur quadratique moyenne (MSE) entre les signaux reconstruits et les sinusoïdes originaux
    mse_100Hz = mean_squared_error(sinusoid_500Hz, reconstructed_signal_1)
    mse_500Hz = mean_squared_error(sinusoid_100Hz, reconstructed_signal_2)

    MSE_1.append(mse_100Hz)
    MSE_2.append(mse_500Hz)


plt.plot([k for k in range(3,31)],MSE_1)
plt.plot([k for k in range(3,31)],MSE_2)
plt.show()






# Création des sinusoïdes
sinusoid_100Hz = np.sin(2 * np.pi * 100 * t)
sinusoid_500Hz = np.sin(2 * np.pi * 500 * t)

# Somme des signaux
composite_signal = sinusoid_100Hz + sinusoid_500Hz


# Décomposition en ondelettes
wavelet = 'coif17'  # Choix de la famille d'ondelettes
coeffs = pywt.wavedec(composite_signal, wavelet)

# Calcul de la norme pour chaque array dans coeffs
norms = [np.average(abs(array)) for array in coeffs]

# Indices des 4 plus grandes arrays
largest_indices = np.argsort(norms)[-4:]    
print(largest_indices)

# Vérification des indices pour éviter les dépassements
largest_indices = [index for index in largest_indices if index < len(coeffs) - 1]

# Création de deux tableaux de zéros de la même taille que la sortie de wavedec
zero_array_1 = [np.zeros_like(array) for array in coeffs]
zero_array_2 = [np.zeros_like(array) for array in coeffs]

# Remplacement des arrays "i, i+1" et "k, k+1"
if len(largest_indices) >= 2:
    i, k = largest_indices[0], largest_indices[2]
    print(i,k)
    zero_array_1[i], zero_array_1[i+1] = coeffs[i], coeffs[i+1]
    zero_array_2[k], zero_array_2[k+1] = coeffs[k], coeffs[k+1]
else:
    raise ValueError("Pas assez d'indices valides pour effectuer la séparation des signaux.")

# Reconstruction des signaux à partir des tableaux modifiés
reconstructed_signal_1 = pywt.waverec(zero_array_1, wavelet)
reconstructed_signal_2 = pywt.waverec(zero_array_2, wavelet)

# Affichage des signaux
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(t, composite_signal)
plt.title('Signal Composite')

plt.subplot(3, 1, 2)
plt.plot(t, reconstructed_signal_1)
plt.title('Signal Reconstruit 1 (i, i+1)')

plt.subplot(3, 1, 3)
plt.plot(t, reconstructed_signal_2)
plt.title('Signal Reconstruit 2 (k, k+1)')

plt.tight_layout()
plt.show()

# Calcul de l'erreur quadratique moyenne (MSE) entre les signaux reconstruits et les sinusoïdes originaux
mse_100Hz = mean_squared_error(sinusoid_500Hz, reconstructed_signal_1)
mse_500Hz = mean_squared_error(sinusoid_100Hz, reconstructed_signal_2)

# Affichage des signaux
plt.figure(figsize=(12, 10))

plt.subplot(4, 1, 1)
plt.plot(t, composite_signal)
plt.title('Signal Composite')

plt.subplot(4, 1, 2)
plt.plot(t, reconstructed_signal_1, label='Reconstruit 1,coif17')
plt.plot(t, sinusoid_500Hz, label='Original 100Hz', linestyle='--')
plt.title(f'Signal Reconstruit 1 (i, i+1)\nMSE: {mse_100Hz:.4f}')
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(t, reconstructed_signal_2, label='Reconstruit 2,coif17')
plt.plot(t, sinusoid_100Hz, label='Original 500Hz', linestyle='--')
plt.title(f'Signal Reconstruit 2 (k, k+1)\nMSE: {mse_500Hz:.4f}')
plt.legend()

plt.subplot(4, 1, 4)
plt.plot(t, sinusoid_100Hz, label='Original 100Hz')
plt.plot(t, sinusoid_500Hz, label='Original 500Hz')
plt.title('Sinusoïdes Originaux')
plt.legend()

plt.tight_layout()
plt.show()


