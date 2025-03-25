import numpy as np
import os
import soundfile as sf
import matplotlib.pyplot as plt
import pywt



def dictionnaire(taille_fenetre):
    wavelets = pywt.wavelist(kind='discrete')
    print(wavelets)
    discrete_wavelets = []
    continuous_wavelets = []

    for w in wavelets:
        try:
            wavelet_obj = pywt.Wavelet(w)
            discrete_wavelets.append(wavelet_obj)
        except ValueError:
            continuous_wavelets.append(w)
    for i,w in enumerate(discrete_wavelets):
        if i%4==0:
            print(w)
        plt.show()

dictionnaire(1024)
ff()

# Génération d'un atome sous forme d'ondelette
def atome(N_, f_ech_, periode, phase):
    
    # Appliquer une ondelette Daubechies de 4 coefficients (db4)
    wavelet = 'db20'  # Ondelette Daubechies de 4 coefficients
    coeffs = pywt.Wavelet(wavelet).wavefun(level=1)  # Calcul des coefficients de l'ondelette
    atome = coeffs[0]  # Extraire l'ondelette
    
    atome = np.pad(atome, (0, N_ - len(atome)), 'constant')
    atome = np.roll(atome, phase)
    # Normaliser l'ondelette pour qu'elle ait une énergie totale de 1
    return atome / np.linalg.norm(atome)

def project(signal_, f_ech, min_period_, max_period_):
    max_proj = -float('inf')
    best_k = 0
    best_phi = 0

    for k in range(min_period_,max_period_):
        for phi in range(k):
            g_k_phi = atome(len(signal_),f_ech, k/f_ech, phi/f_ech)
            proj = np.abs(np.dot(signal_,g_k_phi))
            if proj > max_proj:
                max_proj = proj
                best_k = k
                best_phi = phi
    return atome(len(signal_),f_ech,best_k,best_phi)




#matching pursuit
def OMP(signal, f_ech, f_comp_min, f_comp_max, seuil,maxN):
    period_comp_min = f_ech // f_comp_max
    period_comp_max = f_ech // f_comp_min

    r = [float('inf') for i in range(len(signal))]
    liste_proj = []
    liste_u = []

    meilleur_atome = project(signal, f_ech, period_comp_min, period_comp_max)
    liste_proj.append(meilleur_atome)
    liste_u.append(meilleur_atome)
    r = signal - np.dot(meilleur_atome, signal) * meilleur_atome
    print(r)

    while np.sqrt(np.dot(r, r))/np.sqrt(np.dot(signal,signal)) > seuil or len(liste_u)<maxN:  
        meilleur_atome = project(r, f_ech, period_comp_min, period_comp_max)
        proj_residu_sur_atome = np.dot(meilleur_atome,r)*meilleur_atome
        proj_orthogonal = np.zeros_like(liste_u[0])
        
        for i in range(len(liste_u)):
            print(liste_u[i])
            proj_orthogonal += np.dot(proj_residu_sur_atome, liste_u[i]) * liste_u[i]

        vecteur_innovation = proj_residu_sur_atome - proj_orthogonal
        print(np.dot(proj_orthogonal,proj_orthogonal))
        vecteur_orthogonal = vecteur_innovation/np.sqrt(np.dot(vecteur_innovation, vecteur_innovation))
        r = r - np.dot(vecteur_orthogonal,r)*vecteur_orthogonal


        liste_proj.append(meilleur_atome)
        liste_u.append(vecteur_orthogonal)


    M = np.matmul(np.array(liste_u), np.array(liste_proj).T)
    m = np.array([np.dot(signal, i) for i in liste_u])
    x =np.linalg.solve(M,m)

    approx = np.sum([abs(x[i])*liste_u[i] for i in range(len(x))], axis = 0)
    print(x)
    plt.figure()
    plt.plot(approx, label='approx')
    plt.plot(signal, label='signal')
    plt.plot(r, label='residu')
    plt.legend()
    plt.show()


#signal to process
filepath = os.path.abspath('./audio_partie4/a.mp3')
data, sr = sf.read(filepath)


taille_fenetre = 1024
signal = data[:taille_fenetre]

f_ech = 16000 #Hz
f_comp_min = 200
f_comp_max = 4000
print(sr)
OMP(signal, f_ech, f_comp_min, f_comp_max, 0.05, 100)