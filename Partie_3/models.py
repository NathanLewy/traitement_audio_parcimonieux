import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import numpy as np
import seaborn as sns

def charger_donnees(database_folder):
    # Charger les données d'entraînement et de test
    X_SONS = pd.read_csv(os.path.join(database_folder, 'X_SONS.csv'))
    X_SONS_VC = pd.read_csv(os.path.join(database_folder, 'X_SONS-VC.csv'))
    y_SONS = pd.read_csv(os.path.join(database_folder, 'y_SONS.csv'))['Label']
    y_SONS_VC = pd.read_csv(os.path.join(database_folder, 'y_SONS-VC.csv'))['Label']

    return X_SONS, X_SONS_VC, y_SONS, y_SONS_VC

def tracer_heatmap(scores, gammas, C_list, titre):
    plt.figure(figsize=(12, 8))
    sns.heatmap(scores, annot=False, xticklabels=C_list, yticklabels=gammas, cmap="viridis")
    plt.title(titre)
    plt.xlabel('C')
    plt.ylabel('Gamma')
    plt.show()

# Spécifiez le chemin du répertoire contenant la base de données finale
database_folder = '/home/pierres/ST7/traitement_audio_parcimonieux/TL_3_pierre/database_finale'

# Charger les données
X_SONS, X_SONS_VC, y_SONS, y_SONS_VC = charger_donnees(database_folder)

# Définir une gamme de valeurs pour gamma (RBF)
gammas = np.logspace(-1, 1, 2)  # Réduire pour la visualisation
C_list = np.logspace(-5, 5, 2)  # Réduire pour la visualisation

scaler = StandardScaler()
X_SONS = scaler.fit_transform(X_SONS)

# Initialiser une matrice pour stocker les scores moyens
mean_scores = np.zeros((len(gammas), len(C_list)))
mean_scores2 = np.zeros((len(gammas), len(C_list)))

for i, gamma in enumerate(gammas):
    for j, C in enumerate(C_list):
        # Créer le modèle SVM avec noyau RBF
        svm_model = SVC(kernel='rbf', gamma=gamma, C=C)
        # Effectuer la validation croisée
        scores = cross_val_score(svm_model, X_SONS, y_SONS, cv=5, scoring='accuracy')
        # Calculer la moyenne des scores
        mean_scores[i, j] = np.mean(scores)
        svm_model.fit(X_SONS, y_SONS)
        mean_scores2[i, j] = svm_model.score(X_SONS_VC, y_SONS_VC)

# Tracer la heatmap pour les données de test
tracer_heatmap(mean_scores, gammas, C_list, 'Heatmap de la précision moyenne du SVM avec noyau RBF sur les données de test')

# Tracer la heatmap pour les données de validation
tracer_heatmap(mean_scores2, gammas, C_list, 'Heatmap de la précision moyenne du SVM avec noyau RBF sur les données de validation')
