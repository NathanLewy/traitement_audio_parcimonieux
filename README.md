# Projet de Traitement du Signal

Ce projet est une collection de scripts et de données pour l'analyse et le traitement du signal, avec un focus particulier sur des techniques comme la parcimonie, la décomposition en ondelettes, et l'analyse de signaux audio.

## Structure du Projet

### Partie 1 : Analyse de Signaux Audio
- **Fichiers :**
  - `Partie1_banc_de_filtres.py` : Script pour l'analyse avec un banc de filtres.
  - `Partie1_vibrato.py` : Script pour l'analyse du vibrato.
  - `Partie1_4sinus.wav` et `Partie1_vibrato.wav` : Fichiers audio utilisés pour les analyses.

### Partie 2 : Décomposition en Ondelettes
- **Fichiers :**
  - `TL_2_pt1.py` : Première partie de l'analyse en ondelettes.
  - `TL_2_pt2.py` : Deuxième partie de l'analyse en ondelettes.
  - `#démonstration de wavedec.py` : Exemple d'utilisation de la décomposition en ondelettes.

### Partie 3 : Modélisation et Visualisation
- **Fichiers :**
  - `data.py` : Gestion et prétraitement des données.
  - `models.py` : Implémentation des modèles pour l'analyse.
  - `viz.py` : Scripts pour la visualisation des résultats.
- **Dossiers :**
  - `database_finale/` : Contient les fichiers CSV des données (`X_SONS.csv`, `y_SONS.csv`, etc.).
  - `SONS/`, `SONS_features/`, `SONS_VC_features/`, `SONS-VC/` : Dossiers contenant les données audio et leurs caractéristiques.

### Partie 4 : Compression et Reconstruction
- **Fichiers :**
  - `partie4_basispursuit.py` : Implémentation de la méthode Basis Pursuit.
  - `partie4_taux_compression.py` : Analyse des taux de compression.
  - `partie4_tout_signal.py` : Reconstruction de signaux entiers.
  - `partie4_une_fenetre.py` : Reconstruction sur une fenêtre spécifique.
- **Dossiers :**
  - `audio_partie4/` : Fichiers audio pour cette partie.
  - `tracés/` : Graphiques et visualisations générés.

## Prérequis

Les dépendances nécessaires sont listées dans le fichier [`requirements.txt`](requirements.txt). Installez-les avec la commande suivante :

```bash
pip install -r [requirements.txt](http://_vscodecontentref_/0)