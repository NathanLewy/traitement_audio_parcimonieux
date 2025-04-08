# Projet de Traitement du Signal

Ce repository rassemble toutes les réponses au travaux de laboratoire sur la représentation parcimonieuse des signaux.

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
  - `data.py` : Gestion et prétraitement des données. A LANCER EN PREMIER
  - `models.py` : Implémentation des modèles pour l'analyse. A LANCER APRES AVOIR LANCER data.py
  - `viz.py` : Scripts pour la visualisation des résultats.
- **Dossiers :**
  - `database_finale/` : Contient les fichiers CSV des données (`X_SONS.csv`, `y_SONS.csv`, etc.).
  - `SONS/`, `SONS_features/`, `SONS_VC_features/`, `SONS-VC/` : Dossiers contenant les données audio et leurs caractéristiques des oiseaux.

### Partie 4 : Compression et Reconstruction
- **Fichiers :**
  - `partie4_basispursuit.py` : Implémentation de la méthode Basis Pursuit, qui comme expliquée dans la présentation, ne converge pas.
  - `partie4_taux_compression.py` : Analyse de la définition étendue (expliquée dans la présentation) du taux de compression.
  - `partie4_tout_signal.py` : Reconstruction de signaux entiers selon la compression par Matching Pursuit et Orthogonal Matching Pursuit. On définit un objet compresseur et deux objets solveurs (MP et OMP)
  - `partie4_une_fenetre.py` : Reconstruction sur une fenêtre spécifique par Matching Pursuit et affichage de quelques éléments du dictionnaire.
- **Dossiers :**
  - `audio_partie4/` : Fichier audio en allemand utilisé pour cette partie.
  - `tracés/` : Graphiques et visualisations générés.

## Prérequis

Les dépendances nécessaires sont listées dans le fichier [`requirements.txt`](requirements.txt). Installez-les avec la commande suivante :

```bash
pip install -r requirements.txt
```

Il est possible qu'il faille changer le path d'éxecution dans les fichiers python si vous rencontrez des erreurs.

## Présentation

Afin de comprendre notre démarche, les diapositives de la présentation sont données dans le fichier pdf.
